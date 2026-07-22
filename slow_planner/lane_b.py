from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import (
    CandidateFrontier,
    OrderedImage,
    PlannerDecision,
    PlannerMetrics,
    SlowPlannerProtocolError,
    SlowPlannerRequest,
    StructuredPlannerDecision,
)


LANE_ID = "b"
REV_C_VIEW_ORDER = ("front_left", "front", "front_right", "rear")
REV_C_IMAGE_SIZE = (640, 480)
SNAPSHOT_ID_TEMPLATE = "b::<episode>::<reset>::<sequence>"
FROZEN_INTERNVLA_FALLBACK_CANDIDATE = "a1+b1+c1"

_IDENTIFIER = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_SNAPSHOT_RE = re.compile(
    rf"^b::(?P<episode>{_IDENTIFIER})::(?P<reset>0|[1-9][0-9]*)::(?P<sequence>0|[1-9][0-9]*)$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC_REASON_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class LaneBContractError(SlowPlannerProtocolError):
    """Raised when the private Lane B adapter contract is violated."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise LaneBContractError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LaneBContractError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise LaneBContractError(f"{name} must be finite")
    return number


def _sha256(value: str, name: str) -> str:
    raw = str(value).strip()
    normalized = raw.lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise LaneBContractError(f"{name} must be a lowercase SHA-256 hex digest")
    if raw != normalized:
        raise LaneBContractError(f"{name} must be lowercase")
    return normalized


def _public_reason(value: str, *, default: str = "planner_failure") -> str:
    normalized = str(value).strip()
    return normalized if _PUBLIC_REASON_RE.fullmatch(normalized) else default


@dataclass(frozen=True)
class LaneBSnapshotIdentity:
    episode_id: str
    reset_id: int
    sequence_id: int

    def __post_init__(self) -> None:
        if not re.fullmatch(_IDENTIFIER, self.episode_id):
            raise LaneBContractError(
                "episode_id contains characters forbidden by the Lane B snapshot contract"
            )
        if (
            isinstance(self.reset_id, bool)
            or not isinstance(self.reset_id, int)
            or self.reset_id < 0
        ):
            raise LaneBContractError("reset_id must be a non-negative integer")
        if (
            isinstance(self.sequence_id, bool)
            or not isinstance(self.sequence_id, int)
            or self.sequence_id < 0
        ):
            raise LaneBContractError("sequence_id must be a non-negative integer")

    @property
    def snapshot_id(self) -> str:
        return f"{LANE_ID}::{self.episode_id}::{self.reset_id}::{self.sequence_id}"

    @classmethod
    def parse(cls, snapshot_id: str) -> "LaneBSnapshotIdentity":
        match = _SNAPSHOT_RE.fullmatch(str(snapshot_id))
        if match is None:
            raise LaneBContractError(
                f"snapshot_id must match {SNAPSHOT_ID_TEMPLATE!r}; received {snapshot_id!r}"
            )
        return cls(
            episode_id=match.group("episode"),
            reset_id=int(match.group("reset")),
            sequence_id=int(match.group("sequence")),
        )


def build_lane_b_snapshot_id(episode_id: str, reset_id: int, sequence_id: int) -> str:
    return LaneBSnapshotIdentity(episode_id, reset_id, sequence_id).snapshot_id


def parse_lane_b_snapshot_id(snapshot_id: str) -> LaneBSnapshotIdentity:
    return LaneBSnapshotIdentity.parse(snapshot_id)


class LaneBSnapshotAdvance(str, Enum):
    NEW = "new"
    IDEMPOTENT_REPLAY = "idempotent_replay"


class LaneBSnapshotSequenceTracker:
    """Fail-closed gate for global reset generations and per-reset sequences."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: LaneBSnapshotIdentity | None = None
        self._closed_episodes: set[str] = set()

    @property
    def current(self) -> LaneBSnapshotIdentity | None:
        with self._lock:
            return self._current

    def observe(self, identity: LaneBSnapshotIdentity | str) -> LaneBSnapshotAdvance:
        value = (
            parse_lane_b_snapshot_id(identity)
            if isinstance(identity, str)
            else identity
        )
        if not isinstance(value, LaneBSnapshotIdentity):
            raise LaneBContractError(
                "snapshot identity tracker requires Lane B identity"
            )
        with self._lock:
            current = self._current
            if current is None:
                self._current = value
                return LaneBSnapshotAdvance.NEW
            if value.episode_id != current.episode_id:
                if value.episode_id in self._closed_episodes:
                    raise LaneBContractError("snapshot episode cannot be reopened")
                if value.sequence_id != 0:
                    raise LaneBContractError(
                        "a new episode generation must begin at sequence_id=0"
                    )
                if value.reset_id <= current.reset_id:
                    raise LaneBContractError(
                        "a new episode must advance the global reset generation"
                    )
                self._closed_episodes.add(current.episode_id)
                self._current = value
                return LaneBSnapshotAdvance.NEW
            if value.reset_id < current.reset_id:
                raise LaneBContractError("snapshot reset_id regressed")
            if value.reset_id > current.reset_id:
                if value.sequence_id != 0:
                    raise LaneBContractError(
                        "a new reset generation must begin at sequence_id=0"
                    )
                self._current = value
                return LaneBSnapshotAdvance.NEW
            if value.sequence_id < current.sequence_id:
                raise LaneBContractError("snapshot sequence_id regressed")
            if value.sequence_id == current.sequence_id:
                return LaneBSnapshotAdvance.IDEMPOTENT_REPLAY
            self._current = value
            return LaneBSnapshotAdvance.NEW


@dataclass(frozen=True)
class TimedRevCImage:
    image: OrderedImage
    sim_stamp_s: float
    extrinsic_sha256: str
    source_frame_id: str = ""

    def __post_init__(self) -> None:
        if _finite(self.sim_stamp_s, f"image[{self.image.view_id}].sim_stamp_s") < 0:
            raise LaneBContractError(
                f"image[{self.image.view_id}].sim_stamp_s must be non-negative"
            )
        _sha256(self.extrinsic_sha256, f"image[{self.image.view_id}].extrinsic_sha256")
        if not str(self.source_frame_id).strip():
            raise LaneBContractError(
                f"image[{self.image.view_id}].source_frame_id is required"
            )


@dataclass(frozen=True)
class ValidatedRevCSnapshot:
    identity: LaneBSnapshotIdentity
    snapshot_sim_stamp_s: float
    frames: tuple[TimedRevCImage, ...]
    config_sha256: str
    max_frame_age_s: float
    max_inter_camera_skew_s: float

    @property
    def ordered_images(self) -> tuple[OrderedImage, ...]:
        return tuple(frame.image for frame in self.frames)

    @property
    def frame_ages_s(self) -> tuple[float, ...]:
        return tuple(
            self.snapshot_sim_stamp_s - frame.sim_stamp_s for frame in self.frames
        )

    @property
    def inter_camera_skew_s(self) -> float:
        stamps = [frame.sim_stamp_s for frame in self.frames]
        return max(stamps) - min(stamps)

    def sidecar_record(
        self, *, camera_paths: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        paths = dict(camera_paths or {})
        record = {
            "schema_version": 1,
            "kind": "lane_b_rev_c_snapshot",
            "lane_id": LANE_ID,
            "episode_id": self.identity.episode_id,
            "reset_id": self.identity.reset_id,
            "sequence_id": self.identity.sequence_id,
            "snapshot_id": self.identity.snapshot_id,
            "snapshot_sim_stamp_s": self.snapshot_sim_stamp_s,
            "written_wall_time_s": time.time(),
            "config_sha256": self.config_sha256,
            "view_order": list(REV_C_VIEW_ORDER),
            "inter_camera_skew_s": self.inter_camera_skew_s,
            "max_frame_age_s": self.max_frame_age_s,
            "max_inter_camera_skew_s": self.max_inter_camera_skew_s,
            "images": [
                {
                    "view_id": frame.image.view_id,
                    "source_frame_id": frame.source_frame_id,
                    "sim_stamp_s": frame.sim_stamp_s,
                    "age_s": age_s,
                    "width": frame.image.width,
                    "height": frame.image.height,
                    "pose": list(frame.image.pose),
                    "extrinsic_sha256": frame.extrinsic_sha256,
                    "jpeg_sha256": hashlib.sha256(frame.image.jpeg).hexdigest(),
                    **(
                        {"jpeg_path": paths[frame.image.view_id]}
                        if frame.image.view_id in paths
                        else {}
                    ),
                }
                for frame, age_s in zip(self.frames, self.frame_ages_s)
            ],
        }
        record["snapshot_content_sha256"] = snapshot_content_sha256(record)
        return record


def snapshot_content_sha256(record: Mapping[str, Any]) -> str:
    """Hash the immutable image/config/extrinsic identity of one Lane B snapshot."""

    snapshot_id = str(record.get("snapshot_id") or "")
    parse_lane_b_snapshot_id(snapshot_id)
    config_sha256 = _sha256(str(record.get("config_sha256") or ""), "config_sha256")
    rows = record.get("images")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise LaneBContractError("snapshot images must be an array")
    images = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise LaneBContractError("snapshot image entries must be objects")
        images.append(
            {
                "view_id": str(row.get("view_id") or ""),
                "jpeg_sha256": _sha256(
                    str(row.get("jpeg_sha256") or ""), "image.jpeg_sha256"
                ),
                "extrinsic_sha256": _sha256(
                    str(row.get("extrinsic_sha256") or ""),
                    "image.extrinsic_sha256",
                ),
            }
        )
    if tuple(row["view_id"] for row in images) != REV_C_VIEW_ORDER:
        raise LaneBContractError(
            "snapshot content hash requires the fixed Rev-C view order"
        )
    payload = {
        "snapshot_id": snapshot_id,
        "config_sha256": config_sha256,
        "images": images,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def validate_rev_c_snapshot(
    *,
    episode_id: str,
    snapshot_id: str,
    snapshot_sim_stamp_s: float,
    frames: Sequence[TimedRevCImage],
    config_sha256: str,
    expected_config_sha256: str,
    expected_extrinsic_sha256: Mapping[str, str],
    max_frame_age_s: float,
    max_inter_camera_skew_s: float,
) -> ValidatedRevCSnapshot:
    """Validate the private Rev-C snapshot without changing slow-planner protocol v1."""

    identity = parse_lane_b_snapshot_id(snapshot_id)
    if identity.episode_id != episode_id:
        raise LaneBContractError(
            f"snapshot episode_id={identity.episode_id!r} does not match request episode_id={episode_id!r}"
        )
    stamp = _finite(snapshot_sim_stamp_s, "snapshot_sim_stamp_s")
    if stamp < 0:
        raise LaneBContractError("snapshot_sim_stamp_s must be non-negative")
    max_age = _finite(max_frame_age_s, "max_frame_age_s")
    max_skew = _finite(max_inter_camera_skew_s, "max_inter_camera_skew_s")
    if max_age < 0 or max_skew < 0:
        raise LaneBContractError("snapshot freshness limits must be non-negative")
    config_hash = _sha256(config_sha256, "config_sha256")
    expected_config_hash = _sha256(expected_config_sha256, "expected_config_sha256")
    if config_hash != expected_config_hash:
        raise LaneBContractError("snapshot config hash mismatch")

    normalized = tuple(frames)
    view_ids = tuple(frame.image.view_id for frame in normalized)
    if view_ids != REV_C_VIEW_ORDER:
        raise LaneBContractError(
            f"Rev-C ordered_images must be {REV_C_VIEW_ORDER!r}; received {view_ids!r}"
        )
    expected_width, expected_height = REV_C_IMAGE_SIZE
    for frame in normalized:
        if (frame.image.width, frame.image.height) != (expected_width, expected_height):
            raise LaneBContractError(
                f"image[{frame.image.view_id}] must be {expected_width}x{expected_height}"
            )
        age_s = stamp - frame.sim_stamp_s
        if age_s < 0:
            raise LaneBContractError(
                f"image[{frame.image.view_id}] is from the future in sim time"
            )
        if age_s > max_age:
            raise LaneBContractError(
                f"image[{frame.image.view_id}] age_s={age_s:.6f} exceeds max_frame_age_s={max_age:.6f}"
            )

    image_stamps = [frame.sim_stamp_s for frame in normalized]
    observed_skew = max(image_stamps) - min(image_stamps)
    if observed_skew > max_skew:
        raise LaneBContractError(
            f"inter-camera skew_s={observed_skew:.6f} exceeds max_inter_camera_skew_s={max_skew:.6f}"
        )

    if not isinstance(expected_extrinsic_sha256, Mapping):
        raise LaneBContractError("expected extrinsic hashes are required")
    expected_views = set(REV_C_VIEW_ORDER)
    if set(expected_extrinsic_sha256) != expected_views:
        raise LaneBContractError(
            "expected extrinsic hashes must cover exactly the four Rev-C views"
        )
    for frame in normalized:
        expected = _sha256(
            expected_extrinsic_sha256[frame.image.view_id],
            f"expected_extrinsic_sha256[{frame.image.view_id}]",
        )
        if frame.extrinsic_sha256 != expected:
            raise LaneBContractError(
                f"image[{frame.image.view_id}] extrinsic hash mismatch"
            )

    return ValidatedRevCSnapshot(
        identity=identity,
        snapshot_sim_stamp_s=stamp,
        frames=normalized,
        config_sha256=config_hash,
        max_frame_age_s=max_age,
        max_inter_camera_skew_s=max_skew,
    )


def build_lane_b_request(
    snapshot: ValidatedRevCSnapshot,
    *,
    instruction: str,
    candidate_frontiers: Sequence[CandidateFrontier],
    agent_pose: Sequence[float],
    visited_frontiers: Sequence[int] = (),
    compact_history: Sequence[str] = (),
    wall_timestamp_s: float | None = None,
) -> SlowPlannerRequest:
    """Adapt a validated private snapshot into the unchanged v1 request schema."""

    return SlowPlannerRequest(
        episode_id=snapshot.identity.episode_id,
        snapshot_id=snapshot.identity.snapshot_id,
        instruction=instruction,
        ordered_images=snapshot.ordered_images,
        candidate_frontiers=tuple(candidate_frontiers),
        agent_pose=tuple(float(value) for value in agent_pose),
        visited_frontiers=tuple(int(value) for value in visited_frontiers),
        compact_history=tuple(str(value) for value in compact_history),
        timestamp=time.time()
        if wall_timestamp_s is None
        else _finite(wall_timestamp_s, "wall_timestamp_s"),
    )


class LaneBSnapshotSidecar:
    """Replay-safe JSONL metadata plus immutable snapshot-scoped JPEG files."""

    def __init__(
        self, path: str | Path, *, camera_dir: str | Path | None = None
    ) -> None:
        self.path = Path(path)
        self.camera_dir = Path(camera_dir) if camera_dir is not None else None
        self._lock = threading.Lock()
        self._tracker = LaneBSnapshotSequenceTracker()
        self._records: dict[str, tuple[str, dict[str, Any]]] = {}
        self._loaded = False

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _verified_content_hash(record: Mapping[str, Any]) -> str:
        observed = snapshot_content_sha256(record)
        declared = _sha256(
            str(record.get("snapshot_content_sha256") or ""),
            "snapshot_content_sha256",
        )
        if observed != declared:
            raise LaneBContractError("snapshot content hash mismatch")
        return observed

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LaneBContractError(
                        f"invalid snapshot sidecar JSON at line {line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise LaneBContractError(
                        f"snapshot sidecar line {line_number} must be an object"
                    )
                if record.get("kind") != "lane_b_rev_c_snapshot":
                    raise LaneBContractError(
                        f"snapshot sidecar line {line_number} has invalid kind"
                    )
                identity = parse_lane_b_snapshot_id(
                    str(record.get("snapshot_id") or "")
                )
                if (
                    record.get("episode_id") != identity.episode_id
                    or record.get("reset_id") != identity.reset_id
                    or record.get("sequence_id") != identity.sequence_id
                ):
                    raise LaneBContractError(
                        f"snapshot sidecar line {line_number} identity mismatch"
                    )
                content_hash = self._verified_content_hash(record)
                prior = self._records.get(identity.snapshot_id)
                if prior is not None and prior[0] != content_hash:
                    raise LaneBContractError(
                        f"snapshot_id reuse conflict in sidecar: {identity.snapshot_id}"
                    )
                self._tracker.observe(identity)
                if prior is None:
                    self._records[identity.snapshot_id] = (content_hash, record)
        self._loaded = True

    def _verify_existing_jpegs(self, record: Mapping[str, Any]) -> None:
        if self.camera_dir is None:
            return
        camera_root = self.camera_dir.resolve()
        for row in record.get("images") or []:
            if not isinstance(row, Mapping):
                raise LaneBContractError("snapshot image record must be an object")
            view_id = str(row.get("view_id") or "")
            path = Path(str(row.get("jpeg_path") or "")).resolve()
            if (
                view_id not in REV_C_VIEW_ORDER
                or not path.is_relative_to(camera_root)
                or path.name != f"{view_id}.jpg"
                or not path.is_file()
            ):
                raise LaneBContractError("idempotent snapshot JPEG path is invalid")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row.get("jpeg_sha256"):
                raise LaneBContractError("idempotent snapshot JPEG hash mismatch")

    def append(self, snapshot: ValidatedRevCSnapshot) -> dict[str, Any]:
        with self._lock:
            self._ensure_loaded()
            content_only_record = snapshot.sidecar_record()
            content_hash = self._verified_content_hash(content_only_record)
            snapshot_id = snapshot.identity.snapshot_id
            prior = self._records.get(snapshot_id)
            if prior is not None:
                if prior[0] != content_hash:
                    raise LaneBContractError(
                        f"snapshot_id reuse conflict: {snapshot_id}"
                    )
                self._tracker.observe(snapshot.identity)
                self._verify_existing_jpegs(prior[1])
                return dict(prior[1])

            advance = self._tracker.observe(snapshot.identity)
            if advance is LaneBSnapshotAdvance.IDEMPOTENT_REPLAY:
                raise LaneBContractError(
                    "snapshot identity replay has no canonical sidecar record"
                )

            camera_paths: dict[str, str] = {}
            targets: list[tuple[TimedRevCImage, Path]] = []
            if self.camera_dir is not None:
                snapshot_dir = (
                    self.camera_dir
                    / snapshot.identity.episode_id
                    / f"reset-{snapshot.identity.reset_id}"
                    / f"sequence-{snapshot.identity.sequence_id}"
                )
                targets = [
                    (frame, snapshot_dir / f"{frame.image.view_id}.jpg")
                    for frame in snapshot.frames
                ]
                if any(target.exists() for _, target in targets):
                    raise LaneBContractError(
                        "uncommitted snapshot JPEG path already exists; refusing overwrite"
                    )
                for frame, target in targets:
                    self._atomic_write(target, frame.image.jpeg)
                    camera_paths[frame.image.view_id] = str(target.resolve())

            record = snapshot.sidecar_record(camera_paths=camera_paths)
            if self._verified_content_hash(record) != content_hash:
                raise LaneBContractError(
                    "snapshot changed while preparing sidecar record"
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = (
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
            try:
                if os.write(descriptor, line) != len(line):
                    raise OSError("short write while appending Lane B snapshot sidecar")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._records[snapshot_id] = (content_hash, record)
            return record


class LaneBPlannerMode(str, Enum):
    BOUNDED_ADVISOR = "bounded_advisor"
    DIRECT_HIGH_LEVEL = "direct_high_level"


class LaneBIntent(str, Enum):
    FRONTIER_ADVICE = "frontier_advice"
    FRONTIER_GOAL_CANDIDATE = "frontier_goal_candidate"
    RELATIVE_TARGET_SAFE_HOLD_CANDIDATE = "relative_target_safe_hold_candidate"
    INTERNVLA_FALLBACK_REQUIRED = "INTERNVLA_FALLBACK_REQUIRED"
    DIRECT_SAFE_STOP_REQUIRED = "DIRECT_SAFE_STOP_REQUIRED"


@dataclass(frozen=True)
class LaneBPlannerOutcome:
    episode_id: str
    snapshot_id: str
    mode: LaneBPlannerMode
    source_decision: str
    intent: LaneBIntent
    frontier_id: int | None
    target_relative_xz: tuple[float, float] | None
    confidence: float
    scene_summary: str = ""
    target_evidence: tuple[str, ...] = ()
    blocked_directions: tuple[str, ...] = ()
    recommended_frontier: int | None = None
    target_found: bool = False
    abstain: bool = False
    fallback_used: bool = False
    fallback_reason: str = ""
    requires_arrival_confirmation: bool = False
    requires_internvla_fallback: bool = False
    requires_safe_stop: bool = False
    safe_stop_reason: str = ""

    def to_public_mapping(self) -> dict[str, Any]:
        """Return UI-safe structured data; model raw text is deliberately absent."""

        return {
            "schema_version": 1,
            "episode_id": self.episode_id,
            "snapshot_id": self.snapshot_id,
            "mode": self.mode.value,
            "source_decision": self.source_decision,
            "intent": self.intent.value,
            "frontier_id": self.frontier_id,
            "target_relative_xz": (
                list(self.target_relative_xz)
                if self.target_relative_xz is not None
                else None
            ),
            "confidence": self.confidence,
            "scene_summary": self.scene_summary,
            "target_evidence": list(self.target_evidence),
            "blocked_directions": list(self.blocked_directions),
            "recommended_frontier": self.recommended_frontier,
            "target_found": self.target_found,
            "abstain": self.abstain,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "requires_arrival_confirmation": self.requires_arrival_confirmation,
            "requires_internvla_fallback": self.requires_internvla_fallback,
            "requires_safe_stop": self.requires_safe_stop,
            "safe_stop_reason": self.safe_stop_reason,
            "fallback_owner": (
                "coordinator_frozen_internvla_candidate"
                if self.requires_internvla_fallback
                else None
            ),
            "fallback_candidate": (
                FROZEN_INTERNVLA_FALLBACK_CANDIDATE
                if self.requires_internvla_fallback
                else None
            ),
            "motion_authority": "none",
        }


class LaneBPlannerAdapter:
    """Constrain v1 model decisions to MOVE-only, non-control Lane B intents."""

    def __init__(self, mode: LaneBPlannerMode | str) -> None:
        self.mode = LaneBPlannerMode(mode)

    def _internvla_fallback_required(
        self,
        request: SlowPlannerRequest,
        *,
        source_decision: str,
        reason: str,
        decision: PlannerDecision | None = None,
    ) -> LaneBPlannerOutcome:
        structured = (
            decision if isinstance(decision, StructuredPlannerDecision) else None
        )
        direct = self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL
        normalized_reason = _public_reason(reason)
        return LaneBPlannerOutcome(
            episode_id=request.episode_id,
            snapshot_id=request.snapshot_id,
            mode=self.mode,
            source_decision=source_decision,
            intent=(
                LaneBIntent.DIRECT_SAFE_STOP_REQUIRED
                if direct
                else LaneBIntent.INTERNVLA_FALLBACK_REQUIRED
            ),
            frontier_id=None,
            target_relative_xz=None,
            confidence=0.0,
            scene_summary=(structured.scene_summary if structured else ""),
            target_evidence=(structured.target_evidence if structured else ()),
            blocked_directions=(
                structured.blocked_directions if structured else ()
            ),
            recommended_frontier=None,
            target_found=(structured.target_found if structured else False),
            abstain=True,
            fallback_used=not direct,
            fallback_reason="" if direct else normalized_reason,
            requires_internvla_fallback=not direct,
            requires_safe_stop=direct,
            safe_stop_reason=normalized_reason if direct else "",
        )

    def resolve(
        self, request: SlowPlannerRequest, decision: PlannerDecision
    ) -> LaneBPlannerOutcome:
        try:
            identity = parse_lane_b_snapshot_id(request.snapshot_id)
        except LaneBContractError:
            return self._internvla_fallback_required(
                request,
                source_decision=decision.decision,
                reason="invalid_request_identity",
                decision=decision,
            )
        if identity.episode_id != request.episode_id:
            return self._internvla_fallback_required(
                request,
                source_decision=decision.decision,
                reason="request_episode_snapshot_mismatch",
                decision=decision,
            )
        if (
            decision.episode_id != request.episode_id
            or decision.snapshot_id != request.snapshot_id
        ):
            return self._internvla_fallback_required(
                request,
                source_decision=decision.decision,
                reason="stale_or_mismatched_response",
                decision=decision,
            )
        if decision.fallback_used:
            return self._internvla_fallback_required(
                request,
                source_decision=decision.decision,
                reason="step3_internal_fallback",
                decision=decision,
            )

        current_frontier_ids = {
            frontier.frontier_id for frontier in request.candidate_frontiers
        }
        if decision.decision == "select_frontier":
            if decision.frontier_id not in current_frontier_ids:
                return self._internvla_fallback_required(
                    request,
                    source_decision=decision.decision,
                    reason="frontier_not_current",
                    decision=decision,
                )
            intent = (
                LaneBIntent.FRONTIER_ADVICE
                if self.mode is LaneBPlannerMode.BOUNDED_ADVISOR
                else LaneBIntent.FRONTIER_GOAL_CANDIDATE
            )
            return LaneBPlannerOutcome(
                episode_id=request.episode_id,
                snapshot_id=request.snapshot_id,
                mode=self.mode,
                source_decision=decision.decision,
                intent=intent,
                frontier_id=decision.frontier_id,
                target_relative_xz=None,
                confidence=decision.confidence,
                scene_summary=(
                    decision.scene_summary
                    if isinstance(decision, StructuredPlannerDecision)
                    else ""
                ),
                target_evidence=(
                    decision.target_evidence
                    if isinstance(decision, StructuredPlannerDecision)
                    else ()
                ),
                blocked_directions=(
                    decision.blocked_directions
                    if isinstance(decision, StructuredPlannerDecision)
                    else ()
                ),
                recommended_frontier=decision.frontier_id,
                target_found=False,
                abstain=False,
                fallback_used=decision.fallback_used,
                fallback_reason=_public_reason(decision.fallback_reason, default=""),
            )

        if decision.decision == "target_found":
            if self.mode is LaneBPlannerMode.BOUNDED_ADVISOR:
                return self._internvla_fallback_required(
                    request,
                    source_decision=decision.decision,
                    reason="bounded_advisor_forbids_relative_target",
                    decision=decision,
                )
            if not request.candidate_frontiers:
                return self._internvla_fallback_required(
                    request,
                    source_decision=decision.decision,
                    reason="relative_target_without_frontiers",
                    decision=decision,
                )
            assert decision.target_relative_xz is not None
            target_distance_m = math.hypot(*decision.target_relative_xz)
            if target_distance_m <= 0.0:
                return self._internvla_fallback_required(
                    request,
                    source_decision=decision.decision,
                    reason="relative_target_zero",
                    decision=decision,
                )
            max_frontier_distance_m = max(
                frontier.distance_m for frontier in request.candidate_frontiers
            )
            if target_distance_m > max_frontier_distance_m:
                return self._internvla_fallback_required(
                    request,
                    source_decision=decision.decision,
                    reason="relative_target_out_of_range",
                    decision=decision,
                )
            return LaneBPlannerOutcome(
                episode_id=request.episode_id,
                snapshot_id=request.snapshot_id,
                mode=self.mode,
                source_decision=decision.decision,
                intent=LaneBIntent.RELATIVE_TARGET_SAFE_HOLD_CANDIDATE,
                frontier_id=None,
                target_relative_xz=decision.target_relative_xz,
                confidence=decision.confidence,
                fallback_used=decision.fallback_used,
                fallback_reason=_public_reason(decision.fallback_reason, default=""),
                requires_arrival_confirmation=True,
            )

        return self._internvla_fallback_required(
            request,
            source_decision="abstain",
            reason="step3_abstain",
            decision=decision,
        )

    def resolve_failure(
        self, request: SlowPlannerRequest, reason: str
    ) -> LaneBPlannerOutcome:
        """Fail closed according to the selected Lane-B experiment mode."""

        normalized = _public_reason(reason)
        return self._internvla_fallback_required(
            request,
            source_decision="service_failure",
            reason=normalized,
        )


def frontend_decision_record(
    outcome: LaneBPlannerOutcome,
    metrics: PlannerMetrics | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured public record with no raw generation or reasoning fields."""

    if metrics is None:
        public_metrics: dict[str, Any] = {}
    elif isinstance(metrics, PlannerMetrics):
        public_metrics = metrics.to_mapping()
    else:
        allowed = set(PlannerMetrics.__dataclass_fields__)
        public_metrics = {
            str(key): value for key, value in metrics.items() if str(key) in allowed
        }
    return {
        "kind": "lane_b_planner_decision",
        "published_wall_time_s": time.time(),
        "decision": outcome.to_public_mapping(),
        "metrics": public_metrics,
    }


class LaneBDecisionSidecar:
    """Append only the redacted, structured adapter outcome consumed by the UI."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(
        self,
        outcome: LaneBPlannerOutcome,
        metrics: PlannerMetrics | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = frontend_decision_record(outcome, metrics)
        line = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
            try:
                if os.write(descriptor, line) != len(line):
                    raise OSError("short write while appending Lane B decision sidecar")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return record
