"""Fail-closed binding between live Nav2 frontiers and Lane-B SlowPlanner v1.

This module deliberately does not derive exploration frontiers from a Nav2
plan, costmap, or the frozen replay fixtures.  A coordinator-owned, live Nav2
frontier producer must atomically publish the private JSON contract validated
here.  The existing SlowPlanner wire protocol remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import CandidateFrontier, PlannerDecision, SlowPlannerRequest
from .lane_b import (
    LaneBContractError,
    LaneBIntent,
    LaneBPlannerAdapter,
    LaneBPlannerMode,
    LaneBSnapshotIdentity,
    ValidatedRevCSnapshot,
    build_lane_b_request,
    parse_lane_b_snapshot_id,
)


LIVE_FRONTIER_SCHEMA_VERSION = 1
LIVE_FRONTIER_KIND = "t5_live_nav2_frontier_snapshot"
LIVE_FRONTIER_GEOMETRY_FRAME = "base_link"
LIVE_FRONTIER_AGENT_POSE_ENCODING = "map_xy_yaw_rad"
_SHA256_LENGTH = 64
_FRONTIER_KEYS = frozenset(
    {"frontier_id", "relative_xz", "distance_m", "bearing_deg"}
)
_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "source_node",
        "ros_episode_id",
        "episode_id",
        "reset_id",
        "sequence_id",
        "snapshot_id",
        "captured_sim_time_s",
        "valid_until_sim_time_s",
        "geometry_frame",
        "agent_pose_encoding",
        "agent_pose",
        "candidate_frontiers",
        "frontier_set_sha256",
    }
)


class LiveFrontierContractError(ValueError):
    """A private live-frontier source or arbitration boundary failed closed."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def _contract_error(message: str, code: str) -> LiveFrontierContractError:
    return LiveFrontierContractError(message, code=code)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise _contract_error(
            f"{name} must be numeric, not bool", "INVALID_LIVE_NAV2_FRONTIER_SOURCE"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _contract_error(
            f"{name} must be numeric", "INVALID_LIVE_NAV2_FRONTIER_SOURCE"
        ) from exc
    if not math.isfinite(number):
        raise _contract_error(
            f"{name} must be finite", "INVALID_LIVE_NAV2_FRONTIER_SOURCE"
        )
    return number


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _contract_error(
            f"{name} must be a non-negative integer",
            "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
        )
    return value


def _canonical_frontier_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable source payload covered by ``frontier_set_sha256``."""

    return {
        "schema_version": value["schema_version"],
        "kind": value["kind"],
        "source_node": value["source_node"],
        "ros_episode_id": value["ros_episode_id"],
        "episode_id": value["episode_id"],
        "reset_id": value["reset_id"],
        "sequence_id": value["sequence_id"],
        "snapshot_id": value["snapshot_id"],
        "captured_sim_time_s": value["captured_sim_time_s"],
        "valid_until_sim_time_s": value["valid_until_sim_time_s"],
        "geometry_frame": value["geometry_frame"],
        "agent_pose_encoding": value["agent_pose_encoding"],
        "agent_pose": list(value["agent_pose"]),
        "candidate_frontiers": [
            dict(frontier) for frontier in value["candidate_frontiers"]
        ],
    }


def live_frontier_content_sha256(value: Mapping[str, Any]) -> str:
    """Hash one source-supplied frontier set without creating any frontier."""

    canonical = json.dumps(
        _canonical_frontier_payload(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class LiveNav2FrontierSnapshot:
    """Identity-bound frontier geometry emitted by a real live Nav2 source."""

    source_node: str
    identity: LaneBSnapshotIdentity
    captured_sim_time_s: float
    valid_until_sim_time_s: float
    agent_pose: tuple[float, float, float]
    candidate_frontiers: tuple[CandidateFrontier, ...]
    frontier_set_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LiveNav2FrontierSnapshot":
        if not isinstance(value, Mapping):
            raise _contract_error(
                "live frontier snapshot must be an object",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )
        if set(value) != _SNAPSHOT_KEYS:
            missing = sorted(_SNAPSHOT_KEYS - set(value))
            unknown = sorted(set(value) - _SNAPSHOT_KEYS)
            raise _contract_error(
                f"live frontier snapshot keys differ: missing={missing}, unknown={unknown}",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )
        if value.get("schema_version") != LIVE_FRONTIER_SCHEMA_VERSION:
            raise _contract_error(
                "unsupported live frontier schema_version",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )
        if value.get("kind") != LIVE_FRONTIER_KIND:
            raise _contract_error(
                "frontier source is not a live Nav2 frontier snapshot",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )
        source_node = str(value.get("source_node") or "").strip()
        if not source_node or len(source_node) > 256:
            raise _contract_error(
                "source_node is required and must be bounded",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )
        if value.get("geometry_frame") != LIVE_FRONTIER_GEOMETRY_FRAME:
            raise _contract_error(
                "candidate geometry must be relative to base_link",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )
        if value.get("agent_pose_encoding") != LIVE_FRONTIER_AGENT_POSE_ENCODING:
            raise _contract_error(
                "agent pose must use map_xy_yaw_rad encoding",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )

        episode_id = str(value.get("episode_id") or "")
        reset_id = _non_negative_int(value.get("reset_id"), "reset_id")
        sequence_id = _non_negative_int(value.get("sequence_id"), "sequence_id")
        identity = parse_lane_b_snapshot_id(str(value.get("snapshot_id") or ""))
        if identity != LaneBSnapshotIdentity(episode_id, reset_id, sequence_id):
            raise _contract_error(
                "frontier identity fields disagree with snapshot_id",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )
        if value.get("ros_episode_id") != f"b::{identity.episode_id}":
            raise _contract_error(
                "ros_episode_id is not the lane-prefixed ObservationMetadata identity",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )

        captured = _finite(value.get("captured_sim_time_s"), "captured_sim_time_s")
        valid_until = _finite(
            value.get("valid_until_sim_time_s"), "valid_until_sim_time_s"
        )
        if captured < 0.0 or valid_until < captured:
            raise _contract_error(
                "live frontier validity interval is invalid",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )

        raw_pose = value.get("agent_pose")
        if not isinstance(raw_pose, Sequence) or isinstance(
            raw_pose, (str, bytes, bytearray)
        ):
            raise _contract_error(
                "agent_pose must be an array",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )
        pose = tuple(
            _finite(item, f"agent_pose[{index}]")
            for index, item in enumerate(raw_pose)
        )
        if len(pose) != 3:
            raise _contract_error(
                "agent_pose must contain map x, y and yaw",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )

        raw_frontiers = value.get("candidate_frontiers")
        if not isinstance(raw_frontiers, Sequence) or isinstance(
            raw_frontiers, (str, bytes, bytearray)
        ):
            raise _contract_error(
                "candidate_frontiers must be an array",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )
        for index, item in enumerate(raw_frontiers):
            if not isinstance(item, Mapping) or set(item) != _FRONTIER_KEYS:
                raise _contract_error(
                    f"candidate_frontiers[{index}] must contain only SlowPlanner v1 geometry",
                    "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
                )
        try:
            frontiers = tuple(
                CandidateFrontier.from_mapping(item) for item in raw_frontiers
            )
        except (TypeError, ValueError) as exc:
            raise _contract_error(
                "candidate_frontiers violate SlowPlanner v1 geometry",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            ) from exc
        frontier_ids = [frontier.frontier_id for frontier in frontiers]
        if len(frontier_ids) != len(set(frontier_ids)):
            raise _contract_error(
                "candidate frontier IDs must be unique",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )

        supplied_hash = str(value.get("frontier_set_sha256") or "")
        if (
            len(supplied_hash) != _SHA256_LENGTH
            or supplied_hash.lower() != supplied_hash
            or any(character not in "0123456789abcdef" for character in supplied_hash)
        ):
            raise _contract_error(
                "frontier_set_sha256 must be lowercase SHA-256",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )
        try:
            observed_hash = live_frontier_content_sha256(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise _contract_error(
                "live frontier content cannot be canonically hashed",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            ) from exc
        if supplied_hash != observed_hash:
            raise _contract_error(
                "live frontier content hash mismatch",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )
        return cls(
            source_node=source_node,
            identity=identity,
            captured_sim_time_s=captured,
            valid_until_sim_time_s=valid_until,
            agent_pose=(pose[0], pose[1], pose[2]),
            candidate_frontiers=frontiers,
            frontier_set_sha256=supplied_hash,
        )

    def require_current(
        self,
        *,
        sim_now_s: float,
        max_age_s: float,
        expected_identity: LaneBSnapshotIdentity | None = None,
    ) -> None:
        now = _finite(sim_now_s, "sim_now_s")
        maximum_age = _finite(max_age_s, "max_age_s")
        if maximum_age <= 0.0:
            raise _contract_error(
                "max_age_s must be positive", "INVALID_ARBITER_CONFIGURATION"
            )
        if expected_identity is not None and self.identity != expected_identity:
            raise _contract_error(
                "live frontier identity is no longer current",
                "STALE_LIVE_NAV2_FRONTIER_SOURCE",
            )
        if now < self.captured_sim_time_s:
            raise _contract_error(
                "live frontier snapshot is from the future",
                "STALE_LIVE_NAV2_FRONTIER_SOURCE",
            )
        if (
            now > self.valid_until_sim_time_s
            or now - self.captured_sim_time_s > maximum_age
        ):
            raise _contract_error(
                "live frontier snapshot expired in sim time",
                "STALE_LIVE_NAV2_FRONTIER_SOURCE",
            )


class LiveNav2FrontierFile:
    """Read an atomically replaced current-frontier file; never synthesize one."""

    def __init__(self, path: str | Path, *, maximum_bytes: int = 1_048_576) -> None:
        self.path = Path(path)
        self.maximum_bytes = int(maximum_bytes)
        if self.maximum_bytes <= 0:
            raise ValueError("maximum_bytes must be positive")

    def read(self) -> LiveNav2FrontierSnapshot:
        try:
            payload = self.path.read_bytes()
        except FileNotFoundError as exc:
            raise _contract_error(
                "no live Nav2 frontier producer has published the current snapshot",
                "MISSING_LIVE_NAV2_FRONTIER_SOURCE",
            ) from exc
        except OSError as exc:
            raise _contract_error(
                "live Nav2 frontier snapshot is unreadable",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            ) from exc
        if not payload or len(payload) > self.maximum_bytes:
            raise _contract_error(
                "live Nav2 frontier snapshot has invalid size",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            )
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _contract_error(
                "live Nav2 frontier snapshot is not valid UTF-8 JSON",
                "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
            ) from exc
        return LiveNav2FrontierSnapshot.from_mapping(value)


def build_live_bounded_advisor_request(
    camera_snapshot: ValidatedRevCSnapshot,
    frontier_snapshot: LiveNav2FrontierSnapshot,
    *,
    instruction: str,
    sim_now_s: float,
    max_frontier_age_s: float,
    visited_frontiers: Sequence[int] = (),
    compact_history: Sequence[str] = (),
    wall_timestamp_s: float | None = None,
) -> SlowPlannerRequest:
    """Bind a real frontier set to the unchanged SlowPlanner v1 request."""

    frontier_snapshot.require_current(
        sim_now_s=sim_now_s,
        max_age_s=max_frontier_age_s,
        expected_identity=camera_snapshot.identity,
    )
    if not frontier_snapshot.candidate_frontiers:
        raise _contract_error(
            "live Nav2 source has no current legal frontiers",
            "NO_CURRENT_LEGAL_FRONTIERS",
        )
    return build_lane_b_request(
        camera_snapshot,
        instruction=instruction,
        candidate_frontiers=frontier_snapshot.candidate_frontiers,
        agent_pose=frontier_snapshot.agent_pose,
        visited_frontiers=visited_frontiers,
        compact_history=compact_history,
        wall_timestamp_s=wall_timestamp_s,
    )


@dataclass(frozen=True)
class BoundedFrontierAdvice:
    """The only coordinator-consumable result: one current ID or abstain."""

    episode_id: str
    snapshot_id: str
    decision: str
    frontier_id: int | None
    reason: str
    frontier_set_sha256: str

    def __post_init__(self) -> None:
        if self.decision not in {"select_frontier", "abstain"}:
            raise ValueError("bounded frontier decision must select_frontier or abstain")
        if self.decision == "select_frontier":
            if isinstance(self.frontier_id, bool) or not isinstance(
                self.frontier_id, int
            ):
                raise ValueError("select_frontier requires an integer ID")
        elif self.frontier_id is not None:
            raise ValueError("abstain must not carry a frontier ID")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "episode_id": self.episode_id,
            "snapshot_id": self.snapshot_id,
            "decision": self.decision,
            "frontier_id": self.frontier_id,
            "reason": self.reason,
            "frontier_set_sha256": self.frontier_set_sha256,
            "motion_authority": "none",
            "terminal_stop_authority": "none",
        }


@dataclass(frozen=True)
class LiveBoundedAdvisorSubmission:
    """Private binding retained while one bounded Step3 request is in flight."""

    request: SlowPlannerRequest
    submitted_frontiers: LiveNav2FrontierSnapshot


def adjudicate_live_bounded_advice(
    request: SlowPlannerRequest,
    decision: PlannerDecision,
    *,
    submitted_frontiers: LiveNav2FrontierSnapshot,
    current_frontiers: LiveNav2FrontierSnapshot,
    current_identity: LaneBSnapshotIdentity,
    safe_stop_active: bool,
    sim_now_s: float,
    max_frontier_age_s: float,
) -> BoundedFrontierAdvice:
    """Revalidate source identity/content at consumption and return ID or abstain."""

    def abstain(reason: str) -> BoundedFrontierAdvice:
        return BoundedFrontierAdvice(
            episode_id=submitted_frontiers.identity.episode_id,
            snapshot_id=submitted_frontiers.identity.snapshot_id,
            decision="abstain",
            frontier_id=None,
            reason=reason,
            frontier_set_sha256=submitted_frontiers.frontier_set_sha256,
        )

    try:
        if safe_stop_active:
            raise _contract_error(
                "client safe-stop is active", "SAFE_STOP_ACTIVE"
            )
        if current_identity != submitted_frontiers.identity:
            raise _contract_error(
                "authoritative observation identity advanced while Step3 was evaluating",
                "STALE_LIVE_NAV2_FRONTIER_SOURCE",
            )
        submitted_frontiers.require_current(
            sim_now_s=sim_now_s,
            max_age_s=max_frontier_age_s,
            expected_identity=submitted_frontiers.identity,
        )
        if (
            request.episode_id != submitted_frontiers.identity.episode_id
            or request.snapshot_id != submitted_frontiers.identity.snapshot_id
            or request.candidate_frontiers
            != submitted_frontiers.candidate_frontiers
            or request.agent_pose != submitted_frontiers.agent_pose
        ):
            raise _contract_error(
                "SlowPlanner request is not bound to the submitted live frontier set",
                "REQUEST_FRONTIER_BINDING_MISMATCH",
            )
        current_frontiers.require_current(
            sim_now_s=sim_now_s,
            max_age_s=max_frontier_age_s,
            expected_identity=current_identity,
        )
        if (
            current_frontiers.frontier_set_sha256
            != submitted_frontiers.frontier_set_sha256
        ):
            raise _contract_error(
                "live frontier set changed while Step3 was evaluating",
                "STALE_LIVE_NAV2_FRONTIER_SOURCE",
            )
    except LiveFrontierContractError as exc:
        return abstain(exc.code.lower())

    try:
        outcome = LaneBPlannerAdapter(LaneBPlannerMode.BOUNDED_ADVISOR).resolve(
            request, decision
        )
    except (LaneBContractError, ValueError):
        return abstain("invalid_step3_response")
    if outcome.intent is not LaneBIntent.FRONTIER_ADVICE:
        return abstain(outcome.fallback_reason or "step3_abstain")
    legal_ids = {
        frontier.frontier_id for frontier in current_frontiers.candidate_frontiers
    }
    if outcome.frontier_id not in legal_ids:
        return abstain("frontier_not_current")
    return BoundedFrontierAdvice(
        episode_id=current_frontiers.identity.episode_id,
        snapshot_id=current_frontiers.identity.snapshot_id,
        decision="select_frontier",
        frontier_id=outcome.frontier_id,
        reason="identity_and_frontier_revalidated",
        frontier_set_sha256=current_frontiers.frontier_set_sha256,
    )


class LiveBoundedFrontierArbiter:
    """Read-before-request and re-read-before-consume coordinator boundary."""

    def __init__(
        self,
        source_file: LiveNav2FrontierFile,
        *,
        max_frontier_age_s: float,
    ) -> None:
        maximum_age = _finite(max_frontier_age_s, "max_frontier_age_s")
        if maximum_age <= 0.0:
            raise ValueError("max_frontier_age_s must be positive")
        self.source_file = source_file
        self.max_frontier_age_s = maximum_age

    def prepare(
        self,
        camera_snapshot: ValidatedRevCSnapshot,
        *,
        instruction: str,
        sim_now_s: float,
        visited_frontiers: Sequence[int] = (),
        compact_history: Sequence[str] = (),
        wall_timestamp_s: float | None = None,
    ) -> LiveBoundedAdvisorSubmission:
        submitted = self.source_file.read()
        request = build_live_bounded_advisor_request(
            camera_snapshot,
            submitted,
            instruction=instruction,
            sim_now_s=sim_now_s,
            max_frontier_age_s=self.max_frontier_age_s,
            visited_frontiers=visited_frontiers,
            compact_history=compact_history,
            wall_timestamp_s=wall_timestamp_s,
        )
        return LiveBoundedAdvisorSubmission(request, submitted)

    def consume(
        self,
        submission: LiveBoundedAdvisorSubmission,
        decision: PlannerDecision,
        *,
        current_identity: LaneBSnapshotIdentity,
        safe_stop_active: bool,
        sim_now_s: float,
    ) -> BoundedFrontierAdvice:
        try:
            current = self.source_file.read()
        except LiveFrontierContractError as exc:
            return BoundedFrontierAdvice(
                episode_id=submission.submitted_frontiers.identity.episode_id,
                snapshot_id=submission.submitted_frontiers.identity.snapshot_id,
                decision="abstain",
                frontier_id=None,
                reason=exc.code.lower(),
                frontier_set_sha256=(
                    submission.submitted_frontiers.frontier_set_sha256
                ),
            )
        return adjudicate_live_bounded_advice(
            submission.request,
            decision,
            submitted_frontiers=submission.submitted_frontiers,
            current_frontiers=current,
            current_identity=current_identity,
            safe_stop_active=safe_stop_active,
            sim_now_s=sim_now_s,
            max_frontier_age_s=self.max_frontier_age_s,
        )
