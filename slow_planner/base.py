from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = 1
DECISIONS = frozenset({"select_frontier", "target_found", "abstain"})


class SlowPlannerProtocolError(ValueError):
    """Raised when a request or decision violates the frozen benchmark contract."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SlowPlannerProtocolError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SlowPlannerProtocolError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise SlowPlannerProtocolError(f"{name} must be finite")
    return number


def _finite_tuple(value: Sequence[Any], name: str, *, length: int | None = None) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise SlowPlannerProtocolError(f"{name} must be an array")
    result = tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
    if length is not None and len(result) != length:
        raise SlowPlannerProtocolError(f"{name} must contain exactly {length} values")
    return result


@dataclass(frozen=True)
class OrderedImage:
    view_id: str
    pose: tuple[float, ...]
    jpeg: bytes
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.view_id.strip():
            raise SlowPlannerProtocolError("ordered image view_id is required")
        if not isinstance(self.jpeg, bytes) or not self.jpeg:
            raise SlowPlannerProtocolError(f"image {self.view_id!r} must contain JPEG bytes")
        if self.width <= 0 or self.height <= 0:
            raise SlowPlannerProtocolError(f"image {self.view_id!r} dimensions must be positive")
        _finite_tuple(self.pose, f"image[{self.view_id}].pose")

    def metadata(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "pose": list(self.pose),
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class CandidateFrontier:
    frontier_id: int
    relative_xz: tuple[float, float]
    distance_m: float
    bearing_deg: float

    def __post_init__(self) -> None:
        if isinstance(self.frontier_id, bool) or self.frontier_id < 0:
            raise SlowPlannerProtocolError("frontier_id must be a non-negative integer")
        _finite_tuple(self.relative_xz, f"frontier[{self.frontier_id}].relative_xz", length=2)
        if _finite(self.distance_m, f"frontier[{self.frontier_id}].distance_m") < 0:
            raise SlowPlannerProtocolError("frontier distance must be non-negative")
        _finite(self.bearing_deg, f"frontier[{self.frontier_id}].bearing_deg")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateFrontier":
        frontier_id = value.get("frontier_id")
        if isinstance(frontier_id, bool):
            raise SlowPlannerProtocolError("frontier_id must be an integer")
        try:
            frontier_id = int(frontier_id)
        except (TypeError, ValueError) as exc:
            raise SlowPlannerProtocolError("frontier_id must be an integer") from exc
        relative = _finite_tuple(value.get("relative_xz") or (), "relative_xz", length=2)
        return cls(
            frontier_id=frontier_id,
            relative_xz=(relative[0], relative[1]),
            distance_m=_finite(value.get("distance_m"), "distance_m"),
            bearing_deg=_finite(value.get("bearing_deg"), "bearing_deg"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "frontier_id": self.frontier_id,
            "relative_xz": list(self.relative_xz),
            "distance_m": self.distance_m,
            "bearing_deg": self.bearing_deg,
        }


@dataclass(frozen=True)
class SlowPlannerRequest:
    episode_id: str
    snapshot_id: str
    instruction: str
    ordered_images: tuple[OrderedImage, ...]
    candidate_frontiers: tuple[CandidateFrontier, ...]
    agent_pose: tuple[float, ...]
    visited_frontiers: tuple[int, ...] = ()
    compact_history: tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise SlowPlannerProtocolError(f"unsupported protocol_version={self.protocol_version}")
        if not self.episode_id.strip():
            raise SlowPlannerProtocolError("episode_id is required")
        if not self.snapshot_id.strip():
            raise SlowPlannerProtocolError("snapshot_id is required")
        if not self.instruction.strip():
            raise SlowPlannerProtocolError("instruction is required")
        if not self.ordered_images:
            raise SlowPlannerProtocolError("at least one ordered image is required")
        view_ids = [image.view_id for image in self.ordered_images]
        if len(view_ids) != len(set(view_ids)):
            raise SlowPlannerProtocolError("ordered image view_id values must be unique")
        frontier_ids = [item.frontier_id for item in self.candidate_frontiers]
        if len(frontier_ids) != len(set(frontier_ids)):
            raise SlowPlannerProtocolError("candidate frontier IDs must be unique")
        _finite_tuple(self.agent_pose, "agent_pose")
        _finite(self.timestamp, "timestamp")
        for value in self.visited_frontiers:
            if isinstance(value, bool) or int(value) < 0:
                raise SlowPlannerProtocolError("visited frontier IDs must be non-negative integers")
        if any(not isinstance(item, str) for item in self.compact_history):
            raise SlowPlannerProtocolError("compact_history must contain strings")

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "episode_id": self.episode_id,
            "snapshot_id": self.snapshot_id,
            "instruction": self.instruction,
            "ordered_images": [image.metadata() for image in self.ordered_images],
            "candidate_frontiers": [frontier.to_mapping() for frontier in self.candidate_frontiers],
            "agent_pose": list(self.agent_pose),
            "visited_frontiers": list(self.visited_frontiers),
            "compact_history": list(self.compact_history),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_wire(cls, metadata: Mapping[str, Any], image_parts: Sequence[bytes]) -> "SlowPlannerRequest":
        image_rows = metadata.get("ordered_images") or []
        if len(image_rows) != len(image_parts):
            raise SlowPlannerProtocolError("ordered image metadata/payload count mismatch")
        images = []
        for row, payload in zip(image_rows, image_parts):
            if not isinstance(row, Mapping):
                raise SlowPlannerProtocolError("ordered image metadata must be objects")
            images.append(
                OrderedImage(
                    view_id=str(row.get("view_id") or ""),
                    pose=_finite_tuple(row.get("pose") or (), "image.pose"),
                    jpeg=bytes(payload),
                    width=int(row.get("width") or 0),
                    height=int(row.get("height") or 0),
                )
            )
        frontiers_raw = metadata.get("candidate_frontiers") or []
        frontiers = tuple(CandidateFrontier.from_mapping(row) for row in frontiers_raw)
        visited_raw = metadata.get("visited_frontiers") or []
        return cls(
            episode_id=str(metadata.get("episode_id") or ""),
            snapshot_id=str(metadata.get("snapshot_id") or ""),
            instruction=str(metadata.get("instruction") or ""),
            ordered_images=tuple(images),
            candidate_frontiers=frontiers,
            agent_pose=_finite_tuple(metadata.get("agent_pose") or (), "agent_pose"),
            visited_frontiers=tuple(int(value) for value in visited_raw),
            compact_history=tuple(str(value) for value in (metadata.get("compact_history") or [])),
            timestamp=_finite(metadata.get("timestamp"), "timestamp"),
            protocol_version=int(metadata.get("protocol_version", PROTOCOL_VERSION)),
        )


@dataclass(frozen=True)
class PlannerDecision:
    episode_id: str
    snapshot_id: str
    decision: str
    frontier_id: int | None
    target_relative_xz: tuple[float, float] | None
    confidence: float
    raw_text: str = ""
    parse_attempts: int = 1
    fallback_used: bool = False
    fallback_reason: str = ""
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise SlowPlannerProtocolError("decision protocol version mismatch")
        if not self.episode_id or not self.snapshot_id:
            raise SlowPlannerProtocolError("decision must echo episode_id and snapshot_id")
        if self.decision not in DECISIONS:
            raise SlowPlannerProtocolError(f"invalid decision={self.decision!r}")
        confidence = _finite(self.confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise SlowPlannerProtocolError("confidence must be in [0, 1]")
        if self.decision == "select_frontier":
            if isinstance(self.frontier_id, bool) or not isinstance(self.frontier_id, int):
                raise SlowPlannerProtocolError("select_frontier requires integer frontier_id")
            if self.target_relative_xz is not None:
                raise SlowPlannerProtocolError("select_frontier must not include target_relative_xz")
        elif self.decision == "target_found":
            if self.frontier_id is not None:
                raise SlowPlannerProtocolError("target_found must not include frontier_id")
            if self.target_relative_xz is None:
                raise SlowPlannerProtocolError("target_found requires target_relative_xz")
            _finite_tuple(self.target_relative_xz, "target_relative_xz", length=2)
        elif self.frontier_id is not None or self.target_relative_xz is not None:
            raise SlowPlannerProtocolError("abstain must not include frontier_id or target_relative_xz")
        if self.parse_attempts < 1:
            raise SlowPlannerProtocolError("parse_attempts must be positive")

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        if self.target_relative_xz is not None:
            value["target_relative_xz"] = list(self.target_relative_xz)
        return value


@dataclass(frozen=True)
class StructuredPlannerDecision(PlannerDecision):
    """Validated Step3 semantic summary carried beside the legacy wire decision."""

    scene_summary: str = ""
    target_evidence: tuple[str, ...] = ()
    blocked_directions: tuple[str, ...] = ()
    recommended_frontier: int | None = None
    target_found: bool = False
    abstain: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.scene_summary, str):
            raise SlowPlannerProtocolError("scene_summary must be a string")
        if any(not isinstance(item, str) for item in self.target_evidence):
            raise SlowPlannerProtocolError("target_evidence must contain strings")
        if any(not isinstance(item, str) for item in self.blocked_directions):
            raise SlowPlannerProtocolError("blocked_directions must contain strings")
        if not isinstance(self.target_found, bool) or not isinstance(self.abstain, bool):
            raise SlowPlannerProtocolError("target_found and abstain must be booleans")
        if self.decision == "select_frontier":
            if self.abstain or self.target_found:
                raise SlowPlannerProtocolError(
                    "frontier selection cannot also be target_found or abstain"
                )
            if self.recommended_frontier != self.frontier_id:
                raise SlowPlannerProtocolError(
                    "recommended_frontier must match selected frontier_id"
                )
        elif self.decision == "abstain":
            if not self.abstain or self.recommended_frontier is not None:
                raise SlowPlannerProtocolError(
                    "abstain requires null recommended_frontier"
                )
        else:
            raise SlowPlannerProtocolError(
                "structured Step3 decisions may only select a frontier or abstain"
            )


@dataclass(frozen=True)
class PlannerMetrics:
    image_count: int = 0
    raw_image_resolutions: tuple[tuple[int, int], ...] = ()
    visual_token_count: int = 0
    input_token_count: int = 0
    output_token_count: int = 0
    image_decode_ms: float = 0.0
    prompt_template_ms: float = 0.0
    processor_ms: float = 0.0
    preprocessing_ms: float = 0.0
    prefill_ttft_ms: float = 0.0
    decode_ms: float = 0.0
    decode_tokens_per_s: float = 0.0
    model_generate_ms: float = 0.0
    end_to_end_ms: float = 0.0
    peak_memory_mib: float = 0.0
    network_ms: float = 0.0
    retry_count: int = 0
    model_variant: str = ""
    precision_mode: str = "bf16"

    def merged_attempt(self, other: "PlannerMetrics") -> "PlannerMetrics":
        """Accumulate retry cost while retaining the final attempt's token/image metadata."""
        return replace(
            other,
            image_decode_ms=self.image_decode_ms + other.image_decode_ms,
            prompt_template_ms=self.prompt_template_ms + other.prompt_template_ms,
            processor_ms=self.processor_ms + other.processor_ms,
            preprocessing_ms=self.preprocessing_ms + other.preprocessing_ms,
            prefill_ttft_ms=self.prefill_ttft_ms + other.prefill_ttft_ms,
            decode_ms=self.decode_ms + other.decode_ms,
            model_generate_ms=self.model_generate_ms + other.model_generate_ms,
            end_to_end_ms=self.end_to_end_ms + other.end_to_end_ms,
            peak_memory_mib=max(self.peak_memory_mib, other.peak_memory_mib),
            retry_count=self.retry_count + other.retry_count,
        )

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["raw_image_resolutions"] = [list(item) for item in self.raw_image_resolutions]
        return value


class SlowPlanner(ABC):
    model_variant: str
    precision_mode: str = "bf16"

    @abstractmethod
    def generate_raw(
        self,
        request: SlowPlannerRequest,
        *,
        correction: str = "",
    ) -> tuple[str, PlannerMetrics]:
        raise NotImplementedError

    def health(self) -> dict[str, Any]:
        return {
            "ready": True,
            "model_variant": self.model_variant,
            "precision_mode": self.precision_mode,
            "protocol_version": PROTOCOL_VERSION,
        }

    def parse_decision(
        self, request: SlowPlannerRequest, raw_text: str, *, attempts: int
    ) -> PlannerDecision:
        return parse_decision(request, raw_text, attempts=attempts)

    def close(self) -> None:
        return None


def _json_objects(text: str):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def strict_decision_json_object(raw_text: str) -> dict[str, Any]:
    """Decode one exact decision object with no prefix, suffix, or duplicate keys."""

    text = raw_text.strip()
    if not text.startswith("{"):
        raise SlowPlannerProtocolError("Step3 response must start with a JSON object")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise SlowPlannerProtocolError(f"response repeats key: {key}")
            value[key] = item
        return value

    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    try:
        value, end = decoder.raw_decode(text)
    except json.JSONDecodeError as exc:
        raise SlowPlannerProtocolError("Step3 response is not complete JSON") from exc
    if text[end:].strip():
        raise SlowPlannerProtocolError("Step3 response must contain only one JSON object")
    if not isinstance(value, dict):
        raise SlowPlannerProtocolError("Step3 response must be a JSON object")
    expected = {"decision", "frontier_id", "target_relative_xz", "confidence"}
    if set(value) != expected:
        raise SlowPlannerProtocolError("Step3 response must contain exactly four decision keys")
    return value


def parse_strict_decision(
    request: SlowPlannerRequest, raw_text: str, *, attempts: int
) -> PlannerDecision:
    strict_decision_json_object(raw_text)
    return parse_decision(request, raw_text.strip(), attempts=attempts)


def parse_decision(request: SlowPlannerRequest, raw_text: str, *, attempts: int) -> PlannerDecision:
    objects = list(_json_objects(raw_text.strip()))
    if not objects:
        raise SlowPlannerProtocolError("response does not contain a JSON object")
    value = objects[-1]
    allowed = {"decision", "frontier_id", "target_relative_xz", "confidence"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SlowPlannerProtocolError(f"response contains unknown keys: {unknown}")
    decision = str(value.get("decision") or "")
    frontier_id_raw = value.get("frontier_id")
    frontier_id: int | None
    if frontier_id_raw is None:
        frontier_id = None
    elif isinstance(frontier_id_raw, bool) or not isinstance(frontier_id_raw, int):
        raise SlowPlannerProtocolError("frontier_id must be an integer or null")
    else:
        frontier_id = frontier_id_raw
    target_raw = value.get("target_relative_xz")
    target: tuple[float, float] | None = None
    if target_raw is not None:
        parsed = _finite_tuple(target_raw, "target_relative_xz", length=2)
        target = (parsed[0], parsed[1])
    result = PlannerDecision(
        episode_id=request.episode_id,
        snapshot_id=request.snapshot_id,
        decision=decision,
        frontier_id=frontier_id,
        target_relative_xz=target,
        confidence=_finite(value.get("confidence"), "confidence"),
        raw_text=raw_text,
        parse_attempts=attempts,
    )
    valid_ids = {item.frontier_id for item in request.candidate_frontiers}
    if result.decision == "select_frontier" and result.frontier_id not in valid_ids:
        raise SlowPlannerProtocolError(f"frontier_id={result.frontier_id} is not a current candidate")
    return result


def deterministic_fallback(
    request: SlowPlannerRequest,
    *,
    raw_text: str,
    reason: str,
    attempts: int,
) -> PlannerDecision:
    visited = set(request.visited_frontiers)
    candidates = sorted(
        (item for item in request.candidate_frontiers if item.frontier_id not in visited),
        key=lambda item: (item.frontier_id, item.distance_m),
    )
    # A graph dead-end still needs a deterministic backtracking choice.
    if not candidates:
        candidates = sorted(
            request.candidate_frontiers,
            key=lambda item: (item.frontier_id, item.distance_m),
        )
    if candidates:
        return PlannerDecision(
            episode_id=request.episode_id,
            snapshot_id=request.snapshot_id,
            decision="select_frontier",
            frontier_id=candidates[0].frontier_id,
            target_relative_xz=None,
            confidence=0.0,
            raw_text=raw_text,
            parse_attempts=attempts,
            fallback_used=True,
            fallback_reason=reason,
        )
    return PlannerDecision(
        episode_id=request.episode_id,
        snapshot_id=request.snapshot_id,
        decision="abstain",
        frontier_id=None,
        target_relative_xz=None,
        confidence=0.0,
        raw_text=raw_text,
        parse_attempts=attempts,
        fallback_used=True,
        fallback_reason=reason,
    )


class PlannerRunner:
    """Schema validation, one bounded correction retry, and deterministic fallback."""

    def __init__(self, planner: SlowPlanner, *, max_retries: int = 1) -> None:
        if max_retries < 0 or max_retries > 2:
            raise ValueError("max_retries must be between 0 and 2")
        self.planner = planner
        self.max_retries = max_retries

    def decide(self, request: SlowPlannerRequest) -> tuple[PlannerDecision, PlannerMetrics]:
        started = time.perf_counter()
        aggregate: PlannerMetrics | None = None
        raw_text = ""
        correction = ""
        last_error = ""
        attempts = 0
        for attempts in range(1, self.max_retries + 2):
            raw_text, metrics = self.planner.generate_raw(request, correction=correction)
            metrics = replace(metrics, retry_count=attempts - 1)
            aggregate = metrics if aggregate is None else aggregate.merged_attempt(metrics)
            try:
                decision = self.planner.parse_decision(
                    request, raw_text, attempts=attempts
                )
                elapsed = (time.perf_counter() - started) * 1000.0
                return decision, replace(aggregate, end_to_end_ms=elapsed, retry_count=attempts - 1)
            except SlowPlannerProtocolError as exc:
                last_error = str(exc)
                correction = (
                    "Your previous response was invalid: "
                    + last_error
                    + ". Return only one JSON object matching the required schema; do not add keys or prose."
                )
        assert aggregate is not None
        fallback = deterministic_fallback(
            request,
            raw_text=raw_text,
            reason=f"schema_failure:{last_error}",
            attempts=attempts,
        )
        return fallback, replace(
            aggregate,
            end_to_end_ms=(time.perf_counter() - started) * 1000.0,
            retry_count=max(0, attempts - 1),
        )
