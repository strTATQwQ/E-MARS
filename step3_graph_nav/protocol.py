from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = 1


class GraphNavProtocolError(ValueError):
    """Raised when a graph-nav request or raw model action violates v1."""


class InvalidCandidateId(GraphNavProtocolError):
    """Raised separately because canary must immediately stop on any invalid ID."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise GraphNavProtocolError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GraphNavProtocolError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise GraphNavProtocolError(f"{name} must be finite")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GraphNavProtocolError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class CandidateView:
    candidate_id: int
    target_viewpoint_id: int
    relative_heading_deg: float
    graph_distance_m: float
    jpeg: bytes
    width: int
    height: int

    def __post_init__(self) -> None:
        if _integer(self.candidate_id, "candidate_id") < 0:
            raise GraphNavProtocolError("candidate_id must be non-negative")
        if _integer(self.target_viewpoint_id, "target_viewpoint_id") < 0:
            raise GraphNavProtocolError("target_viewpoint_id must be non-negative")
        heading = _finite(self.relative_heading_deg, "relative_heading_deg")
        if heading < -180.000001 or heading > 180.000001:
            raise GraphNavProtocolError("relative_heading_deg must be normalized to [-180, 180]")
        if _finite(self.graph_distance_m, "graph_distance_m") <= 0.0:
            raise GraphNavProtocolError("graph_distance_m must be positive")
        if not isinstance(self.jpeg, bytes) or not self.jpeg:
            raise GraphNavProtocolError("candidate JPEG must be non-empty bytes")
        if self.width <= 0 or self.height <= 0:
            raise GraphNavProtocolError("candidate image dimensions must be positive")

    def metadata(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target_viewpoint_id": self.target_viewpoint_id,
            "relative_heading_deg": self.relative_heading_deg,
            "graph_distance_m": self.graph_distance_m,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class GraphNavRequest:
    episode_id: str
    snapshot_id: str
    instruction: str
    current_viewpoint_id: int
    step_index: int
    candidates: tuple[CandidateView, ...]
    history: tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise GraphNavProtocolError(f"unsupported protocol_version={self.protocol_version}")
        if not self.episode_id.strip() or not self.snapshot_id.strip():
            raise GraphNavProtocolError("episode_id and snapshot_id are required")
        if not self.instruction.strip():
            raise GraphNavProtocolError("instruction is required")
        if _integer(self.current_viewpoint_id, "current_viewpoint_id") < 0:
            raise GraphNavProtocolError("current_viewpoint_id must be non-negative")
        if _integer(self.step_index, "step_index") < 0:
            raise GraphNavProtocolError("step_index must be non-negative")
        if not self.candidates:
            raise GraphNavProtocolError("at least one candidate is required")
        candidate_ids = [item.candidate_id for item in self.candidates]
        if candidate_ids != list(range(len(candidate_ids))):
            raise GraphNavProtocolError("candidate IDs must be fixed zero-based ordinals")
        targets = [item.target_viewpoint_id for item in self.candidates]
        if targets != sorted(targets) or len(targets) != len(set(targets)):
            raise GraphNavProtocolError("candidate targets must be unique and sorted")
        if any(not isinstance(item, str) for item in self.history):
            raise GraphNavProtocolError("history must contain strings")
        _finite(self.timestamp, "timestamp")

    @property
    def candidate_ids(self) -> frozenset[int]:
        return frozenset(item.candidate_id for item in self.candidates)

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "episode_id": self.episode_id,
            "snapshot_id": self.snapshot_id,
            "instruction": self.instruction,
            "current_viewpoint_id": self.current_viewpoint_id,
            "step_index": self.step_index,
            "candidates": [item.metadata() for item in self.candidates],
            "history": list(self.history),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_wire(cls, metadata: Mapping[str, Any], image_parts: Sequence[bytes]) -> "GraphNavRequest":
        rows = metadata.get("candidates") or []
        if not isinstance(rows, list) or len(rows) != len(image_parts):
            raise GraphNavProtocolError("candidate metadata/payload count mismatch")
        candidates = []
        for row, payload in zip(rows, image_parts):
            if not isinstance(row, Mapping):
                raise GraphNavProtocolError("candidate metadata must be objects")
            candidates.append(
                CandidateView(
                    candidate_id=_integer(row.get("candidate_id"), "candidate_id"),
                    target_viewpoint_id=_integer(row.get("target_viewpoint_id"), "target_viewpoint_id"),
                    relative_heading_deg=_finite(row.get("relative_heading_deg"), "relative_heading_deg"),
                    graph_distance_m=_finite(row.get("graph_distance_m"), "graph_distance_m"),
                    jpeg=bytes(payload),
                    width=int(row.get("width") or 0),
                    height=int(row.get("height") or 0),
                )
            )
        return cls(
            episode_id=str(metadata.get("episode_id") or ""),
            snapshot_id=str(metadata.get("snapshot_id") or ""),
            instruction=str(metadata.get("instruction") or ""),
            current_viewpoint_id=_integer(metadata.get("current_viewpoint_id"), "current_viewpoint_id"),
            step_index=_integer(metadata.get("step_index"), "step_index"),
            candidates=tuple(candidates),
            history=tuple(str(item) for item in (metadata.get("history") or [])),
            timestamp=_finite(metadata.get("timestamp"), "timestamp"),
            protocol_version=int(metadata.get("protocol_version", PROTOCOL_VERSION)),
        )


@dataclass(frozen=True)
class GraphNavAction:
    action: str
    candidate_id: int | None = None

    def __post_init__(self) -> None:
        if self.action == "stop":
            if self.candidate_id is not None:
                raise GraphNavProtocolError("stop must not contain candidate_id")
        elif self.action == "move":
            if self.candidate_id is None:
                raise GraphNavProtocolError("move requires candidate_id")
            _integer(self.candidate_id, "candidate_id")
        else:
            raise GraphNavProtocolError("action must be move or stop")

    def to_mapping(self) -> dict[str, Any]:
        if self.action == "stop":
            return {"action": "stop"}
        return {"action": "move", "candidate_id": self.candidate_id}


def parse_action(raw_text: str, valid_candidate_ids: frozenset[int]) -> GraphNavAction:
    try:
        value = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        raise GraphNavProtocolError(f"raw output is not one JSON object: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise GraphNavProtocolError("raw output must be one JSON object")
    action = value.get("action")
    if action == "stop":
        if set(value) != {"action"}:
            raise GraphNavProtocolError("stop JSON must contain exactly the action key")
        return GraphNavAction(action="stop")
    if action == "move":
        if set(value) != {"action", "candidate_id"}:
            raise GraphNavProtocolError("move JSON must contain exactly action and candidate_id")
        candidate_id = _integer(value.get("candidate_id"), "candidate_id")
        if candidate_id not in valid_candidate_ids:
            raise InvalidCandidateId(f"candidate_id={candidate_id} is not a current candidate")
        return GraphNavAction(action="move", candidate_id=candidate_id)
    raise GraphNavProtocolError("action must be move or stop")
