from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    """Raised when a navigation request or response violates the wire contract."""


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{name} must be numeric") from exc
    if not isfinite(number):
        raise ProtocolError(f"{name} must be finite")
    return number


def _image_bytes(value: Any, name: str, *, required: bool) -> bytes | None:
    if value is None and not required:
        return None
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes) or not value:
        requirement = "non-empty bytes" if required else "bytes or null"
        raise ProtocolError(f"{name} must be {requirement}")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be an object")
    return dict(value)


@dataclass(frozen=True)
class NavigationRequest:
    episode_id: str
    frame_id: int
    timestamp: float
    instruction: str
    rgb_front: bytes
    rgb_left: bytes | None = None
    rgb_right: bytes | None = None
    optional_depth: bytes | None = None
    agent_pose: tuple[float, ...] = ()
    last_action: Mapping[str, Any] = field(default_factory=dict)
    collision_state: Mapping[str, Any] = field(default_factory=dict)
    reset_episode: bool = False
    protocol_version: int = PROTOCOL_VERSION

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "NavigationRequest":
        if not isinstance(payload, Mapping):
            raise ProtocolError("request must be an object")
        episode_id = str(payload.get("episode_id") or "").strip()
        if not episode_id:
            raise ProtocolError("episode_id is required")
        try:
            frame_id = int(payload.get("frame_id"))
        except (TypeError, ValueError) as exc:
            raise ProtocolError("frame_id must be an integer") from exc
        if frame_id < 0:
            raise ProtocolError("frame_id must be non-negative")
        version = int(payload.get("protocol_version", PROTOCOL_VERSION))
        if version != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol_version={version}")
        instruction = str(payload.get("instruction") or "").strip()
        if not instruction:
            raise ProtocolError("instruction is required")
        pose_raw = payload.get("agent_pose") or ()
        if not isinstance(pose_raw, Sequence) or isinstance(pose_raw, (str, bytes, bytearray)):
            raise ProtocolError("agent_pose must be an array")
        pose = tuple(_finite_number(value, f"agent_pose[{idx}]") for idx, value in enumerate(pose_raw))
        return cls(
            episode_id=episode_id,
            frame_id=frame_id,
            timestamp=_finite_number(payload.get("timestamp"), "timestamp"),
            instruction=instruction,
            rgb_front=_image_bytes(payload.get("rgb_front"), "rgb_front", required=True) or b"",
            rgb_left=_image_bytes(payload.get("rgb_left"), "rgb_left", required=False),
            rgb_right=_image_bytes(payload.get("rgb_right"), "rgb_right", required=False),
            optional_depth=_image_bytes(payload.get("optional_depth"), "optional_depth", required=False),
            agent_pose=pose,
            last_action=_mapping(payload.get("last_action"), "last_action"),
            collision_state=_mapping(payload.get("collision_state"), "collision_state"),
            reset_episode=bool(payload.get("reset_episode", False)),
            protocol_version=version,
        )

    def metadata(self) -> dict[str, Any]:
        """Return log-safe metadata without copying image/depth payloads."""
        return {
            "protocol_version": self.protocol_version,
            "episode_id": self.episode_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "instruction": self.instruction,
            "has_left": self.rgb_left is not None,
            "has_right": self.rgb_right is not None,
            "has_depth": self.optional_depth is not None,
            "agent_pose": list(self.agent_pose),
            "last_action": dict(self.last_action),
            "collision_state": dict(self.collision_state),
            "reset_episode": self.reset_episode,
        }


@dataclass(frozen=True)
class NavigationOutput:
    episode_id: str
    frame_id: int
    waypoints: tuple[tuple[float, float], ...]
    heading_sin_cos: tuple[tuple[float, float], ...]
    arrive_or_stop: bool
    confidence: float
    model_latency_ms: float
    vision_latency_ms: float = 0.0
    server_total_latency_ms: float = 0.0
    request_timestamp: float = 0.0
    model_variant: str = ""
    precision_mode: str = "bf16"
    arrive_logits: tuple[float, ...] = ()
    cache_hit: bool = False
    safe_stop_reason: str = ""
    action_head_trained: bool = False
    peak_memory_mib: float = 0.0
    coordinate_frame: str = "base_link_x_forward_y_left"
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ProtocolError("response episode_id is required")
        if self.frame_id < 0:
            raise ProtocolError("response frame_id must be non-negative")
        if len(self.waypoints) != len(self.heading_sin_cos):
            raise ProtocolError("waypoints and heading_sin_cos must have equal length")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ProtocolError("confidence must be in [0, 1]")
        for name, value in (
            ("model_latency_ms", self.model_latency_ms),
            ("vision_latency_ms", self.vision_latency_ms),
            ("server_total_latency_ms", self.server_total_latency_ms),
            ("request_timestamp", self.request_timestamp),
            ("peak_memory_mib", self.peak_memory_mib),
        ):
            _finite_number(value, name)
        for idx, (x, y) in enumerate(self.waypoints):
            _finite_number(x, f"waypoints[{idx}][0]")
            _finite_number(y, f"waypoints[{idx}][1]")
        for idx, (sin_value, cos_value) in enumerate(self.heading_sin_cos):
            _finite_number(sin_value, f"heading_sin_cos[{idx}][0]")
            _finite_number(cos_value, f"heading_sin_cos[{idx}][1]")

    @classmethod
    def safe_stop(
        cls,
        *,
        episode_id: str,
        frame_id: int,
        reason: str,
        request_timestamp: float = 0.0,
        model_variant: str = "",
        precision_mode: str = "bf16",
    ) -> "NavigationOutput":
        return cls(
            episode_id=episode_id,
            frame_id=frame_id,
            waypoints=(),
            heading_sin_cos=(),
            arrive_or_stop=True,
            confidence=1.0,
            model_latency_ms=0.0,
            request_timestamp=request_timestamp,
            model_variant=model_variant,
            precision_mode=precision_mode,
            safe_stop_reason=reason,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "episode_id": self.episode_id,
            "frame_id": self.frame_id,
            "waypoints": [list(point) for point in self.waypoints],
            "heading_sin_cos": [list(value) for value in self.heading_sin_cos],
            "arrive_or_stop": self.arrive_or_stop,
            "confidence": self.confidence,
            "model_latency_ms": self.model_latency_ms,
            "vision_latency_ms": self.vision_latency_ms,
            "server_total_latency_ms": self.server_total_latency_ms,
            "request_timestamp": self.request_timestamp,
            "model_variant": self.model_variant,
            "precision_mode": self.precision_mode,
            "arrive_logits": list(self.arrive_logits),
            "cache_hit": self.cache_hit,
            "safe_stop_reason": self.safe_stop_reason,
            "action_head_trained": self.action_head_trained,
            "peak_memory_mib": self.peak_memory_mib,
            "coordinate_frame": self.coordinate_frame,
        }
