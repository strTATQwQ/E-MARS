"""Identity-bound absolute Nav2 targets for cached System 1 actions."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FrozenSystem1Target:
    """A map/odom-frame target that is safe to reissue after a gate stop.

    The source path itself is deliberately not retained.  It was expressed in
    the robot's old ``base_link`` frame and would be unsafe to reinterpret from
    a later pose.  Instead, the adapter retains only the absolute Nav2 target
    produced from that path, together with its episode/reset/path identity.
    """

    episode_id: str
    reset_generation: int
    source_sequence_id: int
    source_path_sha256: str
    valid_until_ns: int
    frame_id: str
    position_xyz: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    target_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.episode_id or self.reset_generation < 0 or self.source_sequence_id < 0:
            raise ValueError("invalid frozen System 1 target identity")
        if not _SHA256.fullmatch(self.source_path_sha256):
            raise ValueError("invalid frozen System 1 source path hash")
        if self.valid_until_ns <= 0:
            raise ValueError("invalid frozen System 1 target expiry")
        if not self.frame_id or self.frame_id == "base_link":
            raise ValueError("System 1 target must use an absolute Nav2 frame")
        values = (*self.position_xyz, *self.orientation_xyzw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("frozen System 1 target contains NaN/Inf")
        norm = math.sqrt(sum(value * value for value in self.orientation_xyzw))
        if norm <= 1e-9:
            raise ValueError("frozen System 1 target orientation is invalid")
        payload = {
            "episode_id": self.episode_id,
            "frame_id": self.frame_id,
            "orientation_xyzw": list(self.orientation_xyzw),
            "position_xyz": list(self.position_xyz),
            "reset_generation": self.reset_generation,
            "source_path_sha256": self.source_path_sha256,
            "source_sequence_id": self.source_sequence_id,
            "valid_until_ns": self.valid_until_ns,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        object.__setattr__(self, "target_sha256", hashlib.sha256(encoded).hexdigest())

    def require_queue_binding(
        self,
        *,
        episode_id: str,
        reset_generation: int,
        sequence_id: int,
        now_ns: int,
    ) -> None:
        if episode_id != self.episode_id or reset_generation != self.reset_generation:
            raise ValueError("System 1 queue target crossed an episode/reset boundary")
        if sequence_id <= self.source_sequence_id:
            raise ValueError("System 1 queue target sequence did not advance")
        if now_ns <= 0 or now_ns > self.valid_until_ns:
            raise ValueError("System 1 queue target expired; fresh trajectory required")

    def remaining_xy_distance(self, current_x: float, current_y: float) -> float:
        """Return bounded-plan distance remaining in the target's absolute frame."""

        values = (float(current_x), float(current_y))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("current System 1 target pose contains NaN/Inf")
        return math.hypot(
            self.position_xyz[0] - values[0],
            self.position_xyz[1] - values[1],
        )

    def permits_bounded_reissue(
        self,
        current_x: float,
        current_y: float,
        minimum_remaining_m: float,
    ) -> bool:
        """Reject a reissue that cannot contain one more bounded motion step."""

        minimum_remaining_m = float(minimum_remaining_m)
        if not math.isfinite(minimum_remaining_m) or minimum_remaining_m <= 0.0:
            raise ValueError("bounded System 1 reissue distance is invalid")
        return (
            self.remaining_xy_distance(current_x, current_y) + 1e-6
            >= minimum_remaining_m
        )
