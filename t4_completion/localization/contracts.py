"""Frozen dependency-free contracts for the T4.3 localization selector."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


CUVSLAM = "cuvslam"
LIDAR_IMU = "lidar_imu"
ISAAC_GT = "isaac_ground_truth_pose"
SENSOR_ODOMETRY_SOURCES = (CUVSLAM, LIDAR_IMU)
ALL_SOURCES = (*SENSOR_ODOMETRY_SOURCES, ISAAC_GT)

CANONICAL_ODOMETRY_TOPIC = "/odom"
CANONICAL_TF_TOPIC = "/tf"
CANONICAL_PARENT_FRAME = "odom"
CANONICAL_CHILD_FRAME = "base_link"
SOURCE_HEALTH_TOPIC = "/internvla_t4/localization/source_health"
SWITCH_EVENT_TOPIC = "/internvla_t4/localization/switch_event"
DEVIATION_TOPIC = "/internvla_t4/localization/deviation"


def _finite(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


def quaternion_norm(value: tuple[float, float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def quaternion_normalize(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = quaternion_norm(value)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("quaternion_zero_or_nonfinite")
    return tuple(component / norm for component in value)  # type: ignore[return-value]


def quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return quaternion_normalize(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def quaternion_inverse(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, z, w = quaternion_normalize(value)
    return (-x, -y, -z, w)


def quaternion_rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z, w = quaternion_normalize(quaternion)
    vx, vy, vz = vector
    # Equivalent to q * (v, 0) * inverse(q), without normalizing the vector
    # quaternion (which would incorrectly discard translation magnitude).
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def quaternion_distance_rad(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    relative = quaternion_multiply(quaternion_inverse(left), right)
    vector_norm = math.sqrt(sum(component * component for component in relative[:3]))
    # atan2 remains well-conditioned for identical orientations, where acos
    # of a dot product rounded just below one produces a false ~1e-8 rad jump.
    return 2.0 * math.atan2(vector_norm, abs(relative[3]))


@dataclass(frozen=True, slots=True)
class Pose:
    """A parent-to-child rigid transform using ROS quaternion order xyzw."""

    translation: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    @classmethod
    def identity(cls) -> "Pose":
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def normalized(self) -> "Pose":
        if not _finite(self.translation):
            raise ValueError("translation_nonfinite")
        if not _finite(self.quaternion_xyzw):
            raise ValueError("quaternion_nonfinite")
        return Pose(self.translation, quaternion_normalize(self.quaternion_xyzw))

    def compose(self, child: "Pose") -> "Pose":
        left = self.normalized()
        right = child.normalized()
        rotated = quaternion_rotate(left.quaternion_xyzw, right.translation)
        return Pose(
            tuple(a + b for a, b in zip(left.translation, rotated)),  # type: ignore[arg-type]
            quaternion_multiply(left.quaternion_xyzw, right.quaternion_xyzw),
        )

    def inverse(self) -> "Pose":
        normalized = self.normalized()
        inverse_quaternion = quaternion_inverse(normalized.quaternion_xyzw)
        inverse_translation = quaternion_rotate(
            inverse_quaternion,
            tuple(-value for value in normalized.translation),  # type: ignore[arg-type]
        )
        return Pose(inverse_translation, inverse_quaternion)

    def translation_distance(self, other: "Pose") -> float:
        return math.sqrt(
            sum(
                (left - right) ** 2
                for left, right in zip(self.translation, other.translation)
            )
        )

    def rotation_distance_rad(self, other: "Pose") -> float:
        return quaternion_distance_rad(self.quaternion_xyzw, other.quaternion_xyzw)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "translation_xyz": list(self.translation),
            "quaternion_xyzw": list(self.quaternion_xyzw),
        }


@dataclass(frozen=True, slots=True)
class PoseSample:
    """One generation-bound, health-attested candidate pose."""

    source: str
    generation: int
    sequence_id: int
    stamp_ns: int
    received_monotonic_ns: int
    parent_frame: str
    child_frame: str
    pose: Pose
    linear_velocity_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    backend_ready: bool = True
    tracking: bool = True
    health_reason: str = "tracking"


@dataclass(frozen=True, slots=True)
class CanonicalOutput:
    """The single value from which both `/odom` and `/tf` are emitted."""

    source: str
    generation: int
    sequence_id: int
    stamp_ns: int
    parent_frame: str
    child_frame: str
    pose: Pose
    linear_velocity_xyz: tuple[float, float, float]
    angular_velocity_xyz: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "generation": self.generation,
            "sequence_id": self.sequence_id,
            "stamp_ns": self.stamp_ns,
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "pose": self.pose.to_dict(),
            "linear_velocity_xyz": list(self.linear_velocity_xyz),
            "angular_velocity_xyz": list(self.angular_velocity_xyz),
            "odometry_topic": CANONICAL_ODOMETRY_TOPIC,
            "tf_topic": CANONICAL_TF_TOPIC,
        }


@dataclass(frozen=True, slots=True)
class SourceHealth:
    source: str
    configured: bool
    state: str
    reason: str
    generation: int
    last_stamp_ns: int | None
    age_sec: float | None
    stamp_age_sec: float | None
    valid_count: int
    invalid_count: int
    ordered_count: int
    backend_ready: bool
    consecutive_healthy_decisions: int

    @property
    def healthy(self) -> bool:
        return self.state == "HEALTHY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "configured": self.configured,
            "state": self.state,
            "reason": self.reason,
            "generation": self.generation,
            "last_stamp_ns": self.last_stamp_ns,
            "age_sec": self.age_sec,
            "stamp_age_sec": self.stamp_age_sec,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "ordered_count": self.ordered_count,
            "backend_ready": self.backend_ready,
            "consecutive_healthy_decisions": self.consecutive_healthy_decisions,
        }


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    runtime_policy: str
    runtime_target: str
    generation: int
    decision_monotonic_ns: int
    previous_source: str | None
    selected_source: str | None
    switch_reason: str
    switch_event: bool
    source_health: dict[str, SourceHealth]
    output: CanonicalOutput | None
    odometry_deferred: bool
    deviation: dict[str, Any] | None
    switch_translation_jump_m: float | None
    switch_rotation_jump_rad: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event": "localization_selection",
            "runtime_policy": self.runtime_policy,
            "runtime_target": self.runtime_target,
            "generation": self.generation,
            "decision_monotonic_ns": self.decision_monotonic_ns,
            "previous_source": self.previous_source,
            "selected_source": self.selected_source,
            "switch_reason": self.switch_reason,
            "switch_event": self.switch_event,
            "source_health": {
                source: health.to_dict()
                for source, health in sorted(self.source_health.items())
            },
            "output": self.output.to_dict() if self.output is not None else None,
            "odometry_deferred": self.odometry_deferred,
            "deviation": self.deviation,
            "switch_translation_jump_m": self.switch_translation_jump_m,
            "switch_rotation_jump_rad": self.switch_rotation_jump_rad,
        }
