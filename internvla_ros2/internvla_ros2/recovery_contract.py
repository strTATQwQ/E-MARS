"""Shared completion-simulation recovery transaction helpers.

The module has no ROS imports so identity, deadline, idempotency, and trajectory
fingerprint behavior can be verified offline.  ROS nodes pass generated request
and response objects by duck typing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


OP_CANCEL_AND_DISABLE = 1
OP_CLEAR_MODEL_CACHE = 2
OP_REQUEST_REPLAN = 3

STATUS_OK = 0
STATUS_INVALID = 2
STATUS_STALE = 3
STATUS_TIMEOUT = 4
STATUS_CONFLICT = 5
STATUS_INTERNAL = 8

_TOKEN = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_DEADLINE_AHEAD_NS = 120_000_000_000
_NANOSECONDS_PER_SECOND = 1_000_000_000

SYSTEM2_REPLAN_POLICY_ENV = "INTERNVLA_T5_SYSTEM2_REPLAN_POLICY"
SYSTEM2_REPLAN_POLICIES = frozenset(
    {"strict", "observation_bound", "raw_wire_warn"}
)

RESPONSE_FIELDS = (
    "success",
    "status_code",
    "status_message",
    "episode_id",
    "reset_generation",
    "goal_id",
    "recovery_id",
    "operation_id",
    "operation",
    "cache_epoch",
    "trajectory_absolute_sha256",
    "trajectory_shape_sha256",
    "motion_disabled",
    "measured_linear_speed_peak_mps",
    "measured_angular_speed_peak_rps",
    "safety_fresh",
)


class RecoveryContractError(ValueError):
    """A typed recovery request cannot be safely consumed."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class RecoveryIdentity:
    episode_id: str
    reset_generation: int
    goal_id: str
    recovery_id: str
    operation_id: str
    operation: int
    deadline_ns: int


@dataclass(frozen=True, slots=True)
class TrajectorySignature:
    absolute_sha256: str
    shape_sha256: str
    length_m: float
    point_count: int


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _time_ns(value: Any) -> int:
    return int(value.sec) * 1_000_000_000 + int(value.nanosec)


def goal_identity(episode_id: str, reset_generation: int, sequence_id: int) -> str:
    """Return a bounded token shared by the supervisor and active adapter."""

    digest = _canonical_sha256(
        {
            "episode_id": str(episode_id),
            "reset_generation": int(reset_generation),
            "sequence_id": int(sequence_id),
        }
    )
    return f"goal:{digest}"


def recovery_identity(
    episode_id: str, reset_generation: int, recovery_index: int
) -> str:
    digest = _canonical_sha256(
        {
            "episode_id": str(episode_id),
            "reset_generation": int(reset_generation),
            "recovery_index": int(recovery_index),
        }
    )
    return f"recovery:{digest}"


def operation_identity(recovery_id: str, operation: int) -> str:
    return f"operation:{int(operation)}:{hashlib.sha256(recovery_id.encode()).hexdigest()}"


def t5_completion_sim_enabled(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the exact T5 Isaac completion-simulation lane is active."""

    values = os.environ if environment is None else environment
    return bool(
        values.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
        and values.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
        and values.get("INTERNNAV_T5_LANE", "") in {"a", "b"}
    )


def system2_replan_policy(
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve the T5-only System2 recovery comparison policy.

    ``observation_bound`` is the functional default.  The strict and raw-wire
    arms are explicit completion-simulation diagnostics and cannot leak into a
    real-robot or non-T5 process.  Raw-wire is additionally confined to Lane A
    with Recovery A selected.
    """

    values = os.environ if environment is None else environment
    policy = values.get(SYSTEM2_REPLAN_POLICY_ENV, "observation_bound")
    if policy not in SYSTEM2_REPLAN_POLICIES:
        raise ValueError(f"unsupported System2 replan policy: {policy}")
    explicitly_selected = SYSTEM2_REPLAN_POLICY_ENV in values
    if explicitly_selected and not t5_completion_sim_enabled(values):
        raise ValueError(
            "System2 replan comparison policies require T5 Isaac completion_sim"
        )
    if policy == "raw_wire_warn" and (
        values.get("INTERNNAV_T5_LANE", "") != "a"
        or values.get("INTERNNAV_T5_CANDIDATE_PROFILE", "") != "recovery_a"
    ):
        raise ValueError("raw-wire System2 replan is restricted to Lane A Recovery A")
    return policy


def recovery_deadline_ns(now_ns: int, timeout_sec: float) -> int:
    """Create a bounded deadline in the caller-provided clock domain."""

    now_ns = int(now_ns)
    timeout_sec = float(timeout_sec)
    duration_ns = int(timeout_sec * _NANOSECONDS_PER_SECOND)
    if now_ns <= 0:
        raise ValueError("recovery clock is unavailable")
    if (
        not math.isfinite(timeout_sec)
        or duration_ns <= 0
        or duration_ns > _MAX_DEADLINE_AHEAD_NS
    ):
        raise ValueError("recovery timeout is outside the bounded range")
    return now_ns + duration_ns


def semantic_age_sec(now_ns: int | None, stamp_ns: int | None) -> float | None:
    """Return an age only when a positive semantic clock has not regressed."""

    if now_ns is None or stamp_ns is None:
        return None
    now_ns = int(now_ns)
    stamp_ns = int(stamp_ns)
    if now_ns <= 0 or stamp_ns <= 0 or now_ns < stamp_ns:
        return None
    return (now_ns - stamp_ns) / _NANOSECONDS_PER_SECOND


def validate_request(
    request: Any,
    *,
    expected_operation: int,
    active_episode_id: str,
    active_reset_generation: int,
    active_goal_id: str,
    now_ns: int | None = None,
) -> RecoveryIdentity:
    """Validate a request before any state mutation or idempotent lookup."""

    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    if now_ns <= 0:
        raise RecoveryContractError(STATUS_TIMEOUT, "recovery clock is unavailable")
    identity = RecoveryIdentity(
        episode_id=str(request.episode_id),
        reset_generation=int(request.reset_generation),
        goal_id=str(request.goal_id),
        recovery_id=str(request.recovery_id),
        operation_id=str(request.operation_id),
        operation=int(request.operation),
        deadline_ns=_time_ns(request.deadline),
    )
    if (
        not identity.episode_id
        or len(identity.episode_id) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in identity.episode_id)
    ):
        raise RecoveryContractError(STATUS_INVALID, "invalid episode_id")
    for name in ("goal_id", "recovery_id", "operation_id"):
        if _TOKEN.fullmatch(str(getattr(identity, name))) is None:
            raise RecoveryContractError(STATUS_INVALID, f"invalid {name}")
    if identity.reset_generation < 0:
        raise RecoveryContractError(STATUS_INVALID, "negative reset_generation")
    if identity.operation != expected_operation:
        raise RecoveryContractError(STATUS_INVALID, "operation is not supported here")
    if identity.deadline_ns <= now_ns:
        raise RecoveryContractError(STATUS_TIMEOUT, "recovery operation deadline expired")
    if identity.deadline_ns - now_ns > _MAX_DEADLINE_AHEAD_NS:
        raise RecoveryContractError(STATUS_INVALID, "recovery deadline is unbounded")
    if (
        identity.episode_id != active_episode_id
        or identity.reset_generation != active_reset_generation
        or identity.goal_id != active_goal_id
    ):
        raise RecoveryContractError(STATUS_STALE, "active navigation identity changed")
    absolute = tuple(str(value) for value in request.excluded_absolute_sha256)
    shape = tuple(str(value) for value in request.excluded_shape_sha256)
    if len(absolute) > 64 or len(shape) > 64:
        raise RecoveryContractError(STATUS_INVALID, "trajectory exclusion set is too large")
    if any(_SHA256.fullmatch(value) is None for value in (*absolute, *shape)):
        raise RecoveryContractError(STATUS_INVALID, "invalid trajectory fingerprint")
    if expected_operation != OP_REQUEST_REPLAN and (absolute or shape):
        raise RecoveryContractError(
            STATUS_INVALID,
            "trajectory exclusions are only valid for a replan request",
        )
    return identity


def initialize_response(response: Any, request: Any) -> None:
    response.success = False
    response.status_code = STATUS_INTERNAL
    response.status_message = "operation did not complete"
    response.episode_id = str(request.episode_id)
    response.reset_generation = int(request.reset_generation)
    response.goal_id = str(request.goal_id)
    response.recovery_id = str(request.recovery_id)
    response.operation_id = str(request.operation_id)
    response.operation = int(request.operation)
    response.cache_epoch = 0
    response.trajectory_absolute_sha256 = ""
    response.trajectory_shape_sha256 = ""
    response.motion_disabled = False
    response.measured_linear_speed_peak_mps = 0.0
    response.measured_angular_speed_peak_rps = 0.0
    response.safety_fresh = False


def response_snapshot(response: Any) -> dict[str, object]:
    return {name: getattr(response, name) for name in RESPONSE_FIELDS}


def restore_response(response: Any, snapshot: dict[str, object]) -> None:
    for name in RESPONSE_FIELDS:
        setattr(response, name, snapshot[name])


def reject_response(response: Any, error: RecoveryContractError) -> None:
    response.success = False
    response.status_code = error.status_code
    response.status_message = str(error)


def _resample(points: list[tuple[float, float]], count: int = 32) -> list[tuple[float, float]]:
    cumulative = [0.0]
    for first, second in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(first, second))
    total = cumulative[-1]
    output: list[tuple[float, float]] = []
    segment = 0
    for index in range(count):
        target = total * index / (count - 1)
        while segment + 1 < len(cumulative) and cumulative[segment + 1] < target:
            segment += 1
        if segment + 1 >= len(points):
            output.append(points[-1])
            continue
        span = cumulative[segment + 1] - cumulative[segment]
        ratio = 0.0 if span <= 0.0 else (target - cumulative[segment]) / span
        first, second = points[segment], points[segment + 1]
        output.append(
            (
                first[0] + ratio * (second[0] - first[0]),
                first[1] + ratio * (second[1] - first[1]),
            )
        )
    return output


def trajectory_signature(
    points: Iterable[Sequence[float]],
    *,
    resolution_m: float = 0.05,
    maximum_points: int = 2048,
) -> TrajectorySignature:
    """Create density-stable absolute and translated-shape fingerprints."""

    if not 0.0 < resolution_m <= 0.20:
        raise ValueError("resolution_m is outside the supported bound")
    converted: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if index >= maximum_points:
            raise ValueError("trajectory exceeds maximum_points")
        if len(point) != 2:
            raise ValueError("trajectory points must contain x and y")
        candidate = (float(point[0]), float(point[1]))
        if not all(math.isfinite(value) for value in candidate):
            raise ValueError("trajectory contains a non-finite coordinate")
        if not converted or math.dist(converted[-1], candidate) >= resolution_m / 4.0:
            converted.append(candidate)
    if len(converted) < 2:
        raise ValueError("trajectory must contain two distinct points")
    length = sum(math.dist(a, b) for a, b in zip(converted, converted[1:]))
    if length < resolution_m:
        raise ValueError("trajectory is too short for a stable signature")
    sampled = _resample(converted)
    absolute = [
        [int(round(x / resolution_m)), int(round(y / resolution_m))]
        for x, y in sampled
    ]
    origin_x, origin_y = sampled[0]
    shape_resolution = 2.0 * resolution_m
    shape = [
        [
            int(round((x - origin_x) / shape_resolution)),
            int(round((y - origin_y) / shape_resolution)),
        ]
        for x, y in sampled
    ]
    length_bin = int(round(length / resolution_m))
    return TrajectorySignature(
        absolute_sha256=_canonical_sha256(
            {"points": absolute, "length_bin": length_bin}
        ),
        shape_sha256=_canonical_sha256({"points": shape, "length_bin": length_bin}),
        length_m=length,
        point_count=len(converted),
    )


def system2_primitive_signature(
    *,
    action: int,
    episode_id: str,
    reset_generation: int,
    sequence_id: int,
    observation_digest: str,
    x: float,
    y: float,
    yaw_rad: float,
) -> TrajectorySignature:
    """Fingerprint a System2 primitive in absolute and semantic domains.

    The absolute fingerprint names one command at a quantized pose.  The shape
    fingerprint intentionally omits sequence and pose but binds the source
    observation.  Reissuing an excluded action against the same observation
    remains forbidden, while a genuinely new action-observation cycle can
    choose the same discrete primitive again.
    """

    action = int(action)
    reset_generation = int(reset_generation)
    sequence_id = int(sequence_id)
    observation_digest = str(observation_digest)
    pose = (
        float(x),
        float(y),
        math.atan2(math.sin(float(yaw_rad)), math.cos(float(yaw_rad))),
    )
    if (
        action not in {1, 2, 3}
        or not str(episode_id)
        or not observation_digest
        or reset_generation < 0
        or sequence_id < 0
        or not all(math.isfinite(value) for value in pose)
    ):
        raise ValueError("invalid System2 primitive identity")
    semantic = {
        "schema_version": 1,
        "kind": "t5_completion_sim_system2_primitive",
        "action": action,
        "episode_id": str(episode_id),
        "reset_generation": reset_generation,
        "observation_digest": observation_digest,
    }
    absolute = {
        **semantic,
        "sequence_id": sequence_id,
        "pose_bin": {
            "x_5cm": int(round(pose[0] / 0.05)),
            "y_5cm": int(round(pose[1] / 0.05)),
            "yaw_5deg": int(round(pose[2] / math.radians(5.0))),
        },
    }
    return TrajectorySignature(
        absolute_sha256=_canonical_sha256(absolute),
        shape_sha256=_canonical_sha256(semantic),
        length_m=0.0,
        point_count=1,
    )
