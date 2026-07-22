"""Deterministic, identity-scoped and fully bounded T4.5 recovery core.

The core has no ROS, wall-clock, thread, network or actuator dependency.  A
future shared adapter must translate commands into the existing Nav2 safety
chain and return correlated acknowledgements.  Until then this module is an
offline-verifiable contract, not an online motion entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .config import RecoveryConfig


class RecoveryState(str, Enum):
    IDLE = "idle"
    MONITORING = "monitoring"
    CANCEL_GOAL = "cancel_goal"
    CLEAR_LOCAL_CACHE = "clear_local_cache"
    SCAN = "scan"
    REQUEST_REPLAN = "request_replan"
    WAIT_FRESH_PLAN = "wait_fresh_plan"
    ACCEPT_FRESH_PLAN = "accept_fresh_plan"
    SAFE_RETREAT = "safe_retreat"
    COOLDOWN = "cooldown"
    TERMINAL_SAFE_STOP = "terminal_safe_stop"


class RecoveryAction(str, Enum):
    CANCEL_GOAL = "cancel_goal"
    CLEAR_LOCAL_CACHE = "clear_local_cache"
    SCAN = "scan"
    REQUEST_REPLAN = "request_replan"
    ACCEPT_FRESH_PLAN = "accept_fresh_plan"
    SAFE_RETREAT = "safe_retreat"
    LATCH_SAFE_STOP = "latch_safe_stop"


_ACTION_STATES = {
    RecoveryState.CANCEL_GOAL,
    RecoveryState.CLEAR_LOCAL_CACHE,
    RecoveryState.SCAN,
    RecoveryState.REQUEST_REPLAN,
    RecoveryState.ACCEPT_FRESH_PLAN,
    RecoveryState.SAFE_RETREAT,
}
_MOTION_ACTIONS = {RecoveryAction.SCAN, RecoveryAction.SAFE_RETREAT}
_ACTION_BY_STATE = {
    RecoveryState.CANCEL_GOAL: RecoveryAction.CANCEL_GOAL,
    RecoveryState.CLEAR_LOCAL_CACHE: RecoveryAction.CLEAR_LOCAL_CACHE,
    RecoveryState.SCAN: RecoveryAction.SCAN,
    RecoveryState.REQUEST_REPLAN: RecoveryAction.REQUEST_REPLAN,
    RecoveryState.ACCEPT_FRESH_PLAN: RecoveryAction.ACCEPT_FRESH_PLAN,
    RecoveryState.SAFE_RETREAT: RecoveryAction.SAFE_RETREAT,
}
_SAFETY_GATED_ACTIONS = _MOTION_ACTIONS | {RecoveryAction.ACCEPT_FRESH_PLAN}


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _sha(value: object) -> str:
    payload = json.dumps(
        value, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _resample_polyline(
    points: Sequence[tuple[float, float]], sample_count: int = 16
) -> list[tuple[float, float]]:
    cumulative = [0.0]
    for first, second in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(first, second))
    total = cumulative[-1]
    if total <= 0.0:
        raise ValueError("trajectory must have nonzero length")
    output: list[tuple[float, float]] = []
    segment = 0
    for index in range(sample_count):
        target = total * index / (sample_count - 1)
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


@dataclass(frozen=True, slots=True)
class TrajectorySignature:
    absolute_sha256: str
    shape_sha256: str
    length_m: float
    point_count: int

    @property
    def tokens(self) -> tuple[str, str]:
        return (
            f"absolute:{self.absolute_sha256}",
            f"shape:{self.shape_sha256}",
        )


def trajectory_signature(
    points: Iterable[Sequence[float]], *, resolution_m: float, maximum_points: int
) -> TrajectorySignature:
    """Create density-stable absolute and relative-shape signatures."""

    if not 0.0 < resolution_m <= 0.20:
        raise ValueError("resolution_m is outside the supported bound")
    converted: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if index >= maximum_points:
            raise ValueError("trajectory exceeds maximum_points")
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
            raise ValueError("trajectory points must be coordinate sequences")
        if len(point) != 2:
            raise ValueError("trajectory points must contain exactly x and y")
        candidate = (
            _finite_number(point[0], f"points[{index}].x"),
            _finite_number(point[1], f"points[{index}].y"),
        )
        if not converted or math.dist(converted[-1], candidate) >= resolution_m / 4.0:
            converted.append(candidate)
    if len(converted) < 2:
        raise ValueError("trajectory must contain at least two distinct points")
    length = sum(math.dist(a, b) for a, b in zip(converted, converted[1:]))
    if length < resolution_m:
        raise ValueError("trajectory is too short for a stable signature")
    resampled = _resample_polyline(converted)

    def quantize(value: float) -> int:
        return int(round(value / resolution_m))

    absolute = [[quantize(x), quantize(y)] for x, y in resampled]
    origin_x, origin_y = resampled[0]
    # Shape matching is deliberately one bin more tolerant than the absolute
    # signature so small localization shifts or resampling jitter cannot evade
    # the episode taboo set.
    shape_resolution = 2.0 * resolution_m
    shape = [
        [
            int(round((x - origin_x) / shape_resolution)),
            int(round((y - origin_y) / shape_resolution)),
        ]
        for x, y in resampled
    ]
    length_bin = quantize(length)
    return TrajectorySignature(
        absolute_sha256=_sha({"points": absolute, "length_bin": length_bin}),
        shape_sha256=_sha({"points": shape, "length_bin": length_bin}),
        length_m=length,
        point_count=len(converted),
    )


@dataclass(frozen=True, slots=True)
class Observation:
    episode_id: str
    reset_generation: int
    goal_id: str
    monotonic_sec: float
    x_m: float
    y_m: float
    goal_distance_m: float
    goal_active: bool
    motion_requested: bool
    safety_clear: bool
    sim_estop_ready: bool
    sim_estop_active: bool
    rear_clearance_m: float | None = None
    plan_points: tuple[tuple[float, float], ...] | None = None
    plan_update: bool = False


@dataclass(frozen=True, slots=True)
class RecoveryCommand:
    operation_id: str
    action: RecoveryAction
    episode_id: str
    reset_generation: int
    goal_id: str
    recovery_id: str
    attempt: int
    issued_monotonic_sec: float
    deadline_monotonic_sec: float
    requires_ack: bool
    requires_simulation: bool
    requires_estop_ready: bool
    requires_safety_clear: bool
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _Sample:
    stamp: float
    x: float
    y: float
    goal_distance: float


class RecoveryMachine:
    """Bounded state machine.  A terminal stop is latched until begin_episode."""

    def __init__(self, config: RecoveryConfig) -> None:
        config.validate()
        self.config = config
        self.state = RecoveryState.IDLE
        self.episode_id = ""
        self.reset_generation = -1
        self.goal_id = ""
        self.recovery_id = ""
        self._last_now: float | None = None
        self._last_safety_clear = False
        self._last_estop_ready = False
        self._last_estop_active = True
        self._last_safety_observed_at: float | None = None
        self._last_rear_clearance_m: float | None = None
        self._last_pose: tuple[float, float, float] | None = None
        self._samples: deque[_Sample] = deque(maxlen=4096)
        self._plan_arrivals: deque[tuple[float, str]] = deque(
            maxlen=config.taboo_capacity
        )
        self._recent_signatures: dict[str, TrajectorySignature] = {}
        self._seen_plan_shapes: set[str] = set()
        self._active_signature: TrajectorySignature | None = None
        self._accepted_signature: TrajectorySignature | None = None
        self._taboo_order: deque[str] = deque()
        self._taboo: set[str] = set()
        self._pending: RecoveryCommand | None = None
        self._outbox: list[RecoveryCommand] = []
        self._completed_operations: dict[str, bool] = {}
        self._completed_order: deque[str] = deque()
        self._retired_operations: deque[str] = deque(maxlen=256)
        self._phase_attempt = 0
        self._next_issue_not_before = 0.0
        self._recovery_started_at: float | None = None
        self._plan_wait_deadline: float | None = None
        self._awaiting_replan_operation_id: str | None = None
        self._cooldown_until: float | None = None
        self._fallback_used = False
        self._replan_requests_in_stage = 0
        self._plan_rejections_in_stage = 0
        self._commands_this_recovery = 0
        self._recoveries_this_episode = 0
        self._operation_sequence = 0
        self._event_sequence = 0
        self._event_history: deque[dict[str, Any]] = deque(
            maxlen=config.event_history_capacity
        )
        self._events_dropped = 0
        self._event_counts: Counter[str] = Counter()
        self._trigger_counts: Counter[str] = Counter()
        self._action_counts: Counter[str] = Counter()
        self._retry_counts: Counter[str] = Counter()
        self._recoveries_started = 0
        self._recoveries_completed = 0
        self._terminal_stops = 0
        self._old_plan_rejections = 0
        self._old_plan_execution_count = 0
        self._speed_cap_violations = 0
        self._missing_motion_evidence_count = 0
        self._motion_safety_evidence_violation_count = 0
        self._stale_ack_drops = 0
        self._post_recovery_anchor: tuple[float, float, float] | None = None
        self._awaiting_post_recovery_anchor = False
        self._post_recovery_progress_recorded = False

    @property
    def pending_command(self) -> RecoveryCommand | None:
        return self._pending

    @property
    def taboo_tokens(self) -> frozenset[str]:
        return frozenset(self._taboo)

    def events(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._event_history]

    def _record(self, event: str, now: float, **details: Any) -> None:
        self._event_sequence += 1
        self._event_counts[event] += 1
        payload = {
            "schema_version": 1,
            "event_sequence": self._event_sequence,
            "event": event,
            "monotonic_sec": now,
            "profile_id": self.config.profile_id,
            "episode_id": self.episode_id,
            "reset_generation": self.reset_generation,
            "goal_id": self.goal_id,
            "recovery_id": self.recovery_id,
            "state": self.state.value,
            **details,
        }
        if len(self._event_history) == self._event_history.maxlen:
            self._events_dropped += 1
        self._event_history.append(payload)

    def _transition(self, state: RecoveryState, now: float, reason: str) -> None:
        previous = self.state
        self.state = state
        self._phase_attempt = 0
        self._next_issue_not_before = now
        self._record(
            "state_transition",
            now,
            from_state=previous.value,
            to_state=state.value,
            reason=reason,
        )

    def begin_episode(
        self, episode_id: str, reset_generation: int, monotonic_sec: float
    ) -> list[RecoveryCommand]:
        now = _finite_number(monotonic_sec, "monotonic_sec")
        if not isinstance(episode_id, str) or not 1 <= len(episode_id) <= 128:
            raise ValueError("episode_id must be a string of 1..128 characters")
        if type(reset_generation) is not int or reset_generation < 0:
            raise ValueError("reset_generation must be a nonnegative integer")
        if self._last_now is not None and now < self._last_now:
            self._enter_terminal(self._last_now, "episode_time_regression")
            return self._drain_commands()
        if self.reset_generation >= 0 and reset_generation <= self.reset_generation:
            self._enter_terminal(now, "reset_generation_not_increasing")
            return self._drain_commands()
        preempted_generation = self.state != RecoveryState.IDLE
        if self._pending is not None:
            self._retired_operations.append(self._pending.operation_id)
        previous_identity = (self.episode_id, self.reset_generation)
        self.episode_id = episode_id
        self.reset_generation = reset_generation
        self.goal_id = ""
        self.recovery_id = ""
        self.state = RecoveryState.MONITORING
        self._last_now = now
        self._last_safety_clear = False
        self._last_estop_ready = False
        self._last_estop_active = True
        self._last_safety_observed_at = None
        self._last_rear_clearance_m = None
        self._last_pose = None
        self._samples.clear()
        self._plan_arrivals.clear()
        self._recent_signatures.clear()
        self._seen_plan_shapes.clear()
        self._active_signature = None
        self._accepted_signature = None
        self._taboo_order.clear()
        self._taboo.clear()
        self._pending = None
        self._outbox.clear()
        self._phase_attempt = 0
        self._next_issue_not_before = now
        self._recovery_started_at = None
        self._plan_wait_deadline = None
        self._awaiting_replan_operation_id = None
        self._cooldown_until = None
        self._fallback_used = False
        self._replan_requests_in_stage = 0
        self._plan_rejections_in_stage = 0
        self._commands_this_recovery = 0
        self._recoveries_this_episode = 0
        self._post_recovery_anchor = None
        self._awaiting_post_recovery_anchor = False
        self._post_recovery_progress_recorded = False
        self._record(
            "episode_started",
            now,
            previous_episode_id=previous_identity[0],
            previous_reset_generation=previous_identity[1],
        )
        if preempted_generation:
            self._record("generation_reset_safe_stop", now)
            self._queue_safe_stop(now, "episode_reset_preempted_recovery")
        return self._drain_commands()

    def _validate_time(self, now_value: object) -> float:
        try:
            now = _finite_number(now_value, "monotonic_sec")
        except ValueError:
            now = self._last_now if self._last_now is not None else 0.0
            self._enter_terminal(now, "invalid_monotonic_time")
            return now
        if self._last_now is not None and now < self._last_now:
            self._enter_terminal(self._last_now, "monotonic_time_regression")
            return self._last_now
        self._last_now = now
        return now

    def _set_safety(
        self,
        *,
        safety_clear: bool,
        sim_estop_ready: bool,
        sim_estop_active: bool,
        rear_clearance_m: float | None,
    ) -> None:
        if not all(
            type(value) is bool
            for value in (safety_clear, sim_estop_ready, sim_estop_active)
        ):
            raise ValueError("safety flags must be booleans")
        self._last_safety_clear = safety_clear
        self._last_estop_ready = sim_estop_ready
        self._last_estop_active = sim_estop_active
        if rear_clearance_m is None:
            self._last_rear_clearance_m = None
        else:
            clearance = _finite_number(rear_clearance_m, "rear_clearance_m")
            self._last_rear_clearance_m = max(0.0, clearance)

    def _retire_pending(self) -> None:
        if self._pending is not None:
            self._retired_operations.append(self._pending.operation_id)
        self._pending = None

    def _recovery_deadline_ok(self, now: float) -> bool:
        if (
            self._recovery_started_at is not None
            and now - self._recovery_started_at
            > self.config.maximum_recovery_duration_sec
        ):
            self._enter_terminal(now, "recovery_deadline_exhausted")
            return False
        return True

    def _cached_safety_is_fresh(self, now: float) -> bool:
        return (
            self._last_estop_ready
            and not self._last_estop_active
            and self._last_safety_clear
            and self._last_safety_observed_at is not None
            and now - self._last_safety_observed_at
            <= self.config.safety_freshness_timeout_sec
        )

    def _enter_terminal(self, now: float, reason: str) -> None:
        if self.state == RecoveryState.TERMINAL_SAFE_STOP:
            return
        previous = self.state
        self._retire_pending()
        self.state = RecoveryState.TERMINAL_SAFE_STOP
        self._terminal_stops += 1
        self._record(
            "terminal_safe_stop",
            now,
            from_state=previous.value,
            reason=reason,
            commands_this_recovery=self._commands_this_recovery,
        )
        self._queue_safe_stop(now, reason)

    def _queue_safe_stop(self, now: float, reason: str) -> None:
        self._operation_sequence += 1
        command = RecoveryCommand(
            operation_id=(
                f"{self.episode_id}:{self.reset_generation}:terminal:"
                f"{self._operation_sequence}"
            ),
            action=RecoveryAction.LATCH_SAFE_STOP,
            episode_id=self.episode_id,
            reset_generation=self.reset_generation,
            goal_id=self.goal_id,
            recovery_id=self.recovery_id,
            attempt=1,
            issued_monotonic_sec=now,
            deadline_monotonic_sec=now,
            requires_ack=False,
            requires_simulation=True,
            requires_estop_ready=False,
            requires_safety_clear=False,
            arguments=_deep_freeze({"reason": reason, "motion_enabled": False}),
        )
        self._outbox.append(command)
        self._action_counts[command.action.value] += 1

    def _drain_commands(self) -> list[RecoveryCommand]:
        output = list(self._outbox)
        self._outbox.clear()
        return output

    def _add_taboo(self, signature: TrajectorySignature | None, now: float) -> None:
        if signature is None:
            return
        for token in signature.tokens:
            if token in self._taboo:
                continue
            while len(self._taboo_order) >= self.config.taboo_capacity:
                expired = self._taboo_order.popleft()
                self._taboo.discard(expired)
                self._record("taboo_evicted", now, fingerprint=expired)
            self._taboo_order.append(token)
            self._taboo.add(token)
            self._record("trajectory_taboo_added", now, fingerprint=token)

    def _start_recovery(
        self,
        reason: str,
        now: float,
        extra_taboo: Iterable[TrajectorySignature] = (),
    ) -> None:
        if self._recoveries_this_episode >= self.config.maximum_recoveries_per_episode:
            self._enter_terminal(now, "episode_recovery_budget_exhausted")
            return
        self._recoveries_this_episode += 1
        self._recoveries_started += 1
        self.recovery_id = (
            f"{self.episode_id}:{self.reset_generation}:{self._recoveries_this_episode}"
        )
        self._recovery_started_at = now
        self._fallback_used = False
        self._replan_requests_in_stage = 0
        self._plan_rejections_in_stage = 0
        self._commands_this_recovery = 0
        self._accepted_signature = None
        self._add_taboo(self._active_signature, now)
        for signature in extra_taboo:
            self._add_taboo(signature, now)
        self._trigger_counts[reason] += 1
        self._record("recovery_triggered", now, reason=reason)
        self._samples.clear()
        self._transition(RecoveryState.CANCEL_GOAL, now, reason)

    def _command_arguments(self, action: RecoveryAction) -> dict[str, Any]:
        if action == RecoveryAction.CANCEL_GOAL:
            return {"disable_motion_atomically": True}
        if action == RecoveryAction.CLEAR_LOCAL_CACHE:
            return {
                "clear": [
                    "trajectory_candidates",
                    "local_plan",
                    "model_local_history",
                    "observation_window",
                ],
                "preserve": [
                    "episode_identity",
                    "reset_generation",
                    "simulation_estop_latch",
                    "event_statistics",
                    "trajectory_taboo",
                ],
            }
        if action == RecoveryAction.SCAN:
            return {
                "target_yaw_rad": self.config.scan_yaw_rad,
                "maximum_angular_speed_rps": self.config.scan_angular_speed_rps,
                "velocity_chain": "cmd_vel_safe",
            }
        if action == RecoveryAction.REQUEST_REPLAN:
            return {
                "excluded_fingerprints": sorted(self._taboo),
                "require_novel_trajectory": True,
                "consume_exactly_once": True,
            }
        if action == RecoveryAction.ACCEPT_FRESH_PLAN:
            if self._accepted_signature is None:
                raise RuntimeError("accepted signature is missing")
            return {
                "absolute_sha256": self._accepted_signature.absolute_sha256,
                "shape_sha256": self._accepted_signature.shape_sha256,
                "require_fresh_safe_command_barrier": True,
            }
        if action == RecoveryAction.SAFE_RETREAT:
            return {
                "distance_m": -self.config.retreat_distance_m,
                "maximum_linear_speed_mps": self.config.retreat_linear_speed_mps,
                "minimum_rear_clearance_m": self.config.retreat_minimum_rear_clearance_m,
                "velocity_chain": "cmd_vel_safe",
            }
        raise RuntimeError(f"unsupported action: {action}")

    def _issue_current_action(self, now: float) -> None:
        if self.state not in _ACTION_STATES or self._pending is not None:
            return
        action = _ACTION_BY_STATE[self.state]
        if self._commands_this_recovery >= self.config.maximum_commands_per_recovery:
            self._enter_terminal(now, "recovery_command_budget_exhausted")
            return
        if action in _SAFETY_GATED_ACTIONS:
            if not self._cached_safety_is_fresh(now):
                self._enter_terminal(now, "recovery_motion_safety_gate_closed")
                return
            if action == RecoveryAction.SAFE_RETREAT and (
                self._last_rear_clearance_m is None
                or self._last_rear_clearance_m
                < self.config.retreat_minimum_rear_clearance_m
            ):
                self._enter_terminal(now, "rear_clearance_missing_or_insufficient")
                return
        if action == RecoveryAction.REQUEST_REPLAN:
            if (
                self._replan_requests_in_stage
                >= self.config.maximum_replan_requests_per_stage
            ):
                self._start_retreat_or_stop(now, "replan_stage_budget_exhausted")
                return
            self._replan_requests_in_stage += 1
        self._phase_attempt += 1
        self._commands_this_recovery += 1
        self._operation_sequence += 1
        operation_id = (
            f"{self.episode_id}:{self.reset_generation}:{self.recovery_id}:"
            f"{action.value}:{self._operation_sequence}"
        )
        command = RecoveryCommand(
            operation_id=operation_id,
            action=action,
            episode_id=self.episode_id,
            reset_generation=self.reset_generation,
            goal_id=self.goal_id,
            recovery_id=self.recovery_id,
            attempt=self._phase_attempt,
            issued_monotonic_sec=now,
            deadline_monotonic_sec=now + self.config.action_timeout_sec,
            requires_ack=True,
            requires_simulation=True,
            requires_estop_ready=True,
            requires_safety_clear=action in _SAFETY_GATED_ACTIONS,
            arguments=_deep_freeze(self._command_arguments(action)),
        )
        self._pending = command
        self._outbox.append(command)
        self._action_counts[action.value] += 1
        self._record(
            "action_issued",
            now,
            operation_id=operation_id,
            action=action.value,
            attempt=command.attempt,
            deadline_monotonic_sec=command.deadline_monotonic_sec,
            requires_safety_clear=command.requires_safety_clear,
        )

    def _remember_completion(self, operation_id: str, success: bool) -> None:
        self._completed_operations[operation_id] = success
        self._completed_order.append(operation_id)
        while len(self._completed_order) > 256:
            expired = self._completed_order.popleft()
            self._completed_operations.pop(expired, None)

    def _retry_or_fail(self, now: float, reason: str) -> None:
        if self._pending is None:
            raise RuntimeError("retry requested without a pending command")
        action = self._pending.action
        self._retire_pending()
        self._record(
            "action_failed", now, action=action.value, reason=reason, attempt=self._phase_attempt
        )
        # A timed-out or rejected physical action may already have moved the
        # robot.  Issuing a fresh operation ID would allow a second scan or
        # retreat without trustworthy cumulative-motion evidence, so motion
        # actions fail closed instead of using the generic retry path.
        if action in _MOTION_ACTIONS:
            self._enter_terminal(now, f"{action.value}_{reason}_motion_retry_forbidden")
            return
        if self._phase_attempt < self.config.maximum_action_attempts:
            self._retry_counts[action.value] += 1
            self._next_issue_not_before = now + self.config.retry_backoff_sec
            self._record(
                "action_retry_scheduled",
                now,
                action=action.value,
                not_before_monotonic_sec=self._next_issue_not_before,
            )
            return
        if action == RecoveryAction.REQUEST_REPLAN:
            self._start_retreat_or_stop(now, f"replan_{reason}")
            return
        self._enter_terminal(now, f"{action.value}_{reason}_budget_exhausted")

    def _start_retreat_or_stop(self, now: float, reason: str) -> None:
        self._plan_wait_deadline = None
        if self.config.retreat_enabled and not self._fallback_used:
            self._fallback_used = True
            self._transition(RecoveryState.SAFE_RETREAT, now, reason)
            return
        self._enter_terminal(now, reason)

    def _complete_action(self, now: float, action: RecoveryAction) -> None:
        if action == RecoveryAction.CANCEL_GOAL:
            self._transition(RecoveryState.CLEAR_LOCAL_CACHE, now, "goal_cancel_confirmed")
        elif action == RecoveryAction.CLEAR_LOCAL_CACHE:
            self._transition(RecoveryState.SCAN, now, "local_cache_clear_confirmed")
        elif action == RecoveryAction.SCAN:
            self._transition(RecoveryState.REQUEST_REPLAN, now, "scan_confirmed")
        elif action == RecoveryAction.REQUEST_REPLAN:
            self.state = RecoveryState.WAIT_FRESH_PLAN
            self._phase_attempt = 0
            self._plan_wait_deadline = now + self.config.plan_wait_timeout_sec
            completed = self._completed_order[-1] if self._completed_order else None
            self._awaiting_replan_operation_id = completed
            self._record(
                "state_transition",
                now,
                from_state=RecoveryState.REQUEST_REPLAN.value,
                to_state=RecoveryState.WAIT_FRESH_PLAN.value,
                reason="replan_request_confirmed",
                plan_wait_deadline=self._plan_wait_deadline,
            )
        elif action == RecoveryAction.SAFE_RETREAT:
            self._replan_requests_in_stage = 0
            self._plan_rejections_in_stage = 0
            self._transition(RecoveryState.SCAN, now, "bounded_retreat_confirmed")
        elif action == RecoveryAction.ACCEPT_FRESH_PLAN:
            if self._accepted_signature is None:
                self._enter_terminal(now, "accepted_signature_missing")
                return
            if any(token in self._taboo for token in self._accepted_signature.tokens):
                self._old_plan_execution_count += 1
                self._enter_terminal(now, "taboo_plan_reached_execution_barrier")
                return
            if len(self._seen_plan_shapes) >= self.config.taboo_capacity:
                self._enter_terminal(now, "plan_history_capacity_exhausted")
                return
            self._seen_plan_shapes.add(self._accepted_signature.shape_sha256)
            self._active_signature = self._accepted_signature
            self._recoveries_completed += 1
            self._recovery_started_at = None
            # A pose observed before/during scan or retreat cannot prove
            # post-recovery navigation progress.  Anchor only on the first
            # monitoring observation after cooldown.
            self._post_recovery_anchor = None
            self._awaiting_post_recovery_anchor = True
            self._post_recovery_progress_recorded = False
            self.state = RecoveryState.COOLDOWN
            self._cooldown_until = now + self.config.cooldown_sec
            self._record(
                "recovery_completed",
                now,
                accepted_absolute_sha256=self._accepted_signature.absolute_sha256,
                accepted_shape_sha256=self._accepted_signature.shape_sha256,
                cooldown_until=self._cooldown_until,
            )
        else:
            self._enter_terminal(now, "unknown_completed_action")

    def _check_measured_velocity(
        self, now: float, command: RecoveryCommand, details: Mapping[str, Any]
    ) -> bool:
        action = command.action
        required = {
            "measured_max_linear_mps",
            "measured_max_angular_rps",
            "safety_observation_monotonic_sec",
            "simulation_estop_preemption_count",
            "safety_gate_violation_count",
            "velocity_chain",
        }
        if action in _MOTION_ACTIONS and not required.issubset(details):
            self._missing_motion_evidence_count += 1
            self._enter_terminal(now, "missing_measured_velocity_evidence")
            return False
        if action in _MOTION_ACTIONS:
            safety_stamp = _finite_number(
                details["safety_observation_monotonic_sec"],
                "safety_observation_monotonic_sec",
            )
            preemptions = details["simulation_estop_preemption_count"]
            violations = details["safety_gate_violation_count"]
            evidence_valid = (
                type(preemptions) is int
                and preemptions == 0
                and type(violations) is int
                and violations == 0
                and details["velocity_chain"] == "cmd_vel_safe"
                and command.issued_monotonic_sec <= safety_stamp <= now
                and now - safety_stamp <= self.config.safety_freshness_timeout_sec
            )
            if not evidence_valid:
                self._motion_safety_evidence_violation_count += 1
                self._enter_terminal(now, "invalid_motion_safety_evidence")
                return False
        linear = details.get("measured_max_linear_mps", 0.0)
        angular = details.get("measured_max_angular_rps", 0.0)
        linear_value = abs(_finite_number(linear, "measured_max_linear_mps"))
        angular_value = abs(_finite_number(angular, "measured_max_angular_rps"))
        permitted_linear = (
            self.config.retreat_linear_speed_mps
            if action == RecoveryAction.SAFE_RETREAT
            else 0.0
        )
        permitted_angular = (
            self.config.scan_angular_speed_rps if action == RecoveryAction.SCAN else 0.0
        )
        if linear_value > permitted_linear + 1e-9 or angular_value > permitted_angular + 1e-9:
            self._speed_cap_violations += 1
            self._enter_terminal(now, "measured_recovery_velocity_cap_violation")
            return False
        return True

    def acknowledge(
        self,
        *,
        operation_id: str,
        episode_id: str,
        reset_generation: int,
        recovery_id: str,
        success: bool,
        monotonic_sec: float,
        details: Mapping[str, Any] | None = None,
    ) -> list[RecoveryCommand]:
        now = self._validate_time(monotonic_sec)
        identifiers_valid = (
            isinstance(operation_id, str)
            and 1 <= len(operation_id) <= 512
            and isinstance(episode_id, str)
            and 1 <= len(episode_id) <= 128
            and type(reset_generation) is int
            and isinstance(recovery_id, str)
            and 1 <= len(recovery_id) <= 512
        )
        if not identifiers_valid:
            self._enter_terminal(now, "invalid_ack_contract")
            return self._drain_commands()
        if type(success) is not bool:
            self._enter_terminal(now, "ack_success_not_boolean")
            return self._drain_commands()
        if details is not None and not isinstance(details, Mapping):
            self._enter_terminal(now, "ack_details_not_mapping")
            return self._drain_commands()
        if operation_id in self._retired_operations:
            self._stale_ack_drops += 1
            self._record("stale_ack_dropped", now, operation_id=operation_id)
            return self._drain_commands()
        if operation_id in self._completed_operations:
            if self._completed_operations[operation_id] != success:
                self._enter_terminal(now, "conflicting_duplicate_ack")
            else:
                self._record("duplicate_ack_ignored", now, operation_id=operation_id)
            return self._drain_commands()
        if self.state == RecoveryState.TERMINAL_SAFE_STOP:
            self._stale_ack_drops += 1
            self._record("ack_after_terminal_dropped", now, operation_id=operation_id)
            return self._drain_commands()
        if not self._recovery_deadline_ok(now):
            return self._drain_commands()
        if (
            episode_id != self.episode_id
            or reset_generation != self.reset_generation
            or recovery_id != self.recovery_id
        ):
            self._enter_terminal(now, "ack_identity_mismatch")
            return self._drain_commands()
        if self._pending is None or operation_id != self._pending.operation_id:
            self._enter_terminal(now, "unexpected_ack_operation")
            return self._drain_commands()
        if now > self._pending.deadline_monotonic_sec:
            self._retry_or_fail(now, "timeout")
            return self._drain_commands()
        command = self._pending
        if (
            command.action == RecoveryAction.ACCEPT_FRESH_PLAN
            and not self._cached_safety_is_fresh(now)
        ):
            self._enter_terminal(now, "fresh_plan_barrier_safety_stale")
            return self._drain_commands()
        try:
            velocity_ok = self._check_measured_velocity(
                now, command, details or {}
            )
        except (TypeError, ValueError):
            self._enter_terminal(now, "invalid_measured_velocity_evidence")
            return self._drain_commands()
        if not velocity_ok:
            return self._drain_commands()
        self._remember_completion(operation_id, success)
        if not success:
            self._retry_or_fail(now, "rejected")
            return self._drain_commands()
        self._pending = None
        self._record(
            "action_confirmed",
            now,
            operation_id=operation_id,
            action=command.action.value,
            attempt=command.attempt,
        )
        self._complete_action(now, command.action)
        return self.tick(
            monotonic_sec=now,
            safety_clear=self._last_safety_clear,
            sim_estop_ready=self._last_estop_ready,
            sim_estop_active=self._last_estop_active,
            rear_clearance_m=self._last_rear_clearance_m,
            _safety_input_is_cached=True,
        )

    def tick(
        self,
        *,
        monotonic_sec: float,
        safety_clear: bool,
        sim_estop_ready: bool,
        sim_estop_active: bool,
        rear_clearance_m: float | None,
        _safety_input_is_cached: bool = False,
    ) -> list[RecoveryCommand]:
        now = self._validate_time(monotonic_sec)
        try:
            self._set_safety(
                safety_clear=safety_clear,
                sim_estop_ready=sim_estop_ready,
                sim_estop_active=sim_estop_active,
                rear_clearance_m=rear_clearance_m,
            )
        except ValueError:
            self._enter_terminal(now, "invalid_safety_input")
            return self._drain_commands()
        if not _safety_input_is_cached:
            self._last_safety_observed_at = now
        if self.state == RecoveryState.IDLE:
            return self._drain_commands()
        if not sim_estop_ready:
            self._enter_terminal(now, "simulation_estop_not_ready")
        elif sim_estop_active:
            self._enter_terminal(now, "simulation_estop_active")
        elif (
            self._pending is not None
            and self._pending.action in _SAFETY_GATED_ACTIONS
            and (
                not safety_clear
                or self._last_safety_observed_at is None
                or now - self._last_safety_observed_at
                > self.config.safety_freshness_timeout_sec
            )
        ):
            self._enter_terminal(now, "safety_lost_during_recovery_motion")
        elif (
            self._pending is not None
            and self._pending.action == RecoveryAction.SAFE_RETREAT
            and (
                self._last_rear_clearance_m is None
                or self._last_rear_clearance_m
                < self.config.retreat_minimum_rear_clearance_m
            )
        ):
            self._enter_terminal(now, "rear_clearance_lost_during_retreat")
        if self.state == RecoveryState.TERMINAL_SAFE_STOP:
            return self._drain_commands()
        if not self._recovery_deadline_ok(now):
            return self._drain_commands()
        if self._pending is not None and now > self._pending.deadline_monotonic_sec:
            self._retry_or_fail(now, "timeout")
        if (
            self.state == RecoveryState.WAIT_FRESH_PLAN
            and self._plan_wait_deadline is not None
            and now > self._plan_wait_deadline
        ):
            if (
                self._replan_requests_in_stage
                < self.config.maximum_replan_requests_per_stage
            ):
                self._transition(RecoveryState.REQUEST_REPLAN, now, "fresh_plan_timeout")
            else:
                self._start_retreat_or_stop(now, "fresh_plan_timeout_budget_exhausted")
        if (
            self.state == RecoveryState.COOLDOWN
            and self._cooldown_until is not None
            and now >= self._cooldown_until
        ):
            self._transition(RecoveryState.MONITORING, now, "cooldown_complete")
            self._samples.clear()
            self._plan_arrivals.clear()
            self._recent_signatures.clear()
            self._cooldown_until = None
        if (
            self.state in _ACTION_STATES
            and self._pending is None
            and now >= self._next_issue_not_before
        ):
            self._issue_current_action(now)
        return self._drain_commands()

    def _plan_cycle(self, now: float) -> tuple[bool, set[str]]:
        history = [
            token
            for stamp, token in self._plan_arrivals
            if now - stamp <= self.config.loop_window_sec
        ]
        repeats = self.config.plan_cycle_repetitions
        for period in range(1, len(history) // repeats + 1):
            width = period * repeats
            if len(history) < width:
                continue
            suffix = history[-width:]
            pattern = suffix[:period]
            if suffix == pattern * repeats:
                return True, set(pattern)
        return False, set()

    def _progress_trigger(self, now: float) -> str | None:
        loop_samples = [
            sample
            for sample in self._samples
            if now - sample.stamp <= self.config.loop_window_sec
        ]
        if len(loop_samples) >= self.config.minimum_sample_count:
            travel = sum(
                math.hypot(second.x - first.x, second.y - first.y)
                for first, second in zip(loop_samples, loop_samples[1:])
            )
            net = math.hypot(
                loop_samples[-1].x - loop_samples[0].x,
                loop_samples[-1].y - loop_samples[0].y,
            )
            cells = Counter(
                (
                    round(sample.x / self.config.loop_cell_size_m),
                    round(sample.y / self.config.loop_cell_size_m),
                )
                for sample in loop_samples
            )
            if (
                travel >= self.config.loop_min_travel_m
                and net <= self.config.loop_max_net_displacement_m
                and max(cells.values(), default=0) >= self.config.loop_revisit_count
            ):
                return "pose_loop"

        progress_samples = [
            sample
            for sample in self._samples
            if now - sample.stamp <= self.config.no_progress_window_sec
        ]
        if len(progress_samples) < self.config.minimum_sample_count:
            return None
        span = progress_samples[-1].stamp - progress_samples[0].stamp
        if span < (
            self.config.minimum_window_coverage * self.config.no_progress_window_sec
        ):
            return None
        displacement = math.hypot(
            progress_samples[-1].x - progress_samples[0].x,
            progress_samples[-1].y - progress_samples[0].y,
        )
        goal_progress = (
            progress_samples[0].goal_distance - progress_samples[-1].goal_distance
        )
        if (
            displacement < self.config.minimum_displacement_m
            and goal_progress < self.config.minimum_goal_progress_m
        ):
            return "no_progress"
        return None

    def _signature(self, points: Iterable[Sequence[float]]) -> TrajectorySignature:
        return trajectory_signature(
            points,
            resolution_m=self.config.trajectory_resolution_m,
            maximum_points=self.config.trajectory_max_points,
        )

    def observe(self, observation: Observation) -> list[RecoveryCommand]:
        now = self._validate_time(observation.monotonic_sec)
        if self.state == RecoveryState.IDLE:
            raise RuntimeError("begin_episode must be called before observe")
        if (
            not isinstance(observation.episode_id, str)
            or type(observation.reset_generation) is not int
        ):
            self._enter_terminal(now, "invalid_observation_identity_type")
            return self._drain_commands()
        if (
            observation.episode_id != self.episode_id
            or observation.reset_generation != self.reset_generation
        ):
            self._enter_terminal(now, "observation_identity_mismatch")
            return self._drain_commands()
        if not isinstance(observation.goal_id, str) or not 1 <= len(observation.goal_id) <= 128:
            self._enter_terminal(now, "invalid_goal_id")
            return self._drain_commands()
        if not all(
            type(value) is bool
            for value in (
                observation.goal_active,
                observation.motion_requested,
                observation.plan_update,
            )
        ):
            self._enter_terminal(now, "invalid_observation_boolean")
            return self._drain_commands()
        try:
            x = _finite_number(observation.x_m, "observation.x_m")
            y = _finite_number(observation.y_m, "observation.y_m")
            goal_distance = _finite_number(
                observation.goal_distance_m, "observation.goal_distance_m"
            )
        except ValueError:
            self._enter_terminal(now, "invalid_observation_numeric")
            return self._drain_commands()
        if goal_distance < 0.0:
            self._enter_terminal(now, "negative_goal_distance")
            return self._drain_commands()

        incoming_signature: TrajectorySignature | None = None
        if observation.plan_update:
            if observation.plan_points is None:
                self._enter_terminal(now, "plan_update_missing_points")
                return self._drain_commands()
            try:
                incoming_signature = self._signature(observation.plan_points)
            except ValueError:
                self._enter_terminal(now, "invalid_plan_update")
                return self._drain_commands()
        elif observation.plan_points is not None:
            self._enter_terminal(now, "plan_points_without_plan_update")
            return self._drain_commands()

        if self.goal_id and observation.goal_id != self.goal_id:
            if self.state not in {RecoveryState.MONITORING, RecoveryState.COOLDOWN}:
                self._enter_terminal(now, "goal_changed_during_recovery")
                return self._drain_commands()
            self._samples.clear()
            self._plan_arrivals.clear()
            self._recent_signatures.clear()
            self._seen_plan_shapes.clear()
            self._active_signature = None
            self._record(
                "goal_changed",
                now,
                previous_goal_id=self.goal_id,
                new_goal_id=observation.goal_id,
            )
        self.goal_id = observation.goal_id

        tick_commands = self.tick(
            monotonic_sec=now,
            safety_clear=observation.safety_clear,
            sim_estop_ready=observation.sim_estop_ready,
            sim_estop_active=observation.sim_estop_active,
            rear_clearance_m=observation.rear_clearance_m,
        )
        if tick_commands:
            self._outbox.extend(tick_commands)
        if self.state == RecoveryState.TERMINAL_SAFE_STOP:
            return self._drain_commands()
        if not self._recovery_deadline_ok(now):
            return self._drain_commands()
        self._last_pose = (x, y, goal_distance)

        if observation.plan_update:
            if incoming_signature is None:
                raise RuntimeError("validated trajectory signature is missing")
            signature = incoming_signature
            self._active_signature = signature
            self._plan_arrivals.append((now, signature.shape_sha256))
            self._recent_signatures[signature.shape_sha256] = signature
            recent_tokens = {token for _, token in self._plan_arrivals}
            self._recent_signatures = {
                token: value
                for token, value in self._recent_signatures.items()
                if token in recent_tokens
            }
            self._record(
                "plan_arrived",
                now,
                absolute_sha256=signature.absolute_sha256,
                shape_sha256=signature.shape_sha256,
            )
            repeated_shape = signature.shape_sha256 in self._seen_plan_shapes
            if (
                self.state == RecoveryState.MONITORING
                and observation.goal_active
                and observation.motion_requested
            ):
                if repeated_shape:
                    self._start_recovery(
                        "trajectory_cycle",
                        now,
                        list(self._recent_signatures.values()),
                    )
                elif len(self._seen_plan_shapes) >= self.config.taboo_capacity:
                    self._enter_terminal(now, "plan_history_capacity_exhausted")
                else:
                    self._seen_plan_shapes.add(signature.shape_sha256)
            cycle, cycle_tokens = self._plan_cycle(now)
            if (
                cycle
                and self.state == RecoveryState.MONITORING
                and observation.goal_active
                and observation.motion_requested
            ):
                extra = [
                    self._recent_signatures[token]
                    for token in cycle_tokens
                    if token in self._recent_signatures
                ]
                self._start_recovery("trajectory_cycle", now, extra)

        if self.state != RecoveryState.MONITORING:
            return self.tick(
                monotonic_sec=now,
                safety_clear=observation.safety_clear,
                sim_estop_ready=observation.sim_estop_ready,
                sim_estop_active=observation.sim_estop_active,
                rear_clearance_m=observation.rear_clearance_m,
            )
        if not observation.goal_active or not observation.motion_requested:
            self._samples.clear()
            return self._drain_commands()
        if (
            self._samples
            and now - self._samples[-1].stamp
            > self.config.maximum_observation_gap_sec
        ):
            self._record(
                "observation_gap_reset",
                now,
                gap_sec=now - self._samples[-1].stamp,
            )
            self._samples.clear()
        self._samples.append(_Sample(now, x, y, goal_distance))

        if self._awaiting_post_recovery_anchor:
            self._post_recovery_anchor = (x, y, goal_distance)
            self._awaiting_post_recovery_anchor = False
            self._record("post_recovery_progress_anchor", now)
        elif (
            self._post_recovery_anchor is not None
            and not self._post_recovery_progress_recorded
        ):
            anchor_x, anchor_y, anchor_goal = self._post_recovery_anchor
            displacement = math.hypot(x - anchor_x, y - anchor_y)
            goal_progress = anchor_goal - goal_distance
            if (
                displacement >= self.config.minimum_displacement_m
                or goal_progress >= self.config.minimum_goal_progress_m
            ):
                self._post_recovery_progress_recorded = True
                self._record(
                    "post_recovery_positive_progress",
                    now,
                    displacement_m=displacement,
                    goal_progress_m=goal_progress,
                )

        reason = self._progress_trigger(now)
        if reason is not None:
            self._start_recovery(reason, now)
            return self.tick(
                monotonic_sec=now,
                safety_clear=observation.safety_clear,
                sim_estop_ready=observation.sim_estop_ready,
                sim_estop_active=observation.sim_estop_active,
                rear_clearance_m=observation.rear_clearance_m,
            )
        return self._drain_commands()

    def submit_replanned_trajectory(
        self,
        *,
        request_operation_id: str,
        episode_id: str,
        reset_generation: int,
        recovery_id: str,
        points: Iterable[Sequence[float]],
        monotonic_sec: float,
    ) -> list[RecoveryCommand]:
        now = self._validate_time(monotonic_sec)
        if self.state == RecoveryState.TERMINAL_SAFE_STOP:
            return self._drain_commands()
        if not self._recovery_deadline_ok(now):
            return self._drain_commands()
        if (
            episode_id != self.episode_id
            or reset_generation != self.reset_generation
            or recovery_id != self.recovery_id
        ):
            self._enter_terminal(now, "replanned_trajectory_identity_mismatch")
            return self._drain_commands()
        if self.state != RecoveryState.WAIT_FRESH_PLAN:
            self._enter_terminal(now, "replanned_trajectory_in_unexpected_state")
            return self._drain_commands()
        if request_operation_id != self._awaiting_replan_operation_id:
            self._enter_terminal(now, "replanned_trajectory_wrong_request")
            return self._drain_commands()
        if self._plan_wait_deadline is None:
            self._enter_terminal(now, "replanned_trajectory_missing_deadline")
            return self._drain_commands()
        if now > self._plan_wait_deadline:
            self._record(
                "late_replanned_trajectory_dropped",
                now,
                request_operation_id=request_operation_id,
                plan_wait_deadline=self._plan_wait_deadline,
            )
            self._awaiting_replan_operation_id = None
            self._plan_wait_deadline = None
            if (
                self._replan_requests_in_stage
                < self.config.maximum_replan_requests_per_stage
            ):
                self._transition(
                    RecoveryState.REQUEST_REPLAN, now, "late_replanned_trajectory"
                )
            else:
                self._start_retreat_or_stop(
                    now, "late_replanned_trajectory_budget_exhausted"
                )
            return self.tick(
                monotonic_sec=now,
                safety_clear=self._last_safety_clear,
                sim_estop_ready=self._last_estop_ready,
                sim_estop_active=self._last_estop_active,
                rear_clearance_m=self._last_rear_clearance_m,
                _safety_input_is_cached=True,
            )
        try:
            signature = self._signature(points)
        except ValueError as exc:
            self._enter_terminal(now, f"invalid_replanned_trajectory:{exc}")
            return self._drain_commands()
        self._record(
            "replanned_trajectory_arrived",
            now,
            request_operation_id=request_operation_id,
            absolute_sha256=signature.absolute_sha256,
            shape_sha256=signature.shape_sha256,
        )
        if (
            any(token in self._taboo for token in signature.tokens)
            or signature.shape_sha256 in self._seen_plan_shapes
        ):
            self._add_taboo(signature, now)
            self._old_plan_rejections += 1
            self._plan_rejections_in_stage += 1
            self._record(
                "old_trajectory_rejected",
                now,
                request_operation_id=request_operation_id,
                rejection_index=self._plan_rejections_in_stage,
            )
            if (
                self._plan_rejections_in_stage
                >= self.config.maximum_plan_rejections_per_stage
                or self._replan_requests_in_stage
                >= self.config.maximum_replan_requests_per_stage
            ):
                self._start_retreat_or_stop(now, "old_trajectory_budget_exhausted")
            else:
                self._transition(
                    RecoveryState.REQUEST_REPLAN, now, "old_trajectory_rejected"
                )
            return self.tick(
                monotonic_sec=now,
                safety_clear=self._last_safety_clear,
                sim_estop_ready=self._last_estop_ready,
                sim_estop_active=self._last_estop_active,
                rear_clearance_m=self._last_rear_clearance_m,
                _safety_input_is_cached=True,
            )
        if len(self._seen_plan_shapes) >= self.config.taboo_capacity:
            self._enter_terminal(now, "plan_history_capacity_exhausted")
            return self._drain_commands()
        self._accepted_signature = signature
        self._plan_wait_deadline = None
        self._awaiting_replan_operation_id = None
        self._transition(RecoveryState.ACCEPT_FRESH_PLAN, now, "novel_trajectory_verified")
        self._record(
            "novel_trajectory_accepted",
            now,
            absolute_sha256=signature.absolute_sha256,
            shape_sha256=signature.shape_sha256,
        )
        return self.tick(
            monotonic_sec=now,
            safety_clear=self._last_safety_clear,
            sim_estop_ready=self._last_estop_ready,
            sim_estop_active=self._last_estop_active,
            rear_clearance_m=self._last_rear_clearance_m,
            _safety_input_is_cached=True,
        )

    def event_statistics(self) -> dict[str, Any]:
        """Return a machine-readable summary derived from recorded transitions."""

        return {
            "schema_version": 1,
            "profile_id": self.config.profile_id,
            "episode_id": self.episode_id,
            "reset_generation": self.reset_generation,
            "goal_id": self.goal_id,
            "state": self.state.value,
            "event_count": sum(self._event_counts.values()),
            "event_counts": dict(sorted(self._event_counts.items())),
            "trigger_counts": dict(sorted(self._trigger_counts.items())),
            "action_counts": dict(sorted(self._action_counts.items())),
            "retry_counts": dict(sorted(self._retry_counts.items())),
            "recoveries_started": self._recoveries_started,
            "recoveries_completed": self._recoveries_completed,
            "recoveries_this_episode": self._recoveries_this_episode,
            "terminal_safe_stop_count": self._terminal_stops,
            "old_trajectory_rejection_count": self._old_plan_rejections,
            "old_trajectory_execution_count": self._old_plan_execution_count,
            "speed_cap_violation_count": self._speed_cap_violations,
            "missing_motion_evidence_count": self._missing_motion_evidence_count,
            "motion_safety_evidence_violation_count": (
                self._motion_safety_evidence_violation_count
            ),
            "stale_ack_drop_count": self._stale_ack_drops,
            "commands_this_recovery": self._commands_this_recovery,
            "maximum_commands_per_recovery": self.config.maximum_commands_per_recovery,
            "taboo_fingerprint_count": len(self._taboo),
            "retained_event_count": len(self._event_history),
            "dropped_event_count": self._events_dropped,
            "simulation_only": True,
            "simulation_estop_required": True,
            "real_go2_allowed": False,
        }
