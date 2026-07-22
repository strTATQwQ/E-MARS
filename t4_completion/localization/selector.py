"""Health-aware runtime selector with SE(3) continuity on source changes."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .config import SelectorPolicy, SourceSpec
from .contracts import (
    ALL_SOURCES,
    CANONICAL_CHILD_FRAME,
    CANONICAL_PARENT_FRAME,
    CUVSLAM,
    ISAAC_GT,
    LIDAR_IMU,
    SENSOR_ODOMETRY_SOURCES,
    CanonicalOutput,
    Pose,
    PoseSample,
    SelectionDecision,
    SourceHealth,
    quaternion_norm,
)


def _valid_vector(value: Any, length: int) -> bool:
    if not isinstance(value, tuple) or len(value) != length:
        return False
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            return False
    return True


@dataclass(slots=True)
class _SourceState:
    spec: SourceSpec
    sample: PoseSample | None = None
    alignment: Pose | None = None
    last_stamp_ns: int | None = None
    last_sequence_id: int | None = None
    valid_count: int = 0
    invalid_count: int = 0
    ordered_count: int = 0
    last_rejection_reason: str | None = None
    healthy_since_ns: int | None = None
    consecutive_healthy_decisions: int = 0

    def reset(self) -> None:
        self.sample = None
        self.alignment = None
        self.last_stamp_ns = None
        self.last_sequence_id = None
        self.last_rejection_reason = None
        self.healthy_since_ns = None
        self.consecutive_healthy_decisions = 0


class LocalizationSelector:
    """Select cuVSLAM/LIO and explicitly deviate to GT only in completion sim."""

    def __init__(
        self,
        *,
        policy: SelectorPolicy,
        sources: dict[str, SourceSpec],
        initial_generation: int = 0,
    ) -> None:
        if set(sources) != set(ALL_SOURCES):
            raise ValueError("selector_requires_exact_frozen_sources")
        if (
            isinstance(initial_generation, bool)
            or not isinstance(initial_generation, int)
            or initial_generation < 0
        ):
            raise ValueError("initial_generation_must_be_nonnegative")
        self.policy = policy
        self._states = {
            name: _SourceState(spec=sources[name]) for name in ALL_SOURCES
        }
        self.generation = initial_generation
        self.selected_source: str | None = None
        self._last_output: CanonicalOutput | None = None
        self._last_switch_monotonic_ns: int | None = None
        self._last_decision_monotonic_ns: int | None = None
        self._last_decision_sim_time_ns: int | None = None
        self._allow_post_reset_sim_time_rebase = False
        self._last_health: dict[str, SourceHealth] | None = None
        self.switch_count = 0
        self.deviation_count = 0
        self.no_output_count = 0

    @property
    def last_output(self) -> CanonicalOutput | None:
        return self._last_output

    def reset_generation(self, generation: int) -> None:
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ValueError("reset_generation_must_be_integer")
        if generation < self.generation:
            raise ValueError("reset_generation_regression")
        if generation == self.generation:
            return
        self.generation = generation
        self.selected_source = None
        self._last_output = None
        self._last_health = None
        self._last_switch_monotonic_ns = None
        # Monotonic process time must never regress, but a new episode/reset
        # generation may legitimately restart ROS simulation time at zero.
        self._last_decision_sim_time_ns = None
        self._allow_post_reset_sim_time_rebase = True
        for state in self._states.values():
            state.reset()

    def _reject(self, state: _SourceState, reason: str) -> tuple[bool, str]:
        state.invalid_count += 1
        state.last_rejection_reason = reason
        # Fail closed: an invalid identity/stamp/frame invalidates the current
        # sample instead of silently retaining and republishing old pose/TF.
        state.sample = None
        state.healthy_since_ns = None
        state.consecutive_healthy_decisions = 0
        return False, reason

    def invalidate_source(self, source: str, reason: str) -> None:
        if not isinstance(source, str) or source not in self._states:
            raise ValueError("unknown_source")
        if not isinstance(reason, str) or not reason:
            raise ValueError("invalidation_reason_required")
        self._reject(self._states[source], reason)

    def ingest(self, sample: PoseSample) -> tuple[bool, str]:
        if not isinstance(sample, PoseSample):
            return False, "sample_contract_invalid"
        if not isinstance(sample.source, str) or sample.source not in self._states:
            return False, "unknown_source"
        state = self._states[sample.source]
        if not state.spec.enabled:
            return self._reject(state, "source_disabled")
        if isinstance(sample.generation, bool) or not isinstance(
            sample.generation, int
        ):
            return self._reject(state, "generation_must_be_integer")
        if sample.generation != self.generation:
            return self._reject(state, "generation_mismatch")
        if isinstance(sample.sequence_id, bool) or not isinstance(
            sample.sequence_id, int
        ):
            return self._reject(state, "sequence_id_must_be_integer")
        if sample.sequence_id < 0:
            return self._reject(state, "negative_sequence_id")
        if isinstance(sample.stamp_ns, bool) or not isinstance(sample.stamp_ns, int):
            return self._reject(state, "stamp_must_be_integer")
        if sample.stamp_ns <= 0:
            return self._reject(state, "nonpositive_stamp")
        if isinstance(sample.received_monotonic_ns, bool) or not isinstance(
            sample.received_monotonic_ns, int
        ):
            return self._reject(state, "receive_time_must_be_integer")
        if sample.received_monotonic_ns < 0:
            return self._reject(state, "negative_receive_time")
        if sample.parent_frame != state.spec.expected_parent_frame:
            return self._reject(state, "unexpected_parent_frame")
        if sample.child_frame != state.spec.expected_child_frame:
            return self._reject(state, "unexpected_child_frame")
        if state.last_stamp_ns is not None and sample.stamp_ns <= state.last_stamp_ns:
            return self._reject(state, "non_monotonic_stamp")
        if (
            state.last_sequence_id is not None
            and sample.sequence_id <= state.last_sequence_id
        ):
            return self._reject(state, "non_monotonic_sequence")
        if not isinstance(sample.pose, Pose) or not all(
            (
                _valid_vector(sample.pose.translation, 3),
                _valid_vector(sample.pose.quaternion_xyzw, 4),
                _valid_vector(sample.linear_velocity_xyz, 3),
                _valid_vector(sample.angular_velocity_xyz, 3),
            )
        ):
            return self._reject(state, "pose_or_twist_shape_or_value_invalid")
        values = (
            *sample.pose.translation,
            *sample.pose.quaternion_xyzw,
            *sample.linear_velocity_xyz,
            *sample.angular_velocity_xyz,
        )
        try:
            finite = all(math.isfinite(value) for value in values)
        except OverflowError:
            finite = False
        if not finite:
            return self._reject(state, "nonfinite_pose_or_twist")
        if not isinstance(sample.backend_ready, bool) or not isinstance(
            sample.tracking, bool
        ):
            return self._reject(state, "source_health_must_be_boolean")
        if not isinstance(sample.health_reason, str) or not sample.health_reason:
            return self._reject(state, "source_health_reason_invalid")
        norm = quaternion_norm(sample.pose.quaternion_xyzw)
        if abs(norm - 1.0) > self.policy.quaternion_norm_tolerance:
            return self._reject(state, "quaternion_norm_out_of_range")
        normalized = PoseSample(
            source=sample.source,
            generation=sample.generation,
            sequence_id=sample.sequence_id,
            stamp_ns=sample.stamp_ns,
            received_monotonic_ns=sample.received_monotonic_ns,
            parent_frame=sample.parent_frame,
            child_frame=sample.child_frame,
            pose=sample.pose.normalized(),
            linear_velocity_xyz=sample.linear_velocity_xyz,
            angular_velocity_xyz=sample.angular_velocity_xyz,
            backend_ready=sample.backend_ready,
            tracking=sample.tracking,
            health_reason=sample.health_reason,
        )
        state.sample = normalized
        state.last_stamp_ns = normalized.stamp_ns
        state.last_sequence_id = normalized.sequence_id
        state.valid_count += 1
        state.ordered_count += 1
        state.last_rejection_reason = None
        return True, "accepted"

    def _source_health(
        self, source: str, now_ns: int, sim_time_ns: int
    ) -> SourceHealth:
        state = self._states[source]
        sample = state.sample
        configured = state.spec.enabled
        health_state: str
        reason: str
        age_sec: float | None = None
        stamp_age_sec: float | None = None
        backend_ready = False
        if not configured:
            health_state, reason = "DISABLED", "source_disabled"
        elif sample is None:
            health_state = (
                "INVALID" if state.last_rejection_reason else state.spec.startup_state
            )
            reason = state.last_rejection_reason or state.spec.startup_reason
        else:
            backend_ready = sample.backend_ready
            age_ns = now_ns - sample.received_monotonic_ns
            age_sec = age_ns / 1e9
            stamp_age_ns = sim_time_ns - sample.stamp_ns
            stamp_age_sec = stamp_age_ns / 1e9
            if age_ns < 0:
                health_state, reason = "INVALID", "future_receive_time"
            elif stamp_age_ns < -self.policy.maximum_future_stamp_ns:
                health_state, reason = "INVALID", "future_sample_stamp"
            elif not sample.backend_ready:
                health_state, reason = "UNAVAILABLE", "backend_not_ready"
            elif not sample.tracking:
                health_state, reason = "INVALID", sample.health_reason or "tracking_false"
            elif age_ns > self.policy.source_timeout_ns:
                health_state, reason = "STALE", "source_timeout_exceeded"
            elif stamp_age_ns > self.policy.source_timeout_ns:
                health_state, reason = "STALE", "sample_stamp_timeout_exceeded"
            else:
                health_state, reason = "HEALTHY", sample.health_reason or "tracking"
        return SourceHealth(
            source=source,
            configured=configured,
            state=health_state,
            reason=reason,
            generation=self.generation,
            last_stamp_ns=sample.stamp_ns if sample else state.last_stamp_ns,
            age_sec=age_sec,
            stamp_age_sec=stamp_age_sec,
            valid_count=state.valid_count,
            invalid_count=state.invalid_count,
            ordered_count=state.ordered_count,
            backend_ready=backend_ready,
            consecutive_healthy_decisions=state.consecutive_healthy_decisions,
        )

    def health_snapshot(
        self, now_ns: int, sim_time_ns: int
    ) -> dict[str, SourceHealth]:
        if (
            isinstance(now_ns, bool)
            or not isinstance(now_ns, int)
            or isinstance(sim_time_ns, bool)
            or not isinstance(sim_time_ns, int)
        ):
            raise ValueError("decision_time_must_be_integer")
        if now_ns < 0 or sim_time_ns < 0:
            raise ValueError("negative_decision_time")
        return {
            source: self._source_health(source, now_ns, sim_time_ns)
            for source in ALL_SOURCES
        }

    def _advance_health_tracking(
        self, health: dict[str, SourceHealth], now_ns: int
    ) -> dict[str, SourceHealth]:
        advanced: dict[str, SourceHealth] = {}
        for source, snapshot in health.items():
            state = self._states[source]
            if snapshot.healthy:
                if state.healthy_since_ns is None:
                    state.healthy_since_ns = now_ns
                state.consecutive_healthy_decisions += 1
            else:
                state.healthy_since_ns = None
                state.consecutive_healthy_decisions = 0
            advanced[source] = replace(
                snapshot,
                consecutive_healthy_decisions=state.consecutive_healthy_decisions,
            )
        return advanced

    def _recovery_ready(self, source: str, now_ns: int) -> bool:
        state = self._states[source]
        if state.healthy_since_ns is None:
            return False
        if now_ns - state.healthy_since_ns < self.policy.preferred_recovery_hold_ns:
            return False
        if not self._newer_than_last_output(source):
            return False
        return (
            self._last_switch_monotonic_ns is None
            or now_ns - self._last_switch_monotonic_ns >= self.policy.minimum_dwell_ns
        )

    def _newer_than_last_output(self, source: str) -> bool:
        sample = self._states[source].sample
        return sample is not None and (
            self._last_output is None or sample.stamp_ns > self._last_output.stamp_ns
        )

    def _best_sensor_source(self, health: dict[str, SourceHealth]) -> str | None:
        for source in (CUVSLAM, LIDAR_IMU):
            if health[source].healthy and self._newer_than_last_output(source):
                return source
        return None

    def _choose_source(
        self, health: dict[str, SourceHealth], now_ns: int
    ) -> tuple[str | None, str]:
        current = self.selected_source
        best_sensor = self._best_sensor_source(health)
        gt_healthy = health[ISAAC_GT].healthy and self._newer_than_last_output(
            ISAAC_GT
        )
        if current is None:
            if best_sensor == CUVSLAM:
                return CUVSLAM, "INITIAL_PREFERRED_HEALTHY"
            if best_sensor == LIDAR_IMU:
                return LIDAR_IMU, "INITIAL_ALTERNATE_HEALTHY"
            if self.policy.allow_isaac_gt_fallback and gt_healthy:
                return ISAAC_GT, "ALL_SENSOR_ODOMETRY_DEFERRED"
            if not self.policy.allow_isaac_gt_fallback:
                return None, "STRICT_SENSOR_ODOMETRY_UNAVAILABLE_FAIL_CLOSED"
            return None, "NO_HEALTHY_SOURCE_FAIL_CLOSED"

        if health[current].healthy:
            higher_priority: str | None = None
            if current == ISAAC_GT:
                higher_priority = best_sensor
            elif (
                current == LIDAR_IMU
                and health[CUVSLAM].healthy
                and self._newer_than_last_output(CUVSLAM)
            ):
                higher_priority = CUVSLAM
            if higher_priority and self._recovery_ready(higher_priority, now_ns):
                return higher_priority, "RECOVERED_AFTER_HYSTERESIS"
            return current, "ACTIVE_HEALTHY"

        if best_sensor is not None:
            return best_sensor, "ACTIVE_SOURCE_UNHEALTHY"
        if self.policy.allow_isaac_gt_fallback and gt_healthy:
            return ISAAC_GT, "ALL_SENSOR_ODOMETRY_DEFERRED"
        if self.policy.allow_isaac_gt_fallback:
            return None, "GT_UNAVAILABLE_FAIL_CLOSED"
        return None, "STRICT_SENSOR_ODOMETRY_UNAVAILABLE_FAIL_CLOSED"

    def _deviation(
        self, health: dict[str, SourceHealth], selected: str | None
    ) -> dict[str, Any] | None:
        if selected != ISAAC_GT:
            return None
        return {
            "schema_version": 1,
            "severity": "WARN",
            "code": "ODOMETRY_DEFERRED_ISAAC_GT_FALLBACK",
            "runtime_policy": self.policy.runtime_policy,
            "runtime_target": self.policy.runtime_target,
            "fallback_source": ISAAC_GT,
            "sensor_odometry_status": "STRICT_EXTENSION_PENDING",
            "sensor_sources": {
                source: {
                    "state": health[source].state,
                    "reason": health[source].reason,
                }
                for source in SENSOR_ODOMETRY_SOURCES
            },
        }

    def decide(self, now_ns: int, sim_time_ns: int) -> SelectionDecision:
        if (
            isinstance(now_ns, bool)
            or not isinstance(now_ns, int)
            or isinstance(sim_time_ns, bool)
            or not isinstance(sim_time_ns, int)
        ):
            raise ValueError("decision_time_must_be_integer")
        if now_ns < 0 or sim_time_ns < 0:
            raise ValueError("negative_decision_time")
        if (
            self._last_decision_monotonic_ns is not None
            and now_ns <= self._last_decision_monotonic_ns
        ):
            raise ValueError("decision_monotonic_time_regression")
        if (
            self._last_decision_sim_time_ns is not None
            and sim_time_ns < self._last_decision_sim_time_ns
        ):
            if not self._allow_post_reset_sim_time_rebase:
                raise ValueError("decision_sim_time_regression")
        self._last_decision_monotonic_ns = now_ns
        self._last_decision_sim_time_ns = sim_time_ns
        health = self._advance_health_tracking(
            self.health_snapshot(now_ns, sim_time_ns), now_ns
        )
        previous_source = self.selected_source
        desired_source, reason = self._choose_source(health, now_ns)
        if desired_source is None:
            lost_source = previous_source is not None
            if lost_source:
                self.selected_source = None
                self.switch_count += 1
                self._last_switch_monotonic_ns = now_ns
            self.no_output_count += 1
            self._last_health = health
            return SelectionDecision(
                runtime_policy=self.policy.runtime_policy,
                runtime_target=self.policy.runtime_target,
                generation=self.generation,
                decision_monotonic_ns=now_ns,
                previous_source=previous_source,
                selected_source=None,
                switch_reason=reason,
                switch_event=lost_source,
                source_health=health,
                output=None,
                odometry_deferred=False,
                deviation=None,
                switch_translation_jump_m=None,
                switch_rotation_jump_rad=None,
            )

        state = self._states[desired_source]
        sample = state.sample
        if sample is None:  # Defensive: a HEALTHY source must always own a sample.
            raise RuntimeError("healthy_source_missing_sample")
        if self._last_output is not None and sample.stamp_ns <= self._last_output.stamp_ns:
            self.no_output_count += 1
            self._last_health = health
            return SelectionDecision(
                runtime_policy=self.policy.runtime_policy,
                runtime_target=self.policy.runtime_target,
                generation=self.generation,
                decision_monotonic_ns=now_ns,
                previous_source=previous_source,
                selected_source=previous_source,
                switch_reason="CANDIDATE_TIMESTAMP_NOT_NEWER",
                switch_event=False,
                source_health=health,
                output=None,
                odometry_deferred=previous_source == ISAAC_GT,
                deviation=self._deviation(health, previous_source),
                switch_translation_jump_m=None,
                switch_rotation_jump_rad=None,
            )

        switch_event = desired_source != previous_source
        previous_output = self._last_output
        if switch_event:
            state.alignment = (
                Pose.identity()
                if previous_output is None
                else previous_output.pose.compose(sample.pose.inverse())
            )
        if state.alignment is None:
            state.alignment = Pose.identity()
        output_pose = state.alignment.compose(sample.pose)
        translation_jump: float | None = None
        rotation_jump: float | None = None
        if switch_event and previous_output is not None:
            translation_jump = output_pose.translation_distance(previous_output.pose)
            rotation_jump = output_pose.rotation_distance_rad(previous_output.pose)
        output = CanonicalOutput(
            source=desired_source,
            generation=self.generation,
            sequence_id=sample.sequence_id,
            stamp_ns=sample.stamp_ns,
            parent_frame=CANONICAL_PARENT_FRAME,
            child_frame=CANONICAL_CHILD_FRAME,
            pose=output_pose,
            linear_velocity_xyz=(
                (0.0, 0.0, 0.0)
                if switch_event and previous_output is not None
                else sample.linear_velocity_xyz
            ),
            angular_velocity_xyz=(
                (0.0, 0.0, 0.0)
                if switch_event and previous_output is not None
                else sample.angular_velocity_xyz
            ),
        )
        if switch_event:
            self.switch_count += 1
            self._last_switch_monotonic_ns = now_ns
        self.selected_source = desired_source
        self._last_output = output
        self._last_health = health
        self._allow_post_reset_sim_time_rebase = False
        deviation = self._deviation(health, desired_source)
        if deviation is not None:
            self.deviation_count += 1
        return SelectionDecision(
            runtime_policy=self.policy.runtime_policy,
            runtime_target=self.policy.runtime_target,
            generation=self.generation,
            decision_monotonic_ns=now_ns,
            previous_source=previous_source,
            selected_source=desired_source,
            switch_reason=reason,
            switch_event=switch_event,
            source_health=health,
            output=output,
            odometry_deferred=desired_source == ISAAC_GT,
            deviation=deviation,
            switch_translation_jump_m=translation_jump,
            switch_rotation_jump_rad=rotation_jump,
        )

    def summary(self) -> dict[str, Any]:
        current_output_available = (
            self.selected_source is not None
            and self._last_output is not None
            and self._last_health is not None
            and self._last_health[self.selected_source].healthy
            and self._states[self.selected_source].sample is not None
        )
        return {
            "schema_version": 1,
            "runtime_policy": self.policy.runtime_policy,
            "runtime_target": self.policy.runtime_target,
            "generation": self.generation,
            "selected_source": self.selected_source,
            "current_output_available": current_output_available,
            "switch_count": self.switch_count,
            "deviation_count": self.deviation_count,
            "no_output_count": self.no_output_count,
            "odometry_deferred": self.selected_source == ISAAC_GT,
            "sensor_odometry_status": (
                "STRICT_EXTENSION_PENDING"
                if self.deviation_count > 0
                else "SELECTED_OR_NOT_YET_EVALUATED"
            ),
        }
