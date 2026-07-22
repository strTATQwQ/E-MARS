from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GaitServoOutput:
    policy_vx: float
    policy_vy: float
    active: bool
    actual_speed_mps: float
    actual_along_request_mps: float
    actual_lateral_request_mps: float
    effective_along_hard_limit_mps: float
    filtered_speed_mps: float
    filtered_along_request_mps: float
    filtered_lateral_request_mps: float
    reason: str


@dataclass(frozen=True)
class MotionGuardOutput:
    active: bool
    reason: str
    stable_steps: int


class LowSpeedMotionGuard:
    """Latch zero policy input before measured low-speed motion runs away."""

    def __init__(
        self,
        *,
        speed_trigger_mps: float = 0.18,
        yaw_trigger_radps: float = 0.20,
        speed_release_mps: float = 0.08,
        yaw_release_radps: float = 0.08,
        release_steps: int = 25,
    ) -> None:
        if not 0.0 < speed_release_mps < speed_trigger_mps:
            raise ValueError("expected 0 < speed_release < speed_trigger")
        if not 0.0 < yaw_release_radps < yaw_trigger_radps:
            raise ValueError("expected 0 < yaw_release < yaw_trigger")
        if release_steps < 1:
            raise ValueError("release_steps must be positive")
        self.speed_trigger_mps = float(speed_trigger_mps)
        self.yaw_trigger_radps = float(yaw_trigger_radps)
        self.speed_release_mps = float(speed_release_mps)
        self.yaw_release_radps = float(yaw_release_radps)
        self.release_steps = int(release_steps)
        self.active = False
        self.stable_steps = 0
        self.trigger_reason = "clear"

    def reset(self) -> None:
        self.active = False
        self.stable_steps = 0
        self.trigger_reason = "reset"

    def update(self, actual_speed_mps: float, actual_yaw_rate_radps: float) -> MotionGuardOutput:
        speed = abs(float(actual_speed_mps))
        yaw = abs(float(actual_yaw_rate_radps))
        if not self.active and (speed >= self.speed_trigger_mps or yaw >= self.yaw_trigger_radps):
            self.active = True
            self.stable_steps = 0
            self.trigger_reason = "speed" if speed >= self.speed_trigger_mps else "yaw"
        if self.active:
            if speed <= self.speed_release_mps and yaw <= self.yaw_release_radps:
                self.stable_steps += 1
            else:
                self.stable_steps = 0
            if self.stable_steps >= self.release_steps:
                self.active = False
                self.stable_steps = 0
                self.trigger_reason = "released"
        return MotionGuardOutput(self.active, self.trigger_reason, self.stable_steps)


class LowSpeedGaitServo:
    """Pulse a locomotion policy while bounding measured low-speed motion.

    The ROS request remains the safety contract. The larger policy command is
    explicitly exposed in telemetry and is disabled as measured speed reaches
    the hysteresis ceiling.
    """

    def __init__(
        self,
        *,
        gait_command_mps: float = 0.45,
        on_below_mps: float = 0.07,
        off_above_mps: float = 0.15,
        hard_limit_mps: float = 0.20,
        total_hard_limit_mps: float = 0.20,
        filter_alpha: float = 1.0,
        coast_at_requested: bool = False,
    ) -> None:
        if not 0.0 <= on_below_mps < off_above_mps <= hard_limit_mps:
            raise ValueError("expected 0 <= on_below < off_above <= hard_limit")
        if gait_command_mps <= 0.0:
            raise ValueError("gait_command_mps must be positive")
        if total_hard_limit_mps <= 0.0:
            raise ValueError("total_hard_limit_mps must be positive")
        if not 0.0 < filter_alpha <= 1.0:
            raise ValueError("filter_alpha must be in (0, 1]")
        self.gait_command_mps = gait_command_mps
        self.on_below_mps = on_below_mps
        self.off_above_mps = off_above_mps
        self.hard_limit_mps = hard_limit_mps
        self.total_hard_limit_mps = total_hard_limit_mps
        self.filter_alpha = filter_alpha
        self.coast_at_requested = bool(coast_at_requested)
        self.active = False
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0

    def reset(self) -> None:
        self.active = False
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0

    def update(
        self,
        requested_vx: float,
        requested_vy: float,
        actual_vx: float,
        actual_vy: float,
    ) -> GaitServoOutput:
        requested_speed = math.hypot(requested_vx, requested_vy)
        actual_speed = math.hypot(actual_vx, actual_vy)
        self.filtered_vx += self.filter_alpha * (actual_vx - self.filtered_vx)
        self.filtered_vy += self.filter_alpha * (actual_vy - self.filtered_vy)
        filtered_speed = math.hypot(self.filtered_vx, self.filtered_vy)
        if requested_speed <= 1e-4:
            self.active = False
            return GaitServoOutput(
                0.0, 0.0, False, actual_speed, 0.0, 0.0, 0.0,
                filtered_speed, 0.0, 0.0, "no_request",
            )

        direction_x = requested_vx / requested_speed
        direction_y = requested_vy / requested_speed
        actual_along = actual_vx * direction_x + actual_vy * direction_y
        actual_lateral = abs(-actual_vx * direction_y + actual_vy * direction_x)
        filtered_along = self.filtered_vx * direction_x + self.filtered_vy * direction_y
        filtered_lateral = abs(-self.filtered_vx * direction_y + self.filtered_vy * direction_x)
        scaled_on = min(self.on_below_mps, requested_speed * 0.45)
        scaled_off = min(self.off_above_mps, requested_speed * 0.80)
        scaled_hard = min(self.hard_limit_mps, requested_speed)
        combined_along_limit = math.sqrt(
            max(0.0, self.total_hard_limit_mps * self.total_hard_limit_mps - filtered_lateral * filtered_lateral)
        )
        effective_hard = min(scaled_hard, combined_along_limit)

        reason = "hold"
        if filtered_speed >= self.total_hard_limit_mps or effective_hard <= 1e-4:
            self.active = False
            reason = "combined_speed_hard_limit"
        elif filtered_along >= effective_hard:
            self.active = False
            reason = "combined_speed_hard_limit"
        elif self.active and filtered_along >= scaled_off:
            self.active = False
            reason = "hysteresis_off"
        elif not self.active and filtered_along <= scaled_on:
            self.active = True
            reason = "hysteresis_on"

        magnitude = self.gait_command_mps if self.active else (
            min(requested_speed, self.hard_limit_mps) if self.coast_at_requested else 0.0
        )
        return GaitServoOutput(
            policy_vx=magnitude * direction_x,
            policy_vy=magnitude * direction_y,
            active=self.active,
            actual_speed_mps=actual_speed,
            actual_along_request_mps=actual_along,
            actual_lateral_request_mps=actual_lateral,
            effective_along_hard_limit_mps=effective_hard,
            filtered_speed_mps=filtered_speed,
            filtered_along_request_mps=filtered_along,
            filtered_lateral_request_mps=filtered_lateral,
            reason=reason,
        )


@dataclass(frozen=True)
class YawServoOutput:
    policy_wz: float
    active: bool
    actual_wz: float
    filtered_wz: float
    effective_hard_limit: float
    reason: str


class LowSpeedYawServo:
    """Pulse policy yaw input through a measured-rate hysteresis band."""

    def __init__(
        self,
        *,
        gait_command_radps: float = 0.30,
        on_below_radps: float = 0.03,
        off_above_radps: float = 0.12,
        hard_limit_radps: float = 0.24,
        filter_alpha: float = 0.12,
        min_request_radps: float = 0.08,
        pulse_steps: int = 3,
        positive_pulse_steps: int = 12,
        cooldown_steps: int = 8,
    ) -> None:
        if not 0.0 <= on_below_radps < off_above_radps <= hard_limit_radps:
            raise ValueError("expected 0 <= yaw on_below < off_above <= hard_limit")
        if gait_command_radps <= 0.0:
            raise ValueError("gait_command_radps must be positive")
        if not 0.0 < filter_alpha <= 1.0:
            raise ValueError("filter_alpha must be in (0, 1]")
        if not 0.0 <= min_request_radps < hard_limit_radps:
            raise ValueError("min_request_radps must be below hard_limit_radps")
        if pulse_steps < 1 or positive_pulse_steps < 1 or cooldown_steps < 1:
            raise ValueError("pulse_steps, positive_pulse_steps, and cooldown_steps must be positive")
        self.gait_command_radps = gait_command_radps
        self.on_below_radps = on_below_radps
        self.off_above_radps = off_above_radps
        self.hard_limit_radps = hard_limit_radps
        self.filter_alpha = filter_alpha
        self.min_request_radps = min_request_radps
        self.pulse_steps = pulse_steps
        self.positive_pulse_steps = positive_pulse_steps
        self.cooldown_steps = cooldown_steps
        self.active = False
        self.filtered_wz = 0.0
        self.cooldown_remaining = 0
        self.pulse_remaining = 0

    def reset(self) -> None:
        self.active = False
        self.filtered_wz = 0.0
        self.cooldown_remaining = 0
        self.pulse_remaining = 0

    def update(self, requested_wz: float, actual_wz: float) -> YawServoOutput:
        self.filtered_wz += self.filter_alpha * (actual_wz - self.filtered_wz)
        requested_magnitude = abs(requested_wz)
        if requested_magnitude <= 1e-4:
            self.active = False
            self.cooldown_remaining = 0
            self.pulse_remaining = 0
            return YawServoOutput(0.0, False, actual_wz, self.filtered_wz, 0.0, "no_request")
        if requested_magnitude < self.min_request_radps:
            self.active = False
            self.cooldown_remaining = 0
            self.pulse_remaining = 0
            return YawServoOutput(0.0, False, actual_wz, self.filtered_wz, requested_magnitude, "below_min_request")

        direction = math.copysign(1.0, requested_wz)
        actual_along = direction * self.filtered_wz
        scaled_on = min(self.on_below_radps, requested_magnitude * 0.45)
        scaled_off = min(self.off_above_radps, requested_magnitude * 0.80)
        effective_hard = min(self.hard_limit_radps, requested_magnitude)
        reason = "yaw_wait"
        if (
            abs(actual_wz) >= self.hard_limit_radps
            or abs(self.filtered_wz) >= self.hard_limit_radps
            or actual_along >= effective_hard
        ):
            self.active = False
            self.pulse_remaining = 0
            reason = "yaw_hard_limit"
            self.cooldown_remaining = max(self.cooldown_remaining, self.cooldown_steps)
        elif self.pulse_remaining > 0:
            self.active = True
            self.pulse_remaining -= 1
            reason = "yaw_burst"
            if self.pulse_remaining == 0:
                self.cooldown_remaining = self.cooldown_steps
        elif self.cooldown_remaining > 0:
            self.active = False
            self.cooldown_remaining -= 1
            reason = "yaw_cooldown"
        elif abs(actual_wz) <= scaled_on and abs(self.filtered_wz) <= scaled_off and actual_along <= effective_hard:
            self.active = True
            burst_steps = self.positive_pulse_steps if requested_wz > 0.0 else self.pulse_steps
            self.pulse_remaining = burst_steps - 1
            reason = "yaw_burst_start"
            if self.pulse_remaining == 0:
                self.cooldown_remaining = self.cooldown_steps
        else:
            self.active = False
        policy_wz = direction * self.gait_command_radps if self.active else 0.0
        return YawServoOutput(policy_wz, self.active, actual_wz, self.filtered_wz, effective_hard, reason)
