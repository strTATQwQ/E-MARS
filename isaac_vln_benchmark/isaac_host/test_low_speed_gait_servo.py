import pytest

from low_speed_gait_servo import LowSpeedGaitServo, LowSpeedMotionGuard, LowSpeedYawServo


def test_servo_pulses_below_lower_threshold_and_releases_above_upper():
    servo = LowSpeedGaitServo()

    start = servo.update(0.2, 0.0, 0.0, 0.0)
    coast = servo.update(0.2, 0.0, 0.10, 0.0)
    release = servo.update(0.2, 0.0, 0.16, 0.0)

    assert start.active is True
    assert start.policy_vx == pytest.approx(0.45)
    assert coast.active is True
    assert release.active is False
    assert release.policy_vx == 0.0


def test_servo_hard_stops_at_requested_speed_even_below_global_limit():
    servo = LowSpeedGaitServo()
    servo.update(0.1, 0.0, 0.0, 0.0)

    output = servo.update(0.1, 0.0, 0.1, 0.0)

    assert output.active is False
    assert output.reason == "combined_speed_hard_limit"


def test_servo_can_coast_at_bounded_request_instead_of_zero_between_pulses():
    servo = LowSpeedGaitServo(coast_at_requested=True)
    servo.update(0.2, 0.0, 0.0, 0.0)

    output = servo.update(0.2, 0.0, 0.16, 0.0)

    assert output.active is False
    assert output.policy_vx == pytest.approx(0.2)
    assert output.policy_vy == 0.0


def test_servo_preserves_request_direction_and_zero_resets():
    servo = LowSpeedGaitServo()

    left = servo.update(0.0, 0.2, 0.0, 0.0)
    stopped = servo.update(0.0, 0.0, 0.0, 0.0)

    assert left.policy_vx == pytest.approx(0.0)
    assert left.policy_vy == pytest.approx(0.45)
    assert stopped.active is False
    assert stopped.reason == "no_request"


def test_servo_rejects_invalid_hysteresis():
    with pytest.raises(ValueError):
        LowSpeedGaitServo(on_below_mps=0.16, off_above_mps=0.15)


def test_linear_servo_reset_clears_filter_and_active_pulse():
    servo = LowSpeedGaitServo(filter_alpha=0.2)
    servo.update(0.2, 0.0, 0.0, 0.0)
    servo.update(0.2, 0.0, 0.12, 0.04)

    servo.reset()
    stopped = servo.update(0.0, 0.0, 0.0, 0.0)

    assert stopped.active is False
    assert stopped.filtered_speed_mps == 0.0


def test_motion_guard_latches_before_raw_gate_and_requires_stable_release():
    guard = LowSpeedMotionGuard(release_steps=2)

    assert guard.update(0.10, 0.10).active is False
    triggered = guard.update(0.18, 0.10)
    assert triggered.active is True
    assert triggered.reason == "speed"
    assert guard.update(0.04, 0.04).active is True
    released = guard.update(0.04, 0.04)
    assert released.active is False
    assert released.reason == "released"


def test_motion_guard_yaw_trigger_and_reset():
    guard = LowSpeedMotionGuard()
    assert guard.update(0.0, -0.20).active is True
    assert guard.update(0.0, 0.0).active is True
    guard.reset()
    assert guard.update(0.0, 0.0).active is False


def test_lateral_motion_does_not_suppress_forward_pulse_below_combined_limit():
    servo = LowSpeedGaitServo(
        on_below_mps=0.06,
        off_above_mps=0.115,
        hard_limit_mps=0.16,
        total_hard_limit_mps=0.165,
    )

    output = servo.update(0.2, 0.0, 0.02, 0.16)

    assert output.active is True
    assert output.actual_lateral_request_mps == pytest.approx(0.16)
    assert output.effective_along_hard_limit_mps == pytest.approx(0.040311, abs=1e-5)


def test_combined_limit_stops_pulse_before_total_speed_exceeds_bound():
    servo = LowSpeedGaitServo(
        on_below_mps=0.06,
        off_above_mps=0.115,
        hard_limit_mps=0.16,
        total_hard_limit_mps=0.165,
    )
    servo.update(0.2, 0.0, 0.0, 0.0)

    output = servo.update(0.2, 0.0, 0.06, 0.16)

    assert output.active is False
    assert output.reason == "combined_speed_hard_limit"


def test_yaw_servo_pulses_through_low_rate_deadzone():
    servo = LowSpeedYawServo(filter_alpha=1.0)

    start = servo.update(0.15, 0.0)
    release = servo.update(0.15, 0.15)

    assert start.active is True
    assert start.policy_wz == pytest.approx(0.30)
    assert release.active is False


def test_yaw_servo_preserves_sign_and_stops_on_zero_request():
    servo = LowSpeedYawServo(filter_alpha=1.0)

    right = servo.update(-0.15, 0.0)
    stopped = servo.update(0.0, -0.10)

    assert right.policy_wz == pytest.approx(-0.30)
    assert stopped.policy_wz == 0.0
    assert stopped.reason == "no_request"


def test_yaw_servo_does_not_amplify_tiny_request_or_high_actual_rate():
    servo = LowSpeedYawServo(filter_alpha=1.0)

    tiny = servo.update(0.0024, 0.0)
    fast = servo.update(-0.2, 0.8)

    assert tiny.policy_wz == 0.0
    assert tiny.reason == "below_min_request"
    assert fast.policy_wz == 0.0
    assert fast.reason == "yaw_hard_limit"


def test_yaw_servo_uses_single_pulse_then_cooldown():
    servo = LowSpeedYawServo(filter_alpha=1.0, pulse_steps=1, positive_pulse_steps=1, cooldown_steps=2)

    pulse = servo.update(0.15, 0.0)
    cooldown = servo.update(0.15, 0.0)

    assert pulse.active is True
    assert cooldown.active is False
    assert cooldown.reason == "yaw_cooldown"


def test_yaw_servo_burst_is_bounded_then_cools_down():
    servo = LowSpeedYawServo(filter_alpha=1.0, pulse_steps=3, positive_pulse_steps=3, cooldown_steps=2)

    outputs = [servo.update(0.15, 0.0) for _ in range(4)]

    assert [row.active for row in outputs] == [True, True, True, False]
    assert outputs[-1].reason == "yaw_cooldown"


def test_positive_yaw_uses_longer_burst_than_negative_yaw():
    servo = LowSpeedYawServo(filter_alpha=1.0, pulse_steps=2, positive_pulse_steps=4, cooldown_steps=2)
    positive = [servo.update(0.15, 0.0).active for _ in range(4)]
    servo.reset()
    negative = [servo.update(-0.15, 0.0).active for _ in range(4)]

    assert positive == [True, True, True, True]
    assert negative == [True, True, False, False]
