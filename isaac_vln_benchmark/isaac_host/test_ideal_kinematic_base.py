import math

import pytest

from ideal_kinematic_base import IdealKinematicBase


def test_ideal_base_integrates_body_velocity_in_world_frame():
    base = IdealKinematicBase(yaw=math.pi / 2)

    base.integrate(vx_body=0.2, vy_body=0.0, wz=0.1, dt=2.0)

    assert base.x == pytest.approx(0.0, abs=1e-9)
    assert base.y == pytest.approx(0.4)
    assert base.yaw == pytest.approx(math.pi / 2 + 0.2)


def test_ideal_base_reset_is_explicit_and_height_remains_fixed():
    base = IdealKinematicBase(z=0.40)
    base.reset([1.0, -2.0, 3.2])

    assert base.pose() == pytest.approx([1.0, -2.0, 3.2 - 2 * math.pi])
    assert base.z == 0.40
    with pytest.raises(ValueError):
        base.reset([0.0, 0.0])


def test_ideal_base_telemetry_uses_integrated_heading():
    base = IdealKinematicBase(x=1.0, y=2.0, yaw=math.pi / 2.0, z=0.4)

    sample = base.telemetry(vx_body=0.2, vy_body=0.0, wz=-0.3)

    assert sample["heading"] == pytest.approx(math.pi / 2.0)
    assert sample["linear_velocity"] == pytest.approx([0.0, 0.2, 0.0], abs=1e-7)
    assert sample["angular_velocity"] == pytest.approx([0.0, 0.0, -0.3])
