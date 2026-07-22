from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "internvla_nav2_adapter"))
sys.path.insert(0, str(ROOT / "internvla_go2_controller"))
sys.path.insert(0, str(ROOT / "scripts"))

from internvla_go2_controller.runtime import JerkLimitedTwist  # noqa: E402
from internvla_nav2_adapter.motion_primitives import (  # noqa: E402
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    system2_local_poses,
)
from internnav_go2_runtime import _integrate_t5_planar_pose  # noqa: E402


def _yaw(rotation_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(value) for value in rotation_wxyz]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def test_t5_nav2_tolerances_are_stricly_inside_action_gate_slack() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/internnav_t5/nav2_static_lidar.yaml").read_text(
            encoding="utf-8"
        )
    )
    controller = config["controller_server"]["ros__parameters"]
    checker = controller["general_goal_checker"]
    dwb = controller["FollowPath"]
    assert 0.0 < checker["xy_goal_tolerance"] < 0.05
    assert 0.0 < checker["yaw_goal_tolerance"] < math.radians(3.0)
    assert 0.0 < dwb["xy_goal_tolerance"] < 0.05


def test_t5_system2_turns_in_place_but_t4_retains_arc() -> None:
    left = system2_local_poses(
        ACTION_LEFT, t5_completion_sim=True, forward_step_m=0.25
    )
    right = system2_local_poses(
        ACTION_RIGHT, t5_completion_sim=True, forward_step_m=0.25
    )
    assert len(left) == len(right) == 9
    assert all(x == 0.0 and y == 0.0 for x, y, _ in left + right)
    assert math.isclose(left[-1][2], math.radians(15.0), abs_tol=1.0e-12)
    assert math.isclose(right[-1][2], -math.radians(15.0), abs_tol=1.0e-12)

    frozen_t4 = system2_local_poses(
        ACTION_LEFT, t5_completion_sim=False, forward_step_m=0.35
    )
    assert frozen_t4[-1][0] > 0.0
    assert frozen_t4[-1][1] > 0.0
    assert math.isclose(frozen_t4[-1][2], math.radians(15.0), abs_tol=1.0e-12)


def test_t5_forward_primitive_remains_exact_quarter_meter() -> None:
    forward = system2_local_poses(
        ACTION_FORWARD, t5_completion_sim=True, forward_step_m=0.25
    )
    assert forward[0] == (0.0, 0.0, 0.0)
    assert forward[-1] == (0.25, 0.0, 0.0)


def test_planar_integrator_reaches_turn_in_40_ticks_and_forward_in_60() -> None:
    position = np.asarray([1.0, 2.0, 0.31], dtype=np.float64)
    rotation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(40):
        position, rotation = _integrate_t5_planar_pose(
            position, rotation, 0.0, math.radians(15.0) / 2.0, 0.05
        )
    assert np.allclose(position, [1.0, 2.0, 0.31], atol=1.0e-12)
    assert math.isclose(_yaw(rotation), math.radians(15.0), abs_tol=1.0e-12)
    assert rotation[1] == 0.0 and rotation[2] == 0.0

    position = np.asarray([1.0, 2.0, 0.31], dtype=np.float64)
    rotation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(60):
        position, rotation = _integrate_t5_planar_pose(
            position, rotation, 0.25 / 3.0, 0.0, 0.05
        )
    assert np.allclose(position, [1.25, 2.0, 0.31], atol=1.0e-12)
    assert math.isclose(_yaw(rotation), 0.0, abs_tol=1.0e-12)


def test_emergency_stop_zeroes_limiter_before_pose_integration() -> None:
    limiter = JerkLimitedTwist(
        max_linear=0.15,
        max_angular=0.6,
        max_linear_acceleration=0.4,
        max_angular_acceleration=1.2,
        max_linear_jerk=1.5,
        max_angular_jerk=4.0,
    )
    moving = limiter.step(0.15, 0.6, 0.05)
    assert moving.linear_x > 0.0 and moving.angular_z > 0.0
    stopped = limiter.step(0.15, 0.6, 0.05, emergency_stop=True)
    assert stopped.linear_x == stopped.angular_z == 0.0
    before_position = np.asarray([1.0, 2.0, 0.31], dtype=np.float64)
    before_rotation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    after_position, after_rotation = _integrate_t5_planar_pose(
        before_position,
        before_rotation,
        stopped.linear_x,
        stopped.angular_z,
        0.05,
    )
    assert np.array_equal(after_position, before_position)
    assert np.array_equal(after_rotation, before_rotation)


def test_runtime_keeps_planar_write_profile_gated_and_strict_velocity_path() -> None:
    runtime = (ROOT / "scripts/internnav_go2_runtime.py").read_text(encoding="utf-8")
    assert runtime.count("articulation.set_world_pose") == 1
    assert (
        'if self.motion_profile == "t5_completion_planar_root_velocity":\n'
        "                self._apply_t5_planar_root_pose"
    ) in runtime
    assert "else:\n                self._apply_root_velocity" in runtime
    assert "emergency_stop=self.emergency_stop" in runtime
    assert "angular_z,\n                self.dt," in runtime
    assert "self.planar_position: np.ndarray | None = None" in runtime
    assert "self.planar_position = None" in runtime
    assert "self.planar_position = next_position" in runtime
    assert "self.planar_rotation = next_rotation" in runtime
    runner = (ROOT / "scripts/run_t5_distributed_isaac.sh").read_text(
        encoding="utf-8"
    )
    assert "export INTERNVLA_GO2_PHYSICS_HZ=20" in runner
    assert "export INTERNVLA_GO2_CONTROL_HZ=20" in runner
    assert '"bounded_twist_integrated_as_planar_root_pose": True' in runner
    assert '"gravity_dynamics_claimed": False' in runner


def test_preflight_records_t5_deviation_and_strict_prohibition() -> None:
    preflight = (ROOT / "scripts/preflight_go2_continuous.py").read_text(
        encoding="utf-8"
    )
    assert '"scope": "t5_completion_sim_navigation_only"' in preflight
    assert '"deviation": "20hz_bounded_planar_se2_root_pose"' in preflight
    assert '"allowed": False,\n            "scope": "strict_physics"' in preflight
    assert 'runtime_text.count("articulation.set_world_pose") == 1' in preflight
