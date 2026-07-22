from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import omninav_step_scheduler.primitive_executor_node as primitive_module
import omninav_step_scheduler.safe_cmd_mux_node as safe_mux_module
from omninav_step_scheduler.primitive_executor_node import PrimitiveExecutorNode
from omninav_step_scheduler.safe_cmd_mux_node import SafeCmdMuxNode
from omninav_step_scheduler.safe_cmd_mux_node import rate_limit_scalar


def test_live_isaac_config_has_positive_linear_x():
    cfg = _scheduler_config()

    assert cfg["primitive"]["max_linear_x"] > 0.0
    assert cfg["primitive"]["normal_speed_mps"] > 0.0
    assert cfg["safety"]["dry_run"] is False


def test_forced_forward_canonical_config_produces_positive_safe_linear_x():
    primitive_module.Twist = _twist_class
    safe_mux_module.Twist = _twist_class
    cfg = _scheduler_config()
    primitive_node = object.__new__(PrimitiveExecutorNode)
    primitive_node.config = cfg
    primitive_node.current_pose = [0.0, 0.0, 0.0]
    twist = PrimitiveExecutorNode.primitive_to_twist(
        primitive_node,
        {"primitive": "move_forward", "distance_m": 0.5, "ttl_sec": 1.0},
    )
    mux = object.__new__(SafeCmdMuxNode)
    mux.config = cfg
    clamped = SafeCmdMuxNode.clamp(mux, twist)

    assert twist.linear.x > 0.0
    assert clamped.linear.x > 0.0
    assert clamped.linear.x <= cfg["primitive"]["max_linear_x"]


def test_primitive_executor_uses_requested_speed_with_safety_clamp():
    primitive_module.Twist = _twist_class
    cfg = _scheduler_config()
    primitive_node = object.__new__(PrimitiveExecutorNode)
    primitive_node.config = cfg
    primitive_node.current_pose = [0.0, 0.0, 0.0]

    twist = PrimitiveExecutorNode.primitive_to_twist(
        primitive_node,
        {"primitive": "move_forward", "distance_m": 0.5, "speed_mps": cfg["primitive"]["max_linear_x"] + 1.0},
    )

    assert twist.linear.x == cfg["primitive"]["max_linear_x"]


def test_v10_enter_branch_primitive_is_clamped_by_executor_limits():
    primitive_module.Twist = _twist_class
    cfg = _scheduler_config()
    primitive_node = object.__new__(PrimitiveExecutorNode)
    primitive_node.config = cfg
    primitive_node.current_pose = [0.0, 0.0, 0.0]

    twist = PrimitiveExecutorNode.primitive_to_twist(
        primitive_node,
        {
            "primitive": "enter_branch",
            "phase": "enter",
            "linear_x_mps": cfg["primitive"]["max_linear_x"] + 1.0,
            "angular_z_radps": cfg["primitive"]["max_yaw_rate"] + 1.0,
            "max_linear_x_mps": 10.0,
            "max_yaw_rate_radps": 10.0,
        },
    )

    assert twist.linear.x == cfg["primitive"]["max_linear_x"]
    assert twist.angular.z == cfg["primitive"]["max_yaw_rate"]


def test_v10_controller_verify_phase_outputs_stop_twist():
    primitive_module.Twist = _twist_class
    cfg = _scheduler_config()
    primitive_node = object.__new__(PrimitiveExecutorNode)
    primitive_node.config = cfg
    primitive_node.current_pose = [0.0, 0.0, 0.0]

    twist = PrimitiveExecutorNode.primitive_to_twist(
        primitive_node,
        {
            "primitive": "target_relative_approach",
            "phase": "verify",
            "linear_x_mps": 0.2,
            "angular_z_radps": 0.2,
        },
    )

    assert twist.linear.x == 0.0
    assert twist.angular.z == 0.0


def test_rate_limit_scalar_applies_acceleration_limit():
    assert rate_limit_scalar(0.0, 0.20, 0.08, 0.5) == pytest.approx(0.04)
    assert rate_limit_scalar(0.20, 0.0, 0.08, 0.5, 0.40) == pytest.approx(0.0)


def test_rate_limit_scalar_can_be_disabled():
    assert rate_limit_scalar(0.0, 0.20, 0.0, 0.05) == pytest.approx(0.20)


def _twist_class():
    return SimpleNamespace(
        linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )


def _scheduler_config():
    path = Path(__file__).resolve().parents[1] / "config" / "scheduler_isaac_real_models.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
