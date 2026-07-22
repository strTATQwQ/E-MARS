#!/usr/bin/env python3
"""Fail-closed structural preflight for the T3 continuous Go2 path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import internnav_go2_runtime
from internnav_go2_runtime import install_go2_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    install_go2_runtime()
    spec = importlib.util.spec_from_file_location("go2_t3_preflight_cfg", args.config)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load T3 Go2 config")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from internnav.configs.evaluator.vln_default_config import get_config

    config = get_config(module.eval_cfg)
    robot = config.task.robot
    controllers = [item.controller_settings for item in robot.controllers]
    sensors = [item.sensor_settings for item in robot.sensors]
    runtime_source = Path(internnav_go2_runtime.__file__).resolve()
    runtime_text = runtime_source.read_text(encoding="utf-8")
    forbidden_robot_pose_patterns = [
        pattern
        for pattern in ("articulation.set_world_pose", "self.robot.set_world_pose")
        if pattern in runtime_text
    ]
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "execution_mode": "continuous_physics_cmd_vel",
        "robot_type": robot.robot_settings["type"],
        "robot_position": list(robot.robot_settings["position"]),
        "robot_flash": bool(config.task.robot_flash),
        "one_step_stand_still": bool(config.task.one_step_stand_still),
        "controller_names": [item["name"] for item in controllers],
        "controller_types": [item["type"] for item in controllers],
        "sensor_prim_paths": {item["name"]: item["prim_path"] for item in sensors},
        "physics_dt": float(config.env.env_settings["physics_dt"]),
        "rendering_interval": int(config.env.env_settings["rendering_interval"]),
        "robot_offset": list(config.dataset.dataset_settings["robot_offset"]),
        "forbidden_controllers": [],
        "forbidden_robot_world_pose_patterns": forbidden_robot_pose_patterns,
        "runtime_source_sha256": hashlib.sha256(runtime_source.read_bytes()).hexdigest(),
    }
    expected_rendering_interval_text = os.environ.get(
        "INTERNVLA_GO2_EXPECT_RENDERING_INTERVAL", "5"
    )
    if not expected_rendering_interval_text.isdecimal():
        raise RuntimeError("expected rendering interval must be an integer")
    expected_rendering_interval = int(expected_rendering_interval_text)
    if expected_rendering_interval not in {4, 5, 8}:
        raise RuntimeError("expected rendering interval must be 4, 5, or 8")
    payload["expected_rendering_interval"] = expected_rendering_interval
    expected_physics_dt_text = os.environ.get(
        "INTERNVLA_GO2_EXPECT_PHYSICS_DT", "0.005"
    )
    try:
        expected_physics_dt = float(expected_physics_dt_text)
    except ValueError as error:
        raise RuntimeError("expected physics dt must be numeric") from error
    if not math.isfinite(expected_physics_dt) or expected_physics_dt not in {
        0.005,
        0.05,
    }:
        raise RuntimeError("expected physics dt must be 0.005 or 0.05")
    motion_profile = os.environ.get(
        "INTERNVLA_GO2_MOTION_PROFILE", "physics_root_velocity"
    )
    if motion_profile not in {
        "physics_root_velocity",
        "t5_completion_planar_root_velocity",
    }:
        raise RuntimeError("unexpected Go2 motion profile")
    payload["expected_physics_dt"] = expected_physics_dt
    payload["motion_profile"] = motion_profile
    if motion_profile == "t5_completion_planar_root_velocity":
        assert expected_physics_dt == 0.05
        assert expected_rendering_interval in {4, 8}
        assert os.environ.get("INTERNVLA_GO2_PHYSICS_HZ") == "20"
        assert os.environ.get("INTERNVLA_GO2_CONTROL_HZ") == "20"
        expected_sensor_hz = "2.5" if expected_rendering_interval == 8 else "5"
        assert os.environ.get("INTERNVLA_GO2_SENSOR_HZ") == expected_sensor_hz
        payload["root_pose_write_policy"] = {
            "allowed": True,
            "scope": "t5_completion_sim_navigation_only",
            "deviation": "20hz_bounded_planar_se2_root_pose",
            "high_fidelity_go2_dynamics_claimed": False,
        }
    else:
        assert expected_physics_dt == 0.005
        assert expected_rendering_interval == 5
        payload["root_pose_write_policy"] = {
            "allowed": False,
            "scope": "strict_physics",
            "deviation": None,
            "high_fidelity_go2_dynamics_claimed": True,
        }
    forbidden = {"move_by_flash", "move_by_discrete"}.intersection(
        payload["controller_names"]
    )
    payload["forbidden_controllers"] = sorted(forbidden)
    assert payload["robot_type"] == "VLNGo2Robot"
    assert payload["robot_flash"] is False
    assert payload["one_step_stand_still"] is True
    assert payload["controller_names"] == ["stand_still", "vln_dp_move_by_speed"]
    assert payload["controller_types"] == [
        "Go2StandStillController",
        "Go2ContinuousController",
    ]
    assert not forbidden
    # The shared source contains one explicitly profile-gated T5 pose write.
    # Strict physics prohibits selecting that branch; T5 records the deviation
    # above. Any additional robot pose-write call remains fail-closed.
    assert runtime_text.count("articulation.set_world_pose") == 1
    assert forbidden_robot_pose_patterns == ["articulation.set_world_pose"]
    assert (
        'if self.motion_profile == "t5_completion_planar_root_velocity":'
        in runtime_text
    )
    assert payload["sensor_prim_paths"]["pano_camera_0"] == "base/internvla_camera"
    assert "tp_pointcloud" not in payload["sensor_prim_paths"]
    assert abs(payload["physics_dt"] - expected_physics_dt) < 1e-12
    assert payload["rendering_interval"] == expected_rendering_interval
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
