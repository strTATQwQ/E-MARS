#!/usr/bin/env python3
"""Fail-closed structural check for the final Go2 InternNav configuration."""

from __future__ import annotations

import argparse
import importlib.util
import json

from internnav_go2_runtime import install_go2_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    install_go2_runtime()
    spec = importlib.util.spec_from_file_location("go2_preflight_cfg", args.config)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Go2 config")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from internnav.configs.evaluator.vln_default_config import get_config

    config = get_config(module.eval_cfg)
    robot = config.task.robot
    controllers = [item.controller_settings for item in robot.controllers]
    sensors = [item.sensor_settings for item in robot.sensors]
    payload = {
        "schema_version": 1,
        "robot_type": robot.robot_settings["type"],
        "robot_usd_path": robot.robot_settings["usd_path"],
        "robot_position": list(robot.robot_settings["position"]),
        "controller_names": [item["name"] for item in controllers],
        "sensor_prim_paths": {item["name"]: item["prim_path"] for item in sensors},
        "robot_offset": list(config.dataset.dataset_settings["robot_offset"]),
        "success_distance": config.task.metric.metric_setting["metric_config"]["success_distance"],
    }
    assert payload["robot_type"] == "VLNGo2Robot"
    assert payload["controller_names"] == ["stand_still", "move_by_flash"]
    assert payload["sensor_prim_paths"]["pano_camera_0"] == "base/internvla_camera"
    assert "tp_pointcloud" not in payload["sensor_prim_paths"]
    assert payload["success_distance"] == 3.0
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
