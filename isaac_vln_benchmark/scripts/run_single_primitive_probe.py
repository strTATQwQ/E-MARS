#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
SCHED_ROOT = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler"
for path in (PKG_ROOT, SCHED_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaac_vln_benchmark.metrics import path_length
from isaac_vln_benchmark.v4_remote import run_remote_probe
from isaac_vln_benchmark.v5_timebase_utils import evaluate_single_primitive, make_single_primitive_result, write_v5_run_artifacts


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_mock(config: dict[str, Any], output: Path) -> dict[str, Any]:
    primitive = str(config.get("probe", {}).get("primitive", "move_forward"))
    metrics = evaluate_single_primitive(make_single_primitive_result(primitive=primitive, yaw_delta_deg=30.5 if primitive.startswith("turn") else 0.0))
    trajectory = [
        {"episode_id": metrics["episode_id"], "t": 0.0, "pose": [0.0, 0.0, 0.0], "source": "mock"},
        {"episode_id": metrics["episode_id"], "t": 3.0, "pose": [metrics["path_m"], 0.0, math.radians(metrics["yaw_delta_deg"])], "source": "mock"},
    ]
    events = [{"event": "primitive_command", "episode_id": metrics["episode_id"], "primitive": primitive, "attribution": "valid", "discard": False}]
    write_v5_run_artifacts(output, title="single_primitive_probe", metrics=metrics, events=events, trajectory_rows=trajectory)
    return metrics


def run_live(config: dict[str, Any], output: Path) -> dict[str, Any]:
    import os
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String

    from omninav_step_scheduler.schemas import attach_timebase, node_ros_now_sec

    bench = config.get("benchmark", {})
    probe = config.get("probe", {})
    os.environ["ROS_DOMAIN_ID"] = str(int(bench.get("ros_domain_id", 42)))
    if not rclpy.ok():
        rclpy.init(args=None)
        owns_rclpy = True
    else:
        owns_rclpy = False
    node = rclpy.create_node("single_primitive_probe")
    mode_pub = node.create_publisher(String, "/benchmark/mode_json", 10)
    state_pub = node.create_publisher(String, "/scheduler/state", 10)
    primitive_pub = node.create_publisher(String, "/primitive/command_json", 10)
    events: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    candidate_cmd_samples: list[dict[str, float]] = []
    safe_cmd_samples: list[dict[str, float]] = []

    def publish_json(pub, payload: dict[str, Any]) -> None:
        pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def on_metric(msg: Any) -> None:
        try:
            events.append(json.loads(msg.data))
        except Exception:
            events.append({"raw": msg.data})

    def on_odom(msg: Any) -> None:
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        trajectory.append({"episode_id": episode_id, "t": time.monotonic() - started, "pose": [float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), yaw], "source": "odom"})

    def on_candidate_cmd(msg: Twist) -> None:
        candidate_cmd_samples.append(_twist_sample(msg, started))

    def on_safe_cmd(msg: Twist) -> None:
        safe_cmd_samples.append(_twist_sample(msg, started))

    node.create_subscription(String, "/metrics/event_jsonl", on_metric, 100)
    node.create_subscription(Odometry, "/odom", on_odom, 50)
    node.create_subscription(Twist, "/cmd_vel_candidate", on_candidate_cmd, 50)
    node.create_subscription(Twist, "/safe_cmd_vel", on_safe_cmd, 50)
    episode_id = f"single_primitive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    started = time.monotonic()
    mode_payload = {
        "mode": "single_primitive_probe",
        "mode_config": {
            "use_step": False,
            "use_omninav": False,
            "use_internnav": False,
            "step_policy": "never",
            "pending_policy": "stop",
        },
        "episode_id": episode_id,
    }
    for _ in range(3):
        publish_json(mode_pub, mode_payload)
        state_pub.publish(String(data="RUN_FAST"))
        rclpy.spin_once(node, timeout_sec=0.1)
    primitive = str(probe.get("primitive", "move_forward"))
    repeat_sec = float(probe.get("command_repeat_sec", 2.0))
    publish_interval_sec = float(probe.get("command_publish_interval_sec", 0.2))
    command_count = 0
    deadline = time.monotonic() + float(bench.get("max_duration_sec", 8.0))
    repeat_deadline = time.monotonic() + max(0.1, repeat_sec)
    next_publish = 0.0
    while time.monotonic() < deadline:
        state_pub.publish(String(data="RUN_FAST"))
        if time.monotonic() <= repeat_deadline and time.monotonic() >= next_publish:
            t0 = node_ros_now_sec(node)
            request_id = f"single_primitive_probe_{command_count:02d}"
            command = {
                "primitive": primitive,
                "distance_m": float(probe.get("distance_m", 0.5)),
                "yaw_deg": float(probe.get("yaw_deg", 30.0)),
                "confidence": 1.0,
                "source": "single_primitive_probe",
                "request_id": request_id,
                "ttl_sec": float(probe.get("ttl_sec", 1.0)),
            }
            command = attach_timebase(command, node=node, episode_id=episode_id, request_id=request_id, source_stamp=t0, created_ros_time=t0)
            publish_json(primitive_pub, command)
            command_count += 1
            next_publish = time.monotonic() + publish_interval_sec
        rclpy.spin_once(node, timeout_sec=0.05)
    if len(trajectory) >= 2:
        path_m = path_length(trajectory)
        yaw_delta_deg = abs(math.degrees(trajectory[-1]["pose"][2] - trajectory[0]["pose"][2]))
    else:
        path_m = 0.0
        yaw_delta_deg = 0.0
    stale = sum(1 for event in events if str(event.get("event_type") or "").endswith("_stale") or event.get("discard"))
    timebase_errors = sum(1 for event in events if event.get("attribution") == "timebase_error")
    metrics = evaluate_single_primitive(
        {
            "benchmark": "single_primitive_probe",
            "episode_id": episode_id,
            "primitive": primitive,
            "path_m": round(path_m, 3),
            "yaw_delta_deg": round(yaw_delta_deg, 3),
            "collision_count": 0,
            "stale_discard_count": stale,
            "timebase_error_count": timebase_errors,
            "episode_mismatch_count": sum(1 for event in events if event.get("attribution") == "episode_mismatch"),
            "missing_timestamp_count": sum(1 for event in events if event.get("attribution") == "missing_timestamp"),
            "stale_action_executed": 0,
            "actions_through_safe_mux": any(event.get("event_type") == "safe_cmd_mux" for event in events),
            "primitive_command_count": command_count,
            "primitive_accepted_count": sum(1 for event in events if event.get("event_type") == "primitive_command" and event.get("result") == "accepted"),
            "primitive_stopped_count": sum(1 for event in events if event.get("event_type") == "primitive_command" and event.get("result") == "stopped"),
            "candidate_cmd_sample_count": len(candidate_cmd_samples),
            "safe_cmd_sample_count": len(safe_cmd_samples),
            "candidate_nonzero_count": _nonzero_count(candidate_cmd_samples),
            "safe_nonzero_count": _nonzero_count(safe_cmd_samples),
            "max_candidate_linear_x_mps": round(max([abs(row["linear_x"]) for row in candidate_cmd_samples] or [0.0]), 3),
            "max_safe_linear_x_mps": round(max([abs(row["linear_x"]) for row in safe_cmd_samples] or [0.0]), 3),
            "max_candidate_yaw_rate_rps": round(max([abs(row["angular_z"]) for row in candidate_cmd_samples] or [0.0]), 3),
            "max_safe_yaw_rate_rps": round(max([abs(row["angular_z"]) for row in safe_cmd_samples] or [0.0]), 3),
        }
    )
    write_v5_run_artifacts(output, title="single_primitive_probe", metrics=metrics, events=events, trajectory_rows=trajectory)
    node.destroy_node()
    if owns_rclpy and rclpy.ok():
        rclpy.shutdown()
    return metrics


def _yaw_from_quat(q: Any) -> float:
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _twist_sample(msg: Any, started: float) -> dict[str, float]:
    return {
        "t": time.monotonic() - started,
        "linear_x": float(msg.linear.x),
        "linear_y": float(msg.linear.y),
        "angular_z": float(msg.angular.z),
    }


def _nonzero_count(samples: list[dict[str, float]], *, epsilon: float = 1e-4) -> int:
    return sum(1 for row in samples if abs(row["linear_x"]) > epsilon or abs(row["linear_y"]) > epsilon or abs(row["angular_z"]) > epsilon)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "single_primitive_probe.yaml"))
    parser.add_argument("--output", default="")
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remote-live", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--isaac-host", default="")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--ros-domain-id", type=int, default=42)
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output = Path(args.output) if args.output else ROOT / "runs" / f"single_primitive_probe_{stamp}"
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    args.output.mkdir(parents=True, exist_ok=True)

    if args.remote_live:
        metrics = run_live(config, args.output)
    elif args.mock_models or args.dry_run:
        metrics = run_mock(config, args.output)
    else:
        return run_remote_probe(args, script_name="run_single_primitive_probe.py", config_path=config_path)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if metrics.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
