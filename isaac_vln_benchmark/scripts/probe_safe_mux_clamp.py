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
from isaac_vln_benchmark.v6_recovery_utils import (
    cmd_vel_for_primitive,
    evaluate_safe_mux_clamp_records,
    write_csv,
    write_jsonl,
    write_probe_summary,
)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def scheduler_config() -> dict[str, Any]:
    return load_yaml(ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler" / "config" / "scheduler_isaac_real_models.yaml")


def run_mock(config: dict[str, Any], output: Path) -> dict[str, Any]:
    cfg = scheduler_config()
    probe = config.get("probe", {})
    primitive = {
        "primitive": str(probe.get("primitive", "move_forward")),
        "distance_m": float(probe.get("distance_m", 0.5)),
        "speed_mps": float(probe.get("speed_mps", 0.15)),
        "ttl_sec": float(probe.get("ttl_sec", 1.0)),
        "source": "forced_probe",
    }
    candidate = cmd_vel_for_primitive(primitive, cfg)
    max_x = float(cfg.get("primitive", {}).get("max_linear_x", 0.0))
    safe = {"linear_x": max(-max_x, min(max_x, candidate["linear_x"])), "angular_z": candidate["angular_z"]}
    rows = [
        {
            "t": 0.0,
            "primitive": primitive,
            "cmd_vel_candidate": candidate,
            "safe_cmd_vel": safe,
            "safety_status": {"local_costmap_clear": True},
            "max_linear_x_source": max_x,
            "stale": False,
        }
    ]
    metrics = evaluate_safe_mux_clamp_records(rows, path_m=0.45, collision_count=0)
    write_outputs(output, rows, metrics)
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
    cfg = scheduler_config()
    os.environ["ROS_DOMAIN_ID"] = str(int(bench.get("ros_domain_id", 42)))
    if not rclpy.ok():
        rclpy.init(args=None)
        owns_rclpy = True
    else:
        owns_rclpy = False
    node = rclpy.create_node("probe_safe_mux_clamp")
    reset_pub = node.create_publisher(String, "/isaac/reset_episode", 10)
    mode_pub = node.create_publisher(String, "/benchmark/mode_json", 10)
    state_pub = node.create_publisher(String, "/scheduler/state", 10)
    primitive_pub = node.create_publisher(String, "/primitive/command_json", 10)
    events: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    last_candidate = {"linear_x": 0.0, "angular_z": 0.0}
    last_safe = {"linear_x": 0.0, "angular_z": 0.0}
    last_safety: dict[str, Any] = {"local_costmap_clear": True}

    def publish_json(pub, payload: dict[str, Any]) -> None:
        pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def on_metric(msg: Any) -> None:
        events.append(_safe_json(msg.data))

    def on_odom(msg: Any) -> None:
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        trajectory.append({"episode_id": episode_id, "t": time.monotonic() - started, "pose": [float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), yaw], "source": "odom"})

    def on_candidate(msg: Twist) -> None:
        nonlocal last_candidate
        last_candidate = {"linear_x": float(msg.linear.x), "angular_z": float(msg.angular.z)}

    def on_safe(msg: Twist) -> None:
        nonlocal last_safe
        last_safe = {"linear_x": float(msg.linear.x), "angular_z": float(msg.angular.z)}

    def on_safety(msg: Any) -> None:
        nonlocal last_safety
        last_safety = _safe_json(msg.data)

    node.create_subscription(String, "/metrics/event_jsonl", on_metric, 100)
    node.create_subscription(Odometry, "/odom", on_odom, 50)
    node.create_subscription(Twist, "/cmd_vel_candidate", on_candidate, 50)
    node.create_subscription(Twist, "/safe_cmd_vel", on_safe, 50)
    node.create_subscription(String, "/safety/local_status_json", on_safety, 50)
    episode_id = f"safe_mux_clamp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    reset_payload = {
        "task_id": str(probe.get("reset_task_id", "simple_001_seed0")),
        "scene_id": str(probe.get("reset_scene_id", "straight_001")),
        "episode_id": episode_id,
    }
    mode_payload = {
        "mode": "probe_safe_mux_clamp",
        "mode_config": {
            "use_step": False,
            "use_omninav": False,
            "use_internnav": False,
            "pending_policy": "stop",
        },
        "episode_id": episode_id,
    }
    started = time.monotonic()
    reset_wait_until = time.monotonic() + float(probe.get("reset_discovery_timeout_sec", 4.0))
    while time.monotonic() < reset_wait_until and reset_pub.get_subscription_count() <= 0:
        rclpy.spin_once(node, timeout_sec=0.05)
    reset_publishes = 0
    reset_until = time.monotonic() + float(probe.get("reset_settle_sec", 1.5))
    next_reset_publish = 0.0
    while time.monotonic() < reset_until:
        now_mono = time.monotonic()
        if now_mono >= next_reset_publish:
            publish_json(reset_pub, reset_payload)
            reset_publishes += 1
            next_reset_publish = now_mono + float(probe.get("reset_publish_interval_sec", 0.2))
        rclpy.spin_once(node, timeout_sec=0.05)
    for _ in range(3):
        publish_json(mode_pub, mode_payload)
        state_pub.publish(String(data="IDLE"))
        rclpy.spin_once(node, timeout_sec=0.1)

    # Reset telemetry can contain a large pose discontinuity from the previous
    # episode. Start the motion measurement after reset settle so path_m only
    # reflects the forced primitive under test.
    trajectory.clear()
    records.clear()
    started = time.monotonic()

    primitive_base = {
        "primitive": str(probe.get("primitive", "move_forward")),
        "distance_m": float(probe.get("distance_m", 0.5)),
        "speed_mps": float(probe.get("speed_mps", 0.15)),
        "ttl_sec": float(probe.get("ttl_sec", 1.0)),
        "source": "forced_probe",
        "confidence": 1.0,
    }
    deadline = time.monotonic() + float(bench.get("max_duration_sec", 8.0))
    repeat_deadline = time.monotonic() + float(probe.get("command_repeat_sec", 4.0))
    interval = float(probe.get("command_publish_interval_sec", 0.2))
    next_publish = 0.0
    command_count = 0
    target_path = float(config.get("targets", {}).get("path_m_min", 0.3))
    while time.monotonic() < deadline:
        state_pub.publish(String(data="IDLE"))
        now_mono = time.monotonic()
        if now_mono <= repeat_deadline and now_mono >= next_publish:
            request_id = f"safe_mux_clamp_{command_count:02d}"
            ts = node_ros_now_sec(node)
            primitive = attach_timebase(
                dict(primitive_base, request_id=request_id),
                node=node,
                episode_id=episode_id,
                request_id=request_id,
                source_stamp=ts,
                created_ros_time=ts,
            )
            publish_json(primitive_pub, primitive)
            command_count += 1
            next_publish = now_mono + interval
        rclpy.spin_once(node, timeout_sec=0.05)
        records.append(
            {
                "t": time.monotonic() - started,
                "primitive": primitive_base,
                "cmd_vel_candidate": dict(last_candidate),
                "safe_cmd_vel": dict(last_safe),
                "safety_status": dict(last_safety),
                "max_linear_x_source": cfg.get("primitive", {}).get("max_linear_x"),
                "stale": False,
                "safety_reason": _latest_safe_mux_result(events),
            }
        )
        if len(trajectory) >= 2 and path_length(trajectory) > target_path:
            break
    path_m = path_length(trajectory) if len(trajectory) >= 2 else 0.0
    stale = sum(1 for event in events if str(event.get("event_type") or "").endswith("_stale") or event.get("discard"))
    metrics = evaluate_safe_mux_clamp_records(records, path_m=round(path_m, 3), collision_count=0)
    metrics["stale_discard_count"] = stale
    metrics["failures"] = [item for item in metrics.get("failures", []) if item != "stale_discard_count != 0"]
    if stale:
        metrics["pass"] = False
        metrics.setdefault("failures", []).append("stale_discard_count != 0")
    else:
        metrics["pass"] = not metrics.get("failures")
    metrics["primitive_command_count"] = command_count
    metrics["config_max_linear_x"] = cfg.get("primitive", {}).get("max_linear_x")
    metrics["reset_publish_count"] = reset_publishes
    metrics["reset_subscription_count"] = reset_pub.get_subscription_count()
    write_outputs(output, records, metrics, events=events, trajectory=trajectory)
    node.destroy_node()
    if owns_rclpy and rclpy.ok():
        rclpy.shutdown()
    return metrics


def write_outputs(
    output: Path,
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
    trajectory: list[dict[str, Any]] | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(output / "safe_mux_probe.jsonl", rows)
    write_jsonl(output / "events.jsonl", events or rows)
    write_csv(output / "cmd_vel_trace.csv", _flatten_rows(rows))
    write_csv(output / "mode_table.csv", [{"mode": "probe_safe_mux_clamp", "episodes": 1, "samples": len(rows), "pass": metrics.get("pass", False)}])
    write_csv(output / "failure_table.csv", [{"failure_reason": item, "count": 1} for item in metrics.get("failures", [])] or [{"failure_reason": "none", "count": 0}])
    traj_rows = []
    for row in trajectory or []:
        pose = row.get("pose") or [0.0, 0.0, 0.0]
        traj_rows.append({"episode_id": row.get("episode_id", ""), "t": row.get("t", 0.0), "x": pose[0], "y": pose[1], "yaw": pose[2], "source": row.get("source", "odom")})
    write_csv(output / "trajectory.csv", traj_rows or [{"episode_id": "", "t": 0.0, "x": 0.0, "y": 0.0, "yaw": 0.0, "source": "none"}])
    write_probe_summary(output / "summary.md", "Safe Mux Clamp Probe V6", metrics)


def _flatten_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "t": row.get("t", 0.0),
                "candidate_linear_x": row.get("cmd_vel_candidate", {}).get("linear_x", 0.0),
                "candidate_angular_z": row.get("cmd_vel_candidate", {}).get("angular_z", 0.0),
                "safe_linear_x": row.get("safe_cmd_vel", {}).get("linear_x", 0.0),
                "safe_angular_z": row.get("safe_cmd_vel", {}).get("angular_z", 0.0),
                "local_costmap_clear": row.get("safety_status", {}).get("local_costmap_clear", True),
                "safety_reason": row.get("safety_reason", ""),
                "max_linear_x_source": row.get("max_linear_x_source", ""),
            }
        )
    return out


def _latest_safe_mux_result(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if str(event.get("event_type") or "") == "safe_cmd_mux":
            return str(event.get("result") or "")
    return ""


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def _yaw_from_quat(q: Any) -> float:
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "probe_safe_mux_clamp.yaml"))
    parser.add_argument("--output", default="")
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remote-live", action="store_true")
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
    args.output = Path(args.output) if args.output else ROOT / "runs" / f"probe_safe_mux_clamp_{stamp}"
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    args.output.mkdir(parents=True, exist_ok=True)
    if args.remote_live:
        metrics = run_live(config, args.output)
    elif args.mock_models or args.dry_run:
        metrics = run_mock(config, args.output)
    else:
        return run_remote_probe(args, script_name="probe_safe_mux_clamp.py", config_path=config_path)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if metrics.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
