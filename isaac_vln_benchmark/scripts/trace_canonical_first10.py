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

from isaac_vln_benchmark.v4_remote import run_remote_probe
from isaac_vln_benchmark.v6_recovery_utils import classify_first10_root_cause, write_csv, write_jsonl


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_mock(config: dict[str, Any], output: Path) -> dict[str, Any]:
    rows = [
        {
            "t": 0.0,
            "episode_id": "simple_001_seed0",
            "robot_pose": [0.0, 0.0, 0.0],
            "omninav_request_id": "req0",
            "omninav_raw_output": "wp=(0.0,0.45)",
            "raw_waypoint": [0.0, 0.45],
            "raw_action": "move_forward",
            "parsed_primitive": "move_forward",
            "waypoint_forward_axis": "y",
            "fallback_used": False,
            "cmd_vel_candidate": {"linear_x": 0.2, "angular_z": 0.0},
            "safe_cmd_vel": {"linear_x": 0.2, "angular_z": 0.0},
            "safe_mux_blocked": False,
            "safety_reason": None,
            "stale_decision": "valid",
            "stale_reason": None,
            "pose_delta": 0.2,
        }
    ]
    return write_outputs(output, rows)


def run_live(config: dict[str, Any], output: Path) -> dict[str, Any]:
    import os
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String

    bench = config.get("benchmark", {})
    os.environ["ROS_DOMAIN_ID"] = str(int(bench.get("ros_domain_id", 42)))
    if not rclpy.ok():
        rclpy.init(args=None)
        owns_rclpy = True
    else:
        owns_rclpy = False
    node = rclpy.create_node("trace_canonical_first10")
    reset_pub = node.create_publisher(String, "/isaac/reset_episode", 10)
    mode_pub = node.create_publisher(String, "/benchmark/mode_json", 10)
    state_pub = node.create_publisher(String, "/scheduler/state", 10)
    subgoal_pub = node.create_publisher(String, "/scheduler/active_subgoal_json", 10)
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    last_pose = [0.0, 0.0, 0.0]
    start_pose = [0.0, 0.0, 0.0]
    last_request: dict[str, Any] = {}
    last_action: dict[str, Any] = {}
    last_candidate = {"linear_x": 0.0, "angular_z": 0.0}
    last_safe = {"linear_x": 0.0, "angular_z": 0.0}
    last_safe_result = ""
    last_stale: dict[str, Any] = {}
    stale_request_ids: set[str] = set()
    accepted_request_ids: set[str] = set()

    def publish_json(pub, payload: dict[str, Any]) -> None:
        pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def on_metric(msg: Any) -> None:
        nonlocal last_safe_result, last_stale
        payload = _safe_json(msg.data)
        events.append(payload)
        event_type = str(payload.get("event_type") or "")
        if str(payload.get("event_type") or "") == "safe_cmd_mux":
            last_safe_result = str(payload.get("result") or "")
        if event_type == "omninav_action":
            request_id = str(payload.get("request_id") or "")
            if request_id:
                accepted_request_ids.add(request_id)
            last_stale = {}
        if event_type.endswith("_stale"):
            last_stale = payload
            request_id = str(payload.get("request_id") or "")
            if request_id:
                stale_request_ids.add(request_id)

    def on_request(msg: Any) -> None:
        nonlocal last_request
        last_request = _safe_json(msg.data)

    def on_action(msg: Any) -> None:
        nonlocal last_action
        last_action = _safe_json(msg.data)

    def on_candidate(msg: Twist) -> None:
        nonlocal last_candidate
        last_candidate = {"linear_x": float(msg.linear.x), "angular_z": float(msg.angular.z)}

    def on_safe(msg: Twist) -> None:
        nonlocal last_safe
        last_safe = {"linear_x": float(msg.linear.x), "angular_z": float(msg.angular.z)}

    def on_odom(msg: Any) -> None:
        nonlocal last_pose
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        last_pose = [float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), yaw]

    node.create_subscription(String, "/metrics/event_jsonl", on_metric, 100)
    node.create_subscription(String, "/omninav/request_json", on_request, 100)
    node.create_subscription(String, "/omninav/action_candidate_json", on_action, 100)
    node.create_subscription(Twist, "/cmd_vel_candidate", on_candidate, 50)
    node.create_subscription(Twist, "/safe_cmd_vel", on_safe, 50)
    node.create_subscription(Odometry, "/odom", on_odom, 50)

    duration = float(bench.get("duration_sec", 10.0))
    reset_publish_total = 0
    for task_name in list(config.get("tasks") or ["simple_001_seed0", "simple_001_seed1", "simple_002_seed0"]):
        episode_id = f"first10_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        last_request = {}
        last_action = {}
        last_stale = {}
        stale_request_ids.clear()
        accepted_request_ids.clear()
        subgoal = {
            "episode_id": episode_id,
            "subgoal": f"Move forward through {task_name}.",
            "success_condition": "Make forward progress in the simple corridor.",
        }
        mode_payload = {
            "mode": "omninav_only",
            "mode_config": {"use_step": False, "use_omninav": True, "use_internnav": False, "step_policy": "never", "pending_policy": "none"},
            "episode_id": episode_id,
        }
        reset_payload = {
            "task_id": _task_id_from_seed_name(task_name),
            "scene_id": str(bench.get("reset_scene_id", "straight_001")),
            "episode_id": episode_id,
        }
        reset_wait_until = time.monotonic() + float(bench.get("reset_discovery_timeout_sec", 4.0))
        while time.monotonic() < reset_wait_until and reset_pub.get_subscription_count() <= 0:
            rclpy.spin_once(node, timeout_sec=0.05)
        reset_until = time.monotonic() + float(bench.get("reset_settle_sec", 1.5))
        next_reset_publish = 0.0
        while time.monotonic() < reset_until:
            now_mono = time.monotonic()
            if now_mono >= next_reset_publish:
                publish_json(reset_pub, reset_payload)
                reset_publish_total += 1
                next_reset_publish = now_mono + float(bench.get("reset_publish_interval_sec", 0.2))
            rclpy.spin_once(node, timeout_sec=0.05)
        for _ in range(5):
            publish_json(mode_pub, mode_payload)
            publish_json(subgoal_pub, subgoal)
            state_pub.publish(String(data="IDLE"))
            rclpy.spin_once(node, timeout_sec=0.1)
        for _ in range(2):
            publish_json(mode_pub, mode_payload)
            publish_json(subgoal_pub, subgoal)
            state_pub.publish(String(data="RUN_FAST"))
            rclpy.spin_once(node, timeout_sec=0.1)
        start_pose = list(last_pose)
        started = time.monotonic()
        while time.monotonic() - started < duration:
            publish_json(subgoal_pub, subgoal)
            state_pub.publish(String(data="RUN_FAST"))
            rclpy.spin_once(node, timeout_sec=0.05)
            pose_delta = math.hypot(last_pose[0] - start_pose[0], last_pose[1] - start_pose[1])
            current_request_id = str(last_request.get("request_id") or "")
            current_stale = current_request_id in stale_request_ids and current_request_id not in accepted_request_ids
            rows.append(
                {
                    "t": round(time.monotonic() - started, 3),
                    "episode_id": episode_id,
                    "task_name": task_name,
                    "robot_pose": list(last_pose),
                    "omninav_request_id": current_request_id,
                    "omninav_raw_output": str(last_action.get("raw_text") or ""),
                    "raw_waypoint": last_action.get("raw_waypoint") or [],
                    "raw_action": str(last_action.get("raw_action") or ""),
                    "parsed_primitive": str(last_action.get("primitive") or ""),
                    "local_forward_m": _float(last_action.get("local_forward_m")),
                    "local_lateral_m": _float(last_action.get("local_lateral_m")),
                    "waypoint_forward_axis": str(last_action.get("waypoint_forward_axis") or ""),
                    "fallback_used": bool(last_action.get("fallback_used", False)),
                    "cmd_vel_candidate": dict(last_candidate),
                    "safe_cmd_vel": dict(last_safe),
                    "safe_mux_blocked": last_safe_result not in {"", "accepted"},
                    "safety_reason": last_safe_result or None,
                    "stale_decision": "discarded" if current_stale else "valid",
                    "stale_reason": last_stale.get("attribution") if current_stale else None,
                    "pose_delta": round(pose_delta, 3),
                }
            )
        state_pub.publish(String(data="IDLE"))
        rclpy.spin_once(node, timeout_sec=0.2)
    metrics = write_outputs(output, rows, events=events)
    metrics["reset_publish_count"] = reset_publish_total
    metrics["reset_subscription_count"] = reset_pub.get_subscription_count()
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    node.destroy_node()
    if owns_rclpy and rclpy.ok():
        rclpy.shutdown()
    return metrics


def write_outputs(output: Path, rows: list[dict[str, Any]], events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    root = classify_first10_root_cause(rows)
    metrics = {
        "benchmark": "trace_canonical_first10",
        "samples": len(rows),
        "root_cause": root,
        "max_candidate_linear_x_mps": max([abs(float(row.get("cmd_vel_candidate", {}).get("linear_x", 0.0))) for row in rows] or [0.0]),
        "max_safe_linear_x_mps": max([abs(float(row.get("safe_cmd_vel", {}).get("linear_x", 0.0))) for row in rows] or [0.0]),
        "max_pose_delta_m": max([abs(float(row.get("pose_delta", 0.0))) for row in rows] or [0.0]),
        "stale_rows": sum(1 for row in rows if row.get("stale_decision") == "discarded"),
        "pass": root == "unknown" or root == "isaac_does_not_move_despite_cmd" or root == "model_outputs_turn_only",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(output / "trace.jsonl", rows)
    write_jsonl(output / "events.jsonl", events or rows)
    write_csv(output / "cmd_vel_plot.csv", _plot_rows(rows))
    (output / "trace_summary.md").write_text(render_summary(metrics), encoding="utf-8")
    return metrics


def render_summary(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Canonical First10 Trace V6",
            "",
            f"- root_cause: {metrics['root_cause']}",
            f"- samples: {metrics['samples']}",
            f"- max_candidate_linear_x_mps: {metrics['max_candidate_linear_x_mps']}",
            f"- max_safe_linear_x_mps: {metrics['max_safe_linear_x_mps']}",
            f"- max_pose_delta_m: {metrics['max_pose_delta_m']}",
            f"- stale_rows: {metrics['stale_rows']}",
            "",
        ]
    )


def _plot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "episode_id": row.get("episode_id", ""),
                "t": row.get("t", 0.0),
                "candidate_linear_x": row.get("cmd_vel_candidate", {}).get("linear_x", 0.0),
                "candidate_angular_z": row.get("cmd_vel_candidate", {}).get("angular_z", 0.0),
                "safe_linear_x": row.get("safe_cmd_vel", {}).get("linear_x", 0.0),
                "safe_angular_z": row.get("safe_cmd_vel", {}).get("angular_z", 0.0),
                "pose_delta": row.get("pose_delta", 0.0),
                "parsed_primitive": row.get("parsed_primitive", ""),
                "stale_decision": row.get("stale_decision", ""),
                "safety_reason": row.get("safety_reason", ""),
            }
        )
    return out


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def _task_id_from_seed_name(task_name: str) -> str:
    parts = str(task_name).split("_seed", 1)
    return parts[0] if parts[0] else str(task_name)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _yaw_from_quat(q: Any) -> float:
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "trace_canonical_first10.yaml"))
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
    args.output = Path(args.output) if args.output else ROOT / "runs" / f"first10_trace_v6_{stamp}"
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    args.output.mkdir(parents=True, exist_ok=True)
    if args.remote_live:
        metrics = run_live(config, args.output)
    elif args.mock_models or args.dry_run:
        metrics = run_mock(config, args.output)
    else:
        return run_remote_probe(args, script_name="trace_canonical_first10.py", config_path=config_path, mock_omninav=True)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
