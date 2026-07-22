#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from isaac_vln_benchmark.v6_recovery_utils import (
    cmd_vel_for_primitive,
    evaluate_action_parser_records,
    write_csv,
    write_jsonl,
    write_probe_summary,
)
from omninav_step_scheduler.omninav_model_client_node import waypoint_tensor_to_action


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def scheduler_config() -> dict[str, Any]:
    path = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler" / "config" / "scheduler_isaac_real_models.yaml"
    return load_yaml(path)


def run_mock(config: dict[str, Any], output: Path) -> dict[str, Any]:
    cfg = scheduler_config()
    rows: list[dict[str, Any]] = []
    waypoints = [[[0.0, 0.45]], [[0.04, 0.35]], [[0.40, 0.05]], [[0.0, 0.0]], []]
    sample_count = int(config.get("benchmark", {}).get("samples", 20))
    for index in range(sample_count):
        wp = waypoints[index % len(waypoints)]
        action = waypoint_tensor_to_action(wp, arrive_pred=0.05, config=cfg.get("model_clients", {}).get("omninav", {}) | cfg.get("primitive", {}))
        primitive = {
            "primitive": action["primitive"],
            "distance_m": action["distance_m"],
            "yaw_deg": action["yaw_deg"],
        }
        cmd = cmd_vel_for_primitive(primitive, cfg)
        rows.append(record_from_action(index, "mock_parser_probe", action, cmd, config))
    metrics = evaluate_action_parser_records(rows)
    write_outputs(output, rows, metrics)
    return metrics


def run_live(config: dict[str, Any], output: Path) -> dict[str, Any]:
    import os
    import rclpy
    from std_msgs.msg import String

    from omninav_step_scheduler.schemas import attach_timebase, node_ros_now_sec

    cfg = scheduler_config()
    bench = config.get("benchmark", {})
    probe = config.get("probe", {})
    os.environ["ROS_DOMAIN_ID"] = str(int(bench.get("ros_domain_id", 42)))
    if not rclpy.ok():
        rclpy.init(args=None)
        owns_rclpy = True
    else:
        owns_rclpy = False
    node = rclpy.create_node("probe_omninav_action_parser")
    reset_pub = node.create_publisher(String, "/isaac/reset_episode", 10)
    mode_pub = node.create_publisher(String, "/benchmark/mode_json", 10)
    state_pub = node.create_publisher(String, "/scheduler/state", 10)
    subgoal_pub = node.create_publisher(String, "/scheduler/active_subgoal_json", 10)
    request_pub = node.create_publisher(String, "/omninav/request_json", 10)
    actions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    def publish_json(pub, payload: dict[str, Any]) -> None:
        pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def on_action(msg: Any) -> None:
        payload = _safe_json(msg.data)
        actions.append(payload)
        events.append({"event_type": "probe_observed_action_candidate", **payload})

    def on_metric(msg: Any) -> None:
        events.append(_safe_json(msg.data))

    node.create_subscription(String, "/omninav/action_candidate_json", on_action, 100)
    node.create_subscription(String, "/metrics/event_jsonl", on_metric, 100)
    episode_id = f"{probe.get('episode_id_prefix', 'parser_probe')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mode_payload = {
        "mode": "probe_omninav_action_parser",
        "mode_config": {"use_step": False, "use_omninav": False, "use_internnav": False},
        "episode_id": episode_id,
    }
    active_subgoal = dict(probe.get("active_subgoal") or {})
    active_subgoal["episode_id"] = episode_id
    reset_payload = {
        "task_id": str(probe.get("reset_task_id", "simple_001")),
        "scene_id": str(probe.get("reset_scene_id", "straight_001")),
        "episode_id": episode_id,
    }
    reset_wait_until = time.monotonic() + float(probe.get("reset_discovery_timeout_sec", 4.0))
    while time.monotonic() < reset_wait_until and reset_pub.get_subscription_count() <= 0:
        rclpy.spin_once(node, timeout_sec=0.05)
    reset_publish_count = 0
    reset_until = time.monotonic() + float(probe.get("reset_settle_sec", 1.5))
    next_reset_publish = 0.0
    while time.monotonic() < reset_until:
        now_mono = time.monotonic()
        if now_mono >= next_reset_publish:
            publish_json(reset_pub, reset_payload)
            reset_publish_count += 1
            next_reset_publish = now_mono + float(probe.get("reset_publish_interval_sec", 0.2))
        state_pub.publish(String(data="IDLE"))
        rclpy.spin_once(node, timeout_sec=0.05)
    for _ in range(3):
        publish_json(mode_pub, mode_payload)
        publish_json(subgoal_pub, active_subgoal)
        state_pub.publish(String(data="IDLE"))
        rclpy.spin_once(node, timeout_sec=0.1)

    rows: list[dict[str, Any]] = []
    sample_count = int(bench.get("samples", 20))
    wait_response_sec = float(bench.get("wait_response_sec", 4.0))
    request_interval_sec = float(bench.get("request_interval_sec", 0.25))
    used: set[str] = set()
    for index in range(sample_count):
        t0 = node_ros_now_sec(node)
        request_id = f"parser_probe_{index:02d}"
        request = {
            "request_id": request_id,
            "episode_id": episode_id,
            "timestamp": t0,
            "timestamp_request": t0,
            "pose": list(probe.get("pose") or [0.0, 0.0, 0.0]),
            "instruction": str(probe.get("instruction") or "Move forward through the corridor."),
            "subgoal": active_subgoal,
            "frame_count": int(probe.get("frame_count", 3)),
            "use_left_right": bool(probe.get("use_left_right", False)),
            "use_history": bool(probe.get("use_history", False)),
        }
        request = attach_timebase(request, node=node, episode_id=episode_id, request_id=request_id, source_stamp=t0, created_ros_time=t0)
        publish_json(request_pub, request)
        deadline = time.monotonic() + wait_response_sec
        action = None
        while time.monotonic() < deadline:
            state_pub.publish(String(data="IDLE"))
            rclpy.spin_once(node, timeout_sec=0.05)
            for candidate in actions:
                if str(candidate.get("request_id") or "") == request_id and request_id not in used:
                    action = candidate
                    used.add(request_id)
                    break
            if action is not None:
                break
        if action is None:
            action = {
                "request_id": request_id,
                "primitive": "stop",
                "distance_m": 0.0,
                "yaw_deg": 0.0,
                "confidence": 0.0,
                "raw_text": "missing_model_response",
                "raw_waypoint": [],
                "raw_action": "stop:missing_model_response",
                "fallback_used": True,
                "fallback_reason": "missing_model_response",
                "parser_reason": "missing_model_response",
            }
        cmd = cmd_vel_for_primitive(action, cfg)
        rows.append(record_from_action(index, episode_id, action, cmd, config, request=request))
        time.sleep(request_interval_sec)

    metrics = evaluate_action_parser_records(rows)
    metrics["reset_publish_count"] = reset_publish_count
    metrics["reset_subscription_count"] = reset_pub.get_subscription_count()
    write_outputs(output, rows, metrics, events=events)
    node.destroy_node()
    if owns_rclpy and rclpy.ok():
        rclpy.shutdown()
    return metrics


def record_from_action(
    index: int,
    episode_id: str,
    action: dict[str, Any],
    cmd: dict[str, float],
    config: dict[str, Any],
    *,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probe = config.get("probe", {})
    request = request or {}
    return {
        "sample_id": f"sample_{index:02d}",
        "episode_id": episode_id,
        "mission_id": str(action.get("mission_id") or request.get("mission_id") or ""),
        "request_id": str(action.get("request_id") or request.get("request_id") or ""),
        "instruction": str(request.get("instruction") or probe.get("instruction") or ""),
        "active_subgoal": request.get("subgoal") or probe.get("active_subgoal") or {},
        "image_bundle_summary": {
            "front_stamp": action.get("frame_timestamp", action.get("header_stamp_sec", 0.0)),
            "left_stamp": 0.0,
            "right_stamp": 0.0,
            "num_frames": int(request.get("frame_count", probe.get("frame_count", 0)) or 0),
            "history_len": int(bool(request.get("use_history", probe.get("use_history", False)))),
        },
        "raw_model_output": str(action.get("raw_text") or ""),
        "raw_waypoint": action.get("raw_waypoint") or [],
        "raw_action": str(action.get("raw_action") or ""),
        "parsed_primitive": str(action.get("primitive") or "unknown"),
        "local_forward_m": _float(action.get("local_forward_m")),
        "local_lateral_m": _float(action.get("local_lateral_m")),
        "waypoint_forward_axis": str(action.get("waypoint_forward_axis") or ""),
        "distance_m": _float(action.get("distance_m")),
        "yaw_deg": _float(action.get("yaw_deg")),
        "confidence": action.get("confidence"),
        "fallback_used": bool(action.get("fallback_used", False)),
        "fallback_reason": action.get("fallback_reason") or None,
        "parser_reason": str(action.get("parser_reason") or ""),
        "cmd_vel_candidate": cmd,
    }


def write_outputs(output: Path, rows: list[dict[str, Any]], metrics: dict[str, Any], events: list[dict[str, Any]] | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(output / "parser_probe.jsonl", rows)
    write_jsonl(output / "action_parse_trace.jsonl", rows)
    write_jsonl(output / "events.jsonl", events or rows)
    write_csv(output / "action_distribution.csv", [{"primitive": key, "count": value} for key, value in sorted(_counts(rows).items())])
    write_csv(output / "mode_table.csv", [{"mode": "probe_omninav_action_parser", "episodes": 1, "samples": len(rows), "pass": metrics.get("pass", False)}])
    write_csv(output / "failure_table.csv", [{"failure_reason": item, "count": 1} for item in metrics.get("failures", [])] or [{"failure_reason": "none", "count": 0}])
    write_csv(output / "trajectory.csv", [{"episode_id": rows[0]["episode_id"] if rows else "", "t": 0.0, "x": 0.0, "y": 0.0, "yaw": 0.0, "source": "stationary"}])
    write_probe_summary(output / "summary.md", "OmniNav Action Parser Probe V6", metrics)


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get("parsed_primitive") or "unknown")
        out[key] = out.get(key, 0) + 1
    return out


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "probe_omninav_action_parser.yaml"))
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
    args.output = Path(args.output) if args.output else ROOT / "runs" / f"probe_omninav_action_parser_{stamp}"
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    args.output.mkdir(parents=True, exist_ok=True)

    if args.remote_live:
        metrics = run_live(config, args.output)
    elif args.mock_models or args.dry_run:
        metrics = run_mock(config, args.output)
    else:
        return run_remote_probe(args, script_name="probe_omninav_action_parser.py", config_path=config_path, mock_omninav=True)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if metrics.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
