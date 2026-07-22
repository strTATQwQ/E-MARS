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
from isaac_vln_benchmark.v5_timebase_utils import (
    evaluate_timebase_probe,
    make_timebase_probe_rows,
    write_v5_run_artifacts,
)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_mock(config: dict[str, Any], output: Path) -> dict[str, Any]:
    count = int(config.get("benchmark", {}).get("samples", 10))
    rows = make_timebase_probe_rows(count)
    metrics = evaluate_timebase_probe(rows, max_reasonable_age_sec=float(config.get("benchmark", {}).get("max_reasonable_age_sec", 1.0)))
    events = [
        {
            "event": "stale_gate_timebase_probe_sample",
            "episode_id": row["episode_id"],
            "request_id": row["request_id"],
            "attribution": row["discard_reason"],
            "discard": row["discard"],
            "computed_age_sec": row["computed_age_sec"],
        }
        for row in rows
    ]
    write_v5_run_artifacts(output, title="stale_gate_timebase_probe", metrics=metrics, events=events, timebase_rows=rows)
    return metrics


def run_live(config: dict[str, Any], output: Path) -> dict[str, Any]:
    import rclpy
    from std_msgs.msg import String

    from omninav_step_scheduler.schemas import attach_timebase, node_ros_now_sec

    bench = config.get("benchmark", {})
    sample_count = int(bench.get("samples", 10))
    ros_domain_id = int(bench.get("ros_domain_id", 42))
    import os

    os.environ["ROS_DOMAIN_ID"] = str(ros_domain_id)
    if not rclpy.ok():
        rclpy.init(args=None)
        owns_rclpy = True
    else:
        owns_rclpy = False
    node = rclpy.create_node("stale_gate_timebase_probe")
    mode_pub = node.create_publisher(String, "/benchmark/mode_json", 10)
    state_pub = node.create_publisher(String, "/scheduler/state", 10)
    subgoal_pub = node.create_publisher(String, "/scheduler/active_subgoal_json", 10)
    action_pub = node.create_publisher(String, "/omninav/action_candidate_json", 10)
    events: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    def publish_json(pub, payload: dict[str, Any]) -> None:
        pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def on_metric(msg: Any) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {"raw": msg.data}
        events.append(payload)

    def on_request(msg: Any) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {"raw": msg.data}
        requests.append(payload)
        events.append({"event_type": "probe_observed_omninav_request", **payload})

    node.create_subscription(String, "/metrics/event_jsonl", on_metric, 100)
    node.create_subscription(String, "/omninav/request_json", on_request, 100)
    episode_id = f"timebase_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mode_payload = {"mode": "omninav_only", "mode_config": {"step": {"enabled": False}, "omninav": {"enabled": True}}, "episode_id": episode_id}
    for _ in range(3):
        publish_json(mode_pub, mode_payload)
        publish_json(subgoal_pub, {"episode_id": episode_id, "subgoal": "stationary timebase probe", "success_condition": "probe"})
        state_pub.publish(String(data="RUN_FAST"))
        rclpy.spin_once(node, timeout_sec=0.1)

    used_request_ids: set[str] = set()
    for index in range(sample_count):
        deadline = time.time() + 3.0
        req = None
        while time.time() < deadline:
            state_pub.publish(String(data="RUN_FAST"))
            publish_json(subgoal_pub, {"episode_id": episode_id, "subgoal": "stationary timebase probe", "success_condition": "probe"})
            rclpy.spin_once(node, timeout_sec=0.1)
            for candidate in requests:
                request_id = str(candidate.get("request_id") or "")
                if request_id and request_id not in used_request_ids and str(candidate.get("episode_id") or "") == episode_id:
                    req = candidate
                    used_request_ids.add(request_id)
                    break
            if req is not None:
                break
        if req is None:
            req = {
                "request_id": f"timebase_probe_missing_request_{index:02d}",
                "episode_id": episode_id,
                "timestamp": node_ros_now_sec(node),
                "timestamp_request": node_ros_now_sec(node),
                "pose": [0.0, 0.0, 0.0],
                "clock_domain": "unknown",
            }
        request_id = str(req.get("request_id") or f"timebase_probe_{index:02d}")
        t_req = float(req.get("timestamp_request", req.get("timestamp", node_ros_now_sec(node))) or node_ros_now_sec(node))
        t_response = node_ros_now_sec(node)
        action = {
            "request_id": request_id,
            "episode_id": episode_id,
            "timestamp_request": t_req,
            "timestamp_response": t_response,
            "frame_timestamp": req.get("header_stamp_sec", req.get("source_stamp_sec", t_req)),
            "pose_at_snapshot": req.get("pose", [0.0, 0.0, 0.0]),
            "primitive": "stop",
            "distance_m": 0.0,
            "yaw_deg": 0.0,
            "confidence": 1.0,
            "source": "timebase_probe",
            "ttl_sec": 1.0,
        }
        action = attach_timebase(
            action,
            node=node,
            episode_id=episode_id,
            request_id=request_id,
            clock_domain=str(req.get("clock_domain") or "unknown"),
            header_stamp=req.get("header_stamp_sec", t_req),
            source_stamp=t_response,
            created_ros_time=t_response,
        )
        publish_json(action_pub, action)
        time.sleep(0.05)
        rclpy.spin_once(node, timeout_sec=0.15)
        t_gate = node_ros_now_sec(node)
        stale_for_request = [
            event
            for event in events
            if str(event.get("request_id") or "") == request_id and str(event.get("event_type") or "") in {"omninav_stale"}
        ]
        discard_reason = str((stale_for_request[-1].get("attribution") if stale_for_request else "valid") or "valid")
        rows.append(
            {
                "episode_id": episode_id,
                "request_id": request_id,
                "clock_domain": action.get("clock_domain", "unknown"),
                "ros_now": t_gate,
                "wall_now": time.time(),
                "clock_msg_time": action.get("clock_msg_sec"),
                "image_header_stamp": action.get("header_stamp_sec"),
                "frame_bundle_timestamp": action.get("frame_timestamp"),
                "request_start_time": t_req,
                "model_response_time": action.get("timestamp_response"),
                "action_timestamp": action.get("source_stamp_sec"),
                "stale_gate_current_time": t_gate,
                "computed_age_sec": max(0.0, t_gate - float(action.get("source_stamp_sec") or t_gate)),
                "pose_delta_m": 0.0,
                "discard": bool(stale_for_request),
                "discard_reason": discard_reason,
            }
        )
    metrics = evaluate_timebase_probe(rows, max_reasonable_age_sec=float(bench.get("max_reasonable_age_sec", 1.0)))
    write_v5_run_artifacts(output, title="stale_gate_timebase_probe", metrics=metrics, events=events, timebase_rows=rows)
    node.destroy_node()
    if owns_rclpy and rclpy.ok():
        rclpy.shutdown()
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "stale_gate_timebase_probe.yaml"))
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
    if args.max_episodes is not None:
        config.setdefault("benchmark", {})["samples"] = int(args.max_episodes)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output = Path(args.output) if args.output else ROOT / "runs" / f"timebase_probe_{stamp}"
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    args.output.mkdir(parents=True, exist_ok=True)

    if args.remote_live:
        metrics = run_live(config, args.output)
    elif args.mock_models or args.dry_run:
        metrics = run_mock(config, args.output)
    else:
        return run_remote_probe(args, script_name="run_stale_gate_timebase_probe.py", config_path=config_path)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if metrics.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
