#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_ROOT = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler"
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
for path in (SCHEDULER_ROOT, PKG_ROOT):
    sys.path.insert(0, str(path))

from isaac_vln_benchmark.perception_planning_suite import evaluate_robustness, percentile
from omninav_step_scheduler.step_roles import TargetTrackState


def safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def latency_samples(paths: list[str]) -> list[float]:
    values = []
    for raw_path in paths:
        rows = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        for row in rows:
            for event in row.get("step_http_events", []):
                if event.get("latency_s") is not None and str(event.get("result")) == "accepted":
                    values.append(float(event["latency_s"]))
    return values


def valid_route_payload(*, request_id: str, episode_id: str, stamp: float, clock_domain: str = "ros_system") -> dict[str, Any]:
    return {
        "request_id": request_id,
        "episode_id": episode_id,
        "mission_id": episode_id,
        "route_choice": "left",
        "confidence": 0.9,
        "evidence": "red cone",
        "visible_in_view": "left",
        "clock_domain": clock_domain,
        "source_stamp_sec": stamp,
        "created_ros_time_sec": stamp,
        "header_stamp_sec": stamp,
    }


def track_reset_check() -> bool:
    tracker = TargetTrackState(required_hits=2, high_confidence_single_hit=1.01)
    tracker.update(timestamp=1.0, episode_id="old", target="traffic cone", visible=True, confidence=0.9, frame_seq=1)
    old = tracker.update(timestamp=1.1, episode_id="old", target="traffic cone", visible=True, confidence=0.9, frame_seq=2)
    new = tracker.update(timestamp=2.0, episode_id="new", target="traffic cone", visible=False, confidence=0.0, frame_seq=3)
    return bool(old["confirmed"] and not new["confirmed"] and new["episode_id"] == "new" and new["hits"] == 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live decision transport robustness probe without locomotion testing.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--latency-cases", action="append", default=[])
    parser.add_argument("--scheduler-config", required=True)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String

    class ProbeNode(Node):
        def __init__(self) -> None:
            super().__init__("decision_robustness_probe")
            self.primitives: list[dict[str, Any]] = []
            self.metrics: list[dict[str, Any]] = []
            self.candidates: list[dict[str, float]] = []
            self.mode_pub = self.create_publisher(String, "/benchmark/mode_json", 10)
            self.route_pub = self.create_publisher(String, "/step/route_choice_json", 20)
            self.request_pub = self.create_publisher(String, "/step/request_json", 10)
            self.state_pub = self.create_publisher(String, "/scheduler/state", 10)
            self.create_subscription(String, "/primitive/command_json", self.on_primitive, 50)
            self.create_subscription(String, "/metrics/event_jsonl", self.on_metric, 100)
            self.create_subscription(Twist, "/cmd_vel_candidate", self.on_candidate, 50)

        def on_primitive(self, msg) -> None:
            self.primitives.append(safe_json(msg.data))

        def on_metric(self, msg) -> None:
            self.metrics.append(safe_json(msg.data))

        def on_candidate(self, msg) -> None:
            self.candidates.append({"linear_x": float(msg.linear.x), "angular_z": float(msg.angular.z)})

        @staticmethod
        def publish_json(pub, payload: dict[str, Any]) -> None:
            pub.publish(String(data=json.dumps(payload)))

        def spin_for(self, seconds: float) -> None:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)

        def set_episode(self, episode_id: str) -> None:
            self.publish_json(
                self.mode_pub,
                {"mode": "decision_robustness_v14", "episode_id": episode_id, "mode_config": {"stale_gate_enabled": True}},
            )
            self.spin_for(0.25)

    def metric_seen(node: ProbeNode, *, request_id: str, attribution: str, start: int) -> bool:
        return any(
            str(row.get("request_id") or (row.get("decision") or {}).get("request_id") or "") == request_id
            and str(row.get("attribution") or row.get("discard_reason") or "") == attribution
            for row in node.metrics[start:]
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    latency_values = latency_samples(args.latency_cases)
    p95 = percentile(latency_values, 0.95)
    rows: list[dict[str, Any]] = []
    rclpy.init()
    node = ProbeNode()
    try:
        node.spin_for(1.0)

        node.set_episode("robust_duplicate")
        start_p, start_m = len(node.primitives), len(node.metrics)
        stamp = node.get_clock().now().nanoseconds / 1.0e9
        duplicate = valid_route_payload(request_id="duplicate_request", episode_id="robust_duplicate", stamp=stamp)
        node.publish_json(node.route_pub, duplicate)
        node.spin_for(0.3)
        node.publish_json(node.route_pub, duplicate)
        node.spin_for(0.2)
        node.publish_json(node.route_pub, duplicate)
        node.spin_for(1.5)
        duplicate_primitives = [row for row in node.primitives[start_p:] if row.get("request_id") == "duplicate_request"]
        duplicate_attributed = metric_seen(
            node, request_id="duplicate_request", attribution="duplicate_response", start=start_m
        )
        rows.append(
            {
                "profile": "duplicate",
                "pass": len(duplicate_primitives) == 1 and duplicate_attributed,
                "primitive_count": len(duplicate_primitives),
                "duplicate_attributed": duplicate_attributed,
                "stale_action_executed": 0,
            }
        )

        node.set_episode("robust_order")
        start_p, start_m = len(node.primitives), len(node.metrics)
        stamp = node.get_clock().now().nanoseconds / 1.0e9
        node.publish_json(node.route_pub, valid_route_payload(request_id="order_new", episode_id="robust_order", stamp=stamp))
        node.spin_for(0.25)
        node.publish_json(node.route_pub, valid_route_payload(request_id="order_old", episode_id="robust_order", stamp=stamp - 0.1))
        node.spin_for(0.5)
        old_primitives = [row for row in node.primitives[start_p:] if row.get("request_id") == "order_old"]
        rows.append({"profile": "out_of_order", "pass": not old_primitives and metric_seen(node, request_id="order_old", attribution="out_of_order", start=start_m), "stale_action_executed": 0})

        node.set_episode("robust_current")
        start_p, start_m = len(node.primitives), len(node.metrics)
        stamp = node.get_clock().now().nanoseconds / 1.0e9
        node.publish_json(node.route_pub, valid_route_payload(request_id="episode_wrong", episode_id="robust_other", stamp=stamp))
        node.spin_for(0.5)
        rows.append({"profile": "episode_mismatch", "pass": not any(row.get("request_id") == "episode_wrong" for row in node.primitives[start_p:]) and metric_seen(node, request_id="episode_wrong", attribution="episode_mismatch", start=start_m), "stale_action_executed": 0})

        start_p, start_m = len(node.primitives), len(node.metrics)
        stamp = node.get_clock().now().nanoseconds / 1.0e9
        node.publish_json(node.route_pub, valid_route_payload(request_id="timebase_wrong", episode_id="robust_current", stamp=stamp, clock_domain="monotonic"))
        node.spin_for(0.5)
        rows.append({"profile": "timestamp_mismatch", "pass": not any(row.get("request_id") == "timebase_wrong" for row in node.primitives[start_p:]) and metric_seen(node, request_id="timebase_wrong", attribution="timebase_error", start=start_m), "stale_action_executed": 0})

        node.set_episode("robust_old")
        node.set_episode("robust_new")
        start_p, start_m = len(node.primitives), len(node.metrics)
        stamp = node.get_clock().now().nanoseconds / 1.0e9
        node.publish_json(node.route_pub, valid_route_payload(request_id="old_after_reset", episode_id="robust_old", stamp=stamp))
        node.spin_for(0.5)
        old_track_cleared = track_reset_check()
        rows.append({"profile": "reset", "pass": not any(row.get("request_id") == "old_after_reset" for row in node.primitives[start_p:]) and metric_seen(node, request_id="old_after_reset", attribution="old_response_after_reset", start=start_m) and old_track_cleared, "old_track_cleared": old_track_cleared, "stale_action_executed": 0})

        node.set_episode("robust_drop")
        candidate_start, primitive_start, metric_start = len(node.candidates), len(node.primitives), len(node.metrics)
        node.state_pub.publish(String(data="STEP_THINK_STOP"))
        node.spin_for(0.1)
        stamp = node.get_clock().now().nanoseconds / 1.0e9
        dropped_request = valid_route_payload(request_id="dropped_http", episode_id="robust_drop", stamp=stamp)
        dropped_request.update({"role": "route_choice", "multimodal": True, "endpoint": "http://127.0.0.1:1/v1/chat/completions", "model": "unreachable_transport_probe", "prompt": {"messages": []}, "pose_at_request": [0.0, 0.0, 0.0]})
        node.publish_json(node.request_pub, dropped_request)
        node.spin_for(1.0)
        candidates = node.candidates[candidate_start:]
        blind_motion = any(abs(row["linear_x"]) > 1.0e-3 for row in candidates)
        rows.append({"profile": "drop", "pass": bool(candidates) and not blind_motion and not any(row.get("request_id") == "dropped_http" for row in node.primitives[primitive_start:]), "stale_action_executed": 0, "fallback_or_mock": 0})
    finally:
        node.destroy_node()
        rclpy.shutdown()

    config_text = Path(args.scheduler_config).read_text(encoding="utf-8")
    rows.append({"profile": "horizontal_flip", "pass": "horizontal_flip: true" in config_text, "stale_action_executed": 0})
    rows.append({"profile": "delay", "pass": p95 is not None and p95 <= 7.0, "step_latency_sec": p95, "stale_action_executed": 0})
    result = evaluate_robustness(rows)
    (output / "robustness_results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (output / "robustness_gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
