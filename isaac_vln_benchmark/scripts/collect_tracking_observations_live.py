#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_ROOT = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler"
BENCHMARK_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
for path in (SCHEDULER_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaac_vln_benchmark.perception_planning_suite import (
    audit_step_visible_request,
    reset_payload_for_visual_case,
    tracking_visual_case,
)
from omninav_step_scheduler.step_roles import build_semantic_stop_prompt


def safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect genuine per-frame multimodal Step tracking observations.")
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="http://10.100.100.128:8096/v1/chat/completions")
    parser.add_argument("--model", default="Step-3.7-flash-IQ3_XXS-00001-of-00002.gguf")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--max-sequences", type=int, default=0)
    args = parser.parse_args()

    import rclpy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from omninav_step_scheduler.step_http_client_node import ros_image_to_data_url

    class CollectorNode(rclpy.node.Node):
        def __init__(self) -> None:
            super().__init__("step_tracking_observation_collector")
            self.image_count = 0
            self.latest_image = None
            self.responses: dict[str, dict[str, Any]] = {}
            self.reset_acks: list[dict[str, Any]] = []
            self.primitives: list[dict[str, Any]] = []
            self.metrics: list[dict[str, Any]] = []
            self.mode_pub = self.create_publisher(String, "/benchmark/mode_json", 10)
            self.reset_pub = self.create_publisher(String, "/isaac/reset_episode", 10)
            self.instruction_pub = self.create_publisher(String, "/user_instruction", 10)
            self.request_pub = self.create_publisher(String, "/step/request_json", 10)
            self.create_subscription(Image, "/camera/front/image", self.on_image, 2)
            self.create_subscription(String, "/step/semantic_stop_json", self.on_response, 20)
            self.create_subscription(String, "/isaac/reset_ack_json", self.on_reset_ack, 20)
            self.create_subscription(String, "/primitive/command_json", self.on_primitive, 50)
            self.create_subscription(String, "/metrics/event_jsonl", self.on_metric, 100)

        def on_image(self, msg) -> None:
            self.image_count += 1
            self.latest_image = msg

        def on_response(self, msg) -> None:
            payload = safe_json(msg.data)
            request_id = str(payload.get("request_id") or "")
            if request_id:
                self.responses[request_id] = payload

        def on_reset_ack(self, msg) -> None:
            self.reset_acks.append(safe_json(msg.data))

        def on_primitive(self, msg) -> None:
            self.primitives.append(safe_json(msg.data))

        def on_metric(self, msg) -> None:
            self.metrics.append(safe_json(msg.data))

        def publish_json(self, publisher, payload: dict[str, Any]) -> None:
            publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))

        def spin_until(self, predicate, timeout_sec: float) -> bool:
            deadline = time.monotonic() + timeout_sec
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                if predicate():
                    return True
            return False

    sequences = json.loads(Path(args.sequences).read_text(encoding="utf-8"))
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]
    output = Path(args.output)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    (output / "collected_sequences.json").write_text(
        json.dumps(sequences, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    observations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    rclpy.init()
    node = CollectorNode()
    try:
        node.spin_until(
            lambda: node.request_pub.get_subscription_count() > 0
            and node.reset_pub.get_subscription_count() > 0,
            8.0,
        )
        for sequence in sequences:
            episode_id = str(sequence["episode_id"])
            mode = {
                "mode": "perception_tracking_v14",
                "episode_id": episode_id,
                "mode_config": {
                    "use_step": True,
                    "use_omninav": False,
                    "step_roles_only": True,
                    "external_step_triggers_only": True,
                    "stale_gate_enabled": True,
                    "controller_decision_source": "step",
                    "pending_policy": "stop",
                },
            }
            previous: dict[str, Any] | None = None
            for frame in sequence["frames"]:
                frame_index = int(frame["frame_index"])
                if previous is not None and int(frame["frame_seq"]) == int(previous["planned_frame_seq"]):
                    duplicate = dict(previous)
                    duplicate.update(
                        {
                            "frame_index": frame_index,
                            "timestamp": float(previous["timestamp"]) + 0.1,
                            "replayed_duplicate": True,
                            "action_triggered": False,
                        }
                    )
                    observations.append(duplicate)
                    continue

                case = tracking_visual_case(sequence, frame)
                request_id = f"tracking_{sequence['sequence_id']}_{frame_index:02d}"
                reset = reset_payload_for_visual_case(case, episode_id=episode_id)
                ack_start = len(node.reset_acks)
                image_start = node.image_count
                ack = False
                for _ in range(3):
                    node.publish_json(node.mode_pub, mode)
                    node.publish_json(node.reset_pub, reset)
                    ack = node.spin_until(
                        lambda: any(
                            str(row.get("episode_id") or "") == episode_id
                            for row in node.reset_acks[ack_start:]
                        ),
                        1.0,
                    )
                    if ack:
                        break
                node.publish_json(node.mode_pub, mode)
                node.spin_until(lambda: False, 0.1)
                node.publish_json(
                    node.instruction_pub,
                    {"instruction": case["instruction"], "mission_id": episode_id},
                )
                image_start = node.image_count
                image_ready = node.spin_until(lambda: node.image_count >= image_start + 3, 4.0)
                prompt = build_semantic_stop_prompt(
                    instruction=case["instruction"],
                    active_subgoal=case["target"],
                    robot_state={},
                    semantic_summary={},
                    safety_status={},
                    event={"type": "visual_decision_due", "target": case["target"]},
                )
                stamp = node.get_clock().now().nanoseconds / 1.0e9
                request = {
                    "request_id": request_id,
                    "episode_id": episode_id,
                    "mission_id": episode_id,
                    "timestamp_request": stamp,
                    "source_stamp_sec": stamp,
                    "created_ros_time_sec": stamp,
                    "clock_domain": "ros_system",
                    "role": "semantic_stop",
                    "multimodal": True,
                    "pose_at_request": [0.0, 0.0, 0.0],
                    "pending_mode": "stop",
                    "instruction": case["instruction"],
                    "active_subgoal": case["target"],
                    "target": case["target"],
                    "visual_ground_truth_hidden": True,
                    "endpoint": args.endpoint,
                    "model": args.model,
                    "max_tokens": 40,
                    "temperature": 0.0,
                    "prompt": prompt,
                }
                audit = audit_step_visible_request(request)
                primitive_start = len(node.primitives)
                metric_start = len(node.metrics)
                image_path = None
                if node.latest_image is not None and (
                    frame_index in {0, len(sequence["frames"]) // 2, len(sequence["frames"]) - 1}
                    or bool(frame.get("occluded"))
                ):
                    data_url = ros_image_to_data_url(
                        node.latest_image,
                        max_width=384,
                        jpeg_quality=90,
                        horizontal_flip=True,
                        vertical_flip=True,
                    )
                    image_path = images / f"{sequence['sequence_id']}_{frame_index:02d}.jpg"
                    image_path.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
                node.publish_json(node.request_pub, request)
                received = node.spin_until(lambda: request_id in node.responses, args.timeout_sec)
                response = node.responses.get(request_id)
                if not received or not isinstance(response, dict):
                    failures.append(
                        {
                            "sequence_id": sequence["sequence_id"],
                            "frame_index": frame_index,
                            "reason": "missing_step_response",
                        }
                    )
                    continue
                node.spin_until(lambda: False, 0.2)
                snapshot = response.get("image_snapshot") if isinstance(response.get("image_snapshot"), dict) else {}
                frame_seq = int(snapshot.get("frame_seq", -1))
                primitives = [
                    row
                    for row in node.primitives[primitive_start:]
                    if str(row.get("request_id") or "") == request_id
                ]
                http_events = [
                    row
                    for row in node.metrics[metric_start:]
                    if row.get("event_type") == "step_http_response"
                    and str(row.get("request_id") or "") == request_id
                ]
                observation = {
                    "sequence_id": sequence["sequence_id"],
                    "frame_index": frame_index,
                    "timestamp": time.monotonic(),
                    "frame_seq": frame_seq,
                    "planned_frame_seq": int(frame["frame_seq"]),
                    "target_visible": bool(response.get("target_visible")),
                    "confidence": float(response.get("confidence", 0.0)),
                    "visible_in_view": str(response.get("visible_in_view") or ("front" if response.get("target_visible") else "none")),
                    "image_snapshot": snapshot,
                    "track": dict(response.get("track") or {}),
                    "step_latency_sec": response.get("step_latency_sec"),
                    "response": response,
                    "request_audit": audit,
                    "reset_ack_received": ack,
                    "image_ready": image_ready,
                    "action_triggered": bool(primitives),
                    "primitives": primitives,
                    "step_http_events": http_events,
                    "replayed_duplicate": False,
                    "image_path": str(image_path) if image_path else None,
                }
                observations.append(observation)
                previous = observation
    finally:
        node.destroy_node()
        rclpy.shutdown()

    (output / "tracking_observations.json").write_text(
        json.dumps(observations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "sequences": len(sequences),
        "expected_frames": sum(len(sequence["frames"]) for sequence in sequences),
        "observations": len(observations),
        "genuine_step_calls": sum(not row["replayed_duplicate"] for row in observations),
        "duplicate_replays": sum(bool(row["replayed_duplicate"]) for row in observations),
        "oracle_context_leakage": sum(int(row["request_audit"]["oracle_context_leakage"]) for row in observations),
        "failures": failures,
        "qualification_evidence": False,
        "locomotion_fidelity": "ideal_kinematic",
    }
    (output / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if not failures and len(observations) == summary["expected_frames"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
