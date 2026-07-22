#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_ROOT = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler"
BENCHMARK_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
for path in (SCHEDULER_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaac_vln_benchmark.perception_planning_suite import (
    build_tracking_sequences,
    evaluate_tracking_sequences,
    reset_payload_for_visual_case,
    tracking_gate,
    tracking_visual_case,
)
from isaac_vln_benchmark.sensor_perception import (
    build_target_observation,
    camera_intrinsics_from_physical,
    decode_mask_zlib,
    frame_stamps_synchronized,
    percentile,
    ppm_bytes,
)
from omninav_step_scheduler.sensor_only_planning import SpatialTargetTrackState


def safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def post_json(endpoint: str, body: dict[str, Any], timeout_sec: float) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("perception response must be an object")
    return payload, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description="Run actual Isaac RGB-D sequences through Grounded-SAM and tracker.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="http://10.100.100.128:8097/detect_segment")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--sequence-indices", default="")
    parser.add_argument("--timeout-sec", type=float, default=12.0)
    parser.add_argument("--horizontal-flip", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    class CaptureNode(Node):
        def __init__(self) -> None:
            super().__init__("rgbd_tracking_sequence_runner")
            self.rgb: dict[str, Any] | None = None
            self.rgb_count = 0
            self.depth: dict[str, Any] | None = None
            self.intrinsics: dict[str, Any] | None = None
            self.reset_acks: list[dict[str, Any]] = []
            self.scene_objects: dict[str, Any] = {}
            self.scene_objects_count = 0
            self.reset_pub = self.create_publisher(String, "/isaac/reset_episode", 10)
            self.create_subscription(Image, "/camera/front/isaac_image", self.on_rgb, 2)
            self.create_subscription(Image, "/camera/front/depth", self.on_depth, 2)
            self.create_subscription(CameraInfo, "/camera/front/camera_info", self.on_info, 2)
            self.create_subscription(String, "/isaac/reset_ack_json", self.on_ack, 20)
            self.create_subscription(String, "/isaac/objects", self.on_objects, 20)

        def on_rgb(self, msg: Image) -> None:
            self.rgb_count += 1
            self.rgb = image_dict(msg)

        def on_depth(self, msg: Image) -> None:
            self.depth = image_dict(msg)

        def on_info(self, msg: CameraInfo) -> None:
            self.intrinsics = {
                "width": int(msg.width),
                "height": int(msg.height),
                "fx": float(msg.k[0]),
                "fy": float(msg.k[4]),
                "cx": float(msg.k[2]),
                "cy": float(msg.k[5]),
                "k": [float(value) for value in msg.k],
                "frame_id": str(msg.header.frame_id),
            }

        def on_ack(self, msg: String) -> None:
            self.reset_acks.append(safe_json(msg.data))

        def on_objects(self, msg: String) -> None:
            self.scene_objects_count += 1
            self.scene_objects = safe_json(msg.data)

        def publish_reset(self, payload: dict[str, Any]) -> None:
            self.reset_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

        def spin_until(self, predicate, timeout_sec: float) -> bool:
            deadline = time.monotonic() + timeout_sec
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                if predicate():
                    return True
            return False

    sequences = build_tracking_sequences()
    if args.sequence_indices:
        indices = [int(value.strip()) for value in args.sequence_indices.split(",") if value.strip()]
        sequences = [sequences[index] for index in indices]
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]
    output = Path(args.output)
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    (output / "tracking_sequences.json").write_text(json.dumps(sequences, indent=2) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    latencies: list[float] = []
    temporal_requests = 0
    temporal_supported = 0

    rclpy.init()
    node = CaptureNode()
    try:
        node.spin_until(lambda: node.reset_pub.get_subscription_count() > 0, 8.0)
        for sequence in sequences:
            tracker = SpatialTargetTrackState(
                required_hits=max(2, int(sequence.get("required_hits", 2))),
                min_confidence=0.35,
                retain_confidence=0.25,
                max_misses=int(sequence.get("max_misses", 2)),
                max_age_sec=3.0,
                max_distance_jump_m=1.0,
                max_bearing_jump_deg=25.0,
            )
            tracker.reset(episode_id=str(sequence["episode_id"]), target_id=str(sequence["target"]))
            temporal_seed: dict[str, Any] | None = None
            previous_observation: dict[str, Any] | None = None
            was_confirmed = False
            was_lost = False
            for frame in sequence["frames"]:
                frame_index = int(frame["frame_index"])
                frame_seq = int(frame["frame_seq"])
                logical_time = float(frame_index)
                if previous_observation is not None and frame_seq == int(previous_observation["frame_seq"]):
                    track = tracker.update(dict(previous_observation), timestamp=logical_time)
                    model_called = False
                    response = {}
                    latency = 0.0
                    case = tracking_visual_case(sequence, frame)
                else:
                    case = tracking_visual_case(sequence, frame)
                    reset = reset_payload_for_visual_case(case, episode_id=str(sequence["episode_id"]))
                    ack_start = len(node.reset_acks)
                    old_rgb_stamp = float(node.rgb["stamp_sec"]) if node.rgb else -1.0
                    old_rgb_count = node.rgb_count
                    old_scene_objects_count = node.scene_objects_count
                    reset_ok = False
                    for _ in range(3):
                        node.publish_reset(reset)
                        reset_ok = node.spin_until(
                            lambda: any(
                                str(row.get("episode_id") or "") == str(sequence["episode_id"])
                                for row in node.reset_acks[ack_start:]
                            ),
                            1.5,
                        )
                        if reset_ok:
                            break
                    expect_occluder = bool(frame.get("occluded"))

                    def scene_state_matches() -> bool:
                        if node.scene_objects_count <= old_scene_objects_count:
                            return False
                        obstacles = node.scene_objects.get("obstacles")
                        if not isinstance(obstacles, list):
                            return False
                        has_occluder = any(
                            isinstance(row, dict) and str(row.get("id") or "") == "visual_occluder"
                            for row in obstacles
                        )
                        return has_occluder == expect_occluder

                    scene_ok = node.spin_until(scene_state_matches, 5.0)
                    scene_stamp = float(node.scene_objects.get("timestamp") or 0.0) if scene_ok else 0.0
                    scene_rgb_count = node.rgb_count
                    image_ok = node.spin_until(
                        lambda: bool(
                            node.rgb
                            and node.depth
                            and node.rgb_count >= max(old_rgb_count + 3, scene_rgb_count + 6)
                            and float(node.rgb["stamp_sec"]) > old_rgb_stamp
                            and float(node.rgb["stamp_sec"]) >= scene_stamp
                            and frame_stamps_synchronized(node.rgb["stamp_sec"], node.depth["stamp_sec"])
                        ),
                        5.0,
                    )
                    if not reset_ok or not scene_ok or not image_ok or not node.rgb or not node.depth:
                        failures.append(
                            {
                                "sequence_id": sequence["sequence_id"],
                                "frame_index": frame_index,
                                "reason": "reset_or_rgbd_capture_failed",
                                "reset_ok": reset_ok,
                                "scene_ok": scene_ok,
                                "image_ok": image_ok,
                            }
                        )
                        continue
                    rgb, depth = dict(node.rgb), dict(node.depth)
                    intrinsics = node.intrinsics or camera_intrinsics_from_physical(rgb["width"], rgb["height"])
                    image_payload = ppm_bytes(rgb["data"], rgb["width"], rgb["height"])
                    stem = f"{sequence['sequence_id']}_{frame_index:02d}"
                    (frame_dir / f"{stem}.ppm").write_bytes(image_payload)
                    (frame_dir / f"{stem}.depth16").write_bytes(depth["data"])
                    body: dict[str, Any] = {
                        "image_base64": base64.b64encode(image_payload).decode("ascii"),
                        "image_format": "ppm",
                        "target_query": str(sequence["target"]),
                        "box_threshold": 0.30,
                        "text_threshold": 0.25,
                        "max_detections": 4,
                    }
                    endpoint = args.endpoint
                    temporal_used = bool(
                        temporal_seed
                        and 0 < frame_seq - int(temporal_seed["frame_seq"]) <= 8
                    )
                    if temporal_used:
                        temporal_requests += 1
                        body.update(
                            {
                                "previous_image_base64": base64.b64encode(temporal_seed["image"]).decode("ascii"),
                                "previous_mask_zlib_base64": base64.b64encode(
                                    zlib.compress(temporal_seed["mask"])
                                ).decode("ascii"),
                                "episode_id": str(sequence["episode_id"]),
                                "frame_seq": frame_seq,
                            }
                        )
                        endpoint = args.endpoint.rstrip("/").removesuffix("/detect_segment") + "/detect_segment_temporal"
                    try:
                        response, latency = post_json(endpoint, body, args.timeout_sec)
                    except Exception as exc:
                        failures.append(
                            {
                                "sequence_id": sequence["sequence_id"],
                                "frame_index": frame_index,
                                "reason": "perception_request_failed",
                                "error": repr(exc),
                            }
                        )
                        continue
                    latencies.append(latency)
                    temporal_audit = response.get("temporal_propagation") or {}
                    temporal_supported += int(bool(temporal_audit.get("detector_supported")))
                    observation, _preview_mask = build_target_observation(
                        episode_id=str(sequence["episode_id"]),
                        target_id=str(sequence["target"]),
                        frame_seq=frame_seq,
                        source_stamp_sec=float(rgb["stamp_sec"]),
                        width=int(rgb["width"]),
                        height=int(rgb["height"]),
                        intrinsics=intrinsics,
                        depth=depth["data"],
                        response=response,
                        model_latency_sec=latency,
                        horizontal_flip=bool(args.horizontal_flip),
                    )
                    observation["observer_pose"] = list(case.get("camera_pose") or [0.0, 0.0, 0.0])
                    track = tracker.update(observation, timestamp=logical_time)
                    model_called = True
                    if str(track.get("update_result")) in {"hit", "confirmed", "reacquired"}:
                        try:
                            selected = int(track["selected_candidate_index"])
                            detection = response["detections"][selected]
                            selected_mask = decode_mask_zlib(
                                detection,
                                width=int(rgb["width"]),
                                height=int(rgb["height"]),
                            )
                            temporal_seed = {
                                "frame_seq": frame_seq,
                                "image": image_payload,
                                "mask": selected_mask,
                            }
                        except Exception:
                            pass
                    previous_observation = dict(observation)
                    (frame_dir / f"{stem}.json").write_text(
                        json.dumps(
                            {
                                "case": case,
                                "frame_seq": frame_seq,
                                "rgb_stamp_sec": rgb["stamp_sec"],
                                "depth_stamp_sec": depth["stamp_sec"],
                                "intrinsics": intrinsics,
                                "temporal_used": temporal_used,
                                "latency_sec": latency,
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                lost = bool(was_confirmed and not track["confirmed"])
                if lost:
                    was_lost = True
                reacquired = bool(was_lost and track["confirmed"])
                rows.append(
                    {
                        "sequence_id": sequence["sequence_id"],
                        "family": sequence["family"],
                        "episode_id": sequence["episode_id"],
                        "target": sequence["target"],
                        "frame_index": frame_index,
                        "frame_seq": frame_seq,
                        "visible_truth": bool(frame["visible"]),
                        "occluded": bool(frame["occluded"]),
                        "track_episode_id": track["episode_id"],
                        "track_target": track["target_id"],
                        "hits": track["hits"],
                        "misses": track["misses"],
                        "confirmed": track["confirmed"],
                        "track_visible": track["visible"],
                        "track_stale": not bool(track["confirmed"]),
                        "lost": lost,
                        "reacquired": reacquired,
                        "action_triggered": False,
                        "confidence": track["confidence"],
                        "candidate_source": track.get("candidate_source"),
                        "update_result": track.get("update_result"),
                        "model_called": model_called,
                        "latency_sec": latency,
                    }
                )
                observation_rows.append(
                    {
                        "sequence_id": sequence["sequence_id"],
                        "frame_index": frame_index,
                        "frame_seq": frame_seq,
                        "track": track,
                        "temporal_propagation": response.get("temporal_propagation") if response else None,
                    }
                )
                was_confirmed = bool(track["confirmed"])
                if reacquired:
                    was_lost = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

    metrics = evaluate_tracking_sequences(rows)
    metrics.update(
        {
            "sequences": len(sequences),
            "perception_calls": len(latencies),
            "temporal_requests": temporal_requests,
            "temporal_detector_supported": temporal_supported,
            "perception_latency_p95_sec": percentile(latencies, 0.95),
            "actual_isaac_viewport_frames": sum(int(row.get("model_called", False)) for row in rows),
            "model": "GroundingDINO-B + SAM2.1 Hiera Large video propagation",
        }
    )
    gate = tracking_gate(metrics)
    if failures:
        gate["pass"] = False
        gate["failures"].append(f"collection or inference failures: {len(failures)}")
    if (metrics.get("perception_latency_p95_sec") or 99.0) > 5.0:
        gate["pass"] = False
        gate["failures"].append("perception p95 latency > 5 sec")
    output.mkdir(parents=True, exist_ok=True)
    (output / "tracking_results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (output / "observation_trace.json").write_text(
        json.dumps(observation_rows, indent=2) + "\n", encoding="utf-8"
    )
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    (output / "tracking_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0 if gate["pass"] else 2


def image_dict(msg: Any) -> dict[str, Any]:
    return {
        "stamp_sec": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1.0e9,
        "frame_id": str(msg.header.frame_id),
        "width": int(msg.width),
        "height": int(msg.height),
        "encoding": str(msg.encoding),
        "is_bigendian": bool(msg.is_bigendian),
        "step": int(msg.step),
        "data": bytes(msg.data),
    }


if __name__ == "__main__":
    raise SystemExit(main())
