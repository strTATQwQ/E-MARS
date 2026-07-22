from __future__ import annotations

import base64
import json
import time
import urllib.request
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .sensor_perception import (
    build_target_observation,
    camera_intrinsics_from_physical,
    decode_mask_zlib,
    frame_stamps_synchronized,
    ppm_bytes,
    result_matches_active_context,
)

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
except Exception:  # pragma: no cover
    rclpy = None
    ExternalShutdownException = RuntimeError
    Node = object
    CameraInfo = None
    Image = None
    String = None


class GroundedSamPerceptionNode(Node):
    """Actual RGB-D target grounding client; it never consumes simulator truth."""

    def __init__(self) -> None:
        if rclpy is None:
            raise RuntimeError("rclpy is required to run GroundedSamPerceptionNode")
        super().__init__("grounded_sam_perception")
        self.declare_parameter("endpoint", "http://10.100.100.128:8097/detect_segment")
        self.declare_parameter("request_timeout_sec", 12.0)
        self.declare_parameter("max_image_age_sec", 1.0)
        self.declare_parameter("sync_tolerance_sec", 0.002)
        self.declare_parameter("request_rate_hz", 2.0)
        self.declare_parameter("box_threshold", 0.30)
        self.declare_parameter("text_threshold", 0.25)
        self.declare_parameter("horizontal_flip", False)
        self.declare_parameter("max_temporal_seed_gap_frames", 8)
        self.endpoint = str(self.get_parameter("endpoint").value)
        self.enabled = False
        self.episode_id = ""
        self.target_id = ""
        self.query = ""
        self.rgb: dict[str, Any] | None = None
        self.depth: dict[str, Any] | None = None
        self.intrinsics: dict[str, Any] | None = None
        self.last_submitted_stamp: float | None = None
        self.frame_seq = 0
        self.future: Future | None = None
        self.temporal_seed: dict[str, Any] | None = None
        self.pending_frames: dict[int, dict[str, Any]] = {}
        self.step_busy = False
        self.step_busy_since = 0.0
        self.horizontal_flip = bool(self.get_parameter("horizontal_flip").value)
        self.worker_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grounded_sam_http")

        self.observation_pub = self.create_publisher(String, "/perception/target_observation_json", 10)
        self.mask_pub = self.create_publisher(Image, "/perception/target_mask", 2)
        self.preview_mask_pub = self.create_publisher(Image, "/perception/target_mask_preview", 2)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(Image, "/camera/front/isaac_image", self.on_rgb, 2)
        self.create_subscription(Image, "/camera/front/depth", self.on_depth, 2)
        self.create_subscription(CameraInfo, "/camera/front/camera_info", self.on_camera_info, 2)
        self.create_subscription(String, "/perception/query_json", self.on_query, 10)
        self.create_subscription(String, "/perception/target_track_json", self.on_track, 10)
        self.create_subscription(String, "/benchmark/mode_json", self.on_mode, 10)
        self.create_subscription(String, "/episode/reset_lifecycle_json", self.on_reset, 10)
        self.create_subscription(String, "/step/request_json", self.on_step_request, 10)
        self.create_subscription(String, "/step/response_json", self.on_step_response, 10)
        self.create_subscription(String, "/step/semantic_stop_json", self.on_step_response, 10)
        self.create_subscription(String, "/step/route_choice_json", self.on_step_response, 10)
        rate = max(0.1, float(self.get_parameter("request_rate_hz").value))
        self.create_timer(1.0 / rate, self.tick)

    def on_mode(self, msg: String) -> None:
        payload = _json(msg.data)
        mode = payload.get("mode_config") if isinstance(payload.get("mode_config"), dict) else {}
        episode_id = str(payload.get("episode_id") or "")
        if episode_id and episode_id != self.episode_id:
            self.clear_episode(episode_id)
        self.enabled = bool(mode.get("sensor_only_planning", False))
        self.horizontal_flip = bool(
            mode.get("perception_horizontal_flip", mode.get("step_horizontal_flip", self.horizontal_flip))
        )
        query = str(mode.get("perception_target_query") or "").strip()
        target_id = str(mode.get("perception_target_id") or query).strip().lower()
        if query:
            self.query, self.target_id = query, target_id

    def on_reset(self, msg: String) -> None:
        payload = _json(msg.data)
        self.clear_episode(str(payload.get("episode_id") or self.episode_id))

    def clear_episode(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.rgb = None
        self.depth = None
        self.last_submitted_stamp = None
        self.frame_seq = 0
        self.temporal_seed = None
        self.pending_frames = {}

    def on_query(self, msg: String) -> None:
        payload = _json(msg.data)
        episode_id = str(payload.get("episode_id") or "")
        if self.episode_id and episode_id and episode_id != self.episode_id:
            self.publish_metric("perception_query_rejected", result="episode_mismatch")
            return
        query = str(payload.get("target_query") or payload.get("target_id") or "").strip()
        if query:
            next_target_id = str(payload.get("target_id") or query).strip().lower()
            if next_target_id != self.target_id:
                self.temporal_seed = None
            self.query = query
            self.target_id = next_target_id

    def on_step_request(self, msg: String) -> None:
        payload = _json(msg.data)
        if str(payload.get("episode_id") or "") in {"", self.episode_id}:
            self.step_busy = True
            self.step_busy_since = time.monotonic()

    def on_step_response(self, msg: String) -> None:
        payload = _json(msg.data)
        if str(payload.get("episode_id") or "") in {"", self.episode_id}:
            self.step_busy = False
            self.step_busy_since = 0.0

    def on_track(self, msg: String) -> None:
        track = _json(msg.data)
        if not result_matches_active_context(track, episode_id=self.episode_id, target_id=self.target_id):
            return
        try:
            frame_seq = int(track.get("frame_seq"))
            candidate_index = int(track.get("selected_candidate_index"))
        except (TypeError, ValueError):
            return
        pending = self.pending_frames.get(frame_seq)
        accepted = str(track.get("update_result") or "") in {"hit", "confirmed", "reacquired"}
        if not accepted or not pending:
            return
        mask = pending["masks"].get(candidate_index)
        if not mask or not any(mask):
            return
        self.temporal_seed = {
            "episode_id": self.episode_id,
            "target_id": self.target_id,
            "width": pending["width"],
            "height": pending["height"],
            "rgb": pending["rgb"],
            "mask": mask,
            "frame_seq": frame_seq,
        }
        self.publish_mask(
            self.mask_pub,
            mask,
            width=pending["width"],
            height=pending["height"],
            stamp_sec=pending["stamp_sec"],
            frame_id=pending["frame_id"],
        )
        self.pending_frames = {seq: value for seq, value in self.pending_frames.items() if seq > frame_seq}
        self.publish_metric(
            "temporal_seed_committed",
            result="accepted_track_candidate",
            frame_seq=frame_seq,
            selected_candidate_index=candidate_index,
            candidate_source=track.get("candidate_source"),
            mask_sha256=track.get("mask_sha256"),
        )

    def on_rgb(self, msg: Image) -> None:
        self.rgb = _image_dict(msg)

    def on_depth(self, msg: Image) -> None:
        self.depth = _image_dict(msg)

    def on_camera_info(self, msg: CameraInfo) -> None:
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

    def tick(self) -> None:
        if self.future is not None and self.future.done():
            future = self.future
            self.future = None
            try:
                result = future.result()
                self.publish_result(result)
            except Exception as exc:
                self.publish_metric("grounded_sam_request", result="error", error=repr(exc))
        if self.step_busy and time.monotonic() - self.step_busy_since > 15.0:
            self.step_busy = False
            self.step_busy_since = 0.0
            self.publish_metric("grounded_sam_serial_arbiter", result="step_watchdog_release")
        if not self.enabled or self.future is not None or not self.query or self.step_busy:
            return
        if not self.rgb or not self.depth:
            return
        rgb_stamp = float(self.rgb["stamp_sec"])
        depth_stamp = float(self.depth["stamp_sec"])
        if not frame_stamps_synchronized(
            rgb_stamp,
            depth_stamp,
            tolerance_sec=float(self.get_parameter("sync_tolerance_sec").value),
        ):
            self.publish_metric(
                "rgb_depth_sync_rejected",
                result="timestamp_mismatch",
                rgb_stamp_sec=rgb_stamp,
                depth_stamp_sec=depth_stamp,
            )
            return
        if self.last_submitted_stamp is not None and rgb_stamp <= self.last_submitted_stamp:
            return
        age = max(0.0, time.time() - rgb_stamp)
        if age > float(self.get_parameter("max_image_age_sec").value):
            self.publish_metric("perception_frame_rejected", result="stale_image", age_sec=age)
            return
        if self.rgb["width"] != self.depth["width"] or self.rgb["height"] != self.depth["height"]:
            self.publish_metric("rgb_depth_sync_rejected", result="dimension_mismatch")
            return
        if self.intrinsics is None:
            self.intrinsics = camera_intrinsics_from_physical(self.rgb["width"], self.rgb["height"])
        self.last_submitted_stamp = rgb_stamp
        self.frame_seq += 1
        snapshot = {
            "episode_id": self.episode_id,
            "target_id": self.target_id,
            "query": self.query,
            "frame_seq": self.frame_seq,
            "stamp_sec": rgb_stamp,
            "rgb": dict(self.rgb),
            "depth": dict(self.depth),
            "intrinsics": dict(self.intrinsics),
        }
        self.future = self.worker_pool.submit(self.request_model, snapshot)

    def request_model(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        rgb = snapshot["rgb"]
        encoded = base64.b64encode(ppm_bytes(rgb["data"], rgb["width"], rgb["height"])).decode("ascii")
        body = {
            "image_base64": encoded,
            "image_format": "ppm",
            "target_query": snapshot["query"],
            "box_threshold": float(self.get_parameter("box_threshold").value),
            "text_threshold": float(self.get_parameter("text_threshold").value),
            "max_detections": 4,
        }
        endpoint = self.endpoint
        temporal_seed = self.temporal_seed
        if (
            temporal_seed
            and temporal_seed.get("episode_id") == snapshot["episode_id"]
            and temporal_seed.get("target_id") == snapshot["target_id"]
            and temporal_seed.get("width") == rgb["width"]
            and temporal_seed.get("height") == rgb["height"]
            and 0 < snapshot["frame_seq"] - int(temporal_seed.get("frame_seq", -1))
            <= int(self.get_parameter("max_temporal_seed_gap_frames").value)
        ):
            body.update(
                {
                    "previous_image_base64": base64.b64encode(
                        ppm_bytes(temporal_seed["rgb"], temporal_seed["width"], temporal_seed["height"])
                    ).decode("ascii"),
                    "previous_mask_zlib_base64": base64.b64encode(
                        zlib.compress(temporal_seed["mask"])
                    ).decode("ascii"),
                    "episode_id": snapshot["episode_id"],
                    "frame_seq": snapshot["frame_seq"],
                }
            )
            endpoint = self.endpoint.rstrip("/").removesuffix("/detect_segment") + "/detect_segment_temporal"
        started = time.perf_counter()
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=float(self.get_parameter("request_timeout_sec").value),
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("grounded SAM response must be an object")
        return snapshot | {
            "response": payload,
            "latency_sec": time.perf_counter() - started,
            "temporal_requested": "previous_mask_zlib_base64" in body,
        }

    def publish_result(self, result: dict[str, Any]) -> None:
        if not result_matches_active_context(
            result,
            episode_id=self.episode_id,
            target_id=self.target_id,
        ):
            self.publish_metric(
                "grounded_sam_result_discarded",
                result="old_response_after_reset",
                response_episode_id=result.get("episode_id"),
                response_target_id=result.get("target_id"),
            )
            return
        rgb, depth = result["rgb"], result["depth"]
        observation, mask = build_target_observation(
            episode_id=result["episode_id"],
            target_id=result["target_id"],
            frame_seq=result["frame_seq"],
            source_stamp_sec=result["stamp_sec"],
            width=rgb["width"],
            height=rgb["height"],
            intrinsics=result["intrinsics"],
            depth=depth["data"],
            response=result["response"],
            model_latency_sec=result["latency_sec"],
            horizontal_flip=self.horizontal_flip,
        )
        candidate_masks: dict[int, bytes] = {}
        for index, detection in enumerate(result["response"].get("detections") or []):
            if not isinstance(detection, dict):
                continue
            try:
                candidate_masks[index] = decode_mask_zlib(
                    detection,
                    width=int(rgb["width"]),
                    height=int(rgb["height"]),
                )
            except Exception:
                continue
        self.pending_frames[int(result["frame_seq"])] = {
            "width": int(rgb["width"]),
            "height": int(rgb["height"]),
            "rgb": bytes(rgb["data"]),
            "masks": candidate_masks,
            "stamp_sec": float(result["stamp_sec"]),
            "frame_id": str(rgb["frame_id"]),
        }
        for old_seq in sorted(self.pending_frames)[:-12]:
            self.pending_frames.pop(old_seq, None)
        self.publish_json(self.observation_pub, observation)
        self.publish_mask(
            self.preview_mask_pub,
            mask,
            width=int(rgb["width"]),
            height=int(rgb["height"]),
            stamp_sec=float(result["stamp_sec"]),
            frame_id=str(rgb["frame_id"]),
        )
        self.publish_metric(
            "grounded_sam_request",
            result="accepted",
            episode_id=self.episode_id,
            frame_seq=result["frame_seq"],
            latency_sec=result["latency_sec"],
            visible=observation.get("visible"),
            confidence=observation.get("confidence"),
            temporal_requested=bool(result.get("temporal_requested")),
            temporal_propagation=result["response"].get("temporal_propagation"),
        )

    def publish_metric(self, event_type: str, **fields: Any) -> None:
        self.publish_json(
            self.metric_pub,
            {
                "event_type": event_type,
                "model": "groundingdino_b_sam2_1_hiera_large",
                "episode_id": self.episode_id,
                "timestamp": self.get_clock().now().nanoseconds / 1.0e9,
                **fields,
            },
        )

    @staticmethod
    def publish_json(pub, payload: dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)

    @staticmethod
    def publish_mask(pub, mask: bytes, *, width: int, height: int, stamp_sec: float, frame_id: str) -> None:
        mask_msg = Image()
        mask_msg.header.stamp.sec = int(stamp_sec)
        mask_msg.header.stamp.nanosec = int((stamp_sec % 1.0) * 1.0e9)
        mask_msg.header.frame_id = str(frame_id)
        mask_msg.width = int(width)
        mask_msg.height = int(height)
        mask_msg.encoding = "mono8"
        mask_msg.is_bigendian = 0
        mask_msg.step = int(width)
        mask_msg.data = bytes(255 if value else 0 for value in mask)
        pub.publish(mask_msg)

    def destroy_node(self):
        self.worker_pool.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def _stamp_sec(msg: Image | CameraInfo) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1.0e9


def _image_dict(msg: Image) -> dict[str, Any]:
    return {
        "stamp_sec": _stamp_sec(msg),
        "frame_id": str(msg.header.frame_id),
        "width": int(msg.width),
        "height": int(msg.height),
        "encoding": str(msg.encoding),
        "is_bigendian": bool(msg.is_bigendian),
        "step": int(msg.step),
        "data": bytes(msg.data),
    }


def _json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GroundedSamPerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
