from __future__ import annotations

import json
import math
import threading
import time
from io import BytesIO
from typing import Any

from .omninav_model_client_node import (
    _instruction_from_request,
    _pose,
    action_config_for_mode,
    orient_rgb_array,
    ros_image_to_rgb_array,
    waypoint_tensor_to_action,
)
from .schemas import (
    attach_timebase,
    clock_domain_from_node,
    deep_get,
    load_yaml_file,
    make_metric,
    new_id,
    node_ros_now_sec,
    stamp_to_sec,
)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except Exception:  # pragma: no cover - exercised on ROS hosts
    rclpy = None
    Node = object
    Image = None
    String = None


class OmniNavRemoteModelClientNode(Node):
    """Isaac-side ROS2 adapter for the DGX ZeroMQ inference service."""

    def __init__(self) -> None:
        if rclpy is None:
            raise RuntimeError("rclpy is required")
        super().__init__("omninav_remote_model_client")
        self.declare_parameter("config_file", "")
        self.config = load_yaml_file(self.get_parameter("config_file").value)
        cfg = self._cfg()
        from omninav_cosmos.transports.zmq_client import ZmqNavigationClient

        self.client = ZmqNavigationClient(
            str(cfg.get("endpoint", "tcp://10.100.100.128:8100")),
            timeout_ms=int(cfg.get("timeout_ms", 1000)),
        )
        self.action_pub = self.create_publisher(String, "/omninav/action_candidate_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/omninav/request_json", self.on_request, 10)
        self.create_subscription(String, "/benchmark/mode_json", self.on_mode, 10)
        self.frame_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.frames: dict[str, Any] = {}
        self.front_frame_time = 0.0
        self.active_episode_id = ""
        self.frame_id = 0
        self.mode_payload: dict[str, Any] = {}
        self.last_action: dict[str, Any] = {}
        self._subscribe_images()

    def _cfg(self) -> dict[str, Any]:
        return dict(self.config.get("omninav_remote", self.config.get("model_clients", {}).get("omninav_remote", {})))

    def _subscribe_images(self) -> None:
        cfg = self._cfg()
        topics = {
            "front": str(cfg.get("front_image_topic", "/camera/front/isaac_image")),
            "left": str(cfg.get("left_image_topic", "")),
            "right": str(cfg.get("right_image_topic", "")),
        }
        for view, topic in topics.items():
            if topic:
                self.create_subscription(Image, topic, lambda msg, view=view: self.on_image(view, msg), 2)

    def on_image(self, view: str, msg: Any) -> None:
        cfg = self._cfg()
        try:
            frame = ros_image_to_rgb_array(msg)
            frame = orient_rgb_array(
                frame,
                horizontal_flip=bool(cfg.get("horizontal_flip", False)),
                vertical_flip=bool(cfg.get("vertical_flip", False)),
            )
        except Exception as exc:
            self.publish_metric("omninav_remote_image_error", result="parse_error", view=view, error=repr(exc))
            return
        with self.frame_lock:
            self.frames[view] = frame
            if view == "front":
                self.front_frame_time = stamp_to_sec(getattr(getattr(msg, "header", None), "stamp", None)) or node_ros_now_sec(self)

    def on_mode(self, msg: Any) -> None:
        try:
            self.mode_payload = json.loads(msg.data)
        except Exception:
            self.mode_payload = {}

    def on_request(self, msg: Any) -> None:
        try:
            request_payload = json.loads(msg.data)
        except Exception as exc:
            self.publish_metric("omninav_remote_request_error", result="invalid_json", error=repr(exc))
            return
        if not self.request_lock.acquire(blocking=False):
            self.publish_fallback(request_payload, "client_busy")
            return
        threading.Thread(target=self._handle_request, args=(request_payload,), daemon=True).start()

    def _handle_request(self, request_payload: dict[str, Any]) -> None:
        try:
            action, details = self.infer_action(request_payload)
            self.publish_json(self.action_pub, action)
            self.publish_metric("omninav_remote_response", result="accepted", details=details)
        except Exception as exc:
            self.publish_fallback(request_payload, f"remote_error:{type(exc).__name__}", error=repr(exc))
        finally:
            self.request_lock.release()

    def infer_action(self, req: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        from omninav_cosmos.contracts import NavigationRequest

        cfg = self._cfg()
        with self.frame_lock:
            frames = dict(self.frames)
            frame_timestamp = self.front_frame_time
        if "front" not in frames:
            raise RuntimeError("no real front image received")
        image_age = node_ros_now_sec(self) - frame_timestamp
        if image_age > float(cfg.get("max_image_age_sec", 0.75)):
            raise TimeoutError(f"front image stale by {image_age:.3f}s")
        episode_id = str(req.get("episode_id") or "")
        if not episode_id:
            raise ValueError("request episode_id is required")
        reset = episode_id != self.active_episode_id
        if reset:
            self.active_episode_id = episode_id
            self.frame_id = 0
            self.client.reset_episode(episode_id)
        frame_id = self.frame_id
        self.frame_id += 1
        wire_request = NavigationRequest(
            episode_id=episode_id,
            frame_id=frame_id,
            timestamp=time.time(),
            instruction=_instruction_from_request(req),
            rgb_front=self._jpeg(frames["front"]),
            rgb_left=self._jpeg(frames["left"]) if "left" in frames else None,
            rgb_right=self._jpeg(frames["right"]) if "right" in frames else None,
            agent_pose=tuple(_pose(req.get("pose"))),
            last_action=self.last_action,
            collision_state=dict(req.get("collision_state") or {}),
            reset_episode=reset,
        )
        started = time.perf_counter()
        output = self.client.infer(wire_request)
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        network_latency_ms = max(0.0, roundtrip_ms - output.server_total_latency_ms)
        angles = [math.atan2(sin_value, cos_value) for sin_value, cos_value in output.heading_sin_cos]
        action_cfg = action_config_for_mode(self.config, self.mode_payload)
        action_cfg["waypoint_forward_axis"] = "x"
        action_core = waypoint_tensor_to_action(
            output.waypoints,
            arrive_pred=1.0 if output.arrive_or_stop else 0.0,
            recover_angle=angles,
            config=action_cfg,
        )
        if output.safe_stop_reason:
            action_core.update(
                primitive="stop",
                distance_m=0.0,
                yaw_deg=0.0,
                confidence=1.0,
                raw_text=output.safe_stop_reason,
                fallback_used=True,
                fallback_reason=output.safe_stop_reason,
            )
        timestamp_response = node_ros_now_sec(self)
        payload = {
            "request_id": str(req.get("request_id") or new_id("omni_remote")),
            "timestamp_request": float(req.get("timestamp", req.get("timestamp_request", timestamp_response))),
            "timestamp_response": timestamp_response,
            "frame_timestamp": frame_timestamp,
            "pose_at_snapshot": _pose(req.get("pose")),
            "source": "omninav_remote_model_client:isaac_zmq",
            "ttl_sec": float(cfg.get("ttl_sec", 0.75)),
            "model_variant": output.model_variant,
            "precision_mode": output.precision_mode,
            "model_latency_ms": output.model_latency_ms,
            "vision_latency_ms": output.vision_latency_ms,
            "network_latency_ms": network_latency_ms,
            "roundtrip_latency_ms": roundtrip_ms,
            "cache_hit": output.cache_hit,
            "action_head_trained": output.action_head_trained,
            "peak_memory_mib": output.peak_memory_mib,
            "coordinate_frame": output.coordinate_frame,
            **action_core,
        }
        payload = attach_timebase(
            payload,
            node=self,
            episode_id=episode_id,
            mission_id=str(req.get("mission_id") or ""),
            request_id=payload["request_id"],
            clock_domain=str(req.get("clock_domain") or clock_domain_from_node(self)),
            header_stamp=frame_timestamp,
            source_stamp=timestamp_response,
            created_ros_time=timestamp_response,
        )
        self.last_action = dict(action_core)
        return payload, {
            "episode_id": episode_id,
            "frame_id": frame_id,
            "image_age_sec": image_age,
            "model_variant": output.model_variant,
            "precision_mode": output.precision_mode,
            "model_latency_ms": output.model_latency_ms,
            "vision_latency_ms": output.vision_latency_ms,
            "network_latency_ms": network_latency_ms,
            "roundtrip_latency_ms": roundtrip_ms,
            "server_total_latency_ms": output.server_total_latency_ms,
            "cache_hit": output.cache_hit,
            "action_head_trained": output.action_head_trained,
            "peak_memory_mib": output.peak_memory_mib,
            "safe_stop_reason": output.safe_stop_reason,
        }

    def publish_fallback(self, req: dict[str, Any], reason: str, *, error: str | None = None) -> None:
        timestamp = node_ros_now_sec(self)
        payload = attach_timebase(
            {
                "request_id": str(req.get("request_id") or new_id("omni_remote_fallback")),
                "timestamp_request": float(req.get("timestamp", timestamp)),
                "timestamp_response": timestamp,
                "frame_timestamp": timestamp,
                "pose_at_snapshot": _pose(req.get("pose")),
                "primitive": "stop",
                "distance_m": 0.0,
                "yaw_deg": 0.0,
                "confidence": 1.0,
                "raw_text": reason,
                "source": "omninav_remote_model_client:fallback",
                "ttl_sec": 0.25,
                "fallback_used": True,
                "fallback_reason": reason,
            },
            node=self,
            episode_id=str(req.get("episode_id") or ""),
            request_id=str(req.get("request_id") or ""),
        )
        self.publish_json(self.action_pub, payload)
        self.publish_metric("omninav_remote_fallback", result=reason, error=error)

    def action_config(self) -> dict[str, Any]:
        return action_config_for_mode(self.config, self.mode_payload)

    def publish_metric(self, event_type: str, **details: Any) -> None:
        self.publish_json(self.metric_pub, make_metric(event_type, node=self, **details))

    @staticmethod
    def publish_json(publisher: Any, payload: dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        publisher.publish(msg)

    def _jpeg(self, frame: Any) -> bytes:
        from PIL import Image as PILImage

        buffer = BytesIO()
        PILImage.fromarray(frame).save(
            buffer,
            format="JPEG",
            quality=int(self._cfg().get("jpeg_quality", 90)),
            optimize=False,
        )
        return buffer.getvalue()

    def destroy_node(self) -> bool:
        self.client.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    if rclpy is None:
        raise RuntimeError("rclpy is required")
    rclpy.init(args=args)
    node = OmniNavRemoteModelClientNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
