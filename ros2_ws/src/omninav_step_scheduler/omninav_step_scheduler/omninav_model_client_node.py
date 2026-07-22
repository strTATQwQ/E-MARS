from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .schemas import attach_timebase, clock_domain_from_node, deep_get, load_yaml_file, make_metric, new_id, node_ros_now_sec, now, stamp_to_sec

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


def waypoint_tensor_to_action(
    waypoints: Any,
    arrive_pred: Any = None,
    recover_angle: Any = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    points = _collect_waypoints(_to_builtin(waypoints))
    min_norm = float(cfg.get("min_waypoint_norm_m", 0.03))
    stop_distance = float(cfg.get("stop_distance_m", 0.04))
    turn_threshold_deg = float(cfg.get("turn_threshold_deg", 12.0))
    hard_turn_threshold_deg = float(cfg.get("hard_turn_threshold_deg", 28.0))
    forward_yaw_threshold_deg = float(cfg.get("forward_yaw_threshold_deg", turn_threshold_deg))
    forward_min = float(cfg.get("forward_min_m", cfg.get("forward_min_x_m", 0.05)))
    forward_axis = str(cfg.get("waypoint_forward_axis", cfg.get("forward_axis", "x")) or "x").lower()
    max_forward = float(cfg.get("max_forward_horizon_m", 0.75))
    arrive_threshold = float(cfg.get("stop_arrive_threshold", 0.5))

    arrive_prob = _probability(_first_number(_to_builtin(arrive_pred)))
    fallback_used = False
    fallback_reason = ""
    point = next((p for p in points if math.hypot(p[0], p[1]) >= min_norm), points[0] if points else (0.0, 0.0))
    raw_x, raw_y = point
    if forward_axis in {"y", "second", "dy", "forward_y"}:
        forward_m, lateral_m = raw_y, raw_x
        axis_label = "y"
    else:
        forward_m, lateral_m = raw_x, raw_y
        axis_label = "x"
    distance = math.hypot(forward_m, lateral_m)
    yaw_deg = math.degrees(math.atan2(lateral_m, forward_m)) if distance > 1e-6 else _recover_angle_deg(recover_angle)
    yaw_abs = abs(yaw_deg)

    if not points:
        primitive = "stop"
        distance_out = 0.0
        confidence = 0.20
        parser_reason = "no_waypoint_stop"
        raw_action = "stop:no_waypoint"
        fallback_used = True
        fallback_reason = "no_waypoint"
    elif arrive_prob is not None and arrive_prob >= arrive_threshold:
        primitive = "stop"
        distance_out = 0.0
        confidence = max(0.55, min(0.95, arrive_prob))
        parser_reason = "arrive_probability_stop"
        raw_action = "stop:arrive"
    elif distance <= stop_distance:
        primitive = "stop"
        distance_out = 0.0
        confidence = 0.45 if points else 0.20
        parser_reason = "waypoint_below_stop_distance"
        raw_action = "stop:near_zero_waypoint"
    elif forward_m >= forward_min and yaw_abs <= forward_yaw_threshold_deg:
        primitive = "move_forward"
        distance_out = min(distance, max_forward)
        confidence = 0.80
        parser_reason = "forward_waypoint_low_yaw"
        raw_action = "move_forward"
    elif yaw_abs >= hard_turn_threshold_deg or (yaw_abs >= turn_threshold_deg and forward_m <= 0.08):
        primitive = "turn_left" if yaw_deg > 0.0 else "turn_right"
        distance_out = 0.0
        confidence = 0.72
        parser_reason = "yaw_exceeds_turn_threshold"
        raw_action = primitive
    else:
        primitive = "follow_waypoint"
        distance_out = min(distance, max_forward)
        confidence = 0.78
        parser_reason = "waypoint_tracking"
        raw_action = "follow_waypoint"

    if arrive_prob is not None and primitive != "stop":
        confidence = max(0.05, min(0.95, confidence * (1.0 - 0.35 * arrive_prob)))

    return {
        "primitive": primitive,
        "distance_m": float(distance_out),
        "yaw_deg": float(max(-90.0, min(90.0, yaw_deg))),
        "confidence": float(max(0.0, min(1.0, confidence))),
        "raw_text": f"wp=({raw_x:.3f},{raw_y:.3f}) arrive={arrive_prob if arrive_prob is not None else 'n/a'} axis={axis_label}",
        "raw_waypoint": [float(raw_x), float(raw_y)],
        "local_forward_m": float(forward_m),
        "local_lateral_m": float(lateral_m),
        "waypoint_forward_axis": axis_label,
        "raw_action": raw_action,
        "parser_reason": parser_reason,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }


def ros_image_to_rgb_array(msg: Any):
    import numpy as np

    encoding = str(getattr(msg, "encoding", "rgb8")).lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in {"rgb8", "bgr8"}:
        channels = 3
    elif encoding in {"rgba8", "bgra8"}:
        channels = 4
    elif encoding in {"mono8", "8uc1"}:
        channels = 1
    else:
        raise ValueError(f"unsupported image encoding: {msg.encoding!r}")

    rows = data.reshape((height, step))[:, : width * channels]
    image = rows.reshape((height, width, channels))
    if channels == 1:
        image = np.repeat(image, 3, axis=2)
    elif encoding in {"bgr8", "bgra8"}:
        image = image[..., :3][..., ::-1]
    else:
        image = image[..., :3]
    return image.copy()


def orient_rgb_array(frame: Any, *, horizontal_flip: bool = False, vertical_flip: bool = False):
    """Apply the configured camera-contract transform without changing the ROS image."""
    oriented = frame
    if horizontal_flip:
        oriented = oriented[:, ::-1, :]
    if vertical_flip:
        oriented = oriented[::-1, :, :]
    return oriented.copy()


def image_orientation_for_mode(config: dict[str, Any], mode_payload: dict[str, Any]) -> tuple[bool, bool]:
    mode = mode_payload.get("mode_config") if isinstance(mode_payload.get("mode_config"), dict) else {}
    horizontal = mode.get(
        "omninav_horizontal_flip",
        deep_get(config, "model_clients.omninav.horizontal_flip", False),
    )
    vertical = mode.get(
        "omninav_vertical_flip",
        deep_get(config, "model_clients.omninav.vertical_flip", False),
    )
    return bool(horizontal), bool(vertical)


def action_config_for_mode(config: dict[str, Any], mode_payload: dict[str, Any]) -> dict[str, Any]:
    """Build parser thresholds with explicit benchmark-mode overrides."""
    mode = mode_payload.get("mode_config") if isinstance(mode_payload.get("mode_config"), dict) else {}

    def value(mode_key: str, config_path: str, default: Any) -> Any:
        return mode.get(mode_key, deep_get(config, config_path, default))

    return {
        "min_waypoint_norm_m": value("omninav_min_waypoint_norm_m", "model_clients.omninav.min_waypoint_norm_m", 0.03),
        "stop_distance_m": value("omninav_stop_distance_m", "model_clients.omninav.stop_distance_m", 0.04),
        "turn_threshold_deg": value("omninav_turn_threshold_deg", "model_clients.omninav.turn_threshold_deg", 12.0),
        "hard_turn_threshold_deg": value("omninav_hard_turn_threshold_deg", "model_clients.omninav.hard_turn_threshold_deg", 28.0),
        "max_forward_horizon_m": deep_get(config, "primitive.max_forward_horizon_m", 0.75),
        "stop_arrive_threshold": value("omninav_stop_arrive_threshold", "model_clients.omninav.stop_arrive_threshold", 0.5),
        "forward_yaw_threshold_deg": value(
            "omninav_forward_yaw_threshold_deg",
            "model_clients.omninav.forward_yaw_threshold_deg",
            deep_get(config, "model_clients.omninav.turn_threshold_deg", 12.0),
        ),
        "forward_min_x_m": value("omninav_forward_min_x_m", "model_clients.omninav.forward_min_x_m", 0.05),
        "forward_min_m": value(
            "omninav_forward_min_m",
            "model_clients.omninav.forward_min_m",
            deep_get(config, "model_clients.omninav.forward_min_x_m", 0.05),
        ),
        "waypoint_forward_axis": value("omninav_waypoint_forward_axis", "model_clients.omninav.waypoint_forward_axis", "x"),
    }


class OmniNavModelClientNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run OmniNavModelClientNode")
        super().__init__("omninav_model_client")
        self.declare_parameter("config_file", "")
        self.config = load_yaml_file(self.get_parameter("config_file").value)
        self.benchmark_mode_payload: dict[str, Any] = {}
        self.action_pub = self.create_publisher(String, "/omninav/action_candidate_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/omninav/request_json", self.on_request, 10)
        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)

        self.frame_lock = threading.Lock()
        self.front_frame = None
        self.left_frame = None
        self.right_frame = None
        self.front_frame_time = 0.0
        self.history_frames: list[Any] = []
        self.history_limit = int(deep_get(self.config, "model_clients.omninav.history_limit", 32))
        self.horizontal_flip, self.vertical_flip = image_orientation_for_mode(self.config, {})

        self.model_lock = threading.Lock()
        self.pending_request_lock = threading.Lock()
        self.pending_request: dict[str, Any] | None = None
        self.active_episode_id = ""
        self.model_ready = False
        self.model_error: str | None = None
        self.helper = None
        self.processor = None
        self.model = None
        self.omni_cfg = None
        self.predict_scale = float(deep_get(self.config, "model_clients.omninav.predict_scale", 1.0))
        self.step_index = 0

        self._subscribe_images()
        if bool(deep_get(self.config, "model_clients.omninav.autoload", True)):
            self.load_thread = threading.Thread(target=self._load_model_thread, daemon=True)
            self.load_thread.start()

    def _subscribe_images(self):
        if Image is None:
            return
        topics = {
            "front": str(deep_get(self.config, "model_clients.omninav.front_image_topic", "/camera/front/image")),
            "left": str(deep_get(self.config, "model_clients.omninav.left_image_topic", "")),
            "right": str(deep_get(self.config, "model_clients.omninav.right_image_topic", "")),
        }
        if topics["front"]:
            self.create_subscription(Image, topics["front"], lambda msg: self.on_image("front", msg), 2)
        if topics["left"]:
            self.create_subscription(Image, topics["left"], lambda msg: self.on_image("left", msg), 2)
        if topics["right"]:
            self.create_subscription(Image, topics["right"], lambda msg: self.on_image("right", msg), 2)

    def on_image(self, view: str, msg):
        try:
            frame = ros_image_to_rgb_array(msg)
            frame = orient_rgb_array(
                frame,
                horizontal_flip=self.horizontal_flip,
                vertical_flip=self.vertical_flip,
            )
        except Exception as exc:
            self.publish_metric("omninav_image_error", view=view, result="parse_error", error=repr(exc))
            return
        with self.frame_lock:
            setattr(self, f"{view}_frame", frame)
            if view == "front":
                self.front_frame_time = stamp_to_sec(getattr(getattr(msg, "header", None), "stamp", None)) or node_ros_now_sec(self)

    def on_benchmark_mode(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {}
        episode_id = str(payload.get("episode_id") or "")
        if episode_id and episode_id != self.active_episode_id:
            self.active_episode_id = episode_id
            with self.pending_request_lock:
                self.pending_request = None
        self.benchmark_mode_payload = payload
        self.horizontal_flip, self.vertical_flip = image_orientation_for_mode(self.config, payload)
        self.safe_publish_metric(
            "omninav_image_orientation",
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            mode=payload.get("mode"),
            result="configured",
        )

    def _load_model_thread(self):
        try:
            self.load_model()
        except BaseException as exc:  # pragma: no cover - protects long-running ROS thread
            self.model_error = repr(exc)
            self.model_ready = False
            self.get_logger().error(f"OmniNav model load thread crashed: {self.model_error}")
            self.safe_publish_metric("omninav_model_load_error", result="error", error=self.model_error)

    def load_model(self):
        repo = Path(str(deep_get(self.config, "model_clients.omninav.repo", os.environ.get("OMNINAV_REPO", "/home/railgun/ai-stack/src/OmniNav"))))
        model_path = str(deep_get(self.config, "model_clients.omninav.model_path", os.environ.get("OMNINAV_MODEL_PATH", "/home/railgun/ai-stack/models/OmniNav")))
        helper_path = Path(
            str(
                deep_get(
                    self.config,
                    "model_clients.omninav.helper_module_path",
                    os.environ.get("OMNINAV_HELPER_PATH", "/home/railgun/dgx-unitree/remote_bench/omninav_token_sweep.py"),
                )
            )
        )
        attn = str(deep_get(self.config, "model_clients.omninav.attn_implementation", "flash_attention_2"))
        cfg_name = str(deep_get(self.config, "model_clients.omninav.config_name", "F_front_semantic_history_384tok"))
        t0 = time.perf_counter()
        self.get_logger().info(f"Loading OmniNav model config={cfg_name} model_path={model_path}")
        self.safe_publish_metric("omninav_model_load_start", repo=str(repo), model_path=model_path, helper_path=str(helper_path), config_name=cfg_name)
        try:
            add_extra_python_paths(deep_get(self.config, "model_clients.omninav.extra_python_paths", []))
            patch_numpy_legacy_aliases()
            helper = load_helper_module(helper_path)
            helper.add_repo_paths(repo)
            _, processor, model = helper.load_model(model_path, attn)
            configs = {cfg.name: cfg for cfg in helper.CONFIGS}
            if cfg_name not in configs:
                raise KeyError(f"OmniNav config {cfg_name!r} not found in helper CONFIGS")
            self.predict_scale = float(deep_get(self.config, "model_clients.omninav.predict_scale", _load_predict_scale(repo, self.predict_scale)))
            self.helper = helper
            self.processor = processor
            self.model = model
            self.omni_cfg = configs[cfg_name]
            self.model_ready = True
            self.model_error = None
            self.get_logger().info(f"OmniNav model loaded in {time.perf_counter() - t0:.3f}s")
            self.safe_publish_metric("omninav_model_loaded", latency_s=time.perf_counter() - t0, predict_scale=self.predict_scale, result="ready")
        except Exception as exc:
            self.model_error = repr(exc)
            self.model_ready = False
            self.get_logger().error(f"OmniNav model load failed: {self.model_error}")
            self.safe_publish_metric("omninav_model_load_error", latency_s=time.perf_counter() - t0, result="error", error=self.model_error)

    def on_request(self, msg):
        req = _safe_json(msg.data)
        if not self.model_ready:
            reason = "model_error" if self.model_error else "model_loading"
            self.publish_fallback_action(req, reason, confidence=0.0, error=self.model_error)
            return
        if not self.model_lock.acquire(blocking=False):
            with self.pending_request_lock:
                replaced_request_id = str((self.pending_request or {}).get("request_id") or "")
                self.pending_request = req
            self.publish_metric(
                "omninav_model_request_queued",
                request_id=req.get("request_id"),
                replaced_request_id=replaced_request_id,
                result="latest_request_queued",
            )
            return
        threading.Thread(target=self.handle_request, args=(req,), daemon=True).start()

    def handle_request(self, req: dict[str, Any]):
        t0 = time.perf_counter()
        try:
            action, details = self.infer_action(req)
            result = "accepted"
            error = None
        except Exception as exc:
            action = self.make_fallback_payload(req, "inference_error", confidence=0.0, error=repr(exc))
            details = {}
            result = "fallback"
            error = repr(exc)
        finally:
            self.model_lock.release()

        self.publish_json(self.action_pub, action)
        self.publish_metric(
            "omninav_model_response",
            request_id=action["request_id"],
            latency_s=time.perf_counter() - t0,
            result=result,
            primitive=action["primitive"],
            confidence=action["confidence"],
            details=details,
            error=error,
        )
        self.dispatch_pending_request()

    def dispatch_pending_request(self) -> None:
        with self.pending_request_lock:
            if self.pending_request is None or not self.model_lock.acquire(blocking=False):
                return
            req = self.pending_request
            self.pending_request = None
        threading.Thread(target=self.handle_request, args=(req,), daemon=True).start()

    def infer_action(self, req: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        helper = self.helper
        processor = self.processor
        model = self.model
        cfg = self.omni_cfg
        if helper is None or processor is None or model is None or cfg is None:
            raise RuntimeError("OmniNav model is not loaded")

        frames, image_source, frame_timestamp = self.current_frames()
        instruction = _instruction_from_request(req)
        step = self.step_index
        self.step_index += 1

        original_make_rgb = helper.make_rgb
        helper.make_rgb = lambda width, height, step_idx, view: resize_rgb(frames.get(view) if frames.get(view) is not None else frames["front"], width, height)
        try:
            sync = getattr(helper, "sync", lambda: None)
            prep_t0 = time.perf_counter()
            inputs, prep_timing, views, history_count = helper.prepare_inputs(processor, cfg, step, list(self.history_frames), instruction)
            prep_s = time.perf_counter() - prep_t0
        finally:
            helper.make_rgb = original_make_rgb

        import torch

        sync()
        to_cuda_t0 = time.perf_counter()
        inputs = inputs.to("cuda")
        sync()
        forward_t0 = time.perf_counter()
        with torch.inference_mode():
            wp_pred, arrive_pred, sin_angle, cos_angle = model.forward(
                **inputs,
                action_former=True,
                gt_waypoints=0,
                train=False,
                train_branch=["continue"],
            )
        sync()
        post_t0 = time.perf_counter()
        wp_pred = wp_pred * self.predict_scale
        recover_angle = torch.atan2(sin_angle, cos_angle)
        wp_np = wp_pred.detach().float().cpu().numpy()
        arrive_np = arrive_pred.detach().float().cpu().numpy()
        angle_np = recover_angle.detach().float().cpu().numpy()
        post_s = time.perf_counter() - post_t0

        action_core = waypoint_tensor_to_action(wp_np, arrive_np, angle_np, config=self.action_config())
        ts_response = node_ros_now_sec(self)
        payload = {
            "request_id": str(req.get("request_id") or new_id("omni")),
            "timestamp_request": float(req.get("timestamp", req.get("timestamp_request", now()))),
            "timestamp_response": ts_response,
            "frame_timestamp": frame_timestamp,
            "pose_at_snapshot": _pose(req.get("pose")),
            "source": f"omninav_model_client:{image_source}",
            "ttl_sec": float(deep_get(self.config, "model_clients.omninav.ttl_sec", deep_get(self.config, "omninav.max_action_age_sec", 0.5))),
            **action_core,
        }
        payload = attach_timebase(
            payload,
            node=self,
            episode_id=str(req.get("episode_id") or ""),
            mission_id=str(req.get("mission_id") or ""),
            request_id=str(payload["request_id"]),
            clock_domain=str(req.get("clock_domain") or clock_domain_from_node(self)),
            header_stamp=frame_timestamp,
            source_stamp=ts_response,
            created_ros_time=ts_response,
        )
        with self.frame_lock:
            self.history_frames.append(frames["front"])
            self.history_frames = self.history_frames[-self.history_limit :]

        details = {
            "image_source": image_source,
            "views": list(views),
            "history_images": history_count,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "pixel_values_shape": list(inputs["pixel_values"].shape) if "pixel_values" in inputs else None,
            "waypoint_shape": list(wp_np.shape),
            "raw_waypoint": action_core.get("raw_waypoint"),
            "raw_action": action_core.get("raw_action"),
            "parsed_primitive": action_core.get("primitive"),
            "local_forward_m": action_core.get("local_forward_m"),
            "local_lateral_m": action_core.get("local_lateral_m"),
            "waypoint_forward_axis": action_core.get("waypoint_forward_axis"),
            "distance_m": action_core.get("distance_m"),
            "yaw_deg": action_core.get("yaw_deg"),
            "parser_reason": action_core.get("parser_reason"),
            "fallback_used": action_core.get("fallback_used", False),
            "fallback_reason": action_core.get("fallback_reason", ""),
            "arrive_shape": list(arrive_np.shape),
            "prep": prep_timing,
            "prep_total_s": prep_s,
            "to_cuda_s": forward_t0 - to_cuda_t0,
            "forward_s": post_t0 - forward_t0,
            "postprocess_s": post_s,
        }
        return payload, details

    def current_frames(self) -> tuple[dict[str, Any], str, float]:
        with self.frame_lock:
            front = self.front_frame
            left = self.left_frame
            right = self.right_frame
            frame_timestamp = self.front_frame_time or now()
        if front is None:
            if not bool(deep_get(self.config, "model_clients.omninav.synthetic_fallback", True)):
                raise RuntimeError("no front image received and synthetic_fallback is false")
            width = int(deep_get(self.config, "model_clients.omninav.synthetic_width", 640))
            height = int(deep_get(self.config, "model_clients.omninav.synthetic_height", 569))
            front = synthetic_rgb(width, height, self.step_index, "front")
            left = left if left is not None else synthetic_rgb(width, height, self.step_index, "left")
            right = right if right is not None else synthetic_rgb(width, height, self.step_index, "right")
            return {"front": front, "left": left, "right": right}, "synthetic_fallback", now()
        return {"front": front, "left": left if left is not None else front, "right": right if right is not None else front}, "ros_image", frame_timestamp

    def action_config(self) -> dict[str, Any]:
        return action_config_for_mode(self.config, self.benchmark_mode_payload)

    def publish_fallback_action(self, req: dict[str, Any], reason: str, *, confidence: float, error: str | None = None):
        payload = self.make_fallback_payload(req, reason, confidence=confidence, error=error)
        self.publish_json(self.action_pub, payload)
        self.publish_metric("omninav_model_fallback", request_id=payload["request_id"], result=reason, confidence=confidence, error=error)

    def make_fallback_payload(self, req: dict[str, Any], reason: str, *, confidence: float, error: str | None = None) -> dict[str, Any]:
        ts_req = float(req.get("timestamp", req.get("timestamp_request", node_ros_now_sec(self))))
        ts_response = node_ros_now_sec(self)
        payload = {
            "request_id": str(req.get("request_id") or new_id("omni")),
            "timestamp_request": ts_req,
            "timestamp_response": ts_response,
            "frame_timestamp": ts_req,
            "pose_at_snapshot": _pose(req.get("pose")),
            "primitive": "stop",
            "distance_m": 0.0,
            "yaw_deg": 0.0,
            "confidence": confidence,
            "raw_text": reason if error is None else f"{reason}: {error}",
            "source": "omninav_model_client:fallback",
            "ttl_sec": 0.25,
        }
        return attach_timebase(
            payload,
            node=self,
            episode_id=str(req.get("episode_id") or ""),
            mission_id=str(req.get("mission_id") or ""),
            request_id=str(payload["request_id"]),
            clock_domain=str(req.get("clock_domain") or clock_domain_from_node(self)),
            header_stamp=ts_req,
            source_stamp=ts_response,
            created_ros_time=ts_response,
        )

    def publish_metric(self, event_type: str, **kwargs):
        self.publish_json(self.metric_pub, make_metric(event_type, model="omninav_model", **kwargs))

    def safe_publish_metric(self, event_type: str, **kwargs):
        try:
            self.publish_metric(event_type, **kwargs)
        except Exception as exc:  # pragma: no cover - metric publishing must not stop model loading
            self.get_logger().warning(f"Failed to publish metric {event_type}: {exc!r}")

    @staticmethod
    def publish_json(pub, payload):
        msg = String()
        msg.data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)


def load_helper_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    module_name = f"omninav_token_sweep_runtime_{abs(hash(str(path))) & 0xffffffff:x}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def add_extra_python_paths(paths: Any) -> None:
    if isinstance(paths, str):
        paths = [item.strip() for item in paths.split(":") if item.strip()]
    if not isinstance(paths, (list, tuple)):
        return
    for raw_path in reversed(paths):
        path = str(raw_path)
        if path and path not in sys.path:
            sys.path.insert(0, path)


def patch_numpy_legacy_aliases() -> None:
    try:
        import numpy as np
    except Exception:
        return
    if not hasattr(np, "long"):
        np.long = np.int_
    if not hasattr(np, "ulong"):
        np.ulong = np.uint


def resize_rgb(frame: Any, width: int, height: int):
    import numpy as np
    from PIL import Image as PilImage

    arr = np.asarray(frame, dtype=np.uint8)
    if arr.shape[0] == height and arr.shape[1] == width and arr.shape[-1] == 3:
        return arr.copy()
    img = PilImage.fromarray(arr[..., :3])
    return np.asarray(img.resize((width, height), PilImage.BILINEAR), dtype=np.uint8)


def synthetic_rgb(width: int, height: int, step: int, view: str):
    import numpy as np

    view_idx = {"left": 0, "front": 1, "right": 2}.get(view, 1)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[..., 0] = np.mod(x + 0.035 * step + 0.11 * view_idx, 1.0)
    img[..., 1] = np.mod(y + 0.025 * step, 1.0)
    img[..., 2] = 0.20 + 0.10 * view_idx
    if view == "front":
        lane = np.abs(x - 0.50) < 0.055
        img[..., 1] = np.where(lane, 0.92, img[..., 1])
    return (np.clip(img, 0, 1) * 255).astype("uint8")


def _load_predict_scale(repo: Path, default: float) -> float:
    try:
        from agent import waypoint_agent_ovon as omni

        return float(getattr(omni, "PREDICT_SCALE", default))
    except Exception:
        return default


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def _instruction_from_request(req: dict[str, Any]) -> str:
    subgoal = req.get("subgoal")
    if isinstance(subgoal, dict):
        semantic = subgoal.get("semantic_goal") if isinstance(subgoal.get("semantic_goal"), dict) else {}
        if semantic:
            action = str(semantic.get("subgoal_type") or "approach").strip().lower()
            target = str(semantic.get("target") or "").strip()
            relation = str(semantic.get("relation") or "").strip()
            constraints = semantic.get("constraints") if isinstance(semantic.get("constraints"), list) else []
            verb = {
                "find": "Find and navigate toward",
                "pass": "Go past",
                "enter": "Go through",
                "approach": "Go to",
                "verify": "Stop near",
                "ask": "Stop and wait near",
            }.get(action, "Go to")
            parts = [f"{verb} {target}".strip()]
            if relation:
                parts.append(relation)
            if constraints:
                parts.append("Keep these constraints: " + "; ".join(str(item) for item in constraints))
            subgoal_text = ". ".join(parts)
        else:
            subgoal_text = str(subgoal.get("subgoal") or subgoal.get("success_condition") or "")
    else:
        subgoal_text = str(subgoal or "")
    instruction = str(req.get("instruction") or subgoal_text or req.get("mission") or "").strip()
    if instruction:
        return instruction
    return "Move cautiously toward the current navigation subgoal and stop if the goal is reached."


def _pose(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [_float(value[0]), _float(value[1]), _float(value[2])]
    return [0.0, 0.0, 0.0]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_builtin(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _collect_waypoints(value: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []

    def visit(obj: Any):
        if isinstance(obj, (list, tuple)):
            if len(obj) >= 2 and _is_number(obj[0]) and _is_number(obj[1]):
                out.append((float(obj[0]), float(obj[1])))
                return
            for item in obj:
                visit(item)

    visit(value)
    return out


def _first_number(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_number(item)
            if found is not None:
                return found
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _probability(value: float | None) -> float | None:
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def _recover_angle_deg(value: Any) -> float:
    angle = _first_number(_to_builtin(value))
    if angle is None:
        return 0.0
    return math.degrees(angle)


def main(args=None):
    rclpy.init(args=args)
    node = OmniNavModelClientNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
