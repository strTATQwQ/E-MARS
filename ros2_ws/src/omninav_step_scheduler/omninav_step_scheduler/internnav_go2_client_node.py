from __future__ import annotations

import base64
from collections import deque
import json
import math
import pickle
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .internnav_bridge import action_code_to_name, normalize_action_name, primitive_motion

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except Exception:  # pragma: no cover
    rclpy = None
    Node = object
    Twist = None
    Image = None
    Odometry = None
    String = None


CMA_MODEL_SETTINGS = {
    "policy_name": "CMA_Policy",
    "max_step": 200,
    "len_traj_act": 4,
    "state_encoder": {
        "hidden_size": 512,
        "rnn_type": "GRU",
        "num_recurrent_layers": 2,
    },
    "progress_monitor": {
        "use": True,
        "alpha": 1.0,
    },
    "instruction_encoder": {
        "sensor_uuid": "instruction",
        "vocab_size": 2504,
        "use_pretrained_embeddings": True,
        "embedding_file": "data/vln_pe/raw_data/r2r/embeddings.json.gz",
        "dataset_vocab": "data/vln_pe/raw_data/r2r/train/train.json.gz",
        "fine_tune_embeddings": False,
        "embedding_size": 50,
        "hidden_size": 128,
        "rnn_type": "LSTM",
        "final_state_only": True,
        "bidirectional": True,
    },
    "rgb_encoder": {
        "cnn_type": "TorchVisionResNet50",
        "output_size": 256,
        "trainable": False,
    },
    "depth_encoder": {
        "cnn_type": "VlnResnetDepthEncoder",
        "output_size": 128,
        "backbone": "resnet50",
        "ddppo_checkpoint": "checkpoints/ddppo-models/gibson-4plus-mp3d-train-val-test-resnet50.pth",
        "trainable": False,
    },
}


def serialize_obs(obs: Any) -> str:
    return base64.b64encode(pickle.dumps(obs)).decode("utf-8")


def post_json(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def reset_payload(reset_index: list[int] | None = None) -> dict[str, Any]:
    return {"reset_index": reset_index}


def make_agent_config(
    *,
    server_host: str,
    server_port: int,
    model_name: str,
    ckpt_path: str,
    env_num: int = 1,
    proc_num: int = 1,
) -> dict[str, Any]:
    settings = json.loads(json.dumps(CMA_MODEL_SETTINGS))
    settings.update({"env_num": env_num, "proc_num": proc_num})
    return {
        "server_host": server_host,
        "server_port": server_port,
        "model_name": model_name,
        "ckpt_path": ckpt_path,
        "model_settings": settings,
    }


def action_to_twist(action_response: Any, *, forward_speed: float, turn_speed: float) -> tuple[Any, int, str]:
    code = extract_action_code(action_response)
    twist = Twist()
    reason = "stop"
    if code == 1:
        twist.linear.x = float(forward_speed)
        reason = "move_forward"
    elif code == 2:
        twist.angular.z = abs(float(turn_speed))
        reason = "turn_left"
    elif code == 3:
        twist.angular.z = -abs(float(turn_speed))
        reason = "turn_right"
    elif code == -1:
        reason = "stand_still"
    return twist, code, reason


def extract_action_code(action_response: Any) -> int:
    value = action_response
    if isinstance(value, dict) and "action" in value:
        value = value["action"]
    if isinstance(value, list) and value:
        value = value[0]
        if isinstance(value, dict) and "action" in value:
            value = value["action"]
        if isinstance(value, list) and value:
            value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def ros_image_to_rgb(msg: Any) -> np.ndarray:
    require_numpy()
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
        raise ValueError(f"unsupported rgb encoding: {msg.encoding!r}")
    rows = data.reshape((height, step))[:, : width * channels]
    arr = rows.reshape((height, width, channels))
    if channels == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif encoding in {"bgr8", "bgra8"}:
        arr = arr[..., :3][..., ::-1]
    else:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=True)


def ros_image_to_depth(msg: Any) -> np.ndarray:
    require_numpy()
    encoding = str(getattr(msg, "encoding", "32FC1")).lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    if encoding in {"32fc1", "32fc"}:
        raw = np.frombuffer(msg.data, dtype=np.float32)
        cols = step // 4
        arr = raw.reshape((height, cols))[:, :width]
    elif encoding in {"16uc1", "mono16"}:
        raw = np.frombuffer(msg.data, dtype=np.uint16)
        cols = step // 2
        arr = raw.reshape((height, cols))[:, :width].astype(np.float32) / 1000.0
    elif encoding in {"mono8", "8uc1"}:
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        cols = step
        arr = raw.reshape((height, cols))[:, :width].astype(np.float32) / 255.0 * 5.0
    else:
        raise ValueError(f"unsupported depth encoding: {msg.encoding!r}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=0.0)
    return np.clip(arr, 0.0, 10.0).astype(np.float32)[..., None]


def normalize_depth(depth: np.ndarray, max_meters: float) -> np.ndarray:
    require_numpy()
    max_meters = float(max_meters)
    if max_meters <= 0.0:
        return depth.astype(np.float32, copy=True)
    return np.clip(depth.astype(np.float32, copy=False) / max_meters, 0.0, 1.0).astype(np.float32, copy=True)


def resize_nearest(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    require_numpy()
    if arr.shape[0] == height and arr.shape[1] == width:
        return arr.copy()
    ys = np.linspace(0, arr.shape[0] - 1, height).astype(np.int64)
    xs = np.linspace(0, arr.shape[1] - 1, width).astype(np.int64)
    return arr[ys][:, xs].copy()


def synthetic_rgb(step: int, height: int, width: int) -> np.ndarray:
    require_numpy()
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[..., 0] = np.mod(x + 0.03 * step, 1.0)
    img[..., 1] = np.mod(y + 0.02 * step, 1.0)
    img[..., 2] = 0.35
    lane = np.abs(x - 0.5) < 0.06
    img[..., 1] = np.where(lane, 0.95, img[..., 1])
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def synthetic_depth(height: int, width: int) -> np.ndarray:
    require_numpy()
    y = np.linspace(0.6, 3.5, height, dtype=np.float32)[:, None]
    return np.repeat(y, width, axis=1)[..., None].astype(np.float32)


def quaternion_to_yaw(q: Any) -> float:
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_delta_rad(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def planar_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


class ProgressRecoveryGuard:
    def __init__(
        self,
        *,
        enabled: bool,
        window: int,
        min_yaw_progress_rad: float,
        min_xy_progress_m: float,
        forward_steps: int,
    ):
        self.enabled = bool(enabled)
        self.window = max(2, int(window))
        self.min_yaw_progress_rad = max(0.0, float(min_yaw_progress_rad))
        self.min_xy_progress_m = max(0.0, float(min_xy_progress_m))
        self.forward_steps = max(1, int(forward_steps))
        self.samples: deque[tuple[int, tuple[float, float], float]] = deque(maxlen=self.window)
        self.recovery_remaining = 0

    def reset(self):
        self.samples.clear()
        self.recovery_remaining = 0

    def select_action(
        self,
        model_code: int,
        pose_xy: tuple[float, float],
        yaw: float,
    ) -> tuple[int, dict[str, Any]]:
        detail: dict[str, Any] = {
            "override_active": False,
            "override_reason": "",
            "recovery_remaining": self.recovery_remaining,
        }
        if not self.enabled:
            return model_code, detail

        if self.recovery_remaining > 0:
            self.recovery_remaining -= 1
            detail.update(
                {
                    "override_active": True,
                    "override_reason": "deadlock_recovery_forward",
                    "recovery_remaining": self.recovery_remaining,
                }
            )
            return 1, detail

        if model_code not in (2, 3):
            self.samples.clear()
            return model_code, detail

        self.samples.append((model_code, pose_xy, float(yaw)))
        if len(self.samples) < self.window:
            detail["window_fill"] = len(self.samples)
            return model_code, detail

        first_code, first_xy, first_yaw = self.samples[0]
        last_code, last_xy, last_yaw = self.samples[-1]
        same_turn = all(sample[0] == first_code for sample in self.samples)
        yaw_progress = abs(angle_delta_rad(last_yaw, first_yaw))
        xy_progress = planar_distance(last_xy, first_xy)
        detail.update(
            {
                "same_turn": same_turn,
                "yaw_progress_rad": yaw_progress,
                "xy_progress_m": xy_progress,
            }
        )
        if same_turn and yaw_progress < self.min_yaw_progress_rad and xy_progress < self.min_xy_progress_m:
            self.samples.clear()
            self.recovery_remaining = self.forward_steps - 1
            detail.update(
                {
                    "override_active": True,
                    "override_reason": "deadlock_recovery_forward",
                    "recovery_remaining": self.recovery_remaining,
                }
            )
            return 1, detail
        return model_code, detail


class InternNavGo2ClientNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run InternNavGo2ClientNode")
        require_numpy()
        super().__init__("internnav_go2_client")
        self.declare_parameter("server_host", "10.100.100.128")
        self.declare_parameter("server_port", 8087)
        self.declare_parameter("model_name", "cma")
        self.declare_parameter("ckpt_path", "checkpoints/r2r/fine_tuned/cma_plus")
        self.declare_parameter("output_topic", "/go2/cmd_vel")
        self.declare_parameter("front_image_topic", "")
        self.declare_parameter("depth_topic", "")
        self.declare_parameter("odom_topic", "")
        self.declare_parameter("instruction_tokens", "101,102,103,104,105")
        self.declare_parameter("instruction_token_profiles_json", "{}")
        self.declare_parameter("inference_rate_hz", 1.0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("command_ttl_sec", 1.2)
        self.declare_parameter("forward_speed_mps", 0.12)
        self.declare_parameter("turn_speed_radps", 0.35)
        self.declare_parameter("image_width", 256)
        self.declare_parameter("image_height", 256)
        self.declare_parameter("http_timeout_sec", 30.0)
        self.declare_parameter("init_retry_count", 5)
        self.declare_parameter("init_retry_sleep_sec", 2.0)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("synthetic_fallback", True)
        self.declare_parameter("publish_cmd_vel", True)
        self.declare_parameter("action_json_topic", "/internnav/action_json")
        self.declare_parameter("action_ttl_sec", 0.5)
        self.declare_parameter("depth_max_meters", 10.0)
        self.declare_parameter("reset_on_init", True)
        self.declare_parameter("enable_degenerate_recovery", False)
        self.declare_parameter("degenerate_recovery_window", 6)
        self.declare_parameter("degenerate_min_yaw_progress_rad", 0.05)
        self.declare_parameter("degenerate_min_xy_progress_m", 0.02)
        self.declare_parameter("degenerate_recovery_forward_steps", 5)

        self.server_host = str(self.get_parameter("server_host").value)
        self.server_port = int(self.get_parameter("server_port").value)
        self.model_name = str(self.get_parameter("model_name").value)
        self.ckpt_path = str(self.get_parameter("ckpt_path").value)
        self.base_url = f"http://{self.server_host}:{self.server_port}"
        self.timeout_sec = float(self.get_parameter("http_timeout_sec").value)
        self.forward_speed = float(self.get_parameter("forward_speed_mps").value)
        self.turn_speed = float(self.get_parameter("turn_speed_radps").value)
        self.command_ttl_sec = float(self.get_parameter("command_ttl_sec").value)
        self.width = int(self.get_parameter("image_width").value)
        self.height = int(self.get_parameter("image_height").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.synthetic_fallback = bool(self.get_parameter("synthetic_fallback").value)
        self.publish_cmd_vel = bool(self.get_parameter("publish_cmd_vel").value)
        self.action_ttl_sec = float(self.get_parameter("action_ttl_sec").value)
        self.depth_max_meters = float(self.get_parameter("depth_max_meters").value)
        self.instruction_tokens = parse_tokens(str(self.get_parameter("instruction_tokens").value))
        self.instruction_token_profiles = parse_instruction_token_profiles(
            str(self.get_parameter("instruction_token_profiles_json").value)
        )
        self.recovery_guard = ProgressRecoveryGuard(
            enabled=bool(self.get_parameter("enable_degenerate_recovery").value),
            window=int(self.get_parameter("degenerate_recovery_window").value),
            min_yaw_progress_rad=float(self.get_parameter("degenerate_min_yaw_progress_rad").value),
            min_xy_progress_m=float(self.get_parameter("degenerate_min_xy_progress_m").value),
            forward_steps=int(self.get_parameter("degenerate_recovery_forward_steps").value),
        )

        self.cmd_pub = self.create_publisher(Twist, str(self.get_parameter("output_topic").value), 10)
        self.action_pub = self.create_publisher(String, str(self.get_parameter("action_json_topic").value), 50)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)

        self.frame_lock = threading.Lock()
        self.rgb_frame: np.ndarray | None = None
        self.depth_frame: np.ndarray | None = None
        self.instruction_text = ""
        self.active_subgoal: dict[str, Any] = {}
        self.pose_xyz = [0.0, 0.0, 0.0]
        self.pose_yaw = 0.0
        self.rotation_xyzw = [0.0, 0.0, 0.0, 1.0]
        self.step_index = 0

        self.agent_name: str | None = None
        self.inflight = False
        self.last_cmd = Twist()
        self.last_cmd_time = 0.0
        self.last_action_code = 0

        front_topic = str(self.get_parameter("front_image_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        if front_topic:
            self.create_subscription(Image, front_topic, self.on_rgb, 2)
        if depth_topic:
            self.create_subscription(Image, depth_topic, self.on_depth, 2)
        if odom_topic:
            self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        self.create_subscription(String, "/isaac/reset_episode", self.on_episode_reset, 10)
        self.create_subscription(String, "/user_instruction", self.on_user_instruction, 10)
        self.create_subscription(String, "/scheduler/active_subgoal_json", self.on_active_subgoal, 10)

        self.init_agent()
        inference_period = 1.0 / max(0.05, float(self.get_parameter("inference_rate_hz").value))
        publish_period = 1.0 / max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(inference_period, self.inference_tick)
        self.create_timer(publish_period, self.publish_tick)
        self.get_logger().info(
            f"InternNav Go2 client ready server={self.base_url} output={self.get_parameter('output_topic').value} "
            f"dry_run={self.dry_run} synthetic_fallback={self.synthetic_fallback} "
            f"depth_max_meters={self.depth_max_meters} recovery={self.recovery_guard.enabled} "
            f"publish_cmd_vel={self.publish_cmd_vel}"
        )

    def init_agent(self):
        cfg = make_agent_config(
            server_host=self.server_host,
            server_port=self.server_port,
            model_name=self.model_name,
            ckpt_path=self.ckpt_path,
        )
        retry_count = int(self.get_parameter("init_retry_count").value)
        retry_sleep_sec = float(self.get_parameter("init_retry_sleep_sec").value)
        last_error: Exception | None = None
        for attempt in range(max(1, retry_count)):
            t0 = time.perf_counter()
            try:
                response = post_json(f"{self.base_url}/agent/init", {"agent_config": cfg}, self.timeout_sec)
                self.agent_name = str(response.get("agent_name") or self.model_name)
                self.publish_metric(
                    "internnav_go2_init",
                    latency_s=time.perf_counter() - t0,
                    agent_name=self.agent_name,
                    attempt=attempt + 1,
                    reset_on_init=bool(self.get_parameter("reset_on_init").value),
                    depth_max_meters=self.depth_max_meters,
                )
                if bool(self.get_parameter("reset_on_init").value):
                    self.reset_agent(reason="init")
                return
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = exc
                self.get_logger().warning(
                    f"InternNav init failed attempt {attempt + 1}/{retry_count}: {exc!r}"
                )
                if attempt + 1 < retry_count:
                    time.sleep(retry_sleep_sec)
        raise RuntimeError(f"failed to init InternNav agent after {retry_count} attempts: {last_error!r}")

    def on_rgb(self, msg):
        try:
            frame = resize_nearest(ros_image_to_rgb(msg), self.height, self.width)
        except Exception as exc:
            self.publish_metric("internnav_go2_image_error", kind="rgb", error=repr(exc))
            return
        with self.frame_lock:
            self.rgb_frame = frame

    def on_depth(self, msg):
        try:
            frame = resize_nearest(ros_image_to_depth(msg), self.height, self.width)
            frame = normalize_depth(frame, self.depth_max_meters)
        except Exception as exc:
            self.publish_metric("internnav_go2_image_error", kind="depth", error=repr(exc))
            return
        with self.frame_lock:
            self.depth_frame = frame

    def on_odom(self, msg):
        pose = msg.pose.pose
        with self.frame_lock:
            self.pose_xyz = [float(pose.position.x), float(pose.position.y), float(pose.position.z)]
            self.rotation_xyzw = [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ]
            self.pose_yaw = quaternion_to_yaw(pose.orientation)

    def on_episode_reset(self, _msg):
        self.recovery_guard.reset()
        self.step_index = 0
        if self.agent_name:
            threading.Thread(target=self.reset_agent, kwargs={"reason": "episode_reset"}, daemon=True).start()

    def inference_tick(self):
        if self.inflight:
            return
        self.inflight = True
        threading.Thread(target=self.run_inference, daemon=True).start()

    def run_inference(self):
        t0 = time.perf_counter()
        result = "ok"
        error = None
        action_response = None
        try:
            obs, source = self.make_obs()
            instruction_context = strip_local_obs_metadata(obs)
            pose_xy, pose_yaw = self.current_progress_pose()
            pose_at_snapshot = [pose_xy[0], pose_xy[1], pose_yaw]
            payload = {"observation": serialize_obs(obs)}
            response = post_json(f"{self.base_url}/agent/{self.agent_name}/step", payload, self.timeout_sec)
            action_response = response.get("action")
            twist, model_code, model_reason = action_to_twist(
                action_response,
                forward_speed=self.forward_speed,
                turn_speed=self.turn_speed,
            )
            actual_code, override_detail = self.recovery_guard.select_action(model_code, pose_xy, pose_yaw)
            if actual_code != model_code:
                twist, actual_code, actual_reason = action_to_twist(
                    actual_code,
                    forward_speed=self.forward_speed,
                    turn_speed=self.turn_speed,
                )
            else:
                actual_reason = model_reason
            self.last_action_code = actual_code
            if self.dry_run:
                twist = Twist()
                result = "dry_run"
            self.last_cmd = twist
            self.last_cmd_time = time.time()
            self.publish_action_json(
                request_id=f"internnav_{self.step_index:06d}",
                timestamp_request=t0,
                timestamp_response=time.perf_counter(),
                model_code=model_code,
                actual_code=actual_code,
                latency_s=time.perf_counter() - t0,
                obs_source=source,
                used_rgb=source.startswith("ros"),
                used_depth=("synthetic_depth" not in source),
                pose_at_snapshot=pose_at_snapshot,
                instruction_context=instruction_context,
                raw_action=action_response,
                override_detail=override_detail,
            )
            self.publish_metric(
                "internnav_go2_step",
                latency_s=time.perf_counter() - t0,
                result=result,
                action_code=actual_code,
                action_reason=actual_reason,
                model_action_code=model_code,
                model_action_reason=model_reason,
                override_active=override_detail.get("override_active", False),
                override_reason=override_detail.get("override_reason", ""),
                override_detail=override_detail,
                obs_source=source,
                instruction_text=instruction_context.get("instruction", ""),
                subgoal=instruction_context.get("subgoal", ""),
                instruction_token_source=instruction_context.get("token_source", ""),
                raw_action=action_response,
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
            error = repr(exc)
            result = "error"
            self.last_cmd = Twist()
            self.last_cmd_time = time.time()
            self.publish_metric("internnav_go2_step", latency_s=time.perf_counter() - t0, result=result, error=error, raw_action=action_response)
            self.get_logger().warning(f"InternNav step failed: {error}")
        finally:
            self.inflight = False

    def make_obs(self) -> tuple[list[dict[str, Any]], str]:
        with self.frame_lock:
            rgb = None if self.rgb_frame is None else self.rgb_frame.copy()
            depth = None if self.depth_frame is None else self.depth_frame.copy()
            gps = list(self.pose_xyz)
            rot = list(self.rotation_xyzw)
            instruction_text, subgoal_text, success_condition = self.current_instruction_context_locked()
        source = "ros"
        if rgb is None:
            if not self.synthetic_fallback:
                raise RuntimeError("no rgb frame and synthetic_fallback=false")
            rgb = synthetic_rgb(self.step_index, self.height, self.width)
            source = "synthetic"
        if depth is None:
            depth = normalize_depth(synthetic_depth(self.height, self.width), self.depth_max_meters)
            source = f"{source}+synthetic_depth" if source == "ros" else source
        self.step_index += 1
        instruction_for_model = compose_instruction_text(instruction_text, subgoal_text, success_condition)
        tokens, token_source = select_instruction_tokens(
            instruction_for_model,
            self.instruction_tokens,
            self.instruction_token_profiles,
        )
        return [
            {
                "instruction": instruction_for_model,
                "instruction_tokens": list(tokens),
                "instruction_context": {
                    "instruction": instruction_text,
                    "subgoal": subgoal_text,
                    "success_condition": success_condition,
                    "token_source": token_source,
                    "token_count": len(tokens),
                },
                "rgb": rgb.astype(np.uint8, copy=False),
                "depth": depth.astype(np.float32, copy=False),
                "globalgps": gps,
                "globalrotation": rot,
            }
        ], source

    def current_progress_pose(self) -> tuple[tuple[float, float], float]:
        with self.frame_lock:
            return (float(self.pose_xyz[0]), float(self.pose_xyz[1])), float(self.pose_yaw)

    def current_instruction_context_locked(self) -> tuple[str, str, str]:
        subgoal = self.active_subgoal
        return (
            self.instruction_text,
            str(subgoal.get("subgoal") or ""),
            str(subgoal.get("success_condition") or ""),
        )

    def on_user_instruction(self, msg):
        payload = json_object_or_raw(msg.data)
        instruction = str(payload.get("instruction") or payload.get("raw") or "").strip()
        with self.frame_lock:
            self.instruction_text = instruction

    def on_active_subgoal(self, msg):
        payload = json_object_or_raw(msg.data)
        with self.frame_lock:
            self.active_subgoal = payload

    def publish_tick(self):
        if not self.publish_cmd_vel:
            return
        cmd = self.last_cmd
        if time.time() - self.last_cmd_time > self.command_ttl_sec:
            cmd = Twist()
        self.cmd_pub.publish(cmd)

    def publish_action_json(
        self,
        *,
        request_id: str,
        timestamp_request: float,
        timestamp_response: float,
        model_code: int,
        actual_code: int,
        latency_s: float,
        obs_source: str,
        used_rgb: bool,
        used_depth: bool,
        pose_at_snapshot: list[float],
        instruction_context: dict[str, Any],
        raw_action: Any,
        override_detail: dict[str, Any],
    ) -> None:
        model_action = action_code_to_name(model_code)
        applied_action = action_code_to_name(actual_code)
        if normalize_action_name(applied_action) == "unknown":
            applied_action = "stop"
        motion = primitive_motion(applied_action, forward_distance_m=0.35, turn_yaw_deg=15.0)
        msg = String()
        msg.data = json.dumps(
            {
                "source": "internnav",
                "request_id": request_id,
                "timestamp_request": time.time() - max(0.0, latency_s),
                "timestamp_response": time.time(),
                "latency_sec": latency_s,
                "model_action": model_action,
                "applied_action": applied_action,
                "primitive": motion["primitive"],
                "distance_m": motion["distance_m"],
                "yaw_deg": motion["yaw_deg"],
                "confidence": None,
                "obs_source": obs_source,
                "used_rgb": bool(used_rgb),
                "used_depth": bool(used_depth),
                "instruction": instruction_context.get("instruction", ""),
                "subgoal": instruction_context.get("subgoal", ""),
                "success_condition": instruction_context.get("success_condition", ""),
                "instruction_token_source": instruction_context.get("token_source", ""),
                "instruction_token_count": instruction_context.get("token_count", 0),
                "recovery_override": bool(override_detail.get("override_active", False)),
                "recovery_reason": override_detail.get("override_reason") or None,
                "pose_at_snapshot": pose_at_snapshot,
                "ttl_sec": self.action_ttl_sec,
                "raw_action": raw_action,
            },
            ensure_ascii=False,
        )
        self.action_pub.publish(msg)

    def publish_metric(self, event_type: str, **kwargs):
        msg = String()
        payload = {"timestamp": time.time(), "event_type": event_type, "node": "internnav_go2_client", **kwargs}
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.metric_pub.publish(msg)

    def reset_agent(self, *, reason: str = "manual", reset_index: list[int] | None = None) -> bool:
        if not self.agent_name:
            return False
        t0 = time.perf_counter()
        try:
            response = post_json(
                f"{self.base_url}/agent/{self.agent_name}/reset",
                reset_payload(reset_index),
                self.timeout_sec,
            )
            self.publish_metric(
                "internnav_go2_reset",
                latency_s=time.perf_counter() - t0,
                result="ok",
                reason=reason,
                response=response,
            )
            return True
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            self.publish_metric(
                "internnav_go2_reset",
                latency_s=time.perf_counter() - t0,
                result="error",
                reason=reason,
                error=repr(exc),
            )
            self.get_logger().warning(f"InternNav reset failed: {exc!r}")
            return False


def parse_tokens(raw: str) -> list[int]:
    tokens: list[int] = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            tokens.append(int(item))
        except ValueError:
            continue
    return tokens or [101, 102, 103, 104, 105]


def parse_instruction_token_profiles(raw: str | dict[str, Any]) -> dict[str, list[int]]:
    if isinstance(raw, dict):
        payload = raw
    else:
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            payload = {}
    if not isinstance(payload, dict):
        return {}
    profiles: dict[str, list[int]] = {}
    for key, value in payload.items():
        name = str(key).strip().lower()
        if not name:
            continue
        if isinstance(value, str):
            tokens = parse_tokens(value)
        elif isinstance(value, (list, tuple)):
            tokens = []
            for item in value:
                try:
                    tokens.append(int(item))
                except (TypeError, ValueError):
                    continue
            if not tokens:
                tokens = parse_tokens("")
        else:
            continue
        profiles[name] = tokens
    return profiles


def compose_instruction_text(instruction: str, subgoal: str, success_condition: str) -> str:
    parts = []
    if instruction.strip():
        parts.append(instruction.strip())
    if subgoal.strip() and subgoal.strip().lower() not in " ".join(parts).lower():
        parts.append(f"Current subgoal: {subgoal.strip()}.")
    if success_condition.strip():
        parts.append(f"Stop condition: {success_condition.strip()}.")
    return " ".join(parts).strip() or "Move cautiously toward the current navigation target and stop at the goal."


def select_instruction_tokens(
    instruction_text: str,
    default_tokens: list[int],
    profiles: dict[str, list[int]],
) -> tuple[list[int], str]:
    lowered = instruction_text.lower()
    for key in sorted((k for k in profiles if k != "default"), key=len, reverse=True):
        if key in lowered:
            return list(profiles[key]), f"profile:{key}"
    if "default" in profiles:
        return list(profiles["default"]), "profile:default"
    return list(default_tokens), "fallback_static_cma_tokens"


def strip_local_obs_metadata(obs: list[dict[str, Any]]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for index, item in enumerate(obs):
        if not isinstance(item, dict):
            continue
        value = item.pop("instruction_context", None)
        if index == 0 and isinstance(value, dict):
            context = dict(value)
    return context


def json_object_or_raw(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def require_numpy():
    if np is None:
        raise RuntimeError("numpy is required to run internnav_go2_client_node")


def main(args=None):
    rclpy.init(args=args)
    node = InternNavGo2ClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if "context is not valid" not in repr(exc) and "rcl_shutdown already called" not in repr(exc):
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
