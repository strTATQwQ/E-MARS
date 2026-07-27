"""DGX-side typed ROS 2 InternVLA model node."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from internvla_ros2_msgs.action import Step
from internvla_ros2_msgs.msg import ModelState, ObservationMetadata
from internvla_ros2_msgs.srv import Health, Initialize, Reset, Shutdown
from nav_msgs.msg import Path as NavPath
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from .identity import CHECKPOINT_REVISION, MODEL_REVISION
from .fault_injection import fault_profile_enabled, load_fault_restart_session
from .compressed_observation import (
    DEFAULT_MAX_DEPTH_BYTES,
    DEFAULT_MAX_RGB_BYTES,
    ObservationCodecError,
    decode_depth_png,
    decode_rgb_jpeg,
)
from .observation import compressed_observation_digest, observation_digest
from .observation_guard import describe_request_identity
from .protocol import (
    DualClockWindow,
    IdempotencyCache,
    GenerationBarrier,
    ProtocolError,
    RequestIdentity,
    STATUS_CANCELED,
    STATUS_INTERNAL_ERROR,
    STATUS_INVALID_REQUEST,
    STATUS_OBSERVATION_MISSING,
    STATUS_OK,
    STATUS_SHUTTING_DOWN,
    STATUS_STALE,
    STATUS_TIMEOUT,
    validate_action,
    validate_protocol_version,
)
from .trajectory_capture import MetricTrajectoryCapture
from .trajectory_rerank import (
    TrajectoryRerankConfig,
    config_mapping as trajectory_rerank_config_mapping,
    rerank_trajectories,
)


LIFECYCLE_UNINITIALIZED = 0
LIFECYCLE_LOADING = 1
LIFECYCLE_READY = 2
LIFECYCLE_RESETTING = 3
LIFECYCLE_ERROR = 4
LIFECYCLE_SHUTTING_DOWN = 5


_TRAJECTORY_RERANK_SEED_CONTRACT = (
    "internvla_t5_trajectory_rerank_seed_v1:"
    "sha256(model_revision,checkpoint_revision,episode_id,reset_generation,sequence_id):uint32_be"
)


def _time_ns(message: Any) -> int:
    return int(message.sec) * 1_000_000_000 + int(message.nanosec)


def _assign_time(message: Any, value_ns: int) -> None:
    message.sec = int(value_ns // 1_000_000_000)
    message.nanosec = int(value_ns % 1_000_000_000)


def _metadata_identity(metadata: ObservationMetadata) -> RequestIdentity:
    return RequestIdentity(
        str(metadata.episode_id),
        int(metadata.reset_generation),
        int(metadata.sequence_id),
        str(metadata.request_id),
    )


def _trajectory_rerank_seed(
    identity: RequestIdentity,
    *,
    model_revision: str = MODEL_REVISION,
    checkpoint_revision: str = CHECKPOINT_REVISION,
) -> int:
    """Derive the stable per-step rerank seed without observation pixels."""

    material = json.dumps(
        {
            "checkpoint_revision": str(checkpoint_revision),
            "episode_id": str(identity.episode_id),
            "model_revision": str(model_revision),
            "reset_generation": int(identity.reset_generation),
            "sequence_id": int(identity.sequence_id),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(material).digest()[:4], "big", signed=False
    )


def _seed_trajectory_rerank_rngs(seed: int) -> None:
    """Fail closed while seeding every RNG used by the real policy step."""

    import torch

    stable_seed = int(seed)
    random.seed(stable_seed)
    np.random.seed(stable_seed)
    torch.manual_seed(stable_seed)
    torch.cuda.manual_seed_all(stable_seed)


def _identity_mismatch_message(
    expected: RequestIdentity, observed: RequestIdentity
) -> str:
    return (
        "observation metadata identity mismatch: "
        f"expected={describe_request_identity(expected)} "
        f"observed={describe_request_identity(observed)}"
    )


def _decode_rgb(message: Image) -> np.ndarray:
    if message.width != 640 or message.height != 480:
        raise ProtocolError(
            STATUS_INVALID_REQUEST,
            f"RGB must be 640x480, observed {message.width}x{message.height}",
        )
    if message.encoding not in ("rgb8", "bgr8"):
        raise ProtocolError(STATUS_INVALID_REQUEST, f"unsupported RGB encoding {message.encoding!r}")
    row_bytes = int(message.width) * 3
    if int(message.step) < row_bytes:
        raise ProtocolError(STATUS_INVALID_REQUEST, "RGB step is smaller than row payload")
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    expected = int(message.step) * int(message.height)
    if raw.size != expected:
        raise ProtocolError(STATUS_INVALID_REQUEST, "RGB payload size does not match image metadata")
    rgb = raw.reshape(message.height, message.step)[:, :row_bytes].reshape(message.height, message.width, 3)
    if message.encoding == "bgr8":
        rgb = rgb[:, :, ::-1]
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _decode_depth(message: Image) -> np.ndarray:
    if message.width != 640 or message.height != 480:
        raise ProtocolError(
            STATUS_INVALID_REQUEST,
            f"depth must be 640x480, observed {message.width}x{message.height}",
        )
    if message.encoding != "32FC1":
        raise ProtocolError(STATUS_INVALID_REQUEST, f"unsupported depth encoding {message.encoding!r}")
    row_bytes = int(message.width) * 4
    if int(message.step) < row_bytes or int(message.step) % 4:
        raise ProtocolError(STATUS_INVALID_REQUEST, "invalid depth row step")
    dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
    raw = np.frombuffer(bytes(message.data), dtype=dtype)
    expected = int(message.step // 4) * int(message.height)
    if raw.size != expected:
        raise ProtocolError(STATUS_INVALID_REQUEST, "depth payload size does not match image metadata")
    depth = raw.reshape(message.height, message.step // 4)[:, : message.width]
    depth = np.ascontiguousarray(depth, dtype=np.float32)[:, :, None]
    if not np.isfinite(depth).all():
        raise ProtocolError(STATUS_INVALID_REQUEST, "depth contains NaN/Inf")
    if float(depth.min()) < 0.0 or float(depth.max()) > 1.0:
        raise ProtocolError(STATUS_INVALID_REQUEST, "depth is outside frozen normalized range [0,1]")
    return depth


def _extract_action(raw: Any) -> int:
    try:
        return validate_action(int(raw[0]["action"][0]))
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(STATUS_INTERNAL_ERROR, f"invalid model action structure: {raw!r}") from exc


@dataclass(frozen=True)
class StepPayload:
    identity: RequestIdentity
    action: int
    action_source: int
    trajectory: np.ndarray | None
    valid_until_ns: int
    inference_latency_sec: float


@dataclass(frozen=True)
class _CachedObservationComponent:
    revision: int
    message: Any


@dataclass(frozen=True)
class _AcceptedObservationTuple:
    identity: RequestIdentity
    rgb_revision: int
    depth_revision: int
    metadata_revision: int
    epoch: int


class DeterministicFakeAgent:
    """Protocol fault-injection backend; never used for navigation metrics."""

    def __init__(self, delay_sec: float = 0.0):
        self.delay_sec = float(delay_sec)
        self.index = 0
        self.last_trajectory: np.ndarray | None = None

    def step(self, _obs: Any) -> list[dict[str, Any]]:
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        actions = (1, 2, 3, -1, 0)
        action = actions[self.index % len(actions)]
        self.index += 1
        self.last_trajectory = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32)
        return [{"action": [action], "ideal_flag": True}]

    def reset(self, _reset_index: Any = None) -> None:
        self.index = 0
        self.last_trajectory = None


class InternVLAModelNode(Node):
    def __init__(self) -> None:
        super().__init__("internvla_model_node")
        self.declare_parameter("backend", os.environ.get("INTERNVLA_BACKEND", "real"))
        self.declare_parameter("fake_delay_sec", float(os.environ.get("INTERNVLA_FAKE_DELAY_SEC", "0")))
        self.declare_parameter(
            "preload_model",
            os.environ.get("INTERNVLA_PRELOAD_MODEL", "0").strip().lower()
            in {"1", "true", "yes", "on"},
        )
        self.declare_parameter("observation_wait_sec", 2.0)
        self.declare_parameter("observation_transport", "raw")
        self.declare_parameter("maximum_rgb_bytes", DEFAULT_MAX_RGB_BYTES)
        self.declare_parameter("maximum_depth_bytes", DEFAULT_MAX_DEPTH_BYTES)
        self.declare_parameter("observation_cache_size", 32)
        self.declare_parameter("idempotency_cache_size", 512)
        self.backend = str(self.get_parameter("backend").value)
        if self.backend not in ("real", "deterministic_fake"):
            raise RuntimeError(f"unsupported backend: {self.backend}")
        self.observation_transport = str(
            self.get_parameter("observation_transport").value
        )
        if self.observation_transport not in {"raw", "compressed"}:
            raise RuntimeError("observation_transport must be raw or compressed")
        self.maximum_rgb_bytes = int(self.get_parameter("maximum_rgb_bytes").value)
        self.maximum_depth_bytes = int(
            self.get_parameter("maximum_depth_bytes").value
        )
        self.t5_sim_time_only = (
            os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
            and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
            and os.environ.get("INTERNNAV_T5_LANE", "") in {"a", "b"}
        )
        if self.t5_sim_time_only and not bool(
            self.get_parameter("use_sim_time").value
        ):
            raise RuntimeError("T5 completion_sim requires use_sim_time=true")
        system2_queue_horizon = os.environ.get(
            "INTERNVLA_T5_SYSTEM2_QUEUE_HORIZON", "0"
        )
        if system2_queue_horizon not in {"0", "1"}:
            raise RuntimeError(
                "INTERNVLA_T5_SYSTEM2_QUEUE_HORIZON must be 0 or 1"
            )
        self._system2_queue_horizon = int(system2_queue_horizon)
        if self._system2_queue_horizon and not self.t5_sim_time_only:
            raise RuntimeError(
                "System2 receding horizon is restricted to exact T5 Isaac "
                "completion_sim"
            )
        system1_queue_horizon = os.environ.get(
            "INTERNVLA_T5_SYSTEM1_QUEUE_HORIZON", "0"
        )
        if system1_queue_horizon not in {"0", "1"}:
            raise RuntimeError(
                "INTERNVLA_T5_SYSTEM1_QUEUE_HORIZON must be 0 or 1"
            )
        self._system1_queue_horizon = int(system1_queue_horizon)
        if self._system1_queue_horizon and not self.t5_sim_time_only:
            raise RuntimeError(
                "System1 receding horizon is restricted to exact T5 Isaac "
                "completion_sim"
            )
        rerank_flag = os.environ.get("INTERNVLA_T5_TRAJECTORY_RERANK", "0")
        if rerank_flag not in {"0", "1"}:
            raise RuntimeError("INTERNVLA_T5_TRAJECTORY_RERANK must be 0 or 1")
        self._trajectory_rerank_enabled = rerank_flag == "1"
        if self._trajectory_rerank_enabled and not self.t5_sim_time_only:
            raise RuntimeError(
                "trajectory rerank is restricted to exact T5 Isaac completion_sim"
            )
        self._trajectory_rerank_config = TrajectoryRerankConfig()
        result_root = os.environ.get("INTERNVLA_MODEL_RESULT_DIR")
        self._trajectory_rerank_audit_path = (
            Path(result_root).resolve() / "trajectory_rerank_records.jsonl"
            if self._trajectory_rerank_enabled and result_root
            else None
        )
        if self._trajectory_rerank_enabled and self._trajectory_rerank_audit_path is None:
            raise RuntimeError("trajectory rerank requires INTERNVLA_MODEL_RESULT_DIR")
        if (
            self._trajectory_rerank_audit_path is not None
            and self._trajectory_rerank_audit_path.exists()
        ):
            raise RuntimeError("refusing to append existing trajectory rerank audit")
        self._trajectory_rerank_audit_lock = threading.Lock()
        self._t5_sim_time_lock = threading.Lock()
        self._t5_sim_time_high_water_ns = 0

        self._callback_group = ReentrantCallbackGroup()
        self._state_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._cache_condition = threading.Condition()
        self._observation_revision = 0
        self._observation_epoch = 0
        self._rgb_cache: OrderedDict[int, _CachedObservationComponent] = OrderedDict()
        self._depth_cache: OrderedDict[int, _CachedObservationComponent] = OrderedDict()
        self._metadata_cache: OrderedDict[
            int, _CachedObservationComponent
        ] = OrderedDict()
        self._accepted_observation_tuples: OrderedDict[
            int, _AcceptedObservationTuple
        ] = OrderedDict()
        self._observation_cache_size = int(self.get_parameter("observation_cache_size").value)
        self._barrier = GenerationBarrier()
        self._results = IdempotencyCache[StepPayload](
            maximum=int(self.get_parameter("idempotency_cache_size").value)
        )
        self._capture = MetricTrajectoryCapture(
            capture_candidates=self._trajectory_rerank_enabled
        )
        self._system1_queue_remaining = 0
        self._agent: Any = None
        self._lifecycle = LIFECYCLE_UNINITIALIZED
        self._status_code = STATUS_OK
        self._status_message = "uninitialized"
        self._safe_stop = True
        self._shutting_down = False

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=8,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._state_publisher = self.create_publisher(ModelState, "/internvla/model_state", state_qos)
        if self.observation_transport == "compressed":
            self.create_subscription(
                CompressedImage,
                "/internvla/observation/rgb/compressed",
                self._on_rgb,
                image_qos,
                callback_group=self._callback_group,
            )
            self.create_subscription(
                CompressedImage,
                "/internvla/observation/depth/compressed",
                self._on_depth,
                image_qos,
                callback_group=self._callback_group,
            )
        else:
            self.create_subscription(
                Image,
                "/internvla/observation/rgb",
                self._on_rgb,
                image_qos,
                callback_group=self._callback_group,
            )
            self.create_subscription(
                Image,
                "/internvla/observation/depth",
                self._on_depth,
                image_qos,
                callback_group=self._callback_group,
            )
        self.create_subscription(
            ObservationMetadata,
            "/internvla/observation/metadata",
            self._on_metadata,
            image_qos,
            callback_group=self._callback_group,
        )
        self.create_service(
            Health,
            "/internvla/health",
            self._on_health,
            callback_group=self._callback_group,
        )
        self.create_service(
            Initialize,
            "/internvla/initialize",
            self._on_initialize,
            callback_group=self._callback_group,
        )
        self.create_service(
            Reset,
            "/internvla/reset",
            self._on_reset,
            callback_group=self._callback_group,
        )
        self.create_service(
            Shutdown,
            "/internvla/shutdown",
            self._on_shutdown,
            callback_group=self._callback_group,
        )
        self._action_server = ActionServer(
            self,
            Step,
            "/internvla/step",
            execute_callback=self._execute_step,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._callback_group,
            # Jazzy retains terminal action results for 900 seconds by
            # default. Keep that full window explicit so the bounded client
            # re-fetch cannot shorten server-side result availability.
            result_timeout=900.0,
        )
        # Loading the GB10 checkpoint can take longer than the evaluator's
        # frozen per-step deadline.  Preloading is deliberately episode-free:
        # no observation is consumed and GenerationBarrier remains
        # uninitialized until the formal Initialize request arrives.
        if bool(self.get_parameter("preload_model").value):
            self._set_state(LIFECYCLE_LOADING, STATUS_OK, "preloading model", True)
            try:
                self._agent = self._load_agent()
            except BaseException as exc:
                self._set_state(LIFECYCLE_ERROR, STATUS_INTERNAL_ERROR, repr(exc), True)
                raise
            self._set_state(
                LIFECYCLE_UNINITIALIZED,
                STATUS_OK,
                "model preloaded; awaiting initialize",
                True,
            )
        restart_session_path = os.environ.get(
            "INTERNVLA_T5_MODEL_RESTART_SESSION_FILE", ""
        )
        if restart_session_path:
            if not fault_profile_enabled() or self._agent is None:
                raise RuntimeError(
                    "fault session restore requires the exact profile and a preloaded model"
                )
            restart_session = load_fault_restart_session(
                Path(restart_session_path),
                expected_lane=os.environ["INTERNNAV_T5_LANE"],
                expected_action="model_service_restart",
            )
            self._barrier.restore(
                restart_session["episode_id"],
                restart_session["reset_generation"],
                restart_session["last_sequence_id"],
            )
            self._results.clear()
            self._clear_observations()
            self._set_state(
                LIFECYCLE_READY,
                STATUS_OK,
                "fault restart session restored",
                False,
            )
        self._publish_state()
        self.get_logger().info(
            "typed InternVLA model node ready "
            f"backend={self.backend} protocol=1 transport={self.observation_transport} "
            f"preloaded={self._agent is not None}"
        )

    def _on_rgb(self, message: Any) -> None:
        self._cache_message(self._rgb_cache, _time_ns(message.header.stamp), message)

    def _on_depth(self, message: Any) -> None:
        self._cache_message(self._depth_cache, _time_ns(message.header.stamp), message)

    def _on_metadata(self, message: ObservationMetadata) -> None:
        self._cache_message(self._metadata_cache, _time_ns(message.header.stamp), message)

    def _cache_message(
        self,
        cache: OrderedDict[int, _CachedObservationComponent],
        key: int,
        message: Any,
    ) -> None:
        if key <= 0:
            self.get_logger().warning("dropping observation component with zero timestamp")
            return
        with self._cache_condition:
            self._observation_revision += 1
            cache[key] = _CachedObservationComponent(
                revision=self._observation_revision,
                message=message,
            )
            cache.move_to_end(key)
            while len(cache) > self._observation_cache_size:
                cache.popitem(last=False)
            self._cache_condition.notify_all()

    def _on_health(self, request: Health.Request, response: Health.Response) -> Health.Response:
        try:
            validate_protocol_version(request.protocol_version)
            status_code, status_message = self._status_code, self._status_message
        except ProtocolError as exc:
            status_code, status_message = exc.status_code, str(exc)
        with self._state_lock:
            response.status_code = status_code
            response.status_message = status_message
            response.protocol_version = 1
            response.initialized = self._barrier.initialized
            response.lifecycle_state = self._lifecycle
            response.episode_id = self._barrier.episode_id
            response.reset_generation = self._barrier.reset_generation
            response.last_sequence_id = max(0, self._barrier.last_sequence_id)
            response.model_revision = MODEL_REVISION
            response.checkpoint_revision = CHECKPOINT_REVISION
        return response

    def _on_initialize(
        self, request: Initialize.Request, response: Initialize.Response
    ) -> Initialize.Response:
        try:
            validate_protocol_version(request.protocol_version)
            if request.model_revision != MODEL_REVISION:
                raise ProtocolError(STATUS_INVALID_REQUEST, "InternNav revision mismatch")
            if request.checkpoint_revision != CHECKPOINT_REVISION:
                raise ProtocolError(STATUS_INVALID_REQUEST, "checkpoint revision mismatch")
            if not request.episode_id:
                raise ProtocolError(STATUS_INVALID_REQUEST, "episode_id is required")
            with self._inference_lock:
                if self._shutting_down:
                    raise ProtocolError(STATUS_SHUTTING_DOWN, "node is shutting down")
                if self._barrier.initialized:
                    if request.episode_id != self._barrier.episode_id:
                        raise ProtocolError(
                            STATUS_INVALID_REQUEST,
                            "already initialized for a different episode; use reset",
                        )
                else:
                    self._set_state(LIFECYCLE_LOADING, STATUS_OK, "loading model", True)
                    if self._agent is None:
                        self._agent = self._load_agent()
                    self._barrier.initialize(request.episode_id)
                    self._results.clear()
                    self._clear_observations()
                    self._set_state(LIFECYCLE_READY, STATUS_OK, "ready", False)
            response.status_code = STATUS_OK
            response.status_message = "ready"
        except ProtocolError as exc:
            response.status_code = exc.status_code
            response.status_message = str(exc)
        except BaseException as exc:
            self._set_state(LIFECYCLE_ERROR, STATUS_INTERNAL_ERROR, repr(exc), True)
            response.status_code = STATUS_INTERNAL_ERROR
            response.status_message = repr(exc)
        response.protocol_version = 1
        response.initialized = self._barrier.initialized
        response.episode_id = self._barrier.episode_id
        response.reset_generation = self._barrier.reset_generation
        response.model_revision = MODEL_REVISION
        response.checkpoint_revision = CHECKPOINT_REVISION
        return response

    def _load_agent(self) -> Any:
        if self.backend == "deterministic_fake":
            return DeterministicFakeAgent(float(self.get_parameter("fake_delay_sec").value))
        internnav_root = Path(os.environ["INTERNNAV_ROOT"]).resolve()
        config_path = Path(
            os.environ.get(
                "INTERNVLA_MODEL_CONFIG",
                str(internnav_root / "scripts/eval/configs/h1_internvla_n1_async_cfg.py"),
            )
        ).resolve()
        if not config_path.is_file():
            raise RuntimeError(f"model config does not exist: {config_path}")
        os.chdir(internnav_root)
        sys.path.insert(0, str(internnav_root))
        sys.path.insert(0, str(internnav_root / "third_party/diffusion-policy"))
        spec = importlib.util.spec_from_file_location("internvla_t1_model_config", config_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load model config: {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from internnav.agent import Agent
        from internnav.configs.model import internvla_n1_cfg

        # The official evaluator always expands the sparse per-run config with
        # internvla_n1_cfg before it serializes AgentCfg to the legacy server.
        # Reproduce only official vln_default_config.py lines 306-317 here;
        # expanding the unrelated environment defaults on a model-only host
        # would incorrectly require the Isaac-side internutopia package.
        model_settings = internvla_n1_cfg.model_dump()
        model_settings.update(module.eval_cfg.agent.model_settings)
        module.eval_cfg.agent.model_settings = model_settings
        agent = Agent.init(module.eval_cfg.agent)
        policy = getattr(agent, "policy", None)
        if policy is None or not callable(getattr(policy, "named_parameters", None)):
            raise RuntimeError("loaded InternVLA agent does not expose an auditable policy module")
        parameters = list(policy.named_parameters())
        buffers = list(policy.named_buffers())
        meta_parameters = [name for name, value in parameters if bool(value.is_meta)]
        meta_buffers = [name for name, value in buffers if bool(value.is_meta)]
        if meta_parameters or meta_buffers:
            raise RuntimeError(
                "loaded InternVLA policy retains meta tensors: "
                f"parameters={meta_parameters[:8]!r} buffers={meta_buffers[:8]!r}"
            )
        inventory = [
            f"P\t{name}\t{tuple(value.shape)}\t{value.dtype}\t{value.device.type}"
            for name, value in parameters
        ] + [
            f"B\t{name}\t{tuple(value.shape)}\t{value.dtype}\t{value.device.type}"
            for name, value in buffers
        ]
        audit = {
            "schema_version": 1,
            "status": "PASS",
            "backend": self.backend,
            "model_revision": MODEL_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "parameter_tensor_count": len(parameters),
            "buffer_tensor_count": len(buffers),
            "meta_parameter_count": 0,
            "meta_buffer_count": 0,
            "inventory_sha256": hashlib.sha256(
                "\n".join(inventory).encode("utf-8")
            ).hexdigest(),
        }
        result_root = os.environ.get("INTERNVLA_MODEL_RESULT_DIR")
        if result_root:
            audit_path = Path(result_root).resolve() / "model_weight_audit.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        self.get_logger().info(
            "InternVLA policy weight audit " + json.dumps(audit, sort_keys=True)
        )
        self._capture.install()
        return agent

    def _on_reset(self, request: Reset.Request, response: Reset.Response) -> Reset.Response:
        try:
            validate_protocol_version(request.protocol_version)
            with self._inference_lock:
                self._set_state(LIFECYCLE_RESETTING, STATUS_OK, "resetting", True)
                self._barrier.validate_reset(
                    request.next_episode_id,
                    int(request.expected_reset_generation),
                    int(request.reset_barrier_sequence_id),
                )
                self._agent.reset([0])
                self._system1_queue_remaining = 0
                generation = self._barrier.reset(
                    request.next_episode_id,
                    int(request.expected_reset_generation),
                    int(request.reset_barrier_sequence_id),
                )
                self._results.clear()
                self._clear_observations()
                self._set_state(LIFECYCLE_READY, STATUS_OK, "ready", False)
            response.status_code = STATUS_OK
            response.status_message = "reset complete"
            response.reset_generation = generation
        except ProtocolError as exc:
            response.status_code = exc.status_code
            response.status_message = str(exc)
            response.reset_generation = self._barrier.reset_generation
            if self._barrier.initialized:
                self._set_state(LIFECYCLE_READY, exc.status_code, str(exc), True)
        except BaseException as exc:
            response.status_code = STATUS_INTERNAL_ERROR
            response.status_message = repr(exc)
            response.reset_generation = self._barrier.reset_generation
            self._set_state(LIFECYCLE_ERROR, STATUS_INTERNAL_ERROR, repr(exc), True)
        response.episode_id = self._barrier.episode_id
        response.reset_barrier_sequence_id = max(0, self._barrier.last_sequence_id)
        return response

    def _on_shutdown(
        self, request: Shutdown.Request, response: Shutdown.Response
    ) -> Shutdown.Response:
        try:
            validate_protocol_version(request.protocol_version)
            self._shutting_down = True
            self._set_state(
                LIFECYCLE_SHUTTING_DOWN,
                STATUS_SHUTTING_DOWN,
                request.reason or "shutdown requested",
                True,
            )
            response.status_code = STATUS_OK
            response.status_message = "shutdown accepted"
            response.accepted = True
            threading.Timer(0.5, rclpy.shutdown).start()
        except ProtocolError as exc:
            response.status_code = exc.status_code
            response.status_message = str(exc)
            response.accepted = False
        return response

    def _on_goal(self, goal_request: Step.Goal) -> GoalResponse:
        if self._shutting_down:
            return GoalResponse.REJECT
        if len(goal_request.request_id) > 256 or len(goal_request.episode_id) > 256:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _t5_semantic_sim_ns(
        self, observed_ns: int, request_floor_ns: int = 0
    ) -> int:
        observed_ns = int(observed_ns)
        request_floor_ns = int(request_floor_ns)
        with self._t5_sim_time_lock:
            if observed_ns <= 0:
                raise ProtocolError(STATUS_TIMEOUT, "ROS simulation clock is zero")
            if observed_ns < self._t5_sim_time_high_water_ns:
                raise ProtocolError(
                    STATUS_STALE,
                    "ROS simulation clock regressed below persistent high-water: "
                    f"high_water_ns={self._t5_sim_time_high_water_ns} "
                    f"observed_ns={observed_ns}",
                )
            self._t5_sim_time_high_water_ns = observed_ns
        # DDS delivery can put the model's local /clock callback one tick
        # behind the request stamped by the client.  The request stamp is the
        # semantic floor for this request, while the persistent high-water
        # above deliberately tracks only real local /clock observations so a
        # genuine local rollback still fails closed.
        return max(observed_ns, request_floor_ns)

    def _on_cancel(self, _goal_handle: Any) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_step(self, goal_handle: Any) -> Step.Result:
        started = time.perf_counter()
        request = goal_handle.request
        identity = RequestIdentity(
            str(request.episode_id),
            int(request.reset_generation),
            int(request.sequence_id),
            str(request.request_id),
        )
        digest = str(request.observation_digest).lower()
        try:
            validate_protocol_version(request.protocol_version)
            identity.validate()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ProtocolError(STATUS_INVALID_REQUEST, "observation_digest must be lowercase SHA-256")
            sim_stamp_ns = _time_ns(request.sim_stamp)
            if (
                _time_ns(request.observation_stamp) != _time_ns(request.header.stamp)
                or sim_stamp_ns != _time_ns(request.header.stamp)
            ):
                raise ProtocolError(
                    STATUS_INVALID_REQUEST,
                    "header, observation, and explicit sim timestamps differ",
                )
            window = DualClockWindow(
                sim_stamp_ns,
                int(request.client_wall_monotonic_ns),
                _time_ns(request.deadline),
                _time_ns(request.valid_until),
            )
            receive_sim_ns = self.get_clock().now().nanoseconds
            if self.t5_sim_time_only:
                receive_sim_ns = self._t5_semantic_sim_ns(
                    receive_sim_ns, sim_stamp_ns
                )
                window.validate(receive_sim_ns, sim_time_only=True)
                local_deadline_ns = 0
                local_valid_until_ns = 0
            else:
                receive_monotonic_ns = time.monotonic_ns()
                window.validate(receive_sim_ns)
                local_deadline_ns, local_valid_until_ns = (
                    window.local_monotonic_limits(
                        receive_sim_ns, receive_monotonic_ns
                    )
                )

            # Every observation-cache epoch transition is performed while the
            # inference lock is held. Capture the request epoch before the
            # first barrier validation under that same reliable lock boundary,
            # so a reset between validation and observation wait is visible.
            with self._inference_lock:
                with self._state_lock:
                    with self._cache_condition:
                        request_epoch = self._observation_epoch
                    self._barrier.validate_current(identity)
                    cached = self._results.get(identity, digest)
                    if cached is not None:
                        goal_handle.succeed()
                        return self._result_from_payload(cached, replayed=True)
                    self._barrier.validate_new_sequence(identity)

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return self._error_result(identity, STATUS_CANCELED, "canceled before inference")

            stamp_ns = _time_ns(request.observation_stamp)
            observation_wait_ns = int(
                float(self.get_parameter("observation_wait_sec").value) * 1e9
            )
            wait_sim_ns = self.get_clock().now().nanoseconds
            if self.t5_sim_time_only:
                wait_sim_ns = self._t5_semantic_sim_ns(
                    wait_sim_ns, sim_stamp_ns
                )
                window.validate(
                    wait_sim_ns,
                    sim_time_only=True,
                    previous_sim_ns=receive_sim_ns,
                )
            rgb_message, depth_message, metadata, tuple_revisions, wait_sim_ns = (
                self._wait_for_observation(
                    stamp_ns=stamp_ns,
                    identity=identity,
                    expected_epoch=request_epoch,
                    wait_deadline_ns=min(
                        window.deadline_ns,
                        wait_sim_ns + observation_wait_ns,
                    ),
                    wait_monotonic_deadline_ns=(
                        time.monotonic_ns() + observation_wait_ns
                        if self.t5_sim_time_only
                        else min(
                            local_deadline_ns,
                            time.monotonic_ns() + observation_wait_ns,
                        )
                    ),
                    sim_time_floor_ns=(
                        wait_sim_ns if self.t5_sim_time_only else None
                    ),
                )
            )
            self._validate_metadata(metadata, identity, digest, window)
            try:
                if self.observation_transport == "compressed":
                    if not str(rgb_message.format).lower().startswith("jpeg"):
                        raise ObservationCodecError("RGB CompressedImage format is not JPEG")
                    if not str(depth_message.format).lower().startswith("png"):
                        raise ObservationCodecError("depth CompressedImage format is not PNG")
                    rgb_payload = bytes(rgb_message.data)
                    depth_payload = bytes(depth_message.data)
                    calculated_digest = compressed_observation_digest(
                        rgb_payload,
                        depth_payload,
                        metadata.instruction,
                        list(metadata.instruction_tokens),
                        list(metadata.global_gps),
                        list(metadata.global_rotation),
                    )
                    rgb = decode_rgb_jpeg(
                        rgb_payload, maximum_bytes=self.maximum_rgb_bytes
                    )
                    depth = decode_depth_png(
                        depth_payload, maximum_bytes=self.maximum_depth_bytes
                    )
                else:
                    rgb = _decode_rgb(rgb_message)
                    depth = _decode_depth(depth_message)
                    calculated_digest = observation_digest(
                        rgb,
                        depth,
                        metadata.instruction,
                        list(metadata.instruction_tokens),
                        list(metadata.global_gps),
                        list(metadata.global_rotation),
                    )
            except ObservationCodecError as exc:
                raise ProtocolError(STATUS_INVALID_REQUEST, str(exc)) from exc
            if calculated_digest != digest:
                raise ProtocolError(STATUS_INVALID_REQUEST, "observation digest mismatch")
            self._mark_observation_tuple_accepted(stamp_ns, tuple_revisions)

            feedback = Step.Feedback()
            feedback_now = self.get_clock().now()
            if self.t5_sim_time_only:
                feedback_sim_ns = self._t5_semantic_sim_ns(
                    feedback_now.nanoseconds, sim_stamp_ns
                )
                window.validate(
                    feedback_sim_ns,
                    sim_time_only=True,
                    previous_sim_ns=wait_sim_ns,
                )
                wait_sim_ns = feedback_sim_ns
            feedback.header.stamp = feedback_now.to_msg()
            feedback.sequence_id = identity.sequence_id
            feedback.request_id = identity.request_id
            feedback.stage = "inference"
            feedback.elapsed_sec = float(time.perf_counter() - started)
            goal_handle.publish_feedback(feedback)

            with self._inference_lock:
                with self._state_lock:
                    self._barrier.validate_current(identity)
                    cached = self._results.get(identity, digest)
                    if cached is not None:
                        goal_handle.succeed()
                        return self._result_from_payload(cached, replayed=True)
                    self._barrier.validate_new_sequence(identity)
                now_sim_ns = self.get_clock().now().nanoseconds
                if self.t5_sim_time_only:
                    now_sim_ns = self._t5_semantic_sim_ns(
                        now_sim_ns, sim_stamp_ns
                    )
                    window.validate(
                        now_sim_ns,
                        sim_time_only=True,
                        previous_sim_ns=wait_sim_ns,
                    )
                else:
                    now_monotonic_ns = time.monotonic_ns()
                    window.validate(now_sim_ns)
                    if window.deadline_expired(
                        now_sim_ns, now_monotonic_ns, local_deadline_ns
                    ):
                        raise ProtocolError(
                            STATUS_TIMEOUT, "local monotonic deadline expired"
                        )
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return self._error_result(identity, STATUS_CANCELED, "canceled before model call")

                observation = [
                    {
                        "rgb": rgb,
                        "depth": depth,
                        "instruction": metadata.instruction,
                        "instruction_tokens": list(metadata.instruction_tokens),
                        "globalgps": list(metadata.global_gps),
                        "globalrotation": list(metadata.global_rotation),
                    }
                ]
                inference_started = time.perf_counter()
                trajectory_rerank_seed: int | None = None
                if self.backend == "real":
                    if self._trajectory_rerank_enabled:
                        trajectory_rerank_seed = _trajectory_rerank_seed(identity)
                        _seed_trajectory_rerank_rngs(trajectory_rerank_seed)
                    self._capture.begin_step()
                raw_action = self._agent.step(observation)
                trajectory = (
                    self._capture.take()
                    if self.backend == "real"
                    else self._agent.last_trajectory
                )
                if self.backend == "real" and self._trajectory_rerank_enabled:
                    if trajectory_rerank_seed is None:
                        raise RuntimeError("trajectory rerank seed was not applied")
                    baseline_trajectory = trajectory
                    rerank = rerank_trajectories(
                        self._capture.take_candidates(),
                        depth,
                        self._trajectory_rerank_config,
                    )
                    if rerank.selected_trajectory is not None:
                        trajectory = rerank.selected_trajectory
                    self._record_trajectory_rerank(
                        identity=identity,
                        sim_stamp_ns=sim_stamp_ns,
                        baseline_trajectory=baseline_trajectory,
                        rerank=rerank,
                        seed=trajectory_rerank_seed,
                    )
                latency = float(time.perf_counter() - inference_started)
                action = _extract_action(raw_action)
                action_source = self._classify_action_source(trajectory)
                if (
                    self._system1_queue_horizon == 1
                    and action_source == Step.Result.ACTION_SOURCE_SYSTEM1_NEW
                ):
                    self._discard_stale_system1_action_queue(identity)
                if (
                    self._system2_queue_horizon == 1
                    and action_source == Step.Result.ACTION_SOURCE_SYSTEM2
                ):
                    self._discard_stale_system2_action_queue(identity)
                now_ns = self.get_clock().now().nanoseconds
                now_monotonic_ns = (
                    0 if self.t5_sim_time_only else time.monotonic_ns()
                )
                if goal_handle.is_cancel_requested:
                    self._abort_generation_locked("canceled after model mutation")
                    goal_handle.canceled()
                    return self._error_result(
                        identity,
                        STATUS_CANCELED,
                        "canceled; generation invalidated and model reset",
                    )
                if self.t5_sim_time_only:
                    try:
                        now_ns = self._t5_semantic_sim_ns(
                            now_ns, sim_stamp_ns
                        )
                        window.validate_sim_progress(now_ns, now_sim_ns)
                    except ProtocolError:
                        self._abort_generation_locked(
                            "simulation-time contract failed after model mutation"
                        )
                        raise
                if window.deadline_expired(
                    now_ns,
                    now_monotonic_ns,
                    local_deadline_ns,
                    sim_time_only=self.t5_sim_time_only,
                ):
                    self._abort_generation_locked(
                        "deadline expired after model mutation"
                    )
                    goal_handle.abort()
                    return self._error_result(
                        identity,
                        STATUS_TIMEOUT,
                        "deadline expired; generation invalidated and model reset",
                    )
                if window.response_is_stale(
                    now_ns,
                    now_monotonic_ns,
                    local_valid_until_ns,
                    sim_time_only=self.t5_sim_time_only,
                ):
                    self._abort_generation_locked("response validity expired")
                    goal_handle.abort()
                    return self._error_result(
                        identity,
                        STATUS_STALE,
                        "response stale; generation invalidated and model reset",
                    )

                payload = StepPayload(
                    identity,
                    action,
                    action_source,
                    trajectory,
                    window.valid_until_ns,
                    latency,
                )
                with self._state_lock:
                    self._barrier.commit(identity)
                    self._results.put(identity, digest, payload)
                self._set_state(LIFECYCLE_READY, STATUS_OK, "ready", False)
                goal_handle.succeed()
                return self._result_from_payload(payload, replayed=False)
        except ProtocolError as exc:
            goal_handle.abort()
            self._set_state(
                self._lifecycle,
                exc.status_code,
                str(exc),
                exc.status_code != STATUS_OK,
            )
            return self._error_result(identity, exc.status_code, str(exc))
        except BaseException as exc:
            with self._inference_lock:
                if self._barrier.initialized and self._agent is not None:
                    self._abort_generation_locked("internal error after possible model mutation")
            goal_handle.abort()
            self._set_state(LIFECYCLE_ERROR, STATUS_INTERNAL_ERROR, repr(exc), True)
            self.get_logger().error(f"step failed: {exc!r}")
            return self._error_result(identity, STATUS_INTERNAL_ERROR, repr(exc))

    def _record_trajectory_rerank(
        self,
        *,
        identity: RequestIdentity,
        sim_stamp_ns: int,
        baseline_trajectory: np.ndarray | None,
        rerank: Any,
        seed: int,
    ) -> None:
        path = self._trajectory_rerank_audit_path
        if path is None:
            return
        baseline = (
            None
            if baseline_trajectory is None
            else np.ascontiguousarray(baseline_trajectory, dtype=np.float32)
        )
        record = {
            **rerank.audit_mapping(),
            "episode_id": identity.episode_id,
            "reset_generation": identity.reset_generation,
            "sequence_id": identity.sequence_id,
            "request_id": identity.request_id,
            "sim_stamp_ns": int(sim_stamp_ns),
            "seed": int(seed),
            "seed_contract": _TRAJECTORY_RERANK_SEED_CONTRACT,
            "enabled_scope": "exact_t5_isaac_completion_sim",
            "baseline_trajectory_shape": (
                None if baseline is None else list(baseline.shape)
            ),
            "baseline_trajectory_sha256": (
                None
                if baseline is None
                else hashlib.sha256(baseline.tobytes(order="C")).hexdigest()
            ),
            "selected_or_fallback": (
                "selected_candidate"
                if rerank.selected_trajectory is not None
                else "upstream_mean_fallback"
            ),
            "config": trajectory_rerank_config_mapping(
                self._trajectory_rerank_config
            ),
            "wall_time_unix": time.time(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._trajectory_rerank_audit_lock:
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                )

    def _wait_for_observation(
        self,
        stamp_ns: int,
        identity: RequestIdentity,
        expected_epoch: int,
        wait_deadline_ns: int,
        wait_monotonic_deadline_ns: int,
        sim_time_floor_ns: int | None = None,
    ) -> tuple[Any, Any, ObservationMetadata, _AcceptedObservationTuple, int]:
        stale_tuple_baseline: _AcceptedObservationTuple | None = None
        last_sim_ns = 0 if sim_time_floor_ns is None else int(sim_time_floor_ns)
        with self._cache_condition:
            if self._observation_epoch != expected_epoch:
                raise ProtocolError(
                    STATUS_STALE,
                    "observation cache epoch changed before wait: "
                    f"expected_epoch={expected_epoch} "
                    f"current_epoch={self._observation_epoch} "
                    f"request_identity={describe_request_identity(identity)}",
                )
            while True:
                if self._observation_epoch != expected_epoch:
                    raise ProtocolError(
                        STATUS_STALE,
                        "observation cache epoch changed while waiting: "
                        f"expected_epoch={expected_epoch} "
                        f"current_epoch={self._observation_epoch} "
                        f"request_identity={describe_request_identity(identity)}",
                    )
                if sim_time_floor_ns is not None:
                    observed_sim_ns = self._t5_semantic_sim_ns(
                        self.get_clock().now().nanoseconds,
                        sim_time_floor_ns,
                    )
                    if observed_sim_ns <= 0:
                        raise ProtocolError(
                            STATUS_TIMEOUT, "ROS simulation clock is zero while waiting"
                        )
                    if observed_sim_ns < last_sim_ns:
                        raise ProtocolError(
                            STATUS_STALE,
                            "ROS simulation clock regressed while waiting: "
                            f"previous_ns={last_sim_ns} "
                            f"observed_ns={observed_sim_ns}",
                        )
                    last_sim_ns = observed_sim_ns
                    if last_sim_ns > wait_deadline_ns:
                        raise ProtocolError(
                            STATUS_OBSERVATION_MISSING,
                            "ROS simulation-time observation deadline expired",
                        )
                stale_identity: RequestIdentity | None = None
                rgb_entry = self._rgb_cache.get(stamp_ns)
                depth_entry = self._depth_cache.get(stamp_ns)
                metadata_entry = self._metadata_cache.get(stamp_ns)
                if metadata_entry is not None:
                    metadata = metadata_entry.message
                    observed_identity = _metadata_identity(metadata)
                    if observed_identity != identity:
                        stale_identity = observed_identity
                        baseline = stale_tuple_baseline
                        if baseline is None:
                            baseline = self._accepted_observation_tuples.get(stamp_ns)
                        # Keep raising the conservative baseline while stale
                        # metadata remains current, so delayed/duplicate stale
                        # components cannot masquerade as a fresh tuple.
                        stale_tuple_baseline = _AcceptedObservationTuple(
                            identity=observed_identity,
                            rgb_revision=max(
                                0 if baseline is None else baseline.rgb_revision,
                                0 if rgb_entry is None else rgb_entry.revision,
                            ),
                            depth_revision=max(
                                0 if baseline is None else baseline.depth_revision,
                                0 if depth_entry is None else depth_entry.revision,
                            ),
                            metadata_revision=max(
                                0 if baseline is None else baseline.metadata_revision,
                                metadata_entry.revision,
                            ),
                            epoch=expected_epoch,
                        )
                    elif rgb_entry is not None and depth_entry is not None:
                        tuple_revisions = _AcceptedObservationTuple(
                            identity=identity,
                            rgb_revision=rgb_entry.revision,
                            depth_revision=depth_entry.revision,
                            metadata_revision=metadata_entry.revision,
                            epoch=expected_epoch,
                        )
                        accepted = (
                            stale_tuple_baseline
                            if stale_tuple_baseline is not None
                            else self._accepted_observation_tuples.get(stamp_ns)
                        )
                        if accepted is None or accepted.identity == identity or (
                            tuple_revisions.rgb_revision > accepted.rgb_revision
                            and tuple_revisions.depth_revision > accepted.depth_revision
                            and tuple_revisions.metadata_revision
                            > accepted.metadata_revision
                        ):
                            if self._observation_epoch != expected_epoch:
                                raise ProtocolError(
                                    STATUS_STALE,
                                    "observation cache epoch changed before tuple return",
                                )
                            return (
                                rgb_entry.message,
                                depth_entry.message,
                                metadata,
                                tuple_revisions,
                                last_sim_ns,
                            )
                # A prior request may have used the same stamp while the
                # simulator clock was paused. Wait for the matching metadata
                # callback instead of consuming that stale tuple.
                current_sim_ns = (
                    last_sim_ns
                    if sim_time_floor_ns is not None
                    else self.get_clock().now().nanoseconds
                )
                remaining_ns = min(
                    wait_deadline_ns - current_sim_ns,
                    wait_monotonic_deadline_ns - time.monotonic_ns(),
                )
                if remaining_ns <= 0:
                    if stale_identity is not None:
                        raise ProtocolError(
                            STATUS_STALE,
                            _identity_mismatch_message(identity, stale_identity),
                        )
                    raise ProtocolError(
                        STATUS_OBSERVATION_MISSING,
                        "timed out waiting for RGB/depth/metadata tuple",
                    )
                self._cache_condition.wait(min(0.1, remaining_ns / 1e9))

    def _mark_observation_tuple_accepted(
        self, stamp_ns: int, accepted: _AcceptedObservationTuple
    ) -> None:
        with self._cache_condition:
            if accepted.epoch != self._observation_epoch:
                raise ProtocolError(
                    STATUS_STALE,
                    "observation cache epoch changed before acceptance: "
                    f"candidate_epoch={accepted.epoch} "
                    f"current_epoch={self._observation_epoch} "
                    f"candidate_identity={describe_request_identity(accepted.identity)}",
                )
            current = self._accepted_observation_tuples.get(stamp_ns)
            if current is not None and current.epoch != self._observation_epoch:
                raise ProtocolError(
                    STATUS_STALE,
                    "accepted observation baseline has a stale cache epoch",
                )
            if current is None:
                merged = accepted
            elif current.identity == accepted.identity:
                merged = _AcceptedObservationTuple(
                    identity=accepted.identity,
                    rgb_revision=max(current.rgb_revision, accepted.rgb_revision),
                    depth_revision=max(
                        current.depth_revision, accepted.depth_revision
                    ),
                    metadata_revision=max(
                        current.metadata_revision, accepted.metadata_revision
                    ),
                    epoch=self._observation_epoch,
                )
            elif (
                accepted.rgb_revision > current.rgb_revision
                and accepted.depth_revision > current.depth_revision
                and accepted.metadata_revision > current.metadata_revision
            ):
                merged = accepted
            else:
                raise ProtocolError(
                    STATUS_STALE,
                    "refusing observation acceptance revision regression: "
                    f"current_identity={describe_request_identity(current.identity)} "
                    f"current_revisions=({current.rgb_revision},"
                    f"{current.depth_revision},{current.metadata_revision}) "
                    f"candidate_identity={describe_request_identity(accepted.identity)} "
                    f"candidate_revisions=({accepted.rgb_revision},"
                    f"{accepted.depth_revision},{accepted.metadata_revision})",
                )
            self._accepted_observation_tuples[stamp_ns] = merged
            self._accepted_observation_tuples.move_to_end(stamp_ns)
            while len(self._accepted_observation_tuples) > self._observation_cache_size:
                self._accepted_observation_tuples.popitem(last=False)

    def _validate_metadata(
        self,
        metadata: ObservationMetadata,
        identity: RequestIdentity,
        digest: str,
        window: DualClockWindow,
    ) -> None:
        validate_protocol_version(metadata.protocol_version)
        observed_identity = _metadata_identity(metadata)
        if observed_identity != identity:
            raise ProtocolError(
                STATUS_STALE,
                _identity_mismatch_message(identity, observed_identity),
            )
        if metadata.observation_digest.lower() != digest:
            raise ProtocolError(STATUS_INVALID_REQUEST, "metadata observation digest mismatch")
        if (
            int(metadata.client_wall_monotonic_ns)
            != window.client_wall_monotonic_ns
            or _time_ns(metadata.sim_stamp) != window.sim_stamp_ns
            or _time_ns(metadata.header.stamp) != window.sim_stamp_ns
            or _time_ns(metadata.deadline) != window.deadline_ns
            or _time_ns(metadata.valid_until) != window.valid_until_ns
        ):
            raise ProtocolError(
                STATUS_INVALID_REQUEST,
                "metadata dual-clock timestamp/deadline mismatch",
            )

    def _abort_generation_locked(self, reason: str) -> None:
        self._agent.reset([0])
        self._system1_queue_remaining = 0
        self._barrier.abort_generation()
        self._results.clear()
        self._clear_observations()
        self._set_state(LIFECYCLE_READY, STATUS_STALE, reason, True)

    def _clear_observations(self) -> None:
        with self._cache_condition:
            self._rgb_cache.clear()
            self._depth_cache.clear()
            self._metadata_cache.clear()
            self._accepted_observation_tuples.clear()
            self._observation_epoch += 1
            self._cache_condition.notify_all()

    def _result_from_payload(self, payload: StepPayload, replayed: bool) -> Step.Result:
        result = Step.Result()
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = "base_link"
        result.status_code = STATUS_OK
        result.status_message = "idempotent replay" if replayed else "ok"
        result.episode_id = payload.identity.episode_id
        result.reset_generation = payload.identity.reset_generation
        result.sequence_id = payload.identity.sequence_id
        result.request_id = payload.identity.request_id
        result.discrete_action = payload.action
        result.stop = payload.action == 0
        result.replayed = replayed
        result.action_source = payload.action_source
        result.trajectory_source = (
            Step.Result.TRAJECTORY_SYSTEM1_NEW
            if payload.trajectory is not None
            else Step.Result.TRAJECTORY_NONE
        )
        result.trajectory_valid = payload.trajectory is not None
        result.local_path = self._trajectory_path(payload.trajectory, result.header.stamp)
        _assign_time(result.valid_until, payload.valid_until_ns)
        result.inference_latency_sec = float(payload.inference_latency_sec)
        return result

    def _error_result(self, identity: RequestIdentity, status_code: int, message: str) -> Step.Result:
        result = Step.Result()
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = "base_link"
        result.status_code = int(status_code)
        result.status_message = message[:1024]
        result.episode_id = self._barrier.episode_id or identity.episode_id
        result.reset_generation = self._barrier.reset_generation
        result.sequence_id = identity.sequence_id
        result.request_id = identity.request_id
        result.discrete_action = 0
        result.stop = True
        result.replayed = False
        result.action_source = Step.Result.ACTION_SOURCE_UNKNOWN
        result.trajectory_source = Step.Result.TRAJECTORY_NONE
        result.trajectory_valid = False
        result.local_path = self._trajectory_path(None, result.header.stamp)
        result.valid_until = result.header.stamp
        result.inference_latency_sec = 0.0
        return result

    def _classify_action_source(self, trajectory: np.ndarray | None) -> int:
        """Label provenance without changing the upstream agent's queues."""
        if self.backend != "real":
            return Step.Result.ACTION_SOURCE_UNKNOWN
        if trajectory is not None:
            queued = getattr(getattr(self._agent, "s2_output", None), "output_action", None)
            self._system1_queue_remaining = len(queued) if queued is not None else 0
            return Step.Result.ACTION_SOURCE_SYSTEM1_NEW
        if self._system1_queue_remaining > 0:
            self._system1_queue_remaining -= 1
            return Step.Result.ACTION_SOURCE_SYSTEM1_QUEUE
        return Step.Result.ACTION_SOURCE_SYSTEM2

    def _discard_stale_system2_action_queue(
        self, identity: RequestIdentity
    ) -> None:
        """Keep only the just-returned System2 primitive in completion_sim.

        The current action has already been copied out of the upstream agent.
        Remaining queued primitives were planned from the old observation, so
        horizon=1 drops only that consumable queue while preserving the latent
        state and policy history for the next fresh-observation replan.
        """

        output = getattr(self._agent, "s2_output", None)
        output_lock = getattr(self._agent, "s2_output_lock", None)
        if output is None or output_lock is None:
            raise RuntimeError(
                "System2 receding horizon requires the upstream action queue"
            )
        with output_lock:
            queued = getattr(output, "output_action", None)
            discarded = len(queued) if isinstance(queued, (list, tuple)) else 0
            output.output_action = None
        self._system1_queue_remaining = 0
        self.get_logger().info(
            "T5 System2 receding horizon: "
            f"episode={identity.episode_id} reset={identity.reset_generation} "
            f"sequence={identity.sequence_id} discarded={discarded}"
        )

    def _discard_stale_system1_action_queue(
        self, identity: RequestIdentity
    ) -> None:
        """Keep only the first primitive from a new System1 trajectory.

        The current primitive has already been copied out of the upstream
        agent.  Remaining primitives were discretized before that motion was
        measured, so horizon=1 drops only the consumable queue.  The System1
        latent, pixel state, and policy history remain available for a fresh
        observation replan.
        """

        output = getattr(self._agent, "s2_output", None)
        output_lock = getattr(self._agent, "s2_output_lock", None)
        if output is None or output_lock is None:
            raise RuntimeError(
                "System1 receding horizon requires the upstream action queue"
            )
        with output_lock:
            queued = getattr(output, "output_action", None)
            discarded = len(queued) if isinstance(queued, (list, tuple)) else 0
            output.output_action = None
        self._system1_queue_remaining = 0
        self.get_logger().info(
            "T5 System1 receding horizon: "
            f"episode={identity.episode_id} reset={identity.reset_generation} "
            f"sequence={identity.sequence_id} discarded={discarded}"
        )

    @staticmethod
    def _trajectory_path(trajectory: np.ndarray | None, stamp: Any) -> NavPath:
        path = NavPath()
        path.header.stamp = stamp
        path.header.frame_id = "base_link"
        if trajectory is None:
            return path
        points = np.asarray(trajectory, dtype=np.float32)
        for index, point in enumerate(points):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0
            if len(points) == 1:
                yaw = 0.0
            elif index + 1 < len(points):
                delta = points[index + 1] - point
                yaw = math.atan2(float(delta[1]), float(delta[0]))
            else:
                delta = point - points[index - 1]
                yaw = math.atan2(float(delta[1]), float(delta[0]))
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        return path

    def _set_state(self, lifecycle: int, status_code: int, message: str, safe_stop: bool) -> None:
        with self._state_lock:
            self._lifecycle = int(lifecycle)
            self._status_code = int(status_code)
            self._status_message = str(message)[:1024]
            self._safe_stop = bool(safe_stop)
        self._publish_state()

    def _publish_state(self) -> None:
        if not hasattr(self, "_state_publisher"):
            return
        message = ModelState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.protocol_version = 1
        message.status_code = self._status_code
        message.status_message = self._status_message
        message.lifecycle_state = self._lifecycle
        message.initialized = self._barrier.initialized
        message.episode_id = self._barrier.episode_id
        message.reset_generation = self._barrier.reset_generation
        message.last_sequence_id = max(0, self._barrier.last_sequence_id)
        message.safe_stop = self._safe_stop
        message.model_revision = MODEL_REVISION
        message.checkpoint_revision = CHECKPOINT_REVISION
        self._state_publisher.publish(message)

    def destroy_node(self) -> bool:
        self._action_server.destroy()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = InternVLAModelNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
