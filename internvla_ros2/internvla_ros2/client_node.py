"""Isaac-side ROS 2 client node and safe local shared-memory bridge."""

from __future__ import annotations

import json
import base64
import os
import re
import socket
import struct
import threading
import time
import uuid
import zlib
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import resource_tracker
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from internvla_ros2_msgs.action import Step
from geometry_msgs.msg import TransformStamped
from internvla_ros2_msgs.msg import ModelState, NavigationCommand, ObservationMetadata
from internvla_ros2_msgs.srv import Health, Initialize, Reset, ResolveCommand, Shutdown
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Int8
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from .identity import CHECKPOINT_REVISION, MODEL_REVISION
from .model_health import validate_uninitialized_model_health
from .compressed_observation import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_DEPTH_BYTES,
    DEFAULT_MAX_RGB_BYTES,
    DEFAULT_PNG_COMPRESSION,
    ObservationCodecError,
    encode_depth_png,
    encode_rgb_jpeg,
)
from .observation import compressed_observation_digest, observation_digest
from .observation_guard import ObservationStampGate, describe_request_identity
from .protocol import (
    ACTION_STOP,
    PROTOCOL_VERSION,
    RequestIdentity,
    STATUS_CANCELED,
    STATUS_INTERNAL_ERROR,
    STATUS_INVALID_REQUEST,
    STATUS_OBSERVATION_MISSING,
    STATUS_OK,
    STATUS_STALE,
    STATUS_TIMEOUT,
)


MAX_IPC_MESSAGE_BYTES = 4 * 1024 * 1024
MAX_INLINE_COMPRESSED_BYTES = 3 * 1024 * 1024
SHM_NAME_RE = re.compile(r"^/?[A-Za-z0-9_.-]{1,128}$")
T5_STEP_GOAL_ACCEPTANCE_LIVENESS_SEC = 20.0
T5_STEP_RESULT_PRIMARY_LIVENESS_SEC = 60.0
T5_STEP_RESULT_REFETCH_LIVENESS_SEC = 60.0
T5_STEP_ACTION_CLIENT_MAX_COMPLETED_GOALS = 16


class ClientFailure(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = int(status_code)


def _time_ns(message: Any) -> int:
    return int(message.sec) * 1_000_000_000 + int(message.nanosec)


def _assign_time(message: Any, value_ns: int) -> None:
    message.sec = int(value_ns // 1_000_000_000)
    message.nanosec = int(value_ns % 1_000_000_000)


def _future_result(future: Any, timeout_sec: float, description: str) -> Any:
    event = threading.Event()
    future.add_done_callback(lambda _future: event.set())
    if not event.wait(timeout_sec):
        raise ClientFailure(STATUS_TIMEOUT, f"timeout waiting for {description}")
    exception = future.exception()
    if exception is not None:
        raise ClientFailure(STATUS_INTERNAL_ERROR, f"{description} failed: {exception!r}")
    return future.result()


class InternVLAClientNode(Node):
    def __init__(self) -> None:
        super().__init__("internvla_client_node")
        self.declare_parameter("service_timeout_sec", 300.0)
        self.declare_parameter("step_deadline_sec", 10.0)
        self.declare_parameter("step_validity_sec", 12.0)
        self.declare_parameter("discovery_timeout_sec", 15.0)
        self.declare_parameter("ipc_idle_timeout_sec", 300.0)
        self.declare_parameter("control_mode", "model")
        self.declare_parameter("nav2_resolution_timeout_sec", 4.0)
        self.declare_parameter("publish_observation_pose", True)
        self.declare_parameter("observation_transport", "raw")
        self.declare_parameter("rgb_jpeg_quality", DEFAULT_JPEG_QUALITY)
        self.declare_parameter("depth_png_compression", DEFAULT_PNG_COMPRESSION)
        self.declare_parameter("maximum_rgb_bytes", DEFAULT_MAX_RGB_BYTES)
        self.declare_parameter("maximum_depth_bytes", DEFAULT_MAX_DEPTH_BYTES)
        self.declare_parameter("observation_stamp_wait_sec", 1.0)
        self.declare_parameter("result_dir", "")
        self.service_timeout_sec = float(self.get_parameter("service_timeout_sec").value)
        self.step_deadline_sec = float(self.get_parameter("step_deadline_sec").value)
        self.step_validity_sec = float(self.get_parameter("step_validity_sec").value)
        self.ipc_idle_timeout_sec = float(self.get_parameter("ipc_idle_timeout_sec").value)
        self.control_mode = str(self.get_parameter("control_mode").value)
        self.nav2_resolution_timeout_sec = float(
            self.get_parameter("nav2_resolution_timeout_sec").value
        )
        self.publish_observation_pose = bool(
            self.get_parameter("publish_observation_pose").value
        )
        self.observation_transport = str(
            self.get_parameter("observation_transport").value
        )
        self.rgb_jpeg_quality = int(self.get_parameter("rgb_jpeg_quality").value)
        self.depth_png_compression = int(
            self.get_parameter("depth_png_compression").value
        )
        self.maximum_rgb_bytes = int(self.get_parameter("maximum_rgb_bytes").value)
        self.maximum_depth_bytes = int(
            self.get_parameter("maximum_depth_bytes").value
        )
        self.observation_stamp_wait_sec = float(
            self.get_parameter("observation_stamp_wait_sec").value
        )
        result_dir = str(self.get_parameter("result_dir").value)
        self.result_dir = Path(result_dir).resolve() if result_dir else None
        self.records_path = self.result_dir / "client_records.jsonl" if self.result_dir else None
        self.client_started_unix = time.time()
        self.client_step_count = 0
        self.inference_latency_sum = 0.0
        self.inference_latency_max = 0.0
        self.nav2_latency_sum = 0.0
        self.nav2_latency_max = 0.0
        self.encode_latency_sum = 0.0
        self.encode_latency_max = 0.0
        self.action_round_trip_latency_sum = 0.0
        self.action_round_trip_latency_max = 0.0
        self.observation_wire_bytes_sum = 0
        self.reset_barrier_reconcile_count = 0
        self.step_result_refetch_count = 0
        self.observation_tuple_republish_count = 0
        self.identity_prefix = os.environ.get("INTERNNAV_T5_ID_PREFIX", "")
        if self.identity_prefix not in {"", "a::", "b::"}:
            raise RuntimeError("INTERNNAV_T5_ID_PREFIX must be empty, a::, or b::")
        self.t5_sim_time_only = (
            os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
            and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
            and os.environ.get("INTERNNAV_T5_LANE", "") in {"a", "b"}
        )
        if self.t5_sim_time_only and not bool(
            self.get_parameter("use_sim_time").value
        ):
            raise RuntimeError("T5 completion_sim requires use_sim_time=true")
        self._t5_sim_time_lock = threading.Lock()
        self._t5_sim_time_high_water_ns = 0
        self.allow_reset_barrier_reconcile = (
            os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
            and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
        )
        self.model_health_start: dict[str, Any] | None = None
        # The construction-time summary includes the lane-qualified reset ID.
        # Initialize its state before any result artifact can be written.
        self.reset_generation = 0
        if self.result_dir is not None:
            self.result_dir.mkdir(parents=True, exist_ok=True)
            if self.records_path is not None and self.records_path.exists():
                raise RuntimeError("refusing to append existing client records")
            self._write_client_summary("READY")
        if self.control_mode not in {"model", "nav2"}:
            raise RuntimeError("control_mode must be model or nav2")
        if self.observation_transport not in {"raw", "compressed"}:
            raise RuntimeError("observation_transport must be raw or compressed")
        if self.step_validity_sec < self.step_deadline_sec:
            raise RuntimeError("step_validity_sec must be >= step_deadline_sec")
        if (
            not np.isfinite(self.observation_stamp_wait_sec)
            or self.observation_stamp_wait_sec <= 0.0
            or self.observation_stamp_wait_sec > 5.0
        ):
            raise RuntimeError(
                "observation_stamp_wait_sec must be finite and inside (0, 5]"
            )

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
        if self.observation_transport == "compressed":
            self.rgb_publisher = self.create_publisher(
                CompressedImage, "/internvla/observation/rgb/compressed", image_qos
            )
            self.depth_publisher = self.create_publisher(
                CompressedImage, "/internvla/observation/depth/compressed", image_qos
            )
        else:
            self.rgb_publisher = self.create_publisher(
                Image, "/internvla/observation/rgb", image_qos
            )
            self.depth_publisher = self.create_publisher(
                Image, "/internvla/observation/depth", image_qos
            )
        self.metadata_publisher = self.create_publisher(
            ObservationMetadata, "/internvla/observation/metadata", image_qos
        )
        self.state_publisher = self.create_publisher(ModelState, "/internvla/client_state", state_qos)
        self.path_publisher = self.create_publisher(NavPath, "/internvla/local_path", image_qos)
        self.action_publisher = self.create_publisher(Int8, "/internvla/discrete_action", image_qos)
        self.stop_publisher = self.create_publisher(Bool, "/internvla/stop", image_qos)
        self.navigation_command_publisher = self.create_publisher(
            NavigationCommand, "/internvla/navigation_command", image_qos
        )
        self.odom_publisher = self.create_publisher(Odometry, "/odom", image_qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = self.get_clock().now().to_msg()
        map_to_odom.header.frame_id = "map"
        map_to_odom.child_frame_id = "odom"
        map_to_odom.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(map_to_odom)
        self.health_client = self.create_client(Health, "/internvla/health")
        self.initialize_client = self.create_client(Initialize, "/internvla/initialize")
        self.reset_client = self.create_client(Reset, "/internvla/reset")
        self.shutdown_client = self.create_client(Shutdown, "/internvla/shutdown")
        self.resolve_client = self.create_client(
            ResolveCommand, "/internvla/nav2_resolve"
        )
        # The T4 odometry callbacks can wait for ``operation_lock`` while an
        # IPC step owns it.  Keep Step responses out of that default callback
        # group so the action future can still settle and release the lock.
        self._step_action_callback_group = MutuallyExclusiveCallbackGroup()
        self.step_client = self._new_step_action_client()
        # A completed action result is the only safe boundary at which to
        # retire the client waitable.  Keeping the lifetime below the first
        # observed fixed-five transport stall prevents a lost result future
        # from poisoning later goal acceptance without replaying any goal.
        self.step_action_client_completed_goals = 0
        self.step_action_client_rotation_count = 0
        self.step_action_client_transport_poisoned = False

        self.operation_lock = threading.RLock()
        self._observation_stamp_gate = ObservationStampGate()
        self.current_goal_lock = threading.Lock()
        self.current_goal: Any = None
        self.initialized = False
        self.episode_id = ""
        self.next_sequence_id = 0
        self.last_committed_sequence = -1
        self.safe_stop = True
        self.last_status_code = STATUS_OK
        self.last_status_message = "uninitialized"
        self._publish_state()

    def wait_for_graph(self) -> None:
        timeout = float(self.get_parameter("discovery_timeout_sec").value)
        deadline = time.monotonic() + timeout
        clients = (self.health_client, self.initialize_client, self.reset_client)
        while time.monotonic() < deadline:
            resolver_ready = self.control_mode != "nav2" or self.resolve_client.service_is_ready()
            if (
                all(client.service_is_ready() for client in clients)
                and self.step_client.server_is_ready()
                and resolver_ready
            ):
                return
            for client in clients:
                client.wait_for_service(timeout_sec=0.1)
            self.step_client.wait_for_server(timeout_sec=0.1)
            if self.control_mode == "nav2":
                self.resolve_client.wait_for_service(timeout_sec=0.1)
        raise ClientFailure(STATUS_TIMEOUT, "typed InternVLA graph discovery timed out")

    def health(self, timeout_sec: float = 10.0) -> dict[str, Any]:
        request = Health.Request()
        request.protocol_version = PROTOCOL_VERSION
        response = _future_result(self.health_client.call_async(request), timeout_sec, "health")
        return {
            "status_code": int(response.status_code),
            "status_message": response.status_message,
            "initialized": bool(response.initialized),
            "lifecycle_state": int(response.lifecycle_state),
            "episode_id": response.episode_id,
            "reset_generation": int(response.reset_generation),
            "last_sequence_id": int(response.last_sequence_id),
            "model_revision": response.model_revision,
            "checkpoint_revision": response.checkpoint_revision,
        }

    def initialize(self, episode_id: str) -> dict[str, Any]:
        with self.operation_lock:
            request = Initialize.Request()
            request.protocol_version = PROTOCOL_VERSION
            request.episode_id = str(episode_id)
            request.model_revision = MODEL_REVISION
            request.checkpoint_revision = CHECKPOINT_REVISION
            response = _future_result(
                self.initialize_client.call_async(request),
                self.service_timeout_sec,
                "initialize",
            )
            if int(response.status_code) != STATUS_OK or not response.initialized:
                self._safe_stop(int(response.status_code), response.status_message)
                raise ClientFailure(int(response.status_code), response.status_message)
            self.initialized = True
            self.episode_id = response.episode_id
            self.reset_generation = int(response.reset_generation)
            self.next_sequence_id = 0
            self.last_committed_sequence = -1
            self._clear_safe_stop("ready")
            if (
                os.environ.get("INTERNNAV_T5_FAULT_INJECTION_PROFILE", "off")
                == "completion_sim_minimal_v1"
            ):
                self._write_client_summary("RUNNING")
            return {
                "status_code": int(response.status_code),
                "episode_id": self.episode_id,
                "reset_generation": self.reset_generation,
            }

    def reset(self, next_episode_id: str) -> dict[str, Any]:
        with self.operation_lock:
            self.cancel_current()
            request = Reset.Request()
            request.protocol_version = PROTOCOL_VERSION
            request.next_episode_id = str(next_episode_id)
            request.expected_reset_generation = self.reset_generation
            request.reset_barrier_sequence_id = max(0, self.last_committed_sequence)
            response = _future_result(
                self.reset_client.call_async(request),
                self.service_timeout_sec,
                "reset",
            )
            local_barrier = int(request.reset_barrier_sequence_id)
            remote_barrier = int(response.reset_barrier_sequence_id)
            if (
                self.allow_reset_barrier_reconcile
                and int(response.status_code) == STATUS_STALE
                and response.status_message
                == "reset barrier is behind the committed sequence"
                and str(response.episode_id) == self.episode_id
                and int(response.reset_generation) == self.reset_generation
                and remote_barrier == local_barrier + 1
            ):
                request.reset_barrier_sequence_id = remote_barrier
                response = _future_result(
                    self.reset_client.call_async(request),
                    self.service_timeout_sec,
                    "bounded completion reset reconciliation",
                )
                if int(response.status_code) == STATUS_OK:
                    self.reset_barrier_reconcile_count += 1
            if int(response.status_code) != STATUS_OK:
                self._safe_stop(int(response.status_code), response.status_message)
                raise ClientFailure(int(response.status_code), response.status_message)
            self.episode_id = response.episode_id
            self.reset_generation = int(response.reset_generation)
            self.next_sequence_id = 0
            self.last_committed_sequence = -1
            self._clear_safe_stop("reset complete")
            if (
                os.environ.get("INTERNNAV_T5_FAULT_INJECTION_PROFILE", "off")
                == "completion_sim_minimal_v1"
            ):
                self._write_client_summary("RUNNING")
            return {
                "status_code": int(response.status_code),
                "episode_id": self.episode_id,
                "reset_generation": self.reset_generation,
            }

    def _publish_observation_tuple(
        self,
        rgb_message: Any,
        depth_message: Any,
        metadata: ObservationMetadata,
    ) -> None:
        self.rgb_publisher.publish(rgb_message)
        self.depth_publisher.publish(depth_message)
        self.metadata_publisher.publish(metadata)

    def _wait_for_accepted_step_result(
        self,
        goal_handle: Any,
        fallback_timeout_sec: float,
    ) -> tuple[Any, int]:
        if not self.t5_sim_time_only:
            return (
                _future_result(
                    goal_handle.get_result_async(),
                    fallback_timeout_sec,
                    "step result",
                ),
                0,
            )
        result_future = goal_handle.get_result_async()
        try:
            return (
                _future_result(
                    result_future,
                    T5_STEP_RESULT_PRIMARY_LIVENESS_SEC,
                    "step result",
                ),
                0,
            )
        except ClientFailure as exc:
            if exc.status_code != STATUS_TIMEOUT:
                raise
            # Continue waiting on the original result request.  Issuing a
            # second GetResult while the first future is still pending leaves
            # two response futures on the same transport and does not repair a
            # callback-group stall.
            self.get_logger().warning(
                "primary Step result wait failed; bounded same-future result "
                f"continuation follows: {exc}"
            )
        # Retain the existing metric name for artifact compatibility.  It now
        # counts a bounded continuation of the same result future.
        self.step_result_refetch_count += 1
        return (
            _future_result(
                result_future,
                T5_STEP_RESULT_REFETCH_LIVENESS_SEC,
                "same-goal step result continuation",
            ),
            1,
        )

    def _new_step_action_client(self) -> ActionClient:
        return ActionClient(
            self,
            Step,
            "/internvla/step",
            callback_group=self._step_action_callback_group,
        )

    def _rotate_step_action_client_if_needed(self) -> None:
        """Replace a quiescent T5 ActionClient at a bounded safe boundary.

        The current model identity is never replayed.  A poisoned client is
        only retired before a *new* goal, after the failed request has already
        safe-stopped and unwound its current-goal handle.
        """

        if not self.t5_sim_time_only:
            return
        completed = int(
            getattr(self, "step_action_client_completed_goals", 0)
        )
        poisoned = bool(
            getattr(self, "step_action_client_transport_poisoned", False)
        )
        if (
            not poisoned
            and completed < T5_STEP_ACTION_CLIENT_MAX_COMPLETED_GOALS
        ):
            return
        with self.current_goal_lock:
            if self.current_goal is not None:
                raise ClientFailure(
                    STATUS_INTERNAL_ERROR,
                    "refusing to rotate Step ActionClient with an active goal",
                )
        previous = self.step_client
        try:
            previous.destroy()
        except BaseException as exc:
            self._safe_stop(
                STATUS_INTERNAL_ERROR,
                "failed to retire poisoned Step ActionClient",
            )
            raise ClientFailure(
                STATUS_INTERNAL_ERROR,
                f"failed to retire Step ActionClient: {exc!r}",
            ) from exc
        self.step_client = self._new_step_action_client()
        if not self.step_client.wait_for_server(timeout_sec=5.0):
            self.step_action_client_transport_poisoned = True
            self._safe_stop(
                STATUS_TIMEOUT,
                "replacement Step ActionClient did not discover its server",
            )
            raise ClientFailure(
                STATUS_TIMEOUT,
                "replacement Step ActionClient did not discover its server",
            )
        self.step_action_client_completed_goals = 0
        self.step_action_client_transport_poisoned = False
        self.step_action_client_rotation_count = int(
            getattr(self, "step_action_client_rotation_count", 0)
        ) + 1
        self.get_logger().warning(
            "rotated quiescent Step ActionClient at bounded T5 transport "
            f"boundary; rotations={self.step_action_client_rotation_count}"
        )

    def _send_step_goal_once(
        self,
        goal: Step.Goal,
        action_liveness_timeout_sec: float,
        result_liveness_timeout_sec: float,
    ) -> tuple[Any, int]:
        self._rotate_step_action_client_if_needed()
        try:
            goal_handle = _future_result(
                self.step_client.send_goal_async(goal),
                min(T5_STEP_GOAL_ACCEPTANCE_LIVENESS_SEC, action_liveness_timeout_sec),
                "step goal acceptance",
            )
        except ClientFailure as exc:
            if self.t5_sim_time_only and exc.status_code == STATUS_TIMEOUT:
                self.step_action_client_transport_poisoned = True
                self._safe_stop(
                    STATUS_TIMEOUT,
                    "Step goal acceptance transport timed out; client marked poisoned",
                )
            raise
        if not goal_handle.accepted:
            self._safe_stop(STATUS_INVALID_REQUEST, "step goal rejected")
            raise ClientFailure(STATUS_INVALID_REQUEST, "step goal rejected")
        with self.current_goal_lock:
            self.current_goal = goal_handle
        try:
            wrapped, result_refetch_count = self._wait_for_accepted_step_result(
                goal_handle,
                result_liveness_timeout_sec,
            )
        except ClientFailure:
            if self.t5_sim_time_only:
                self.step_action_client_transport_poisoned = True
            timeout_message = (
                "step result timeout after bounded same-goal re-fetch; "
                "cancel requested"
                if self.t5_sim_time_only
                else "step result timeout; cancel requested"
            )
            self._safe_stop(
                STATUS_TIMEOUT,
                timeout_message,
            )
            try:
                goal_handle.cancel_goal_async()
            except BaseException as cancel_exc:
                # Preserve the original result-delivery failure. The physical
                # safe-stop is already active; cancellation is best-effort.
                self.get_logger().warning(
                    "best-effort Step cancel failed after safe-stop: "
                    f"{cancel_exc!r}"
                )
            raise
        finally:
            with self.current_goal_lock:
                if self.current_goal is goal_handle:
                    self.current_goal = None
        if self.t5_sim_time_only:
            self.step_action_client_completed_goals = int(
                getattr(self, "step_action_client_completed_goals", 0)
            ) + 1
        return wrapped, result_refetch_count

    def _request_step_with_bounded_observation_retry(
        self,
        *,
        goal: Step.Goal,
        rgb_message: Any,
        depth_message: Any,
        metadata: ObservationMetadata,
        action_liveness_timeout_sec: float,
        result_liveness_timeout_sec: float,
    ) -> tuple[Any, int, int]:
        wrapped, result_refetch_count = self._send_step_goal_once(
            goal,
            action_liveness_timeout_sec,
            result_liveness_timeout_sec,
        )
        observation_republish_count = 0
        if (
            self.t5_sim_time_only
            and int(wrapped.result.status_code) == STATUS_OBSERVATION_MISSING
            and str(wrapped.result.episode_id) == str(goal.episode_id)
            and int(wrapped.result.reset_generation)
            == int(goal.reset_generation)
            and int(wrapped.result.sequence_id) == int(goal.sequence_id)
            and str(wrapped.result.request_id) == str(goal.request_id)
        ):
            # Republish the exact encoded tuple and resend the exact protocol
            # identity once.  No stamp, digest, request ID, or sequence is
            # regenerated, and the failed missing-observation goal did not
            # reach model inference or commit the generation barrier.
            self.get_logger().warning(
                "Step reported missing observation; republishing exact tuple "
                "and retrying the same protocol identity once"
            )
            self._publish_observation_tuple(rgb_message, depth_message, metadata)
            observation_republish_count = 1
            self.observation_tuple_republish_count += 1
            wrapped, retry_refetch_count = self._send_step_goal_once(
                goal,
                action_liveness_timeout_sec,
                result_liveness_timeout_sec,
            )
            result_refetch_count += retry_refetch_count
        return wrapped, result_refetch_count, observation_republish_count

    def step_arrays(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
        instruction: str,
        instruction_tokens: list[int],
        global_gps: list[float],
        global_rotation: list[float],
        sequence_id: int | None = None,
        request_id: str | None = None,
        deadline_sec: float | None = None,
        validity_sec: float | None = None,
        commit_sequence: bool = True,
        camera_sensor_sequence: int | None = None,
        camera_sensor_stamp_ns: int | None = None,
        camera_sensor_schema_version: int | None = None,
        camera_sensor_source: str | None = None,
    ) -> dict[str, Any]:
        with self.operation_lock:
            if not self.initialized:
                raise ClientFailure(STATUS_INVALID_REQUEST, "client is not initialized")
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            depth = np.ascontiguousarray(depth, dtype=np.float32)
            if rgb.shape != (480, 640, 3) or depth.shape != (480, 640, 1):
                raise ClientFailure(STATUS_INVALID_REQUEST, "observation shape violates frozen contract")
            if not np.isfinite(depth).all() or float(depth.min()) < 0.0 or float(depth.max()) > 1.0:
                raise ClientFailure(STATUS_INVALID_REQUEST, "depth violates finite normalized contract")
            if len(global_gps) != 3 or len(global_rotation) != 4:
                raise ClientFailure(STATUS_INVALID_REQUEST, "pose arrays must be length 3/4")

            sequence = self.next_sequence_id if sequence_id is None else int(sequence_id)
            request_name = request_id or f"{self.episode_id}:{self.reset_generation}:{sequence}"
            identity = RequestIdentity(
                self.episode_id,
                self.reset_generation,
                sequence,
                request_name,
            )
            encode_started = time.perf_counter()
            try:
                if self.observation_transport == "compressed":
                    rgb_payload = encode_rgb_jpeg(
                        rgb,
                        quality=self.rgb_jpeg_quality,
                        maximum_bytes=self.maximum_rgb_bytes,
                    )
                    depth_payload = encode_depth_png(
                        depth,
                        compression=self.depth_png_compression,
                        maximum_bytes=self.maximum_depth_bytes,
                    )
                    digest = compressed_observation_digest(
                        rgb_payload,
                        depth_payload,
                        instruction,
                        instruction_tokens,
                        global_gps,
                        global_rotation,
                    )
                    observation_wire_bytes = len(rgb_payload) + len(depth_payload)
                else:
                    rgb_payload = rgb.tobytes()
                    depth_payload = depth.astype("<f4", copy=False).tobytes()
                    digest = observation_digest(
                        rgb,
                        depth,
                        instruction,
                        instruction_tokens,
                        global_gps,
                        global_rotation,
                    )
                    observation_wire_bytes = len(rgb_payload) + len(depth_payload)
            except ObservationCodecError as exc:
                raise ClientFailure(STATUS_INVALID_REQUEST, str(exc)) from exc
            encode_latency_sec = float(time.perf_counter() - encode_started)

            deadline_duration = self.step_deadline_sec if deadline_sec is None else float(deadline_sec)
            validity_duration = self.step_validity_sec if validity_sec is None else float(validity_sec)
            if deadline_duration <= 0.0 or validity_duration <= 0.0:
                raise ClientFailure(STATUS_INVALID_REQUEST, "deadline/validity must be positive")
            sim_now = self._wait_for_observation_stamp(identity, deadline_duration)
            now_ns = sim_now.nanoseconds
            deadline_ns = now_ns + int(deadline_duration * 1e9)
            valid_until_ns = now_ns + int(validity_duration * 1e9)
            if validity_duration < deadline_duration:
                raise ClientFailure(STATUS_INVALID_REQUEST, "validity precedes deadline")
            client_wall_monotonic_ns = time.monotonic_ns()
            if not self.t5_sim_time_only:
                local_wall_deadline_ns = client_wall_monotonic_ns + int(
                    deadline_duration * 1e9
                )
                local_wall_valid_until_ns = client_wall_monotonic_ns + int(
                    validity_duration * 1e9
                )

            stamp = sim_now.to_msg()
            if self.observation_transport == "compressed":
                rgb_message = CompressedImage()
                rgb_message.header.stamp = stamp
                rgb_message.header.frame_id = "internvla_camera"
                rgb_message.format = "jpeg; encoding=rgb8"
                rgb_message.data = rgb_payload
                depth_message = CompressedImage()
                depth_message.header.stamp = stamp
                depth_message.header.frame_id = "internvla_camera"
                depth_message.format = "png; encoding=16UC1; source=32FC1-normalized"
                depth_message.data = depth_payload
            else:
                rgb_message = Image()
                rgb_message.header.stamp = stamp
                rgb_message.header.frame_id = "internvla_camera"
                rgb_message.height = 480
                rgb_message.width = 640
                rgb_message.encoding = "rgb8"
                rgb_message.is_bigendian = False
                rgb_message.step = 640 * 3
                rgb_message.data = rgb_payload
                depth_message = Image()
                depth_message.header.stamp = stamp
                depth_message.header.frame_id = "internvla_camera"
                depth_message.height = 480
                depth_message.width = 640
                depth_message.encoding = "32FC1"
                depth_message.is_bigendian = False
                depth_message.step = 640 * 4
                depth_message.data = depth_payload
            metadata = ObservationMetadata()
            metadata.header.stamp = stamp
            metadata.header.frame_id = "internvla_camera"
            metadata.protocol_version = PROTOCOL_VERSION
            metadata.episode_id = self.episode_id
            metadata.reset_generation = self.reset_generation
            metadata.sequence_id = sequence
            metadata.request_id = request_name
            metadata.observation_digest = digest
            metadata.client_wall_monotonic_ns = client_wall_monotonic_ns
            metadata.sim_stamp = stamp
            _assign_time(metadata.deadline, deadline_ns)
            _assign_time(metadata.valid_until, valid_until_ns)
            metadata.instruction = str(instruction)
            metadata.instruction_tokens = [int(value) for value in instruction_tokens]
            metadata.global_gps = [float(value) for value in global_gps]
            metadata.global_rotation = [float(value) for value in global_rotation]

            if self.publish_observation_pose:
                transform = TransformStamped()
                transform.header.stamp = stamp
                transform.header.frame_id = "odom"
                transform.child_frame_id = "base_link"
                transform.transform.translation.x = float(global_gps[0])
                transform.transform.translation.y = float(global_gps[1])
                transform.transform.translation.z = float(global_gps[2])
                # Isaac returns scalar-first (w,x,y,z); ROS messages are (x,y,z,w).
                transform.transform.rotation.x = float(global_rotation[1])
                transform.transform.rotation.y = float(global_rotation[2])
                transform.transform.rotation.z = float(global_rotation[3])
                transform.transform.rotation.w = float(global_rotation[0])
                self.tf_broadcaster.sendTransform(transform)
                odometry = Odometry()
                odometry.header.stamp = stamp
                odometry.header.frame_id = "odom"
                odometry.child_frame_id = "base_link"
                odometry.pose.pose.position.x = float(global_gps[0])
                odometry.pose.pose.position.y = float(global_gps[1])
                odometry.pose.pose.position.z = float(global_gps[2])
                odometry.pose.pose.orientation = transform.transform.rotation
                odometry.pose.covariance[0] = 1e-6
                odometry.pose.covariance[7] = 1e-6
                odometry.pose.covariance[35] = 1e-6
                self.odom_publisher.publish(odometry)

            self._publish_observation_tuple(rgb_message, depth_message, metadata)

            goal = Step.Goal()
            goal.header.stamp = stamp
            goal.header.frame_id = "internvla_camera"
            goal.protocol_version = PROTOCOL_VERSION
            goal.episode_id = self.episode_id
            goal.reset_generation = self.reset_generation
            goal.sequence_id = sequence
            goal.request_id = request_name
            goal.observation_digest = digest
            goal.client_wall_monotonic_ns = client_wall_monotonic_ns
            goal.sim_stamp = stamp
            goal.observation_stamp = stamp
            _assign_time(goal.deadline, deadline_ns)
            _assign_time(goal.valid_until, valid_until_ns)

            action_liveness_timeout_sec = (
                self.service_timeout_sec
                if self.t5_sim_time_only
                else deadline_duration
            )
            if not self.step_client.wait_for_server(
                timeout_sec=min(5.0, action_liveness_timeout_sec)
            ):
                self._safe_stop(STATUS_TIMEOUT, "step action server unavailable")
                raise ClientFailure(STATUS_TIMEOUT, "step action server unavailable")
            action_round_trip_started = time.perf_counter()
            result_liveness_timeout_sec = (
                max(0.1, self.service_timeout_sec)
                if self.t5_sim_time_only
                else max(
                    0.1,
                    (local_wall_deadline_ns - time.monotonic_ns()) / 1e9 + 1.0,
                )
            )
            (
                wrapped,
                result_refetch_count,
                observation_republish_count,
            ) = self._request_step_with_bounded_observation_retry(
                goal=goal,
                rgb_message=rgb_message,
                depth_message=depth_message,
                metadata=metadata,
                action_liveness_timeout_sec=action_liveness_timeout_sec,
                result_liveness_timeout_sec=result_liveness_timeout_sec,
            )
            action_round_trip_latency_sec = float(
                time.perf_counter() - action_round_trip_started
            )

            result = wrapped.result
            if int(result.status_code) != STATUS_OK:
                if int(result.reset_generation) != self.reset_generation:
                    self.reset_generation = int(result.reset_generation)
                    self.next_sequence_id = 0
                    self.last_committed_sequence = -1
                self._safe_stop(int(result.status_code), result.status_message)
                raise ClientFailure(int(result.status_code), result.status_message)
            if (
                result.episode_id != self.episode_id
                or int(result.reset_generation) != self.reset_generation
                or int(result.sequence_id) != sequence
                or result.request_id != request_name
            ):
                self._safe_stop(STATUS_STALE, "stale/mismatched response discarded")
                raise ClientFailure(STATUS_STALE, "stale/mismatched response discarded")
            if self.t5_sim_time_only:
                response_sim_ns = self._t5_semantic_sim_ns(
                    self.get_clock().now().nanoseconds
                )
                if response_sim_ns <= 0:
                    message = "ROS simulation clock is zero at response acceptance"
                    self._safe_stop(STATUS_TIMEOUT, message)
                    raise ClientFailure(STATUS_TIMEOUT, message)
                if response_sim_ns < now_ns:
                    message = (
                        "ROS simulation clock regressed before response acceptance: "
                        f"request_ns={now_ns} observed_ns={response_sim_ns}"
                    )
                    self._safe_stop(STATUS_STALE, message)
                    raise ClientFailure(STATUS_STALE, message)
                if response_sim_ns > deadline_ns:
                    message = "response exceeded ROS simulation-time deadline"
                    self._safe_stop(STATUS_TIMEOUT, message)
                    raise ClientFailure(STATUS_TIMEOUT, message)
            else:
                if time.monotonic_ns() > local_wall_deadline_ns:
                    self._safe_stop(
                        STATUS_TIMEOUT,
                        "response exceeded local monotonic deadline",
                    )
                    raise ClientFailure(
                        STATUS_TIMEOUT,
                        "response exceeded local monotonic deadline",
                    )
                response_sim_ns = self.get_clock().now().nanoseconds
            if response_sim_ns > _time_ns(result.valid_until):
                self._safe_stop(STATUS_STALE, "response expired before client acceptance")
                raise ClientFailure(STATUS_STALE, "response expired before client acceptance")
            if (
                not self.t5_sim_time_only
                and time.monotonic_ns() > local_wall_valid_until_ns
            ):
                self._safe_stop(STATUS_STALE, "response exceeded local monotonic validity")
                raise ClientFailure(STATUS_STALE, "response exceeded local monotonic validity")

            if commit_sequence and not result.replayed:
                if sequence != self.next_sequence_id:
                    self._safe_stop(STATUS_STALE, "unexpected sequence commit")
                    raise ClientFailure(STATUS_STALE, "unexpected sequence commit")
                self.last_committed_sequence = sequence
                self.next_sequence_id = sequence + 1
            command = NavigationCommand()
            command.header.stamp = stamp
            command.header.frame_id = "base_link"
            command.protocol_version = PROTOCOL_VERSION
            command.episode_id = result.episode_id
            command.reset_generation = int(result.reset_generation)
            command.sequence_id = int(result.sequence_id)
            command.request_id = result.request_id
            command.observation_digest = digest
            command.valid_until = result.valid_until
            command.discrete_action = int(result.discrete_action)
            command.stop = bool(result.stop)
            command.action_source = int(result.action_source)
            command.trajectory_source = int(result.trajectory_source)
            command.trajectory_valid = bool(result.trajectory_valid)
            command.local_path = result.local_path
            command.local_path.header.stamp = stamp
            command.local_path.header.frame_id = "base_link"
            for pose in command.local_path.poses:
                pose.header = command.local_path.header
            command.global_gps = [float(value) for value in global_gps]
            command.global_rotation_wxyz = [float(value) for value in global_rotation]
            self.navigation_command_publisher.publish(command)
            selected_action = int(result.discrete_action)
            selected_stop = bool(result.stop)
            nav2_goal_sent = False
            nav2_plan_valid = False
            resolution_latency_sec = 0.0
            if self.control_mode == "nav2":
                resolution = self._resolve_nav2(command)
                selected_action = int(resolution.discrete_action)
                selected_stop = bool(resolution.stop)
                nav2_goal_sent = bool(resolution.nav2_goal_sent)
                nav2_plan_valid = bool(resolution.nav2_plan_valid)
                resolution_latency_sec = float(resolution.resolution_latency_sec)
            action_message = Int8()
            action_message.data = selected_action
            self.action_publisher.publish(action_message)
            stop_message = Bool()
            stop_message.data = selected_stop
            self.stop_publisher.publish(stop_message)
            self._clear_safe_stop("step ok")
            return {
                "status_code": int(result.status_code),
                "status_message": result.status_message,
                "episode_id": result.episode_id,
                "reset_generation": int(result.reset_generation),
                "sequence_id": int(result.sequence_id),
                "request_id": result.request_id,
                "observation_digest": digest,
                "observation_transport": self.observation_transport,
                "rgb_wire_bytes": len(rgb_payload),
                "depth_wire_bytes": len(depth_payload),
                "observation_wire_bytes": observation_wire_bytes,
                "observation_encode_latency_sec": encode_latency_sec,
                "action_round_trip_latency_sec": action_round_trip_latency_sec,
                "step_result_refetch_count": result_refetch_count,
                "observation_tuple_republish_count": observation_republish_count,
                "step_action_client_rotation_count": int(
                    self.step_action_client_rotation_count
                ),
                "network_ros_residual_latency_sec": max(
                    0.0,
                    action_round_trip_latency_sec
                    - float(result.inference_latency_sec),
                ),
                "client_wall_monotonic_ns": client_wall_monotonic_ns,
                "sim_stamp_ns": now_ns,
                "discrete_action": selected_action,
                "model_discrete_action": int(result.discrete_action),
                "stop": selected_stop,
                "model_stop": bool(result.stop),
                "control_mode": self.control_mode,
                "nav2_goal_sent": nav2_goal_sent,
                "nav2_plan_valid": nav2_plan_valid,
                "nav2_resolution_latency_sec": resolution_latency_sec,
                "replayed": bool(result.replayed),
                "action_source": int(result.action_source),
                "trajectory_source": int(result.trajectory_source),
                "trajectory_valid": bool(result.trajectory_valid),
                "local_path_frame": result.local_path.header.frame_id,
                "local_path": [
                    [float(pose.pose.position.x), float(pose.pose.position.y)]
                    for pose in result.local_path.poses
                ],
                "valid_until_ns": _time_ns(result.valid_until),
                "inference_latency_sec": float(result.inference_latency_sec),
            }

    def _wait_for_observation_stamp(
        self, identity: RequestIdentity, request_deadline_sec: float
    ) -> Any:
        wait_sec = min(
            float(request_deadline_sec), self.observation_stamp_wait_sec
        )
        wait_deadline_ns = time.monotonic_ns() + int(wait_sec * 1e9)
        previous_identity = self._observation_stamp_gate.last_identity
        previous_stamp_ns = self._observation_stamp_gate.last_stamp_ns
        observed_stamp_ns = 0
        while True:
            sim_now = self.get_clock().now()
            observed_stamp_ns = int(sim_now.nanoseconds)
            if getattr(self, "t5_sim_time_only", False):
                observed_stamp_ns = self._t5_semantic_sim_ns(observed_stamp_ns)
            if self._observation_stamp_gate.accepts(identity, observed_stamp_ns):
                self._observation_stamp_gate.commit(identity, observed_stamp_ns)
                return sim_now
            remaining_ns = wait_deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                previous = (
                    "none"
                    if previous_identity is None
                    else describe_request_identity(previous_identity)
                )
                message = (
                    "sim clock did not provide a strictly increasing stamp for "
                    "a new observation request within the bounded wait: "
                    f"wait_sec={wait_sec:.3f} "
                    f"previous_identity={previous} "
                    f"requested_identity={describe_request_identity(identity)} "
                    f"previous_stamp_ns={previous_stamp_ns} "
                    f"observed_stamp_ns={observed_stamp_ns}"
                )
                self._safe_stop(STATUS_TIMEOUT, message)
                raise ClientFailure(STATUS_TIMEOUT, message)
            time.sleep(min(0.01, remaining_ns / 1e9))

    def _t5_semantic_sim_ns(self, observed_ns: int) -> int:
        observed_ns = int(observed_ns)
        with self._t5_sim_time_lock:
            if observed_ns <= 0:
                status_code = STATUS_TIMEOUT
                message = "ROS simulation clock is zero"
            elif observed_ns < self._t5_sim_time_high_water_ns:
                status_code = STATUS_STALE
                message = (
                    "ROS simulation clock regressed below persistent high-water: "
                    f"high_water_ns={self._t5_sim_time_high_water_ns} "
                    f"observed_ns={observed_ns}"
                )
            else:
                self._t5_sim_time_high_water_ns = observed_ns
                return observed_ns
        self._safe_stop(status_code, message)
        raise ClientFailure(status_code, message)

    def record_step(self, result: dict[str, Any]) -> None:
        if self.records_path is None:
            return
        record = {"schema_version": 1, "wall_time_unix": time.time(), **result}
        with self.records_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        inference = float(result["inference_latency_sec"])
        nav2 = float(result["nav2_resolution_latency_sec"])
        encode = float(result["observation_encode_latency_sec"])
        action_round_trip = float(result["action_round_trip_latency_sec"])
        self.client_step_count += 1
        self.inference_latency_sum += inference
        self.inference_latency_max = max(self.inference_latency_max, inference)
        self.nav2_latency_sum += nav2
        self.nav2_latency_max = max(self.nav2_latency_max, nav2)
        self.encode_latency_sum += encode
        self.encode_latency_max = max(self.encode_latency_max, encode)
        self.action_round_trip_latency_sum += action_round_trip
        self.action_round_trip_latency_max = max(
            self.action_round_trip_latency_max, action_round_trip
        )
        self.observation_wire_bytes_sum += int(result["observation_wire_bytes"])
        if (
            self.client_step_count % 20 == 0
            or os.environ.get("INTERNNAV_T5_FAULT_INJECTION_PROFILE", "off")
            == "completion_sim_minimal_v1"
        ):
            self._write_client_summary("RUNNING")

    def _write_client_summary(self, status: str) -> None:
        if self.result_dir is None:
            return
        count = self.client_step_count
        payload = {
            "schema_version": 1,
            "status": status,
            "started_unix": self.client_started_unix,
            "updated_unix": time.time(),
            "step_count": count,
            "mean_inference_latency_sec": self.inference_latency_sum / count if count else 0.0,
            "maximum_inference_latency_sec": self.inference_latency_max,
            "mean_nav2_resolution_latency_sec": self.nav2_latency_sum / count if count else 0.0,
            "maximum_nav2_resolution_latency_sec": self.nav2_latency_max,
            "observation_transport": self.observation_transport,
            "mean_observation_encode_latency_sec": self.encode_latency_sum / count if count else 0.0,
            "maximum_observation_encode_latency_sec": self.encode_latency_max,
            "mean_action_round_trip_latency_sec": self.action_round_trip_latency_sum / count if count else 0.0,
            "maximum_action_round_trip_latency_sec": self.action_round_trip_latency_max,
            "mean_observation_wire_bytes": self.observation_wire_bytes_sum / count if count else 0.0,
            "reset_barrier_reconcile_count": self.reset_barrier_reconcile_count,
            "step_result_refetch_count": self.step_result_refetch_count,
            "observation_tuple_republish_count": self.observation_tuple_republish_count,
            "step_action_client_rotation_count": int(
                getattr(self, "step_action_client_rotation_count", 0)
            ),
            "step_action_client_transport_poisoned": bool(
                getattr(self, "step_action_client_transport_poisoned", False)
            ),
            "lane_identity_prefix": self.identity_prefix,
            "reset_id": f"{self.identity_prefix}{self.reset_generation}",
            "model_health_start": self.model_health_start,
        }
        if (
            os.environ.get("INTERNNAV_T5_FAULT_INJECTION_PROFILE", "off")
            == "completion_sim_minimal_v1"
        ):
            payload["fault_restart_session"] = {
                "episode_id": str(getattr(self, "episode_id", "")),
                "reset_generation": int(getattr(self, "reset_generation", 0)),
                "next_sequence_id": int(getattr(self, "next_sequence_id", 0)),
                "last_committed_sequence": int(
                    getattr(self, "last_committed_sequence", -1)
                ),
            }
        temporary = self.result_dir / "client_summary.json.tmp"
        final = self.result_dir / "client_summary.json"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, final)

    def finish_records(self) -> None:
        self._write_client_summary("FINISHED")

    def _resolve_nav2(self, command: NavigationCommand) -> ResolveCommand.Response:
        if not self.resolve_client.wait_for_service(timeout_sec=1.0):
            self._safe_stop(STATUS_TIMEOUT, "Nav2 resolver unavailable")
            raise ClientFailure(STATUS_TIMEOUT, "Nav2 resolver unavailable")
        request = ResolveCommand.Request()
        request.command = command
        response = _future_result(
            self.resolve_client.call_async(request),
            self.nav2_resolution_timeout_sec,
            "Nav2 command resolution",
        )
        if int(response.status_code) != STATUS_OK:
            self._safe_stop(int(response.status_code), response.status_message)
            raise ClientFailure(int(response.status_code), response.status_message)
        if (
            response.episode_id != command.episode_id
            or int(response.reset_generation) != int(command.reset_generation)
            or int(response.sequence_id) != int(command.sequence_id)
            or response.request_id != command.request_id
        ):
            self._safe_stop(STATUS_STALE, "mismatched Nav2 resolution discarded")
            raise ClientFailure(STATUS_STALE, "mismatched Nav2 resolution discarded")
        return response

    def cancel_current(self) -> bool:
        with self.current_goal_lock:
            goal = self.current_goal
        if goal is None:
            return False
        try:
            goal.cancel_goal_async()
            self._safe_stop(STATUS_CANCELED, "in-flight goal cancel requested")
            return True
        except BaseException as exc:
            self._safe_stop(STATUS_INTERNAL_ERROR, f"cancel failed: {exc!r}")
            return False

    def request_model_shutdown(self, reason: str) -> dict[str, Any]:
        request = Shutdown.Request()
        request.protocol_version = PROTOCOL_VERSION
        request.reason = reason
        request.force = False
        response = _future_result(
            self.shutdown_client.call_async(request), 10.0, "shutdown"
        )
        self._safe_stop(STATUS_CANCELED, "model shutdown")
        return {
            "status_code": int(response.status_code),
            "status_message": response.status_message,
            "accepted": bool(response.accepted),
        }

    def _safe_stop(self, status_code: int, message: str) -> None:
        self.safe_stop = True
        self.last_status_code = int(status_code)
        self.last_status_message = str(message)[:1024]
        action_message = Int8()
        action_message.data = ACTION_STOP
        self.action_publisher.publish(action_message)
        stop_message = Bool()
        stop_message.data = True
        self.stop_publisher.publish(stop_message)
        self._publish_state()

    def _clear_safe_stop(self, message: str) -> None:
        self.safe_stop = False
        self.last_status_code = STATUS_OK
        self.last_status_message = message
        self._publish_state()

    def _publish_state(self) -> None:
        message = ModelState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.protocol_version = PROTOCOL_VERSION
        message.status_code = self.last_status_code
        message.status_message = self.last_status_message
        message.lifecycle_state = 2 if self.initialized else 0
        message.initialized = self.initialized
        message.episode_id = self.episode_id
        message.reset_generation = self.reset_generation
        message.last_sequence_id = max(0, self.last_committed_sequence)
        message.safe_stop = self.safe_stop
        message.model_revision = MODEL_REVISION
        message.checkpoint_revision = CHECKPOINT_REVISION
        self.state_publisher.publish(message)


class ClientRuntime:
    def __init__(self) -> None:
        self.node = InternVLAClientNode()
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)

    def start(self) -> None:
        self.thread.start()
        self.node.wait_for_graph()
        self.node.model_health_start = validate_uninitialized_model_health(
            self.node.health()
        )
        self.node._write_client_summary("READY")

    def stop(self) -> None:
        self.node.cancel_current()
        self.node.finish_records()
        self.executor.shutdown()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self.thread.join(timeout=5.0)


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_json(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _recv_exact(connection, 4))[0]
    if size < 2 or size > MAX_IPC_MESSAGE_BYTES:
        raise ClientFailure(STATUS_INVALID_REQUEST, "invalid IPC message length")
    value = json.loads(_recv_exact(connection, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise ClientFailure(STATUS_INVALID_REQUEST, "IPC message must be an object")
    return value


def _send_json(connection: socket.socket, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_IPC_MESSAGE_BYTES:
        raise ClientFailure(STATUS_INTERNAL_ERROR, "IPC response exceeds bound")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def _shared_array(descriptor: dict[str, Any], expected_shape: tuple[int, ...], expected_dtype: str) -> np.ndarray:
    name = str(descriptor.get("name", ""))
    shape = tuple(int(value) for value in descriptor.get("shape", []))
    dtype = str(descriptor.get("dtype", ""))
    if not SHM_NAME_RE.fullmatch(name):
        raise ClientFailure(STATUS_INVALID_REQUEST, "invalid shared-memory name")
    if shape != expected_shape or dtype != expected_dtype:
        raise ClientFailure(STATUS_INVALID_REQUEST, "shared-memory array contract mismatch")
    data_type = np.dtype(dtype)
    expected_bytes = int(np.prod(shape)) * data_type.itemsize
    shared = SharedMemory(name=name, create=False)
    try:
        if shared.size < expected_bytes:
            raise ClientFailure(STATUS_INVALID_REQUEST, "shared-memory segment is undersized")
        return np.ndarray(shape, dtype=data_type, buffer=shared.buf[:expected_bytes]).copy()
    finally:
        shared.close()
        # The Isaac producer owns unlink. Prevent this attachment-only process
        # from trying to unlink the same segment again at interpreter exit.
        try:
            resource_tracker.unregister(shared._name, "shared_memory")
        except (AttributeError, KeyError):
            pass


def _inline_array(
    descriptor: dict[str, Any],
    expected_shape: tuple[int, ...],
    expected_dtype: str,
) -> np.ndarray:
    shape = tuple(int(value) for value in descriptor.get("shape", []))
    dtype = str(descriptor.get("dtype", ""))
    if shape != expected_shape or dtype != expected_dtype:
        raise ClientFailure(STATUS_INVALID_REQUEST, "inline array contract mismatch")
    if descriptor.get("encoding") != "zlib+base64":
        raise ClientFailure(STATUS_INVALID_REQUEST, "unsupported inline array encoding")
    try:
        compressed = base64.b64decode(
            str(descriptor.get("data", "")).encode("ascii"), validate=True
        )
    except (UnicodeEncodeError, ValueError) as exc:
        raise ClientFailure(STATUS_INVALID_REQUEST, "invalid inline array base64") from exc
    if not compressed or len(compressed) > MAX_INLINE_COMPRESSED_BYTES:
        raise ClientFailure(STATUS_INVALID_REQUEST, "inline compressed array exceeds bound")
    data_type = np.dtype(dtype)
    expected_bytes = int(np.prod(shape)) * data_type.itemsize
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(compressed, expected_bytes + 1)
    except zlib.error as exc:
        raise ClientFailure(STATUS_INVALID_REQUEST, "invalid inline array compression") from exc
    if (
        len(raw) != expected_bytes
        or not inflater.eof
        or inflater.unused_data
        or inflater.unconsumed_tail
    ):
        raise ClientFailure(STATUS_INVALID_REQUEST, "inline array payload size mismatch")
    return np.frombuffer(raw, dtype=data_type).reshape(shape).copy()


class LocalIPCServer:
    def __init__(self, node: InternVLAClientNode, socket_path: Path):
        self.node = node
        self.socket_path = socket_path
        self.stop_event = threading.Event()

    def serve(self) -> None:
        if self.socket_path.exists():
            if self.socket_path.is_socket():
                self.socket_path.unlink()
            else:
                raise RuntimeError(f"refusing non-socket IPC path: {self.socket_path}")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        listener.listen(1)
        listener.settimeout(1.0)
        try:
            while not self.stop_event.is_set() and rclpy.ok():
                try:
                    connection, _ = listener.accept()
                except (socket.timeout, TimeoutError):
                    continue
                with connection:
                    connection.settimeout(self.node.ipc_idle_timeout_sec)
                    try:
                        while not self.stop_event.is_set():
                            request = _recv_json(connection)
                            response = self._handle(request)
                            _send_json(connection, response)
                    except (EOFError, BrokenPipeError, ConnectionResetError, socket.timeout):
                        self.node._safe_stop(STATUS_TIMEOUT, "local evaluator IPC disconnected")
                    except ClientFailure as exc:
                        self.node._safe_stop(exc.status_code, str(exc))
                        try:
                            _send_json(
                                connection,
                                {
                                    "schema_version": 1,
                                    "status": "error",
                                    "status_code": exc.status_code,
                                    "message": str(exc),
                                    "discrete_action": ACTION_STOP,
                                    "stop": True,
                                },
                            )
                        except OSError:
                            pass
        finally:
            listener.close()
            if self.socket_path.exists():
                self.socket_path.unlink()

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        if int(request.get("schema_version", 0)) != 1:
            raise ClientFailure(STATUS_INVALID_REQUEST, "IPC schema_version must be 1")
        operation = str(request.get("operation", ""))
        if operation == "health":
            return {"schema_version": 1, "status": "ok", **self.node.health()}
        if operation == "initialize":
            return {
                "schema_version": 1,
                "status": "ok",
                **self.node.initialize(str(request.get("episode_id", ""))),
            }
        if operation == "reset":
            return {
                "schema_version": 1,
                "status": "ok",
                **self.node.reset(str(request.get("next_episode_id", ""))),
            }
        if operation == "cancel":
            return {"schema_version": 1, "status": "ok", "cancel_requested": self.node.cancel_current()}
        if operation == "shutdown_model":
            return {
                "schema_version": 1,
                "status": "ok",
                **self.node.request_model_shutdown(str(request.get("reason", "IPC shutdown"))),
            }
        if operation == "shutdown_client":
            self.stop_event.set()
            return {"schema_version": 1, "status": "ok", "accepted": True}
        if operation != "step":
            raise ClientFailure(STATUS_INVALID_REQUEST, f"unsupported IPC operation {operation!r}")
        if request.get("array_transport") == "inline_zlib":
            rgb = _inline_array(
                request.get("rgb", {}), (480, 640, 3), "uint8"
            )
            depth = _inline_array(
                request.get("depth", {}), (480, 640, 1), "float32"
            )
        else:
            rgb = _shared_array(request.get("rgb", {}), (480, 640, 3), "uint8")
            depth = _shared_array(request.get("depth", {}), (480, 640, 1), "float32")
        camera_metadata = request.get("camera_sensor_metadata")
        if camera_metadata is not None and not isinstance(camera_metadata, dict):
            raise ClientFailure(
                STATUS_INVALID_REQUEST, "camera_sensor_metadata must be an object"
            )
        camera_metadata = camera_metadata or {}
        exact_t5_camera_contract = bool(
            getattr(self.node, "_motion_gate_enabled", False)
        )
        if exact_t5_camera_contract and (
            camera_metadata.get("schema_version") != 1
            or camera_metadata.get("source") != "x86_isaac_pano_camera_0"
        ):
            raise ClientFailure(
                STATUS_INVALID_REQUEST,
                "exact T5 camera source metadata is missing or invalid",
            )
        result = self.node.step_arrays(
            rgb=rgb,
            depth=depth,
            instruction=str(request.get("instruction", "")),
            instruction_tokens=[int(value) for value in request.get("instruction_tokens", [])],
            global_gps=[float(value) for value in request.get("global_gps", [0.0, 0.0, 0.0])],
            global_rotation=[float(value) for value in request.get("global_rotation", [0.0, 0.0, 0.0, 1.0])],
            sequence_id=request.get("sequence_id"),
            request_id=request.get("request_id"),
            deadline_sec=request.get("deadline_sec"),
            validity_sec=request.get("validity_sec"),
            commit_sequence=bool(request.get("commit_sequence", True)),
            camera_sensor_sequence=(
                camera_metadata.get("sequence")
                if exact_t5_camera_contract
                else camera_metadata.get(
                    "sequence", request.get("camera_sensor_sequence")
                )
            ),
            camera_sensor_stamp_ns=(
                camera_metadata.get("sim_stamp_ns")
                if exact_t5_camera_contract
                else camera_metadata.get(
                    "sim_stamp_ns", request.get("camera_sensor_stamp_ns")
                )
            ),
            camera_sensor_schema_version=camera_metadata.get("schema_version"),
            camera_sensor_source=camera_metadata.get("source"),
        )
        self.node.record_step(result)
        return {"schema_version": 1, "status": "ok", **result}


class TCPIPCServer(LocalIPCServer):
    """Bounded x86-evaluator to DGX_EDGE observation ingress."""

    def __init__(
        self,
        node: InternVLAClientNode,
        bind_host: str,
        bind_port: int,
        expected_peer_ip: str,
    ):
        super().__init__(node, Path("/dev/null"))
        self.bind_host = str(bind_host)
        self.bind_port = int(bind_port)
        self.expected_peer_ip = str(expected_peer_ip)
        if not 1024 <= self.bind_port <= 65535:
            raise RuntimeError("TCP evaluator ingress port is outside [1024,65535]")
        for label, value in (
            ("bind", self.bind_host),
            ("peer", self.expected_peer_ip),
        ):
            try:
                socket.inet_pton(socket.AF_INET, value)
            except OSError as exc:
                raise RuntimeError(f"invalid IPv4 {label} address: {value!r}") from exc

    def serve(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.bind_host, self.bind_port))
        listener.listen(1)
        listener.settimeout(1.0)
        try:
            while not self.stop_event.is_set() and rclpy.ok():
                try:
                    connection, address = listener.accept()
                except (socket.timeout, TimeoutError):
                    continue
                with connection:
                    peer_ip = str(address[0])
                    if peer_ip != self.expected_peer_ip:
                        self.node._safe_stop(
                            STATUS_INVALID_REQUEST,
                            f"rejected evaluator peer {peer_ip}",
                        )
                        continue
                    connection.settimeout(self.node.ipc_idle_timeout_sec)
                    try:
                        while not self.stop_event.is_set():
                            request = _recv_json(connection)
                            response = self._handle(request)
                            _send_json(connection, response)
                    except (
                        EOFError,
                        BrokenPipeError,
                        ConnectionResetError,
                        socket.timeout,
                    ):
                        self.node._safe_stop(
                            STATUS_TIMEOUT, "remote evaluator IPC disconnected"
                        )
                    except ClientFailure as exc:
                        self.node._safe_stop(exc.status_code, str(exc))
                        try:
                            _send_json(
                                connection,
                                {
                                    "schema_version": 1,
                                    "status": "error",
                                    "status_code": exc.status_code,
                                    "message": str(exc),
                                    "discrete_action": ACTION_STOP,
                                    "stop": True,
                                },
                            )
                        except OSError:
                            pass
        finally:
            listener.close()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    runtime = ClientRuntime()
    bind_host = os.environ.get("INTERNVLA_CLIENT_TCP_BIND_HOST", "")
    if bind_host:
        bind_port = int(os.environ.get("INTERNVLA_CLIENT_TCP_PORT", "25139"))
        expected_peer = os.environ.get(
            "INTERNVLA_CLIENT_TCP_EXPECTED_PEER", "10.100.120.111"
        )
        server: LocalIPCServer = TCPIPCServer(
            runtime.node, bind_host, bind_port, expected_peer
        )
        endpoint = f"tcp://{bind_host}:{bind_port}"
    else:
        socket_path = Path(
            os.environ.get("INTERNVLA_CLIENT_SOCKET", "/tmp/internvla_client.sock")
        )
        server = LocalIPCServer(runtime.node, socket_path)
        endpoint = str(socket_path)
    try:
        runtime.start()
        runtime.node.get_logger().info(f"evaluator IPC ready at {endpoint}")
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop_event.set()
        runtime.stop()


if __name__ == "__main__":
    main()
