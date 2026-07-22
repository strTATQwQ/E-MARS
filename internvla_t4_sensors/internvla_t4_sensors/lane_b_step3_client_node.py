"""Lane-B-only Step3 coordinator layered over the frozen T4 client node."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from internvla_ros2.client_node import (
    ClientFailure,
    InternVLAClientNode,
    LocalIPCServer,
    STATUS_INVALID_REQUEST,
    STATUS_STALE,
    STATUS_TIMEOUT,
    TCPIPCServer,
    validate_uninitialized_model_health,
)
from internvla_ros2.observation import observation_digest
from internvla_ros2.protocol import ACTION_FORWARD, ACTION_STAND_STILL, PROTOCOL_VERSION
from internvla_ros2_msgs.msg import NavigationCommand
from rclpy.executors import MultiThreadedExecutor

from .client_node import T4OdometryClientNode
from .lane_b_step3_live import LaneBStep3LiveCoordinator


class _LaneBStep3ExecutionBridge(InternVLAClientNode):
    """Insert a Lane-B-only execution hook after the frozen T4 gate in the MRO."""

    def step_arrays(self, **kwargs: Any) -> dict[str, Any]:
        if getattr(self, "step3_direct_high_level", False):
            return self._execute_direct_step3(**kwargs)  # type: ignore[attr-defined]
        return super().step_arrays(**kwargs)


class LaneBStep3ClientNode(T4OdometryClientNode, _LaneBStep3ExecutionBridge):
    """Use the private advisor only when this explicit Lane-B node is launched."""

    def __init__(self) -> None:
        advisor = os.environ.get("INTERNNAV_T5_STEP3_LIVE_ADVISOR", "0") == "1"
        direct = os.environ.get("INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL", "0") == "1"
        if not (
            advisor != direct
            and os.environ.get("INTERNNAV_T5_LANE", "") == "b"
            and os.environ.get("INTERNNAV_T5_LANE_NAMESPACE", "") == "/t5/lane_b"
        ):
            raise RuntimeError("exactly one Lane-B Step3 runtime mode is required")
        self.step3_direct_high_level = direct
        super().__init__()
        if not self._t5_sim_time_semantics:
            raise RuntimeError("Step3 live client requires frozen T5 sim-time semantics")
        self.step3_live_coordinator = LaneBStep3LiveCoordinator(self)
        self._step3_pending_context: dict[str, Any] | None = None

    def wait_for_graph(self) -> None:
        if not self.step3_direct_high_level:
            return super().wait_for_graph()
        deadline = time.monotonic() + float(
            self.get_parameter("discovery_timeout_sec").value
        )
        while time.monotonic() < deadline:
            if (
                self.step3_live_coordinator.prepare_client.service_is_ready()
                and self.step3_live_coordinator.commit_client.service_is_ready()
            ):
                return
            self.step3_live_coordinator.prepare_client.wait_for_service(timeout_sec=0.1)
            self.step3_live_coordinator.commit_client.wait_for_service(timeout_sec=0.1)
        raise ClientFailure(STATUS_TIMEOUT, "Step3 direct graph discovery timed out")

    def health(self, timeout_sec: float = 10.0) -> dict[str, Any]:
        if not self.step3_direct_high_level:
            return super().health(timeout_sec=timeout_sec)
        del timeout_sec
        return {
            "status_code": 0,
            "status_message": "Step3 direct client ready",
            "initialized": self.initialized,
            "episode_id": self.episode_id,
            "reset_generation": self.reset_generation,
            "last_sequence_id": self.last_committed_sequence,
            "planner_mode": "direct_high_level",
            "internvla_model_loaded": False,
            "internvla_fallback_allowed": False,
            "protocol_version": PROTOCOL_VERSION,
        }

    def initialize(self, episode_id: str) -> dict[str, Any]:
        if not self.step3_direct_high_level:
            return super().initialize(episode_id)
        with self.operation_lock:
            if self.initialized:
                raise ClientFailure(
                    STATUS_INVALID_REQUEST, "Step3 direct client is already initialized"
                )
            if not str(episode_id).startswith("b::"):
                raise ClientFailure(STATUS_INVALID_REQUEST, "invalid Lane-B episode")
            self.initialized = True
            self.episode_id = str(episode_id)
            self.reset_generation = 0
            self.next_sequence_id = 0
            self.last_committed_sequence = -1
            self._clear_safe_stop("Step3 direct ready")
            self._arm_odometry_barrier()
            self._reset_motion_gate("Step3 direct initialize")
            self._publish_episode_prime()
            return {
                "status_code": 0,
                "episode_id": self.episode_id,
                "reset_generation": self.reset_generation,
                "planner_mode": "direct_high_level",
                "internvla_model_loaded": False,
            }

    def reset(self, next_episode_id: str) -> dict[str, Any]:
        if not self.step3_direct_high_level:
            return super().reset(next_episode_id)
        with self.operation_lock:
            if not self.initialized or not str(next_episode_id).startswith("b::"):
                raise ClientFailure(STATUS_INVALID_REQUEST, "invalid direct reset")
            confirmed_stop = self._confirm_stop_before_identity_change(
                "Step3 direct episode reset"
            )
            self.episode_id = str(next_episode_id)
            self.reset_generation += 1
            self.next_sequence_id = 0
            self.last_committed_sequence = -1
            self._arm_odometry_barrier()
            self._install_post_reset_sensor_barrier(
                "Step3 direct episode reset", confirmed_stop
            )
            self._publish_episode_prime()
            return {
                "status_code": 0,
                "episode_id": self.episode_id,
                "reset_generation": self.reset_generation,
                "planner_mode": "direct_high_level",
                "internvla_model_loaded": False,
            }

    def request_model_shutdown(self, reason: str) -> dict[str, Any]:
        if not self.step3_direct_high_level:
            return super().request_model_shutdown(reason)
        return {"accepted": True, "model_loaded": False, "reason": str(reason)}

    def _direct_safe_stop(self, command: Any, reason: str) -> Any:
        normalized = str(reason or "direct_runtime_failure")[:128]
        status = STATUS_STALE if "stale" in normalized else STATUS_TIMEOUT
        self._safe_stop(status, f"Step3 direct safe-stop: {normalized}")
        raise ClientFailure(status, f"Step3 direct safe-stop: {normalized}")

    @staticmethod
    def _direct_local_path(action: int) -> list[list[float]]:
        return [[0.0, 0.0], [0.25, 0.0]] if action == ACTION_FORWARD else []

    def _execute_direct_step3(self, **kwargs: Any) -> dict[str, Any]:
        if not self.step3_direct_high_level:
            raise RuntimeError("direct Step3 execution hook used in advisor mode")
        rgb = np.ascontiguousarray(kwargs["rgb"], dtype=np.uint8)
        depth = np.ascontiguousarray(kwargs["depth"], dtype=np.float32)
        instruction = str(kwargs["instruction"])
        instruction_tokens = [int(value) for value in kwargs["instruction_tokens"]]
        global_gps = [float(value) for value in kwargs["global_gps"]]
        global_rotation = [float(value) for value in kwargs["global_rotation"]]
        if rgb.shape != (480, 640, 3) or depth.shape != (480, 640, 1):
            raise ClientFailure(STATUS_INVALID_REQUEST, "direct observation shape drifted")
        if not np.isfinite(depth).all() or float(depth.min()) < 0.0 or float(depth.max()) > 1.0:
            raise ClientFailure(STATUS_INVALID_REQUEST, "direct depth contract drifted")
        sequence_value = kwargs.get("sequence_id")
        sequence = self.next_sequence_id if sequence_value is None else int(sequence_value)
        request_name = str(
            kwargs.get("request_id")
            or f"{self.episode_id}:{self.reset_generation}:{sequence}"
        )
        if sequence != self.next_sequence_id:
            self._direct_safe_stop(None, "stale_direct_sequence")
        digest = observation_digest(
            rgb,
            depth,
            instruction,
            instruction_tokens,
            global_gps,
            global_rotation,
        )
        sim_now_ns = int(self.get_clock().now().nanoseconds)
        if sim_now_ns <= 0:
            self._direct_safe_stop(None, "direct_sim_clock_unavailable")
        validity_sec = float(kwargs.get("validity_sec") or 12.0)
        if validity_sec < 12.0:
            raise ClientFailure(
                STATUS_INVALID_REQUEST, "direct validity must cover 12-second deadline"
            )
        command = NavigationCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = "base_link"
        command.protocol_version = PROTOCOL_VERSION
        command.episode_id = self.episode_id
        command.reset_generation = self.reset_generation
        command.sequence_id = sequence
        command.request_id = request_name
        command.observation_digest = digest
        valid_until_ns = sim_now_ns + int(validity_sec * 1_000_000_000)
        command.valid_until.sec = valid_until_ns // 1_000_000_000
        command.valid_until.nanosec = valid_until_ns % 1_000_000_000
        command.discrete_action = ACTION_STAND_STILL
        command.stop = False
        command.action_source = 0
        command.trajectory_source = 0
        command.trajectory_valid = False
        command.local_path.header = command.header
        command.global_gps = global_gps
        command.global_rotation_wxyz = global_rotation
        context = self._step3_pending_context
        if context is None:
            self._direct_safe_stop(command, "direct_context_unavailable")
        started = time.perf_counter()
        resolution = self.step3_live_coordinator.resolve(
            command,
            advisor_snapshot=context["advisor_snapshot"],
            instruction=instruction,
            agent_pose=tuple(global_gps + global_rotation),
            request_sim_ns=sim_now_ns,
        )
        planning_latency_sec = time.perf_counter() - started
        action = int(resolution.discrete_action)
        if (
            int(resolution.status_code) != 0
            or not bool(resolution.nav2_goal_sent)
            or not bool(resolution.nav2_plan_valid)
            or action not in {1, 2, 3}
        ):
            self._direct_safe_stop(command, "direct_commit_not_executed")
        if int(self.get_clock().now().nanoseconds) != sim_now_ns:
            self._direct_safe_stop(command, "direct_sim_time_advanced")
        self.last_committed_sequence = sequence
        self.next_sequence_id = sequence + 1
        self._clear_safe_stop("Step3 direct frontier committed")
        return {
            "status_code": 0,
            "status_message": "Step3 direct frontier committed",
            "episode_id": self.episode_id,
            "reset_generation": self.reset_generation,
            "sequence_id": sequence,
            "request_id": request_name,
            "observation_digest": digest,
            "discrete_action": action,
            "model_discrete_action": action,
            "stop": False,
            "model_stop": False,
            "control_mode": "nav2",
            "nav2_goal_sent": True,
            "nav2_plan_valid": True,
            "resolution_latency_sec": float(resolution.resolution_latency_sec),
            "nav2_resolution_latency_sec": float(
                resolution.resolution_latency_sec
            ),
            "planning_latency_sec": planning_latency_sec,
            "inference_latency_sec": planning_latency_sec,
            "command_age_sec": planning_latency_sec,
            "observation_transport": "lane_b_private_tcp",
            "observation_encode_latency_sec": 0.0,
            "action_round_trip_latency_sec": planning_latency_sec,
            "observation_wire_bytes": int(rgb.nbytes + depth.nbytes),
            "valid_until_ns": valid_until_ns,
            "replayed": False,
            "action_source": 0,
            "trajectory_source": 0,
            "trajectory_valid": False,
            "local_path": self._direct_local_path(action),
            "planner_mode": "direct_high_level",
            "internvla_model_loaded": False,
            "internvla_fallback_used": False,
        }

    def _resolve_frozen_nav2(self, command: Any) -> Any:
        return super()._resolve_nav2(command)

    def _resolve_nav2(self, command: Any) -> Any:
        context = self._step3_pending_context
        if context is None:
            return self._resolve_frozen_nav2(command)
        return self.step3_live_coordinator.resolve(
            command,
            advisor_snapshot=context["advisor_snapshot"],
            instruction=context["instruction"],
            agent_pose=context["agent_pose"],
            request_sim_ns=int(self.get_clock().now().nanoseconds),
        )


class _LaneBStep3ServerMixin:
    node: LaneBStep3ClientNode

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        if str(request.get("operation", "")) != "step":
            return super()._handle(request)  # type: ignore[misc]
        snapshot = request.get("advisor_snapshot")
        if snapshot is not None and not isinstance(snapshot, dict):
            raise ClientFailure(
                STATUS_INVALID_REQUEST, "advisor_snapshot must be an object"
            )
        if self.node._step3_pending_context is not None:
            raise ClientFailure(
                STATUS_INVALID_REQUEST, "concurrent Step3 step is forbidden"
            )
        gps = tuple(float(value) for value in request.get("global_gps", [0, 0, 0]))
        rotation = tuple(
            float(value)
            for value in request.get("global_rotation", [0, 0, 0, 1])
        )
        self.node._step3_pending_context = {
            "advisor_snapshot": snapshot,
            "instruction": str(request.get("instruction", "")),
            "agent_pose": gps + rotation,
        }
        try:
            return super()._handle(request)  # type: ignore[misc]
        finally:
            self.node._step3_pending_context = None


class LaneBStep3LocalIPCServer(_LaneBStep3ServerMixin, LocalIPCServer):
    pass


class LaneBStep3TCPIPCServer(_LaneBStep3ServerMixin, TCPIPCServer):
    pass


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LaneBStep3ClientNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    bind_host = os.environ.get("INTERNVLA_CLIENT_TCP_BIND_HOST", "")
    if bind_host:
        bind_port = int(os.environ.get("INTERNVLA_CLIENT_TCP_PORT", "25139"))
        server: LocalIPCServer = LaneBStep3TCPIPCServer(
            node,
            bind_host,
            bind_port,
            os.environ.get("INTERNVLA_CLIENT_TCP_EXPECTED_PEER", "10.100.120.111"),
        )
        endpoint = f"tcp://{bind_host}:{bind_port}"
    else:
        socket_path = Path(
            os.environ.get("INTERNVLA_CLIENT_SOCKET", "/tmp/internvla_client.sock")
        )
        server = LaneBStep3LocalIPCServer(node, socket_path)
        endpoint = str(socket_path)
    try:
        thread.start()
        node.wait_for_graph()
        if node.step3_direct_high_level:
            node.model_health_start = node.health()
        else:
            node.model_health_start = validate_uninitialized_model_health(node.health())
        node._write_client_summary("READY")
        node.get_logger().info(f"Lane-B Step3 evaluator IPC ready at {endpoint}")
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop_event.set()
        node.cancel_current()
        node.finish_records()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
