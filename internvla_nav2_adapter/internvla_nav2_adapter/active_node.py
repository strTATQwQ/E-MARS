"""Sampled closed-loop bridge from InternVLA metric goals to Nav2 cmd_vel."""

from __future__ import annotations

import json
import hashlib
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from internvla_ros2_msgs.msg import NavigationCommand
from internvla_ros2_msgs.srv import ResolveCommand
from nav2_msgs.action import FollowPath, NavigateToPose
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .motion_primitives import system2_local_poses
from .system1_target import FrozenSystem1Target


STATUS_OK = 0
STATUS_INVALID_REQUEST = 2
STATUS_STALE = 3
STATUS_TIMEOUT = 4
STATUS_INTERNAL_ERROR = 8
ACTION_STAND = -1
ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
SOURCE_SYSTEM2 = 1
SOURCE_SYSTEM1_NEW = 2
SOURCE_SYSTEM1_QUEUE = 3
CONTINUOUS_ODOM_FRESHNESS_SEC = 1.0
TERMINAL_GOAL_STATUSES = frozenset(
    {
        GoalStatus.STATUS_SUCCEEDED,
        GoalStatus.STATUS_CANCELED,
        GoalStatus.STATUS_ABORTED,
    }
)


def _time_ns(message: Any) -> int:
    return int(message.sec) * 1_000_000_000 + int(message.nanosec)


def _yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _future_result(future: Any, timeout_sec: float) -> Any:
    event = threading.Event()
    future.add_done_callback(lambda _: event.set())
    if not event.wait(timeout_sec):
        raise TimeoutError
    exception = future.exception()
    if exception is not None:
        raise exception
    return future.result()


def _semantic_now(node: Any) -> float:
    """Return the functional clock while preserving legacy wall semantics."""

    if not getattr(node, "_t5_sim_time_semantics", False):
        return time.monotonic()
    now_ns = int(node.get_clock().now().nanoseconds)
    if now_ns <= 0:
        return 0.0
    clock_lock = getattr(node, "_semantic_clock_lock", None)
    if clock_lock is None:
        last_ns = int(getattr(node, "_last_semantic_clock_ns", 0))
        if now_ns < last_ns:
            return 0.0
        node._last_semantic_clock_ns = now_ns
    else:
        with clock_lock:
            last_ns = int(getattr(node, "_last_semantic_clock_ns", 0))
            if now_ns < last_ns:
                return 0.0
            node._last_semantic_clock_ns = now_ns
    return now_ns / 1_000_000_000


def _semantic_age(node: Any, stamp: float) -> float | None:
    now = _semantic_now(node)
    if getattr(node, "_t5_sim_time_semantics", False) and (
        now <= 0.0 or stamp <= 0.0 or now < stamp
    ):
        return None
    return now - stamp


def _load_onboard_restart_identity() -> tuple[str, int, int] | None:
    session_value = os.environ.get(
        "INTERNVLA_T5_ONBOARD_RESTART_SESSION_FILE", ""
    )
    if not session_value:
        return None
    from internvla_ros2.fault_injection import (
        fault_profile_enabled,
        load_fault_restart_session,
    )

    if not fault_profile_enabled():
        raise RuntimeError(
            "onboard session restore requires the exact T5 fault profile"
        )
    lane = os.environ.get("INTERNNAV_T5_LANE", "")
    session = load_fault_restart_session(
        Path(session_value),
        expected_lane=lane,
        expected_action="dgx_ros_node_restart",
    )
    return (
        str(session["episode_id"]),
        int(session["reset_generation"]),
        int(session["last_sequence_id"]),
    )


class ActiveNav2Adapter(Node):
    def __init__(self) -> None:
        super().__init__("internvla_nav2_active_adapter")
        self._t5_sim_time_semantics = bool(
            os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
            and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
            and os.environ.get("INTERNNAV_T5_LANE", "") in {"a", "b"}
        )
        self.declare_parameter(
            "result_dir", os.environ.get("INTERNVLA_NAV2_ACTIVE_RESULT_DIR", "")
        )
        self.declare_parameter("goal_max_distance_m", 2.0)
        self.declare_parameter("goal_min_distance_m", 0.05)
        self.declare_parameter("nav2_server_timeout_sec", 3.0)
        self.declare_parameter("cmd_wait_timeout_sec", 4.0)
        self.declare_parameter("reset_settle_sec", 0.75)
        self.declare_parameter("navigate_goal_update_distance_m", 0.40)
        self.declare_parameter("navigate_goal_update_period_sec", 3.0)
        self.declare_parameter("angular_threshold_rad_sec", 0.08)
        self.declare_parameter("flash_forward_step_m", 0.25)
        self.declare_parameter("flash_turn_threshold_deg", 15.0)
        self.declare_parameter("command_mode", "navigate_to_pose")
        self.declare_parameter("execution_mode", "sampled_flash")
        self.declare_parameter("allow_command_pose_anchor_fallback", False)
        result_value = str(self.get_parameter("result_dir").value)
        if not result_value:
            raise RuntimeError("result_dir is required")
        self.result_dir = Path(result_value).resolve()
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.result_dir / "active_records.jsonl"
        if self.records_path.exists():
            raise RuntimeError("refusing to append existing active adapter record")
        self.maximum_distance = float(self.get_parameter("goal_max_distance_m").value)
        self.minimum_distance = float(self.get_parameter("goal_min_distance_m").value)
        if self.maximum_distance <= 0.0 or not (
            0.0 < self.minimum_distance < self.maximum_distance
        ):
            raise RuntimeError("invalid local navigation distance bounds")
        self.server_timeout = float(self.get_parameter("nav2_server_timeout_sec").value)
        self.cmd_wait_timeout = float(self.get_parameter("cmd_wait_timeout_sec").value)
        self.reset_settle_sec = float(self.get_parameter("reset_settle_sec").value)
        self.navigate_goal_update_distance = float(
            self.get_parameter("navigate_goal_update_distance_m").value
        )
        self.navigate_goal_update_period = float(
            self.get_parameter("navigate_goal_update_period_sec").value
        )
        if (
            self.navigate_goal_update_distance <= 0.0
            or self.navigate_goal_update_period <= 0.0
        ):
            raise RuntimeError("invalid NavigateToPose rolling-goal update thresholds")
        self.angular_threshold = float(
            self.get_parameter("angular_threshold_rad_sec").value
        )
        self.flash_forward_step = float(self.get_parameter("flash_forward_step_m").value)
        self.flash_turn_threshold = math.radians(
            float(self.get_parameter("flash_turn_threshold_deg").value)
        )
        self.command_mode = str(self.get_parameter("command_mode").value)
        if self.command_mode not in {"navigate_to_pose", "follow_path"}:
            raise RuntimeError("command_mode must be navigate_to_pose or follow_path")
        self.execution_mode = str(self.get_parameter("execution_mode").value)
        if self.execution_mode not in {"sampled_flash", "continuous"}:
            raise RuntimeError("execution_mode must be sampled_flash or continuous")
        self.allow_command_pose_anchor_fallback = bool(
            self.get_parameter("allow_command_pose_anchor_fallback").value
        )
        if self.allow_command_pose_anchor_fallback and not (
            os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
            and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
        ):
            raise RuntimeError(
                "command-pose anchor fallback is restricted to Isaac completion_sim"
            )
        if self._t5_sim_time_semantics and (
            not self.has_parameter("use_sim_time")
            or not bool(self.get_parameter("use_sim_time").value)
        ):
            raise RuntimeError("T5 completion_sim requires use_sim_time=true")
        self._semantic_clock_lock = threading.Lock()
        self._last_semantic_clock_ns = 0
        self.callback_group = ReentrantCallbackGroup()
        self.operation_lock = threading.RLock()
        self.cmd_condition = threading.Condition()
        self.latest_cmd = (0.0, 0.0)
        self.latest_cmd_monotonic = 0.0
        self.cmd_serial = 0
        self.plan_condition = threading.Condition()
        self.latest_plan: list[tuple[float, float]] = []
        self.latest_plan_monotonic = 0.0
        self.plan_serial = 0
        self.odom_lock = threading.Lock()
        self.latest_odom: tuple[float, float, float] | None = None
        self.latest_odom_monotonic = 0.0
        self.active_goal: Any = None
        self.last_navigate_goal: tuple[float, float, float] | None = None
        self.last_navigate_goal_monotonic = 0.0
        self.navigate_goal_reuse_count = 0
        self.frozen_system1_target: FrozenSystem1Target | None = None
        self.system1_target_reissue_count = 0
        self.current_system1_target_reissued = False
        self.current_system1_target_sha256: str | None = None
        restart_identity = _load_onboard_restart_identity()
        if restart_identity is None:
            self.active_episode = ""
            self.active_generation = -1
            self.last_sequence = -1
        else:
            (
                self.active_episode,
                self.active_generation,
                self.last_sequence,
            ) = restart_identity
        self.command_count = 0
        self.goal_count = 0
        self.nav2_control_count = 0
        self.system2_passthrough_count = 0
        self.system2_nav2_count = 0
        self.quantization_count = 0
        self.direct_motion_bypass_count = 0
        self.failure_count = 0
        self.command_pose_anchor_fallback_count = 0
        self.current_path_anchor_source: str | None = None
        self.current_path_anchor_odom_age_sec: float | None = None
        self.current_command_pose_anchor_fallback = False
        self.last_path_anchor_source: str | None = None
        self.last_path_anchor_odom_age_sec: float | None = None
        self.reset_settle_until = 0.0
        self.started = time.time()
        self.navigate_client = ActionClient(
            self,
            NavigateToPose,
            "navigate_to_pose",
            callback_group=self.callback_group,
        )
        self.follow_client = ActionClient(
            self,
            FollowPath,
            "follow_path",
            callback_group=self.callback_group,
        )
        self.motion_publisher = self.create_publisher(
            Bool, "/internvla/nav2_motion_enabled", 20
        )
        self.active_path_publisher = self.create_publisher(
            NavPath, "/internvla/nav2_active_path", 20
        )
        self.create_subscription(
            Twist,
            "cmd_vel_safe",
            self._on_cmd,
            20,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            NavPath,
            "plan",
            self._on_plan,
            20,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Odometry,
            "/odom",
            self._on_odom,
            50,
            callback_group=self.callback_group,
        )
        self.stop_ack_publisher = None
        if self._t5_sim_time_semantics:
            self.create_subscription(
                Bool,
                "/internvla/stop",
                self._on_safe_stop,
                20,
                callback_group=self.callback_group,
            )
            self.create_subscription(
                String,
                "/internvla/t5_nav2_stop_request",
                self._on_stop_request,
                20,
                callback_group=self.callback_group,
            )
            self.stop_ack_publisher = self.create_publisher(
                String, "/internvla/t5_nav2_stop_ack", 20
            )
        self.create_service(
            ResolveCommand,
            "/internvla/nav2_resolve",
            self._resolve,
            callback_group=self.callback_group,
        )
        self._write_summary("READY")

    def _on_cmd(self, message: Twist) -> None:
        with self.cmd_condition:
            self.latest_cmd = (float(message.linear.x), float(message.angular.z))
            self.latest_cmd_monotonic = _semantic_now(self)
            self.cmd_serial += 1
            self.cmd_condition.notify_all()

    def _on_plan(self, message: NavPath) -> None:
        points = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in message.poses
        ]
        if not points:
            return
        with self.plan_condition:
            self.latest_plan = points
            self.latest_plan_monotonic = _semantic_now(self)
            self.plan_serial += 1
            self.plan_condition.notify_all()
        # NavigateToPose obtains its executable path from Nav2's /plan topic.
        # Recovery must supervise that same world-frame path; publishing only
        # FollowPath command paths leaves the default NavigateToPose mode blind.
        self.active_path_publisher.publish(message)

    def _on_odom(self, message: Odometry) -> None:
        orientation = message.pose.pose.orientation
        with self.odom_lock:
            self.latest_odom = (
                float(message.pose.pose.position.x),
                float(message.pose.pose.position.y),
                _yaw(
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                ),
            )
            self.latest_odom_monotonic = _semantic_now(self)

    def _on_safe_stop(self, message: Bool) -> None:
        if not bool(message.data):
            return
        # Cancellation is confirmed through the identity-bound request/ack
        # topic below.  This Bool independently disables command forwarding.
        with self.operation_lock:
            if self.execution_mode == "continuous":
                self._publish_motion(False)

    def _on_stop_request(self, message: String) -> None:
        if not self._t5_sim_time_semantics or self.stop_ack_publisher is None:
            return
        value: dict[str, Any] = {}
        token = ""
        episode_id = ""
        reset_generation = -1
        sequence_id = -2
        request_id = ""
        try:
            decoded = json.loads(str(message.data))
            if not isinstance(decoded, dict) or decoded.get("schema_version") != 1:
                raise ValueError("invalid stop request schema")
            value = decoded
            token = str(value.get("token", ""))
            episode_id = str(value.get("episode_id", ""))
            reset_generation = int(value.get("reset_generation", -1))
            sequence_id = int(value.get("sequence_id", -2))
            request_id = str(value.get("request_id", ""))
            if (
                not token
                or len(token) > 256
                or not episode_id
                or reset_generation < 0
                or sequence_id < -1
                or not request_id
            ):
                raise ValueError("invalid stop request identity")
            with self.operation_lock:
                if self.execution_mode == "continuous":
                    self._publish_motion(False)
                preserve_candidate = bool(
                    self.execution_mode == "continuous"
                    and episode_id == self.active_episode
                    and reset_generation == self.active_generation
                    and sequence_id == self.last_sequence
                    and str(value.get("reason", ""))
                    == "measured motion reached the preregistered completion threshold"
                )
                remaining_distance = None
                preserve_system1_target = False
                if preserve_candidate and self.frozen_system1_target is not None:
                    with self.odom_lock:
                        current_odom = self.latest_odom
                    if current_odom is not None:
                        remaining_distance = (
                            self.frozen_system1_target.remaining_xy_distance(
                                current_odom[0], current_odom[1]
                            )
                        )
                        # A queued command is judged as another bounded forward
                        # primitive by the client gate.  Do not reissue an old
                        # absolute goal when less than one such primitive remains:
                        # Nav2 will decelerate inside its goal tolerance and the
                        # fresh-primitive gate could otherwise never reach 0.20 m.
                        preserve_system1_target = (
                            self.frozen_system1_target.permits_bounded_reissue(
                                current_odom[0],
                                current_odom[1],
                                self.flash_forward_step,
                            )
                        )
                cancel_warning = ""
                try:
                    if preserve_system1_target:
                        cancelled = self._cancel_active(
                            wait=True, preserve_system1_target=True
                        )
                    else:
                        cancelled = self._cancel_active(wait=True)
                except TimeoutError:
                    # completion_sim has already disabled command forwarding above.
                    # A late Nav2 cancel acknowledgement must not abort the entire
                    # episode: leave a second best-effort cancel in flight, clear the
                    # local active handle, and admit only the next identity-bound
                    # primitive.  Empty/negative acknowledgements remain fatal, and
                    # this callback is not enabled by the real-Go2 configuration.
                    cancelled = self._cancel_active(
                        wait=False,
                        preserve_system1_target=preserve_system1_target,
                    )
                    if cancelled:
                        cancel_warning = (
                            "WARN completion_sim Nav2 cancel acknowledgement exceeded "
                            "1.0 wall-s; motion disabled and best-effort cancel issued"
                        )
                        self.get_logger().warning(cancel_warning)
                if not cancelled:
                    raise RuntimeError("Nav2 cancellation was not confirmed")
            status = "ok"
            detail = cancel_warning or "Nav2 goal absent or cancellation confirmed"
            if preserve_candidate and remaining_distance is not None:
                detail += (
                    "; frozen System1 target "
                    + ("preserved" if preserve_system1_target else "cleared")
                    + f" with {remaining_distance:.6f} m remaining"
                )
        except BaseException as exc:
            status = "error"
            detail = repr(exc)[:512]
            token = str(value.get("token", ""))
            episode_id = str(value.get("episode_id", ""))
            request_id = str(value.get("request_id", ""))
            try:
                reset_generation = int(value.get("reset_generation", -1))
                sequence_id = int(value.get("sequence_id", -2))
            except (TypeError, ValueError):
                reset_generation = -1
                sequence_id = -2
        ack = String()
        ack.data = json.dumps(
            {
                "schema_version": 1,
                "token": token,
                "episode_id": episode_id,
                "reset_generation": reset_generation,
                "sequence_id": sequence_id,
                "request_id": request_id,
                "status": status,
                "detail": detail,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self.stop_ack_publisher.publish(ack)

    def _path_anchor(
        self, command: NavigationCommand
    ) -> tuple[float, float, float, str]:
        if self.execution_mode == "continuous":
            with self.odom_lock:
                odom = self.latest_odom
                odom_age = (
                    None
                    if odom is None
                    else _semantic_age(self, self.latest_odom_monotonic)
                )
            self.current_path_anchor_odom_age_sec = odom_age
            self.last_path_anchor_odom_age_sec = odom_age
            if (
                odom is not None
                and odom_age is not None
                and odom_age <= CONTINUOUS_ODOM_FRESHNESS_SEC
            ):
                self.current_path_anchor_source = "live_navigation_odometry"
                self.last_path_anchor_source = self.current_path_anchor_source
                return odom[0], odom[1], odom[2], "odom"
            if not self.allow_command_pose_anchor_fallback:
                self.current_path_anchor_source = (
                    "missing_navigation_odometry_rejected"
                    if odom is None
                    else "stale_navigation_odometry_rejected"
                )
                self.last_path_anchor_source = self.current_path_anchor_source
                raise TimeoutError
            try:
                anchor = self._validated_command_pose_anchor(command)
            except (TypeError, ValueError, OverflowError) as exc:
                self.current_path_anchor_source = (
                    "command_navigation_odometry_fallback_invalid"
                )
                self.last_path_anchor_source = self.current_path_anchor_source
                raise ValueError("invalid command navigation pose fallback") from exc
            self.command_pose_anchor_fallback_count += 1
            self.current_command_pose_anchor_fallback = True
            self.current_path_anchor_source = (
                "command_navigation_odometry_fallback"
            )
            self.last_path_anchor_source = self.current_path_anchor_source
            return anchor
        rotation = command.global_rotation_wxyz
        return (
            float(command.global_gps[0]),
            float(command.global_gps[1]),
            _yaw(rotation[1], rotation[2], rotation[3], rotation[0]),
            "map",
        )

    @staticmethod
    def _validated_command_pose_anchor(
        command: NavigationCommand,
    ) -> tuple[float, float, float, str]:
        gps = [float(value) for value in command.global_gps]
        rotation = [float(value) for value in command.global_rotation_wxyz]
        if len(gps) != 3 or len(rotation) != 4:
            raise ValueError("command navigation pose has invalid dimensions")
        if not all(math.isfinite(value) for value in gps + rotation):
            raise ValueError("command navigation pose contains NaN/Inf")
        norm = math.sqrt(sum(value * value for value in rotation))
        if norm < 0.5:
            raise ValueError("command navigation pose quaternion is invalid")
        rotation = [value / norm for value in rotation]
        return (
            gps[0],
            gps[1],
            _yaw(rotation[1], rotation[2], rotation[3], rotation[0]),
            "odom",
        )

    def _resolve(
        self, request: ResolveCommand.Request, response: ResolveCommand.Response
    ) -> ResolveCommand.Response:
        started = time.monotonic()
        command = request.command
        with self.operation_lock:
            self.command_count += 1
            self.current_path_anchor_source = None
            self.current_path_anchor_odom_age_sec = None
            self.current_command_pose_anchor_fallback = False
            self.current_system1_target_reissued = False
            self.current_system1_target_sha256 = None
            response.episode_id = command.episode_id
            response.reset_generation = command.reset_generation
            response.sequence_id = command.sequence_id
            response.request_id = command.request_id
            response.discrete_action = ACTION_STOP
            response.stop = True
            record: dict[str, Any] = {
                "schema_version": 1,
                "command_index": self.command_count - 1,
                "episode_id": command.episode_id,
                "reset_generation": int(command.reset_generation),
                "sequence_id": int(command.sequence_id),
                "request_id": command.request_id,
                "model_action": int(command.discrete_action),
                "action_source": int(command.action_source),
                "trajectory_valid": bool(command.trajectory_valid),
                "nav2_goal_sent": False,
                "nav2_plan_valid": False,
            }
            if command.trajectory_valid:
                local_path_xy = [
                    [float(pose.pose.position.x), float(pose.pose.position.y)]
                    for pose in command.local_path.poses
                ]
                encoded_path = json.dumps(
                    local_path_xy, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
                record["local_path_xy"] = local_path_xy
                record["local_path_sha256"] = hashlib.sha256(encoded_path).hexdigest()
            try:
                self._check_identity(command)
                now_ns = int(self.get_clock().now().nanoseconds)
                if (
                    (getattr(self, "_t5_sim_time_semantics", False) and now_ns <= 0)
                    or now_ns > _time_ns(command.valid_until)
                ):
                    raise ValueError("stale command")
                action, goal_sent, plan_valid, cmd = self._select_action(command)
                response.status_code = STATUS_OK
                response.status_message = "resolved"
                response.discrete_action = action
                response.stop = action == ACTION_STOP
                response.nav2_goal_sent = goal_sent
                response.nav2_plan_valid = plan_valid
                record.update(
                    {
                        "selected_action": action,
                        "nav2_goal_sent": goal_sent,
                        "nav2_plan_valid": plan_valid,
                        "cmd_vel_linear_x": cmd[0],
                        "cmd_vel_angular_z": cmd[1],
                        "system1_target_reissued": (
                            self.current_system1_target_reissued
                        ),
                    }
                )
            except ValueError as exc:
                self.failure_count += 1
                response.status_code = (
                    STATUS_STALE if str(exc) == "stale command" else STATUS_INVALID_REQUEST
                )
                response.status_message = str(exc)
                record["error"] = str(exc)
            except TimeoutError:
                self.failure_count += 1
                response.status_code = STATUS_TIMEOUT
                response.status_message = "Nav2 goal/cmd_vel timeout"
                record["error"] = response.status_message
            except BaseException as exc:
                self.failure_count += 1
                response.status_code = STATUS_INTERNAL_ERROR
                response.status_message = f"Nav2 resolver error: {exc!r}"[:512]
                record["error"] = response.status_message
            record.update(
                {
                    "command_pose_anchor_fallback": (
                        self.current_command_pose_anchor_fallback
                    ),
                    "command_pose_anchor_fallback_count": (
                        self.command_pose_anchor_fallback_count
                    ),
                    "path_anchor_odom_age_sec": (
                        self.current_path_anchor_odom_age_sec
                    ),
                    "path_anchor_source": self.current_path_anchor_source,
                    "system1_target_sha256": self.current_system1_target_sha256,
                }
            )
            response.resolution_latency_sec = float(time.monotonic() - started)
            record["resolution_latency_sec"] = response.resolution_latency_sec
            self._append(record)
            self._write_summary("RUNNING" if not self.failure_count else "FAIL_SO_FAR")
            return response

    def _check_identity(self, command: NavigationCommand) -> None:
        generation = int(command.reset_generation)
        sequence = int(command.sequence_id)
        if self.active_generation < 0:
            if sequence != 0:
                raise ValueError("first sequence is not zero")
        elif generation == self.active_generation:
            if command.episode_id != self.active_episode or sequence != self.last_sequence + 1:
                raise ValueError("non-contiguous command identity")
        elif generation > self.active_generation:
            if sequence != 0:
                raise ValueError("invalid reset barrier")
            self._cancel_active(wait=True)
            if self.execution_mode == "continuous":
                self._publish_motion(False)
            with self.plan_condition:
                self.latest_plan = []
                self.latest_plan_monotonic = 0.0
            if getattr(self, "_t5_sim_time_semantics", False):
                with self.odom_lock:
                    self.latest_odom = None
                    self.latest_odom_monotonic = 0.0
                self.reset_settle_until = 0.0
            else:
                self.reset_settle_until = time.monotonic() + self.reset_settle_sec
        else:
            raise ValueError("invalid reset generation")
        self.active_generation = generation
        self.active_episode = command.episode_id
        self.last_sequence = sequence

    def _select_action(
        self, command: NavigationCommand
    ) -> tuple[int, bool, bool, tuple[float, float]]:
        if self.execution_mode == "continuous":
            return self._select_continuous(command)
        model_action = int(command.discrete_action)
        if command.stop or model_action == ACTION_STOP:
            self._cancel_active()
            return ACTION_STOP, False, False, (0.0, 0.0)
        if int(command.action_source) == SOURCE_SYSTEM2:
            self.system2_passthrough_count += 1
            self.direct_motion_bypass_count += int(model_action not in {ACTION_STAND, ACTION_STOP})
            return model_action, False, False, (0.0, 0.0)
        if int(command.action_source) not in {SOURCE_SYSTEM1_NEW, SOURCE_SYSTEM1_QUEUE}:
            raise ValueError("unknown model action provenance")

        before_plan_serial = self.plan_serial
        before_cmd_serial = self.cmd_serial
        goal_sent = False
        if command.trajectory_valid:
            remaining_settle = (
                0.0
                if getattr(self, "_t5_sim_time_semantics", False)
                else self.reset_settle_until - time.monotonic()
            )
            if remaining_settle > 0.0:
                time.sleep(remaining_settle)
            navigation_target = (
                self._follow_path_from_command(command)
                if self.command_mode == "follow_path"
                else self._goal_from_path(command)
            )
            if navigation_target is None:
                # A degenerate fresh trajectory retains the official safe action.
                return model_action, False, False, (0.0, 0.0)
            if self.command_mode == "follow_path":
                self._send_path(navigation_target)
            else:
                goal_sent = self._send_goal(navigation_target)
            if self.command_mode == "follow_path":
                goal_sent = True
            self.goal_count += int(goal_sent)
        if self.active_goal is None:
            return model_action, False, False, (0.0, 0.0)

        deadline = time.monotonic() + self.cmd_wait_timeout
        if self.command_mode == "follow_path":
            with self.cmd_condition:
                while time.monotonic() < deadline:
                    linear, angular = self.latest_cmd
                    cmd_age = _semantic_age(self, self.latest_cmd_monotonic)
                    fresh_nonzero = (
                        self.cmd_serial > before_cmd_serial
                        and cmd_age is not None
                        and cmd_age <= 1.0
                        and (abs(linear) > 1e-4 or abs(angular) > self.angular_threshold)
                    )
                    if fresh_nonzero:
                        break
                    self.cmd_condition.wait(max(0.0, deadline - time.monotonic()))
                else:
                    if int(command.action_source) == SOURCE_SYSTEM1_QUEUE:
                        return model_action, False, False, (0.0, 0.0)
                    raise TimeoutError
                cmd = self.latest_cmd
            action = self._quantize(cmd)
            self.quantization_count += 1
            self.nav2_control_count += 1
            return action, goal_sent, True, cmd

        if goal_sent:
            with self.plan_condition:
                while self.plan_serial <= before_plan_serial and time.monotonic() < deadline:
                    self.plan_condition.wait(deadline - time.monotonic())
        with self.plan_condition:
            plan = list(self.latest_plan)
            plan_age = _semantic_age(self, self.latest_plan_monotonic)
        if not plan or plan_age is None or plan_age > 10.0:
            if int(command.action_source) == SOURCE_SYSTEM1_QUEUE:
                return model_action, False, False, (0.0, 0.0)
            raise TimeoutError
        with self.cmd_condition:
            cmd = self.latest_cmd
        action = self._quantize_plan(command, plan)
        self.quantization_count += 1
        self.nav2_control_count += 1
        return action, goal_sent, True, cmd

    def _select_continuous(
        self, command: NavigationCommand
    ) -> tuple[int, bool, bool, tuple[float, float]]:
        model_action = int(command.discrete_action)
        if command.stop or model_action == ACTION_STOP:
            self._cancel_active()
            self._publish_motion(False)
            return ACTION_STOP, False, False, (0.0, 0.0)

        source = int(command.action_source)
        if source == SOURCE_SYSTEM2:
            self._clear_frozen_system1_target()
            if model_action == ACTION_STAND:
                self._cancel_active()
                self._publish_motion(False)
                return ACTION_STAND, False, False, (0.0, 0.0)
            if model_action not in {ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT}:
                raise ValueError("unsupported System 2 continuous action")
            path = self._system2_path(command, model_action)
            if self.command_mode == "navigate_to_pose":
                goal_sent = self._send_goal(path.poses[-1])
            else:
                self._send_path(path)
                goal_sent = True
            self.system2_nav2_count += 1
            self.goal_count += int(goal_sent)
            cmd = self._latest_continuous_cmd()
            self.nav2_control_count += 1
            self._publish_motion(True)
            return ACTION_STAND, goal_sent, True, cmd

        if source not in {SOURCE_SYSTEM1_NEW, SOURCE_SYSTEM1_QUEUE}:
            raise ValueError("unknown model action provenance")

        goal_sent = False
        if command.trajectory_valid:
            remaining_settle = (
                0.0
                if getattr(self, "_t5_sim_time_semantics", False)
                else self.reset_settle_until - time.monotonic()
            )
            if remaining_settle > 0.0:
                time.sleep(remaining_settle)
            navigation_target = (
                self._goal_from_path(command)
                if self.command_mode == "navigate_to_pose"
                else self._follow_path_from_command(command)
            )
            if navigation_target is None:
                self._cancel_active()
                self._publish_motion(False)
                return ACTION_STAND, False, False, (0.0, 0.0)
            if self.command_mode == "navigate_to_pose":
                goal_sent = self._send_goal(navigation_target)
                self._freeze_system1_target(command, navigation_target)
            else:
                self._send_path(navigation_target)
                goal_sent = True
                # A FollowPath command is already transformed into an absolute
                # frame, but this minimal T5 fix intentionally does not invent a
                # new path-cache contract.  A queued action after cancellation
                # therefore fails closed and requests a fresh trajectory.
                self._clear_frozen_system1_target()
            self.goal_count += int(goal_sent)
        elif source == SOURCE_SYSTEM1_QUEUE:
            frozen_target = self._require_frozen_system1_target(command)
            self.current_system1_target_sha256 = frozen_target.target_sha256
            if self.active_goal is None:
                if self.command_mode != "navigate_to_pose":
                    raise ValueError(
                        "System 1 FollowPath queue requires a fresh absolute trajectory"
                    )
                goal_sent = self._send_goal(
                    self._materialize_frozen_system1_target(command, frozen_target)
                )
                if not goal_sent or self.active_goal is None:
                    raise RuntimeError("System 1 absolute target reissue was not accepted")
                self.system1_target_reissue_count += 1
                self.current_system1_target_reissued = True
                self.goal_count += 1
        if self.active_goal is None:
            self._publish_motion(False)
            return ACTION_STAND, goal_sent, False, (0.0, 0.0)
        cmd = self._latest_continuous_cmd()
        self.nav2_control_count += 1
        self._publish_motion(True)
        return ACTION_STAND, goal_sent, True, cmd

    @staticmethod
    def _source_path_sha256(command: NavigationCommand) -> str:
        local_path_xy = [
            [float(pose.pose.position.x), float(pose.pose.position.y)]
            for pose in command.local_path.poses
        ]
        if not local_path_xy:
            raise ValueError("System 1 trajectory is empty")
        encoded = json.dumps(
            local_path_xy, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _freeze_system1_target(
        self, command: NavigationCommand, target: PoseStamped
    ) -> None:
        pose = target.pose
        frozen = FrozenSystem1Target(
            episode_id=str(command.episode_id),
            reset_generation=int(command.reset_generation),
            source_sequence_id=int(command.sequence_id),
            source_path_sha256=self._source_path_sha256(command),
            valid_until_ns=_time_ns(command.valid_until),
            frame_id=str(target.header.frame_id),
            position_xyz=(
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ),
            orientation_xyzw=(
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
        )
        self.frozen_system1_target = frozen
        self.current_system1_target_sha256 = frozen.target_sha256

    def _clear_frozen_system1_target(self) -> None:
        self.frozen_system1_target = None

    def _require_frozen_system1_target(
        self, command: NavigationCommand
    ) -> FrozenSystem1Target:
        frozen = self.frozen_system1_target
        if frozen is None:
            raise ValueError(
                "System 1 queue has no identity-bound absolute target; "
                "fresh trajectory required"
            )
        frozen.require_queue_binding(
            episode_id=str(command.episode_id),
            reset_generation=int(command.reset_generation),
            sequence_id=int(command.sequence_id),
            now_ns=int(self.get_clock().now().nanoseconds),
        )
        return frozen

    @staticmethod
    def _materialize_frozen_system1_target(
        command: NavigationCommand, frozen: FrozenSystem1Target
    ) -> PoseStamped:
        target = PoseStamped()
        target.header.stamp = command.header.stamp
        target.header.frame_id = frozen.frame_id
        (
            target.pose.position.x,
            target.pose.position.y,
            target.pose.position.z,
        ) = frozen.position_xyz
        (
            target.pose.orientation.x,
            target.pose.orientation.y,
            target.pose.orientation.z,
            target.pose.orientation.w,
        ) = frozen.orientation_xyzw
        return target

    def _latest_continuous_cmd(self) -> tuple[float, float]:
        """Return current Nav2 state without blocking the physics producer.

        Isaac's evaluator calls the resolver synchronously and advances the
        simulation only after it returns.  Continuous control therefore cannot
        wait for a post-goal cmd_vel sample: that would also pause odometry and
        depth, exactly when an obstacle makes Nav2 hold or abort a path.  Goal
        acceptance is the plan-valid signal; a missing/stale velocity is a
        safe zero until the asynchronous controller publishes again.
        """
        with self.cmd_condition:
            age = _semantic_age(self, self.latest_cmd_monotonic)
            if (
                self.cmd_serial > 0
                and age is not None
                and age <= 0.5
            ):
                return self.latest_cmd
        return (0.0, 0.0)

    def _wait_for_fresh_cmd(
        self, before_serial: int, *, require_new: bool = True
    ) -> tuple[float, float]:
        deadline = time.monotonic() + self.cmd_wait_timeout
        with self.cmd_condition:
            while time.monotonic() < deadline:
                serial_ok = self.cmd_serial > before_serial if require_new else self.cmd_serial > 0
                age = _semantic_age(self, self.latest_cmd_monotonic)
                fresh = age is not None and age <= 0.5
                # A fresh zero or sub-threshold velocity is still a valid Nav2
                # control result.  RPP legitimately converges through small
                # rotate-to-heading commands before it starts translating;
                # treating those commands as missing stalls path refreshes and
                # eventually lets RPP prune the old path to zero poses.
                if serial_ok and fresh:
                    return self.latest_cmd
                self.cmd_condition.wait(max(0.0, deadline - time.monotonic()))
        raise TimeoutError

    def _system2_path(
        self, command: NavigationCommand, action: int
    ) -> NavPath:
        anchor_x, anchor_y, base_yaw, frame_id = self._path_anchor(command)
        # Frozen T4 keeps its 0.35 m forward primitive and 0.45 m turn arc.
        # Exact T5 completion_sim uses the measured 0.25 m step and rotates
        # about the footprint center without unintended translation.
        forward_step = self.flash_forward_step if self._t5_sim_time_semantics else 0.35
        local_points = system2_local_poses(
            action,
            t5_completion_sim=self._t5_sim_time_semantics,
            forward_step_m=forward_step,
        )
        cosine, sine = math.cos(base_yaw), math.sin(base_yaw)
        path = NavPath()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = frame_id
        for local_x, local_y, yaw_delta in local_points:
            yaw = base_yaw + yaw_delta
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = (
                anchor_x + cosine * local_x - sine * local_y
            )
            pose.pose.position.y = (
                anchor_y + sine * local_x + cosine * local_y
            )
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        return path

    def _publish_motion(self, enabled: bool) -> None:
        message = Bool()
        message.data = bool(enabled)
        self.motion_publisher.publish(message)

    def _goal_from_path(self, command: NavigationCommand) -> PoseStamped | None:
        points = np.asarray(
            [[pose.pose.position.x, pose.pose.position.y] for pose in command.local_path.poses],
            dtype=np.float64,
        )
        if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
            raise ValueError("invalid local path shape")
        if not np.isfinite(points).all():
            raise ValueError("local path contains NaN/Inf")
        radial = np.linalg.norm(points - points[0], axis=1)
        candidates = np.nonzero(radial <= self.maximum_distance + 1e-6)[0]
        target_index = int(candidates[-1]) if len(candidates) else 0
        if float(radial[target_index]) < self.minimum_distance:
            return None
        rotation = command.global_rotation_wxyz
        base_yaw = _yaw(rotation[1], rotation[2], rotation[3], rotation[0])
        local = points[target_index]
        cosine, sine = math.cos(base_yaw), math.sin(base_yaw)
        map_x = float(command.global_gps[0]) + cosine * float(local[0]) - sine * float(local[1])
        map_y = float(command.global_gps[1]) + sine * float(local[0]) + cosine * float(local[1])
        if target_index > 0:
            delta = points[target_index] - points[target_index - 1]
            goal_yaw = base_yaw + math.atan2(float(delta[1]), float(delta[0]))
        else:
            goal_yaw = base_yaw
        goal = PoseStamped()
        goal.header.stamp = command.header.stamp
        goal.header.frame_id = "map"
        goal.pose.position.x = map_x
        goal.pose.position.y = map_y
        goal.pose.orientation.z = math.sin(goal_yaw / 2.0)
        goal.pose.orientation.w = math.cos(goal_yaw / 2.0)
        return goal

    def _follow_path_from_command(self, command: NavigationCommand) -> NavPath | None:
        points = np.asarray(
            [[pose.pose.position.x, pose.pose.position.y] for pose in command.local_path.poses],
            dtype=np.float64,
        )
        if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
            raise ValueError("invalid local path shape")
        if not np.isfinite(points).all():
            raise ValueError("local path contains NaN/Inf")
        radial = np.linalg.norm(points - points[0], axis=1)
        candidates = np.nonzero(radial <= self.maximum_distance + 1e-6)[0]
        target_index = int(candidates[-1]) if len(candidates) else 0
        if float(radial[target_index]) < self.minimum_distance:
            return None
        anchor_x, anchor_y, base_yaw, frame_id = self._path_anchor(command)
        cosine, sine = math.cos(base_yaw), math.sin(base_yaw)
        path = NavPath()
        path.header.stamp = (
            self.get_clock().now().to_msg()
            if self.execution_mode == "continuous"
            else command.header.stamp
        )
        path.header.frame_id = frame_id
        for index, local in enumerate(points[: target_index + 1]):
            if index + 1 <= target_index:
                delta = points[index + 1] - local
            elif index:
                delta = local - points[index - 1]
            else:
                delta = np.asarray([1.0, 0.0])
            pose_yaw = base_yaw + math.atan2(float(delta[1]), float(delta[0]))
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = (
                anchor_x + cosine * float(local[0])
                - sine * float(local[1])
            )
            pose.pose.position.y = (
                anchor_y + sine * float(local[0])
                + cosine * float(local[1])
            )
            pose.pose.orientation.z = math.sin(pose_yaw / 2.0)
            pose.pose.orientation.w = math.cos(pose_yaw / 2.0)
            path.poses.append(pose)
        return path

    def _send_goal(self, pose: PoseStamped) -> bool:
        now = _semantic_now(self)
        pose_yaw = _yaw(
            float(pose.pose.orientation.x),
            float(pose.pose.orientation.y),
            float(pose.pose.orientation.z),
            float(pose.pose.orientation.w),
        )
        candidate = (
            float(pose.pose.position.x),
            float(pose.pose.position.y),
            pose_yaw,
        )
        if (
            self.execution_mode == "continuous"
            and self.active_goal is not None
            and self.last_navigate_goal is not None
        ):
            distance = math.hypot(
                candidate[0] - self.last_navigate_goal[0],
                candidate[1] - self.last_navigate_goal[1],
            )
            yaw_delta = abs(
                math.atan2(
                    math.sin(candidate[2] - self.last_navigate_goal[2]),
                    math.cos(candidate[2] - self.last_navigate_goal[2]),
                )
            )
            goal_age = _semantic_age(self, self.last_navigate_goal_monotonic)
            if (
                distance < self.navigate_goal_update_distance
                and yaw_delta < math.radians(30.0)
                and goal_age is not None
                and goal_age < self.navigate_goal_update_period
            ):
                self.navigate_goal_reuse_count += 1
                return False
        deadline = time.monotonic() + self.server_timeout
        handle = None
        while time.monotonic() < deadline:
            if not self.navigate_client.wait_for_server(timeout_sec=0.2):
                continue
            # Continuous NavigateToPose goals are rolling local-goal updates.
            # Let Nav2 preempt the previous accepted goal atomically; an
            # explicit asynchronous cancel can tear down the replacement BT.
            if self.execution_mode != "continuous":
                self._cancel_active()
            goal = NavigateToPose.Goal()
            goal.pose = pose
            remaining = max(0.1, deadline - time.monotonic())
            handle = _future_result(
                self.navigate_client.send_goal_async(goal), remaining
            )
            if handle.accepted:
                break
            handle = None
            time.sleep(0.1)
        if handle is None:
            raise TimeoutError
        self.active_goal = handle
        self.last_navigate_goal = candidate
        self.last_navigate_goal_monotonic = now
        result_future = handle.get_result_async()
        result_future.add_done_callback(lambda _: self._goal_finished(handle))
        return True

    def _send_path(self, path: NavPath) -> None:
        deadline = time.monotonic() + self.server_timeout
        handle = None
        while time.monotonic() < deadline:
            if not self.follow_client.wait_for_server(timeout_sec=0.2):
                continue
            # FollowPath's action server accepts a new goal as an in-place path
            # update.  Explicitly canceling the previous handle here races the
            # accepted replacement and can clear the controller plan.  Resets,
            # STOP and shutdown still cancel through _cancel_active().
            goal = FollowPath.Goal()
            goal.path = path
            remaining = max(0.1, deadline - time.monotonic())
            handle = _future_result(self.follow_client.send_goal_async(goal), remaining)
            if handle.accepted:
                break
            handle = None
            time.sleep(0.1)
        if handle is None:
            raise TimeoutError
        self.active_goal = handle
        self.active_path_publisher.publish(path)
        result_future = handle.get_result_async()
        result_future.add_done_callback(lambda _: self._goal_finished(handle))

    def _goal_finished(self, handle: Any) -> None:
        with self.operation_lock:
            if self.active_goal is handle:
                self.active_goal = None

    def _cancel_active(
        self, wait: bool = False, *, preserve_system1_target: bool = False
    ) -> bool:
        handle = self.active_goal
        if handle is None:
            self.last_navigate_goal = None
            self.last_navigate_goal_monotonic = 0.0
            if not preserve_system1_target:
                self._clear_frozen_system1_target()
            return True
        try:
            future = handle.cancel_goal_async()
            if wait:
                result = _future_result(future, 1.0)
                cancel_confirmed = bool(
                    result is not None and getattr(result, "goals_canceling", [])
                )
                goal_terminal = int(getattr(handle, "status", 0)) in (
                    TERMINAL_GOAL_STATUSES
                )
                # Nav2 may finish the goal between the stop request and its cancel
                # response.  In that race goals_canceling is correctly empty, but
                # a terminal action status proves that no old motion remains.  The
                # result callback cannot clear active_goal while this callback owns
                # operation_lock, so inspect the handle status directly.
                if (
                    not cancel_confirmed
                    and not goal_terminal
                    and self.active_goal is handle
                ):
                    raise RuntimeError("Nav2 did not confirm goal cancellation")
            if self.active_goal is handle:
                self.active_goal = None
            self.last_navigate_goal = None
            self.last_navigate_goal_monotonic = 0.0
            if not preserve_system1_target:
                self._clear_frozen_system1_target()
            return True
        except BaseException:
            if wait:
                raise
            return False

    def _quantize(self, cmd: tuple[float, float]) -> int:
        linear, angular = cmd
        if linear > 1e-4:
            predicted_turn = angular * self.flash_forward_step / max(linear, 0.05)
            if predicted_turn > self.flash_turn_threshold:
                return ACTION_LEFT
            if predicted_turn < -self.flash_turn_threshold:
                return ACTION_RIGHT
            return ACTION_FORWARD
        if angular > self.angular_threshold:
            return ACTION_LEFT
        if angular < -self.angular_threshold:
            return ACTION_RIGHT
        return ACTION_STAND

    def _quantize_plan(
        self, command: NavigationCommand, plan: list[tuple[float, float]]
    ) -> int:
        current = np.asarray(command.global_gps[:2], dtype=np.float64)
        points = np.asarray(plan, dtype=np.float64)
        nearest = int(np.argmin(np.linalg.norm(points - current, axis=1)))
        target = points[-1]
        for point in points[nearest + 1 :]:
            if float(np.linalg.norm(point - current)) >= self.flash_forward_step:
                target = point
                break
        vector = target - current
        if float(np.linalg.norm(vector)) < 0.05:
            return ACTION_STAND
        rotation = command.global_rotation_wxyz
        base_yaw = _yaw(rotation[1], rotation[2], rotation[3], rotation[0])
        angle = math.atan2(float(vector[1]), float(vector[0])) - base_yaw
        angle = math.atan2(math.sin(angle), math.cos(angle))
        if angle > self.flash_turn_threshold:
            return ACTION_LEFT
        if angle < -self.flash_turn_threshold:
            return ACTION_RIGHT
        return ACTION_FORWARD

    def _append(self, record: dict[str, Any]) -> None:
        with self.records_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _write_summary(self, status: str) -> None:
        payload = {
            "schema_version": 1,
            "status": status,
            "mode": (
                f"{self.command_mode}_continuous_physics"
                if self.execution_mode == "continuous"
                else f"{self.command_mode}_sampled_closed_loop"
            ),
            "execution_mode": self.execution_mode,
            "goal_max_distance_m": self.maximum_distance,
            "goal_min_distance_m": self.minimum_distance,
            "navigate_goal_update_distance_m": self.navigate_goal_update_distance,
            "navigate_goal_update_period_sec": self.navigate_goal_update_period,
            "continuous_odom_freshness_sec": CONTINUOUS_ODOM_FRESHNESS_SEC,
            "allow_command_pose_anchor_fallback": (
                self.allow_command_pose_anchor_fallback
            ),
            "command_pose_anchor_fallback_count": (
                self.command_pose_anchor_fallback_count
            ),
            "last_path_anchor_source": self.last_path_anchor_source,
            "last_path_anchor_odom_age_sec": self.last_path_anchor_odom_age_sec,
            "command_count": self.command_count,
            "goal_count": self.goal_count,
            "navigate_goal_reuse_count": self.navigate_goal_reuse_count,
            "system1_target_reissue_count": self.system1_target_reissue_count,
            "nav2_control_count": self.nav2_control_count,
            "system2_passthrough_count": self.system2_passthrough_count,
            "system2_nav2_count": self.system2_nav2_count,
            "cmd_vel_quantization_count": self.quantization_count,
            "direct_motion_bypass_count": self.direct_motion_bypass_count,
            "failure_count": self.failure_count,
            "started_unix": self.started,
            "updated_unix": time.time(),
        }
        temporary = self.result_dir / "active_summary.json.tmp"
        final = self.result_dir / "active_summary.json"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, final)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ActiveNav2Adapter()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._cancel_active()
        except BaseException:
            pass
        if node.execution_mode == "continuous" and rclpy.ok():
            try:
                node._publish_motion(False)
            except BaseException:
                pass
        node._write_summary("FINISHED" if not node.failure_count else "FAIL")
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
