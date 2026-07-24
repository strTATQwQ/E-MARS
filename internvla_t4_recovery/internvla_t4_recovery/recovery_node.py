"""Bounded stuck/oscillation detector and Nav2-native recovery sequence."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import rclpy
from internvla_ros2.recovery_contract import (
    OP_CANCEL_AND_DISABLE,
    OP_CLEAR_MODEL_CACHE,
    OP_REQUEST_REPLAN,
    goal_identity,
    operation_identity,
    recovery_deadline_ns,
    recovery_identity,
    semantic_age_sec,
    system2_primitive_signature,
    t5_completion_sim_enabled,
    trajectory_signature,
)
from internvla_ros2.protocol import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT
from internvla_ros2_msgs.msg import NavigationCommand
from internvla_ros2_msgs.srv import RecoveryControl
from nav2_msgs.action import BackUp, Spin
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String


_SIM_CLOCK_STALL_WATCHDOG_SEC = 300.0
_SYSTEM2_ACTION_SOURCE = 1
_SYSTEM2_MOTION_ACTIONS = frozenset({ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT})
_SYSTEM2_TURN_ACTIONS = frozenset({ACTION_LEFT, ACTION_RIGHT})
_SYSTEM2_TURN_PROGRESS_RAD = math.radians(12.0)
_T5_ACTION_NO_PROGRESS_GRACE_SEC = 3.0


def _wrapped_angle(angle_rad: float) -> float:
    """Return the shortest signed angle in [-pi, pi]."""

    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _yaw_from_quaternion(orientation: Any) -> float:
    values = tuple(
        float(value)
        for value in (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
    )
    norm = math.sqrt(sum(value * value for value in values))
    if not all(math.isfinite(value) for value in values) or norm < 0.5:
        raise ValueError("odometry orientation is invalid")
    x, y, z, w = (value / norm for value in values)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _directed_system2_yaw_progress(
    samples: list[tuple[int, float, float, float]],
    action: int | None,
    motion_started_ns: int,
) -> float:
    """Accumulate only the commanded turn direction after execution starts."""

    if action not in _SYSTEM2_TURN_ACTIONS or motion_started_ns <= 0:
        return 0.0
    yaw_samples = [sample for sample in samples if sample[0] >= motion_started_ns]
    if len(yaw_samples) < 2 or not all(
        math.isfinite(sample[3]) for sample in yaw_samples
    ):
        return 0.0
    unwrapped_delta = sum(
        _wrapped_angle(current[3] - previous[3])
        for previous, current in zip(yaw_samples, yaw_samples[1:])
    )
    commanded_delta = (
        unwrapped_delta if action == ACTION_LEFT else -unwrapped_delta
    )
    return max(0.0, commanded_delta)


def _no_progress_recovery_eligible(
    *, t5_completion_sim: bool, command_age_sec: float | None
) -> bool:
    """Do not let an old progress window preempt a fresh bounded action."""

    if not t5_completion_sim:
        return True
    return bool(
        command_age_sec is not None
        and command_age_sec > _T5_ACTION_NO_PROGRESS_GRACE_SEC
    )


class _SemanticDeadlineExpired(TimeoutError):
    pass


class _RecoveryOwnershipLost(RuntimeError):
    pass


def _wait(future: Any, timeout_sec: float) -> Any:
    event = threading.Event()
    future.add_done_callback(lambda _: event.set())
    if not event.wait(timeout_sec):
        raise TimeoutError
    exception = future.exception()
    if exception is not None:
        raise exception
    return future.result()


class RecoverySupervisor(Node):
    def __init__(self) -> None:
        super().__init__("internvla_t4_recovery_supervisor")
        self.declare_parameter("result_dir", "")
        self.declare_parameter("progress_horizon_sec", 2.0)
        self.declare_parameter("minimum_progress_m", 0.08)
        self.declare_parameter("oscillation_travel_m", 0.30)
        self.declare_parameter("trajectory_validity_sec", 3.0)
        self.declare_parameter("trajectory_refresh_distance_m", 0.30)
        self.declare_parameter("trajectory_refresh_time_sec", 2.0)
        self.declare_parameter("trajectory_deviation_m", 0.60)
        self.declare_parameter("recovery_scan_yaw_rad", math.pi / 3.0)
        self.declare_parameter("recovery_scan_speed_rps", 0.3)
        self.declare_parameter("replan_deadline_sec", 30.0)
        self.declare_parameter("recovery_cooldown_sec", 2.0)
        self.declare_parameter("maximum_recoveries_per_episode", 3)
        self.declare_parameter("enable_short_backup", False)
        self.declare_parameter("enable_scheduled_refresh", False)
        self.declare_parameter("maximum_scheduled_refreshes_per_episode", 1)
        self.declare_parameter("maximum_recovery_duration_sec", 60.0)
        self.declare_parameter("recovery_profile_id", "completion-default")
        self.declare_parameter("recovery_profile_sha256", "none")
        value = str(self.get_parameter("result_dir").value)
        if not value:
            raise RuntimeError("result_dir is required")
        self.result_dir = Path(value).resolve()
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.result_dir / "recovery_records.jsonl"
        if self.audit_path.exists():
            raise FileExistsError(self.audit_path)
        self.horizon = float(self.get_parameter("progress_horizon_sec").value)
        self.minimum_progress = float(self.get_parameter("minimum_progress_m").value)
        self.oscillation_travel = float(
            self.get_parameter("oscillation_travel_m").value
        )
        self.trajectory_validity = float(
            self.get_parameter("trajectory_validity_sec").value
        )
        self.refresh_distance = float(
            self.get_parameter("trajectory_refresh_distance_m").value
        )
        self.refresh_time = float(
            self.get_parameter("trajectory_refresh_time_sec").value
        )
        self.trajectory_deviation = float(
            self.get_parameter("trajectory_deviation_m").value
        )
        self.scan_yaw = float(self.get_parameter("recovery_scan_yaw_rad").value)
        self.scan_speed = float(
            self.get_parameter("recovery_scan_speed_rps").value
        )
        self.replan_deadline = float(
            self.get_parameter("replan_deadline_sec").value
        )
        self.cooldown = float(self.get_parameter("recovery_cooldown_sec").value)
        self.maximum_recoveries = int(
            self.get_parameter("maximum_recoveries_per_episode").value
        )
        self.enable_backup = bool(self.get_parameter("enable_short_backup").value)
        self.enable_scheduled_refresh = bool(
            self.get_parameter("enable_scheduled_refresh").value
        )
        self.maximum_scheduled_refreshes = int(
            self.get_parameter("maximum_scheduled_refreshes_per_episode").value
        )
        self.maximum_recovery_duration = float(
            self.get_parameter("maximum_recovery_duration_sec").value
        )
        self._recovery_uses_sim_time = t5_completion_sim_enabled()
        self.allow_spin_timeout_warn_only = self._recovery_uses_sim_time
        if self._recovery_uses_sim_time and not bool(
            self.get_parameter("use_sim_time").value
        ):
            raise RuntimeError("T5 completion_sim recovery requires use_sim_time=true")
        self.recovery_profile_id = str(
            self.get_parameter("recovery_profile_id").value
        )
        self.recovery_profile_sha256 = str(
            self.get_parameter("recovery_profile_sha256").value
        )
        if not (
            0.5 <= self.horizon <= 5.0
            and 0.02 <= self.minimum_progress <= 0.30
            and 0.10 <= self.oscillation_travel <= 1.0
            and 0.5 <= self.trajectory_validity <= 10.0
            and 0.2 <= self.refresh_distance <= 0.4
            and 1.0 <= self.refresh_time <= 3.0
            and 0.2 <= self.trajectory_deviation <= 1.5
            and 0.1 <= abs(self.scan_yaw) <= math.pi
            and 0.05 <= self.scan_speed <= 2.0
            and 5.0 <= self.replan_deadline <= 120.0
            and 1 <= self.maximum_recoveries <= 5
            and 0 <= self.maximum_scheduled_refreshes <= 3
            and 5.0 <= self.maximum_recovery_duration <= 60.0
        ):
            raise RuntimeError("invalid bounded recovery parameters")
        if self.recovery_profile_sha256 != "none" and (
            len(self.recovery_profile_sha256) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.recovery_profile_sha256
            )
        ):
            raise RuntimeError("recovery profile SHA-256 is invalid")
        self.callback_group = ReentrantCallbackGroup()
        self._semantic_clock_lock = threading.Lock()
        self._last_semantic_clock_ns = 0
        self.samples: deque[tuple[int, float, float, float]] = deque(maxlen=1000)
        self.path_hashes: deque[tuple[int, str]] = deque(maxlen=16)
        self.motion_enabled = False
        self.latest_path_semantic_ns = 0
        self.latest_path: list[tuple[float, float]] = []
        self.latest_command_signature: Any | None = None
        self.latest_signature_source = ""
        self.latest_command_semantic_ns = 0
        self.latest_system2_turn_action: int | None = None
        self._system2_turn_progress_armed = False
        self._system2_turn_motion_started_ns = 0
        self._motion_true_edge_serial = 0
        self._consumed_motion_true_edge_serial = 0
        self._motion_true_edge_semantic_ns = 0
        self.path_start_xy: tuple[float, float] | None = None
        self.episode_id = ""
        self.generation = -1
        self._episode_epoch = 0
        self.last_sequence_id = -1
        self.recoveries_this_episode = 0
        self.recovery_events_this_episode = 0
        self.scheduled_refreshes_this_episode = 0
        self.recovery_active = False
        self.last_recovery_finished_semantic_ns = 0
        self.terminal_stop = False
        self.awaiting_fresh_path_hash: str | None = None
        self.post_recovery_anchor: tuple[float, float] | None = None
        self.post_recovery_index = 0
        self._spin_measurement_active = False
        self._spin_yaw_rate_abs_samples: list[float] = []
        self._lock = threading.RLock()
        self.adapter_recovery_client = self.create_client(
            RecoveryControl,
            "/internvla/t4_recovery_adapter",
            callback_group=self.callback_group,
        )
        self.model_recovery_client = self.create_client(
            RecoveryControl,
            "/internvla/t4_recovery_model",
            callback_group=self.callback_group,
        )
        self.client_recovery_client = self.create_client(
            RecoveryControl,
            "/internvla/t4_recovery_client",
            callback_group=self.callback_group,
        )
        self.spin_client = ActionClient(
            self,
            Spin,
            "spin",
            callback_group=self.callback_group,
        )
        self.backup_client = ActionClient(
            self,
            BackUp,
            "backup",
            callback_group=self.callback_group,
        )
        self.motion_publisher = self.create_publisher(
            Bool, "/internvla/nav2_motion_enabled", 20
        )
        self.stop_publisher = self.create_publisher(Bool, "/internvla/stop", 20)
        self.create_subscription(
            Odometry,
            "/odom",
            self._on_odom,
            50,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            NavPath,
            "/internvla/nav2_active_path",
            self._on_path,
            20,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Bool,
            "/internvla/nav2_motion_enabled",
            self._on_motion,
            20,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            NavigationCommand,
            "/internvla/navigation_command",
            self._on_command,
            20,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            String,
            "/internvla_t4/episode_prime",
            self._on_episode_prime,
            20,
            callback_group=self.callback_group,
        )
        self.create_timer(0.20, self._tick, callback_group=self.callback_group)

    def _append(self, payload: dict[str, Any]) -> None:
        payload = {
            "schema_version": 1,
            "episode_id": self.episode_id,
            "reset_generation": self.generation,
            "recovery_profile_id": self.recovery_profile_id,
            "recovery_profile_sha256": self.recovery_profile_sha256,
            **payload,
            "wall_time_unix": time.time(),
        }
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")

    def _on_odom(self, message: Odometry) -> None:
        now_ns = self._semantic_now_ns()
        if now_ns is None:
            return
        x = float(message.pose.pose.position.x)
        y = float(message.pose.pose.position.y)
        try:
            yaw = _yaw_from_quaternion(message.pose.pose.orientation)
        except (AttributeError, TypeError, ValueError):
            # Preserve the legacy XY sample path.  An invalid quaternion only
            # prevents the completion_sim System2 turn from claiming yaw
            # progress; it must not alter strict-T4 translational supervision.
            yaw = math.nan
        if not all(math.isfinite(value) for value in (x, y)):
            return
        yaw_rate = abs(float(message.twist.twist.angular.z))
        with self._lock:
            self.samples.append((now_ns, x, y, yaw))
            if self._spin_measurement_active and math.isfinite(yaw_rate):
                self._spin_yaw_rate_abs_samples.append(yaw_rate)

    def _read_sim_clock_ns(self) -> tuple[int | None, bool]:
        try:
            if not bool(self.get_parameter("use_sim_time").value):
                return None, False
            value = int(self.get_clock().now().nanoseconds)
        except (AttributeError, TypeError, ValueError):
            return None, False
        return (value, True) if value > 0 else (None, False)

    def _semantic_now_ns(self) -> int | None:
        if not self._recovery_uses_sim_time:
            return time.monotonic_ns()
        value, valid = self._read_sim_clock_ns()
        if not valid or value is None:
            return None
        with self._semantic_clock_lock:
            if value < self._last_semantic_clock_ns:
                return None
            self._last_semantic_clock_ns = value
        return value

    def _require_semantic_now_ns(self) -> int:
        value = self._semantic_now_ns()
        if value is None:
            raise RuntimeError("recovery simulation clock is unavailable or regressed")
        return value

    def _contract_now_ns(self) -> int:
        if self._recovery_uses_sim_time:
            return self._require_semantic_now_ns()
        return time.time_ns()

    def _recovery_owner_matches_locked(
        self, recovery_epoch: int, episode_id: str, generation: int
    ) -> bool:
        return bool(
            self._episode_epoch == recovery_epoch
            and self.episode_id == episode_id
            and self.generation == generation
        )

    def _require_recovery_owner(
        self, recovery_epoch: int, episode_id: str, generation: int
    ) -> None:
        with self._lock:
            if not self._recovery_owner_matches_locked(
                recovery_epoch, episode_id, generation
            ):
                raise _RecoveryOwnershipLost(
                    "recovery episode identity changed"
                )

    def _owned_recovery_effect(
        self,
        recovery_epoch: int,
        episode_id: str,
        generation: int,
        effect: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with self._lock:
            if not self._recovery_owner_matches_locked(
                recovery_epoch, episode_id, generation
            ):
                raise _RecoveryOwnershipLost(
                    "recovery episode identity changed"
                )
            return effect(*args, **kwargs)

    def _wait_future_until_semantic_deadline(
        self,
        future: Any,
        deadline_ns: int,
        *,
        recovery_epoch: int,
        episode_id: str,
        generation: int,
    ) -> Any:
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        self._require_recovery_owner(recovery_epoch, episode_id, generation)
        last_clock_ns = self._require_semantic_now_ns()
        last_clock_progress_wall = time.monotonic()
        while True:
            self._require_recovery_owner(recovery_epoch, episode_id, generation)
            if event.is_set():
                exception = future.exception()
                if exception is not None:
                    raise exception
                return future.result()
            now_ns = self._require_semantic_now_ns()
            if now_ns >= deadline_ns:
                raise _SemanticDeadlineExpired("recovery semantic deadline expired")
            if now_ns > last_clock_ns:
                last_clock_ns = now_ns
                last_clock_progress_wall = time.monotonic()
            elif (
                self._recovery_uses_sim_time
                and time.monotonic() - last_clock_progress_wall
                >= _SIM_CLOCK_STALL_WATCHDOG_SEC
            ):
                raise RuntimeError("recovery simulation clock stalled")
            event.wait(0.10)

    def _wait_future_while_owned(
        self,
        future: Any,
        timeout_sec: float,
        *,
        recovery_epoch: int,
        episode_id: str,
        generation: int,
    ) -> Any:
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        deadline = time.monotonic() + timeout_sec
        while True:
            self._require_recovery_owner(recovery_epoch, episode_id, generation)
            if event.is_set():
                exception = future.exception()
                if exception is not None:
                    raise exception
                return future.result()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError
            event.wait(min(0.10, remaining))

    @staticmethod
    def _cancel_goal_when_ready(future: Any) -> None:
        try:
            handle = future.result()
            if handle is not None and bool(handle.accepted):
                handle.cancel_goal_async()
        except BaseException:
            pass

    def _begin_spin_telemetry(self) -> dict[str, Any]:
        wall_started = time.monotonic()
        sim_started_ns, sim_started_valid = self._read_sim_clock_ns()
        semantic_started_ns = self._semantic_now_ns()
        with self._lock:
            command_semantic_ns = self.latest_command_semantic_ns
            self._spin_yaw_rate_abs_samples.clear()
            self._spin_measurement_active = True
        return {
            "wall_started": wall_started,
            "sim_started_ns": sim_started_ns,
            "sim_started_valid": sim_started_valid,
            "command_age_sec_at_spin_start": semantic_age_sec(
                semantic_started_ns, command_semantic_ns
            ),
        }

    def _finish_spin_telemetry(
        self, record: dict[str, Any], telemetry: dict[str, Any]
    ) -> None:
        wall_ended = time.monotonic()
        sim_ended_ns, sim_ended_valid = self._read_sim_clock_ns()
        semantic_ended_ns = self._semantic_now_ns()
        with self._lock:
            self._spin_measurement_active = False
            yaw_rates = tuple(self._spin_yaw_rate_abs_samples)
            self._spin_yaw_rate_abs_samples.clear()
            command_semantic_ns = self.latest_command_semantic_ns

        wall_duration = max(0.0, wall_ended - float(telemetry["wall_started"]))
        sim_started_ns = telemetry["sim_started_ns"]
        sim_valid = bool(
            telemetry["sim_started_valid"]
            and sim_ended_valid
            and sim_started_ns is not None
            and sim_ended_ns is not None
            and sim_ended_ns >= sim_started_ns
        )
        sim_duration = (
            (sim_ended_ns - sim_started_ns) / 1_000_000_000.0
            if sim_valid
            else None
        )
        spin_rtf = (
            sim_duration / wall_duration
            if sim_duration is not None and wall_duration > 0.0
            else None
        )
        measured_mean = sum(yaw_rates) / len(yaw_rates) if yaw_rates else None
        if measured_mean is not None and not math.isfinite(measured_mean):
            measured_mean = None
        if spin_rtf is not None and not math.isfinite(spin_rtf):
            spin_rtf = None

        record.update(
            {
                "spin_wall_duration_sec": wall_duration,
                "spin_sim_duration_sec": sim_duration,
                "spin_sim_duration_valid": sim_valid,
                "spin_rtf": spin_rtf,
                "commanded_yaw_rate_rps": self.scan_speed,
                "measured_yaw_rate_rps": measured_mean,
                "measured_yaw_rate_sample_count": len(yaw_rates),
                "measured_yaw_rate_peak_rps": max(yaw_rates) if yaw_rates else None,
                "command_age_sec_at_spin_start": telemetry[
                    "command_age_sec_at_spin_start"
                ],
                "command_age_sec_at_spin_end": semantic_age_sec(
                    semantic_ended_ns, command_semantic_ns
                ),
            }
        )

    def _on_path(self, message: NavPath) -> None:
        coordinates = [
            (round(float(pose.pose.position.x), 2), round(float(pose.pose.position.y), 2))
            for pose in message.poses
        ]
        try:
            path_signature = trajectory_signature(coordinates)
        except (TypeError, ValueError):
            path_signature = None
        digest = hashlib.sha256(
            json.dumps(coordinates, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        now_ns = self._semantic_now_ns()
        if now_ns is None:
            return
        with self._lock:
            self.latest_path_semantic_ns = now_ns
            self.latest_path = coordinates
            if self.latest_command_signature is None and path_signature is not None:
                # System 2 paths are generated inside the adapter and therefore
                # do not exist in NavigationCommand.local_path.  The actual
                # Nav2 plan is the only valid old-trajectory exclusion source.
                self.latest_command_signature = path_signature
                self.latest_signature_source = "nav2_active_path"
            self.path_start_xy = (
                (self.samples[-1][1], self.samples[-1][2]) if self.samples else None
            )
            if not self.path_hashes or self.path_hashes[-1][1] != digest:
                self.path_hashes.append((self.latest_path_semantic_ns, digest))
            if self.awaiting_fresh_path_hash is not None:
                previous_digest = self.awaiting_fresh_path_hash
                fresh = digest != previous_digest
                self.awaiting_fresh_path_hash = None
                self.post_recovery_anchor = (
                    (self.samples[-1][1], self.samples[-1][2])
                    if self.samples
                    else None
                )
                self._append(
                    {
                        "event": "post_recovery_trajectory",
                        "recovery_index": self.post_recovery_index,
                        "fresh_path_hash": fresh,
                        "old_path_sha256": previous_digest,
                        "new_path_sha256": digest,
                    }
                )

    def _on_motion(self, message: Bool) -> None:
        enabled = bool(message.data)
        now_ns = (
            self._semantic_now_ns()
            if self._recovery_uses_sim_time and enabled
            else None
        )
        with self._lock:
            if not self._recovery_uses_sim_time:
                self.motion_enabled = enabled
                return
            was_enabled = self.motion_enabled
            self.motion_enabled = enabled
            if not enabled:
                self._system2_turn_progress_armed = False
                self._system2_turn_motion_started_ns = 0
                self._consumed_motion_true_edge_serial = (
                    self._motion_true_edge_serial
                )
                self._motion_true_edge_semantic_ns = 0
            elif not was_enabled:
                self._motion_true_edge_serial += 1
                self._motion_true_edge_semantic_ns = now_ns or 0
                if (
                    self.latest_system2_turn_action in _SYSTEM2_TURN_ACTIONS
                    and now_ns is not None
                ):
                    self._system2_turn_progress_armed = True
                    self._system2_turn_motion_started_ns = now_ns
                    self._consumed_motion_true_edge_serial = (
                        self._motion_true_edge_serial
                    )
            elif (
                self.latest_system2_turn_action in _SYSTEM2_TURN_ACTIONS
                and self._system2_turn_progress_armed
            ):
                # A duplicate true sample is not a new execution edge and must
                # never move the start of an already measured turn.
                pass

    def _adopt_episode_identity_locked(
        self, episode_id: str, generation: int
    ) -> bool:
        """Adopt a current or newer identity and invalidate old recovery work."""

        if generation < self.generation:
            return False
        if generation == self.generation and self.episode_id:
            return episode_id == self.episode_id
        self._episode_epoch += 1
        self.episode_id = episode_id
        self.generation = generation
        self.recoveries_this_episode = 0
        self.recovery_events_this_episode = 0
        self.scheduled_refreshes_this_episode = 0
        self.recovery_active = False
        self.last_recovery_finished_semantic_ns = 0
        self.samples.clear()
        self.path_hashes.clear()
        self.latest_path = []
        self.latest_path_semantic_ns = 0
        self.path_start_xy = None
        self.motion_enabled = False
        self.terminal_stop = False
        self.awaiting_fresh_path_hash = None
        self.post_recovery_anchor = None
        self.post_recovery_index = 0
        self.latest_command_signature = None
        self.latest_signature_source = ""
        self.latest_command_semantic_ns = 0
        self.latest_system2_turn_action = None
        self._system2_turn_progress_armed = False
        self._system2_turn_motion_started_ns = 0
        self._motion_true_edge_serial = 0
        self._consumed_motion_true_edge_serial = 0
        self._motion_true_edge_semantic_ns = 0
        self.last_sequence_id = -1
        self._spin_measurement_active = False
        self._spin_yaw_rate_abs_samples.clear()
        return True

    def _on_episode_prime(self, message: String) -> None:
        try:
            value = json.loads(message.data)
            if int(value.get("schema_version", 0)) != 1:
                raise ValueError("unsupported episode-prime schema")
            episode_id = str(value.get("episode_id", ""))
            generation = int(value.get("reset_generation", -1))
            if not episode_id or generation < 0:
                raise ValueError("invalid episode-prime identity")
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"ignoring malformed episode prime: {exc}")
            return
        with self._lock:
            self._adopt_episode_identity_locked(episode_id, generation)

    def _on_command(self, message: NavigationCommand) -> None:
        now_ns = self._semantic_now_ns()
        with self._lock:
            generation = int(message.reset_generation)
            if not self._adopt_episode_identity_locked(
                str(message.episode_id), generation
            ):
                return
            self.last_sequence_id = int(message.sequence_id)
            self.latest_command_semantic_ns = now_ns if now_ns is not None else 0
            action = int(message.discrete_action)
            system2_motion = bool(
                self._recovery_uses_sim_time
                and not bool(message.stop)
                and int(message.action_source) == _SYSTEM2_ACTION_SOURCE
                and action in _SYSTEM2_MOTION_ACTIONS
            )
            self.latest_system2_turn_action = (
                action if system2_motion and action in _SYSTEM2_TURN_ACTIONS else None
            )
            self._system2_turn_progress_armed = False
            self._system2_turn_motion_started_ns = 0
            if self.latest_system2_turn_action is None:
                # A non-turn command cannot consume a prior motion edge as
                # evidence for a future turn.
                self._consumed_motion_true_edge_serial = (
                    self._motion_true_edge_serial
                )
                self._motion_true_edge_semantic_ns = 0
            elif (
                self.motion_enabled
                and self._motion_true_edge_serial
                > self._consumed_motion_true_edge_serial
                and self._motion_true_edge_semantic_ns > 0
            ):
                # DDS may deliver the adapter's execution edge before this
                # command topic.  Consume that one edge exactly once.
                self._system2_turn_progress_armed = True
                self._system2_turn_motion_started_ns = (
                    self._motion_true_edge_semantic_ns
                )
                self._consumed_motion_true_edge_serial = (
                    self._motion_true_edge_serial
                )
            points = [
                (float(pose.pose.position.x), float(pose.pose.position.y))
                for pose in message.local_path.poses
            ]
            if bool(message.stop):
                self.latest_command_signature = None
                self.latest_signature_source = ""
            elif system2_motion and self.samples:
                try:
                    current = self.samples[-1]
                    self.latest_command_signature = system2_primitive_signature(
                        action=action,
                        episode_id=str(message.episode_id),
                        reset_generation=generation,
                        sequence_id=int(message.sequence_id),
                        observation_digest=str(message.observation_digest),
                        x=current[1],
                        y=current[2],
                        yaw_rad=current[3],
                    )
                    self.latest_signature_source = "system2_primitive"
                except (TypeError, ValueError):
                    self.latest_command_signature = None
                    self.latest_signature_source = ""
            elif system2_motion:
                self.latest_command_signature = None
                self.latest_signature_source = ""
            elif points:
                try:
                    self.latest_command_signature = trajectory_signature(points)
                    self.latest_signature_source = "navigation_command"
                except (TypeError, ValueError):
                    self.latest_command_signature = None
                    self.latest_signature_source = ""

    def _progress_metrics(
        self, now_ns: int
    ) -> tuple[float, float, float, int] | None:
        horizon_ns = int(self.horizon * 1_000_000_000)
        samples = [
            sample
            for sample in self.samples
            if 0 <= now_ns - sample[0] <= horizon_ns
        ]
        if (
            len(samples) < 4
            or samples[-1][0] - samples[0][0] < int(0.8 * horizon_ns)
        ):
            return None
        displacement = math.hypot(
            samples[-1][1] - samples[0][1], samples[-1][2] - samples[0][2]
        )
        travel = sum(
            math.hypot(current[1] - previous[1], current[2] - previous[2])
            for previous, current in zip(samples, samples[1:])
        )
        recent_hashes = {
            digest
            for stamp, digest in self.path_hashes
            if 0 <= now_ns - stamp <= 2 * horizon_ns
        }
        repeated = len(self.path_hashes) >= 3 and len(recent_hashes) == 1
        yaw_progress = (
            _directed_system2_yaw_progress(
                samples,
                self.latest_system2_turn_action,
                self._system2_turn_motion_started_ns,
            )
            if self._recovery_uses_sim_time
            and self._system2_turn_progress_armed
            else 0.0
        )
        return displacement, travel, yaw_progress, int(repeated)

    def _tick(self) -> None:
        now_ns = self._semantic_now_ns()
        if now_ns is None:
            return
        with self._lock:
            path_age = semantic_age_sec(now_ns, self.latest_path_semantic_ns)
            cooldown_age = semantic_age_sec(
                now_ns, self.last_recovery_finished_semantic_ns
            )
            if (
                self.recovery_active
                or self.terminal_stop
                or path_age is None
                or path_age > self.trajectory_validity
                or (
                    self.last_recovery_finished_semantic_ns > 0
                    and (cooldown_age is None or cooldown_age < self.cooldown)
                )
            ):
                return
            metrics = self._progress_metrics(now_ns)
            if metrics is None:
                return
            displacement, travel, yaw_progress, repeated = metrics
            yaw_progress_sufficient = bool(
                self._recovery_uses_sim_time
                and self._system2_turn_progress_armed
                and self.latest_system2_turn_action in _SYSTEM2_TURN_ACTIONS
                and yaw_progress + 1e-9 >= _SYSTEM2_TURN_PROGRESS_RAD
            )
            reason = None
            full_recovery = True
            current_xy = (self.samples[-1][1], self.samples[-1][2])
            if self.post_recovery_anchor is not None:
                post_progress = math.hypot(
                    current_xy[0] - self.post_recovery_anchor[0],
                    current_xy[1] - self.post_recovery_anchor[1],
                )
                if post_progress >= self.minimum_progress:
                    self._append(
                        {
                            "event": "post_recovery_positive_progress",
                            "recovery_index": self.post_recovery_index,
                            "progress_m": post_progress,
                        }
                    )
                    self.post_recovery_anchor = None
            deviation = (
                min(
                    math.hypot(current_xy[0] - point[0], current_xy[1] - point[1])
                    for point in self.latest_path
                )
                if self.latest_path
                else 0.0
            )
            command_age = semantic_age_sec(
                now_ns, self.latest_command_semantic_ns
            )
            progress_recovery_eligible = _no_progress_recovery_eligible(
                t5_completion_sim=self._recovery_uses_sim_time,
                command_age_sec=command_age,
            )
            if self.motion_enabled:
                if (
                    displacement < self.minimum_progress and not yaw_progress_sufficient
                    and progress_recovery_eligible
                ):
                    reason = "no_progress"
                if (
                    progress_recovery_eligible
                    and travel >= self.oscillation_travel
                    and displacement < 1.5 * self.minimum_progress
                    and not yaw_progress_sufficient
                ):
                    reason = "short_period_oscillation"
                if (
                    progress_recovery_eligible
                    and repeated
                    and displacement < 2.0 * self.minimum_progress
                    and not yaw_progress_sufficient
                ):
                    reason = "repeated_trajectory_no_progress"
                if deviation > self.trajectory_deviation:
                    reason = "trajectory_deviation"
            if (
                reason is None
                and self.enable_scheduled_refresh
                and self.scheduled_refreshes_this_episode
                < self.maximum_scheduled_refreshes
                and self.path_start_xy is not None
                and self.latest_command_signature is not None
                and command_age is not None
                and command_age <= self.trajectory_validity
            ):
                traveled_from_path_start = math.hypot(
                    current_xy[0] - self.path_start_xy[0],
                    current_xy[1] - self.path_start_xy[1],
                )
                if traveled_from_path_start >= self.refresh_distance:
                    reason = "scheduled_distance_refresh"
                    full_recovery = False
                elif path_age >= self.refresh_time:
                    reason = "scheduled_time_refresh"
                    full_recovery = False
            if reason is None:
                return
            if self.latest_command_signature is None:
                # A recovery transaction must never start unless it can name
                # and exclude the trajectory that is being canceled.
                return
            if (
                full_recovery
                and self.recoveries_this_episode >= self.maximum_recoveries
            ):
                self.terminal_stop = True
                stop = Bool()
                stop.data = True
                self.stop_publisher.publish(stop)
                self._append(
                    {
                        "event": "terminal_safe_stop",
                        "reason": "maximum_recoveries_reached",
                        "displacement_m": displacement,
                        "travel_m": travel,
                    }
                )
                return
            self.recovery_active = True
            self.recovery_events_this_episode += 1
            if full_recovery:
                self.recoveries_this_episode += 1
            else:
                self.scheduled_refreshes_this_episode += 1
            recovery_epoch = self._episode_epoch
            recovery_episode_id = self.episode_id
            recovery_generation = self.generation
        threading.Thread(
            target=self._recover,
            args=(
                reason,
                displacement,
                travel,
                bool(repeated),
                full_recovery,
                recovery_epoch,
                recovery_episode_id,
                recovery_generation,
            ),
            daemon=True,
        ).start()

    def _recovery_request(
        self,
        operation: int,
        recovery_id: str,
        *,
        recovery_epoch: int,
        episode_id: str,
        generation: int,
        sequence_id: int,
        include_trajectory_exclusion: bool = False,
        goal_id_override: str | None = None,
        signature_override: Any | None = None,
        timeout_sec: float = 8.0,
        deadline_ns_override: int | None = None,
    ) -> RecoveryControl.Request:
        with self._lock:
            if not self._recovery_owner_matches_locked(
                recovery_epoch, episode_id, generation
            ):
                raise _RecoveryOwnershipLost(
                    "recovery episode identity changed"
                )
            signature = (
                signature_override
                if signature_override is not None
                else self.latest_command_signature
            )
        if sequence_id < 0:
            raise RuntimeError("recovery has no active navigation command")
        if include_trajectory_exclusion and signature is None:
            raise RuntimeError("recovery cannot exclude the active trajectory")
        request = RecoveryControl.Request()
        request.episode_id = episode_id
        request.reset_generation = generation
        request.goal_id = goal_id_override or goal_identity(
            episode_id, generation, sequence_id
        )
        request.recovery_id = recovery_id
        request.operation_id = operation_identity(recovery_id, operation)
        request.operation = operation
        deadline_ns = (
            int(deadline_ns_override)
            if deadline_ns_override is not None
            else recovery_deadline_ns(self._contract_now_ns(), timeout_sec)
        )
        if deadline_ns <= 0:
            raise RuntimeError("recovery deadline override must be positive")
        request.deadline.sec = deadline_ns // 1_000_000_000
        request.deadline.nanosec = deadline_ns % 1_000_000_000
        if include_trajectory_exclusion:
            request.excluded_absolute_sha256 = [signature.absolute_sha256]
            request.excluded_shape_sha256 = [signature.shape_sha256]
        return request

    @staticmethod
    def _call_typed(
        client: Any,
        request: RecoveryControl.Request,
        *,
        service_name: str,
        timeout_sec: float,
        ownership_check: Any,
        owned_effect: Any,
    ) -> RecoveryControl.Response:
        ownership_check()
        if not client.wait_for_service(timeout_sec=2.0):
            raise TimeoutError(f"{service_name} unavailable")
        ownership_check()
        last_error: BaseException | None = None
        for _ in range(2):
            try:
                response = _wait(
                    owned_effect(client.call_async, request), timeout_sec
                )
                ownership_check()
                if (
                    str(response.episode_id) != str(request.episode_id)
                    or int(response.reset_generation) != int(request.reset_generation)
                    or str(response.goal_id) != str(request.goal_id)
                    or str(response.recovery_id) != str(request.recovery_id)
                    or str(response.operation_id) != str(request.operation_id)
                    or int(response.operation) != int(request.operation)
                ):
                    raise RuntimeError(f"{service_name} returned mismatched identity")
                if not bool(response.success):
                    raise RuntimeError(
                        f"{service_name} failed: {response.status_message}"
                    )
                return response
            except _RecoveryOwnershipLost:
                raise
            except (TimeoutError, RuntimeError) as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _recover(
        self,
        reason: str,
        displacement: float,
        travel: float,
        repeated: bool,
        full_recovery: bool,
        recovery_epoch: int,
        recovery_episode_id: str,
        recovery_generation: int,
    ) -> None:
        with self._lock:
            if not self._recovery_owner_matches_locked(
                recovery_epoch, recovery_episode_id, recovery_generation
            ):
                return
            recovery_index = self.recovery_events_this_episode
            active_sequence_id = self.last_sequence_id
            active_goal_id = goal_identity(
                recovery_episode_id,
                recovery_generation,
                active_sequence_id,
            )
            active_signature = self.latest_command_signature
            active_signature_source = self.latest_signature_source
            active_recovery_id = recovery_identity(
                recovery_episode_id,
                recovery_generation,
                recovery_index,
            )
            recoveries_at_start = self.recoveries_this_episode

        def require_owner() -> None:
            self._require_recovery_owner(
                recovery_epoch, recovery_episode_id, recovery_generation
            )

        def owned_effect(effect: Any, *args: Any, **kwargs: Any) -> Any:
            return self._owned_recovery_effect(
                recovery_epoch,
                recovery_episode_id,
                recovery_generation,
                effect,
                *args,
                **kwargs,
            )

        started_ns = self._semantic_now_ns()
        recovery_semantic_deadline_ns = (
            recovery_deadline_ns(started_ns, self.maximum_recovery_duration)
            if started_ns is not None
            else None
        )
        record: dict[str, Any] = {
            "event": "recovery",
            "reason": reason,
            "displacement_m": displacement,
            "travel_m": travel,
            "repeated_path": repeated,
            "full_recovery": full_recovery,
            "recovery_index": recovery_index,
            "recovery_id": active_recovery_id,
            "cancel_success": False,
            "history_clear_success": False,
            "scan_success": False,
            "backup_attempted": False,
            "backup_success": False,
            "fresh_trajectory_requested": False,
            "typed_transaction": True,
            "active_signature_source": active_signature_source,
        }
        disabled = Bool()
        disabled.data = False
        try:
            if recovery_semantic_deadline_ns is None:
                raise RuntimeError(
                    "recovery simulation clock is unavailable or regressed"
                )
            owned_effect(self.motion_publisher.publish, disabled)
            with self._lock:
                if not self._recovery_owner_matches_locked(
                    recovery_epoch, recovery_episode_id, recovery_generation
                ):
                    raise _RecoveryOwnershipLost(
                        "recovery episode identity changed"
                    )
                self.awaiting_fresh_path_hash = (
                    self.path_hashes[-1][1]
                    if self.path_hashes
                    else "no_prior_path"
                )
                self.post_recovery_index = recovery_index
            cancel_request = self._recovery_request(
                OP_CANCEL_AND_DISABLE,
                active_recovery_id,
                recovery_epoch=recovery_epoch,
                episode_id=recovery_episode_id,
                generation=recovery_generation,
                sequence_id=active_sequence_id,
                goal_id_override=active_goal_id,
            )
            cancel = self._call_typed(
                self.adapter_recovery_client,
                cancel_request,
                service_name="adapter recovery service",
                timeout_sec=3.0,
                ownership_check=require_owner,
                owned_effect=owned_effect,
            )
            record["cancel_operation_id"] = str(cancel.operation_id)
            record["cancel_success"] = bool(
                cancel.success and cancel.motion_disabled
            )
            record["cancel_safety_fresh"] = bool(cancel.safety_fresh)
            record["cancel_linear_speed_mps"] = float(
                cancel.measured_linear_speed_peak_mps
            )
            record["cancel_angular_speed_rps"] = float(
                cancel.measured_angular_speed_peak_rps
            )
            if not record["cancel_success"]:
                raise RuntimeError("adapter did not confirm motion disable")

            client_replan_request = self._recovery_request(
                OP_REQUEST_REPLAN,
                active_recovery_id,
                recovery_epoch=recovery_epoch,
                episode_id=recovery_episode_id,
                generation=recovery_generation,
                sequence_id=active_sequence_id,
                include_trajectory_exclusion=True,
                signature_override=active_signature,
                timeout_sec=self.replan_deadline,
            )
            replanned = self._call_typed(
                self.client_recovery_client,
                client_replan_request,
                service_name="client recovery service",
                timeout_sec=3.0,
                ownership_check=require_owner,
                owned_effect=owned_effect,
            )
            record["replan_operation_id"] = str(replanned.operation_id)
            record["excluded_absolute_sha256"] = list(
                client_replan_request.excluded_absolute_sha256
            )
            record["excluded_shape_sha256"] = list(
                client_replan_request.excluded_shape_sha256
            )
            record["fresh_trajectory_requested"] = bool(replanned.success)
            shared_replan_deadline_ns = (
                int(client_replan_request.deadline.sec) * 1_000_000_000
                + int(client_replan_request.deadline.nanosec)
            )

            clear_request = self._recovery_request(
                OP_CLEAR_MODEL_CACHE,
                active_recovery_id,
                recovery_epoch=recovery_epoch,
                episode_id=recovery_episode_id,
                generation=recovery_generation,
                sequence_id=active_sequence_id,
            )
            cleared = self._call_typed(
                self.model_recovery_client,
                clear_request,
                service_name="model recovery service",
                timeout_sec=4.0,
                ownership_check=require_owner,
                owned_effect=owned_effect,
            )
            record["history_clear_success"] = bool(cleared.success)
            record["history_clear_operation_id"] = str(cleared.operation_id)
            record["cache_epoch"] = int(cleared.cache_epoch)
            if not cleared.success or int(cleared.cache_epoch) <= 0:
                raise RuntimeError("model did not confirm a new cache epoch")
            if full_recovery:
                enabled = Bool()
                enabled.data = True
                owned_effect(self.motion_publisher.publish, enabled)
                require_owner()
                if not self.spin_client.wait_for_server(timeout_sec=2.0):
                    raise TimeoutError("Nav2 Spin server unavailable")
                require_owner()
                goal = Spin.Goal()
                goal.target_yaw = self.scan_yaw
                goal.time_allowance.sec = 12
                spin_goal_future = owned_effect(
                    self.spin_client.send_goal_async, goal
                )
                try:
                    handle = self._wait_future_while_owned(
                        spin_goal_future,
                        3.0,
                        recovery_epoch=recovery_epoch,
                        episode_id=recovery_episode_id,
                        generation=recovery_generation,
                    )
                except _RecoveryOwnershipLost:
                    spin_goal_future.add_done_callback(
                        self._cancel_goal_when_ready
                    )
                    raise
                if not handle.accepted:
                    raise RuntimeError("Nav2 Spin rejected")
                with self._lock:
                    if not self._recovery_owner_matches_locked(
                        recovery_epoch,
                        recovery_episode_id,
                        recovery_generation,
                    ):
                        handle.cancel_goal_async()
                        raise _RecoveryOwnershipLost(
                            "recovery episode identity changed"
                        )
                    spin_telemetry = self._begin_spin_telemetry()
                spin_result = None
                spin_result_future = owned_effect(handle.get_result_async)
                try:
                    try:
                        post_spin_reserve_sec = (
                            13.0 if self.allow_spin_timeout_warn_only else 0.5
                        )
                        spin_semantic_deadline_ns = (
                            recovery_semantic_deadline_ns
                            - int(post_spin_reserve_sec * 1_000_000_000)
                        )
                        if (
                            self._require_semantic_now_ns()
                            >= spin_semantic_deadline_ns
                        ):
                            raise _SemanticDeadlineExpired(
                                "recovery semantic budget exhausted before Spin"
                            )
                        spin_result = self._wait_future_until_semantic_deadline(
                            spin_result_future,
                            spin_semantic_deadline_ns,
                            recovery_epoch=recovery_epoch,
                            episode_id=recovery_episode_id,
                            generation=recovery_generation,
                        )
                    except Exception as exc:
                        ownership_lost = isinstance(
                            exc, _RecoveryOwnershipLost
                        )
                        if not ownership_lost:
                            try:
                                owned_effect(
                                    self.motion_publisher.publish, disabled
                                )
                            except _RecoveryOwnershipLost as lost:
                                exc = lost
                                ownership_lost = True
                        if ownership_lost and bool(
                            getattr(spin_result_future, "done", lambda: False)()
                        ):
                            raise exc
                        cancel_future = handle.cancel_goal_async()
                        cancel_response = _wait(cancel_future, 2.0)
                        cancel_confirmed = bool(
                            getattr(cancel_response, "goals_canceling", [])
                        )
                        if ownership_lost:
                            if cancel_confirmed:
                                try:
                                    _wait(spin_result_future, 2.0)
                                except BaseException:
                                    pass
                            raise exc
                        if not cancel_confirmed:
                            raise RuntimeError(
                                "Nav2 Spin cancellation not confirmed"
                            ) from exc
                        if not isinstance(exc, _SemanticDeadlineExpired):
                            raise
                        if not self.allow_spin_timeout_warn_only:
                            raise _SemanticDeadlineExpired(
                                "Nav2 Spin exceeded recovery semantic budget"
                            ) from exc
                        try:
                            terminal_result = _wait(spin_result_future, 2.0)
                        except TimeoutError as terminal_exc:
                            raise RuntimeError(
                                "Nav2 Spin cancellation did not reach terminal state"
                            ) from terminal_exc
                        record["scan_cancel_terminal_status"] = int(
                            terminal_result.status
                        )
                        record["scan_timeout_warn_only"] = True
                        record["scan_warning"] = (
                            "Nav2 Spin exceeded recovery semantic budget"
                        )
                finally:
                    with self._lock:
                        if self._recovery_owner_matches_locked(
                            recovery_epoch,
                            recovery_episode_id,
                            recovery_generation,
                        ):
                            self._finish_spin_telemetry(
                                record, spin_telemetry
                            )
                record["scan_success"] = bool(
                    spin_result is not None and int(spin_result.status) == 4
                )
                if (
                    self.enable_backup
                    and not self.allow_spin_timeout_warn_only
                    and recoveries_at_start >= 2
                ):
                    require_owner()
                    backup_available = self.backup_client.wait_for_server(
                        timeout_sec=1.0
                    )
                    require_owner()
                    if backup_available:
                        record["backup_attempted"] = True
                        backup = BackUp.Goal()
                        backup.target.x = -0.15
                        backup.speed = 0.05
                        backup.time_allowance.sec = 6
                        backup_goal_future = owned_effect(
                            self.backup_client.send_goal_async, backup
                        )
                        try:
                            backup_handle = self._wait_future_while_owned(
                                backup_goal_future,
                                2.0,
                                recovery_epoch=recovery_epoch,
                                episode_id=recovery_episode_id,
                                generation=recovery_generation,
                            )
                        except _RecoveryOwnershipLost:
                            backup_goal_future.add_done_callback(
                                self._cancel_goal_when_ready
                            )
                            raise
                        if backup_handle.accepted:
                            backup_result_future = owned_effect(
                                backup_handle.get_result_async
                            )
                            try:
                                backup_result = self._wait_future_while_owned(
                                    backup_result_future,
                                    8.0,
                                    recovery_epoch=recovery_epoch,
                                    episode_id=recovery_episode_id,
                                    generation=recovery_generation,
                                )
                            except _RecoveryOwnershipLost:
                                backup_handle.cancel_goal_async()
                                raise
                            record["backup_success"] = (
                                int(backup_result.status) == 4
                            )
            owned_effect(self.motion_publisher.publish, disabled)
            adapter_replan_request = self._recovery_request(
                OP_REQUEST_REPLAN,
                active_recovery_id,
                recovery_epoch=recovery_epoch,
                episode_id=recovery_episode_id,
                generation=recovery_generation,
                sequence_id=active_sequence_id,
                include_trajectory_exclusion=True,
                goal_id_override=active_goal_id,
                signature_override=active_signature,
                deadline_ns_override=shared_replan_deadline_ns,
            )
            armed = self._call_typed(
                self.adapter_recovery_client,
                adapter_replan_request,
                service_name="adapter recovery replan gate",
                timeout_sec=3.0,
                ownership_check=require_owner,
                owned_effect=owned_effect,
            )
            record["adapter_replan_gate_armed"] = bool(armed.success)
        except _RecoveryOwnershipLost:
            pass
        except BaseException as exc:
            record["error"] = repr(exc)[:1024]
            stop = Bool()
            stop.data = True
            try:
                owned_effect(self.stop_publisher.publish, stop)
            except _RecoveryOwnershipLost:
                pass
        finally:
            try:
                require_owner()
                owned_effect(self.motion_publisher.publish, disabled)
                ended_ns = self._semantic_now_ns()
                record["duration_sec"] = semantic_age_sec(ended_ns, started_ns)
                if record["duration_sec"] is None:
                    record["error"] = "recovery simulation clock became invalid"
                    stop = Bool()
                    stop.data = True
                    owned_effect(self.stop_publisher.publish, stop)
                elif record["duration_sec"] > self.maximum_recovery_duration:
                    record["error"] = (
                        "maximum recovery semantic duration exceeded"
                    )
                    stop = Bool()
                    stop.data = True
                    owned_effect(self.stop_publisher.publish, stop)
                with self._lock:
                    if not self._recovery_owner_matches_locked(
                        recovery_epoch,
                        recovery_episode_id,
                        recovery_generation,
                    ):
                        raise _RecoveryOwnershipLost(
                            "recovery episode identity changed"
                        )
                    self._append(record)
                    self.samples.clear()
                    self.path_hashes.clear()
                    self.latest_path = []
                    self.path_start_xy = None
                    self.latest_path_semantic_ns = 0
                    self.recovery_active = False
                    self.last_recovery_finished_semantic_ns = ended_ns or 0
                    if record["duration_sec"] is None:
                        self.terminal_stop = True
            except _RecoveryOwnershipLost:
                pass


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RecoverySupervisor()
    executor = MultiThreadedExecutor(num_threads=8)
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
