"""Pure sim-time motion/observation sequencing for T5 completion_sim.

The evaluator advances Isaac only after an IPC step returns, so motion must not
be awaited inside the model RPC.  This state machine instead spans evaluator
requests: arm after a Nav2 action, observe navigation odometry asynchronously,
safe-stop when the bounded motion completes (or times out), then require a new
post-stop observation before another model inference is admitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any


ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
MOTION_ACTIONS = frozenset({ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT})
SYSTEM1_QUEUE_ACTION_SOURCE = 3

DECISION_READY = "ready"
DECISION_HOLD_MOTION = "hold_motion"
DECISION_HOLD_CANCEL_ACK = "hold_cancel_ack"
DECISION_HOLD_POST_STOP = "hold_post_stop"
DECISION_SAFE_STOP_COMPLETE = "safe_stop_complete"
DECISION_SAFE_STOP_TIMEOUT = "safe_stop_timeout"
DECISION_SAFE_STOP_STALE = "safe_stop_stale"

ODOMETRY_STAMP_ADVANCED = "advanced"
ODOMETRY_STAMP_DUPLICATE = "duplicate"
ODOMETRY_STAMP_REGRESSED = "regressed"


def classify_odometry_stamp(previous_stamp_ns: int, stamp_ns: int) -> str:
    """Classify an already time-validated odometry stamp."""

    if previous_stamp_ns > 0 and stamp_ns == previous_stamp_ns:
        return ODOMETRY_STAMP_DUPLICATE
    if previous_stamp_ns > 0 and stamp_ns < previous_stamp_ns:
        return ODOMETRY_STAMP_REGRESSED
    return ODOMETRY_STAMP_ADVANCED


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw_rad: float

    def validate(self) -> None:
        if not all(math.isfinite(value) for value in (self.x, self.y, self.yaw_rad)):
            raise ValueError("motion-gate pose contains NaN/Inf")


@dataclass(frozen=True)
class MotionGateConfig:
    turn_command_deg: float = 15.0
    turn_required_deg: float = 12.0
    turn_timeout_sec: float = 2.0
    forward_command_m: float = 0.25
    forward_required_m: float = 0.20
    forward_timeout_sec: float = 3.0

    def validate(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
            raise ValueError("motion-gate bounds must be finite and positive")
        if self.turn_required_deg > self.turn_command_deg:
            raise ValueError("required turn exceeds commanded turn")
        if self.forward_required_m > self.forward_command_m:
            raise ValueError("required forward distance exceeds commanded distance")


def is_system1_queue_plan_motion(
    *,
    action_source: int,
    trajectory_valid: bool,
    nav2_goal_sent: bool,
    nav2_plan_valid: bool,
    stop: bool,
    model_action: int,
) -> bool:
    """Return whether a validated queue response still owns a usable Nav2 plan."""

    return bool(
        action_source == SYSTEM1_QUEUE_ACTION_SOURCE
        and not trajectory_valid
        and nav2_goal_sent
        and nav2_plan_valid
        and not stop
        and model_action in MOTION_ACTIONS
    )


def resolved_motion_bounds(
    action: int,
    config: MotionGateConfig,
    local_path: Any,
    *,
    use_preregistered_bounds: bool = False,
) -> tuple[float, float]:
    """Resolve gate bounds without requiring a path for an accepted queue plan."""

    if action not in MOTION_ACTIONS:
        raise ValueError("motion-gate action is not a motion action")
    if action != ACTION_FORWARD:
        return (
            math.radians(config.turn_command_deg),
            math.radians(config.turn_required_deg),
        )
    if use_preregistered_bounds:
        return config.forward_command_m, config.forward_required_m
    if not isinstance(local_path, list):
        raise ValueError("resolved forward path is unavailable")
    forward_reach = max(
        (
            float(point[0])
            for point in local_path
            if isinstance(point, (list, tuple))
            and len(point) == 2
            and math.isfinite(float(point[0]))
        ),
        default=0.0,
    )
    commanded = min(config.forward_command_m, forward_reach)
    if commanded <= 0.05:
        raise ValueError("resolved forward path is too short for measured motion")
    required = min(config.forward_required_m, commanded * 0.8)
    return commanded, required


@dataclass(frozen=True)
class GateDecision:
    kind: str
    reason: str
    permits_model_step: bool
    requires_safe_stop: bool
    keep_safe_stop: bool
    progress: float = 0.0
    required_progress: float = 0.0
    commanded_progress: float = 0.0
    elapsed_sim_sec: float = 0.0

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _PendingMotion:
    episode_id: str
    reset_generation: int
    sequence_id: int
    request_id: str
    action: int
    start_pose: Pose2D
    start_sim_ns: int
    start_odom_stamp_ns: int
    start_odom_serial: int
    required_progress: float
    commanded_progress: float
    timeout_sec: float


@dataclass(frozen=True)
class _StopBarrier:
    stop_requested_sim_ns: int
    stop_requested_odom_stamp_ns: int
    stop_requested_odom_serial: int
    stop_requested_camera_stamp_ns: int
    stop_requested_camera_serial: int
    cancel_acknowledged: bool
    cancel_ack_sim_ns: int
    cancel_ack_odom_stamp_ns: int
    cancel_ack_odom_serial: int
    cancel_ack_camera_stamp_ns: int
    cancel_ack_camera_serial: int
    reason: str
    action: int
    sequence_id: int
    request_id: str


def _wrapped_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class MotionObservationGate:
    """Deterministic state machine; callers provide synchronization."""

    def __init__(self, config: MotionGateConfig | None = None) -> None:
        self.config = config or MotionGateConfig()
        self.config.validate()
        self.active_episode_id = ""
        self.active_reset_generation = -1
        self.pending: _PendingMotion | None = None
        self.stop_barrier: _StopBarrier | None = None

    @property
    def state(self) -> str:
        if self.pending is not None:
            return "executing"
        if self.stop_barrier is not None:
            return "awaiting_post_stop_observation"
        return "ready"

    def reset(self, episode_id: str, reset_generation: int) -> None:
        if not episode_id:
            raise ValueError("motion-gate episode_id is required")
        if int(reset_generation) < 0:
            raise ValueError("motion-gate reset_generation must be nonnegative")
        self.active_episode_id = str(episode_id)
        self.active_reset_generation = int(reset_generation)
        self.pending = None
        self.stop_barrier = None

    def arm(
        self,
        *,
        episode_id: str,
        reset_generation: int,
        sequence_id: int,
        request_id: str,
        action: int,
        start_pose: Pose2D,
        start_sim_ns: int,
        start_odom_stamp_ns: int,
        start_odom_serial: int,
        resolved_commanded_progress: float | None = None,
        resolved_required_progress: float | None = None,
    ) -> GateDecision:
        self._validate_identity(episode_id, reset_generation)
        if self.state != "ready":
            raise RuntimeError(f"cannot arm motion gate from state {self.state}")
        if int(action) not in MOTION_ACTIONS:
            raise ValueError(f"unsupported motion action {action}")
        if int(sequence_id) < 0 or not request_id:
            raise ValueError("motion-gate request identity is invalid")
        if min(int(start_sim_ns), int(start_odom_stamp_ns), int(start_odom_serial)) <= 0:
            raise ValueError("motion-gate arm requires positive sim/odometry stamps")
        start_pose.validate()
        required, commanded, timeout_sec = self._bounds(int(action))
        if resolved_commanded_progress is not None:
            commanded = float(resolved_commanded_progress)
        if resolved_required_progress is not None:
            required = float(resolved_required_progress)
        if (
            not math.isfinite(required)
            or not math.isfinite(commanded)
            or required <= 0.0
            or commanded <= 0.0
            or required > commanded
        ):
            raise ValueError("resolved motion bounds are invalid")
        self.pending = _PendingMotion(
            episode_id=str(episode_id),
            reset_generation=int(reset_generation),
            sequence_id=int(sequence_id),
            request_id=str(request_id),
            action=int(action),
            start_pose=start_pose,
            start_sim_ns=int(start_sim_ns),
            start_odom_stamp_ns=int(start_odom_stamp_ns),
            start_odom_serial=int(start_odom_serial),
            required_progress=required,
            commanded_progress=commanded,
            timeout_sec=timeout_sec,
        )
        return GateDecision(
            kind=DECISION_HOLD_MOTION,
            reason="Nav2 motion armed; waiting for measured odometry progress",
            permits_model_step=False,
            requires_safe_stop=False,
            keep_safe_stop=False,
            required_progress=required,
            commanded_progress=commanded,
        )

    def observe(
        self,
        *,
        episode_id: str,
        reset_generation: int,
        sim_stamp_ns: int,
        odom_stamp_ns: int,
        odom_serial: int,
        camera_sensor_stamp_ns: int = 0,
        camera_sensor_serial: int = 0,
        pose: Pose2D,
    ) -> GateDecision:
        self._validate_identity(episode_id, reset_generation)
        pose.validate()
        sim_stamp_ns = int(sim_stamp_ns)
        odom_stamp_ns = int(odom_stamp_ns)
        odom_serial = int(odom_serial)
        if self.stop_barrier is not None:
            barrier = self.stop_barrier
            if not barrier.cancel_acknowledged:
                return GateDecision(
                    kind=DECISION_HOLD_CANCEL_ACK,
                    reason="safe-stop requested; waiting for confirmed Nav2 cancellation",
                    permits_model_step=False,
                    requires_safe_stop=False,
                    keep_safe_stop=True,
                )
            fresh = (
                sim_stamp_ns > barrier.cancel_ack_sim_ns
                and odom_stamp_ns > barrier.cancel_ack_odom_stamp_ns
                and odom_serial > barrier.cancel_ack_odom_serial
                and int(camera_sensor_stamp_ns) > barrier.cancel_ack_camera_stamp_ns
                and int(camera_sensor_serial) > barrier.cancel_ack_camera_serial
            )
            if fresh:
                self.stop_barrier = None
                return GateDecision(
                    kind=DECISION_READY,
                    reason=(
                        "new post-cancel camera observation and odometry admitted"
                    ),
                    permits_model_step=True,
                    requires_safe_stop=False,
                    keep_safe_stop=True,
                )
            return GateDecision(
                kind=DECISION_HOLD_POST_STOP,
                reason=(
                    "waiting for camera observation and odometry newer than cancel ack"
                ),
                permits_model_step=False,
                requires_safe_stop=False,
                keep_safe_stop=True,
            )
        if self.pending is None:
            return GateDecision(
                kind=DECISION_READY,
                reason="motion gate is ready",
                permits_model_step=True,
                requires_safe_stop=False,
                keep_safe_stop=False,
            )

        pending = self.pending
        if (
            sim_stamp_ns <= 0
            or odom_stamp_ns <= 0
            or odom_serial <= 0
            or sim_stamp_ns < pending.start_sim_ns
            or odom_stamp_ns < pending.start_odom_stamp_ns
            or odom_serial < pending.start_odom_serial
            or odom_stamp_ns > sim_stamp_ns
        ):
            return self._transition_to_stop(
                pending,
                sim_stamp_ns=max(sim_stamp_ns, pending.start_sim_ns),
                odom_stamp_ns=max(odom_stamp_ns, pending.start_odom_stamp_ns),
                odom_serial=max(odom_serial, pending.start_odom_serial),
                kind=DECISION_SAFE_STOP_STALE,
                reason="sim time or odometry regressed while motion was active",
            )

        required = pending.required_progress
        commanded = pending.commanded_progress
        timeout_sec = pending.timeout_sec
        progress = self._progress(pending.action, pending.start_pose, pose)
        elapsed = (sim_stamp_ns - pending.start_sim_ns) / 1_000_000_000
        # Deadline is exclusive: reaching the threshold at/after the deadline is
        # a timeout, never a late PASS.
        if elapsed >= timeout_sec:
            return self._transition_to_stop(
                pending,
                sim_stamp_ns=sim_stamp_ns,
                odom_stamp_ns=odom_stamp_ns,
                odom_serial=odom_serial,
                kind=DECISION_SAFE_STOP_TIMEOUT,
                reason="measured motion missed the sim-time deadline",
                progress=progress,
                required=required,
                commanded=commanded,
                elapsed=elapsed,
            )
        if progress + 1e-9 >= required:
            return self._transition_to_stop(
                pending,
                sim_stamp_ns=sim_stamp_ns,
                odom_stamp_ns=odom_stamp_ns,
                odom_serial=odom_serial,
                kind=DECISION_SAFE_STOP_COMPLETE,
                reason="measured motion reached the preregistered completion threshold",
                progress=progress,
                required=required,
                commanded=commanded,
                elapsed=elapsed,
            )
        return GateDecision(
            kind=DECISION_HOLD_MOTION,
            reason="Nav2 motion remains inside its sim-time execution window",
            permits_model_step=False,
            requires_safe_stop=False,
            keep_safe_stop=False,
            progress=progress,
            required_progress=required,
            commanded_progress=commanded,
            elapsed_sim_sec=elapsed,
        )

    def require_stop_barrier(
        self,
        *,
        episode_id: str,
        reset_generation: int,
        sequence_id: int,
        request_id: str,
        sim_stamp_ns: int,
        odom_stamp_ns: int,
        odom_serial: int,
        camera_sensor_stamp_ns: int,
        camera_sensor_serial: int,
        reason: str,
    ) -> GateDecision:
        """Create an explicit stop-request barrier without an active motion.

        Initialize/reset uses this path so an old Nav2 goal cannot cross an
        episode boundary.  Cancellation still has to be acknowledged before a
        later camera/odometry pair can release the barrier.
        """

        self._validate_identity(episode_id, reset_generation)
        if self.pending is not None:
            raise RuntimeError("cannot replace an active motion with a reset barrier")
        self.stop_barrier = _StopBarrier(
            stop_requested_sim_ns=max(0, int(sim_stamp_ns)),
            stop_requested_odom_stamp_ns=max(0, int(odom_stamp_ns)),
            stop_requested_odom_serial=max(0, int(odom_serial)),
            stop_requested_camera_stamp_ns=max(0, int(camera_sensor_stamp_ns)),
            stop_requested_camera_serial=max(0, int(camera_sensor_serial)),
            cancel_acknowledged=False,
            cancel_ack_sim_ns=0,
            cancel_ack_odom_stamp_ns=0,
            cancel_ack_odom_serial=0,
            cancel_ack_camera_stamp_ns=0,
            cancel_ack_camera_serial=0,
            reason=str(reason),
            action=0,
            sequence_id=int(sequence_id),
            request_id=str(request_id),
        )
        return GateDecision(
            kind="reset_safe_stop",
            reason=str(reason),
            permits_model_step=False,
            requires_safe_stop=True,
            keep_safe_stop=True,
        )

    def fail_active_motion(
        self,
        *,
        episode_id: str,
        reset_generation: int,
        sim_stamp_ns: int,
        odom_stamp_ns: int,
        odom_serial: int,
        reason: str,
    ) -> GateDecision:
        self._validate_identity(episode_id, reset_generation)
        pending = self.pending
        if pending is None:
            raise RuntimeError("no active motion can be failed")
        return self._transition_to_stop(
            pending,
            sim_stamp_ns=max(int(sim_stamp_ns), pending.start_sim_ns),
            odom_stamp_ns=max(int(odom_stamp_ns), pending.start_odom_stamp_ns),
            odom_serial=max(int(odom_serial), pending.start_odom_serial),
            kind=DECISION_SAFE_STOP_STALE,
            reason=str(reason),
        )

    def acknowledge_stop(
        self,
        *,
        episode_id: str,
        reset_generation: int,
        sim_stamp_ns: int,
        odom_stamp_ns: int,
        odom_serial: int,
        camera_sensor_stamp_ns: int,
        camera_sensor_serial: int,
    ) -> GateDecision:
        self._validate_identity(episode_id, reset_generation)
        barrier = self.stop_barrier
        if barrier is None:
            raise RuntimeError("cancel ack has no matching stop request")
        if barrier.cancel_acknowledged:
            raise RuntimeError("stop request was already acknowledged")
        baselines = (
            int(sim_stamp_ns),
            int(odom_stamp_ns),
            int(odom_serial),
            int(camera_sensor_stamp_ns),
            int(camera_sensor_serial),
        )
        if any(value < 0 for value in baselines) or baselines[0] <= 0:
            raise ValueError("cancel ack barrier baselines are invalid")
        self.stop_barrier = replace(
            barrier,
            cancel_acknowledged=True,
            cancel_ack_sim_ns=max(baselines[0], barrier.stop_requested_sim_ns),
            cancel_ack_odom_stamp_ns=max(
                baselines[1], barrier.stop_requested_odom_stamp_ns
            ),
            cancel_ack_odom_serial=max(
                baselines[2], barrier.stop_requested_odom_serial
            ),
            cancel_ack_camera_stamp_ns=max(
                baselines[3], barrier.stop_requested_camera_stamp_ns
            ),
            cancel_ack_camera_serial=max(
                baselines[4], barrier.stop_requested_camera_serial
            ),
        )
        return GateDecision(
            kind="cancel_ack",
            reason="Nav2 cancellation acknowledged; waiting for post-stop sensors",
            permits_model_step=False,
            requires_safe_stop=False,
            keep_safe_stop=True,
        )

    def inherit_confirmed_stop_barrier(
        self,
        *,
        episode_id: str,
        reset_generation: int,
        sequence_id: int,
        request_id: str,
        sim_stamp_ns: int,
        odom_stamp_ns: int,
        odom_serial: int,
        camera_sensor_stamp_ns: int,
        camera_sensor_serial: int,
        reason: str,
    ) -> GateDecision:
        """Carry a proven pre-reset cancellation into the new identity.

        The Nav2 cancellation itself is acknowledged while the old episode
        identity is still active.  After the model reset changes identity this
        method installs only the post-stop sensor cutoff; it never fabricates a
        cancellation acknowledgement for the new identity.
        """

        self._validate_identity(episode_id, reset_generation)
        if self.pending is not None or self.stop_barrier is not None:
            raise RuntimeError("new identity motion gate is not empty")
        baselines = (
            int(sim_stamp_ns),
            int(odom_stamp_ns),
            int(odom_serial),
            int(camera_sensor_stamp_ns),
            int(camera_sensor_serial),
        )
        if any(value < 0 for value in baselines) or baselines[0] <= 0:
            raise ValueError("inherited stop barrier baselines are invalid")
        self.stop_barrier = _StopBarrier(
            stop_requested_sim_ns=baselines[0],
            stop_requested_odom_stamp_ns=baselines[1],
            stop_requested_odom_serial=baselines[2],
            stop_requested_camera_stamp_ns=baselines[3],
            stop_requested_camera_serial=baselines[4],
            cancel_acknowledged=True,
            cancel_ack_sim_ns=baselines[0],
            cancel_ack_odom_stamp_ns=baselines[1],
            cancel_ack_odom_serial=baselines[2],
            cancel_ack_camera_stamp_ns=baselines[3],
            cancel_ack_camera_serial=baselines[4],
            reason=str(reason),
            action=0,
            sequence_id=int(sequence_id),
            request_id=str(request_id),
        )
        return GateDecision(
            kind="inherited_cancel_ack",
            reason=str(reason),
            permits_model_step=False,
            requires_safe_stop=False,
            keep_safe_stop=True,
        )

    def snapshot(self) -> dict[str, Any]:
        pending = self.pending
        barrier = self.stop_barrier
        return {
            "schema_version": 1,
            "state": self.state,
            "episode_id": self.active_episode_id,
            "reset_generation": self.active_reset_generation,
            "config": asdict(self.config),
            "pending": None
            if pending is None
            else {
                **asdict(pending),
                "start_pose": asdict(pending.start_pose),
            },
            "stop_barrier": None if barrier is None else asdict(barrier),
        }

    def _validate_identity(self, episode_id: str, reset_generation: int) -> None:
        if (
            str(episode_id) != self.active_episode_id
            or int(reset_generation) != self.active_reset_generation
        ):
            raise ValueError(
                "motion-gate identity mismatch: "
                f"active={self.active_episode_id}:{self.active_reset_generation} "
                f"observed={episode_id}:{reset_generation}"
            )

    def _bounds(self, action: int) -> tuple[float, float, float]:
        if action == ACTION_FORWARD:
            return (
                self.config.forward_required_m,
                self.config.forward_command_m,
                self.config.forward_timeout_sec,
            )
        return (
            math.radians(self.config.turn_required_deg),
            math.radians(self.config.turn_command_deg),
            self.config.turn_timeout_sec,
        )

    @staticmethod
    def _progress(action: int, start: Pose2D, current: Pose2D) -> float:
        if action == ACTION_FORWARD:
            delta_x = current.x - start.x
            delta_y = current.y - start.y
            return delta_x * math.cos(start.yaw_rad) + delta_y * math.sin(
                start.yaw_rad
            )
        yaw_delta = _wrapped_angle(current.yaw_rad - start.yaw_rad)
        return yaw_delta if action == ACTION_LEFT else -yaw_delta

    def _transition_to_stop(
        self,
        pending: _PendingMotion,
        *,
        sim_stamp_ns: int,
        odom_stamp_ns: int,
        odom_serial: int,
        kind: str,
        reason: str,
        progress: float = 0.0,
        required: float = 0.0,
        commanded: float = 0.0,
        elapsed: float = 0.0,
    ) -> GateDecision:
        self.pending = None
        self.stop_barrier = _StopBarrier(
            stop_requested_sim_ns=int(sim_stamp_ns),
            stop_requested_odom_stamp_ns=int(odom_stamp_ns),
            stop_requested_odom_serial=int(odom_serial),
            stop_requested_camera_stamp_ns=0,
            stop_requested_camera_serial=0,
            cancel_acknowledged=False,
            cancel_ack_sim_ns=0,
            cancel_ack_odom_stamp_ns=0,
            cancel_ack_odom_serial=0,
            cancel_ack_camera_stamp_ns=0,
            cancel_ack_camera_serial=0,
            reason=str(reason),
            action=pending.action,
            sequence_id=pending.sequence_id,
            request_id=pending.request_id,
        )
        return GateDecision(
            kind=kind,
            reason=reason,
            permits_model_step=False,
            requires_safe_stop=True,
            keep_safe_stop=True,
            progress=progress,
            required_progress=required,
            commanded_progress=commanded,
            elapsed_sim_sec=elapsed,
        )
