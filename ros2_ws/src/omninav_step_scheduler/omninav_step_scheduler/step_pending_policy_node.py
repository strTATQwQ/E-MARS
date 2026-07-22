from __future__ import annotations

import json
import time
from typing import Any

from .runtime_mode import mode_payload_from_json
from .schemas import SchedulerState, attach_timebase, deep_get, load_yaml_file, make_metric, parse_safety_status
from .stale_gate import pose_delta_m, yaw_delta_deg

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover
    rclpy = None
    Node = object
    String = None
    Twist = None


def safe_to_move_while_thinking(
    safety,
    current_committed_primitive: dict[str, Any] | None,
    step_request_multimodal: bool,
    pose_at_step_request: list[float],
    current_pose: list[float],
    config: dict[str, Any] | None = None,
) -> bool:
    if not deep_get(config, "pending_policy.allow_move_while_step", True):
        return False
    if step_request_multimodal and deep_get(config, "pending_policy.stop_if_step_multimodal", True):
        return False
    if not current_committed_primitive or not current_committed_primitive.get("interruptible", True):
        return False
    if not safety.local_costmap_clear or safety.estop or safety.robot_fallen_or_unstable:
        return False
    if safety.obstacle_distance_m is not None and safety.obstacle_distance_m <= float(deep_get(config, "pending_policy.stop_if_obstacle_within_m", 0.8)):
        return False
    if safety.human_distance_m is not None and safety.human_distance_m <= float(deep_get(config, "pending_policy.stop_if_human_within_m", 2.0)):
        return False
    if safety.near_doorway or safety.near_intersection or safety.dynamic_obstacle or safety.on_slope_or_stairs:
        return False
    if pose_delta_m(pose_at_step_request, current_pose) >= float(deep_get(config, "pending_policy.thinking_max_distance_m", 0.30)):
        return False
    if yaw_delta_deg(pose_at_step_request[2], current_pose[2]) >= float(deep_get(config, "pending_policy.thinking_max_yaw_deg", 10.0)):
        return False
    return True


def choose_pending_policy_mode(
    state: str,
    safety,
    current_committed_primitive: dict[str, Any] | None,
    step_request_multimodal: bool,
    pose_at_step_request: list[float],
    current_pose: list[float],
    config: dict[str, Any] | None = None,
) -> str:
    if state == SchedulerState.STEP_THINK_SCAN.value:
        if safety.local_costmap_clear and not safety.estop and not safety.dynamic_obstacle:
            return "safe_scan"
        return "stop"
    if state == SchedulerState.STEP_THINK_MOVE.value and safe_to_move_while_thinking(
        safety, current_committed_primitive, step_request_multimodal, pose_at_step_request, current_pose, config
    ):
        return "move_slow"
    return "stop"


def pending_wait_exceeded(started_at: float, current_time: float, max_wait_sec: float) -> bool:
    return bool(float(started_at) > 0.0 and float(current_time) - float(started_at) >= max(0.1, float(max_wait_sec)))


class StepPendingPolicyNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run StepPendingPolicyNode")
        super().__init__("step_pending_policy")
        self.declare_parameter("config_file", "")
        self.config = load_yaml_file(self.get_parameter("config_file").value)
        self.state = SchedulerState.IDLE.value
        self.safety = parse_safety_status({})
        self.current_primitive: dict[str, Any] | None = None
        self.step_request_multimodal = False
        self.pose_at_step_request = [0.0, 0.0, 0.0]
        self.current_pose = [0.0, 0.0, 0.0]
        self.active_episode_id = ""
        self.pending_started_at = 0.0
        self.pending_wait_reported = False
        self.mode_pub = self.create_publisher(String, "/scheduler/step_pending_mode", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_candidate", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/scheduler/state", self.on_state, 10)
        self.create_subscription(String, "/safety/local_status_json", self.on_safety, 10)
        self.create_subscription(String, "/primitive/command_json", self.on_primitive, 10)
        self.create_subscription(String, "/step/request_json", self.on_step_request, 10)
        self.create_subscription(String, "/robot_state_json", self.on_robot_state, 10)
        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
        self.create_subscription(String, "/episode/reset_lifecycle_json", self.on_reset, 10)
        self.create_timer(0.1, self.tick)

    def on_benchmark_mode(self, msg):
        payload = mode_payload_from_json(msg.data)
        episode_id = str(payload.get("episode_id", ""))
        if episode_id and episode_id != self.active_episode_id:
            self.active_episode_id = episode_id
            self.clear_pending("benchmark_mode_change")

    def on_reset(self, msg):
        payload = _safe_json(msg.data)
        episode_id = str(payload.get("episode_id") or "")
        if episode_id:
            self.active_episode_id = episode_id
        self.clear_pending(str(payload.get("type") or "episode_reset"))

    def clear_pending(self, reason: str) -> None:
        self.current_primitive = None
        self.step_request_multimodal = False
        self.pose_at_step_request = [0.0, 0.0, 0.0]
        self.pending_started_at = 0.0
        self.pending_wait_reported = False
        self.publish_metric("step_pending_policy_reset", reason=reason, result="cleared")

    def on_state(self, msg):
        self.state = msg.data
        if self.state not in {SchedulerState.STEP_THINK_STOP.value, SchedulerState.STEP_THINK_MOVE.value, SchedulerState.STEP_THINK_SCAN.value}:
            self.pending_started_at = 0.0
            self.pending_wait_reported = False

    def on_safety(self, msg):
        self.safety = parse_safety_status(msg.data)

    def on_primitive(self, msg):
        self.current_primitive = _safe_json(msg.data)

    def on_step_request(self, msg):
        payload = _safe_json(msg.data)
        self.step_request_multimodal = bool(payload.get("multimodal", False))
        self.pose_at_step_request = list(payload.get("pose_at_request") or [0.0, 0.0, 0.0])
        self.pending_started_at = time.monotonic()
        self.pending_wait_reported = False

    def on_robot_state(self, msg):
        payload = _safe_json(msg.data)
        self.current_pose = list(payload.get("pose") or self.current_pose)

    def tick(self):
        if self.state not in {SchedulerState.STEP_THINK_STOP.value, SchedulerState.STEP_THINK_MOVE.value, SchedulerState.STEP_THINK_SCAN.value}:
            return
        mode = choose_pending_policy_mode(
            self.state,
            self.safety,
            self.current_primitive,
            self.step_request_multimodal,
            self.pose_at_step_request,
            self.current_pose,
            self.config,
        )
        current_time = time.monotonic()
        wait_sec = current_time - self.pending_started_at if self.pending_started_at > 0.0 else 0.0
        if pending_wait_exceeded(
            self.pending_started_at,
            current_time,
            float(deep_get(self.config, "pending_policy.max_wait_sec", 15.0)),
        ):
            mode = "stop"
            if not self.pending_wait_reported:
                self.pending_wait_reported = True
                self.publish_metric(
                    "step_pending_wait_exceeded",
                    pending_mode="stop",
                    wait_sec=round(wait_sec, 3),
                    result="hold_position",
                )
        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)
        self.cmd_pub.publish(self.build_twist(mode))
        self.publish_metric("step_pending_policy", pending_mode=mode)

    def build_twist(self, mode: str):
        twist = Twist()
        if mode == "move_slow":
            twist.linear.x = min(
                float(deep_get(self.config, "pending_policy.thinking_speed_limit_mps", 0.12)),
                float(deep_get(self.config, "primitive.max_linear_x", 0.25)),
            )
        elif mode == "safe_scan":
            twist.angular.z = float(deep_get(self.config, "pending_policy.scan_yaw_rate", 0.15))
        return twist

    def publish_metric(self, event_type: str, **kwargs):
        msg = String()
        msg.data = json.dumps(attach_timebase(make_metric(event_type, model="none", state=self.state, **kwargs), node=self, episode_id=self.active_episode_id), ensure_ascii=False)
        self.metric_pub.publish(msg)


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def main(args=None):
    rclpy.init(args=args)
    node = StepPendingPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
