from __future__ import annotations

import json
import os
import time

from .runtime_mode import mode_payload_from_json
from .schemas import SchedulerState, attach_timebase, deep_get, load_yaml_file, make_metric, now, parse_safety_status
from .primitive_executor_node import is_escape_twist

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


class SafeCmdMuxNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run SafeCmdMuxNode")
        super().__init__("safe_cmd_mux")
        self.declare_parameter("config_file", "")
        self.declare_parameter("dry_run", None)
        self.config = load_yaml_file(self.get_parameter("config_file").value)
        dry_param = self.get_parameter("dry_run").value
        self.dry_run = bool(deep_get(self.config, "safety.dry_run", True)) if dry_param is None else bool(dry_param)
        self.state = SchedulerState.IDLE.value
        self.safety = parse_safety_status({})
        self.latest_cmd = Twist()
        self.latest_cmd_time = 0.0
        self.last_output = Twist()
        self.last_output_time = time.monotonic()
        self.active_episode_id = ""
        output_topic = deep_get(self.config, "safety.output_topic", "/safe_cmd_vel")
        self.safe_pub = self.create_publisher(Twist, output_topic, 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(Twist, "/cmd_vel_candidate", self.on_candidate, 10)
        self.create_subscription(String, "/safety/local_status_json", self.on_safety, 10)
        self.create_subscription(String, "/scheduler/state", self.on_state, 10)
        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
        self.create_subscription(String, "/episode/reset_lifecycle_json", self.on_reset, 10)
        self.create_timer(0.05, self.tick)

    def on_benchmark_mode(self, msg):
        payload = mode_payload_from_json(msg.data)
        episode_id = str(payload.get("episode_id", ""))
        if episode_id and episode_id != self.active_episode_id:
            self.active_episode_id = episode_id
            self.clear_latest("benchmark_mode_change")

    def on_reset(self, msg):
        payload = _safe_json(msg.data)
        episode_id = str(payload.get("episode_id") or "")
        if episode_id:
            self.active_episode_id = episode_id
        self.clear_latest(str(payload.get("type") or "episode_reset"))

    def clear_latest(self, reason: str) -> None:
        self.latest_cmd = Twist()
        self.latest_cmd_time = 0.0
        self.reset_rate_limiter()
        self.safe_pub.publish(Twist())
        self.publish_metric("safe_cmd_mux_reset", result="cleared", reason=reason)

    def on_candidate(self, msg):
        self.latest_cmd = msg
        self.latest_cmd_time = now()

    def on_safety(self, msg):
        self.safety = parse_safety_status(msg.data)

    def on_state(self, msg):
        self.state = msg.data

    def tick(self):
        cmd = self.filtered_cmd()
        self.safe_pub.publish(cmd)

    def filtered_cmd(self):
        zero = Twist()
        age = now() - self.latest_cmd_time if self.latest_cmd_time else 999.0
        ttl = float(deep_get(self.config, "primitive.cmd_vel_ttl_sec", 0.3))
        result = "accepted"
        if self.dry_run:
            result = "dry_run_zero"
            self.publish_metric("safe_cmd_mux", result=result, cmd_vel=_twist_dict(self.latest_cmd), dry_run=True)
            return zero
        env_name = str(deep_get(self.config, "safety.enable_motion_env_name", "GO2_ENABLE_MOTION"))
        if deep_get(self.config, "safety.require_enable_motion_env", True) and os.environ.get(env_name) != "1":
            result = "missing_enable_env"
        elif deep_get(self.config, "safety.require_deadman_for_real_motion", True) and not self.safety.deadman:
            result = "deadman_false"
        elif self.safety.estop or self.safety.robot_fallen_or_unstable:
            result = "safety_stop"
        elif not self.safety.local_costmap_clear and not is_escape_twist(self.latest_cmd):
            result = "safety_stop"
        elif self.state == SchedulerState.FAILSAFE.value:
            result = "failsafe"
        elif age > ttl:
            result = "stale_cmd"
        else:
            cmd = self.rate_limit(self.clamp(self.latest_cmd))
            self.publish_metric("safe_cmd_mux", result=result, cmd_vel=_twist_dict(cmd), dry_run=False)
            return cmd
        self.reset_rate_limiter()
        self.publish_metric("safe_cmd_mux", result=result, cmd_vel=_twist_dict(zero), dry_run=False)
        return zero

    def clamp(self, cmd):
        out = Twist()
        max_x = float(deep_get(self.config, "primitive.max_linear_x", 0.25))
        max_yaw = float(deep_get(self.config, "primitive.max_yaw_rate", 0.35))
        out.linear.x = max(-max_x, min(max_x, cmd.linear.x))
        out.angular.z = max(-max_yaw, min(max_yaw, cmd.angular.z))
        return out

    def rate_limit(self, cmd):
        current = time.monotonic()
        dt = max(0.0, min(current - self.last_output_time, 0.25))
        linear_accel = float(deep_get(self.config, "primitive.accel_limit_mps2", 0.0))
        linear_decel = float(deep_get(self.config, "primitive.decel_limit_mps2", linear_accel))
        yaw_accel = float(deep_get(self.config, "primitive.yaw_accel_limit_radps2", 0.0))
        yaw_decel = float(deep_get(self.config, "primitive.yaw_decel_limit_radps2", yaw_accel))
        out = Twist()
        out.linear.x = rate_limit_scalar(self.last_output.linear.x, cmd.linear.x, linear_accel, dt, linear_decel)
        out.angular.z = rate_limit_scalar(self.last_output.angular.z, cmd.angular.z, yaw_accel, dt, yaw_decel)
        self.last_output = out
        self.last_output_time = current
        return out

    def reset_rate_limiter(self) -> None:
        self.last_output = Twist()
        self.last_output_time = time.monotonic()

    def publish_metric(self, event_type: str, **kwargs):
        msg = String()
        payload = attach_timebase(make_metric(event_type, model="none", state=self.state, **kwargs), node=self, episode_id=self.active_episode_id)
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.metric_pub.publish(msg)


def _safe_json(raw: str) -> dict:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def _twist_dict(twist) -> dict:
    return {
        "linear": {"x": twist.linear.x, "y": twist.linear.y, "z": twist.linear.z},
        "angular": {"x": twist.angular.x, "y": twist.angular.y, "z": twist.angular.z},
    }


def rate_limit_scalar(
    previous: float,
    requested: float,
    accel_limit_per_sec: float,
    dt: float,
    decel_limit_per_sec: float | None = None,
) -> float:
    limit_per_sec = accel_limit_per_sec
    if previous * requested >= 0.0 and abs(requested) < abs(previous):
        limit_per_sec = decel_limit_per_sec if decel_limit_per_sec is not None else accel_limit_per_sec
    if limit_per_sec <= 0.0 or dt <= 0.0:
        return requested
    max_delta = limit_per_sec * dt
    return previous + max(-max_delta, min(max_delta, requested - previous))


def main(args=None):
    rclpy.init(args=args)
    node = SafeCmdMuxNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
