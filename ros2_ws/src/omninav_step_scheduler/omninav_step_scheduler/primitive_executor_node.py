from __future__ import annotations

import json
import math

from .runtime_mode import mode_payload_from_json
from .schemas import attach_timebase, deep_get, load_yaml_file, make_metric, node_ros_now_sec, now, parse_safety_status

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


class PrimitiveExecutorNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run PrimitiveExecutorNode")
        super().__init__("primitive_executor")
        self.declare_parameter("config_file", "")
        self.config = load_yaml_file(self.get_parameter("config_file").value)
        self.safety = parse_safety_status({})
        self.current_pose = [0.0, 0.0, 0.0]
        self.active_twist = Twist()
        self.active_until = 0.0
        self.active_episode_id = ""
        self.low_speed_primitive_stabilizer = False
        self.blocked_turn_direction = 0.0
        self.blocked_turn_until = 0.0
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_candidate", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/primitive/command_json", self.on_primitive, 10)
        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
        self.create_subscription(String, "/episode/reset_lifecycle_json", self.on_reset, 10)
        self.create_subscription(String, "/safety/local_status_json", self.on_safety, 10)
        self.create_subscription(String, "/robot_state_json", self.on_robot_state, 10)
        self.create_timer(0.05, self.tick)

    def on_benchmark_mode(self, msg):
        payload = mode_payload_from_json(msg.data)
        mode_config = payload.get("mode_config") if isinstance(payload.get("mode_config"), dict) else {}
        self.low_speed_primitive_stabilizer = bool(
            mode_config.get("low_speed_primitive_stabilizer", False)
            or str(payload.get("mode") or "").endswith("sim2real_low_speed")
        )
        episode_id = str(payload.get("episode_id", ""))
        if episode_id and episode_id != self.active_episode_id:
            self.active_episode_id = episode_id
            self.clear_active("benchmark_mode_change")

    def on_reset(self, msg):
        payload = _safe_json(msg.data)
        episode_id = str(payload.get("episode_id") or "")
        if episode_id:
            self.active_episode_id = episode_id
        self.clear_active(str(payload.get("type") or "episode_reset"))

    def clear_active(self, reason: str) -> None:
        self.active_twist = Twist()
        self.active_until = 0.0
        self.blocked_turn_direction = 0.0
        self.blocked_turn_until = 0.0
        self.cmd_pub.publish(Twist())
        self.publish_metric("primitive_executor_reset", reason=reason, episode_id=self.active_episode_id, result="cleared")

    def on_safety(self, msg):
        self.safety = parse_safety_status(msg.data)

    def on_robot_state(self, msg):
        payload = _safe_json(msg.data)
        pose = payload.get("pose")
        if isinstance(pose, list) and len(pose) >= 3:
            try:
                self.current_pose = [float(pose[0]), float(pose[1]), float(pose[2])]
            except (TypeError, ValueError):
                pass

    def on_primitive(self, msg):
        primitive = _safe_json(msg.data)
        primitive_episode_id = str(primitive.get("episode_id") or "")
        if self.active_episode_id and primitive_episode_id and primitive_episode_id != self.active_episode_id:
            self.publish_metric(
                "primitive_command_stale",
                action_type=primitive.get("primitive"),
                request_id=primitive.get("request_id"),
                attribution="episode_mismatch",
                result="discarded",
            )
            return
        requested_kind = str(primitive.get("primitive") or "stop")
        effective_primitive = primitive
        lock_override = False
        if self.low_speed_primitive_stabilizer:
            escape_rate = lane_centering_yaw_rate(self.current_pose, max_yaw_rate=1.0)
            escape_direction = 1.0 if escape_rate >= 0.0 else -1.0
            effective_kind, self.blocked_turn_direction, self.blocked_turn_until, lock_override = blocked_turn_lock_action(
                requested_kind,
                local_costmap_clear=self.safety.local_costmap_clear,
                lock_direction=self.blocked_turn_direction,
                locked_until=self.blocked_turn_until,
                timestamp=now(),
                escape_direction=escape_direction,
            )
            if effective_kind != requested_kind:
                effective_primitive = dict(primitive)
                effective_primitive["primitive"] = effective_kind
        twist = self.primitive_to_twist(effective_primitive)
        if motion_blocked_by_safety(twist, self.safety):
            twist = Twist()
            result = "stopped"
        else:
            result = "accepted"
        self.active_twist = twist
        self.active_until = now() + self.primitive_ttl(primitive)
        self.cmd_pub.publish(twist)
        self.publish_metric(
            "primitive_command",
            action_type=primitive.get("primitive"),
            request_id=primitive.get("request_id"),
            episode_id=primitive_episode_id or self.active_episode_id,
            cmd_vel=_twist_dict(twist),
            primitive=primitive,
            effective_action_type=effective_primitive.get("primitive"),
            blocked_turn_lock_active=lock_override,
            result=result,
        )

    def tick(self):
        if now() > self.active_until:
            return
        if motion_blocked_by_safety(self.active_twist, self.safety):
            self.cmd_pub.publish(Twist())
            return
        self.cmd_pub.publish(self.active_twist)

    def primitive_ttl(self, primitive: dict) -> float:
        try:
            ttl = float(primitive.get("ttl_sec", deep_get(self.config, "primitive.cmd_vel_ttl_sec", 0.3)))
        except (TypeError, ValueError):
            ttl = float(deep_get(self.config, "primitive.cmd_vel_ttl_sec", 0.3))
        return max(0.05, min(1.5, ttl))

    def primitive_to_twist(self, primitive: dict):
        twist = Twist()
        kind = primitive.get("primitive", "stop")
        normal_speed = float(deep_get(self.config, "primitive.normal_speed_mps", 0.20))
        try:
            requested_speed = float(primitive.get("speed_mps", normal_speed))
        except (TypeError, ValueError):
            requested_speed = normal_speed
        thinking_speed = float(deep_get(self.config, "primitive.thinking_speed_mps", 0.12))
        max_linear_x = float(deep_get(self.config, "primitive.max_linear_x", 0.25))
        max_yaw_rate = float(deep_get(self.config, "primitive.max_yaw_rate", 0.35))
        max_forward = float(deep_get(self.config, "primitive.max_forward_horizon_m", 0.75))
        if kind == "move_forward":
            distance = min(abs(float(primitive.get("distance_m", 0.25))), max_forward)
            twist.linear.x = min(max(0.0, requested_speed), max_linear_x) if distance > 0.0 else 0.0
            if twist.linear.x > 0.0 and bool(deep_get(self.config, "primitive.lane_centering_enabled", False)):
                twist.angular.z = lane_centering_yaw_rate(
                    self.current_pose,
                    center_y=float(deep_get(self.config, "primitive.lane_center_y_m", 0.0)),
                    lookahead_m=float(deep_get(self.config, "primitive.lane_center_lookahead_m", 2.0)),
                    gain=float(deep_get(self.config, "primitive.lane_center_gain", 0.8)),
                    max_yaw_rate=min(max_yaw_rate, float(deep_get(self.config, "primitive.lane_center_max_yaw_rate", 0.28))),
                )
        elif kind == "follow_waypoint":
            distance = min(abs(float(primitive.get("distance_m", 0.25))), max_forward)
            yaw = max(-45.0, min(45.0, float(primitive.get("yaw_deg", 0.0))))
            twist.linear.x = min(max(0.0, requested_speed), max_linear_x) if distance > 0.0 else 0.0
            if abs(yaw) > 20.0:
                twist.linear.x *= 0.5
            twist.angular.z = max(-max_yaw_rate, min(max_yaw_rate, math.radians(yaw)))
        elif kind == "back_off":
            twist.linear.x = -min(thinking_speed, max_linear_x)
        elif kind == "turn_left":
            yaw = min(abs(float(primitive.get("yaw_deg", 15.0))), 45.0)
            requested_yaw = min(max_yaw_rate, math.radians(yaw))
            if self.low_speed_primitive_stabilizer:
                twist.linear.x, twist.angular.z = low_speed_turn_command(
                    1.0,
                    local_costmap_clear=self.safety.local_costmap_clear,
                    max_linear_x=max_linear_x,
                    max_yaw_rate=min(max_yaw_rate, requested_yaw),
                )
            else:
                twist.angular.z = requested_yaw
        elif kind == "turn_right":
            yaw = min(abs(float(primitive.get("yaw_deg", 15.0))), 45.0)
            requested_yaw = min(max_yaw_rate, math.radians(yaw))
            if self.low_speed_primitive_stabilizer:
                twist.linear.x, twist.angular.z = low_speed_turn_command(
                    -1.0,
                    local_costmap_clear=self.safety.local_costmap_clear,
                    max_linear_x=max_linear_x,
                    max_yaw_rate=min(max_yaw_rate, requested_yaw),
                )
            else:
                twist.angular.z = -requested_yaw
        elif kind == "look_around":
            default_scan = float(deep_get(self.config, "pending_policy.scan_yaw_rate", 0.15))
            try:
                requested_scan = float(primitive.get("angular_z_radps", default_scan))
            except (TypeError, ValueError):
                requested_scan = default_scan
            twist.angular.z = max(-max_yaw_rate, min(max_yaw_rate, requested_scan))
        elif kind in {"enter_branch", "target_relative_approach"}:
            phase = str(primitive.get("phase") or "")
            if phase not in {"verify_branch", "verify", "stop", "done", "failsafe"}:
                try:
                    requested_linear = float(primitive.get("linear_x_mps", 0.0))
                except (TypeError, ValueError):
                    requested_linear = 0.0
                try:
                    requested_angular = float(primitive.get("angular_z_radps", 0.0))
                except (TypeError, ValueError):
                    requested_angular = 0.0
                try:
                    primitive_max_linear = float(primitive.get("max_linear_x_mps", max_linear_x))
                except (TypeError, ValueError):
                    primitive_max_linear = max_linear_x
                try:
                    primitive_max_yaw = float(primitive.get("max_yaw_rate_radps", max_yaw_rate))
                except (TypeError, ValueError):
                    primitive_max_yaw = max_yaw_rate
                linear_limit = min(max_linear_x, abs(primitive_max_linear))
                yaw_limit = min(max_yaw_rate, abs(primitive_max_yaw))
                twist.linear.x = max(-linear_limit, min(linear_limit, requested_linear))
                twist.angular.z = max(-yaw_limit, min(yaw_limit, requested_angular))
        return twist

    def publish_metric(self, event_type: str, **kwargs):
        msg = String()
        payload = attach_timebase(make_metric(event_type, model="none", **kwargs), node=self, episode_id=self.active_episode_id)
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


def is_escape_twist(twist, *, epsilon: float = 0.02) -> bool:
    linear_x = float(twist.linear.x)
    linear_y = float(twist.linear.y)
    angular_z = float(twist.angular.z)
    if linear_x < -epsilon:
        return True
    return abs(linear_x) <= epsilon and abs(linear_y) <= epsilon and abs(angular_z) > epsilon


def motion_blocked_by_safety(twist, safety) -> bool:
    if safety.estop or safety.robot_fallen_or_unstable:
        return True
    if safety.local_costmap_clear:
        return False
    return not is_escape_twist(twist)


def lane_centering_yaw_rate(
    pose: list[float],
    *,
    center_y: float = 0.0,
    lookahead_m: float = 2.0,
    gain: float = 0.8,
    max_yaw_rate: float = 0.28,
) -> float:
    if len(pose) < 3:
        return 0.0
    y = float(pose[1])
    yaw = float(pose[2])
    lookahead = max(0.1, float(lookahead_m))
    desired_yaw = math.atan2(float(center_y) - y, lookahead)
    error = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
    rate = float(gain) * error
    limit = abs(float(max_yaw_rate))
    return max(-limit, min(limit, rate))


def low_speed_turn_command(
    direction: float,
    *,
    local_costmap_clear: bool,
    crawl_speed_mps: float = 0.08,
    yaw_rate_radps: float = 0.20,
    blocked_reverse_speed_mps: float = 0.08,
    max_linear_x: float = 0.20,
    max_yaw_rate: float = 0.30,
) -> tuple[float, float]:
    sign = 1.0 if float(direction) >= 0.0 else -1.0
    if local_costmap_clear:
        linear = min(abs(float(crawl_speed_mps)), abs(float(max_linear_x)))
    else:
        linear = -min(abs(float(blocked_reverse_speed_mps)), abs(float(max_linear_x)))
    angular = sign * min(abs(float(yaw_rate_radps)), abs(float(max_yaw_rate)))
    return linear, angular


def blocked_turn_lock_action(
    requested_kind: str,
    *,
    local_costmap_clear: bool,
    lock_direction: float,
    locked_until: float,
    timestamp: float,
    hold_sec: float = 4.0,
    escape_direction: float = 1.0,
) -> tuple[str, float, float, bool]:
    """Hold one pure-turn direction while a static obstacle blocks forward motion."""

    kind = str(requested_kind or "stop")
    now_sec = float(timestamp)
    lock_active = abs(float(lock_direction)) > 0.0 and now_sec < float(locked_until)
    if local_costmap_clear:
        if lock_active and kind in {"move_forward", "follow_waypoint", "look_around", "turn_left", "turn_right"}:
            effective = "turn_left" if float(lock_direction) > 0.0 else "turn_right"
            return effective, float(lock_direction), float(locked_until), True
        if lock_active and kind == "stop":
            return kind, float(lock_direction), float(locked_until), True
        return kind, 0.0, 0.0, False

    if not lock_active and kind in {"turn_left", "turn_right"}:
        direction = 1.0 if kind == "turn_left" else -1.0
        return kind, direction, now_sec + max(0.1, float(hold_sec)), True

    if not lock_active and kind in {"move_forward", "follow_waypoint"}:
        direction = 1.0 if float(escape_direction) >= 0.0 else -1.0
        effective = "turn_left" if direction > 0.0 else "turn_right"
        return effective, direction, now_sec + max(0.1, float(hold_sec)), True

    if not lock_active:
        return kind, 0.0, 0.0, False

    if kind in {"move_forward", "follow_waypoint", "look_around", "turn_left", "turn_right"}:
        effective = "turn_left" if float(lock_direction) > 0.0 else "turn_right"
        return effective, float(lock_direction), float(locked_until), True

    # An explicit stop always wins, while retaining the lock for a subsequent
    # navigation command inside the bounded hold window.
    return kind, float(lock_direction), float(locked_until), True


def main(args=None):
    rclpy.init(args=args)
    node = PrimitiveExecutorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
