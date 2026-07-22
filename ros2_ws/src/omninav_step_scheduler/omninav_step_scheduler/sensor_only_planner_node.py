from __future__ import annotations

import json
import time
from typing import Any

from .runtime_mode import mode_payload_from_json
from .schemas import attach_timebase, make_metric, node_ros_now_sec
from .sensor_only_planning import (
    SensorRouteController,
    SpatialTargetTrackState,
    instruction_visual_target,
    payload_oracle_fields,
    semantic_track_primitive,
)

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover
    rclpy = None
    Node = object
    String = None


class SensorOnlyPlannerNode(Node):
    """Convert fresh spatial sensor tracks into normal-chain primitives."""

    def __init__(self) -> None:
        if rclpy is None:
            raise RuntimeError("rclpy is required to run SensorOnlyPlannerNode")
        super().__init__("sensor_only_planner")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.enabled = False
        self.active_episode_id = ""
        self.task_type = ""
        self.pose = [0.0, 0.0, 0.0]
        self.free_space: dict[str, Any] = {}
        self.last_observation_monotonic = 0.0
        self.last_primitive_signature = ""
        self.last_primitive_publish = 0.0
        self.threshold_event_sent = False
        self.route_event_sent = False
        self.mode: dict[str, Any] = {}
        self.track = SpatialTargetTrackState()
        self.route = SensorRouteController()

        self.track_pub = self.create_publisher(String, "/perception/target_track_json", 10)
        self.primitive_pub = self.create_publisher(String, "/primitive/command_json", 10)
        self.mission_event_pub = self.create_publisher(String, "/mission/event_json", 10)
        self.query_pub = self.create_publisher(String, "/perception/query_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
        self.create_subscription(String, "/episode/reset_lifecycle_json", self.on_reset, 10)
        self.create_subscription(String, "/robot_state_json", self.on_robot_state, 10)
        self.create_subscription(String, "/perception/target_observation_json", self.on_observation, 10)
        self.create_subscription(String, "/planner/route_decision_json", self.on_route_decision, 10)
        self.create_subscription(String, "/user_instruction", self.on_instruction, 10)
        period = 1.0 / max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(period, self.tick)

    def on_benchmark_mode(self, msg: String) -> None:
        payload = mode_payload_from_json(msg.data)
        mode = payload.get("mode_config") if isinstance(payload.get("mode_config"), dict) else {}
        episode_id = str(payload.get("episode_id") or "")
        if episode_id and episode_id != self.active_episode_id:
            self.reset_state(episode_id=episode_id, target_id=str(mode.get("perception_target_id") or ""))
        self.mode = dict(mode)
        self.enabled = bool(mode.get("sensor_only_planning", False))
        self.task_type = str(mode.get("sensor_planning_task_type") or "")
        self.track.required_hits = max(2, int(mode.get("sensor_track_required_hits", 2)))
        self.track.min_confidence = float(mode.get("sensor_track_min_confidence", 0.60))
        self.track.retain_confidence = float(mode.get("sensor_track_retain_confidence", 0.25))
        self.track.max_misses = int(mode.get("sensor_track_max_misses", 2))
        self.track.max_age_sec = float(mode.get("sensor_track_max_age_sec", 3.0))
        self.track.max_distance_jump_m = float(mode.get("sensor_track_max_distance_jump_m", 1.0))
        self.track.max_bearing_jump_deg = float(mode.get("sensor_track_max_bearing_jump_deg", 25.0))
        self.route.turn_angle_deg = float(mode.get("sensor_route_turn_angle_deg", 75.0))
        self.route.post_turn_forward_m = float(mode.get("sensor_route_post_forward_m", 1.0))
        self.route.approach_trigger_distance_m = float(mode.get("sensor_route_approach_trigger_distance_m", 3.3))
        self.route.max_approach_distance_m = float(mode.get("sensor_route_max_approach_m", 3.4))
        self.route.target_stop_distance_m = float(mode.get("sensor_route_target_stop_distance_m", 1.6))
        self.route.obstacle_bypass_trigger_m = float(mode.get("sensor_route_obstacle_bypass_trigger_m", 1.85))
        self.route.obstacle_bypass_turn_deg = float(mode.get("sensor_route_obstacle_bypass_turn_deg", 40.0))
        self.route.obstacle_bypass_forward_m = float(mode.get("sensor_route_obstacle_bypass_forward_m", 1.4))
        self.route.obstacle_bypass_speed_mps = float(mode.get("sensor_route_obstacle_bypass_speed_mps", 0.15))
        self.route.obstacle_bypass_min_advance_clearance_m = float(
            mode.get("sensor_route_obstacle_bypass_min_advance_clearance_m", 1.35)
        )

    def on_instruction(self, msg: String) -> None:
        if not self.enabled:
            return
        payload = _json(msg.data)
        query = instruction_visual_target(str(payload.get("instruction") or ""))
        if not query:
            return
        self.track.reset(episode_id=self.active_episode_id, target_id=query)
        self.publish_json(
            self.query_pub,
            {
                "episode_id": self.active_episode_id,
                "target_id": query,
                "target_query": query,
                "source": "user_instruction_visual_phrase",
            },
        )
        self.publish_metric("sensor_perception_query", result="published", target_id=query)

    def on_reset(self, msg: String) -> None:
        payload = _json(msg.data)
        self.reset_state(
            episode_id=str(payload.get("episode_id") or self.active_episode_id),
            target_id=str(self.mode.get("perception_target_id") or ""),
        )

    def reset_state(self, *, episode_id: str, target_id: str) -> None:
        self.active_episode_id = episode_id
        self.track.reset(episode_id=episode_id, target_id=target_id)
        self.route.reset()
        self.free_space = {}
        self.last_observation_monotonic = 0.0
        self.last_primitive_signature = ""
        self.last_primitive_publish = 0.0
        self.threshold_event_sent = False
        self.route_event_sent = False

    def on_robot_state(self, msg: String) -> None:
        payload = _json(msg.data)
        pose = payload.get("pose")
        if isinstance(pose, list) and len(pose) >= 3:
            try:
                self.pose = [float(pose[0]), float(pose[1]), float(pose[2])]
            except (TypeError, ValueError):
                return

    def on_observation(self, msg: String) -> None:
        if not self.enabled:
            return
        payload = _json(msg.data)
        forbidden = payload_oracle_fields(payload)
        if forbidden:
            self.publish_metric("sensor_observation_rejected", result="oracle_leakage", forbidden_fields=forbidden)
            return
        episode_id = str(payload.get("episode_id") or "")
        if self.active_episode_id and episode_id != self.active_episode_id:
            self.publish_metric("sensor_observation_rejected", result="episode_mismatch", observation=payload)
            return
        try:
            observation = dict(payload)
            observation["observer_pose"] = list(self.pose)
            observation["observer_yaw_rad"] = float(self.pose[2])
            track = self.track.update(observation, timestamp=node_ros_now_sec(self))
        except Exception as exc:
            self.publish_metric("sensor_observation_rejected", result="parse_error", error=repr(exc))
            return
        free_space = payload.get("free_space_m")
        if isinstance(free_space, dict):
            self.free_space = dict(free_space)
        self.last_observation_monotonic = time.monotonic()
        self.publish_json(self.track_pub, track)
        self.publish_metric(
            "sensor_target_track",
            result=track.get("update_result"),
            track=track,
            free_space_m=self.free_space,
        )
        if (
            self.task_type == "route_choice"
            and not self.route_event_sent
            and bool(track.get("confirmed"))
            and bool(track.get("fresh"))
        ):
            self.route_event_sent = True
            self.publish_json(
                self.mission_event_pub,
                {
                    "type": "route_choice_upcoming",
                    "reason": "route_landmark_detected_in_actual_viewport",
                    "episode_id": self.active_episode_id,
                    "mission_id": self.active_episode_id,
                    "frame_seq": payload.get("frame_seq"),
                    "image_source": "actual_isaac_viewport",
                    "source": "sensor_only_route_trigger",
                },
            )
            self.publish_metric("sensor_route_trigger", result="route_choice_upcoming")

    def on_route_decision(self, msg: String) -> None:
        if not self.enabled:
            return
        payload = _json(msg.data)
        if payload_oracle_fields(payload):
            self.publish_metric("sensor_route_decision_rejected", result="oracle_leakage", decision=payload)
            return
        episode_id = str(payload.get("episode_id") or "")
        if self.active_episode_id and episode_id != self.active_episode_id:
            self.publish_metric("sensor_route_decision_rejected", result="episode_mismatch", decision=payload)
            return
        try:
            self.route.start(payload, self.pose)
        except Exception as exc:
            self.publish_metric("sensor_route_decision_rejected", result="parse_error", error=repr(exc))
            return
        self.task_type = "route_choice"
        self.publish_metric("sensor_route_decision_accepted", result="accepted", decision=payload)

    def tick(self) -> None:
        if not self.enabled:
            return
        if self.task_type == "route_choice" or self.route.phase != "idle":
            primitive = self.route.command(
                self.pose,
                self.free_space,
                self.track.summary(timestamp=node_ros_now_sec(self)),
            )
        elif self.task_type == "semantic_target":
            track = self.track.summary(timestamp=node_ros_now_sec(self))
            primitive = semantic_track_primitive(
                track,
                max_linear_mps=float(self.mode.get("sensor_target_max_linear_mps", 0.20)),
                max_yaw_radps=float(self.mode.get("sensor_target_max_yaw_radps", 0.30)),
                stop_distance_m=float(self.mode.get("sensor_target_stop_distance_m", 2.0)),
            )
            if primitive.get("phase") == "await_step_stop" and not self.threshold_event_sent:
                self.threshold_event_sent = True
                self.publish_json(
                    self.mission_event_pub,
                    {
                        "type": "candidate_goal_reached",
                        "reason": "confirmed_sensor_track_inside_stop_threshold",
                        "episode_id": self.active_episode_id,
                        "mission_id": self.active_episode_id,
                        "target_id": track.get("target_id"),
                        "target": str(self.mode.get("perception_target_query") or track.get("target_id") or ""),
                        "distance_m": track.get("distance_m"),
                        "distance_to_target_m": track.get("distance_m"),
                        "allow_distance_context": True,
                        "distance_basis": "actual_mask_depth",
                        "track": track,
                        "source": "sensor_only_semantic_planner",
                    },
                )
        else:
            return
        primitive = dict(primitive)
        primitive.update(
            {
                "episode_id": self.active_episode_id,
                "ttl_sec": float(self.mode.get("sensor_primitive_ttl_sec", 0.45)),
                "controller_input_policy": "actual_sensor_only",
            }
        )
        forbidden = payload_oracle_fields(primitive)
        if forbidden:
            self.publish_metric("sensor_primitive_rejected", result="oracle_leakage", forbidden_fields=forbidden)
            return
        signature = json.dumps(primitive, sort_keys=True, ensure_ascii=True)
        now = time.monotonic()
        if signature == self.last_primitive_signature and now - self.last_primitive_publish < 0.20:
            return
        self.last_primitive_signature = signature
        self.last_primitive_publish = now
        self.publish_json(self.primitive_pub, attach_timebase(primitive, node=self, episode_id=self.active_episode_id))
        self.publish_metric("sensor_planner_primitive", result="published", primitive=primitive)

    def publish_metric(self, event_type: str, **fields: Any) -> None:
        self.publish_json(
            self.metric_pub,
            attach_timebase(
                make_metric(event_type, model="sensor_only_planner", **fields),
                node=self,
                episode_id=self.active_episode_id,
            ),
        )

    @staticmethod
    def publish_json(pub, payload: dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)


def _json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SensorOnlyPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
