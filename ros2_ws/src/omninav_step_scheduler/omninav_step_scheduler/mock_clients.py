from __future__ import annotations

import json
import random
from typing import Any

from .schemas import attach_timebase, clock_domain_from_node, make_metric, new_id, node_ros_now_sec, now

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover
    rclpy = None
    Node = object
    String = None


class MockStepClientNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run MockStepClientNode")
        super().__init__("mock_step_client")
        self.declare_parameter("response_delay_sec", 0.1)
        self.response_pub = self.create_publisher(String, "/step/response_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/step/request_json", self.on_request, 10)

    def on_request(self, msg):
        req = _safe_json(msg.data)
        delay = float(self.get_parameter("response_delay_sec").value)
        timer = None

        def fire(req=req):
            if timer is not None:
                timer.cancel()
            self.publish_response(req)

        timer = self.create_timer(delay, fire)

    def publish_response(self, req):
        ts = node_ros_now_sec(self)
        mission, subgoal, success_condition = _step_mission_fields(req)
        response = {
            "request_id": req.get("request_id", new_id("step")),
            "timestamp_request": req.get("timestamp_request", ts),
            "timestamp_response": ts,
            "multimodal": bool(req.get("multimodal", False)),
            "pose_at_request": req.get("pose_at_request", [0.0, 0.0, 0.0]),
            "navila_or_omninav_instruction": mission,
            "subgoal": subgoal,
            "success_condition": success_condition,
            "constraints": {
                "max_speed_mps": 0.2,
                "avoid_people": True,
                "stop_if_uncertain": True,
                "forbidden_zones": [],
            },
            "replan_triggers": ["low_confidence", "blocked_path", "target_not_visible"],
            "recommended_pending_mode": req.get("pending_mode", "stop") if req.get("pending_mode") in {"stop", "move_slow", "safe_scan"} else "stop",
            "confidence": 0.82,
            "raw_json": {"mock": True},
        }
        response = attach_timebase(
            response,
            node=self,
            episode_id=str(req.get("episode_id") or ""),
            mission_id=str(req.get("mission_id") or ""),
            request_id=str(response["request_id"]),
            clock_domain=str(req.get("clock_domain") or clock_domain_from_node(self)),
            source_stamp=ts,
            created_ros_time=ts,
        )
        self.publish_json(self.response_pub, response)
        self.publish_json(self.metric_pub, make_metric("step_response", model="step", request_id=response["request_id"], result="accepted"))

    @staticmethod
    def publish_json(pub, payload):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)


class MockOmniNavClientNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run MockOmniNavClientNode")
        super().__init__("mock_omninav_client")
        self.declare_parameter("response_delay_sec", 0.02)
        self.action_pub = self.create_publisher(String, "/omninav/action_candidate_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/omninav/request_json", self.on_request, 10)

    def on_request(self, msg):
        req = _safe_json(msg.data)
        delay = float(self.get_parameter("response_delay_sec").value)
        timer = None

        def fire(req=req):
            if timer is not None:
                timer.cancel()
            self.publish_action(req)

        timer = self.create_timer(delay, fire)

    def publish_action(self, req):
        ts_req = float(req.get("timestamp", req.get("timestamp_request", node_ros_now_sec(self))))
        ts = node_ros_now_sec(self)
        action = {
            "request_id": req.get("request_id", new_id("omni")),
            "timestamp_request": ts_req,
            "timestamp_response": ts,
            "frame_timestamp": ts_req,
            "pose_at_snapshot": req.get("pose", [0.0, 0.0, 0.0]),
            "primitive": "move_forward",
            "distance_m": 0.35,
            "yaw_deg": 0.0,
            "confidence": 0.72 + random.random() * 0.1,
            "raw_text": "FORWARD_35",
            "source": "omninav",
            "ttl_sec": 0.5,
        }
        action = attach_timebase(
            action,
            node=self,
            episode_id=str(req.get("episode_id") or ""),
            mission_id=str(req.get("mission_id") or ""),
            request_id=str(action["request_id"]),
            clock_domain=str(req.get("clock_domain") or clock_domain_from_node(self)),
            source_stamp=ts,
            created_ros_time=ts,
        )
        self.publish_json(self.action_pub, action)
        self.publish_json(self.metric_pub, make_metric("omninav_response", model="omninav", request_id=action["request_id"], result="accepted"))

    @staticmethod
    def publish_json(pub, payload):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)


class MockSafetyStatusNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run MockSafetyStatusNode")
        super().__init__("mock_safety_status")
        self.pub = self.create_publisher(String, "/safety/local_status_json", 10)
        self.state_pub = self.create_publisher(String, "/robot_state_json", 10)
        self.create_timer(0.1, self.tick)

    def tick(self):
        safety = {
            "timestamp": now(),
            "local_costmap_clear": True,
            "obstacle_distance_m": 2.5,
            "human_distance_m": None,
            "on_slope_or_stairs": False,
            "near_doorway": False,
            "near_intersection": False,
            "dynamic_obstacle": False,
            "estop": False,
            "deadman": False,
            "robot_fallen_or_unstable": False,
            "battery_ok": True,
        }
        state = {"timestamp": now(), "pose": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}
        self.publish_json(self.pub, safety)
        self.publish_json(self.state_pub, state)

    @staticmethod
    def publish_json(pub, payload):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)


def _safe_json(raw: str) -> dict:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def _step_mission_fields(req: dict[str, Any]) -> tuple[str, str, str]:
    prompt = req.get("prompt") if isinstance(req.get("prompt"), dict) else {}
    messages = prompt.get("messages") if isinstance(prompt.get("messages"), list) else []
    mission = str(req.get("instruction") or req.get("mission") or "").strip()
    subgoal = str(req.get("current_subgoal") or req.get("subgoal") or "").strip()
    success = str(req.get("success_condition") or "").strip()

    for message in reversed(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        event = payload.get("Event reason") if isinstance(payload.get("Event reason"), dict) else {}
        mission = mission or str(payload.get("Mission") or event.get("instruction") or event.get("mission") or "").strip()
        subgoal = subgoal or str(payload.get("Current subgoal") or event.get("current_subgoal") or event.get("subgoal") or "").strip()
        success = success or str(event.get("success_condition") or "").strip()
        break

    if not mission:
        reason = str(req.get("reason") or "mission target")
        mission = f"Continue toward the mission target; current Step trigger is {reason}."
    if not subgoal:
        subgoal = mission
    if not success:
        success = "mission target reached"
    return mission, subgoal, success


def _spin(node_cls, args=None):
    rclpy.init(args=args)
    node = node_cls()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if exc.__class__.__name__ != "ExternalShutdownException":
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main_step(args=None):
    _spin(MockStepClientNode, args)


def main_omninav(args=None):
    _spin(MockOmniNavClientNode, args)


def main_safety(args=None):
    _spin(MockSafetyStatusNode, args)
