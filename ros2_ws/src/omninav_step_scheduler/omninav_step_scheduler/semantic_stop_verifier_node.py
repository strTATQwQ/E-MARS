from __future__ import annotations

import json
from typing import Any

from .schemas import attach_timebase, load_yaml_file, make_metric, new_id, now
from .step_roles import SemanticStopGate, semantic_stop_verifier, step_role_for_event

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover - exercised on ROS hosts
    rclpy = None
    Node = object
    String = None


class SemanticStopVerifierNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run SemanticStopVerifierNode")
        super().__init__("semantic_stop_verifier")
        self.declare_parameter("config_file", "")
        self.config = load_yaml_file(self.get_parameter("config_file").value)
        gate_cfg = self.config.get("semantic_stop_gate", {}) if isinstance(self.config.get("semantic_stop_gate"), dict) else {}
        self.gate = SemanticStopGate(
            target_visible_required_frames=int(gate_cfg.get("target_visible_required_frames", 2)),
            max_distance_to_target_m=float(gate_cfg.get("max_distance_to_target_m", 2.5)),
            force_stop_if_distance_less_than_m=float(gate_cfg.get("force_stop_if_distance_less_than_m", 2.0)),
            force_stop_after_visible_sec=float(gate_cfg.get("force_stop_after_visible_sec", 1.5)),
            min_confidence=float(gate_cfg.get("min_confidence", 0.55)),
        )
        self.instruction = ""
        self.active_subgoal = ""
        self.semantic_summary: dict[str, Any] = {}
        self.latest_status: dict[str, Any] = {}
        self.active_mission_id = ""
        self.active_mode_config: dict[str, Any] = {}

        self.stop_pub = self.create_publisher(String, "/step/semantic_stop_json", 10)
        self.primitive_pub = self.create_publisher(String, "/primitive/command_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/user_instruction", self.on_instruction, 10)
        self.create_subscription(String, "/scheduler/active_subgoal_json", self.on_active_subgoal, 10)
        self.create_subscription(String, "/semantic_summary_json", self.on_semantic_summary, 10)
        self.create_subscription(String, "/isaac/episode_status_json", self.on_status, 10)
        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
        self.create_subscription(String, "/step/semantic_stop_trigger_json", self.on_trigger, 10)
        self.create_subscription(String, "/mission/event_json", self.on_trigger, 10)

    def reset_gate(self) -> None:
        self.gate = SemanticStopGate(
            target_visible_required_frames=self.gate.target_visible_required_frames,
            max_distance_to_target_m=self.gate.max_distance_to_target_m,
            force_stop_if_distance_less_than_m=self.gate.force_stop_if_distance_less_than_m,
            force_stop_after_visible_sec=self.gate.force_stop_after_visible_sec,
            min_confidence=self.gate.min_confidence,
        )

    def on_benchmark_mode(self, msg):
        payload = _safe_json(msg.data)
        mode_config = payload.get("mode_config") if isinstance(payload.get("mode_config"), dict) else {}
        self.active_mode_config = dict(mode_config)
        episode_id = str(payload.get("episode_id") or "")
        if episode_id and episode_id != self.active_mission_id:
            self.active_mission_id = episode_id
            self.reset_gate()

    def on_instruction(self, msg):
        payload = _safe_json(msg.data)
        self.instruction = str(payload.get("instruction") or msg.data or "")

    def on_active_subgoal(self, msg):
        payload = _safe_json(msg.data)
        self.active_subgoal = str(payload.get("subgoal") or payload.get("active_subgoal") or "")

    def on_semantic_summary(self, msg):
        self.semantic_summary = _safe_json(msg.data)

    def on_status(self, msg):
        self.latest_status = _safe_json(msg.data)

    def on_trigger(self, msg):
        if self.active_mode_config and not bool(self.active_mode_config.get("use_step", False)):
            return
        if bool(self.active_mode_config.get("real_step_required", False)):
            return
        event = _safe_json(msg.data)
        if step_role_for_event(event) != "semantic_stop":
            return
        mission_id = str(event.get("mission_id") or event.get("episode_id") or "")
        if mission_id and mission_id != self.active_mission_id:
            self.active_mission_id = mission_id
            self.reset_gate()
        target = str(event.get("target") or self.latest_status.get("target") or self.active_subgoal or self.instruction)
        distance = _float(
            event.get(
                "distance_to_target_m",
                event.get(
                    "distance_to_target",
                    self.latest_status.get("distance_to_target_m", self.latest_status.get("distance_to_target")),
                ),
            )
        )
        visible = bool(event.get("target_visible", self.latest_status.get("target_visible", self.semantic_summary.get("target_visible", False))))
        result = semantic_stop_verifier(
            target=target,
            target_visible=visible,
            distance_m=distance,
            active_subgoal=self.active_subgoal,
            semantic_summary=self.semantic_summary,
            max_distance_m=self.gate.max_distance_to_target_m,
            min_confidence=self.gate.min_confidence,
        )
        result = attach_timebase(
            dict(result) | {"request_id": str(event.get("request_id") or "")},
            node=self,
            episode_id=str(event.get("episode_id") or self.active_mission_id),
            mission_id=mission_id,
            request_id=str(event.get("request_id") or ""),
        )
        gate_result = self.gate.update(timestamp=now(), target_visible=result["target_visible"], distance_m=distance, verifier_json=result)
        self.publish_json(self.stop_pub, result)
        if gate_result["force_stop"] and bool(self.active_mode_config.get("semantic_stop_direct_primitive", False)):
            self.publish_json(
                self.primitive_pub,
                attach_timebase(
                {
                    "primitive": "stop",
                    "request_id": new_id("semantic_stop_gate"),
                    "episode_id": str(event.get("episode_id") or self.active_mission_id),
                    "mission_id": mission_id,
                    "reason": gate_result["reason"],
                    "ttl_sec": 0.4,
                },
                node=self,
                episode_id=str(event.get("episode_id") or self.active_mission_id),
                mission_id=mission_id,
                ),
            )
        self.publish_metric("step_semantic_stop_json", event=event, output=result, gate=gate_result, result="published")

    def publish_metric(self, event_type: str, **kwargs):
        self.publish_json(self.metric_pub, make_metric(event_type, model="step_semantic_stop_verifier", **kwargs))

    @staticmethod
    def publish_json(pub, payload):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw, "timestamp": now()}


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(args=None):
    rclpy.init(args=args)
    node = SemanticStopVerifierNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
