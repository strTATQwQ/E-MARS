from __future__ import annotations

import json
import time


def reset_episode_id(previous: str, request_id: str) -> str:
    suffix = str(request_id or "decision")[-12:].replace(" ", "_")
    return f"{previous or 'episode'}_transport_reset_{suffix}"


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class ResetOnDecision(Node):
        def __init__(self) -> None:
            super().__init__("reset_on_decision")
            self.declare_parameter("delay_sec", 0.10)
            self.latest_mode: dict = {}
            self.pending: dict | None = None
            self.reset_sent = False
            self.mode_pub = self.create_publisher(String, "/benchmark/mode_json", 10)
            self.reset_pub = self.create_publisher(String, "/isaac/reset_episode", 10)
            self.audit_pub = self.create_publisher(String, "/benchmark/reset_stress_json", 10)
            self.create_subscription(String, "/benchmark/mode_json", self.on_mode, 10)
            for topic in (
                "/stress/raw/step/response_json",
                "/stress/raw/step/route_choice_json",
                "/stress/raw/step/semantic_stop_json",
            ):
                self.create_subscription(String, topic, self.on_raw_decision, 10)
            self.create_timer(0.02, self.tick)

        def on_mode(self, msg: String) -> None:
            try:
                value = json.loads(msg.data)
            except Exception:
                return
            if isinstance(value, dict) and "_transport_reset_" not in str(value.get("episode_id") or ""):
                self.latest_mode = value

        def on_raw_decision(self, msg: String) -> None:
            if self.pending is not None or self.reset_sent:
                return
            try:
                decision = json.loads(msg.data)
            except Exception:
                decision = {}
            if not isinstance(decision, dict):
                decision = {}
            self.pending = {
                "decision": decision,
                "deadline": time.monotonic() + float(self.get_parameter("delay_sec").value),
            }

        def tick(self) -> None:
            if self.pending is None or self.reset_sent or time.monotonic() < float(self.pending["deadline"]):
                return
            decision = dict(self.pending["decision"])
            old_episode = str(decision.get("episode_id") or self.latest_mode.get("episode_id") or "")
            request_id = str(decision.get("request_id") or "")
            new_episode = reset_episode_id(old_episode, request_id)
            mode = dict(self.latest_mode)
            mode["episode_id"] = new_episode
            reset = {"episode_id": new_episode, "reason": "decision_transport_reset_stress"}
            self.mode_pub.publish(String(data=json.dumps(mode, ensure_ascii=False)))
            self.reset_pub.publish(String(data=json.dumps(reset, ensure_ascii=False)))
            audit = {
                "event_type": "reset_injected",
                "old_episode_id": old_episode,
                "new_episode_id": new_episode,
                "request_id": request_id,
                "delay_sec": float(self.get_parameter("delay_sec").value),
            }
            self.audit_pub.publish(String(data=json.dumps(audit, ensure_ascii=False)))
            self.reset_sent = True

    rclpy.init()
    node = ResetOnDecision()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
