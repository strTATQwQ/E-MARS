from __future__ import annotations

import json

from .metrics import DelayInjectorCore


def delay_audit_payload(record: dict, profile: str) -> dict:
    payload = {k: v for k, v in record.items() if k != "message"}
    payload["profile"] = str(profile)
    raw = record.get("message")
    try:
        message = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        message = None
    if isinstance(message, dict):
        for key in ("episode_id", "mission_id", "request_id", "role", "route_choice", "stop"):
            if key in message:
                payload[key] = message[key]
    return payload


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class DelayInjector(Node):
        def __init__(self) -> None:
            super().__init__("isaac_vln_delay_injector")
            self.declare_parameter("input_topic", "/step/response_json")
            self.declare_parameter("output_topic", "/delayed/step/response_json")
            self.declare_parameter("delay_sec", 0.0)
            self.declare_parameter("drop_rate", 0.0)
            self.declare_parameter("seed", 42)
            self.declare_parameter("profile", "unnamed")
            self.core = DelayInjectorCore(
                float(self.get_parameter("delay_sec").value),
                float(self.get_parameter("drop_rate").value),
                seed=int(self.get_parameter("seed").value),
            )
            self.pub = self.create_publisher(String, str(self.get_parameter("output_topic").value), 10)
            self.audit_pub = self.create_publisher(String, "/benchmark/delay_audit_json", 10)
            self.sub = self.create_subscription(String, str(self.get_parameter("input_topic").value), self.on_msg, 10)
            self.timer = self.create_timer(0.02, self.tick)

        def now(self) -> float:
            return self.get_clock().now().nanoseconds / 1e9

        def on_msg(self, msg: String) -> None:
            accepted = self.core.enqueue(str(self.get_parameter("input_topic").value), msg.data, self.now())
            if not accepted:
                self.publish_audit(self.core.records[-1])

        def tick(self) -> None:
            for record in self.core.drain(self.now()):
                message = self.with_delay_metadata(record["message"], float(record["delay_sec"]))
                self.pub.publish(String(data=message))
                self.publish_audit(record)

        def with_delay_metadata(self, raw: str, delay_sec: float) -> str:
            try:
                payload = json.loads(raw)
            except Exception:
                return raw
            if not isinstance(payload, dict):
                return raw
            payload["queue_delay_sec"] = delay_sec
            payload["transport_delay_sec"] = delay_sec
            payload["transport_profile"] = str(self.get_parameter("profile").value)
            return json.dumps(payload, ensure_ascii=False)

        def publish_audit(self, record: dict) -> None:
            payload = delay_audit_payload(record, str(self.get_parameter("profile").value))
            self.audit_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    rclpy.init()
    node = DelayInjector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
