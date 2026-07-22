from __future__ import annotations

import csv
import json
from pathlib import Path

from .schemas import deep_get, load_yaml_file, now

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover
    rclpy = None
    Node = object
    String = None


class MetricsLoggerNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run MetricsLoggerNode")
        super().__init__("metrics_logger")
        self.declare_parameter("config_file", "")
        self.config = load_yaml_file(self.get_parameter("config_file").value)
        self.output_dir = Path(str(deep_get(self.config, "logging.output_dir", "runs/latest_scheduler")))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "events.jsonl"
        self.csv_path = self.output_dir / "summary.csv"
        self.events: list[dict] = []
        self.event_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        for topic in [
            "/scheduler/state",
            "/scheduler/step_trigger_json",
            "/scheduler/step_pending_mode",
            "/step/request_json",
            "/step/response_json",
            "/omninav/request_json",
            "/omninav/action_candidate_json",
            "/primitive/command_json",
            "/mission/event_json",
            "/episode/reset_lifecycle_json",
        ]:
            self.create_subscription(String, topic, lambda msg, t=topic: self.log_topic(t, msg.data), 100)
        self.create_subscription(String, "/metrics/event_jsonl", self.on_metric, 100)
        self.create_timer(2.0, self.write_summary)

    def on_metric(self, msg):
        payload = _safe_json(msg.data)
        self.write_event(payload)

    def log_topic(self, topic: str, data: str):
        self.write_event({"timestamp": now(), "event_type": "topic", "topic": topic, "payload": _safe_json(data)})

    def write_event(self, payload: dict):
        self.events.append(payload)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_summary(self):
        rows = [
            ("omninav_call_count", sum(1 for e in self.events if e.get("event_type") in {"omninav_request", "omninav_action"})),
            ("step_call_count", sum(1 for e in self.events if e.get("event_type") == "step_request")),
            ("stop_and_think_count", sum(1 for e in self.events if e.get("pending_mode") == "stop")),
            ("move_while_thinking_count", sum(1 for e in self.events if e.get("pending_mode") == "move_slow")),
            ("stale_result_count", sum(1 for e in self.events if e.get("stale") or e.get("event_type") in {"step_response_stale", "omninav_stale", "internnav_stale"})),
            ("timebase_error_count", sum(1 for e in self.events if e.get("attribution") == "timebase_error")),
            ("episode_mismatch_count", sum(1 for e in self.events if e.get("attribution") == "episode_mismatch")),
            ("missing_timestamp_count", sum(1 for e in self.events if e.get("attribution") == "missing_timestamp")),
            ("safety_stop_count", sum(1 for e in self.events if e.get("result") in {"safety_stop", "stopped"})),
        ]
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            writer.writerows(rows)


def _safe_json(raw: str) -> dict:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def main(args=None):
    rclpy.init(args=args)
    node = MetricsLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.write_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
