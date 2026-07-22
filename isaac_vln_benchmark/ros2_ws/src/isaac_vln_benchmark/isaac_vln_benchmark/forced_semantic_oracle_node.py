from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .semantic_navigation_benchmark import load_task_set, oracle_plan_payload

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover
    rclpy = None
    Node = object
    String = None


class ForcedSemanticOracleNode(Node):
    """Publishes semantic subgoals through the same stale-gated Step role topic."""

    def __init__(self) -> None:
        if rclpy is None:
            raise RuntimeError("rclpy is required to run ForcedSemanticOracleNode")
        super().__init__("forced_semantic_oracle")
        self.declare_parameter("task_file", "")
        task_file = str(self.get_parameter("task_file").value)
        if not task_file:
            raise RuntimeError("task_file is required")
        task_set = load_task_set(Path(task_file), require_full=False)
        self.tasks = {str(task["task_id"]): task for task in task_set["tasks"]}
        self.enabled = False
        self.episode_id = ""
        self.task_id = ""
        self.plan: list[dict[str, Any]] = []
        self.next_index = 0
        self.decision_pub = self.create_publisher(String, "/step/semantic_subgoal_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/benchmark/mode_json", self.on_mode, 10)
        self.create_subscription(String, "/scheduler/step_trigger_json", self.on_trigger, 10)

    def on_mode(self, msg: Any) -> None:
        payload = _safe_json(msg.data)
        mode = str(payload.get("mode") or payload.get("benchmark_mode") or "")
        mode_cfg = payload.get("mode_config") if isinstance(payload.get("mode_config"), dict) else {}
        self.enabled = mode == "forced_semantic_oracle" or bool(mode_cfg.get("forced_semantic_oracle", False))
        self.episode_id = str(payload.get("episode_id") or "")
        self.task_id = str(payload.get("task_id") or mode_cfg.get("task_id") or "")
        self.next_index = 0
        task = self.tasks.get(self.task_id)
        self.plan = oracle_plan_payload(task, episode_id=self.episode_id) if task else []
        self.publish_metric(
            "forced_semantic_oracle_configured",
            result="ready" if self.enabled and self.plan else "disabled_or_missing_task",
            task_id=self.task_id,
            episode_id=self.episode_id,
            subgoal_count=len(self.plan),
        )

    def on_trigger(self, msg: Any) -> None:
        if not self.enabled:
            return
        event = _safe_json(msg.data)
        event_type = str(event.get("type") or "")
        if event_type not in {"semantic_plan_requested", "semantic_subgoal_completed"}:
            return
        if self.next_index >= len(self.plan):
            self.publish_metric("forced_semantic_oracle_plan_exhausted", result="complete", task_id=self.task_id)
            return
        payload = self._stamp(self.plan[self.next_index])
        self.next_index += 1
        self.decision_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.publish_metric(
            "forced_semantic_oracle_subgoal",
            result="published",
            task_id=self.task_id,
            subgoal_index=payload["subgoal_index"],
            subgoal_type=payload["subgoal_type"],
            requires_stale_gate=True,
            publishes_motion=False,
        )

    def _stamp(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        stamp = float(self.get_clock().now().nanoseconds) * 1e-9
        wall = time.time()
        request_id = f"forced_semantic_{uuid.uuid4().hex[:12]}"
        clock_domain = "ros_sim" if bool(self.get_parameter("use_sim_time").value) else "ros_system"
        timebase = {
            "episode_id": self.episode_id,
            "mission_id": "",
            "request_id": request_id,
            "clock_domain": clock_domain,
            "ros_now_sec": stamp,
            "wall_now_sec": wall,
            "clock_msg_sec": stamp,
            "header_stamp_sec": stamp,
            "source_stamp_sec": stamp,
            "created_ros_time_sec": stamp,
            "created_wall_time_sec": wall,
        }
        result.update(timebase)
        result.update({"request_id": request_id, "timestamp_response": stamp, "timebase": timebase})
        return result

    def publish_metric(self, event_type: str, **kwargs: Any) -> None:
        payload = {"event": event_type, "timestamp": time.time(), **kwargs}
        self.metric_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ForcedSemanticOracleNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
