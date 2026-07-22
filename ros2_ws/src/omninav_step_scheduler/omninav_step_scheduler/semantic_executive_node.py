from __future__ import annotations

import json
from typing import Any

from .schemas import (
    attach_timebase,
    clock_domain_from_node,
    deep_get,
    load_yaml_file,
    make_metric,
    new_id,
    node_ros_now_sec,
)
from .semantic_executive import SemanticExecutiveCore, parse_semantic_subgoal_json
from .stale_gate import evaluate_role_decision

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover
    rclpy = None
    Node = object
    String = None


class SemanticExecutiveNode(Node):
    """Accepts stale-gated semantic decisions and emits motion-free OmniNav goals."""

    def __init__(self) -> None:
        if rclpy is None:
            raise RuntimeError("rclpy is required to run SemanticExecutiveNode")
        super().__init__("semantic_executive")
        self.declare_parameter("config_file", "")
        self.config = load_yaml_file(self.get_parameter("config_file").value)
        self.enabled = bool(deep_get(self.config, "semantic_executive.enabled", False))
        self.strict_timebase = bool(deep_get(self.config, "semantic_executive.strict_timebase", True))
        self.core = SemanticExecutiveCore(
            min_confidence=float(deep_get(self.config, "semantic_executive.min_confidence", 0.55))
        )
        self.active_episode_id = ""
        self.last_source_stamp: float | None = None
        self.next_subgoal_index = 0

        self.goal_pub = self.create_publisher(String, "/omninav/semantic_goal_json", 10)
        self.state_pub = self.create_publisher(String, "/semantic_executive/state_json", 10)
        self.accepted_pub = self.create_publisher(String, "/semantic_executive/accepted_subgoal_json", 10)
        self.trigger_pub = self.create_publisher(String, "/scheduler/step_trigger_json", 10)
        self.recovery_pub = self.create_publisher(String, "/semantic_executive/recovery_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)

        self.create_subscription(String, "/benchmark/mode_json", self.on_mode, 10)
        self.create_subscription(String, "/step/semantic_subgoal_json", self.on_subgoal, 10)
        self.create_subscription(String, "/mission/event_json", self.on_event, 10)
        self.create_subscription(String, "/user_instruction", self.on_instruction, 10)

    def on_mode(self, msg: Any) -> None:
        payload = _safe_json(msg.data)
        episode_id = str(payload.get("episode_id") or "")
        if episode_id and episode_id != self.active_episode_id:
            self.active_episode_id = episode_id
            self.last_source_stamp = None
            self.next_subgoal_index = 0
            self.core.reset(episode_id)
            self.publish_state(self.core._result("NEEDS_PLAN", reason="episode_reset"))

    def on_instruction(self, msg: Any) -> None:
        if not self.enabled:
            return
        payload = _safe_json(msg.data)
        instruction = str(payload.get("instruction") or msg.data or "").strip()
        self.publish_trigger(
            {
                "type": "semantic_plan_requested",
                "episode_id": self.active_episode_id,
                "instruction": instruction,
                "next_subgoal_index": self.next_subgoal_index,
                "source": "semantic_executive",
                "multimodal": True,
            }
        )

    def on_subgoal(self, msg: Any) -> None:
        if not self.enabled:
            return
        payload = _safe_json(msg.data)
        source_stamp = _optional_float((payload.get("timebase") or {}).get("source_stamp_sec"))
        old_response = bool(
            source_stamp is not None
            and self.last_source_stamp is not None
            and source_stamp <= self.last_source_stamp
        )
        stale = evaluate_role_decision(
            payload,
            current_time=node_ros_now_sec(self),
            config=deep_get(self.config, "stale_gate", {}) or {},
            current_episode_id=self.active_episode_id,
            current_clock_domain=clock_domain_from_node(self),
            strict_timebase=self.strict_timebase,
            old_response_after_reset=old_response,
        )
        if not stale.valid:
            self.publish_metric("semantic_subgoal_discarded", **stale.to_metric_fields())
            self.publish_recovery("stop", stale.attribution)
            return
        try:
            subgoal = parse_semantic_subgoal_json(payload)
        except Exception as exc:
            self.publish_metric("semantic_subgoal_schema_error", error=repr(exc), result="discarded")
            self.publish_recovery("stop", "schema_error")
            return
        result = self.core.accept(subgoal)
        self.last_source_stamp = source_stamp
        if result["status"] == "RUNNING":
            explicit_index = subgoal.metadata.get("subgoal_index")
            expected_index = int(getattr(self, "next_subgoal_index", 0))
            subgoal_index = expected_index if explicit_index is None else int(explicit_index)
            if subgoal_index != expected_index:
                self.publish_metric(
                    "semantic_subgoal_sequence_error",
                    result="discarded",
                    expected_index=expected_index,
                    received_index=subgoal_index,
                )
                self.clear_omninav_goal("subgoal_sequence_error")
                self.publish_recovery("stop", "subgoal_sequence_error")
                return
            self.publish_json(self.accepted_pub, subgoal.to_dict() | {"subgoal_index": subgoal_index})
            self.publish_json(
                self.goal_pub,
                dict(result["semantic_goal"])
                | {
                    "subgoal_index": subgoal_index,
                    "episode_id": self.active_episode_id,
                    "source": "semantic_executive",
                    "stale_gate": "accepted",
                },
            )
        elif result["status"] == "RECOVERY":
            self.clear_omninav_goal(str(result.get("reason") or "recovery"))
            self.publish_recovery(str(result.get("recovery") or "stop"), str(result.get("reason") or "rejected"))
        self.publish_state(result)
        self.publish_metric(
            "semantic_subgoal_processed",
            result=result["status"].lower(),
            subgoal_type=subgoal.subgoal_type,
            confidence=subgoal.confidence,
            stale_gate="accepted",
            publishes_motion=False,
        )

    def on_event(self, msg: Any) -> None:
        if not self.enabled:
            return
        event = _safe_json(msg.data)
        event_type = str(event.get("type") or event.get("reason") or "")
        event_episode = str(event.get("episode_id") or "")
        if event_episode and self.active_episode_id and event_episode != self.active_episode_id:
            self.publish_metric("semantic_event_episode_mismatch", result="discarded", event=event)
            return
        if self.core.active is None:
            recovery_completed = event_type == "semantic_recovery_completed"
            reason = "recovery_retry_required" if recovery_completed else "no_active_subgoal"
            next_subgoal_index = int(getattr(self, "next_subgoal_index", 0))
            self.publish_state(self.core._result("NEEDS_PLAN", reason=reason))
            self.publish_metric(
                "semantic_event_without_active_subgoal",
                result="retry_triggered" if recovery_completed else "observed_without_trigger",
                observed_event_type=event_type or "unknown",
                next_subgoal_index=next_subgoal_index,
            )
            if recovery_completed:
                self.publish_trigger(
                    {
                        "type": "semantic_plan_requested",
                        "episode_id": self.active_episode_id,
                        "next_subgoal_index": next_subgoal_index,
                        "reason": "recovery_retry_required",
                        "recovery": str(event.get("recovery") or ""),
                        "source": "semantic_executive",
                        "multimodal": True,
                    }
                )
            return
        result = self.core.handle_event(event)
        self.publish_state(result)
        if result["status"] == "RECOVERY":
            self.clear_omninav_goal(str(result.get("reason") or "recovery"))
            self.publish_recovery(str(result.get("recovery") or "stop"), str(result.get("reason") or "recovery"))
        elif result["status"] == "NEEDS_PLAN":
            if result.get("completed_subgoal"):
                completed_index = int(getattr(self, "next_subgoal_index", 0))
                self.next_subgoal_index = completed_index + 1
                self.hold_omninav_goal(str(result.get("reason") or "subgoal_completed"))
            else:
                completed_index = None
                self.clear_omninav_goal(str(result.get("reason") or "needs_plan"))
            self.publish_trigger(
                {
                    "type": "semantic_subgoal_completed"
                    if result.get("completed_subgoal")
                    else "semantic_plan_requested",
                    "episode_id": self.active_episode_id,
                    "completed_subgoal": result.get("completed_subgoal"),
                    "completed_subgoal_index": completed_index,
                    "next_subgoal_index": self.next_subgoal_index,
                    "source": "semantic_executive",
                    "multimodal": True,
                }
            )
        elif result["status"] == "COMPLETE":
            completed_index = int(getattr(self, "next_subgoal_index", 0))
            self.next_subgoal_index = completed_index + 1
            self.hold_omninav_goal(str(result.get("reason") or "mission_verified"))
            self.publish_metric(
                "semantic_mission_verified",
                result="complete",
                completed_subgoal_index=completed_index,
                publishes_motion=False,
            )
        elif result["status"] == "RUNNING" and result.get("reason") == "semantic_recovery_completed":
            semantic_goal = result.get("semantic_goal")
            if isinstance(semantic_goal, dict):
                self.publish_json(
                    self.goal_pub,
                    dict(semantic_goal)
                    | {
                        "episode_id": self.active_episode_id,
                        "source": "semantic_executive_recovery_resume",
                        "stale_gate": "accepted",
                    },
                )
                self.publish_metric(
                    "semantic_subgoal_resumed",
                    result="goal_republished",
                    recovery=result.get("recovery"),
                    subgoal_type=semantic_goal.get("subgoal_type"),
                )

    def clear_omninav_goal(self, reason: str) -> None:
        self.publish_json(
            self.goal_pub,
            {
                "clear": True,
                "episode_id": self.active_episode_id,
                "source": "semantic_executive",
                "reason": reason,
            },
        )

    def hold_omninav_goal(self, reason: str) -> None:
        self.publish_json(
            self.goal_pub,
            {
                "clear": True,
                "hold": True,
                "episode_id": self.active_episode_id,
                "source": "semantic_executive",
                "reason": reason,
            },
        )

    def publish_recovery(self, recovery: str, reason: str) -> None:
        payload = {
            "episode_id": self.active_episode_id,
            "recovery": recovery if recovery in {"scan", "backtrack", "ask", "stop"} else "stop",
            "reason": reason,
            "publishes_motion": False,
            "requires_normal_execution_chain": True,
        }
        stamped = attach_timebase(
            payload,
            node=self,
            episode_id=self.active_episode_id,
            request_id=new_id("semantic_recovery"),
        )
        self.publish_json(self.recovery_pub, stamped)
        self.publish_metric("semantic_recovery_requested", **stamped)

    def publish_trigger(self, payload: dict[str, Any]) -> None:
        stamped = attach_timebase(
            payload,
            node=self,
            episode_id=self.active_episode_id,
            mission_id=str(payload.get("mission_id") or ""),
            request_id=str(payload.get("request_id") or ""),
        )
        self.publish_json(self.trigger_pub, stamped)

    def publish_state(self, payload: dict[str, Any]) -> None:
        self.publish_json(self.state_pub, payload)

    def publish_metric(self, event_type: str, **kwargs: Any) -> None:
        self.publish_json(self.metric_pub, make_metric(event_type, **kwargs))

    @staticmethod
    def publish_json(pub: Any, payload: Any) -> None:
        pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {"value": payload}
    except Exception:
        return {"instruction": str(raw)}


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticExecutiveNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
