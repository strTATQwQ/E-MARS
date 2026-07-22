from __future__ import annotations

import json
from typing import Any

from .runtime_mode import mode_payload_from_json
from .schemas import (
    SchemaError,
    attach_timebase,
    clock_domain_from_node,
    deep_get,
    load_yaml_file,
    make_metric,
    new_id,
    node_ros_now_sec,
    now,
)
from .stale_gate import evaluate_role_decision

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover - exercised on ROS hosts
    rclpy = None
    Node = object
    String = None


RECOVERY_ACTIONS = {"scan", "backtrack", "ask", "stop"}


def recovery_to_primitive(
    payload: dict[str, Any], config: dict[str, Any] | None = None
) -> tuple[dict[str, Any], float]:
    """Map a semantic recovery enum to one bounded normal-chain primitive."""

    cfg = config or {}
    recovery = str(payload.get("recovery") or "").strip().lower()
    if recovery not in RECOVERY_ACTIONS:
        raise SchemaError(f"unsupported semantic recovery: {recovery}")
    source_request_id = str(payload.get("request_id") or new_id("semantic_recovery"))
    base = {
        "request_id": f"{source_request_id}_primitive",
        "source_request_id": source_request_id,
        "source": "semantic_recovery_bridge",
        "recovery": recovery,
        "reason": str(payload.get("reason") or "semantic_recovery"),
    }
    if recovery == "scan":
        duration = _bounded_duration(deep_get(cfg, "semantic_recovery.scan_duration_sec", 0.8))
        return (
            base
            | {
                "primitive": "look_around",
                "angular_z_radps": float(deep_get(cfg, "semantic_recovery.scan_yaw_rate_radps", 0.20)),
                "ttl_sec": duration,
            },
            duration,
        )
    if recovery == "backtrack":
        duration = _bounded_duration(deep_get(cfg, "semantic_recovery.backtrack_duration_sec", 1.0))
        return (
            base
            | {
                "primitive": "back_off",
                "distance_m": float(deep_get(cfg, "semantic_recovery.backtrack_distance_m", 0.20)),
                "ttl_sec": duration,
            },
            duration,
        )
    duration = _bounded_duration(deep_get(cfg, "semantic_recovery.stop_hold_sec", 0.15))
    return base | {"primitive": "stop", "ttl_sec": duration}, duration


class SemanticRecoveryBridgeNode(Node):
    """Runs bounded recovery through primitive_executor and safe_mux."""

    def __init__(self) -> None:
        if rclpy is None:
            raise RuntimeError("rclpy is required to run SemanticRecoveryBridgeNode")
        super().__init__("semantic_recovery_bridge")
        self.declare_parameter("config_file", "")
        self.config = load_yaml_file(self.get_parameter("config_file").value)
        self.enabled = bool(deep_get(self.config, "semantic_recovery.enabled", True))
        self.active_episode_id = ""
        self.retired_episode_ids: set[str] = set()
        self.seen_request_ids: set[str] = set()
        self.active: dict[str, Any] | None = None

        self.primitive_pub = self.create_publisher(String, "/primitive/command_json", 10)
        self.override_pub = self.create_publisher(String, "/oracle/control_override_json", 10)
        self.event_pub = self.create_publisher(String, "/mission/event_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/benchmark/mode_json", self.on_mode, 10)
        self.create_subscription(String, "/semantic_executive/recovery_json", self.on_recovery, 10)
        self.create_timer(0.05, self.tick)

    def on_mode(self, msg: Any) -> None:
        payload = mode_payload_from_json(msg.data)
        episode_id = str(payload.get("episode_id") or "")
        if not episode_id or episode_id == self.active_episode_id:
            return
        if self.active_episode_id:
            self.retired_episode_ids.add(self.active_episode_id)
        self._cancel_active("episode_reset")
        self.active_episode_id = episode_id
        self.seen_request_ids.clear()
        self.publish_metric("episode_reset", result="cleared", reset_scope="semantic_recovery_bridge")

    def on_recovery(self, msg: Any) -> None:
        if not self.enabled:
            return
        payload = _safe_json(msg.data)
        request_id = str(payload.get("request_id") or "")
        episode_id = str(payload.get("episode_id") or "")
        stale = evaluate_role_decision(
            payload,
            current_time=node_ros_now_sec(self),
            config=deep_get(self.config, "stale_gate", {}) or {},
            current_episode_id=self.active_episode_id,
            current_clock_domain=clock_domain_from_node(self),
            strict_timebase=True,
            old_response_after_reset=bool(episode_id and episode_id in self.retired_episode_ids),
        )
        if not stale.valid:
            self.publish_metric("semantic_recovery_discarded", result="discarded", **stale.to_metric_fields())
            return
        if request_id and request_id in self.seen_request_ids:
            self.publish_metric(
                "semantic_recovery_discarded",
                result="discarded",
                attribution="duplicate_response",
                request_id=request_id,
            )
            return
        try:
            primitive, duration = recovery_to_primitive(payload, self.config)
        except Exception as exc:
            self.publish_metric("semantic_recovery_parse_error", result="parse_error", error=repr(exc))
            return
        if request_id:
            self.seen_request_ids.add(request_id)
        self._cancel_active("superseded")
        stamped_primitive = self._stamp(primitive, request_id=str(primitive["request_id"]))
        self._publish_override(True, str(payload.get("recovery") or ""), request_id)
        self.publish_json(self.primitive_pub, stamped_primitive)
        self.active = {
            "deadline": now() + duration,
            "recovery": str(payload.get("recovery") or ""),
            "request_id": request_id,
        }
        self.publish_metric(
            "semantic_recovery_started",
            result="primitive_published",
            recovery=self.active["recovery"],
            request_id=request_id,
            primitive=stamped_primitive,
        )

    def tick(self) -> None:
        if self.active is None or now() < float(self.active["deadline"]):
            return
        active = dict(self.active)
        self.active = None
        stop = self._stamp(
            {
                "primitive": "stop",
                "ttl_sec": 0.15,
                "source": "semantic_recovery_bridge",
                "reason": "semantic_recovery_completed",
                "recovery": active["recovery"],
            },
            request_id=new_id("semantic_recovery_stop"),
        )
        self.publish_json(self.primitive_pub, stop)
        self._publish_override(False, str(active["recovery"]), str(active["request_id"]))
        event = self._stamp(
            {
                "type": "semantic_recovery_completed",
                "source": "semantic_recovery_bridge",
                "recovery": active["recovery"],
                "source_request_id": active["request_id"],
            },
            request_id=new_id("semantic_recovery_event"),
        )
        self.publish_json(self.event_pub, event)
        self.publish_metric(
            "semantic_recovery_completed",
            result="completed",
            recovery=active["recovery"],
            request_id=active["request_id"],
        )

    def _cancel_active(self, reason: str) -> None:
        if self.active is None:
            return
        active = dict(self.active)
        self.active = None
        stop = self._stamp(
            {
                "primitive": "stop",
                "ttl_sec": 0.15,
                "source": "semantic_recovery_bridge",
                "reason": reason,
            },
            request_id=new_id("semantic_recovery_cancel"),
        )
        self.publish_json(self.primitive_pub, stop)
        self._publish_override(False, str(active.get("recovery") or ""), str(active.get("request_id") or ""))

    def _publish_override(self, active: bool, recovery: str, request_id: str) -> None:
        payload = self._stamp(
            {
                "active": bool(active),
                "episode_id": self.active_episode_id,
                "source": "semantic_recovery_bridge",
                "phase": f"semantic_recovery_{recovery}",
                "source_request_id": request_id,
            },
            request_id=new_id("semantic_recovery_override"),
        )
        self.publish_json(self.override_pub, payload)

    def _stamp(self, payload: dict[str, Any], *, request_id: str) -> dict[str, Any]:
        return attach_timebase(
            payload,
            node=self,
            episode_id=self.active_episode_id,
            request_id=request_id,
            source_stamp=node_ros_now_sec(self),
            created_ros_time=node_ros_now_sec(self),
        )

    def publish_metric(self, event_type: str, **kwargs: Any) -> None:
        self.publish_json(
            self.metric_pub,
            attach_timebase(
                make_metric(event_type, model="semantic_recovery_bridge", **kwargs),
                node=self,
                episode_id=self.active_episode_id,
            ),
        )

    @staticmethod
    def publish_json(pub: Any, payload: dict[str, Any]) -> None:
        pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def _bounded_duration(value: Any) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        duration = 0.15
    return max(0.05, min(1.5, duration))


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticRecoveryBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
