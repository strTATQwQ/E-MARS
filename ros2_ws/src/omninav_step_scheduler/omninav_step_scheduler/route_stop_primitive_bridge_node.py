from __future__ import annotations

import json
from typing import Any

from .runtime_mode import mode_payload_from_json
from .schemas import SchemaError, attach_timebase, clock_domain_from_node, deep_get, load_yaml_file, make_metric, new_id, node_ros_now_sec
from .stale_gate import evaluate_role_decision
from .step_roles import parse_route_choice_json, parse_semantic_stop_json

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover - exercised on ROS hosts
    rclpy = None
    Node = object
    String = None


def forced_route_choice_json(raw: str | dict[str, Any]) -> dict[str, Any]:
    payload = _safe_json(raw)
    route = str(payload.get("route_choice") or "").strip().lower()
    if route not in {"left", "right", "front", "scan", "stop"}:
        raise SchemaError(f"unsupported forced route_choice: {route}")
    result = {
        "route_choice": route,
        "confidence": _float(payload.get("confidence"), 1.0),
        "source": str(payload.get("source") or "forced_oracle"),
    }
    _copy_metadata(payload, result)
    return result


def forced_stop_json(raw: str | dict[str, Any]) -> dict[str, Any]:
    payload = _safe_json(raw)
    if "stop" not in payload:
        raise SchemaError("missing forced stop field")
    result = {
        "stop": bool(payload.get("stop")),
        "confidence": _float(payload.get("confidence"), 1.0),
        "source": str(payload.get("source") or "forced_oracle"),
    }
    _copy_metadata(payload, result)
    return result


def route_choice_to_primitive(decision: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    cfg = config or {}
    route = str(decision.get("route_choice") or "").strip().lower()
    source = str(decision.get("source") or decision.get("role") or "route_choice")
    yaw_deg = abs(float(deep_get(cfg, "route_choice_bridge.turn_yaw_deg", 35.0)))
    distance_m = float(deep_get(cfg, "route_choice_bridge.forward_distance_m", 0.35))
    ttl_sec = float(deep_get(cfg, "route_choice_bridge.ttl_sec", 0.75))
    request_id = str(decision.get("request_id") or new_id("route_choice_primitive"))
    if route == "left":
        return {
            "primitive": "follow_waypoint",
            "distance_m": distance_m,
            "yaw_deg": yaw_deg,
            "ttl_sec": ttl_sec,
            "request_id": request_id,
            "source": source,
            "route_choice": route,
        }
    if route == "right":
        return {
            "primitive": "follow_waypoint",
            "distance_m": distance_m,
            "yaw_deg": -yaw_deg,
            "ttl_sec": ttl_sec,
            "request_id": request_id,
            "source": source,
            "route_choice": route,
        }
    if route == "front":
        return {
            "primitive": "move_forward",
            "distance_m": distance_m,
            "yaw_deg": 0.0,
            "ttl_sec": ttl_sec,
            "request_id": request_id,
            "source": source,
            "route_choice": route,
        }
    if route == "scan":
        return {
            "primitive": "look_around",
            "ttl_sec": ttl_sec,
            "request_id": request_id,
            "source": source,
            "route_choice": route,
        }
    if route == "stop":
        return {
            "primitive": "stop",
            "ttl_sec": min(ttl_sec, 0.5),
            "request_id": request_id,
            "source": source,
            "route_choice": route,
        }
    return None


def stop_decision_to_primitive(decision: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if not bool(decision.get("stop", False)):
        return None
    cfg = config or {}
    source = str(decision.get("source") or decision.get("role") or "semantic_stop")
    if source.startswith("step") and bool(deep_get(cfg, "semantic_stop_bridge.require_confirmed_track", False)):
        track = decision.get("track") if isinstance(decision.get("track"), dict) else {}
        if not (bool(track.get("confirmed", False)) and bool(track.get("visible", False))):
            return None
    ttl_sec = float(deep_get(cfg, "semantic_stop_bridge.ttl_sec", 0.5))
    return {
        "primitive": "stop",
        "ttl_sec": ttl_sec,
        "request_id": str(decision.get("request_id") or new_id("stop_primitive")),
        "source": source,
        "reason": str(decision.get("reason") or "stop_decision"),
        "target_track": decision.get("track", {}),
    }


def duplicate_request_id(payload: dict[str, Any], seen: set[str]) -> bool:
    request_id = str(payload.get("request_id") or "")
    if not request_id:
        return False
    if request_id in seen:
        return True
    seen.add(request_id)
    return False


def source_stamp_out_of_order(payload: dict[str, Any], last_source_stamp: float | None) -> tuple[bool, float | None]:
    raw = payload.get("source_stamp_sec", payload.get("header_stamp_sec"))
    try:
        source_stamp = float(raw)
    except (TypeError, ValueError):
        return False, last_source_stamp
    if last_source_stamp is not None and source_stamp + 1.0e-6 < last_source_stamp:
        return True, last_source_stamp
    return False, source_stamp if last_source_stamp is None else max(last_source_stamp, source_stamp)


class RouteStopPrimitiveBridgeNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run RouteStopPrimitiveBridgeNode")
        super().__init__("route_stop_primitive_bridge")
        self.declare_parameter("config_file", "")
        self.config = load_yaml_file(self.get_parameter("config_file").value)
        self.active_episode_id = ""
        self.retired_episode_ids: set[str] = set()
        self.seen_request_ids: set[str] = set()
        self.last_source_stamp_by_role: dict[str, float | None] = {"route": None, "stop": None}
        self.sensor_only_planning = False
        self.primitive_pub = self.create_publisher(String, "/primitive/command_json", 10)
        self.planner_route_pub = self.create_publisher(String, "/planner/route_decision_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
        self.create_subscription(String, "/oracle/route_choice_json", self.on_forced_route, 10)
        self.create_subscription(String, "/step/route_choice_json", self.on_step_route, 10)
        self.create_subscription(String, "/oracle/semantic_stop_json", self.on_forced_stop, 10)
        self.create_subscription(String, "/oracle/stop_json", self.on_forced_stop, 10)
        self.create_subscription(String, "/step/semantic_stop_json", self.on_step_stop, 10)
        self.create_subscription(String, "/scheduler/semantic_stop_release_json", self.on_step_stop, 10)

    def on_benchmark_mode(self, msg):
        payload = mode_payload_from_json(msg.data)
        mode_config = payload.get("mode_config") if isinstance(payload.get("mode_config"), dict) else {}
        self.sensor_only_planning = bool(mode_config.get("sensor_only_planning", False))
        episode_id = str(payload.get("episode_id", ""))
        if episode_id and episode_id != self.active_episode_id:
            if self.active_episode_id:
                self.retired_episode_ids.add(self.active_episode_id)
                if len(self.retired_episode_ids) > 64:
                    self.retired_episode_ids = set(sorted(self.retired_episode_ids)[-32:])
            self.active_episode_id = episode_id
            self.seen_request_ids.clear()
            self.last_source_stamp_by_role = {"route": None, "stop": None}
            self.publish_metric("episode_reset", episode_id=episode_id, reset_scope="route_stop_primitive_bridge", result="cleared")

    def on_forced_route(self, msg):
        self.handle_route(msg.data, forced=True)

    def on_step_route(self, msg):
        self.handle_route(msg.data, forced=False)

    def on_forced_stop(self, msg):
        self.handle_stop(msg.data, forced=True)

    def on_step_stop(self, msg):
        self.handle_stop(msg.data, forced=False)

    def handle_route(self, raw: str, *, forced: bool) -> None:
        try:
            decision = forced_route_choice_json(raw) if forced else parse_route_choice_json(raw)
            if not forced:
                decision = dict(decision) | {"source": "step_route_choice"}
            decision = self.with_timebase(decision)
            stale = self.role_decision_stale(decision)
            if stale.discard:
                self.publish_metric("route_choice_bridge_stale", result="discarded", **stale.to_metric_fields(), decision=decision)
                return
            out_of_order, source_stamp = source_stamp_out_of_order(
                decision, self.last_source_stamp_by_role["route"]
            )
            if out_of_order:
                self.publish_metric(
                    "route_choice_bridge_stale",
                    result="discarded",
                    attribution="out_of_order",
                    decision=decision,
                )
                return
            if duplicate_request_id(decision, self.seen_request_ids):
                self.publish_metric(
                    "route_choice_bridge_stale",
                    result="discarded",
                    attribution="duplicate_response",
                    decision=decision,
                )
                return
            self.last_source_stamp_by_role["route"] = source_stamp
            primitive = None if self.sensor_only_planning else route_choice_to_primitive(decision, self.config)
        except Exception as exc:
            self.publish_metric("route_choice_bridge_parse_error", result="parse_error", error=repr(exc), raw=raw)
            return
        if self.sensor_only_planning:
            self.publish_json(self.planner_route_pub, decision)
            self.publish_metric(
                "route_choice_bridge",
                result="sensor_planner_decision_published",
                decision=decision,
            )
            return
        if primitive is None:
            self.publish_metric("route_choice_bridge_noop", result="noop", decision=decision)
            return
        _copy_metadata(decision, primitive)
        primitive = self.with_timebase(primitive, request_id=str(primitive.get("request_id") or ""))
        self.publish_json(self.primitive_pub, primitive)
        self.publish_metric("route_choice_bridge", result="primitive_published", decision=decision, primitive=primitive)

    def handle_stop(self, raw: str, *, forced: bool) -> None:
        try:
            decision = forced_stop_json(raw) if forced else parse_semantic_stop_json(raw)
            if not forced:
                decision = dict(decision) | {"source": "step_semantic_stop"}
            decision = self.with_timebase(decision)
            stale = self.role_decision_stale(decision)
            if stale.discard:
                self.publish_metric("semantic_stop_bridge_stale", result="discarded", **stale.to_metric_fields(), decision=decision)
                return
            out_of_order, source_stamp = source_stamp_out_of_order(
                decision, self.last_source_stamp_by_role["stop"]
            )
            if out_of_order:
                self.publish_metric(
                    "semantic_stop_bridge_stale",
                    result="discarded",
                    attribution="out_of_order",
                    decision=decision,
                )
                return
            if duplicate_request_id(decision, self.seen_request_ids):
                self.publish_metric(
                    "semantic_stop_bridge_stale",
                    result="discarded",
                    attribution="duplicate_response",
                    decision=decision,
                )
                return
            self.last_source_stamp_by_role["stop"] = source_stamp
            primitive = stop_decision_to_primitive(decision, self.config)
        except Exception as exc:
            self.publish_metric("semantic_stop_bridge_parse_error", result="parse_error", error=repr(exc), raw=raw)
            return
        if primitive is None:
            self.publish_metric("semantic_stop_bridge_noop", result="noop", decision=decision)
            return
        _copy_metadata(decision, primitive)
        primitive = self.with_timebase(primitive, request_id=str(primitive.get("request_id") or ""))
        self.publish_json(self.primitive_pub, primitive)
        self.publish_metric("semantic_stop_bridge", result="primitive_published", decision=decision, primitive=primitive)

    def with_timebase(self, payload: dict[str, Any], *, request_id: str | None = None) -> dict[str, Any]:
        source_stamp = payload.get("source_stamp_sec") or payload.get("created_ros_time_sec") or node_ros_now_sec(self)
        return attach_timebase(
            payload,
            node=self,
            episode_id=str(payload.get("episode_id") or self.active_episode_id),
            mission_id=str(payload.get("mission_id") or ""),
            request_id=str(request_id or payload.get("request_id") or new_id("route_stop")),
            clock_domain=str(payload.get("clock_domain") or clock_domain_from_node(self)),
            source_stamp=source_stamp,
            created_ros_time=node_ros_now_sec(self),
        )

    def role_decision_stale(self, payload: dict[str, Any]):
        episode_id = str(payload.get("episode_id") or "")
        return evaluate_role_decision(
            payload,
            node_ros_now_sec(self),
            self.config.get("stale_gate", self.config),
            current_episode_id=self.active_episode_id,
            current_clock_domain=clock_domain_from_node(self),
            strict_timebase=True,
            old_response_after_reset=bool(episode_id and episode_id in self.retired_episode_ids),
        )

    def publish_metric(self, event_type: str, **kwargs):
        self.publish_json(self.metric_pub, make_metric(event_type, model="route_stop_primitive_bridge", **kwargs))

    @staticmethod
    def publish_json(pub, payload: dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)


def _safe_json(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        value = json.loads(str(raw or ""))
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _copy_metadata(source: dict[str, Any], target: dict[str, Any]) -> None:
    for key in (
        "episode_id",
        "mission_id",
        "request_id",
        "clock_domain",
        "ros_now_sec",
        "wall_now_sec",
        "clock_msg_sec",
        "header_stamp_sec",
        "source_stamp_sec",
        "created_ros_time_sec",
        "created_wall_time_sec",
        "ros_now",
        "wall_now",
        "clock_msg_time",
        "header_stamp",
        "source_stamp",
        "created_ros_time",
        "created_wall_time",
        "timebase",
    ):
        if key in source:
            target[key] = source[key]


def main(args=None):
    rclpy.init(args=args)
    node = RouteStopPrimitiveBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
