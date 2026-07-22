from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SchemaError(ValueError):
    pass


class SchedulerState(str, Enum):
    IDLE = "IDLE"
    STEP_THINK_STOP = "STEP_THINK_STOP"
    STEP_THINK_MOVE = "STEP_THINK_MOVE"
    STEP_THINK_SCAN = "STEP_THINK_SCAN"
    RUN_FAST = "RUN_FAST"
    EXECUTE_PRIMITIVE = "EXECUTE_PRIMITIVE"
    VERIFY = "VERIFY"
    REPLAN = "REPLAN"
    FAILSAFE = "FAILSAFE"
    DONE = "DONE"


PRIMITIVES = {
    "move_forward",
    "turn_left",
    "turn_right",
    "stop",
    "inspect",
    "back_off",
    "look_around",
    "follow_waypoint",
}
PENDING_MODES = {"stop", "move_slow", "safe_scan"}


@dataclass
class UserInstruction:
    instruction: str
    mission_id: str
    timestamp: float


TIMEBASE_FIELDS = [
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
]

TIMEBASE_ALIASES = {
    "ros_now": "ros_now_sec",
    "wall_now": "wall_now_sec",
    "clock_msg_time": "clock_msg_sec",
    "header_stamp": "header_stamp_sec",
    "source_stamp": "source_stamp_sec",
    "created_ros_time": "created_ros_time_sec",
    "created_wall_time": "created_wall_time_sec",
}


@dataclass
class TimebaseMeta:
    episode_id: str = ""
    mission_id: str = ""
    request_id: str = ""
    clock_domain: str = "unknown"
    ros_now_sec: float | None = None
    wall_now_sec: float | None = None
    clock_msg_sec: float | None = None
    header_stamp_sec: float | None = None
    source_stamp_sec: float | None = None
    created_ros_time_sec: float | None = None
    created_wall_time_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "mission_id": self.mission_id,
            "request_id": self.request_id,
            "clock_domain": self.clock_domain,
            "ros_now_sec": self.ros_now_sec,
            "wall_now_sec": self.wall_now_sec,
            "clock_msg_sec": self.clock_msg_sec,
            "header_stamp_sec": self.header_stamp_sec,
            "source_stamp_sec": self.source_stamp_sec,
            "created_ros_time_sec": self.created_ros_time_sec,
            "created_wall_time_sec": self.created_wall_time_sec,
        }

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        for name in ("episode_id", "request_id", "clock_domain"):
            value = getattr(self, name)
            if value is None or str(value).strip() == "" or (name == "clock_domain" and value == "unknown"):
                missing.append(name)
        for name in ("created_ros_time_sec", "created_wall_time_sec", "source_stamp_sec"):
            if getattr(self, name) is None:
                missing.append(name)
        return missing


@dataclass
class OmniNavAction:
    request_id: str
    timestamp_request: float
    timestamp_response: float
    frame_timestamp: float
    pose_at_snapshot: list[float]
    primitive: str
    distance_m: float
    yaw_deg: float
    confidence: float
    raw_text: str = ""
    raw_waypoint: list[float] = field(default_factory=list)
    raw_action: str = ""
    parser_reason: str = ""
    fallback_used: bool = False
    fallback_reason: str = ""
    source: str = "omninav"
    ttl_sec: float = 0.5
    episode_id: str = ""
    mission_id: str = ""
    clock_domain: str = "unknown"
    timebase: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepConstraints:
    max_speed_mps: float = 0.2
    avoid_people: bool = True
    stop_if_uncertain: bool = True
    forbidden_zones: list[str] = field(default_factory=list)


@dataclass
class StepPlan:
    request_id: str
    timestamp_request: float
    timestamp_response: float
    multimodal: bool
    pose_at_request: list[float]
    navila_or_omninav_instruction: str
    subgoal: str
    success_condition: str
    constraints: StepConstraints
    replan_triggers: list[str]
    recommended_pending_mode: str
    confidence: float
    raw_json: dict[str, Any] = field(default_factory=dict)
    episode_id: str = ""
    mission_id: str = ""
    clock_domain: str = "unknown"
    timebase: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyStatus:
    timestamp: float
    local_costmap_clear: bool = True
    obstacle_distance_m: float | None = None
    human_distance_m: float | None = None
    on_slope_or_stairs: bool = False
    near_doorway: bool = False
    near_intersection: bool = False
    dynamic_obstacle: bool = False
    estop: bool = False
    deadman: bool = False
    robot_fallen_or_unstable: bool = False
    battery_ok: bool = True


def now() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def stamp_to_sec(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    sec = getattr(value, "sec", None)
    nanosec = getattr(value, "nanosec", None)
    if sec is not None and nanosec is not None:
        try:
            return float(sec) + float(nanosec) * 1e-9
        except (TypeError, ValueError):
            return None
    stamp = getattr(value, "stamp", None)
    if stamp is not None:
        return stamp_to_sec(stamp)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def node_ros_now_sec(node: Any | None = None) -> float:
    if node is None:
        return now()
    try:
        stamp = node.get_clock().now()
        return float(stamp.nanoseconds) * 1e-9
    except Exception:
        return now()


def clock_domain_from_node(node: Any | None = None) -> str:
    if node is None:
        return "wall"
    try:
        clock = node.get_clock()
        active = bool(getattr(clock, "ros_time_is_active", False))
        clock_type = str(getattr(clock, "clock_type", "")).lower()
        if "ros_time" in clock_type:
            return "ros_sim" if active else "ros_system"
        if "system" in clock_type:
            return "ros_system"
    except Exception:
        return "unknown"
    return "unknown"


def timebase_from_payload(raw: dict[str, Any] | None, *, fallback_request_id: str = "", fallback_mission_id: str = "") -> TimebaseMeta:
    payload = raw if isinstance(raw, dict) else {}
    nested = payload.get("timebase") if isinstance(payload.get("timebase"), dict) else {}
    merged: dict[str, Any] = {}
    merged.update(nested)
    for key in TIMEBASE_FIELDS:
        if key in payload:
            merged[key] = payload[key]
    for alias, canonical in TIMEBASE_ALIASES.items():
        if canonical not in merged and alias in payload:
            merged[canonical] = payload[alias]
        if canonical not in merged and alias in nested:
            merged[canonical] = nested[alias]
    return TimebaseMeta(
        episode_id=str(merged.get("episode_id") or payload.get("episode_id") or ""),
        mission_id=str(merged.get("mission_id") or payload.get("mission_id") or fallback_mission_id or ""),
        request_id=str(merged.get("request_id") or payload.get("request_id") or fallback_request_id or ""),
        clock_domain=str(merged.get("clock_domain") or payload.get("clock_domain") or "unknown"),
        ros_now_sec=_optional_float(merged.get("ros_now_sec")),
        wall_now_sec=_optional_float(merged.get("wall_now_sec")),
        clock_msg_sec=_optional_float(merged.get("clock_msg_sec")),
        header_stamp_sec=_optional_float(merged.get("header_stamp_sec")),
        source_stamp_sec=_optional_float(merged.get("source_stamp_sec")),
        created_ros_time_sec=_optional_float(merged.get("created_ros_time_sec")),
        created_wall_time_sec=_optional_float(merged.get("created_wall_time_sec")),
    )


def make_timebase_meta(
    *,
    node: Any | None = None,
    episode_id: str = "",
    mission_id: str = "",
    request_id: str = "",
    clock_domain: str | None = None,
    clock_msg_time: Any | None = None,
    header_stamp: Any | None = None,
    source_stamp: Any | None = None,
    created_ros_time: float | None = None,
    created_wall_time: float | None = None,
) -> TimebaseMeta:
    wall = now()
    ros_now = node_ros_now_sec(node)
    domain = str(clock_domain or clock_domain_from_node(node) or "unknown")
    source_sec = stamp_to_sec(source_stamp)
    if source_sec is None:
        source_sec = created_ros_time if created_ros_time is not None else ros_now
    return TimebaseMeta(
        episode_id=str(episode_id or ""),
        mission_id=str(mission_id or ""),
        request_id=str(request_id or ""),
        clock_domain=domain,
        ros_now_sec=ros_now,
        wall_now_sec=wall,
        clock_msg_sec=stamp_to_sec(clock_msg_time) if clock_msg_time is not None else ros_now,
        header_stamp_sec=stamp_to_sec(header_stamp) if header_stamp is not None else source_sec,
        source_stamp_sec=source_sec,
        created_ros_time_sec=created_ros_time if created_ros_time is not None else ros_now,
        created_wall_time_sec=created_wall_time if created_wall_time is not None else wall,
    )


def attach_timebase(
    payload: dict[str, Any],
    *,
    node: Any | None = None,
    episode_id: str | None = None,
    mission_id: str | None = None,
    request_id: str | None = None,
    clock_domain: str | None = None,
    clock_msg_time: Any | None = None,
    header_stamp: Any | None = None,
    source_stamp: Any | None = None,
    created_ros_time: float | None = None,
    created_wall_time: float | None = None,
) -> dict[str, Any]:
    out = dict(payload)
    rid = str(request_id or out.get("request_id") or "")
    mid = str(mission_id if mission_id is not None else out.get("mission_id", ""))
    eid = str(episode_id if episode_id is not None else out.get("episode_id", ""))
    meta = make_timebase_meta(
        node=node,
        episode_id=eid,
        mission_id=mid,
        request_id=rid,
        clock_domain=clock_domain,
        clock_msg_time=clock_msg_time,
        header_stamp=header_stamp,
        source_stamp=source_stamp,
        created_ros_time=created_ros_time,
        created_wall_time=created_wall_time,
    )
    tb = meta.to_dict()
    out.update(tb)
    out.update(
        {
            "ros_now": tb["ros_now_sec"],
            "wall_now": tb["wall_now_sec"],
            "clock_msg_time": tb["clock_msg_sec"],
            "header_stamp": tb["header_stamp_sec"],
            "source_stamp": tb["source_stamp_sec"],
            "created_ros_time": tb["created_ros_time_sec"],
            "created_wall_time": tb["created_wall_time_sec"],
            "timebase": tb,
        }
    )
    return out


def loads_json(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SchemaError("JSON payload must be an object")
    return loaded


def dumps_dataclass(value: Any) -> str:
    return json.dumps(to_dict(value), ensure_ascii=False)


def to_dict(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_dict(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [to_dict(v) for v in value]
    if isinstance(value, dict):
        return {k: to_dict(v) for k, v in value.items()}
    return value


def require(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise SchemaError(f"missing required field: {key}")
    return payload[key]


def as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name} must be a float") from exc


def as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise SchemaError(f"{name} must be a bool")


def as_pose(value: Any, name: str = "pose") -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SchemaError(f"{name} must be [x, y, yaw]")
    return [as_float(v, name) for v in value]


def parse_user_instruction(raw: str | dict[str, Any]) -> UserInstruction:
    payload = loads_json(raw) if not isinstance(raw, str) or raw.strip().startswith("{") else {"instruction": raw}
    instruction = str(require(payload, "instruction")).strip()
    if not instruction:
        raise SchemaError("instruction cannot be empty")
    return UserInstruction(
        instruction=instruction,
        mission_id=str(payload.get("mission_id") or new_id("mission")),
        timestamp=as_float(payload.get("timestamp", now()), "timestamp"),
    )


def parse_omninav_action(raw: str | dict[str, Any]) -> OmniNavAction:
    payload = loads_json(raw)
    primitive = str(require(payload, "primitive"))
    if primitive not in PRIMITIVES:
        raise SchemaError(f"unsupported primitive: {primitive}")
    confidence = as_float(require(payload, "confidence"), "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise SchemaError("confidence must be in [0, 1]")
    request_id = str(payload.get("request_id") or new_id("omni"))
    meta = timebase_from_payload(payload, fallback_request_id=request_id)
    return OmniNavAction(
        request_id=request_id,
        timestamp_request=as_float(require(payload, "timestamp_request"), "timestamp_request"),
        timestamp_response=as_float(require(payload, "timestamp_response"), "timestamp_response"),
        frame_timestamp=as_float(payload.get("frame_timestamp", payload.get("timestamp_request")), "frame_timestamp"),
        pose_at_snapshot=as_pose(require(payload, "pose_at_snapshot"), "pose_at_snapshot"),
        primitive=primitive,
        distance_m=as_float(payload.get("distance_m", 0.0), "distance_m"),
        yaw_deg=as_float(payload.get("yaw_deg", 0.0), "yaw_deg"),
        confidence=confidence,
        raw_text=str(payload.get("raw_text", "")),
        raw_waypoint=[as_float(v, "raw_waypoint") for v in payload.get("raw_waypoint", [])]
        if isinstance(payload.get("raw_waypoint"), list)
        else [],
        raw_action=str(payload.get("raw_action", "")),
        parser_reason=str(payload.get("parser_reason", "")),
        fallback_used=bool(payload.get("fallback_used", False)),
        fallback_reason=str(payload.get("fallback_reason", "")),
        source=str(payload.get("source", "omninav")),
        ttl_sec=as_float(payload.get("ttl_sec", 0.5), "ttl_sec"),
        episode_id=meta.episode_id,
        mission_id=meta.mission_id,
        clock_domain=meta.clock_domain,
        timebase=meta.to_dict(),
    )


def parse_step_plan(raw: str | dict[str, Any]) -> StepPlan:
    payload = loads_json(raw)
    constraints_raw = payload.get("constraints") or {}
    if not isinstance(constraints_raw, dict):
        raise SchemaError("constraints must be an object")
    mode = str(require(payload, "recommended_pending_mode"))
    if mode not in PENDING_MODES:
        raise SchemaError(f"unsupported recommended_pending_mode: {mode}")
    request_id = str(payload.get("request_id") or new_id("step"))
    meta = timebase_from_payload(payload, fallback_request_id=request_id)
    return StepPlan(
        request_id=request_id,
        timestamp_request=as_float(require(payload, "timestamp_request"), "timestamp_request"),
        timestamp_response=as_float(require(payload, "timestamp_response"), "timestamp_response"),
        multimodal=bool(payload.get("multimodal", False)),
        pose_at_request=as_pose(require(payload, "pose_at_request"), "pose_at_request"),
        navila_or_omninav_instruction=str(require(payload, "navila_or_omninav_instruction")),
        subgoal=str(require(payload, "subgoal")),
        success_condition=str(require(payload, "success_condition")),
        constraints=StepConstraints(
            max_speed_mps=as_float(constraints_raw.get("max_speed_mps", 0.2), "constraints.max_speed_mps"),
            avoid_people=bool(constraints_raw.get("avoid_people", True)),
            stop_if_uncertain=bool(constraints_raw.get("stop_if_uncertain", True)),
            forbidden_zones=list(constraints_raw.get("forbidden_zones") or []),
        ),
        replan_triggers=list(payload.get("replan_triggers") or []),
        recommended_pending_mode=mode,
        confidence=as_float(require(payload, "confidence"), "confidence"),
        raw_json=payload,
        episode_id=meta.episode_id,
        mission_id=meta.mission_id,
        clock_domain=meta.clock_domain,
        timebase=meta.to_dict(),
    )


def parse_safety_status(raw: str | dict[str, Any] | None) -> SafetyStatus:
    payload = loads_json(raw or {})
    return SafetyStatus(
        timestamp=as_float(payload.get("timestamp", now()), "timestamp"),
        local_costmap_clear=bool(payload.get("local_costmap_clear", True)),
        obstacle_distance_m=_nullable_float(payload.get("obstacle_distance_m"), "obstacle_distance_m"),
        human_distance_m=_nullable_float(payload.get("human_distance_m"), "human_distance_m"),
        on_slope_or_stairs=bool(payload.get("on_slope_or_stairs", False)),
        near_doorway=bool(payload.get("near_doorway", False)),
        near_intersection=bool(payload.get("near_intersection", False)),
        dynamic_obstacle=bool(payload.get("dynamic_obstacle", False)),
        estop=bool(payload.get("estop", False)),
        deadman=bool(payload.get("deadman", False)),
        robot_fallen_or_unstable=bool(payload.get("robot_fallen_or_unstable", False)),
        battery_ok=bool(payload.get("battery_ok", True)),
    )


def _nullable_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return as_float(value, name)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def make_metric(event_type: str, **kwargs: Any) -> dict[str, Any]:
    payload = {"timestamp": now(), "event_type": event_type}
    payload.update(kwargs)
    return payload


def make_event(event_type: str, **kwargs: Any) -> dict[str, Any]:
    payload = {"timestamp": now(), "type": event_type}
    payload.update(kwargs)
    return payload


def load_yaml_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load scheduler config files") from exc
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise SchemaError("config file must contain a YAML object")
    return loaded


def deep_get(config: dict[str, Any] | None, dotted: str, default: Any = None) -> Any:
    cur: Any = config or {}
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
