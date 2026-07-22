from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .schemas import SchemaError


SUBGOAL_TYPES = {"find", "pass", "enter", "approach", "verify", "ask"}
RECOVERY_ACTIONS = {"scan", "backtrack", "ask", "stop"}
SEMANTIC_RESPONSE_KEYS = {
    "subgoal_type",
    "target",
    "relation",
    "constraints",
    "completion_evidence",
    "recovery",
    "confidence",
}
METADATA_KEYS = {
    "request_id",
    "episode_id",
    "mission_id",
    "clock_domain",
    "source_stamp_sec",
    "created_ros_time_sec",
    "created_wall_time_sec",
    "timebase",
    "timestamp_response",
    "source",
    "multimodal",
    "step_latency_sec",
    "image_snapshot",
    "fallback_reason",
    "completion_evidence_source",
    "task_id",
    "subgoal_index",
    "ros_now_sec",
    "wall_now_sec",
    "clock_msg_sec",
    "header_stamp_sec",
    "ros_now",
    "wall_now",
    "clock_msg_time",
    "header_stamp",
    "source_stamp",
    "created_ros_time",
    "created_wall_time",
}
FORBIDDEN_CONTROL_KEYS = {
    "cmd_vel",
    "linear",
    "angular",
    "primitive",
    "waypoint",
    "waypoints",
    "twist",
    "speed",
    "yaw",
    "yaw_rate",
    "target_pose",
    "goal_pose",
    "trajectory",
}
FORBIDDEN_ORACLE_KEYS = {
    "oracle_plan",
    "oracle_visibility",
    "expected_branch",
    "correct_branch",
    "branch_polygon",
    "branch_polygons",
    "ground_truth",
    "judge",
    "success_truth",
}
PROMPT_TRANSPORT_KEYS = {
    "request_id",
    "mission_id",
    "clock_domain",
    "source_stamp_sec",
    "created_ros_time_sec",
    "created_wall_time_sec",
    "timestamp_response",
    "ros_now_sec",
    "wall_now_sec",
    "clock_msg_sec",
    "header_stamp_sec",
    "ros_now",
    "wall_now",
    "clock_msg_time",
    "header_stamp",
    "source_stamp",
    "created_ros_time",
    "created_wall_time",
    "timebase",
}


@dataclass(frozen=True)
class SemanticSubgoal:
    subgoal_type: str
    target: str
    relation: str
    constraints: list[str]
    completion_evidence: str
    recovery: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        metadata = payload.pop("metadata")
        if include_metadata:
            payload.update(metadata)
        return payload


def parse_semantic_subgoal_json(raw: str | dict[str, Any]) -> SemanticSubgoal:
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"invalid semantic subgoal JSON: {exc}") from exc
    else:
        payload = dict(raw)
    if not isinstance(payload, dict):
        raise SchemaError("semantic subgoal must be a JSON object")
    _reject_forbidden_fields(payload)
    unknown = set(payload) - SEMANTIC_RESPONSE_KEYS - METADATA_KEYS
    missing = SEMANTIC_RESPONSE_KEYS - set(payload)
    if missing:
        raise SchemaError(f"missing semantic subgoal fields: {sorted(missing)}")
    if unknown:
        raise SchemaError(f"unsupported semantic subgoal fields: {sorted(unknown)}")

    subgoal_type = _short_text(payload["subgoal_type"], "subgoal_type", 24).lower()
    if subgoal_type not in SUBGOAL_TYPES:
        raise SchemaError(f"unsupported subgoal_type: {subgoal_type}")
    recovery = _short_text(payload["recovery"], "recovery", 24).lower()
    if recovery not in RECOVERY_ACTIONS:
        raise SchemaError(f"unsupported recovery: {recovery}")
    constraints = payload["constraints"]
    if not isinstance(constraints, list) or len(constraints) > 8:
        raise SchemaError("constraints must be a list with at most 8 entries")
    normalized_constraints = [_semantic_constraint(item) for item in constraints]
    confidence = _confidence(payload["confidence"])
    metadata = {key: payload[key] for key in METADATA_KEYS if key in payload}
    return SemanticSubgoal(
        subgoal_type=subgoal_type,
        target=_short_text(payload["target"], "target", 120),
        relation=_short_text(payload["relation"], "relation", 160, allow_empty=True),
        constraints=normalized_constraints,
        completion_evidence=_short_text(payload["completion_evidence"], "completion_evidence", 240),
        recovery=recovery,
        confidence=confidence,
        metadata=metadata,
    )


def safe_semantic_fallback(*, reason: str = "model response invalid") -> dict[str, Any]:
    return {
        "subgoal_type": "ask",
        "target": "operator clarification",
        "relation": "",
        "constraints": ["hold position"],
        "completion_evidence": "operator provides a clarified instruction",
        "recovery": "stop",
        "confidence": 0.0,
        "source": "safe_fallback",
        "fallback_reason": str(reason)[:240],
    }


def build_semantic_executive_prompt(
    *,
    instruction: str,
    active_subgoal: dict[str, Any] | None,
    observation_history: list[dict[str, Any]],
    event: dict[str, Any],
) -> dict[str, Any]:
    safe_active = sanitize_observable_context(active_subgoal or {})
    safe_history = [sanitize_observable_context(item) for item in observation_history[-8:]]
    safe_event = sanitize_observable_context(event)
    system = (
        "You are the slow semantic executive in a hierarchical robot navigation system. "
        "OmniNav owns local path following and obstacle avoidance. Select exactly one next semantic subgoal. "
        "Use only the instruction, camera-derived observations, track history, and execution events provided. "
        "Never output velocity, yaw, coordinates, waypoints, paths, trajectories, or motor primitives. "
        "When evidence is insufficient or progress is unsafe, choose ask or a recovery of scan, backtrack, ask, or stop. "
        "A terminal verify subgoal must use recovery stop. "
        "For a terminal verify stage, target must repeat the concrete semantic target from the preceding stage; "
        "never use a generic target such as arrival, goal, destination, or completion. "
        "Follow explicit numbered instruction stages in order. Select exactly the zero-based next_subgoal_index stage "
        "and never merge or skip a find stage just because its target is visible. "
        "Use an empty relation unless the instruction explicitly links the target to another object with a spatial "
        "relation such as beside, after, or inside. Nearest, pass, toward, arrival, and arrived at are not relations. "
        "The recovery field must exactly follow any recovery explicitly assigned to the current stage in the instruction. "
        "Never replace an assigned backtrack or stop recovery with scan. Before responding, verify completion_evidence "
        "is non-empty. "
        "Keep target and relation concise, use a non-empty completion_evidence of two to eight words, and use an empty "
        "constraints list unless the instruction states a semantic constraint. "
        "Return one strict JSON object with exactly these keys: subgoal_type,target,relation,constraints,"
        "completion_evidence,recovery,confidence."
    )
    user = {
        "instruction": str(instruction),
        "active_subgoal": safe_active,
        "observation_history": safe_history,
        "event": safe_event,
        "schema": {
            "subgoal_type": "find|pass|enter|approach|verify|ask",
            "target": "semantic target text",
            "relation": "semantic relation or empty string",
            "constraints": ["semantic constraint"],
            "completion_evidence": "observable evidence required for completion",
            "recovery": "scan|backtrack|ask|stop",
            "confidence": "number 0..1",
        },
    }
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
    }


def sanitize_observable_context(value: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, list):
        return [sanitize_observable_context(item, path + (str(index),)) for index, item in enumerate(value)]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        normalized = str(key).lower()
        if normalized in PROMPT_TRANSPORT_KEYS:
            continue
        if normalized in FORBIDDEN_CONTROL_KEYS or normalized in FORBIDDEN_ORACLE_KEYS:
            continue
        if normalized in {"distance_to_target_m", "target_visible"} and not _sensor_path(path):
            continue
        if normalized == "source" and "oracle" in str(item).lower():
            continue
        result[str(key)] = sanitize_observable_context(item, path + (normalized,))
    return result


def oracle_leakage_findings(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    findings: list[str] = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(oracle_leakage_findings(item, path + (str(index),)))
        return findings
    if not isinstance(value, dict):
        return findings
    for key, item in value.items():
        normalized = str(key).lower()
        current = path + (normalized,)
        dotted = ".".join(current)
        if normalized in FORBIDDEN_ORACLE_KEYS:
            findings.append(dotted)
        elif normalized in {"distance_to_target_m", "target_visible"} and not _sensor_path(path):
            findings.append(dotted)
        elif normalized == "source" and "oracle" in str(item).lower():
            findings.append(dotted)
        findings.extend(oracle_leakage_findings(item, current))
    return sorted(set(findings))


def validate_semantic_goal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SchemaError("semantic goal must be an object")
    _reject_forbidden_fields(payload)
    required = {"subgoal_type", "target", "relation", "constraints", "completion_evidence"}
    missing = required - set(payload)
    if missing:
        raise SchemaError(f"semantic goal missing fields: {sorted(missing)}")
    subgoal_type = _short_text(payload["subgoal_type"], "subgoal_type", 24).lower()
    if subgoal_type not in SUBGOAL_TYPES:
        raise SchemaError(f"unsupported subgoal_type: {subgoal_type}")
    constraints = payload["constraints"]
    if not isinstance(constraints, list) or len(constraints) > 8:
        raise SchemaError("constraints must be a list with at most 8 entries")
    return {
        "subgoal_type": subgoal_type,
        "target": _short_text(payload["target"], "target", 120),
        "relation": _short_text(payload["relation"], "relation", 160, allow_empty=True),
        "constraints": [_semantic_constraint(item) for item in constraints],
        "completion_evidence": _short_text(payload["completion_evidence"], "completion_evidence", 240),
    }


class SemanticExecutiveCore:
    """State holder for one event-triggered semantic subgoal at a time."""

    COMPLETION_EVENTS = {
        "find": {"target_track_confirmed"},
        "pass": {"landmark_passed"},
        "enter": {"region_entered"},
        "approach": {"target_within_stop_distance"},
        "verify": {"completion_verified"},
        "ask": {"clarification_received"},
    }
    RECOVERY_EVENTS = {
        "blocked_path_detected",
        "clarification_received",
        "no_progress_timeout",
        "target_track_lost",
        "semantic_low_confidence",
        "semantic_timeout",
    }

    def __init__(self, *, min_confidence: float = 0.55) -> None:
        self.min_confidence = float(min_confidence)
        self.episode_id = ""
        self.active: SemanticSubgoal | None = None
        self.history: list[dict[str, Any]] = []

    def reset(self, episode_id: str) -> None:
        self.episode_id = str(episode_id or "")
        self.active = None
        self.history = []

    def accept(self, subgoal: SemanticSubgoal) -> dict[str, Any]:
        episode_id = str(subgoal.metadata.get("episode_id") or "")
        if self.episode_id and episode_id and episode_id != self.episode_id:
            return self._result("REJECTED", reason="episode_mismatch")
        if subgoal.confidence < self.min_confidence:
            self.active = None
            return self._result("RECOVERY", reason="low_confidence", recovery="stop")
        self.active = subgoal
        self.history.append({"event": "subgoal_accepted", "subgoal": subgoal.to_dict()})
        return self._result("RUNNING", reason="accepted")

    def handle_event(self, event: dict[str, Any]) -> dict[str, Any]:
        event_type = str(event.get("type") or event.get("reason") or "")
        if not self.active:
            return self._result("NEEDS_PLAN", reason="no_active_subgoal")
        if event_type == "semantic_recovery_completed":
            resumed = self.active.to_dict()
            recovery = str(event.get("recovery") or "")
            self.history.append(
                {
                    "event": event_type,
                    "result": "recovered",
                    "recovery": recovery,
                    "subgoal": resumed,
                }
            )
            return self._result(
                "RUNNING",
                reason="semantic_recovery_completed",
                resumed_subgoal=resumed,
                recovery=recovery,
            )
        if event_type in self.COMPLETION_EVENTS[self.active.subgoal_type]:
            terminal = self.active.subgoal_type == "verify"
            completed = self.active.to_dict()
            self.history.append({"event": event_type, "result": "completed", "subgoal": completed})
            self.active = None
            return self._result(
                "COMPLETE" if terminal else "NEEDS_PLAN",
                reason="mission_verified" if terminal else "subgoal_completed",
                completed_subgoal=completed,
            )
        if event_type in self.RECOVERY_EVENTS:
            recovery = self.active.recovery
            self.history.append({"event": event_type, "result": "recovery", "recovery": recovery})
            return self._result("RECOVERY", reason=event_type, recovery=recovery)
        return self._result("RUNNING", reason="event_observed")

    def _result(self, status: str, *, reason: str, **extra: Any) -> dict[str, Any]:
        semantic_goal = None
        if self.active is not None:
            semantic_goal = {
                "subgoal_type": self.active.subgoal_type,
                "target": self.active.target,
                "relation": self.active.relation,
                "constraints": list(self.active.constraints),
                "completion_evidence": self.active.completion_evidence,
            }
        return {
            "status": status,
            "reason": reason,
            "episode_id": self.episode_id,
            "active_subgoal": self.active.to_dict() if self.active else None,
            "semantic_goal": semantic_goal,
            "publishes_motion": False,
            **extra,
        }


def _reject_forbidden_fields(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, path + (str(index),))
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        normalized = str(key).lower()
        if normalized in FORBIDDEN_CONTROL_KEYS or normalized in FORBIDDEN_ORACLE_KEYS:
            raise SchemaError(f"forbidden field in semantic subgoal: {'.'.join(path + (normalized,))}")
        _reject_forbidden_fields(item, path + (normalized,))


def _sensor_path(path: tuple[str, ...]) -> bool:
    return any(part in {"sensor_track", "perception", "observation", "observation_history"} for part in path)


def _short_text(value: Any, name: str, limit: int, *, allow_empty: bool = False) -> str:
    text = str(value or "").strip()
    if not text and not allow_empty:
        raise SchemaError(f"{name} cannot be empty")
    if len(text) > limit:
        raise SchemaError(f"{name} exceeds {limit} characters")
    return text


def _semantic_constraint(value: Any) -> str:
    text = _short_text(value, "constraint", 120)
    lowered = text.lower()
    forbidden_terms = ("cmd_vel", "waypoint", "m/s", "rad/s", "degrees", "coordinate", "trajectory")
    if any(term in lowered for term in forbidden_terms):
        raise SchemaError("constraints must be semantic, not geometric or motor commands")
    return text


def _confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError("confidence must be a number") from exc
    if not 0.0 <= confidence <= 1.0:
        raise SchemaError("confidence must be in [0, 1]")
    return confidence
