from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .schemas import SchemaError


ROUTE_CHOICES = {"left", "right", "front", "scan", "stop"}
VISIBLE_VIEWS = {"left", "front", "right", "none"}
FORBIDDEN_MOTION_KEYS = {
    "cmd_vel",
    "linear",
    "angular",
    "primitive",
    "waypoint",
    "twist",
    "speed",
    "yaw_rate",
}


def extract_json_object(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw or "").strip()
    if not text:
        raise SchemaError("empty JSON payload")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise SchemaError("no JSON object found")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise SchemaError("JSON payload must be an object")
    return value


def parse_route_choice_json(raw: str | dict[str, Any]) -> dict[str, Any]:
    payload = extract_json_object(raw)
    _reject_motion_fields(payload)
    route_choice = str(_require(payload, "route_choice")).strip().lower()
    visible = str(_require(payload, "visible_in_view")).strip().lower()
    if route_choice not in ROUTE_CHOICES:
        raise SchemaError(f"unsupported route_choice: {route_choice}")
    if visible not in VISIBLE_VIEWS:
        raise SchemaError(f"unsupported visible_in_view: {visible}")
    result = {
        "route_choice": route_choice,
        "confidence": _confidence(_require(payload, "confidence")),
        "evidence": _short_text(_require(payload, "evidence"), "evidence"),
        "visible_in_view": visible,
    }
    _copy_metadata(payload, result)
    return result


def parse_semantic_stop_json(raw: str | dict[str, Any]) -> dict[str, Any]:
    payload = extract_json_object(raw)
    _reject_motion_fields(payload)
    result = {
        "stop": _bool(_require(payload, "stop"), "stop"),
        "target_visible": _bool(_require(payload, "target_visible"), "target_visible"),
        "estimated_distance_ok": _bool(_require(payload, "estimated_distance_ok"), "estimated_distance_ok"),
        "confidence": _confidence(_require(payload, "confidence")),
        "reason": _short_text(_require(payload, "reason"), "reason"),
    }
    _copy_metadata(payload, result)
    return result


def route_choice_verifier(
    *,
    instruction: str = "",
    active_subgoal: str = "",
    semantic_summary: dict[str, Any] | None = None,
    intersection_context: dict[str, Any] | None = None,
    min_confidence: float = 0.55,
) -> dict[str, Any]:
    text = f"{instruction} {active_subgoal}".lower()
    summary = semantic_summary or {}
    context = intersection_context or {}
    expected = infer_expected_route(text)
    target = infer_target_object(text)
    visible = visible_view_for_target(target, summary, context)
    confidence = 0.35
    evidence = "no explicit route cue"

    if expected in {"left", "right", "front", "stop"}:
        confidence = 0.92
        evidence = f"instruction says {expected}"
    elif target and visible != "none":
        expected = visible
        confidence = 0.72
        evidence = f"target visible in {visible}"

    if expected not in ROUTE_CHOICES:
        expected = "scan"
    if confidence < min_confidence and expected != "stop":
        expected = "scan"
        evidence = "confidence below threshold"
    return parse_route_choice_json(
        {
            "route_choice": expected,
            "confidence": round(confidence, 3),
            "evidence": evidence,
            "visible_in_view": visible,
        }
    )


def semantic_stop_verifier(
    *,
    target: str = "",
    target_visible: bool = False,
    distance_m: float | None = None,
    active_subgoal: str = "",
    semantic_summary: dict[str, Any] | None = None,
    max_distance_m: float = 2.5,
    min_confidence: float = 0.55,
) -> dict[str, Any]:
    summary = semantic_summary or {}
    visible = bool(target_visible or target_visible_in_summary(target or active_subgoal, summary))
    distance = _nullable_float(distance_m)
    distance_ok = bool(distance is not None and distance <= max_distance_m)
    confidence = 0.90 if visible and distance_ok else 0.45 if visible else 0.20
    stop = bool(visible and distance_ok and confidence >= min_confidence)
    if stop:
        reason = "target visible within stop distance"
    elif visible:
        reason = "target visible but distance not ready"
    else:
        reason = "target not visible"
    return parse_semantic_stop_json(
        {
            "stop": stop,
            "target_visible": visible,
            "estimated_distance_ok": distance_ok,
            "confidence": round(confidence, 3),
            "reason": reason,
        }
    )


def coerce_route_choice_response(req: dict[str, Any], model_text: str, *, error: str | None = None) -> dict[str, Any]:
    try:
        return parse_route_choice_json(model_text)
    except Exception:
        return route_choice_verifier(
            instruction=str(req.get("instruction") or req.get("mission") or ""),
            active_subgoal=str(req.get("active_subgoal") or req.get("subgoal") or ""),
            semantic_summary=req.get("semantic_summary") if isinstance(req.get("semantic_summary"), dict) else {},
            intersection_context=req.get("intersection_context") if isinstance(req.get("intersection_context"), dict) else {},
        )


def coerce_semantic_stop_response(req: dict[str, Any], model_text: str, *, error: str | None = None) -> dict[str, Any]:
    try:
        return parse_semantic_stop_json(model_text)
    except Exception:
        return semantic_stop_verifier(
            target=str(req.get("target") or req.get("active_subgoal") or req.get("subgoal") or ""),
            target_visible=bool(req.get("target_visible", False)),
            distance_m=_nullable_float(req.get("distance_m")),
            active_subgoal=str(req.get("active_subgoal") or req.get("subgoal") or ""),
            semantic_summary=req.get("semantic_summary") if isinstance(req.get("semantic_summary"), dict) else {},
        )


def step_role_for_event(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or event.get("reason") or "").strip()
    if event_type in {"route_choice_upcoming", "doorway_choice", "turn_task", "multiple_branches_visible"}:
        return "route_choice"
    if event_type in {
        "candidate_goal_reached",
        "completion_verification",
        "object_candidate_visible",
        "target_visible",
        "distance_to_target_below_threshold",
    }:
        return "semantic_stop"
    if event_type in {
        "semantic_plan_requested",
        "semantic_subgoal_completed",
        "semantic_recovery_required",
    }:
        return "semantic_executive"
    return "disabled"


def build_route_choice_prompt(
    *,
    instruction: str,
    active_subgoal: str,
    robot_state: dict[str, Any],
    semantic_summary: dict[str, Any],
    safety_status: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any]:
    instruction_route_hint = infer_expected_route(instruction)
    target_landmark = str(event.get("target") or active_subgoal)
    system = (
        "Return one strict JSON object with all four keys exactly: "
        "route_choice,confidence,evidence,visible_in_view. "
        "Inspect the attached front camera image. The instruction_route_hint is parsed from the operator instruction; "
        "copy it to route_choice when it is left, right, front, or stop. Use the image only to verify the landmark "
        "and report its screen location in visible_in_view. Never replace an explicit instruction direction with the "
        "landmark's screen location. When instruction_route_hint is scan, choose the branch matching the requested "
        "landmark's visible_in_view; use scan only when that landmark is absent. For screen location, classify the "
        "target center in the left 40% as left, middle 20% as front, and right 40% as right. "
        "When the requested landmark is visible and instruction_route_hint is scan, route_choice must exactly equal "
        "visible_in_view. Never return scan when visible_in_view is left, front, or right. "
        "Evidence is at most two words. Do not output motor commands, waypoints, or plans."
    )
    user = {
        "role": "route_choice_verifier",
        "instruction": instruction,
        "active_subgoal": active_subgoal,
        "target_landmark": target_landmark,
        "target_visual_attributes": target_visual_attributes(target_landmark),
        "instruction_route_hint": instruction_route_hint,
        "event_type": event.get("type", "route_choice_upcoming"),
        "required_json_schema": {
            "route_choice": "left|right|front|scan|stop",
            "confidence": "number 0..1",
            "evidence": "max 2 words",
            "visible_in_view": "left|front|right|none",
        },
    }
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]}


def build_semantic_stop_prompt(
    *,
    instruction: str,
    active_subgoal: str,
    robot_state: dict[str, Any],
    semantic_summary: dict[str, Any],
    safety_status: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any]:
    distance = event.get("distance_to_target_m", event.get("distance_to_target"))
    if distance is None:
        distance = semantic_summary.get("distance_to_target_m")
    has_distance_context = distance is not None and bool(
        event.get("allow_distance_context", event.get("type") != "visual_decision_due")
    )
    distance_instruction = (
        "Copy within_stop_distance exactly to estimated_distance_ok. "
        if has_distance_context
        else "Estimate whether the target is within the configured stop distance from image perspective only. "
    )
    system = (
        "Return one strict JSON object with all five keys exactly: "
        "stop,target_visible,estimated_distance_ok,confidence,reason. "
        "Treat the target text as a query, never as evidence. Inspect the attached front camera image and internally "
        "verify every required_visual_attributes value against pixels. Set target_visible only when the exact requested "
        "target category is clearly visible. The instruction, target name, color, and any distance context are not "
        "evidence that it is present. Do not reinterpret another object as the target. Named attributes must occur on "
        "the same object. If any required attribute cannot be verified from pixels, set target_visible=false and "
        "stop=false. The image is authoritative for presence. "
        + distance_instruction +
        "Set stop to target_visible AND estimated_distance_ok. Reason is at most two words. "
        "Do not output motor commands, waypoints, or plans."
    )
    try:
        within_stop_distance = float(distance) <= 2.5
    except (TypeError, ValueError):
        within_stop_distance = False
    target = semantic_target_query(str(event.get("target") or active_subgoal))
    user = {
        "role": "semantic_stop_verifier",
        "instruction": instruction,
        "active_subgoal": active_subgoal,
        "target_query": target,
        "required_visual_attributes": target_visual_attributes(target),
        "verification_question": f"Is an actual {target} visually present in the image pixels?",
        "required_json_schema": {
            "stop": "boolean",
            "target_visible": "boolean",
            "estimated_distance_ok": "boolean",
            "confidence": "number 0..1",
            "reason": "max 2 words",
        },
    }
    if has_distance_context:
        user["distance_to_target_m"] = distance
        user["within_stop_distance"] = within_stop_distance
        user["distance_basis"] = "geometric_context"
    else:
        user["stop_distance_m"] = 2.0
        user["distance_basis"] = "image_estimate"
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]}


@dataclass
class VisibleToStopMonitor:
    target_visible_required_frames: int = 2
    max_distance_to_target_m: float = 2.5
    first_target_visible_time: float | None = None
    stop_command_time: float | None = None
    target_visible_frames: int = 0
    distance_at_first_visible: float | None = None
    distance_at_stop: float | None = None

    def update(
        self,
        *,
        timestamp: float,
        target_visible: bool,
        distance_m: float | None,
        stop_command: bool = False,
    ) -> dict[str, Any]:
        distance = _nullable_float(distance_m)
        if target_visible:
            self.target_visible_frames += 1
            if self.first_target_visible_time is None:
                self.first_target_visible_time = float(timestamp)
                self.distance_at_first_visible = distance
        else:
            self.target_visible_frames = 0
        if stop_command and self.stop_command_time is None:
            self.stop_command_time = float(timestamp)
            self.distance_at_stop = distance
        return self.summary()

    def summary(self) -> dict[str, Any]:
        latency = None
        if self.first_target_visible_time is not None and self.stop_command_time is not None:
            latency = max(0.0, self.stop_command_time - self.first_target_visible_time)
        stopped_too_late = bool(latency is not None and latency > 2.0)
        if self.distance_at_stop is not None and self.distance_at_stop > self.max_distance_to_target_m:
            stopped_too_late = True
        stopped_too_early = bool(self.stop_command_time is not None and self.first_target_visible_time is None)
        return {
            "first_target_visible_time": self.first_target_visible_time,
            "stop_command_time": self.stop_command_time,
            "visible_to_stop_latency_sec": latency,
            "target_visible_frames": self.target_visible_frames,
            "distance_at_first_visible": self.distance_at_first_visible,
            "distance_at_stop": self.distance_at_stop,
            "stopped_too_late": stopped_too_late,
            "stopped_too_early": stopped_too_early,
        }


@dataclass
class TargetTrackState:
    required_hits: int = 2
    high_confidence_single_hit: float = 0.85
    min_confidence: float = 0.60
    max_misses: int = 2
    max_age_sec: float = 3.0
    episode_id: str = ""
    target: str = ""
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    visible: bool = False
    confidence: float = 0.0
    visible_in_view: str = "none"
    first_seen_time: float | None = None
    last_seen_time: float | None = None
    frame_seq: int | None = None

    def reset(self, *, episode_id: str, target: str) -> None:
        self.episode_id = str(episode_id or "")
        self.target = str(target or "").strip().lower()
        self.hits = 0
        self.misses = 0
        self.confirmed = False
        self.visible = False
        self.confidence = 0.0
        self.visible_in_view = "none"
        self.first_seen_time = None
        self.last_seen_time = None
        self.frame_seq = None

    def update(
        self,
        *,
        timestamp: float,
        episode_id: str,
        target: str,
        visible: bool,
        confidence: float,
        visible_in_view: str = "none",
        frame_seq: int | None = None,
    ) -> dict[str, Any]:
        normalized_target = str(target or "").strip().lower()
        if str(episode_id or "") != self.episode_id or normalized_target != self.target:
            self.reset(episode_id=episode_id, target=normalized_target)
        self.confidence = max(0.0, min(1.0, float(confidence)))
        view = str(visible_in_view or "none").lower()
        self.visible_in_view = view if view in VISIBLE_VIEWS else "none"
        previous_frame_seq = self.frame_seq
        is_new_frame = frame_seq is None or previous_frame_seq is None or frame_seq != previous_frame_seq
        self.frame_seq = frame_seq
        accepted_hit = bool(visible and self.confidence >= self.min_confidence)
        self.visible = accepted_hit
        if accepted_hit and is_new_frame:
            self.hits += 1
            self.misses = 0
            if self.first_seen_time is None:
                self.first_seen_time = float(timestamp)
            self.last_seen_time = float(timestamp)
            if self.hits >= self.required_hits or self.confidence >= self.high_confidence_single_hit:
                self.confirmed = True
        elif not accepted_hit and is_new_frame:
            self.hits = 0
            self.misses += 1
            if self.misses > self.max_misses:
                self.confirmed = False
        return self.summary(timestamp=float(timestamp))

    def summary(self, *, timestamp: float) -> dict[str, Any]:
        age = None if self.last_seen_time is None else max(0.0, float(timestamp) - self.last_seen_time)
        fresh = bool(age is not None and age <= self.max_age_sec)
        return {
            "episode_id": self.episode_id,
            "target": self.target,
            "confirmed": bool(self.confirmed and fresh),
            "visible": self.visible,
            "confidence": round(self.confidence, 3),
            "visible_in_view": self.visible_in_view,
            "hits": self.hits,
            "misses": self.misses,
            "first_seen_time": self.first_seen_time,
            "last_seen_time": self.last_seen_time,
            "age_sec": None if age is None else round(age, 3),
            "frame_seq": self.frame_seq,
        }


@dataclass
class SemanticStopGate:
    target_visible_required_frames: int = 2
    max_distance_to_target_m: float = 2.5
    force_stop_if_distance_less_than_m: float = 2.0
    force_stop_after_visible_sec: float = 1.5
    min_confidence: float = 0.55

    def __post_init__(self) -> None:
        self.monitor = VisibleToStopMonitor(
            target_visible_required_frames=self.target_visible_required_frames,
            max_distance_to_target_m=self.max_distance_to_target_m,
        )

    def update(
        self,
        *,
        timestamp: float,
        target_visible: bool,
        distance_m: float | None,
        verifier_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        verifier = None
        if verifier_json:
            try:
                verifier = parse_semantic_stop_json(verifier_json)
            except Exception:
                verifier = None
        monitor = self.monitor.update(timestamp=timestamp, target_visible=target_visible, distance_m=distance_m)
        distance = _nullable_float(distance_m)
        visible_ready = bool(target_visible and monitor["target_visible_frames"] >= self.target_visible_required_frames)
        visible_duration = 0.0
        if self.monitor.first_target_visible_time is not None:
            visible_duration = max(0.0, float(timestamp) - self.monitor.first_target_visible_time)

        force = False
        reason = "no_gate_condition"
        if verifier and verifier["stop"] and verifier["confidence"] >= self.min_confidence:
            force = True
            reason = "semantic_stop_verifier"
        elif visible_ready and distance is not None and distance < self.force_stop_if_distance_less_than_m:
            force = True
            reason = "visible_and_distance_below_force_threshold"
        elif visible_ready and distance is not None and distance <= self.max_distance_to_target_m and visible_duration >= self.force_stop_after_visible_sec:
            force = True
            reason = "visible_duration_exceeded"

        if force:
            monitor = self.monitor.update(timestamp=timestamp, target_visible=target_visible, distance_m=distance, stop_command=True)
        return {
            "force_stop": force,
            "reason": reason,
            "monitor": monitor,
            "visible_duration_sec": round(visible_duration, 3),
            "actions_through_safe_mux": True,
        }


def infer_expected_route(text: str) -> str:
    lowered = str(text or "").lower()
    if re.search(r"\b(turn|go|take|veer)\s+left\b|\bleft\b", lowered):
        return "left"
    if re.search(r"\b(turn|go|take|veer)\s+right\b|\bright\b", lowered):
        return "right"
    if re.search(r"\b(straight|front|forward)\b", lowered):
        return "front"
    if re.search(r"\bstop\b", lowered):
        return "stop"
    return "scan"


def target_visual_attributes(target: str) -> dict[str, str]:
    words = re.findall(r"[a-z0-9]+", str(target or "").lower().replace("_", " "))
    colors = {"red", "blue", "yellow", "green", "orange", "purple", "black", "white", "gray", "grey"}
    color = next((word for word in words if word in colors), "")
    category_words = [word for word in words if word not in colors]
    attributes = {"category": " ".join(category_words) or "unknown"}
    if color:
        attributes["color"] = color
    return attributes


def semantic_target_query(text: str) -> str:
    """Reduce an action instruction to the visual noun phrase used by Step."""

    raw = str(text or "").strip()
    inferred = infer_target_object(raw)
    if not inferred:
        return raw
    category = inferred.replace("_", " ")
    words = re.findall(r"[a-z0-9]+", raw.lower().replace("_", " "))
    colors = {"red", "blue", "yellow", "green", "orange", "purple", "black", "white", "gray", "grey"}
    color = next((word for word in words if word in colors), "")
    return f"{color} {category}".strip()


def infer_target_object(text: str) -> str:
    lowered = str(text or "").lower()
    for target in ("red_cone", "blue_box", "fire_extinguisher", "red_exit_sign", "red_box"):
        if target.replace("_", " ") in lowered or target in lowered:
            return target
    return ""


def visible_view_for_target(target: str, semantic_summary: dict[str, Any], context: dict[str, Any] | None = None) -> str:
    target = str(target or "").lower()
    if not target:
        return str((context or {}).get("visible_in_view") or "none").lower() if str((context or {}).get("visible_in_view") or "none").lower() in VISIBLE_VIEWS else "none"
    for obj in _objects_from_summary(semantic_summary):
        label = str(obj.get("label") or obj.get("name") or obj.get("class") or obj.get("id") or "").lower()
        if target and target.replace("_", " ") not in label and target not in label:
            continue
        view = str(obj.get("view") or obj.get("bearing") or obj.get("camera") or "none").lower()
        if view in VISIBLE_VIEWS:
            return view
        x = _nullable_float(obj.get("x") or obj.get("image_x"))
        if x is not None:
            if x < 0.33:
                return "left"
            if x > 0.67:
                return "right"
            return "front"
    return "none"


def target_visible_in_summary(target: str, semantic_summary: dict[str, Any]) -> bool:
    target = str(target or "").lower().replace("_", " ")
    if not target:
        return bool(semantic_summary.get("target_visible", False))
    for obj in _objects_from_summary(semantic_summary):
        label = str(obj.get("label") or obj.get("name") or obj.get("class") or obj.get("id") or "").lower().replace("_", " ")
        if target in label and bool(obj.get("visible", True)):
            return True
    return bool(semantic_summary.get("target_visible", False))


def _objects_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("objects", "visible_objects", "detections"):
        value = summary.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if isinstance(summary.get("target"), dict):
        return [summary["target"]]
    return []


def _reject_motion_fields(payload: dict[str, Any]) -> None:
    forbidden = FORBIDDEN_MOTION_KEYS.intersection(payload.keys())
    if forbidden:
        raise SchemaError(f"Step verifier JSON cannot contain motion fields: {sorted(forbidden)}")


def _require(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise SchemaError(f"missing required field: {key}")
    return payload[key]


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError("confidence must be a number") from exc
    if not 0.0 <= number <= 1.0:
        raise SchemaError("confidence must be in [0, 1]")
    return number


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise SchemaError(f"{name} must be a bool")


def _short_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SchemaError(f"{name} cannot be empty")
    if len(text) > 160:
        text = text[:157].rstrip() + "..."
    return text


def _nullable_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        "track",
        "target",
        "image_snapshot",
        "multimodal",
    ):
        if key in source:
            target[key] = source[key]
