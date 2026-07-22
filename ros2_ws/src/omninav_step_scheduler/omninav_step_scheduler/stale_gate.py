from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .schemas import OmniNavAction, StepPlan, TimebaseMeta, timebase_from_payload


DEFAULT_CONFIG = {
    "step_result_ttl_sec": 3.0,
    "step_pose_delta_m": 0.30,
    "step_yaw_delta_deg": 15.0,
    "multimodal_step_pose_delta_m": 0.20,
    "multimodal_step_yaw_delta_deg": 10.0,
    "omninav_pose_delta_m": 0.30,
    "omninav_yaw_delta_deg": 15.0,
    "max_action_age_sec": 0.5,
    "max_clock_skew_sec": 0.25,
}

STALE_ATTRIBUTIONS = {
    "valid",
    "stale_due_to_age",
    "stale_due_to_pose_delta",
    "timebase_error",
    "episode_mismatch",
    "missing_timestamp",
    "queue_delay",
    "old_response_after_reset",
    "unknown",
}

COMPARABLE_CLOCK_DOMAIN_SETS = [
    {"wall", "ros_system"},
    {"ros_sim"},
]


@dataclass
class StaleDecision:
    valid: bool
    attribution: str
    reason: str
    discard: bool = False
    age_sec: float | None = None
    pose_delta_m: float | None = None
    yaw_delta_deg: float | None = None
    threshold_sec: float | None = None
    pose_threshold_m: float | None = None
    yaw_threshold_deg: float | None = None
    clock_domain: str = "unknown"
    current_clock_domain: str = "unknown"
    episode_id: str = ""
    current_episode_id: str = ""
    mission_id: str = ""
    request_id: str = ""
    queue_delay_sec: float | None = None
    clock_skew_sec: float | None = None
    missing_fields: list[str] | None = None

    def to_metric_fields(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "discard": self.discard,
            "stale": self.discard and self.attribution.startswith("stale_due_to"),
            "attribution": self.attribution,
            "discard_reason": self.attribution,
            "reason": self.reason,
            "age_sec": self.age_sec,
            "pose_delta_m": self.pose_delta_m,
            "yaw_delta_deg": self.yaw_delta_deg,
            "threshold_sec": self.threshold_sec,
            "pose_threshold_m": self.pose_threshold_m,
            "yaw_threshold_deg": self.yaw_threshold_deg,
            "clock_domain": self.clock_domain,
            "current_clock_domain": self.current_clock_domain,
            "episode_id": self.episode_id,
            "current_episode_id": self.current_episode_id,
            "mission_id": self.mission_id,
            "request_id": self.request_id,
            "queue_delay_sec": self.queue_delay_sec,
            "clock_skew_sec": self.clock_skew_sec,
            "missing_fields": list(self.missing_fields or []),
        }


def valid_decision(**kwargs: Any) -> StaleDecision:
    return StaleDecision(valid=True, discard=False, attribution="valid", reason="valid", **kwargs)


def discard_decision(attribution: str, reason: str, **kwargs: Any) -> StaleDecision:
    if attribution not in STALE_ATTRIBUTIONS:
        attribution = "unknown"
    return StaleDecision(valid=False, discard=True, attribution=attribution, reason=reason, **kwargs)


def pose_delta_m(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def yaw_delta_deg(a_yaw: float, b_yaw: float) -> float:
    diff = (float(a_yaw) - float(b_yaw) + 180.0) % 360.0 - 180.0
    return abs(diff)


def _cfg(config: dict[str, Any] | None, key: str) -> Any:
    if not config:
        return DEFAULT_CONFIG[key]
    return config.get(key, DEFAULT_CONFIG[key])


def evaluate_step_plan(
    plan: StepPlan,
    current_pose: list[float],
    current_time: float,
    config: dict[str, Any] | None = None,
    *,
    current_episode_id: str = "",
    current_clock_domain: str = "wall",
    strict_timebase: bool = True,
    old_response_after_reset: bool = False,
) -> StaleDecision:
    if plan.multimodal:
        max_pose = float(_cfg(config, "multimodal_step_pose_delta_m"))
        max_yaw = float(_cfg(config, "multimodal_step_yaw_delta_deg"))
    else:
        max_pose = float(_cfg(config, "step_pose_delta_m"))
        max_yaw = float(_cfg(config, "step_yaw_delta_deg"))
    max_age = float(_cfg(config, "step_result_ttl_sec"))
    return _evaluate_common(
        plan,
        current_pose,
        current_time,
        max_age,
        max_pose,
        max_yaw,
        snapshot_pose=plan.pose_at_request,
        legacy_timestamp=plan.timestamp_response,
        config=config,
        current_episode_id=current_episode_id,
        current_clock_domain=current_clock_domain,
        strict_timebase=strict_timebase,
        old_response_after_reset=old_response_after_reset,
    )


def evaluate_omninav_action(
    action: OmniNavAction,
    current_pose: list[float],
    current_time: float,
    config: dict[str, Any] | None = None,
    *,
    current_episode_id: str = "",
    current_clock_domain: str = "wall",
    strict_timebase: bool = True,
    old_response_after_reset: bool = False,
) -> StaleDecision:
    max_age = min(float(_cfg(config, "max_action_age_sec")), float(action.ttl_sec))
    return _evaluate_common(
        action,
        current_pose,
        current_time,
        max_age,
        float(_cfg(config, "omninav_pose_delta_m")),
        float(_cfg(config, "omninav_yaw_delta_deg")),
        snapshot_pose=action.pose_at_snapshot,
        legacy_timestamp=action.timestamp_response,
        config=config,
        current_episode_id=current_episode_id,
        current_clock_domain=current_clock_domain,
        strict_timebase=strict_timebase,
        old_response_after_reset=old_response_after_reset,
    )


def evaluate_role_decision(
    payload: dict[str, Any],
    current_time: float,
    config: dict[str, Any] | None = None,
    *,
    current_episode_id: str = "",
    current_clock_domain: str = "wall",
    strict_timebase: bool = True,
    old_response_after_reset: bool = False,
) -> StaleDecision:
    """Validate a route/stop role output before it becomes a primitive."""

    meta = timebase_from_payload(payload)
    max_age = float((config or {}).get("role_decision_ttl_sec", _cfg(config, "step_result_ttl_sec")))
    queue_delay = _optional_float(payload.get("queue_delay_sec"))
    common = {
        "threshold_sec": max_age,
        "clock_domain": meta.clock_domain,
        "current_clock_domain": current_clock_domain or "unknown",
        "episode_id": meta.episode_id,
        "current_episode_id": current_episode_id,
        "mission_id": meta.mission_id,
        "request_id": str(payload.get("request_id") or meta.request_id or ""),
        "queue_delay_sec": queue_delay,
    }
    if old_response_after_reset:
        return discard_decision("old_response_after_reset", "response belongs to an already reset episode", **common)
    if current_episode_id and meta.episode_id and meta.episode_id != current_episode_id:
        return discard_decision("episode_mismatch", "message episode_id does not match active episode", **common)
    if strict_timebase:
        missing = meta.missing_fields()
        if missing:
            return discard_decision(
                "missing_timestamp",
                "required role-decision timebase field is missing",
                missing_fields=missing,
                **common,
            )
        if not clock_domains_comparable(meta.clock_domain, current_clock_domain):
            return discard_decision("timebase_error", "clock domains are not comparable", **common)
    source_time = _source_time(meta, _optional_float(payload.get("timestamp_response")) or 0.0, strict_timebase)
    if source_time is None:
        return discard_decision("missing_timestamp", "source timestamp is missing", missing_fields=["source_stamp_sec"], **common)
    age = float(current_time) - float(source_time)
    if not math.isfinite(age):
        return discard_decision("timebase_error", "computed age is non-finite", age_sec=age, **common)
    if age < 0.0:
        max_skew = float((config or {}).get("max_clock_skew_sec", DEFAULT_CONFIG["max_clock_skew_sec"]))
        if age < -max_skew:
            return discard_decision(
                "timebase_error",
                "computed age is negative beyond clock skew tolerance",
                age_sec=age,
                clock_skew_sec=abs(age),
                **common,
            )
        common["clock_skew_sec"] = abs(age)
        age = 0.0
    if queue_delay is not None and queue_delay > float((config or {}).get("max_queue_delay_sec", max_age)):
        return discard_decision("queue_delay", "response waited too long in queue", age_sec=age, **common)
    if age > max_age:
        return discard_decision("stale_due_to_age", "role decision age exceeds TTL", age_sec=age, **common)
    return valid_decision(age_sec=age, **common)


def is_step_plan_valid(plan: StepPlan, current_pose: list[float], current_time: float, config: dict[str, Any] | None = None) -> bool:
    return evaluate_step_plan(plan, current_pose, current_time, config, strict_timebase=False).valid


def is_omninav_action_valid(action: OmniNavAction, current_pose: list[float], current_time: float, config: dict[str, Any] | None = None) -> bool:
    return evaluate_omninav_action(action, current_pose, current_time, config, strict_timebase=False).valid


def _evaluate_common(
    item: Any,
    current_pose: list[float],
    current_time: float,
    max_age: float,
    max_pose: float,
    max_yaw: float,
    *,
    snapshot_pose: list[float],
    legacy_timestamp: float,
    config: dict[str, Any] | None,
    current_episode_id: str,
    current_clock_domain: str,
    strict_timebase: bool,
    old_response_after_reset: bool,
) -> StaleDecision:
    payload = _payload_from_obj(item)
    meta = _meta_from_obj(item, payload)
    request_id = str(payload.get("request_id") or meta.request_id or "")
    queue_delay = _optional_float(payload.get("queue_delay_sec"))
    common = {
        "threshold_sec": max_age,
        "pose_threshold_m": max_pose,
        "yaw_threshold_deg": max_yaw,
        "clock_domain": meta.clock_domain,
        "current_clock_domain": current_clock_domain or "unknown",
        "episode_id": meta.episode_id,
        "current_episode_id": current_episode_id,
        "mission_id": meta.mission_id,
        "request_id": request_id,
        "queue_delay_sec": queue_delay,
    }
    if old_response_after_reset:
        return discard_decision("old_response_after_reset", "response belongs to an already reset request", **common)
    if current_episode_id and meta.episode_id and meta.episode_id != current_episode_id:
        return discard_decision("episode_mismatch", "message episode_id does not match active episode", **common)
    if strict_timebase:
        missing = meta.missing_fields()
        if current_episode_id and not meta.episode_id and "episode_id" not in missing:
            missing.append("episode_id")
        if missing:
            return discard_decision("missing_timestamp", "required v5 timebase field is missing", missing_fields=missing, **common)
    source_time = _source_time(meta, legacy_timestamp, strict_timebase)
    if source_time is None:
        return discard_decision("missing_timestamp", "source timestamp is missing", missing_fields=["source_stamp_sec"], **common)
    current_domain = current_clock_domain or meta.clock_domain
    if strict_timebase and not clock_domains_comparable(meta.clock_domain, current_domain):
        return discard_decision("timebase_error", "clock domains are not comparable", **common)
    age = float(current_time) - float(source_time)
    if not math.isfinite(age):
        return discard_decision("timebase_error", "computed age is negative or non-finite", age_sec=age, **common)
    if age < 0.0:
        max_skew = float((config or {}).get("max_clock_skew_sec", DEFAULT_CONFIG["max_clock_skew_sec"]))
        if age < -max_skew:
            return discard_decision(
                "timebase_error",
                "computed age is negative beyond clock skew tolerance",
                age_sec=age,
                clock_skew_sec=abs(age),
                **common,
            )
        age = 0.0
        common["clock_skew_sec"] = abs(float(current_time) - float(source_time))
    if queue_delay is not None and queue_delay > float((config or {}).get("max_queue_delay_sec", max_age)):
        return discard_decision("queue_delay", "response waited too long in queue", age_sec=age, **common)
    if age > max_age:
        return discard_decision("stale_due_to_age", "message age exceeds TTL", age_sec=age, **common)
    p_delta = pose_delta_m(snapshot_pose, current_pose)
    y_delta = yaw_delta_deg(snapshot_pose[2], current_pose[2])
    if p_delta >= max_pose or y_delta >= max_yaw:
        return discard_decision(
            "stale_due_to_pose_delta",
            "robot moved too far since source snapshot",
            age_sec=age,
            pose_delta_m=p_delta,
            yaw_delta_deg=y_delta,
            **common,
        )
    return valid_decision(age_sec=age, pose_delta_m=p_delta, yaw_delta_deg=y_delta, **common)


def clock_domains_comparable(left: str, right: str) -> bool:
    left = str(left or "unknown")
    right = str(right or "unknown")
    if left == "unknown" or right == "unknown":
        return False
    if left == right:
        return True
    return any(left in group and right in group for group in COMPARABLE_CLOCK_DOMAIN_SETS)


def _payload_from_obj(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    payload = dict(getattr(item, "__dict__", {}) or {})
    return payload


def _meta_from_obj(item: Any, payload: dict[str, Any]) -> TimebaseMeta:
    if isinstance(payload.get("timebase"), dict):
        meta = timebase_from_payload(payload, fallback_request_id=str(payload.get("request_id") or ""))
    else:
        meta = timebase_from_payload(
            payload | {"timebase": getattr(item, "timebase", {}) or {}},
            fallback_request_id=str(payload.get("request_id") or ""),
        )
    if not meta.episode_id and getattr(item, "episode_id", ""):
        meta.episode_id = str(getattr(item, "episode_id"))
    if not meta.mission_id and getattr(item, "mission_id", ""):
        meta.mission_id = str(getattr(item, "mission_id"))
    if meta.clock_domain == "unknown" and getattr(item, "clock_domain", ""):
        meta.clock_domain = str(getattr(item, "clock_domain"))
    return meta


def _source_time(meta: TimebaseMeta, legacy_timestamp: float, strict_timebase: bool) -> float | None:
    if meta.source_stamp_sec is not None:
        return meta.source_stamp_sec
    if meta.created_ros_time_sec is not None:
        return meta.created_ros_time_sec
    if meta.created_wall_time_sec is not None:
        return meta.created_wall_time_sec
    if not strict_timebase:
        return float(legacy_timestamp)
    return None


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
