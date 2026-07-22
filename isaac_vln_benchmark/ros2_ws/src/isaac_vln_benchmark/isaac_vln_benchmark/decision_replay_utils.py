from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable


REQUIRED_ATTRIBUTIONS = {
    "duplicate_response",
    "out_of_order",
    "episode_mismatch",
    "timebase_error",
    "old_response_after_reset",
}


def _payload(value: Any) -> dict[str, Any]:
    raw = getattr(value, "data", value)
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _linear_x(value: Any) -> float:
    if isinstance(value, dict):
        linear = value.get("linear") if isinstance(value.get("linear"), dict) else {}
        return float(linear.get("x", 0.0) or 0.0)
    return float(getattr(getattr(value, "linear", None), "x", 0.0))


def _text(value: Any) -> str:
    raw = getattr(value, "data", value)
    return str(raw if not isinstance(raw, dict) else raw.get("data") or raw.get("raw") or "")


def evaluate_decision_replay(
    records: Iterable[tuple[str, Any]], expected_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    topic_counts: Counter[str] = Counter()
    attributions: Counter[str] = Counter()
    primitive_request_ids: set[str] = set()
    pending_candidate_linear_max = 0.0
    scheduler_state = ""
    stale_action_executed = 0
    for topic, value in records:
        topic_counts[topic] += 1
        payload = _payload(value)
        if topic == "/metrics/event_jsonl":
            attribution = str(payload.get("attribution") or payload.get("discard_reason") or "")
            if attribution:
                attributions[attribution] += 1
            stale_action_executed += int(str(payload.get("event_type") or "") == "stale_action_executed")
        elif topic == "/primitive/command_json":
            primitive_request_ids.add(str(payload.get("request_id") or ""))
        elif topic == "/scheduler/state":
            scheduler_state = _text(value)
        elif topic == "/cmd_vel_candidate" and scheduler_state.startswith("STEP_THINK_"):
            pending_candidate_linear_max = max(pending_candidate_linear_max, abs(_linear_x(value)))
    expected_profiles = {str(row.get("profile") or "") for row in expected_rows}
    expected_pass = all(bool(row.get("pass")) for row in expected_rows)
    rejected_ids = {"order_old", "episode_wrong", "timebase_wrong", "old_after_reset", "dropped_http"}
    required_topics = {
        "/benchmark/mode_json",
        "/step/route_choice_json",
        "/step/request_json",
        "/primitive/command_json",
        "/metrics/event_jsonl",
        "/cmd_vel_candidate",
        "/scheduler/state",
    }
    checks = {
        "records_present": sum(topic_counts.values()) > 0,
        "required_topics_present": required_topics.issubset(topic_counts),
        "all_robustness_profiles_passed": expected_pass and len(expected_profiles) == 8,
        "required_discard_attributions_present": REQUIRED_ATTRIBUTIONS.issubset(attributions),
        "rejected_responses_never_became_primitives": not bool(rejected_ids & primitive_request_ids),
        "pending_or_timeout_never_blind_forward": pending_candidate_linear_max <= 1.0e-3,
        "stale_action_executed_zero": stale_action_executed == 0,
    }
    return {
        "schema_version": 1,
        "pass": all(checks.values()),
        "replay_kind": "rosbag2_decision_transport_offline_replay",
        "checks": checks,
        "topic_counts": dict(sorted(topic_counts.items())),
        "attributions": dict(sorted(attributions.items())),
        "primitive_request_ids": sorted(primitive_request_ids),
        "pending_candidate_linear_max_mps": pending_candidate_linear_max,
        "stale_action_executed": stale_action_executed,
        "qualification_evidence": False,
        "real_robot_motion_enabled": False,
    }
