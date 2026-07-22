from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_FIELDS = frozenset(
    {
        "target_pose",
        "nav_target_pose",
        "expected_branch",
        "branch_polygon",
        "branch_polygons",
        "branch_membership",
        "oracle_visibility",
        "oracle_context",
    }
)
CONTROL_EVENT_TYPES = frozenset(
    {
        "sensor_target_track",
        "sensor_planner_primitive",
        "sensor_route_decision_accepted",
        "sensor_route_trigger",
        "grounded_sam_request",
        "primitive_command",
        "route_choice_bridge",
        "semantic_stop_bridge",
    }
)
DISALLOWED_EVIDENCE_SOURCES = ("mock", "fallback", "synthetic")


def normalized_runtime_event(row: dict[str, Any]) -> dict[str, Any]:
    """Expose the payload stored by the metrics-event JSONL wrapper."""
    event_name = str(row.get("event") or row.get("event_type") or row.get("type") or "")
    details = row.get("details")
    if event_name == "metrics_event_jsonl" and isinstance(details, dict):
        normalized = dict(row)
        normalized.update(details)
        return normalized
    return row


def nested_forbidden_fields(value: Any, *, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child_value in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key) in FORBIDDEN_FIELDS:
                found.append(child_path)
            found.extend(nested_forbidden_fields(child_value, path=child_path))
    elif isinstance(value, list):
        for index, child_value in enumerate(value):
            found.extend(nested_forbidden_fields(child_value, path=f"{path}[{index}]"))
    return found


def audit_runtime_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [normalized_runtime_event(row) for row in events]
    controller_events = [row for row in rows if str(row.get("event_type") or row.get("type") or "") in CONTROL_EVENT_TYPES]
    leaks: list[dict[str, Any]] = []
    disallowed_sources: list[dict[str, Any]] = []
    actual_viewport_calls = 0
    for index, row in enumerate(controller_events):
        fields = nested_forbidden_fields(row)
        if fields:
            leaks.append({"event_index": index, "event_type": row.get("event_type"), "fields": fields})
        serialized = json.dumps(row, sort_keys=True).lower()
        source = str(row.get("image_source") or row.get("source") or "").lower()
        if any(token in source for token in DISALLOWED_EVIDENCE_SOURCES):
            disallowed_sources.append({"event_index": index, "source": source})
        if "actual_isaac_viewport" in serialized or "groundingdino_b_sam2_1_hiera_large" in serialized:
            actual_viewport_calls += 1
    required_event_counts = {
        event_type: sum(str(row.get("event_type") or row.get("type") or "") == event_type for row in controller_events)
        for event_type in ("grounded_sam_request", "sensor_target_track", "sensor_planner_primitive", "primitive_command")
    }
    failures: list[str] = []
    if leaks:
        failures.append("oracle_context_leakage")
    if disallowed_sources:
        failures.append("mock_fallback_or_synthetic_source")
    if actual_viewport_calls == 0:
        failures.append("no_actual_viewport_perception_evidence")
    for event_type, count in required_event_counts.items():
        if count == 0:
            failures.append(f"missing_{event_type}")
    return {
        "pass": not failures,
        "controller_event_count": len(controller_events),
        "oracle_context_leakage": sum(len(row["fields"]) for row in leaks),
        "leaks": leaks,
        "disallowed_sources": disallowed_sources,
        "actual_viewport_evidence_count": actual_viewport_calls,
        "required_event_counts": required_event_counts,
        "failures": failures,
        "qualification_evidence": False,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def audit_run(run_dir: Path) -> dict[str, Any]:
    root_events = run_dir / "events.jsonl"
    if root_events.is_file():
        candidates = [root_events]
    else:
        candidates = [run_dir / "metrics_events.jsonl"]
        candidates.extend(sorted(run_dir.glob("**/events.jsonl")))
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        rows.extend(read_jsonl(path))
    result = audit_runtime_events(rows)
    result["run_dir"] = str(run_dir.resolve())
    result["event_files"] = [str(path) for path in seen if path.is_file()]
    return result
