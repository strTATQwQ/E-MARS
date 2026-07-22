from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def event_details(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("details")
    return value if isinstance(value, dict) else {}


def evaluate_transport_profile(
    run_dir: Path,
    *,
    profile: str,
    requested_delay_sec: float,
    requested_drop_rate: float,
    reset_expected: bool = False,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    episodes = metrics.get("episodes") if isinstance(metrics.get("episodes"), list) else []
    events = load_jsonl(run_dir / "events.jsonl")
    audits = [event_details(row) for row in events if row.get("event") == "benchmark_delay_audit_json"]
    raw = [
        event_details(row)
        for row in events
        if str(row.get("event") or "").startswith("stress_raw_step_")
    ]
    resets = [event_details(row) for row in events if row.get("event") == "benchmark_reset_stress_json"]
    metric_events = [event_details(row) for row in events if row.get("event") == "metrics_event_jsonl"]
    stale = [
        row
        for row in metric_events
        if str(row.get("event_type") or "") in {"route_choice_bridge_stale", "semantic_stop_bridge_stale"}
        and str(row.get("result") or "") == "discarded"
    ]
    primitives = [event_details(row) for row in events if row.get("event") == "primitive_command_json"]
    dropped = [row for row in audits if bool(row.get("dropped"))]
    delivered = [row for row in audits if not bool(row.get("dropped"))]
    protected_ids = {str(row.get("request_id") or "") for row in dropped + stale if row.get("request_id")}
    primitive_ids = {str(row.get("request_id") or "") for row in primitives if row.get("request_id")}
    forbidden_primitive_ids = sorted(protected_ids & primitive_ids)
    collision_count = sum(int(row.get("num_collisions", 0) or 0) for row in episodes)
    stale_action_executed = sum(1 for row in episodes if float(row.get("stale_action_rate", 0.0) or 0.0) > 0.0)
    timebase_error = sum(1 for row in metric_events if str(row.get("attribution") or "") == "timebase_error")
    parse_error = sum(1 for row in metric_events if "parse_error" in str(row.get("event_type") or ""))
    max_delay = max((float(row.get("delay_sec", 0.0) or 0.0) for row in delivered), default=0.0)
    reset_stale = [row for row in stale if str(row.get("attribution") or "") == "old_response_after_reset"]
    live_step_observed = bool(raw) or any(int(row.get("num_step_calls", 0) or 0) > 0 for row in episodes)
    checks = {
        "metrics_present": bool(episodes),
        "live_step_observed": live_step_observed,
        "delay_audit_observed": bool(audits),
        "collision_zero": collision_count == 0,
        "stale_action_executed_zero": stale_action_executed == 0,
        "timebase_error_zero": timebase_error == 0,
        "parse_error_zero": parse_error == 0,
        "protected_requests_never_became_primitives": not forbidden_primitive_ids,
    }
    if requested_drop_rate >= 1.0:
        checks["deterministic_drop_observed"] = bool(dropped) and not delivered
    else:
        checks["delivery_observed"] = bool(delivered)
        checks["requested_delay_observed"] = max_delay + 0.03 >= requested_delay_sec
    if reset_expected:
        checks["reset_injected"] = bool(resets)
        checks["old_response_after_reset_discarded"] = bool(reset_stale)
    else:
        checks["reset_not_injected"] = not resets
    if profile == "latency":
        checks["task_success_under_mild_delay"] = bool(episodes) and all(bool(row.get("success")) for row in episodes)
    result = {
        "schema_version": 1,
        "profile": profile,
        "pass": all(checks.values()),
        "checks": checks,
        "requested_delay_sec": requested_delay_sec,
        "requested_drop_rate": requested_drop_rate,
        "raw_decision_count": len(raw),
        "delivered_count": len(delivered),
        "dropped_count": len(dropped),
        "max_observed_delay_sec": max_delay,
        "reset_count": len(resets),
        "reset_stale_discard_count": len(reset_stale),
        "forbidden_primitive_request_ids": forbidden_primitive_ids,
        "collision_count": collision_count,
        "stale_action_executed": stale_action_executed,
        "timebase_error": timebase_error,
        "parse_error": parse_error,
        "episode_results": [
            {key: row.get(key) for key in ("episode_id", "task_type", "success", "failure_reason")}
            for row in episodes
        ],
    }
    return result


def evaluate_transport_stress(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    names = {str(row.get("profile") or "") for row in profiles}
    required = {"latency", "packet_loss", "reset"}
    return {
        "schema_version": 1,
        "pass": required.issubset(names) and all(bool(row.get("pass")) for row in profiles),
        "required_profiles": sorted(required),
        "observed_profiles": sorted(names),
        "profiles": profiles,
        "real_robot_motion_enabled": False,
    }


def write_transport_stress(output: Path, result: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "stress_gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Decision Transport Stress", "", f"- pass: {result['pass']}", "- real robot motion enabled: false", ""]
    for row in result["profiles"]:
        lines.append(f"## {row['profile']}")
        lines.append("")
        lines.append(f"- pass: {row['pass']}")
        lines.append(f"- delivered/dropped: {row['delivered_count']}/{row['dropped_count']}")
        lines.append(f"- max delay: {row['max_observed_delay_sec']:.3f}s")
        lines.append(f"- reset stale discards: {row['reset_stale_discard_count']}")
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
