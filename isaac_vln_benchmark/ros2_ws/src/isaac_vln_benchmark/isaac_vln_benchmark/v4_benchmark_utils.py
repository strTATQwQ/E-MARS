from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .v3_benchmark_utils import dump_json


STALE_CATEGORIES = {
    "stale_step_route_choice",
    "stale_step_stop_verify",
    "stale_omninav_waypoint",
    "stale_due_to_queue_delay",
    "stale_due_to_robot_motion",
    "stale_due_to_timestamp_mismatch",
    "stale_due_to_timeout",
    "unknown",
}


def judge_forced_route_episode(record: dict[str, Any]) -> dict[str, Any]:
    expected = str(record.get("expected_route") or "").lower()
    first_turn = str(record.get("first_turn_action") or record.get("first_turn") or "none").lower()
    yaw_5 = _float(record.get("yaw_after_5s", record.get("yaw_5s")), 0.0)
    yaw_10 = _float(record.get("yaw_after_10s", record.get("yaw_10s")), 0.0)
    final_y = _float(record.get("final_y", record.get("y_after_20s")), 0.0)
    if first_turn == "none":
        if yaw_5 > 0.05 or yaw_10 > 0.05:
            first_turn = "left"
        elif yaw_5 < -0.05 or yaw_10 < -0.05:
            first_turn = "right"
    entered_correct = bool(
        (expected == "left" and (final_y > 0.20 or yaw_10 > 0.15 or first_turn == "left"))
        or (expected == "right" and (final_y < -0.20 or yaw_10 < -0.15 or first_turn == "right"))
        or (expected == "front" and abs(final_y) <= 0.20 and first_turn in {"front", "none"})
    )
    wrong_branch = bool(
        (expected == "left" and (final_y < -0.20 or first_turn == "right"))
        or (expected == "right" and (final_y > 0.20 or first_turn == "left"))
    )
    no_turn = bool(expected in {"left", "right"} and abs(yaw_10) < 0.05 and first_turn in {"front", "none"})
    safety_block = int(record.get("safety_block_events", record.get("safety_block", 0)) or 0)
    stale_discards = int(record.get("stale_discards", record.get("stale_discard_count", 0)) or 0)
    if entered_correct:
        failure = "none"
    elif safety_block:
        failure = "safety_block"
    elif no_turn:
        failure = "primitive_mapping_error"
    elif wrong_branch:
        failure = "yaw_controller_error"
    elif stale_discards:
        failure = "stale_gate"
    else:
        failure = str(record.get("failure_reason") or "unknown")
    return {
        "first_turn_action": first_turn,
        "entered_correct_branch": entered_correct,
        "wrong_branch": wrong_branch,
        "no_turn": no_turn,
        "yaw_after_5s": yaw_5,
        "yaw_after_10s": yaw_10,
        "failure_reason": failure,
    }


def judge_forced_stop_episode(record: dict[str, Any], *, max_latency_sec: float = 1.5) -> dict[str, Any]:
    target_visible = bool(record.get("target_visible_at_stop", record.get("target_visible", False)))
    distance = _float(record.get("distance_at_stop", record.get("distance_to_target")), 999.0)
    latency = _float(record.get("visible_to_stop_latency_sec"), 999.0)
    stop_seen = bool(record.get("stop_cmd_received", record.get("stop_decision", record.get("stop", False))))
    distance_ok = 0.0 <= distance <= _float(record.get("max_distance_to_target_m"), 2.5)
    correct = bool(stop_seen and target_visible and distance_ok and latency <= max_latency_sec)
    if correct:
        failure = "none"
    elif not target_visible:
        failure = "visibility_status_error"
    elif not stop_seen:
        failure = "cmd_vel_stop_error"
    elif not distance_ok:
        failure = "distance_alias_error"
    elif latency > max_latency_sec:
        failure = "stop_gate_error"
    else:
        failure = str(record.get("failure_reason") or "unknown")
    return {
        "stop_decision_accuracy": correct,
        "target_visible_at_stop": target_visible,
        "distance_at_stop": distance,
        "visible_to_stop_latency_sec": latency,
        "stopped_too_late": latency > max_latency_sec,
        "stopped_too_early": bool(record.get("stopped_too_early", False)),
        "never_stopped": not stop_seen,
        "failure_reason": failure,
    }


def summarize_route_rows(rows: list[dict[str, Any]], *, target_rate: float = 0.80) -> dict[str, Any]:
    judged = [dict(row) | judge_forced_route_episode(row) for row in rows]
    correct = sum(1 for row in judged if row["entered_correct_branch"])
    collisions = sum(int(row.get("collision_count", row.get("num_collisions", 0)) or 0) for row in judged)
    stale_executed = sum(int(row.get("stale_action_executed", 0) or 0) for row in judged)
    stale_discards = sum(int(row.get("stale_discards", row.get("stale_discard_count", 0)) or 0) for row in judged)
    failures = [row["failure_reason"] for row in judged if row["failure_reason"] != "none"]
    rate = correct / len(judged) if judged else 0.0
    return {
        "episodes": len(judged),
        "correct_branch_count": correct,
        "correct_branch_rate": round(rate, 3),
        "target_correct_branch_rate": target_rate,
        "collision_count": collisions,
        "stale_action_executed": stale_executed,
        "stale_discard_count": stale_discards,
        "pass": bool(rate >= target_rate and collisions == 0 and stale_executed == 0),
        "failure_top1": _top1(failures),
        "rows": judged,
    }


def summarize_stop_rows(rows: list[dict[str, Any]], *, target_acc: float = 0.90, target_latency_sec: float = 1.5) -> dict[str, Any]:
    judged = [dict(row) | judge_forced_stop_episode(row, max_latency_sec=target_latency_sec) for row in rows]
    correct = sum(1 for row in judged if row["stop_decision_accuracy"])
    latencies = [float(row["visible_to_stop_latency_sec"]) for row in judged if row["visible_to_stop_latency_sec"] < 900]
    mean_latency = sum(latencies) / len(latencies) if latencies else 999.0
    p95_latency = _nearest_rank_percentile(latencies, 0.95) if latencies else 999.0
    collisions = sum(int(row.get("collision_count", row.get("num_collisions", 0)) or 0) for row in judged)
    stale_executed = sum(int(row.get("stale_action_executed", 0) or 0) for row in judged)
    stale_discards = sum(int(row.get("stale_discards", row.get("stale_discard_count", 0)) or 0) for row in judged)
    failures = [row["failure_reason"] for row in judged if row["failure_reason"] != "none"]
    acc = correct / len(judged) if judged else 0.0
    return {
        "episodes": len(judged),
        "stop_correct_count": correct,
        "stop_decision_accuracy": round(acc, 3),
        "visible_to_stop_latency_sec": round(mean_latency, 3),
        "visible_to_stop_latency_p95_sec": round(p95_latency, 3),
        "target_stop_decision_accuracy": target_acc,
        "target_visible_to_stop_latency_sec": target_latency_sec,
        "collision_count": collisions,
        "stale_action_executed": stale_executed,
        "stale_discard_count": stale_discards,
        "pass": bool(acc >= target_acc and p95_latency <= target_latency_sec and collisions == 0 and stale_executed == 0),
        "failure_top1": _top1(failures),
        "rows": judged,
    }


def _nearest_rank_percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 999.0
    rank = max(1, math.ceil(max(0.0, min(1.0, quantile)) * len(ordered)))
    return ordered[rank - 1]


def categorize_stale_event(event: dict[str, Any]) -> str:
    text = json.dumps(event, ensure_ascii=False).lower()
    details = event.get("details") if isinstance(event.get("details"), dict) else event
    model = str(details.get("model") or event.get("model") or "").lower() if isinstance(details, dict) else ""
    role = str(details.get("role") or details.get("event_type") or details.get("reason") or "").lower() if isinstance(details, dict) else ""
    if "route_choice" in text or role == "route_choice":
        return "stale_step_route_choice"
    if "semantic_stop" in text or "stop_verify" in text or role == "semantic_stop":
        return "stale_step_stop_verify"
    if model == "omninav" or "omninav" in text:
        return "stale_omninav_waypoint"
    if "queue" in text or "inflight" in text:
        return "stale_due_to_queue_delay"
    if "pose_delta" in text or "yaw_delta" in text or "robot_motion" in text:
        return "stale_due_to_robot_motion"
    if "timestamp" in text or "clock" in text:
        return "stale_due_to_timestamp_mismatch"
    if "timeout" in text or "ttl" in text or "age" in text:
        return "stale_due_to_timeout"
    return "unknown"


def analyze_stale_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    stale = []
    counts = {key: 0 for key in sorted(STALE_CATEGORIES)}
    for event in events:
        text = json.dumps(event, ensure_ascii=False).lower()
        if "stale" not in text:
            continue
        category = categorize_stale_event(event)
        counts[category] += 1
        stale.append({"category": category, "event": event})
    return {"total_stale_discards": len(stale), "counts": counts, "records": stale}


def evaluate_sim2real_readiness_v4(metrics: dict[str, Any], gate_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    gate_cfg = gate_cfg or {}
    failures: list[str] = []
    checks = {
        "forced_route_oracle_correct_branch_rate": (
            _float(metrics.get("forced_route_oracle_correct_branch_rate"), -1.0),
            ">=",
            _float(gate_cfg.get("forced_route_oracle_correct_branch_rate"), 0.80),
        ),
        "forced_stop_oracle_accuracy": (
            _float(metrics.get("forced_stop_oracle_accuracy"), -1.0),
            ">=",
            _float(gate_cfg.get("forced_stop_oracle_accuracy"), 0.90),
        ),
        "step_route_choice_correct_branch_rate": (
            _float(metrics.get("step_route_choice_correct_branch_rate"), -1.0),
            ">=",
            _float(gate_cfg.get("step_route_choice_correct_branch_rate"), 0.60),
        ),
        "step_semantic_stop_accuracy": (
            _float(metrics.get("step_semantic_stop_accuracy"), -1.0),
            ">=",
            _float(gate_cfg.get("step_semantic_stop_accuracy"), 0.75),
        ),
        "visible_to_stop_latency_sec": (
            _float(metrics.get("visible_to_stop_latency_sec"), 999.0),
            "<=",
            _float(gate_cfg.get("visible_to_stop_latency_sec"), 2.0),
        ),
        "collision_count": (int(metrics.get("collision_count", 999)), "==", 0),
        "stale_action_executed": (int(metrics.get("stale_action_executed", 999)), "==", 0),
        "stale_discard_count": (int(metrics.get("stale_discard_count", 999)), "==", int(gate_cfg.get("stale_discard_count", 0))),
        "parse_error": (int(metrics.get("parse_error", 999)), "==", 0),
        "max_linear_x_mps": (
            _float(metrics.get("max_linear_x_mps"), 999.0),
            "<=",
            _float(gate_cfg.get("max_linear_x_mps"), 0.20),
        ),
    }
    for name, (actual, op, expected) in checks.items():
        if op == ">=" and not actual >= expected:
            failures.append(f"{name}={actual} < {expected}")
        elif op == "<=" and not actual <= expected:
            failures.append(f"{name}={actual} > {expected}")
        elif op == "==" and not actual == expected:
            failures.append(f"{name}={actual} != {expected}")
    if bool(metrics.get("actions_through_safe_mux", False)) is not True:
        failures.append("actions_through_safe_mux is not true")
    if bool(metrics.get("real_robot_motion_enabled", True)) is not False:
        failures.append("real_robot_motion_enabled is not false")
    if bool(metrics.get("mock_models", False)) is True:
        failures.append("mock_models is true; v4 sim2real gate requires live Isaac evidence")
    return {
        "ready": not failures,
        "status": "READY FOR SENSOR-ONLY DRY-RUN" if not failures else "NOT READY FOR REAL ROBOT AUTONOMY",
        "failures": failures,
        "allowed_next_steps": [
            "sensor-only dry-run",
            "bag replay",
            "offline scoring",
            "stationary camera validation",
            "manual-triggered primitive test",
        ],
        "disallowed_next_steps": [
            "real robot autonomous navigation",
            "dynamic obstacle real test",
            "semantic target approach on real robot",
        ],
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_stale_analysis(path: str | Path, analysis: dict[str, Any]) -> None:
    lines = [
        "# Stale Discard Analysis",
        "",
        f"- total_stale_discards: {analysis.get('total_stale_discards', 0)}",
        "",
        "| category | count |",
        "| --- | ---: |",
    ]
    counts = analysis.get("counts") if isinstance(analysis.get("counts"), dict) else {}
    for category in sorted(STALE_CATEGORIES):
        lines.append(f"| {category} | {int(counts.get(category, 0))} |")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(number) or math.isinf(number):
        return float(default)
    return number


def _top1(values: list[str]) -> str:
    if not values:
        return "none"
    return max(set(values), key=values.count)
