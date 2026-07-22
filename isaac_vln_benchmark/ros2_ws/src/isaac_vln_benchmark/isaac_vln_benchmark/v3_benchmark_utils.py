from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def judge_route_choice_episode(record: dict[str, Any]) -> dict[str, Any]:
    expected = str(record.get("expected_route") or "").lower()
    first_turn = str(record.get("first_turn") or record.get("first_turn_action") or "none").lower()
    output = str(record.get("route_choice_output") or "").lower()
    entered_correct = bool(record.get("entered_correct_branch", False))
    if expected and first_turn in {"left", "right", "front"}:
        entered_correct = first_turn == expected
    elif expected and output in {"left", "right", "front"}:
        entered_correct = output == expected
    entered_wrong = bool(expected and first_turn in {"left", "right", "front"} and first_turn != expected)
    if entered_correct:
        failure = "none"
    elif record.get("near_intersection_blocks", 0):
        failure = "safety_block"
    elif not output or output == "scan":
        failure = "route_choice_prompt"
    elif output != expected:
        failure = "model_output"
    else:
        failure = str(record.get("failure_reason") or "unknown")
    return {
        "entered_correct_branch": entered_correct,
        "entered_wrong_branch": entered_wrong,
        "first_turn_action": first_turn,
        "yaw_after_5s": float(record.get("yaw_after_5s", record.get("yaw_5s", 0.0))),
        "yaw_after_10s": float(record.get("yaw_after_10s", record.get("yaw_10s", 0.0))),
        "route_choice_output": output or "scan",
        "near_intersection_block_events": int(record.get("near_intersection_blocks", record.get("near_intersection_block_events", 0))),
        "turn_cmd_vel_count": int(record.get("turn_cmd_vel_count", 0)),
        "failure_reason": failure,
    }


def judge_semantic_stop_episode(record: dict[str, Any]) -> dict[str, Any]:
    target_visible = bool(record.get("target_visible_at_stop", record.get("target_visible", False)))
    distance = _float(record.get("distance_at_stop", record.get("distance_m")))
    latency = _float(record.get("visible_to_stop_latency_sec"))
    max_distance = float(record.get("max_distance_to_target_m", 2.5))
    min_distance = float(record.get("min_distance_to_target_m", 0.2))
    stop = bool(record.get("stop_decision", record.get("stop", False)))
    distance_ok = bool(distance is not None and min_distance <= distance <= max_distance)
    correct = bool(stop and target_visible and distance_ok and (latency is None or latency <= 2.0))
    if correct:
        failure = "none"
    elif not target_visible:
        failure = "target_not_visible"
    elif not stop:
        failure = "never_stopped"
    elif distance is not None and distance > max_distance:
        failure = "stopped_too_early"
    elif latency is not None and latency > 2.0:
        failure = "stopped_too_late"
    else:
        failure = str(record.get("failure_reason") or "unknown")
    return {
        "stop_decision_accuracy": correct,
        "target_visible_at_stop": target_visible,
        "visible_to_stop_latency_sec": latency,
        "distance_at_stop": distance,
        "stopped_too_late": failure == "stopped_too_late",
        "stopped_too_early": failure == "stopped_too_early",
        "failure_reason": failure,
    }


def evaluate_sim2real_readiness(metrics: dict[str, Any], gate_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    gate_cfg = gate_cfg or {}
    failures: list[str] = []
    checks = {
        "route_choice_correct_branch_rate": (float(metrics.get("route_choice_correct_branch_rate", -1.0)), ">=", float(gate_cfg.get("route_choice_correct_branch_rate", 0.60))),
        "semantic_stop_accuracy": (float(metrics.get("semantic_stop_accuracy", -1.0)), ">=", float(gate_cfg.get("semantic_stop_accuracy", 0.75))),
        "visible_to_stop_latency_sec": (float(metrics.get("visible_to_stop_latency_sec", 999.0)), "<=", float(gate_cfg.get("visible_to_stop_latency_sec", 2.0))),
        "collision_count": (int(metrics.get("collision_count", 999)), "==", 0),
        "stale_action_executed": (int(metrics.get("stale_action_executed", 999)), "==", 0),
        "stale_discard_count": (int(metrics.get("stale_discard_count", 0)), "==", int(gate_cfg.get("stale_discard_count", 0))),
        "parse_error": (int(metrics.get("parse_error", 999)), "==", 0),
        "max_linear_x_mps": (float(metrics.get("max_linear_x_mps", 999.0)), "<=", float(gate_cfg.get("max_linear_x_mps", 0.20))),
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
        failures.append("mock_models is true; sim2real gate requires live Isaac evidence")
    return {
        "ready": not failures,
        "status": "READY FOR SENSOR-ONLY DRY-RUN" if not failures else "NOT READY FOR REAL ROBOT AUTONOMY",
        "failures": failures,
        "allowed_next_steps": [
            "real robot sensor-only dry-run",
            "bag replay",
            "offline Step / OmniNav scoring",
            "stationary camera validation",
        ],
        "disallowed_next_steps": [
            "real robot autonomous navigation",
            "dynamic obstacle real test",
            "semantic target approach on real robot",
        ],
    }


def parse_summary_metrics(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    metrics: dict[str, Any] = {}
    patterns = {
        "route_choice_correct_branch_rate": r"route_choice_correct_branch_rate\s*[:=]\s*([0-9.]+)",
        "semantic_stop_accuracy": r"semantic_stop_accuracy\s*[:=]\s*([0-9.]+)",
        "visible_to_stop_latency_sec": r"visible_to_stop_latency_sec\s*[:=]\s*([0-9.]+)",
        "collision_count": r"collision_count\s*[:=]\s*(\d+)",
        "stale_action_executed": r"stale_action_executed\s*[:=]\s*(\d+)",
        "stale_discard_count": r"stale_discard_count\s*[:=]\s*(\d+)",
        "parse_error": r"parse_error\s*[:=]\s*(\d+)",
        "max_linear_x_mps": r"max_linear_x_mps\s*[:=]\s*([0-9.]+)",
        "actions_through_safe_mux": r"actions_through_safe_mux\s*[:=]\s*(true|false)",
        "real_robot_motion_enabled": r"real_robot_motion_enabled\s*[:=]\s*(true|false)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1)
        if value.lower() in {"true", "false"}:
            metrics[key] = value.lower() == "true"
        elif key in {"collision_count", "stale_action_executed", "stale_discard_count", "parse_error"}:
            metrics[key] = int(value)
        else:
            metrics[key] = float(value)
    return metrics


def dump_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
