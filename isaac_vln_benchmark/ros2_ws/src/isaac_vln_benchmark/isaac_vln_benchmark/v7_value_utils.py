from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .metrics import aggregate_metrics
from .v5_timebase_utils import analyze_stale_events_v5


GOLDEN = {
    "success_count": 6,
    "episodes": 15,
    "clean_success_count": 4,
    "mean_path_m": 4.0403,
    "stale_discard_count": 12,
    "timebase_error_count": 0,
    "collision_count": 0,
    "stale_action_executed": 0,
    "max_linear_x_mps": 0.45,
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def load_jsonl(path: Path, limit: int = 500000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
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


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = headers or sorted({key for row in rows for key in row.keys()})
    if not headers:
        headers = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mode_rows(metrics: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    return [row for row in metrics.get("episodes", []) if row.get("mode") == mode]


def mode_success(metrics: dict[str, Any], mode: str) -> dict[str, Any]:
    rows = mode_rows(metrics, mode)
    agg = aggregate_metrics(rows)
    return {
        "mode": mode,
        "episodes": len(rows),
        "success_count": sum(1 for row in rows if row.get("success")),
        "clean_success_count": sum(1 for row in rows if row.get("clean_success")),
        "mean_path_m": agg.get("mean_path_length", 0.0),
        "collision_count": agg.get("collision_count", 0),
        "stale_discard_count": agg.get("stale_discard_count", 0),
        "failure_top1": agg.get("failure_top1"),
        "task_type_success": task_type_success(rows),
    }


def task_type_success(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        key = str(row.get("task_type") or "")
        bucket = result.setdefault(key, {"success": 0, "episodes": 0})
        bucket["episodes"] += 1
        if row.get("success"):
            bucket["success"] += 1
    return result


def task_success_count(metrics: dict[str, Any], mode: str, task_type: str) -> int:
    return sum(1 for row in mode_rows(metrics, mode) if row.get("task_type") == task_type and row.get("success"))


def max_linear_x_from_events(events: list[dict[str, Any]]) -> float:
    best = 0.0
    for event in events:
        details = _details(event)
        event_type = _event_type(event)
        if event_type not in {"safe_cmd_vel", "cmd_vel_candidate", "safe_cmd_mux", "primitive_command"}:
            continue
        cmd = details.get("cmd_vel") if isinstance(details.get("cmd_vel"), dict) else details
        linear = cmd.get("linear") if isinstance(cmd.get("linear"), dict) else {}
        best = max(best, abs(_float(linear.get("x"), 0.0)))
    return best


def primitive_counts_from_events(events: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "move_forward_count": 0,
        "follow_waypoint_count": 0,
        "turn_left_count": 0,
        "turn_right_count": 0,
        "stop_count": 0,
    }
    for event in events:
        details = _details(event)
        event_type = _event_type(event)
        primitive = ""
        if event_type in {"primitive_command", "omninav_action", "omninav_model_response"}:
            payload = details.get("primitive") if isinstance(details.get("primitive"), dict) else {}
            primitive = str(payload.get("primitive") or details.get("primitive") or details.get("action_type") or "")
        elif event_type in {"route_choice_bridge", "semantic_stop_bridge"}:
            payload = details.get("primitive") if isinstance(details.get("primitive"), dict) else {}
            primitive = str(payload.get("primitive") or "")
        primitive = primitive.strip()
        if primitive == "move_forward":
            counts["move_forward_count"] += 1
        elif primitive == "follow_waypoint":
            counts["follow_waypoint_count"] += 1
        elif primitive == "turn_left":
            counts["turn_left_count"] += 1
        elif primitive == "turn_right":
            counts["turn_right_count"] += 1
        elif primitive == "stop":
            counts["stop_count"] += 1
    return counts


def extract_route_decisions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        details = _details(event)
        event_type = _event_type(event)
        if event_type not in {"step_route_choice_json", "route_choice_bridge", "route_choice_bridge_parse_error", "route_choice_bridge_stale"}:
            continue
        decision = details.get("decision") if isinstance(details.get("decision"), dict) else {}
        output = details.get("output") if isinstance(details.get("output"), dict) else {}
        primitive = details.get("primitive") if isinstance(details.get("primitive"), dict) else {}
        source = decision.get("source") or output.get("source") or primitive.get("source")
        rows.append(
            {
                "mode": event.get("mode"),
                "episode_id": event.get("episode_id"),
                "t": event.get("t"),
                "event_type": event_type,
                "result": details.get("result", ""),
                "source": source or "",
                "route_choice": decision.get("route_choice") or output.get("route_choice") or primitive.get("route_choice") or "",
                "confidence": decision.get("confidence") or output.get("confidence") or "",
                "visible_in_view": decision.get("visible_in_view") or output.get("visible_in_view") or "",
                "primitive": primitive.get("primitive") or "",
                "yaw_deg": primitive.get("yaw_deg") or "",
                "error": details.get("error", ""),
            }
        )
    return rows


def extract_stop_decisions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        details = _details(event)
        event_type = _event_type(event)
        if event_type not in {"step_semantic_stop_json", "semantic_stop_bridge", "semantic_stop_bridge_parse_error", "semantic_stop_bridge_stale", "visible_to_stop_monitor"}:
            continue
        decision = details.get("decision") if isinstance(details.get("decision"), dict) else {}
        output = details.get("output") if isinstance(details.get("output"), dict) else {}
        gate = details.get("gate") if isinstance(details.get("gate"), dict) else {}
        monitor = gate.get("monitor") if isinstance(gate.get("monitor"), dict) else details
        primitive = details.get("primitive") if isinstance(details.get("primitive"), dict) else {}
        source = decision.get("source") or output.get("source") or primitive.get("source")
        rows.append(
            {
                "mode": event.get("mode"),
                "episode_id": event.get("episode_id"),
                "t": event.get("t"),
                "event_type": event_type,
                "result": details.get("result", ""),
                "source": source or "",
                "stop": decision.get("stop", output.get("stop", "")),
                "target_visible": decision.get("target_visible", output.get("target_visible", "")),
                "estimated_distance_ok": decision.get("estimated_distance_ok", output.get("estimated_distance_ok", "")),
                "confidence": decision.get("confidence") or output.get("confidence") or "",
                "visible_to_stop_latency_sec": monitor.get("visible_to_stop_latency_sec", ""),
                "distance_at_stop": monitor.get("distance_at_stop", ""),
                "force_stop": gate.get("force_stop", ""),
                "primitive": primitive.get("primitive") or "",
                "error": details.get("error", ""),
            }
        )
    return rows


def stop_metrics(stop_rows: list[dict[str, Any]]) -> dict[str, Any]:
    decision_rows = [row for row in stop_rows if row.get("event_type") in {"step_semantic_stop_json", "semantic_stop_bridge"}]
    true_rows = [row for row in decision_rows if _truthy(row.get("stop")) and _truthy(row.get("target_visible")) and _truthy(row.get("estimated_distance_ok"))]
    latencies = [_float(row.get("visible_to_stop_latency_sec"), None) for row in stop_rows]
    latencies = [value for value in latencies if value is not None]
    return {
        "stop_decision_count": len(decision_rows),
        "stop_decision_accuracy": len(true_rows) / max(1, len(decision_rows)),
        "visible_to_stop_latency_sec": max(latencies) if latencies else None,
    }


def write_trajectory_aggregate(output: Path, metrics: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for episode in metrics.get("episodes", []):
        mode = str(episode.get("mode") or "")
        task_id = str(episode.get("task_id") or "")
        path = output / mode / task_id / "trajectory.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["mode"] = mode
                row["task_id"] = task_id
                row["episode_id"] = str(episode.get("episode_id") or "")
                rows.append(row)
    write_csv(output / "trajectory.csv", rows)


def postprocess_v7_run(output: Path, *, run_kind: str) -> dict[str, Any]:
    metrics = load_json(output / "metrics.json")
    events = load_jsonl(output / "events.jsonl")
    stale = analyze_stale_events_v5(events)
    route_rows = extract_route_decisions(events)
    stop_rows = extract_stop_decisions(events)
    primitive_counts = primitive_counts_from_events(events)
    write_csv(output / "stale_attribution.csv", [{"attribution": key, "count": value} for key, value in stale["counts"].items()])
    write_csv(output / "step_route_decisions.csv", route_rows)
    write_csv(output / "step_stop_decisions.csv", stop_rows)
    write_trajectory_aggregate(output, metrics)
    parse_errors = sum(1 for row in route_rows + stop_rows if "parse_error" in str(row.get("event_type")))
    max_linear = max_linear_x_from_events(events)
    stale_count = max(int(stale.get("total_discards", 0)), sum(int(row.get("num_stale_step_results", 0) or 0) + int(row.get("num_stale_omninav_actions", 0) or 0) for row in metrics.get("episodes", [])))
    v7 = {
        "run_kind": run_kind,
        "modes": [mode_success(metrics, mode) for mode in sorted({str(row.get("mode")) for row in metrics.get("episodes", [])})],
        "max_linear_x_mps": max_linear,
        "stale_discard_count": stale_count,
        "timebase_error_count": stale["counts"].get("timebase_error", 0),
        "episode_mismatch_count": stale["counts"].get("episode_mismatch", 0),
        "missing_timestamp_count": stale["counts"].get("missing_timestamp", 0),
        "collision_count": sum(int(row.get("num_collisions", 0) or 0) for row in metrics.get("episodes", [])),
        "stale_action_executed": sum(1 for row in metrics.get("episodes", []) if row.get("failure_reason") == "stale_action_executed"),
        "parse_error_count": parse_errors,
        "route_choice_decision_count": len(route_rows),
        "route_choice_correct_branch_rate": route_choice_success_rate(metrics),
        **primitive_counts,
        **stop_metrics(stop_rows),
    }
    for key, value in primitive_counts.items():
        v7[key] = max(int(v7.get(key, 0) or 0), int(metrics.get(key, 0) or 0))
    metrics["v7_summary"] = v7
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


def evaluate_golden_smoke(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = metrics.get("episodes", [])
    agg = aggregate_metrics(rows)
    v7 = metrics.get("v7_summary", {})
    failures: list[str] = []
    if agg.get("mean_path_length", 0.0) <= 2.0:
        failures.append(f"mean_path_m={agg.get('mean_path_length', 0.0):g} <= 2")
    if int(v7.get("move_forward_count", 0) or 0) <= 0:
        failures.append("move_forward_count <= 0")
    if v7.get("max_linear_x_mps", 0.0) <= 0.0:
        failures.append("max_linear_x_mps <= 0")
    if v7.get("timebase_error_count", 0) != 0:
        failures.append(f"timebase_error_count={v7.get('timebase_error_count')} != 0")
    if v7.get("collision_count", 0) != 0:
        failures.append(f"collision_count={v7.get('collision_count')} != 0")
    if v7.get("stale_action_executed", 0) != 0:
        failures.append(f"stale_action_executed={v7.get('stale_action_executed')} != 0")
    return {
        "pass": not failures,
        "failures": failures,
        "mean_path_m": agg.get("mean_path_length", 0.0),
        "move_forward_count": int(v7.get("move_forward_count", 0) or 0),
        "max_linear_x_mps": v7.get("max_linear_x_mps", 0.0),
    }


def evaluate_forced_oracle(metrics: dict[str, Any]) -> dict[str, Any]:
    base = mode_success(metrics, "omninav_only_v6_golden")
    oracle = mode_success(metrics, "omninav_forced_route_stop_oracle_v7")
    turn_success = task_success_count(metrics, "omninav_forced_route_stop_oracle_v7", "turn_choice")
    semantic_success = task_success_count(metrics, "omninav_forced_route_stop_oracle_v7", "semantic_target")
    v7 = metrics.get("v7_summary", {})
    failures: list[str] = []
    if oracle["success_count"] < base["success_count"]:
        failures.append("oracle success below OmniNav-only")
    if oracle["clean_success_count"] < base["clean_success_count"]:
        failures.append("oracle clean below OmniNav-only")
    if turn_success <= 0 and semantic_success <= 0:
        failures.append("oracle produced no turn_choice or semantic_target success")
    if v7.get("stale_discard_count", 999) > 20:
        failures.append(f"stale_discard_count={v7.get('stale_discard_count')} > 20")
    if v7.get("collision_count", 999) != 0:
        failures.append(f"collision_count={v7.get('collision_count')} != 0")
    if v7.get("stale_action_executed", 999) != 0:
        failures.append(f"stale_action_executed={v7.get('stale_action_executed')} != 0")
    return {
        "pass": not failures,
        "failures": failures,
        "baseline": base,
        "oracle": oracle,
        "oracle_turn_choice_success": turn_success,
        "oracle_semantic_target_success": semantic_success,
    }


def evaluate_step_signal(metrics: dict[str, Any], mode: str, *, kind: str) -> dict[str, Any]:
    current = mode_success(metrics, mode)
    v7 = metrics.get("v7_summary", {})
    turn_success = task_success_count(metrics, mode, "turn_choice")
    semantic_success = task_success_count(metrics, mode, "semantic_target")
    failures: list[str] = []
    if kind == "route" and turn_success <= 0:
        failures.append("turn_choice success <= 0")
    if kind == "stop":
        if semantic_success <= 0:
            failures.append("semantic_target success <= 0")
        if float(v7.get("stop_decision_accuracy", 0.0) or 0.0) < 0.75:
            failures.append("stop_decision_accuracy < 0.75")
        latency = v7.get("visible_to_stop_latency_sec")
        if latency is None or float(latency) > 2.0:
            failures.append("visible_to_stop_latency_sec missing or > 2.0")
    if v7.get("collision_count", 999) != 0:
        failures.append("collision_count != 0")
    if v7.get("stale_action_executed", 999) != 0:
        failures.append("stale_action_executed != 0")
    return {
        "pass": not failures,
        "positive_signal": (turn_success > 0 if kind == "route" else semantic_success > 0),
        "failures": failures,
        "mode": current,
        "turn_choice_success": turn_success,
        "semantic_target_success": semantic_success,
    }


def evaluate_route_stop(metrics: dict[str, Any]) -> dict[str, Any]:
    mode = "omninav_step_route_stop_v7"
    current = mode_success(metrics, mode)
    turn_success = task_success_count(metrics, mode, "turn_choice")
    semantic_success = task_success_count(metrics, mode, "semantic_target")
    v7 = metrics.get("v7_summary", {})
    failures: list[str] = []
    if current["success_count"] < 8:
        failures.append("success_count < 8")
    if current["clean_success_count"] < 6:
        failures.append("clean_success_count < 6")
    if turn_success <= 0:
        failures.append("turn_choice success <= 0")
    if semantic_success <= 0:
        failures.append("semantic_target success <= 0")
    if v7.get("collision_count", 999) != 0:
        failures.append("collision_count != 0")
    if v7.get("stale_action_executed", 999) != 0:
        failures.append("stale_action_executed != 0")
    return {"pass": not failures, "failures": failures, "mode": current, "turn_choice_success": turn_success, "semantic_target_success": semantic_success}


def evaluate_sim2real_v7(metrics: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    gate = gate.get("required", gate) if isinstance(gate.get("required"), dict) else gate
    v7 = metrics.get("v7_summary", {})
    failures: list[str] = []
    max_linear = float(v7.get("max_linear_x_mps", 0.0) or 0.0)
    if max_linear > float(gate.get("max_linear_x_mps_real_gate", 0.20)):
        failures.append(f"max_linear_x_mps={max_linear:g} > {float(gate.get('max_linear_x_mps_real_gate', 0.20)):g}")
    route_rate = float(v7.get("route_choice_correct_branch_rate", 0.0) or 0.0)
    if route_rate < float(gate.get("route_choice_correct_branch_rate_min", 0.60)):
        failures.append(f"route_choice_correct_branch_rate={route_rate:g} < {float(gate.get('route_choice_correct_branch_rate_min', 0.60)):g}")
    stop_acc = float(v7.get("stop_decision_accuracy", 0.0) or 0.0)
    if stop_acc < float(gate.get("semantic_stop_accuracy_min", 0.75)):
        failures.append(f"semantic_stop_accuracy={stop_acc:g} < {float(gate.get('semantic_stop_accuracy_min', 0.75)):g}")
    latency = v7.get("visible_to_stop_latency_sec")
    if latency is None or float(latency) > float(gate.get("visible_to_stop_latency_max_sec", 2.0)):
        failures.append("visible_to_stop_latency_sec missing or > gate")
    for metric_key, gate_key in (
        ("collision_count", "collision_count_required"),
        ("stale_action_executed", "stale_action_executed_required"),
        ("timebase_error_count", "timebase_error_required"),
        ("parse_error_count", "parse_error_required"),
    ):
        if int(v7.get(metric_key, 999) or 0) != int(gate.get(gate_key, 0)):
            failures.append(f"{metric_key}={v7.get(metric_key)} != {gate.get(gate_key, 0)}")
    if int(v7.get("stale_discard_count", 999) or 0) > int(gate.get("stale_discard_count_max", 0)):
        failures.append(f"stale_discard_count={v7.get('stale_discard_count')} > {gate.get('stale_discard_count_max', 0)}")
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
            "semantic target approach on real robot",
            "dynamic obstacle real test",
            "autonomous turn/route-choice real test",
        ],
    }


def route_choice_success_rate(metrics: dict[str, Any]) -> float:
    rows = [row for row in metrics.get("episodes", []) if row.get("task_type") in {"turn_choice", "turn_microbench"}]
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("success")) / max(1, len(rows))


def step_effect_counts(step_metrics: dict[str, Any], baseline_metrics: dict[str, Any] | None = None) -> dict[str, int]:
    baseline_rows = {}
    if baseline_metrics:
        baseline_rows = {str(row.get("task_id")): bool(row.get("success")) for row in baseline_metrics.get("episodes", []) if row.get("mode") == "omninav_only_v6_golden"}
    helped = hurt = no_effect = 0
    for row in step_metrics.get("episodes", []):
        if not str(row.get("mode", "")).startswith("omninav_step_"):
            continue
        key = str(row.get("task_id"))
        step_success = bool(row.get("success"))
        if key in baseline_rows:
            if step_success and not baseline_rows[key]:
                helped += 1
            elif not step_success and baseline_rows[key]:
                hurt += 1
            else:
                no_effect += 1
        else:
            no_effect += 1
    return {"helped": helped, "hurt": hurt, "no_effect": no_effect}


def render_sim2real_gate(result: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [f"# Sim2Real Readiness V7: {result['status']}", "", "## V7 Summary", json.dumps(metrics.get("v7_summary", {}), indent=2, ensure_ascii=False), "", "## Failures"]
    lines.extend([f"- {item}" for item in result["failures"]] or ["- none"])
    lines.extend(["", "## Allowed Next Steps"])
    lines.extend([f"- {item}" for item in result["allowed_next_steps"]])
    lines.extend(["", "## Disallowed Next Steps"])
    lines.extend([f"- {item}" for item in result["disallowed_next_steps"]])
    return "\n".join(lines) + "\n"


def render_v7_summary(output: Path, metrics: dict[str, Any], evaluation: dict[str, Any], *, title: str) -> str:
    modes = metrics.get("v7_summary", {}).get("modes", [])
    lines = [
        f"# {title}",
        "",
        "## V6 Golden Recap",
        "",
        f"- success: {GOLDEN['success_count']}/{GOLDEN['episodes']}",
        f"- clean: {GOLDEN['clean_success_count']}/{GOLDEN['episodes']}",
        f"- mean_path_m: {GOLDEN['mean_path_m']}",
        f"- stale_discard_count: {GOLDEN['stale_discard_count']}",
        f"- max_linear_x_mps: {GOLDEN['max_linear_x_mps']}",
        "",
        "## Modes",
        "",
    ]
    for row in modes:
        lines.append(
            f"- {row['mode']}: success={row['success_count']}/{row['episodes']}, clean={row['clean_success_count']}/{row['episodes']}, mean_path_m={float(row['mean_path_m']):.3f}, failure_top1={row.get('failure_top1')}"
        )
    lines.extend(["", "## Evaluation", "", json.dumps(evaluation, indent=2, ensure_ascii=False), "", "## Artifacts", ""])
    for name in [
        "metrics.json",
        "events.jsonl",
        "mode_table.csv",
        "failure_table.csv",
        "step_route_decisions.csv",
        "step_stop_decisions.csv",
        "stale_attribution.csv",
        "trajectory.csv",
        "sim2real_gate_v7.md",
    ]:
        lines.append(f"- {name}: {output / name}")
    return "\n".join(lines) + "\n"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float(value: Any, default: Any = 0.0) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _details(event: dict[str, Any]) -> dict[str, Any]:
    return event.get("details") if isinstance(event.get("details"), dict) else event


def _event_type(event: dict[str, Any]) -> str:
    details = _details(event)
    return str(details.get("event_type") or event.get("event_type") or event.get("event") or "")
