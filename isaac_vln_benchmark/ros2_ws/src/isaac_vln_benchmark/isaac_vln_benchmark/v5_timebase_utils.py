from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any


V5_ATTRIBUTIONS = [
    "valid",
    "stale_due_to_age",
    "stale_due_to_pose_delta",
    "timebase_error",
    "episode_mismatch",
    "missing_timestamp",
    "queue_delay",
    "old_response_after_reset",
    "unknown",
]

TIMEBASE_COLUMNS = [
    "episode_id",
    "request_id",
    "clock_domain",
    "ros_now",
    "wall_now",
    "clock_msg_time",
    "image_header_stamp",
    "frame_bundle_timestamp",
    "request_start_time",
    "model_response_time",
    "action_timestamp",
    "stale_gate_current_time",
    "computed_age_sec",
    "pose_delta_m",
    "discard",
    "discard_reason",
]


def make_timebase_probe_rows(count: int = 10, *, episode_id: str = "timebase_probe_ep0") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_wall = time.time()
    for index in range(count):
        request_t = base_wall + index * 0.05
        response_t = request_t + 0.01
        stale_gate_t = response_t + 0.02
        rows.append(
            {
                "episode_id": episode_id,
                "request_id": f"timebase_probe_{index:02d}",
                "clock_domain": "wall",
                "ros_now": stale_gate_t,
                "wall_now": stale_gate_t,
                "clock_msg_time": stale_gate_t,
                "image_header_stamp": request_t,
                "frame_bundle_timestamp": request_t,
                "request_start_time": request_t,
                "model_response_time": response_t,
                "action_timestamp": response_t,
                "stale_gate_current_time": stale_gate_t,
                "computed_age_sec": round(stale_gate_t - response_t, 6),
                "pose_delta_m": 0.0,
                "discard": False,
                "discard_reason": "valid",
            }
        )
    return rows


def evaluate_timebase_probe(rows: list[dict[str, Any]], *, max_reasonable_age_sec: float = 1.0) -> dict[str, Any]:
    analysis = analyze_stale_rows(rows)
    ages = [_float(row.get("computed_age_sec"), math.nan) for row in rows]
    bad_age = [age for age in ages if not math.isfinite(age) or age < -1e-6 or age > max_reasonable_age_sec]
    discard_count = sum(1 for row in rows if bool(row.get("discard")))
    metrics = {
        "benchmark": "stale_gate_timebase_probe",
        "samples": len(rows),
        "discard_count": discard_count,
        "stale_discard_count": discard_count,
        "timebase_error_count": analysis["counts"].get("timebase_error", 0),
        "episode_mismatch_count": analysis["counts"].get("episode_mismatch", 0),
        "missing_timestamp_count": analysis["counts"].get("missing_timestamp", 0),
        "bad_age_count": len(bad_age),
        "max_computed_age_sec": max([age for age in ages if math.isfinite(age)] or [0.0]),
        "pass": bool(
            rows
            and discard_count == 0
            and analysis["counts"].get("timebase_error", 0) == 0
            and analysis["counts"].get("episode_mismatch", 0) == 0
            and analysis["counts"].get("missing_timestamp", 0) == 0
            and not bad_age
        ),
    }
    return metrics | {"stale_attribution": analysis}


def make_single_primitive_result(
    *,
    primitive: str = "move_forward",
    path_m: float = 0.42,
    yaw_delta_deg: float = 0.0,
    episode_id: str = "single_primitive_ep0",
) -> dict[str, Any]:
    turn = primitive.startswith("turn")
    return {
        "benchmark": "single_primitive_probe",
        "episode_id": episode_id,
        "primitive": primitive,
        "path_m": 0.0 if turn else path_m,
        "yaw_delta_deg": yaw_delta_deg if turn else 0.0,
        "collision_count": 0,
        "stale_discard_count": 0,
        "timebase_error_count": 0,
        "episode_mismatch_count": 0,
        "missing_timestamp_count": 0,
        "stale_action_executed": 0,
        "actions_through_safe_mux": True,
        "pass": bool((turn and yaw_delta_deg > 20.0) or ((not turn) and path_m > 0.3)),
    }


def evaluate_single_primitive(metrics: dict[str, Any]) -> dict[str, Any]:
    primitive = str(metrics.get("primitive") or "move_forward")
    turn = primitive.startswith("turn")
    motion_ok = _float(metrics.get("yaw_delta_deg"), 0.0) > 20.0 if turn else _float(metrics.get("path_m"), 0.0) > 0.3
    failures: list[str] = []
    if int(metrics.get("stale_discard_count", 0) or 0) != 0:
        failures.append("stale_discard_count != 0")
    if int(metrics.get("timebase_error_count", 0) or 0) != 0:
        failures.append("timebase_error_count != 0")
    if int(metrics.get("episode_mismatch_count", 0) or 0) != 0:
        failures.append("episode_mismatch_count != 0")
    if int(metrics.get("collision_count", 0) or 0) != 0:
        failures.append("collision_count != 0")
    if not motion_ok:
        failures.append("motion threshold not reached")
    return metrics | {"pass": not failures, "failures": failures}


def analyze_stale_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter({key: 0 for key in V5_ATTRIBUTIONS})
    records: list[dict[str, Any]] = []
    for row in rows:
        attr = classify_attribution(row)
        counts[attr] += 1
        if attr != "valid" or bool(row.get("discard")):
            records.append({"attribution": attr, "record": row})
    return {"total_discards": sum(1 for row in rows if bool(row.get("discard"))), "counts": dict(counts), "records": records}


def analyze_stale_events_v5(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter({key: 0 for key in V5_ATTRIBUTIONS})
    records: list[dict[str, Any]] = []
    for event in events:
        attr = classify_attribution(event)
        details = event.get("details") if isinstance(event.get("details"), dict) else event
        event_type = str(details.get("event_type") or event.get("event_type") or event.get("event") or "")
        is_discard = bool(details.get("discard") or details.get("stale") or event_type in {"step_response_stale", "omninav_stale", "internnav_stale", "stale_result"})
        if attr == "valid" and not is_discard:
            continue
        counts[attr] += 1
        records.append({"attribution": attr, "event": event})
    return {"total_discards": len(records), "counts": dict(counts), "records": records}


def classify_attribution(record: dict[str, Any]) -> str:
    details = record.get("details") if isinstance(record.get("details"), dict) else record
    attr = str(details.get("attribution") or details.get("discard_reason") or details.get("reason") or "").strip()
    if attr in V5_ATTRIBUTIONS:
        return attr
    attr_lower = attr.lower()
    if attr_lower in {"timestamp_mismatch", "clock_domain_mismatch", "clock_mismatch", "timebase_mismatch"}:
        return "timebase_error"
    if bool(details.get("timebase_error")):
        return "timebase_error"
    text = json.dumps(record, ensure_ascii=False).lower()
    if "timestamp_mismatch" in text or '"timebase_error"' in text:
        return "timebase_error"
    if not _looks_like_discard_or_stale(details):
        return "valid"
    if "episode_mismatch" in text:
        return "episode_mismatch"
    if "missing_timestamp" in text:
        return "missing_timestamp"
    if "old_response_after_reset" in text:
        return "old_response_after_reset"
    if "queue" in text:
        return "queue_delay"
    if "pose_delta" in text or "robot_motion" in text or "yaw_delta" in text:
        return "stale_due_to_pose_delta"
    if "timeout" in text or "ttl" in text or "age" in text or "stale_omninav" in text:
        return "stale_due_to_age"
    if bool(details.get("discard")):
        return "unknown"
    return "valid"


def _looks_like_discard_or_stale(details: dict[str, Any]) -> bool:
    event_type = str(details.get("event_type") or details.get("event") or "").lower()
    result = str(details.get("result") or "").lower()
    if bool(details.get("discard") or details.get("stale")):
        return True
    if event_type in {"step_response_stale", "omninav_stale", "internnav_stale", "stale_result"}:
        return True
    if event_type.endswith("_stale"):
        return True
    return result in {"discarded", "timeout", "expired", "stale", "dropped"}


def write_v5_run_artifacts(
    output: str | Path,
    *,
    title: str,
    metrics: dict[str, Any],
    events: list[dict[str, Any]] | None = None,
    timebase_rows: list[dict[str, Any]] | None = None,
    trajectory_rows: list[dict[str, Any]] | None = None,
    mode_rows: list[dict[str, Any]] | None = None,
    failure_rows: list[dict[str, Any]] | None = None,
) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    events = events or []
    timebase_rows = timebase_rows or []
    trajectory_rows = trajectory_rows or []
    mode_rows = mode_rows or [{"mode": metrics.get("benchmark", title), "episodes": metrics.get("episodes", metrics.get("samples", 1)), "pass": metrics.get("pass", False)}]
    failure_rows = failure_rows or _failure_rows_from_metrics(metrics)
    stale_analysis = analyze_stale_rows(timebase_rows) if timebase_rows else analyze_stale_events_v5(events)

    (output / "metrics.json").write_text(json.dumps(metrics | {"stale_attribution": stale_analysis}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_jsonl(output / "events.jsonl", events)
    _write_csv(output / "timebase_table.csv", TIMEBASE_COLUMNS, [_normal_timebase_row(row) for row in timebase_rows])
    _write_csv(output / "stale_attribution.csv", ["attribution", "count"], [{"attribution": key, "count": stale_analysis["counts"].get(key, 0)} for key in V5_ATTRIBUTIONS])
    _write_csv(output / "mode_table.csv", sorted({key for row in mode_rows for key in row.keys()} or {"mode", "episodes", "pass"}), mode_rows)
    _write_csv(output / "failure_table.csv", sorted({key for row in failure_rows for key in row.keys()} or {"failure_reason", "count"}), failure_rows)
    _write_csv(output / "trajectory.csv", ["episode_id", "t", "x", "y", "yaw", "source"], [_normal_traj_row(row) for row in trajectory_rows])
    (output / "summary.md").write_text(_summary_text(title, output, metrics, stale_analysis), encoding="utf-8")


def evaluate_sim2real_readiness_v5(metrics: dict[str, Any], gate_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    gate_cfg = gate_cfg or {}
    failures: list[str] = []

    def value_for(*names: str) -> Any:
        for name in names:
            if name in metrics and metrics[name] is not None:
                return metrics[name]
        return None

    def compare(name: str, actual: Any, op: str, expected: int | float) -> None:
        if actual is None:
            failures.append(f"{name}=missing/unverified")
            return
        if op == "==" and not actual == expected:
            failures.append(f"{name}={actual} != {expected}")
        elif op == "<=" and not actual <= expected:
            failures.append(f"{name}={actual} > {expected}")
        elif op == ">=" and not actual >= expected:
            failures.append(f"{name}={actual} < {expected}")

    exact_zero_checks = (
        "timebase_error_count",
        "episode_mismatch_count",
        "missing_timestamp_count",
        "collision_count",
        "stale_action_executed",
    )
    for name in exact_zero_checks:
        actual = value_for(name)
        compare(name, int(actual) if actual is not None else None, "==", 0)

    stale_discard_count = value_for("stale_discard_count")
    compare(
        "stale_discard_count",
        int(stale_discard_count) if stale_discard_count is not None else None,
        "<=",
        int(gate_cfg.get("stale_discard_count", 0)),
    )
    compare(
        "route_choice_correct_branch_rate",
        None
        if value_for("route_choice_correct_branch_rate", "step_route_choice_correct_branch_rate") is None
        else _float(value_for("route_choice_correct_branch_rate", "step_route_choice_correct_branch_rate"), -1.0),
        ">=",
        _float(gate_cfg.get("route_choice_correct_branch_rate"), 0.60),
    )
    compare(
        "semantic_stop_accuracy",
        None
        if value_for("semantic_stop_accuracy", "step_semantic_stop_accuracy") is None
        else _float(value_for("semantic_stop_accuracy", "step_semantic_stop_accuracy"), -1.0),
        ">=",
        _float(gate_cfg.get("semantic_stop_accuracy"), 0.75),
    )
    compare(
        "visible_to_stop_latency_sec",
        None if value_for("visible_to_stop_latency_sec") is None else _float(value_for("visible_to_stop_latency_sec"), 999.0),
        "<=",
        _float(gate_cfg.get("visible_to_stop_latency_sec"), 2.0),
    )
    parse_error = value_for("parse_error", "parse_errors")
    compare("parse_error", int(parse_error) if parse_error is not None else None, "==", 0)
    max_linear_x = value_for("max_linear_x_mps")
    compare(
        "max_linear_x_mps",
        _float(max_linear_x, 999.0) if max_linear_x is not None else None,
        "<=",
        _float(gate_cfg.get("max_linear_x_mps"), 0.20),
    )

    safe_mux_results = metrics.get("safe_mux_results") if isinstance(metrics.get("safe_mux_results"), dict) else {}
    went_through_safe_mux = bool(metrics.get("actions_through_safe_mux", False)) or int(safe_mux_results.get("accepted", 0)) > 0
    if not went_through_safe_mux:
        failures.append("actions_through_safe_mux=missing/unverified")
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


def _summary_text(title: str, output: Path, metrics: dict[str, Any], stale_analysis: dict[str, Any]) -> str:
    lines = [
        f"# {title}",
        "",
        f"- run_dir: {output}",
        f"- pass: {metrics.get('pass', False)}",
        f"- stale_discard_count: {metrics.get('stale_discard_count', metrics.get('discard_count', 0))}",
        f"- timebase_error_count: {metrics.get('timebase_error_count', stale_analysis['counts'].get('timebase_error', 0))}",
        f"- episode_mismatch_count: {metrics.get('episode_mismatch_count', stale_analysis['counts'].get('episode_mismatch', 0))}",
        f"- missing_timestamp_count: {metrics.get('missing_timestamp_count', stale_analysis['counts'].get('missing_timestamp', 0))}",
        "",
        "## Stale Attribution",
        "",
        "| attribution | count |",
        "| --- | ---: |",
    ]
    for key in V5_ATTRIBUTIONS:
        lines.append(f"| {key} | {stale_analysis['counts'].get(key, 0)} |")
    if metrics.get("failures"):
        lines.extend(["", "## Failures", ""])
        lines.extend([f"- {failure}" for failure in metrics["failures"]])
    return "\n".join(lines) + "\n"


def _normal_timebase_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key, "") for key in TIMEBASE_COLUMNS}


def _normal_traj_row(row: dict[str, Any]) -> dict[str, Any]:
    pose = row.get("pose") or [row.get("x", 0.0), row.get("y", 0.0), row.get("yaw", 0.0)]
    return {
        "episode_id": row.get("episode_id", ""),
        "t": row.get("t", 0.0),
        "x": pose[0] if len(pose) > 0 else 0.0,
        "y": pose[1] if len(pose) > 1 else 0.0,
        "yaw": pose[2] if len(pose) > 2 else 0.0,
        "source": row.get("source", ""),
    }


def _failure_rows_from_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    failures = metrics.get("failures")
    if isinstance(failures, list) and failures:
        return [{"failure_reason": str(item), "count": 1} for item in failures]
    if metrics.get("pass", False):
        return [{"failure_reason": "none", "count": 0}]
    return [{"failure_reason": str(metrics.get("failure_reason") or "unknown"), "count": 1}]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    headers = list(headers)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number
