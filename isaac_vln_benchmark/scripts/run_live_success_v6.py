#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.v5_timebase_utils import analyze_stale_events_v5
from isaac_vln_benchmark.v6_recovery_utils import (
    classify_first10_root_cause,
    evaluate_omninav_regression_v6,
    write_csv,
    write_jsonl,
)


def load_jsonl(path: Path, limit: int = 250000) -> list[dict[str, Any]]:
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


def ensure_v6_artifacts(output: Path) -> dict[str, Any]:
    events = load_jsonl(output / "events.jsonl")
    stale = analyze_stale_events_v5(events)
    action_rows = action_trace_from_events(events)
    safe_rows = safe_trace_from_events(events)
    first10_root = classify_first10_root_cause(action_rows[:500] + safe_rows[:500])
    metrics = load_metrics(output)
    metrics.update(aggregate_v6_metrics(metrics, events, action_rows, safe_rows, stale))
    evaluated = evaluate_omninav_regression_v6(metrics)
    metrics.update(evaluated)
    metrics["first10_root_cause"] = first10_root
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(output / "action_parse_trace.jsonl", action_rows)
    write_jsonl(output / "safe_mux_trace.jsonl", safe_rows)
    write_csv(output / "stale_attribution.csv", [{"attribution": key, "count": value} for key, value in stale["counts"].items()])
    if not (output / "timebase_table.csv").exists():
        write_csv(output / "timebase_table.csv", [])
    if not metrics.get("pass", False):
        write_root_cause(output / "REGRESSION_ROOT_CAUSE.md", metrics, first10_root)
    return metrics


def load_metrics(output: Path) -> dict[str, Any]:
    path = output / "metrics.json"
    metrics: dict[str, Any] = {}
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                metrics.update(value)
        except json.JSONDecodeError:
            pass
    summary_path = output / "summary_aggregate.json"
    if summary_path.exists():
        try:
            summary_value = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary_value = {}
        for row in summary_value.get("summary", []) if isinstance(summary_value, dict) else []:
            if "omninav_only" not in str(row.get("mode", "")):
                continue
            metrics["success_count"] = int(float(row.get("success", 0) or 0))
            metrics["completed_episodes"] = int(float(row.get("episodes", 0) or 0))
            metrics["mean_path_m"] = _float(row.get("mean_path_m"), metrics.get("mean_path_m", 0.0))
            metrics["clean_success_count"] = int(round(_float(row.get("clean_success_rate"), 0.0) * metrics["completed_episodes"]))
            break
    aggregate = metrics.get("aggregate_by_mode") if isinstance(metrics.get("aggregate_by_mode"), dict) else {}
    omni = aggregate.get("omninav_only") if isinstance(aggregate.get("omninav_only"), dict) else {}
    if omni:
        metrics.setdefault("completed_episodes", int(float(omni.get("episodes", 0) or 0)))
        metrics.setdefault("mean_path_m", _float(omni.get("mean_path_length"), 0.0))
        metrics.setdefault("success_count", int(round(_float(omni.get("success_rate"), 0.0) * int(metrics.get("completed_episodes", 0) or 0))))
        metrics.setdefault("collision_count", int(omni.get("collision_count", 0) or 0))
        metrics.setdefault("stale_discard_count", int(omni.get("stale_discard_count", 0) or 0))
    mode_table = output / "mode_table.csv"
    if mode_table.exists() and "success_count" not in metrics:
        with mode_table.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if "omninav_only" in str(row.get("mode", "")):
                    success, episodes = parse_fraction(row.get("success"))
                    metrics["success_count"] = success
                    metrics["completed_episodes"] = episodes
                    metrics["mean_path_m"] = _float(row.get("mean_path_m"), 0.0)
                    break
    return metrics


def aggregate_v6_metrics(
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    safe_rows: list[dict[str, Any]],
    stale: dict[str, Any],
) -> dict[str, Any]:
    primitives = [str(row.get("parsed_primitive") or "") for row in action_rows]
    safe_max = max([abs(_float(row.get("safe_linear_x"), 0.0)) for row in safe_rows] or [0.0])
    candidate_max = max([abs(_float(row.get("candidate_linear_x"), 0.0)) for row in safe_rows] or [0.0])
    return {
        "move_forward_count": primitives.count("move_forward"),
        "follow_waypoint_count": primitives.count("follow_waypoint"),
        "turn_left_count": primitives.count("turn_left"),
        "turn_right_count": primitives.count("turn_right"),
        "stop_count": primitives.count("stop"),
        "max_linear_x_mps": max(_float(metrics.get("max_linear_x_mps"), 0.0), safe_max),
        "max_candidate_linear_x_mps": candidate_max,
        "stale_discard_count": int(metrics.get("stale_discard_count", stale.get("total_discards", 0))),
        "timebase_error_count": stale["counts"].get("timebase_error", int(metrics.get("timebase_error_count", 0) or 0)),
        "episode_mismatch_count": stale["counts"].get("episode_mismatch", int(metrics.get("episode_mismatch_count", 0) or 0)),
        "missing_timestamp_count": stale["counts"].get("missing_timestamp", int(metrics.get("missing_timestamp_count", 0) or 0)),
        "collision_count": int(metrics.get("collision_count", metrics.get("num_collisions", 0)) or 0),
        "stale_action_executed": int(metrics.get("stale_action_executed", 0) or 0),
    }


def action_trace_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        details = event.get("details") if isinstance(event.get("details"), dict) else event
        event_type = str(details.get("event_type") or event.get("event") or "")
        if event_type not in {"omninav_model_response", "omninav_action", "primitive_command"}:
            continue
        primitive = details.get("primitive") or details.get("action_type")
        primitive_payload = details.get("primitive") if isinstance(details.get("primitive"), dict) else {}
        rows.append(
            {
                "t": event.get("t", details.get("timestamp", "")),
                "episode_id": event.get("episode_id", details.get("episode_id", "")),
                "request_id": details.get("request_id", primitive_payload.get("request_id", "")),
                "omninav_raw_output": primitive_payload.get("raw_text") or details.get("raw_text") or "",
                "raw_waypoint": primitive_payload.get("raw_waypoint") or details.get("raw_waypoint") or [],
                "raw_action": primitive_payload.get("raw_action") or details.get("raw_action") or "",
                "parsed_primitive": primitive_payload.get("primitive") or primitive or "",
                "fallback_used": primitive_payload.get("fallback_used", details.get("fallback_used", False)),
                "parser_reason": primitive_payload.get("parser_reason", details.get("parser_reason", "")),
                "cmd_vel_candidate": details.get("cmd_vel", {}),
            }
        )
    return rows


def safe_trace_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last_candidate = {"linear_x": 0.0, "angular_z": 0.0}
    for event in events:
        details = event.get("details") if isinstance(event.get("details"), dict) else event
        event_type = str(details.get("event_type") or event.get("event") or "")
        if event_type == "primitive_command":
            cmd = details.get("cmd_vel") or {}
            last_candidate = {
                "linear_x": _float((cmd.get("linear") or {}).get("x"), 0.0),
                "angular_z": _float((cmd.get("angular") or {}).get("z"), 0.0),
            }
        if event_type != "safe_cmd_mux":
            continue
        cmd = details.get("cmd_vel") or {}
        rows.append(
            {
                "t": event.get("t", details.get("timestamp", "")),
                "episode_id": event.get("episode_id", details.get("episode_id", "")),
                "candidate_linear_x": last_candidate["linear_x"],
                "candidate_angular_z": last_candidate["angular_z"],
                "safe_linear_x": _float((cmd.get("linear") or {}).get("x"), 0.0),
                "safe_angular_z": _float((cmd.get("angular") or {}).get("z"), 0.0),
                "result": details.get("result", ""),
            }
        )
    return rows


def write_root_cause(path: Path, metrics: dict[str, Any], first10_root: str) -> None:
    lines = [
        "# V6 Regression Root Cause",
        "",
        f"- first10_root_cause: {first10_root}",
        f"- pass: {metrics.get('pass', False)}",
        f"- success_count: {metrics.get('success_count', 0)}",
        f"- mean_path_m: {metrics.get('mean_path_m', 0)}",
        f"- move_forward_count: {metrics.get('move_forward_count', 0)}",
        f"- max_linear_x_mps: {metrics.get('max_linear_x_mps', 0)}",
        f"- stale_discard_count: {metrics.get('stale_discard_count', 0)}",
        "",
        "## Failures",
        "",
    ]
    lines.extend([f"- {item}" for item in metrics.get("failures", [])] or ["- none"])
    lines.extend(["", "Decision: OmniNav baseline not restored; downstream Step value evaluation skipped.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_fraction(value: Any) -> tuple[int, int]:
    text = str(value or "")
    if "/" in text:
        left, right = text.split("/", 1)
        return int(float(left)), int(float(right))
    return int(float(text or 0)), 0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="omninav_only")
    parser.add_argument("--config", default=str(ROOT / "configs" / "live_success_v6_omninav_regression.yaml"))
    parser.add_argument("--output", default="")
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--real-omninav", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=15)
    parser.add_argument("--isaac-host", default="")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--internnav-server", default="http://10.100.100.128:8087")
    parser.add_argument("--dgx-user", default="")
    parser.add_argument("--dgx-password", default="")
    args = parser.parse_args(argv)
    if args.mode != "omninav_only":
        raise SystemExit("v6 recovery gate only allows --mode omninav_only before baseline is restored")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else ROOT / "runs" / f"v6_omninav_only_regression_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_live_success_benchmark.py"),
        "--config",
        str(args.config),
        "--modes",
        "omninav_only",
        "--max-episodes",
        str(args.max_episodes),
        "--output",
        str(out_dir),
    ]
    if args.mock_models or args.dry_run:
        cmd.append("--mock-models")
    if args.real_omninav:
        cmd.append("--real-omninav")
    if args.isaac_host and not (args.mock_models or args.dry_run):
        cmd += ["--isaac-host", args.isaac_host, "--isaac-user", args.isaac_user, "--isaac-password", args.isaac_password]
    if args.isaac_hostkey:
        cmd += ["--isaac-hostkey", args.isaac_hostkey]
    if args.internnav_server:
        cmd += ["--internnav-server", args.internnav_server]
    if args.dgx_user:
        cmd += ["--dgx-user", args.dgx_user]
    if args.dgx_password:
        cmd += ["--dgx-password", args.dgx_password]
    result = subprocess.run(cmd, check=False)
    metrics = ensure_v6_artifacts(out_dir)
    (out_dir / "v6_mode_alias.json").write_text(json.dumps({"requested_mode": args.mode, "benchmark_mode": "omninav_only"}, indent=2) + "\n", encoding="utf-8")
    if result.returncode != 0:
        return result.returncode
    return 0 if metrics.get("pass", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
