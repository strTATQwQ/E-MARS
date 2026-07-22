#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from isaac_vln_benchmark.sensor_only_audit import audit_run


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def details(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("details")
    return value if isinstance(value, dict) else {}


def event_type(row: dict[str, Any]) -> str:
    value = details(row)
    return str(value.get("event_type") or row.get("event") or row.get("type") or "")


def percentile(values: list[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return round(ordered[low], 6)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 6)


def first_safe_zero_after(events: list[dict[str, Any]], start_t: float) -> float | None:
    for row in events:
        if float(row.get("t", -1.0)) < start_t or row.get("event") != "safe_cmd_vel":
            continue
        value = details(row)
        linear = value.get("linear") if isinstance(value.get("linear"), dict) else {}
        angular = value.get("angular") if isinstance(value.get("angular"), dict) else {}
        if abs(float(linear.get("x", 0.0))) < 1.0e-6 and abs(float(angular.get("z", 0.0))) < 1.0e-6:
            return float(row.get("t", 0.0))
    return None


def performance_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"samples": 0, "status": "missing"}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    def numbers(name: str) -> list[float]:
        result: list[float] = []
        for row in rows:
            try:
                result.append(float(row.get(name, 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
        return result

    def maximum(name: str) -> float | None:
        values = numbers(name)
        return round(max(values), 4) if values else None

    def mean(name: str) -> float | None:
        values = numbers(name)
        return round(sum(values) / len(values), 4) if values else None

    available = numbers("mem_available_kib")
    return {
        "samples": len(rows),
        "gpu_util_mean_pct": mean("gpu_util_pct"),
        "gpu_util_max_pct": maximum("gpu_util_pct"),
        "power_mean_w": mean("power_w"),
        "power_max_w": maximum("power_w"),
        "temperature_max_c": maximum("temp_c"),
        "mem_available_min_gib": round(min(available) / (1024.0 * 1024.0), 4) if available else None,
        "step_rss_max_gib": round((maximum("step_rss_kib") or 0.0) / (1024.0 * 1024.0), 4),
        "grounded_sam_rss_max_gib": round((maximum("grounded_sam_rss_kib") or 0.0) / (1024.0 * 1024.0), 4),
        "omninav_rss_max_gib": round((maximum("omninav_rss_kib") or 0.0) / (1024.0 * 1024.0), 4),
        "internnav_rss_max_gib": round((maximum("internnav_rss_kib") or 0.0) / (1024.0 * 1024.0), 4),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_evidence(run_dir: Path) -> dict[str, Any]:
    events = read_events(run_dir / "events.jsonl")
    metrics = read_json(run_dir / "metrics.json")
    episodes = metrics.get("episodes") if isinstance(metrics.get("episodes"), list) else []
    audit = audit_run(run_dir)
    perception = [
        float(details(row).get("latency_sec"))
        for row in events
        if event_type(row) == "grounded_sam_request" and details(row).get("latency_sec") is not None
    ]
    step = [
        float(details(row).get("latency_s"))
        for row in events
        if event_type(row) == "step_http_response" and details(row).get("latency_s") is not None
    ]
    tracks = [details(row).get("track") for row in events if event_type(row) == "sensor_target_track"]
    tracks = [row for row in tracks if isinstance(row, dict)]
    phases = Counter(
        str((details(row).get("primitive") or {}).get("phase") or "")
        for row in events
        if event_type(row) == "sensor_planner_primitive"
    )
    phases.pop("", None)
    threshold_events = [
        float(row.get("t", 0.0))
        for row in events
        if event_type(row) == "mission_event_step_trigger"
        and str(details(row).get("event") or "") == "candidate_goal_reached"
    ]
    stop_latencies: list[float] = []
    for start in threshold_events:
        stopped = first_safe_zero_after(events, start)
        if stopped is not None:
            stop_latencies.append(max(0.0, stopped - start))

    collisions = sum(int(row.get("num_collisions", 0) or 0) for row in episodes)
    safety_blocks = sum(int(row.get("safety_block_ticks", 0) or 0) for row in episodes)
    stale_results = sum(
        int(row.get("num_stale_step_results", 0) or 0) + int(row.get("num_stale_omninav_actions", 0) or 0)
        for row in episodes
    )
    timebase_errors = sum(
        "timebase" in event_type(row).lower() and str(details(row).get("result") or "").lower() in {"error", "rejected"}
        for row in events
    )
    stale_action_executed = sum(event_type(row) == "stale_action_executed" for row in events)
    task_types = {str(row.get("task_type") or "") for row in episodes}
    semantic = "semantic_target" in task_types
    route = "turn_choice" in task_types
    common_gate = {
        "all_success": bool(episodes) and all(bool(row.get("success")) for row in episodes),
        "all_clean": bool(episodes) and all(bool(row.get("clean_success")) for row in episodes),
        "perception_p95_le_5s": bool(perception) and (percentile(perception, 0.95) or 99.0) <= 5.0,
        "step_p95_le_7s": bool(step) and (percentile(step, 0.95) or 99.0) <= 7.0,
        "collision_zero": collisions == 0,
        "safety_block_zero": safety_blocks == 0,
        "runtime_stale_zero": stale_results == 0,
        "timebase_error_zero": timebase_errors == 0,
        "stale_action_executed_zero": stale_action_executed == 0,
        "oracle_leakage_zero": audit["pass"],
    }
    if semantic:
        common_gate["threshold_to_safe_stop_p95_le_2s"] = bool(stop_latencies) and (percentile(stop_latencies, 0.95) or 99.0) <= 2.0
        common_gate["two_frame_track_confirmed"] = any(int(row.get("hits", 0) or 0) >= 2 and row.get("confirmed") for row in tracks)
    if route:
        common_gate["route_phase_sequence_complete"] = all(
            phase in phases
            for phase in ("approach_intersection", "rotate_to_branch", "advance_into_branch", "verify_branch")
        )
    return {
        "run_dir": str(run_dir.resolve()),
        "task_types": sorted(task_types),
        "episodes": episodes,
        "latency": {
            "perception_count": len(perception),
            "perception_p95_sec": percentile(perception, 0.95),
            "perception_max_sec": max(perception) if perception else None,
            "step_count": len(step),
            "step_p95_sec": percentile(step, 0.95),
            "threshold_to_safe_stop_count": len(stop_latencies),
            "threshold_to_safe_stop_p95_sec": percentile(stop_latencies, 0.95),
        },
        "track": {
            "event_count": len(tracks),
            "distinct_frame_count": len({row.get("frame_seq") for row in tracks if row.get("frame_seq") is not None}),
            "max_hits": max((int(row.get("hits", 0) or 0) for row in tracks), default=0),
            "confirmed_event_count": sum(bool(row.get("confirmed")) for row in tracks),
            "duplicate_frames": max((int(row.get("duplicate_frames", 0) or 0) for row in tracks), default=0),
            "out_of_order_frames": max((int(row.get("out_of_order_frames", 0) or 0) for row in tracks), default=0),
            "discontinuity_rejections": max((int(row.get("discontinuity_rejections", 0) or 0) for row in tracks), default=0),
        },
        "planner_phase_counts": dict(phases),
        "safety": {
            "collision": collisions,
            "safety_block_ticks": safety_blocks,
            "runtime_stale": stale_results,
            "timebase_error": timebase_errors,
            "stale_action_executed": stale_action_executed,
        },
        "oracle_leakage_audit": audit,
        "dgx_performance": performance_summary(run_dir / "dgx_performance.csv"),
        "gates": common_gate,
        "pass": all(common_gate.values()),
        "qualification_evidence": False,
        "locomotion_fidelity": "ideal_kinematic",
        "internnav_identity": "CmaAgent/system1/fallback_static_cma_tokens",
    }


def write_outputs(run_dir: Path, result: dict[str, Any]) -> None:
    gate_path = run_dir / "v16_sensor_only_gate.json"
    gate_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = [name for name, passed in result["gates"].items() if not passed]
    latency = result["latency"]
    perf = result["dgx_performance"]
    summary = (
        "# V16 sensor-only run evidence\n\n"
        f"- gate: {'PASS' if result['pass'] else 'FAIL'}\n"
        f"- task types: {', '.join(result['task_types'])}\n"
        f"- failed checks: {', '.join(failed) if failed else 'none'}\n"
        f"- perception p95: {latency['perception_p95_sec']} s\n"
        f"- Step p95: {latency['step_p95_sec']} s\n"
        f"- threshold-to-safe-stop p95: {latency['threshold_to_safe_stop_p95_sec']} s\n"
        f"- track distinct frames / max hits: {result['track']['distinct_frame_count']} / {result['track']['max_hits']}\n"
        f"- planner phases: {json.dumps(result['planner_phase_counts'], sort_keys=True)}\n"
        f"- safety: {json.dumps(result['safety'], sort_keys=True)}\n"
        f"- DGX samples / GPU max / power max: {perf.get('samples')} / {perf.get('gpu_util_max_pct')}% / {perf.get('power_max_w')} W\n"
        "- qualification_evidence: false\n"
        "- Sim2Real: NOT READY\n"
    )
    (run_dir / "v16_sensor_only_summary.md").write_text(summary, encoding="utf-8")
    artifact_names = [
        "config.yaml",
        "scheduler_config.yaml",
        "metrics.json",
        "events.jsonl",
        "dgx_performance.csv",
        "sensor_only_oracle_leakage_audit.json",
        "v16_sensor_only_gate.json",
        "model_lock_v16.yaml",
    ]
    artifacts = []
    for name in artifact_names:
        path = run_dir / name
        if path.is_file():
            artifacts.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    visual_dir = run_dir / "visual"
    if visual_dir.is_dir():
        for path in sorted(value for value in visual_dir.rglob("*") if value.is_file()):
            artifacts.append(
                {
                    "path": str(path.relative_to(run_dir)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "run_dir": str(run_dir.resolve()),
        "artifacts": artifacts,
        "qualification_evidence": False,
    }
    (run_dir / "canonical_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    result = build_evidence(run_dir)
    write_outputs(run_dir, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
