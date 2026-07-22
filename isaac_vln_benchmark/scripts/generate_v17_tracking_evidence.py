#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 4)


def performance(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"samples": 0, "metrics": {}}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = (
        "gpu_util_pct",
        "gpu_mem_util_pct",
        "power_w",
        "temp_c",
        "mem_available_kib",
        "step_rss_kib",
        "grounded_sam_rss_kib",
        "omninav_rss_kib",
        "internnav_rss_kib",
        "load1",
    )
    metrics: dict[str, Any] = {}
    for field in fields:
        values = []
        for row in rows:
            try:
                values.append(float(row[field]))
            except (KeyError, TypeError, ValueError):
                pass
        if values:
            metrics[field] = {
                "mean": round(sum(values) / len(values), 4),
                "p95": percentile(values, 0.95),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
    return {"samples": len(rows), "metrics": metrics}


def route_summary(path: Path) -> dict[str, Any]:
    metrics_path = path / "metrics.json"
    if not metrics_path.is_file():
        return {"run_id": path.name, "available": False}
    document = load_json(metrics_path)
    episodes = document.get("episodes") if isinstance(document.get("episodes"), list) else []
    episode = episodes[0] if episodes else {}
    events_path = path / "events.jsonl"
    temporal_requests = 0
    seed_commits = 0
    phases: dict[str, int] = {}
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            if event.get("event") == "metrics_event_jsonl" and details.get("event_type") == "grounded_sam_request":
                temporal_requests += int(bool(details.get("temporal_requested")))
            if event.get("event") == "metrics_event_jsonl" and details.get("event_type") == "temporal_seed_committed":
                seed_commits += 1
            if event.get("event") == "metrics_event_jsonl" and details.get("event_type") == "sensor_planner_primitive":
                primitive = details.get("primitive") if isinstance(details.get("primitive"), dict) else {}
                phase = str(primitive.get("phase") or "unknown")
                phases[phase] = phases.get(phase, 0) + 1
    return {
        "run_id": path.name,
        "available": True,
        "success": bool(episode.get("success")),
        "clean_success": bool(episode.get("clean_success")),
        "failure_reason": episode.get("failure_reason"),
        "path_length_m": episode.get("path_length_m"),
        "final_distance_to_target_m": episode.get("final_distance_to_target_m"),
        "collisions": int(episode.get("num_collisions", 0) or 0),
        "stale_step": int(episode.get("num_stale_step_results", 0) or 0),
        "stale_omninav": int(episode.get("num_stale_omninav_actions", 0) or 0),
        "safety_stops": int(episode.get("num_safety_stops", 0) or 0),
        "step_latency_ms": episode.get("step_mean_latency_ms"),
        "temporal_requests": temporal_requests,
        "seed_commits": seed_commits,
        "phases": phases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V17 RGB-D temporal tracking evidence.")
    parser.add_argument("--tracking-run", required=True)
    parser.add_argument("--route-run", action="append", default=[])
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    tracking_run = Path(args.tracking_run).resolve()
    gate = load_json(tracking_run / "result" / "tracking_gate.json")
    perf = performance(tracking_run / "dgx_perf.csv")
    routes = [route_summary(Path(value).resolve()) for value in args.route_run]
    left_run = any("left" in row.get("run_id", "") for row in routes)
    right_run = any("right" in row.get("run_id", "") for row in routes)
    left_pass = any(row.get("success") and "left" in row.get("run_id", "") for row in routes)
    right_pass = any(row.get("success") and "right" in row.get("run_id", "") for row in routes)
    tracking_pass = bool(gate.get("pass"))
    route_gate_pass = bool(left_pass and right_pass)
    result = {
        "schema_version": 1,
        "version": "V17",
        "scope": "perception_tracking_route_planning_only",
        "tracking_run_id": tracking_run.name,
        "tracking_gate": gate,
        "route_diagnostics": routes,
        "route_gate": {
            "pass": route_gate_pass,
            "left_run": left_run,
            "left_pass": left_pass,
            "right_run": right_run,
            "right_pass": right_pass,
        },
        "dgx_performance": perf,
        "next_stage": {
            "visual_90_allowed": bool(tracking_pass and route_gate_pass),
            "route_semantic_30x30_allowed": False,
            "screening_15_allowed": False,
            "paired_45_allowed": False,
        },
        "value_claim": "true multimodal OmniNav+Step full route-stop value remains unproven",
        "sim2real": "NOT READY FOR REAL ROBOT AUTONOMY",
        "qualification_evidence": False,
        "locomotion_fidelity": "ideal_kinematic",
        "internnav_identity": "CmaAgent/system1/fallback_static_cma_tokens",
    }
    evidence_dir = workspace / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    gate_path = evidence_dir / "v17_perception_tracking_gate.json"
    gate_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    perf_path = tracking_run / "dgx_performance_summary.json"
    perf_path.write_text(json.dumps(perf, indent=2) + "\n", encoding="utf-8")
    key_files = [tracking_run / "result" / "tracking_gate.json", tracking_run / "result" / "tracking_results.json", perf_path, gate_path]
    manifest = {
        "schema_version": 1,
        "run_id": tracking_run.name,
        "remote_raw_evidence": f"/home/song/dgx-unitree/isaac_vln_benchmark/runs/{tracking_run.name}",
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in key_files
            if path.is_file()
        ],
    }
    manifest_path = tracking_run / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    metrics = gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {}
    gpu = perf.get("metrics", {}).get("gpu_util_pct", {})
    power = perf.get("metrics", {}).get("power_w", {})
    temperature = perf.get("metrics", {}).get("temp_c", {})
    memory = perf.get("metrics", {}).get("mem_available_kib", {})
    step_rss = perf.get("metrics", {}).get("step_rss_kib", {})
    grounded_rss = perf.get("metrics", {}).get("grounded_sam_rss_kib", {})
    omninav_rss = perf.get("metrics", {}).get("omninav_rss_kib", {})
    report = [
        "# V17 RGB-D Temporal Tracking And Route Status",
        "",
        f"- tracking full30: **{'PASS' if tracking_pass else 'FAIL'}**",
        f"- actual viewport route: **{'PASS' if route_gate_pass else 'FAIL'}** "
        f"(left={'PASS' if left_pass else 'FAIL' if left_run else 'NOT RUN'}, "
        f"right={'PASS' if right_pass else 'FAIL' if right_run else 'NOT RUN'})",
        "- formal value screening: **LOCKED**",
        "- Sim2Real: **NOT READY FOR REAL ROBOT AUTONOMY**",
        "- no RL, gait, dynamics, locomotion policy, or real Go2 changes were used",
        "",
        "## Tracking",
        "",
        f"- sequences / rows / model calls: {metrics.get('sequences')}/{metrics.get('frames')}/{metrics.get('perception_calls')}",
        f"- duplicate miscounts / cross-talk / stale actions: {metrics.get('duplicate_frame_miscounts')}/{metrics.get('episode_target_cross_talk')}/{metrics.get('stale_track_action_count')}",
        f"- confirmed visible false-target rate: {metrics.get('confirmed_false_target_rate')}",
        f"- occlusion reacquisition: {metrics.get('occlusion_reacquired_sequences')}/{metrics.get('occlusion_sequences')} ({metrics.get('reacquisition_rate')})",
        f"- temporal requests / detector-supported: {metrics.get('temporal_requests')}/{metrics.get('temporal_detector_supported')}",
        f"- perception p95: {metrics.get('perception_latency_p95_sec')} s (gate <=5 s)",
        "",
        "## Route",
        "",
    ]
    for row in routes:
        report.append(
            f"- `{row.get('run_id')}`: success={row.get('success')}, failure={row.get('failure_reason')}, "
            f"path={row.get('path_length_m')} m, safety_stops={row.get('safety_stops')}, "
            f"temporal_requests={row.get('temporal_requests')}, seed_commits={row.get('seed_commits')}"
        )
    report.extend(
        [
            "",
            "## DGX-Spark",
            "",
            f"- samples: {perf.get('samples')}",
            f"- GPU util mean/p95/max: {gpu.get('mean')}/{gpu.get('p95')}/{gpu.get('max')} %",
            f"- power mean/p95/max: {power.get('mean')}/{power.get('p95')}/{power.get('max')} W",
            f"- temperature mean/p95/max: {temperature.get('mean')}/{temperature.get('p95')}/{temperature.get('max')} C",
            f"- unified memory available min: {round(float(memory.get('min', 0.0)) / 1048576.0, 3)} GiB",
            f"- RSS max GiB (Step/Grounded-SAM/OmniNav): "
            f"{round(float(step_rss.get('max', 0.0)) / 1048576.0, 3)}/"
            f"{round(float(grounded_rss.get('max', 0.0)) / 1048576.0, 3)}/"
            f"{round(float(omninav_rss.get('max', 0.0)) / 1048576.0, 3)}",
            "- GB10 nvidia-smi GPU-memory percentage was unavailable and recorded as 0; use unified-memory and RSS figures above.",
            "",
            "## Conclusion",
            "",
            "Tracking progress does not unlock the value experiment while actual-viewport left/right route micro remains failed.",
            "True multimodal OmniNav+Step full route-stop value remains unproven. Sim2Real remains NOT READY.",
        ]
    )
    report_text = "\n".join(report) + "\n"
    report_path = workspace / "V17_RGBD_TEMPORAL_TRACKING_REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")
    current_report = workspace / "docs" / "current" / "V17_RGBD_TEMPORAL_TRACKING_REPORT.md"
    current_report.parent.mkdir(parents=True, exist_ok=True)
    current_report.write_text(report_text, encoding="utf-8")
    print(json.dumps({"gate": str(gate_path), "manifest": str(manifest_path), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
