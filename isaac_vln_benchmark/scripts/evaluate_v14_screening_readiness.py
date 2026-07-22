#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


BASELINE_MODE = "omninav_only_v11_matched"
CANDIDATE_MODE = "omninav_step_route_stop_v12_screen"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def step_calls(events_path: Path) -> list[dict[str, Any]]:
    calls = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            details = row.get("details") if isinstance(row.get("details"), dict) else {}
            if row.get("event") != "metrics_event_jsonl" or details.get("event_type") != "step_http_response":
                continue
            output = details.get("output") if isinstance(details.get("output"), dict) else {}
            snapshot = details.get("image_snapshot") if isinstance(details.get("image_snapshot"), dict) else {}
            calls.append(
                {
                    "episode_id": row.get("episode_id"),
                    "role": details.get("role"),
                    "result": details.get("result"),
                    "multimodal": bool(details.get("multimodal")),
                    "fresh_image": int(snapshot.get("frame_seq", -1)) >= 0 and float(snapshot.get("age_sec", 999.0)) <= 0.75,
                    "frame_seq": snapshot.get("frame_seq"),
                    "image_age_sec": snapshot.get("age_sec"),
                    "horizontal_flip": snapshot.get("horizontal_flip"),
                    "vertical_flip": snapshot.get("vertical_flip"),
                    "latency_sec": details.get("latency_s"),
                    "route_choice": output.get("route_choice"),
                    "stop": output.get("stop"),
                    "target_visible": output.get("target_visible"),
                    "track_confirmed": bool((output.get("track") or {}).get("confirmed", False)),
                    "oracle_context_leakage": int(output.get("oracle_context_leakage", 0) or 0),
                }
            )
    return calls


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception-suite", required=True)
    parser.add_argument("--invalid-screen", required=True)
    parser.add_argument("--final-smoke", required=True)
    parser.add_argument("--camera-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    suite = Path(args.perception_suite)
    invalid = Path(args.invalid_screen)
    smoke = Path(args.final_smoke)
    camera = Path(args.camera_audit)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    suite_gate = load_json(suite / "gate.json")
    invalid_value = load_json(invalid / "gate.json")
    invalid_multimodal = load_json(invalid / "multimodal_value_gate.json")
    invalid_dgx = load_json(invalid / "dgx_performance_summary.json")
    metrics = load_json(smoke / "metrics.json")
    episodes = metrics.get("episodes", [])
    calls = step_calls(smoke / "events.jsonl")
    by_task = {(row.get("mode"), row.get("task_type")): row for row in episodes}
    baseline_simple = by_task.get((BASELINE_MODE, "simple_navigation"), {})
    route = by_task.get((CANDIDATE_MODE, "turn_choice"), {})
    semantic = by_task.get((CANDIDATE_MODE, "semantic_target"), {})
    route_call = next((row for row in calls if row.get("role") == "route_choice"), {})
    semantic_call = next((row for row in calls if row.get("role") == "semantic_stop"), {})
    latencies = sorted(float(row["latency_sec"]) for row in calls if row.get("latency_sec") is not None)
    p95 = latencies[-1] if latencies else None

    checks = {
        "perception_suite_pass": bool(suite_gate.get("pass")),
        "invalid_90_episode_run_rejected": not bool(invalid_multimodal.get("pass")),
        "invalid_run_had_no_candidate_multimodal_calls": int(
            ((invalid_value.get("real_step_http") or {}).get(CANDIDATE_MODE) or {}).get("accepted_multimodal", 0)
        ) == 0,
        "final_smoke_episode_count_6": len(episodes) == 6,
        "final_smoke_route_call_fresh_multimodal": bool(
            route_call.get("result") == "accepted" and route_call.get("multimodal") and route_call.get("fresh_image")
        ),
        "final_smoke_semantic_call_fresh_multimodal": bool(
            semantic_call.get("result") == "accepted" and semantic_call.get("multimodal") and semantic_call.get("fresh_image")
        ),
        "step_latency_p95_le_7s": p95 is not None and p95 <= 7.0,
        "oracle_context_leakage_zero": sum(int(row.get("oracle_context_leakage", 0)) for row in calls) == 0,
        "baseline_simple_success": bool(baseline_simple.get("success")),
        "candidate_route_success": bool(route.get("success")),
        "candidate_semantic_success": bool(semantic.get("success")),
        "candidate_semantic_stop_true": bool(semantic_call.get("stop")),
        "candidate_semantic_track_confirmed": bool(semantic_call.get("track_confirmed")),
    }
    failures = [name for name, passed in checks.items() if not passed]
    gate = {
        "schema_version": 1,
        "pass": False,
        "screen_retry_allowed": False,
        "paired_confirmation": "LOCKED",
        "perception_tracking_planning_suite_pass": bool(suite_gate.get("pass")),
        "invalid_screen": {
            "run_id": invalid.name,
            "episodes": sum(int((row or {}).get("episodes", 0)) for row in (invalid_value.get("modes") or {}).values()),
            "formal_value_evidence": False,
            "reason": "V14 scheduler config was fingerprinted but not forwarded to the V12 launcher; candidate Step calls were not multimodal",
            "dgx_performance": invalid_dgx,
        },
        "camera_contract": {
            "omninav_topic": "/camera/front/image",
            "omninav_source": "V6-compatible synthetic semantic renderer",
            "step_topic": "/camera/front/isaac_image",
            "step_horizontal_flip": True,
            "step_vertical_flip": True,
            "raw_audit": load_json(camera / "metadata.json"),
        },
        "final_smoke": {
            "run_id": smoke.name,
            "episodes": len(episodes),
            "baseline_simple_success": bool(baseline_simple.get("success")),
            "baseline_simple_path_m": baseline_simple.get("path_length_m"),
            "route_success": bool(route.get("success")),
            "semantic_success": bool(semantic.get("success")),
            "step_calls": calls,
            "step_latency_p95_sec": round(p95, 6) if p95 is not None else None,
        },
        "checks": checks,
        "failures": failures,
        "failure_top1": "omninav_simple_visual_input_contract_regression",
        "failure_top2": "step_semantic_target_not_visible_or_unconfirmed",
        "qualification_evidence": False,
        "locomotion_fidelity": "ideal_kinematic",
        "value_claim": "true multimodal OmniNav+Step full route-stop value remains unproven",
        "sim2real_gate": "NOT READY FOR REAL ROBOT AUTONOMY",
    }
    gate_path = output / "gate.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    with (output / "failure_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "failure", "blocking"])
        writer.writeheader()
        for rank, failure in enumerate(failures, 1):
            writer.writerow({"rank": rank, "failure": failure, "blocking": True})
    lines = [
        "# V14 Multimodal Screening Readiness",
        "",
        "- gate: **FAIL**",
        "- 45-pair confirmation: **LOCKED**",
        f"- final smoke: route={int(bool(route.get('success')))}/1, semantic={int(bool(semantic.get('success')))}/1, baseline simple={int(bool(baseline_simple.get('success')))}/1",
        f"- fresh multimodal calls: {sum(1 for row in calls if row.get('fresh_image') and row.get('multimodal'))}/{len(calls)}",
        f"- Step p95: {p95:.3f}s (gate <=5s)" if p95 is not None else "- Step p95: unavailable",
        f"- failure_top1: `{gate['failure_top1']}`",
        f"- failure_top2: `{gate['failure_top2']}`",
        "- qualification_evidence: `false`",
        "- Sim2Real: **NOT READY FOR REAL ROBOT AUTONOMY**",
        "",
        "The completed 90-episode run is retained as diagnostic evidence but is not formal value evidence because the V14 scheduler profile was not forwarded to the launcher.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = []
    for path in [gate_path, output / "failure_table.csv", output / "summary.md"]:
        artifacts.append({"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = {"schema_version": 1, "run_id": output.name, "artifacts": artifacts}
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
