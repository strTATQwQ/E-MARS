#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def value(mapping: dict[str, Any], key: str, default: Any = "n/a") -> Any:
    result = mapping.get(key, default)
    return default if result is None else result


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the canonical pure perception/planning report from structured gates.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--asset-catalog", required=True)
    parser.add_argument("--dgx-summary", default="")
    parser.add_argument("--multimodal-value-gate", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    suite = Path(args.suite)
    gate = read_json(suite / "gate.json", {})
    generation = read_json(suite / "generation_summary.json", {})
    catalog = read_json(Path(args.asset_catalog), {})
    dgx = read_json(Path(args.dgx_summary), {}) if args.dgx_summary else {}
    value_gate = read_json(Path(args.multimodal_value_gate), {}) if args.multimodal_value_gate else {}
    gates = gate.get("gates") or {}
    controlled = (gates.get("controlled_visual") or {}).get("metrics") or {}
    mp3d = (gates.get("mp3d_pe_visual") or {}).get("metrics") or {}
    tracking = (gates.get("tracking") or {}).get("metrics") or {}
    planning = (gates.get("planning") or {}).get("metrics") or {}
    robustness = gates.get("robustness") or {}
    dgx_metrics = dgx.get("metrics") or {}
    gpu = dgx_metrics.get("gpu_util_pct") or {}
    power = dgx_metrics.get("power_w") or {}
    temperature = dgx_metrics.get("temperature_c") or {}
    rss = dgx_metrics.get("llama_rss_kib") or {}
    lines = [
        "# Pure Perception, Tracking, And Route Planning Gate",
        "",
        f"- suite pass: **{bool(gate.get('pass'))}**",
        f"- multimodal screening allowed: **{bool(gate.get('screening_allowed'))}**",
        "- locomotion backend: `ideal_kinematic`",
        "- qualification evidence: `false`",
        "- InternNav identity: `CmaAgent/system1/fallback_static_cma_tokens`",
        "- Sim2Real: **NOT READY FOR REAL ROBOT AUTONOMY**",
        "- no RL, locomotion policy, gait, or dynamics changes were used",
        "",
        "## Assets And Visual Domain",
        "",
        f"- mp3d_pe archive SHA256: `{value(catalog, 'archive_sha256')}`",
        f"- installed bytes: {value(catalog, 'directory_size_bytes')}",
        f"- indexed metric scenes: {value(catalog, 'scene_count')}",
        f"- generated controlled/mp3d cases: {value(generation, 'controlled_visual_cases')}/{value(generation, 'mp3d_pe_visual_cases')}",
        f"- controlled route accuracy: {value(controlled, 'route_accuracy')}",
        f"- controlled semantic precision/recall/FPR: {value(controlled, 'semantic_precision')}/{value(controlled, 'semantic_recall')}/{value(controlled, 'semantic_false_positive_rate')}",
        f"- mp3d route accuracy: {value(mp3d, 'route_accuracy')}",
        f"- mp3d semantic precision/recall/FPR: {value(mp3d, 'semantic_precision')}/{value(mp3d, 'semantic_recall')}/{value(mp3d, 'semantic_false_positive_rate')}",
        f"- fresh images / parse errors / oracle leakage: {value(mp3d, 'fresh_image_rate')}/{value(mp3d, 'json_parse_errors')}/{value(mp3d, 'oracle_context_leakage')}",
        "",
        "## Tracking",
        "",
        f"- frames: {value(tracking, 'frames')}",
        f"- duplicate-frame hit miscounts: {value(tracking, 'duplicate_frame_miscounts')}",
        f"- episode/target cross-talk: {value(tracking, 'episode_target_cross_talk')}",
        f"- confirmed false-target rate: {value(tracking, 'confirmed_false_target_rate')}",
        f"- occlusion reacquisition rate: {value(tracking, 'reacquisition_rate')}",
        f"- stale-track actions: {value(tracking, 'stale_track_action_count')}",
        "",
        "## Planning And Stop",
        "",
        f"- correct branches: {value(planning, 'route_correct')}/{value(planning, 'route_total')}",
        f"- left/right correct: {value(planning.get('route_side_correct') or {}, 'left')}/{value(planning.get('route_side_correct') or {}, 'right')}",
        f"- wrong branch after confirmation: {value(planning, 'confirmed_wrong_branch_entries')}",
        f"- semantic coverage/success positives: {value(planning, 'semantic_positive_coverage')}/{value(planning, 'semantic_success')}",
        f"- negative false stops: {value(planning, 'semantic_negative_false_stop')}",
        f"- decreasing-distance approach fraction: {value(planning, 'approach_distance_decrease_rate')}",
        f"- threshold-to-safe-stop p95: {value(planning, 'threshold_to_safe_stop_p95_sec')} s (gate <=2 s)",
        "",
        "## Decision Robustness",
        "",
        f"- pass: {bool(robustness.get('pass'))}",
        f"- Step request p95: {value(robustness, 'step_latency_p95_sec')} s (gate <=7 s)",
        f"- stale action executed: {value(robustness, 'stale_action_executed')}",
        f"- reset old-track clearing: {value(robustness, 'old_track_clear_rate')}",
        "",
        "## DGX-Spark",
        "",
        f"- samples / active samples: {value(dgx, 'samples')}/{value(dgx, 'active_samples')}",
        f"- active GPU mean / p95 / max: {value(gpu, 'active_mean')}/{value(gpu, 'p95')}/{value(gpu, 'max')} %",
        f"- active power mean / max: {value(power, 'active_mean')}/{value(power, 'max')} W",
        f"- temperature max: {value(temperature, 'max')} C",
        f"- llama-server RSS max: {value(rss, 'max')} KiB",
        "",
        "## Value Claim",
        "",
    ]
    if value_gate:
        readiness = value_gate.get("final_smoke") or {}
        invalid_screen = value_gate.get("invalid_screen") or {}
        invalid_dgx = invalid_screen.get("dgx_performance") or {}
        invalid_dgx_metrics = invalid_dgx.get("metrics") or {}
        lines += [
            f"- fresh-image paired value gate pass: **{bool(value_gate.get('pass'))}**",
            f"- claim: {value(value_gate, 'value_claim')}",
            f"- formal 15-episode screening retry allowed: **{bool(value_gate.get('screen_retry_allowed'))}**",
            f"- 45-pair confirmation: **{value(value_gate, 'paired_confirmation')}**",
            f"- final smoke baseline simple / route / semantic: {int(bool(readiness.get('baseline_simple_success')))}/1, {int(bool(readiness.get('route_success')))}/1, {int(bool(readiness.get('semantic_success')))}/1",
            f"- final smoke Step p95: {value(readiness, 'step_latency_p95_sec')} s (gate <=7 s)",
            f"- failure_top1: `{value(value_gate, 'failure_top1')}`",
            f"- failure_top2: `{value(value_gate, 'failure_top2')}`",
            f"- rejected diagnostic run: {value(invalid_screen, 'run_id')} ({value(invalid_screen, 'episodes')} episodes, formal_value_evidence={str(bool(invalid_screen.get('formal_value_evidence'))).lower()})",
            f"- diagnostic DGX GPU mean/p95/max: {value(invalid_dgx_metrics.get('gpu_util_pct') or {}, 'mean')}/{value(invalid_dgx_metrics.get('gpu_util_pct') or {}, 'p95')}/{value(invalid_dgx_metrics.get('gpu_util_pct') or {}, 'max')} %",
            f"- diagnostic DGX power mean/p95/max: {value(invalid_dgx_metrics.get('power_w') or {}, 'mean')}/{value(invalid_dgx_metrics.get('power_w') or {}, 'p95')}/{value(invalid_dgx_metrics.get('power_w') or {}, 'max')} W",
            f"- diagnostic DGX temperature mean/p95/max: {value(invalid_dgx_metrics.get('temperature_c') or {}, 'mean')}/{value(invalid_dgx_metrics.get('temperature_c') or {}, 'p95')}/{value(invalid_dgx_metrics.get('temperature_c') or {}, 'max')} C",
        ]
    else:
        lines += [
            "- true multimodal full-task value remains **unproven** until the 15-episode screen and 45-pair confirmation both pass the fresh-image evidence gate.",
        ]
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
