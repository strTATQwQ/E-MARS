#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.perception_planning_suite import (
    STEP_REQUEST_LATENCY_P95_MAX_SEC,
    THRESHOLD_TO_SAFE_STOP_P95_MAX_SEC,
    audit_step_visible_request,
    build_controlled_visual_cases,
    build_mp3d_visual_cases,
    build_planning_cases,
    build_tracking_sequences,
    evaluate_planning,
    evaluate_robustness,
    evaluate_tracking_sequences,
    evaluate_visual_cases,
    mp3d_background_transform,
    planning_gate,
    step_visible_request,
    tracking_gate,
    visual_gate,
    visual_rows_from_micro,
    write_artifact_manifest,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def scene_index(asset_root: Path) -> list[str]:
    if not asset_root.is_dir():
        return []
    scene_paths = sorted(
        path
        for path in asset_root.rglob("isaacsim_*.usd")
        if path.is_file() and not path.name.endswith("_non_metric.usd")
    )
    return [path.relative_to(asset_root).as_posix() for path in scene_paths]


def generate(output: Path, asset_root: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    scenes = scene_index(asset_root)
    controlled = build_controlled_visual_cases()
    mp3d = build_mp3d_visual_cases(scenes)
    catalog_path = asset_root / "scene_catalog.json"
    catalog_rows = {}
    if catalog_path.is_file():
        catalog = read_json(catalog_path)
        catalog_rows = {
            str(row.get("usd_path")): row
            for row in catalog.get("scenes", [])
            if isinstance(row, dict)
        }
    for case in mp3d:
        if case.get("scene_id"):
            case["scene_usd_path"] = str((asset_root / str(case["scene_id"])).resolve())
            catalog_row = catalog_rows.get(str(case["scene_id"])) or {}
            bounds = catalog_row.get("obj_bounds") if isinstance(catalog_row.get("obj_bounds"), dict) else {}
            case.update(mp3d_background_transform(bounds))
    tracking = build_tracking_sequences()
    planning = build_planning_cases()
    audits = [
        {"case_id": case["case_id"], **audit_step_visible_request(step_visible_request(case))}
        for case in controlled + mp3d
    ]
    write_json(output / "controlled_visual_cases.json", controlled)
    write_json(output / "mp3d_pe_visual_cases.json", mp3d)
    write_json(output / "tracking_sequences.json", tracking)
    write_json(output / "route_planning_cases.json", planning["route"])
    write_json(output / "semantic_planning_cases.json", planning["semantic"])
    write_json(output / "step_visible_request_audit.json", audits)
    inventory = {
        "asset_root": str(asset_root),
        "installed": asset_root.is_dir(),
        "scene_file_count": len(scenes),
        "scene_index_materialized": bool(scenes),
        "scene_index": scenes,
        "blocked_reason": None if scenes else "mp3d_pe is unavailable or contains no USD scene files",
    }
    write_json(output / "mp3d_pe_inventory.json", inventory)
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "controlled_visual_cases": len(controlled),
        "controlled_route_cases": sum(case["role"] == "route_choice" for case in controlled),
        "controlled_semantic_cases": sum(case["role"] == "semantic_stop" for case in controlled),
        "mp3d_pe_visual_cases": len(mp3d),
        "tracking_sequences": len(tracking),
        "route_planning_episodes": len(planning["route"]),
        "semantic_planning_episodes": len(planning["semantic"]),
        "oracle_context_leakage": sum(row["oracle_context_leakage"] for row in audits),
        "mp3d_pe_ready": bool(scenes),
        "qualification_evidence": False,
        "locomotion_fidelity": "ideal_kinematic",
    }
    write_json(output / "generation_summary.json", summary)
    return summary


def evaluate(output: Path, results: Path) -> dict[str, Any]:
    controlled_rows = read_json(results / "controlled_visual_results.json") if (results / "controlled_visual_results.json").is_file() else []
    mp3d_rows = read_json(results / "mp3d_pe_visual_results.json") if (results / "mp3d_pe_visual_results.json").is_file() else []
    track_rows = read_json(results / "tracking_results.json") if (results / "tracking_results.json").is_file() else []
    route_rows = read_json(results / "route_planning_results.json") if (results / "route_planning_results.json").is_file() else []
    semantic_rows = read_json(results / "semantic_planning_results.json") if (results / "semantic_planning_results.json").is_file() else []
    robustness_rows = read_json(results / "robustness_results.json") if (results / "robustness_results.json").is_file() else []
    controlled_gate = visual_gate(evaluate_visual_cases(controlled_rows), domain="controlled")
    mp3d_gate = visual_gate(evaluate_visual_cases(mp3d_rows), domain="mp3d_pe")
    tracking_result = tracking_gate(evaluate_tracking_sequences(track_rows))
    planning_result = planning_gate(evaluate_planning(route_rows, semantic_rows))
    robustness_result = evaluate_robustness(robustness_rows)
    generation = read_json(output / "generation_summary.json")
    gates = {
        "controlled_visual": controlled_gate,
        "mp3d_pe_visual": mp3d_gate,
        "tracking": tracking_result,
        "planning": planning_result,
        "robustness": robustness_result,
    }
    screening_allowed = bool(generation.get("mp3d_pe_ready")) and all(bool(gate.get("pass")) for gate in gates.values())
    result = {
        "schema_version": 1,
        "pass": screening_allowed,
        "screening_allowed": screening_allowed,
        "paired_confirmation_allowed": False,
        "gates": gates,
        "qualification_evidence": False,
        "locomotion_fidelity": "ideal_kinematic",
        "sim2real_gate": "NOT READY FOR REAL ROBOT AUTONOMY",
        "value_claim": "unproven for true multimodal full route-stop",
        "latency_policy": {
            "step_request_p95_max_sec": STEP_REQUEST_LATENCY_P95_MAX_SEC,
            "threshold_to_safe_stop_p95_max_sec": THRESHOLD_TO_SAFE_STOP_P95_MAX_SEC,
            "step_request_timeout_behavior": "scan_or_stop; never blind forward",
        },
    }
    write_json(output / "gate.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and gate the pure perception/tracking/planning suite.")
    parser.add_argument("--output", default="")
    parser.add_argument("--asset-root", default="/home/song/InternNav/data/scene_data/mp3d_pe")
    parser.add_argument("--results", default="")
    parser.add_argument("--controlled-micro", default="")
    parser.add_argument("--mp3d-micro", default="")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else ROOT / "runs" / f"perception_planning_suite_{stamp}"
    summary = generate(output, Path(args.asset_root))
    results_path = Path(args.results) if args.results else output / "formal_results"
    if args.controlled_micro or args.mp3d_micro:
        results_path.mkdir(parents=True, exist_ok=True)
    if args.controlled_micro:
        write_json(
            results_path / "controlled_visual_results.json",
            visual_rows_from_micro(read_json(Path(args.controlled_micro))),
        )
    if args.mp3d_micro:
        write_json(
            results_path / "mp3d_pe_visual_results.json",
            visual_rows_from_micro(read_json(Path(args.mp3d_micro))),
        )
    has_results = bool(args.results or args.controlled_micro or args.mp3d_micro)
    result = evaluate(output, results_path) if has_results else {
        "pass": False,
        "screening_allowed": False,
        "blocked_reason": "formal result files have not been supplied",
        "mp3d_pe_ready": summary["mp3d_pe_ready"],
        "qualification_evidence": False,
        "locomotion_fidelity": "ideal_kinematic",
        "sim2real_gate": "NOT READY FOR REAL ROBOT AUTONOMY",
        "latency_policy": {
            "step_request_p95_max_sec": STEP_REQUEST_LATENCY_P95_MAX_SEC,
            "threshold_to_safe_stop_p95_max_sec": THRESHOLD_TO_SAFE_STOP_P95_MAX_SEC,
            "step_request_timeout_behavior": "scan_or_stop; never blind forward",
        },
    }
    if not has_results:
        write_json(output / "gate.json", result)
    write_artifact_manifest(output, run_id=output.name, metadata={"generation": summary, "gate": result})
    print(json.dumps({"output": str(output), "generation": summary, "gate": result}, indent=2, ensure_ascii=False))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
