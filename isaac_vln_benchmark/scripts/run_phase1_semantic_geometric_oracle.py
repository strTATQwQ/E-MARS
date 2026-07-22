#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.v10_geometry_utils import postprocess_target_approach_run
from run_v10_branch_entry_controller import add_args, resolve_output
from run_v8_route_oracle_coverage import common_live_args


MODES = [
    "phase1_semantic_pure_forward",
    "phase1_semantic_pure_rotate",
    "phase1_semantic_rotate_then_forward",
    "phase1_semantic_proportional",
    "phase1_semantic_waypoint",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_gate_artifacts(output: Path, metrics: dict) -> dict:
    summary = dict(metrics.get("v10_target_relative_approach") or {})
    phase1_pass = bool(summary.get("pass"))
    gate = {
        "schema_version": 1,
        "phase": "semantic_geometric_oracle",
        "pass": phase1_pass,
        "best_controller": summary.get("best_controller"),
        "semantic": summary,
        "v11_forced_oracle_full": "ALLOWED" if phase1_pass else "NOT ALLOWED",
        "step": "FROZEN",
        "real_robot": "DISABLED",
        "sim2real": "NOT READY",
    }
    gate_path = output / "gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifact_names = [
        "config.yaml",
        "selected_tasks.yaml",
        "metrics.json",
        "summary.md",
        "target_approach_metrics.csv",
        "controller_trace.csv",
        "trajectory.csv",
        "stale_attribution.csv",
        "failure_table.csv",
        "gate.json",
    ]
    manifest = {
        "schema_version": 1,
        "run_id": output.name,
        "gate": gate,
        "artifacts": [
            {"path": name, "sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in artifact_names
            if (output / name).is_file()
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run low-speed semantic geometric oracle micro-scenes.")
    add_args(parser, 15)
    args = parser.parse_args(argv)
    output = resolve_output(args.output, "phase1_semantic_geometric_oracle")
    config = Path(args.config) if args.config else ROOT / "configs" / "phase1_semantic_geometric_oracle.yaml"
    if args.live and not args.mock_models and not args.isaac_host:
        args.isaac_host = "10.100.120.111"
    if args.dry_run:
        print(json.dumps({"dry_run": True, "output": str(output), "config": str(config), "modes": MODES}, indent=2))
        return 0
    if not args.analyze_existing:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_live_success_benchmark.py"),
            "--config",
            str(config),
            "--modes",
            *MODES,
            "--max-episodes",
            str(args.max_episodes),
            "--output",
            str(output),
        ] + common_live_args(args)
        return_code = subprocess.run(command, check=False).returncode
        if return_code:
            return int(return_code)
    metrics = postprocess_target_approach_run(output)
    gate = write_gate_artifacts(output, metrics)
    print(json.dumps({"output": str(output), "gate": gate}, indent=2, ensure_ascii=False))
    return 0 if args.mock_models or gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
