#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCH_PACKAGE = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
SCHEDULER_PACKAGE = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler"
SCRIPTS = ROOT / "scripts"
for source_path in (BENCH_PACKAGE, SCHEDULER_PACKAGE, SCRIPTS):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from isaac_vln_benchmark.compositional_micro_gate import evaluate_v19_compositional_micro
from isaac_vln_benchmark.compositional_progress_analysis import (
    analyze_compositional_run,
    write_compositional_analysis,
)
from isaac_vln_benchmark.config_loader import load_data
from isaac_vln_benchmark.semantic_navigation_benchmark import load_task_set
from run_v18_forced_semantic_oracle_upper_bound import (
    audit_model_evidence,
    ensure_dgx_omninav,
    monitor_dgx,
    write_csv,
    write_manifest,
)


def write_episode_table(output: Path, metrics: dict[str, Any], task_set: dict[str, Any]) -> None:
    tasks = {str(task["task_id"]): task for task in task_set["tasks"]}
    rows = []
    for episode in metrics.get("episodes", []):
        task = tasks.get(str(episode.get("task_id") or ""), {})
        runtime = task.get("semantic_runtime") or {}
        rows.append(
            {
                "task_id": episode.get("task_id"),
                "parent_task_id": runtime.get("parent_task_id"),
                "micro_stage": runtime.get("micro_stage"),
                "seed": task.get("seed"),
                "success": bool(episode.get("success", False)),
                "failure_reason": episode.get("failure_reason") or "",
                "mission_time_sec": episode.get("mission_time_sec"),
                "path_length_m": episode.get("path_length_m"),
                "subgoals_completed": episode.get("semantic_subgoals_completed"),
                "subgoals_expected": episode.get("semantic_subgoals_expected"),
                "real_omninav_calls": episode.get("num_omninav_real_image_calls"),
                "fallback_calls": episode.get("num_omninav_fallback_calls"),
                "semantic_errors": episode.get("semantic_errors"),
                "collision": episode.get("num_collisions"),
                "runtime_stale": episode.get("runtime_stale"),
                "timebase_error": episode.get("timebase_error"),
                "stale_action_executed": episode.get("stale_action_executed"),
            }
        )
    fields = list(rows[0]) if rows else ["task_id"]
    with (output / "episode_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# V19 Compositional Local Progress Micro",
        "",
        f"- stage: `{gate['micro_stage']}`",
        f"- gate: **{'PASS' if gate['pass'] else 'FAIL'}**",
        f"- success: `{gate['success']}/{gate['episodes_expected']}`",
        f"- parent success: `{json.dumps(gate['parent_success'], sort_keys=True)}`",
        f"- model evidence: `{'PASS' if gate['model_evidence_gate'] else 'FAIL'}`",
        f"- safety: `{json.dumps(gate['safety'], sort_keys=True)}`",
        f"- failure_top1: `{gate['failure_top1']}`",
        "- Step remains frozen; this micro never unlocks formal screening by itself.",
        "- execution chain: forced oracle -> stale gate -> semantic executive -> OmniNav -> primitive_executor -> safe_mux -> Isaac",
        "- judge geometry is oracle-only and never enters the controller request.",
        "- qualification_evidence: `false`",
        "- Sim2Real: `NOT READY FOR REAL ROBOT AUTONOMY`",
    ]
    (output / "V19_COMPOSITIONAL_MICRO_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V19 compositional progress micro suite.")
    parser.add_argument("--stage", choices=("single", "chain"), required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--analyze-existing", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=18)
    parser.add_argument("--isaac-host", default="10.100.120.111")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--dgx-host", default="10.100.100.128")
    parser.add_argument("--dgx-user", default="railgun")
    parser.add_argument("--dgx-password", default="spark")
    parser.add_argument("--plink", default=r"C:\Program Files\PuTTY\plink.exe")
    parser.add_argument("--omninav-startup-timeout-sec", type=float, default=180.0)
    args = parser.parse_args(argv)

    config = ROOT / "configs" / f"live_success_v19_compositional_{args.stage}.yaml"
    config_doc = load_data(config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else ROOT / "runs" / f"v19_compositional_{args.stage}_{stamp}"
    if not output.is_absolute():
        output = ROOT.parent / output if output.parts and output.parts[0] == ROOT.name else ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    return_code = 0
    monitor_rows: list[dict[str, Any]] = []
    if not args.analyze_existing:
        ensure_dgx_omninav(args)
        stop = threading.Event()
        monitor = threading.Thread(target=monitor_dgx, args=(stop, monitor_rows, args), daemon=True)
        monitor.start()
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_live_success_benchmark.py"),
            "--config", str(config),
            "--modes", "forced_semantic_oracle",
            "--max-episodes", str(args.max_episodes),
            "--output", str(output),
            "--real-omninav",
            "--scheduler-config", str(ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler" / "config" / "scheduler_isaac_real_models.yaml"),
            "--isaac-host", args.isaac_host,
            "--isaac-user", args.isaac_user,
            "--isaac-password", args.isaac_password,
            "--dgx-user", args.dgx_user,
            "--dgx-password", args.dgx_password,
        ]
        if args.isaac_hostkey:
            command += ["--isaac-hostkey", args.isaac_hostkey]
        try:
            return_code = subprocess.run(command, check=False).returncode
        finally:
            stop.set()
            monitor.join(timeout=20)
            write_csv(output / "dgx_performance.csv", monitor_rows)

    metrics_path = output / "metrics.json"
    selected_tasks_path = output / "selected_tasks.yaml"
    if not metrics_path.is_file() or not selected_tasks_path.is_file():
        raise FileNotFoundError("micro run did not produce metrics.json and selected_tasks.yaml")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    task_set = load_task_set(selected_tasks_path, require_full=False)
    model_evidence = audit_model_evidence(output, metrics)
    gate = evaluate_v19_compositional_micro(metrics, task_set, model_evidence, config_doc.get("gates", {}))
    gate["launcher_return_code"] = return_code
    gate["model_evidence"] = model_evidence
    (output / "v19_compositional_micro_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    write_episode_table(output, metrics, task_set)
    episode_root = output / "forced_semantic_oracle"
    if episode_root.is_dir():
        progress = analyze_compositional_run(episode_root)
        write_compositional_analysis(progress, output / "progress_analysis")
    write_report(output, gate)
    if not (output / "dgx_performance.csv").exists():
        write_csv(output / "dgx_performance.csv", monitor_rows)
    write_manifest(output, gate)
    print(json.dumps({"output": str(output), "gate": gate}, indent=2))
    if return_code:
        return return_code
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
