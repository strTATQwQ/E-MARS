#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PACKAGE = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
SCHEDULER_PACKAGE = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler"
SCRIPTS = ROOT / "scripts"
for source_path in (BENCH_PACKAGE, SCHEDULER_PACKAGE, SCRIPTS):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from isaac_vln_benchmark.compositional_progress_analysis import (
    analyze_compositional_run,
    write_compositional_analysis,
)
from isaac_vln_benchmark.semantic_navigation_benchmark import (
    evaluate_semantic_navigation_run,
    load_task_set,
)
from run_v18_forced_semantic_oracle_upper_bound import (
    audit_model_evidence,
    ensure_dgx_omninav,
    monitor_dgx,
    normalized_records,
    write_csv,
    write_episode_tables,
    write_manifest,
)


def write_report(output: Path, gate: dict) -> None:
    lines = [
        "# V19 Forced Semantic Oracle Full30 Upper Bound",
        "",
        f"- gate: **{'PASS' if gate['forced_semantic_oracle_gate'] and gate['model_evidence_gate'] else 'FAIL'}**",
        f"- success: `{gate['success']}/{gate['episodes_expected']}`",
        f"- category success: `{json.dumps(gate['category_success'], sort_keys=True)}`",
        f"- model evidence: `{'PASS' if gate['model_evidence_gate'] else 'FAIL'}`",
        f"- safety: `{json.dumps(gate['safety'], sort_keys=True)}`",
        f"- failure_top1: `{gate['failure_top1']}`",
        f"- Step semantic micro allowed: `{gate['step_semantic_micro_allowed']}`",
        "- Step full screening remains locked until real Step semantic-quality micro passes.",
        "- local navigation owner: OmniNav; semantic decision owner: forced oracle.",
        "- execution: stale gate -> semantic executive -> OmniNav -> primitive_executor -> safe_mux -> Isaac.",
        "- scenes use instrumented semantic markers and an oracle-only judge.",
        "- true multimodal OmniNav+Step value remains UNPROVEN.",
        "- qualification_evidence: `false`; Sim2Real remains NOT READY.",
    ]
    (output / "V19_FORCED_SEMANTIC_ORACLE_UPPER_BOUND_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V19 forced semantic oracle full30 upper bound.")
    parser.add_argument("--output", default="")
    parser.add_argument("--analyze-existing", action="store_true")
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

    config = ROOT / "configs" / "live_success_v19_forced_semantic_oracle.yaml"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else ROOT / "runs" / f"v19_forced_semantic_oracle_full30_{stamp}"
    if not output.is_absolute():
        output = ROOT.parent / output if output.parts and output.parts[0] == ROOT.name else ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    return_code = 0
    monitor_rows: list[dict] = []
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
            "--max-episodes", "30",
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
    tasks_path = output / "selected_tasks.yaml"
    if not metrics_path.is_file() or not tasks_path.is_file():
        raise FileNotFoundError("full30 run did not produce metrics.json and selected_tasks.yaml")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    task_set = load_task_set(tasks_path, require_full=True)
    model_evidence = audit_model_evidence(output, metrics)
    write_episode_tables(output, metrics, task_set)
    gate = evaluate_semantic_navigation_run(normalized_records(metrics), task_set, mode="forced_semantic_oracle")
    upper_bound_pass = bool(gate["forced_semantic_oracle_gate"] and gate["model_evidence_gate"])
    gate.update(
        {
            "pass": upper_bound_pass,
            "screening_allowed": False,
            "step_semantic_micro_allowed": upper_bound_pass,
            "step_full_screening_allowed": False,
            "launcher_return_code": return_code,
            "runtime_profile": "v19_local_marker_v2",
            "model_evidence": model_evidence,
            "instrumented_semantic_markers": True,
            "geometry_judge_is_oracle_only": True,
            "value_claim": "unproven",
            "qualification_evidence": False,
            "sim2real": "NOT READY FOR REAL ROBOT AUTONOMY",
        }
    )
    (output / "v19_forced_semantic_oracle_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    episode_root = output / "forced_semantic_oracle"
    if episode_root.is_dir():
        write_compositional_analysis(analyze_compositional_run(episode_root), output / "compositional_progress")
    write_report(output, gate)
    if not (output / "dgx_performance.csv").exists():
        write_csv(output / "dgx_performance.csv", monitor_rows)
    write_manifest(output, gate)
    print(json.dumps({"output": str(output), "gate": gate}, indent=2))
    if return_code:
        return return_code
    return 0 if upper_bound_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
