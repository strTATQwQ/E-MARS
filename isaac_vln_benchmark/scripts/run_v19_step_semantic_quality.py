#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from isaac_vln_benchmark.step_semantic_quality import evaluate_step_semantic_quality
from run_v18_forced_semantic_oracle_upper_bound import ensure_dgx_omninav, write_csv, write_manifest
from run_v19_compositional_micro import monitor_dgx


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output: Path, gate: dict) -> None:
    lines = [
        "# V19 Real-Step Semantic Quality Micro",
        "",
        f"- gate: **{'PASS' if gate['pass'] else 'FAIL'}**",
        f"- closed-loop success: `{gate['closed_loop_success']}/{gate['episodes']}`",
        f"- type accuracy: `{gate['type_accuracy']:.3f}`",
        f"- target/relation accuracy: `{gate['target_relation_accuracy']:.3f}`",
        f"- sequence accuracy: `{gate['sequence_accuracy']:.3f}`",
        f"- recovery accuracy: `{gate['recovery_accuracy']:.3f}`",
        f"- Step latency p95: `{gate['step_latency_p95_sec']:.3f}s` (gate `<= {gate['latency_gate_sec']:.1f}s`)",
        f"- responses: `{gate['accepted_responses']}/{gate['responses']}` strict accepted",
        f"- completion evidence client defaults: `{gate['completion_evidence_defaulted_responses']}`",
        f"- fresh actual viewport: `{gate['fresh_image_responses']}/{gate['responses']}`",
        f"- oracle leakage findings: `{len(gate['oracle_leakage_findings'])}`",
        f"- safety: `{json.dumps(gate['safety'], sort_keys=True)}`",
        f"- Step full screening allowed: `{gate['step_full_screening_allowed']}`",
        "- scope: instrumented public marker language; this is not open-vocabulary semantic grounding evidence.",
        "- local navigation and obstacle avoidance remain owned by OmniNav.",
        "- true multimodal OmniNav+Step route-stop value remains UNPROVEN.",
        "- qualification_evidence: `false`; Sim2Real remains NOT READY.",
    ]
    (output / "V19_STEP_SEMANTIC_QUALITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V19 real-Step semantic quality micro.")
    parser.add_argument("--output", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--analyze-existing", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=6)
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

    config = Path(args.config) if args.config else ROOT / "configs" / "live_success_v19_step_semantic_quality.yaml"
    if not config.is_absolute():
        config = ROOT / config
    scheduler = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler" / "config" / "scheduler_isaac_real_models.yaml"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else ROOT / "runs" / f"v19_step_semantic_quality_{stamp}"
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
            "--modes", "omninav_step_semantic_executive",
            "--max-episodes", str(args.max_episodes),
            "--output", str(output),
            "--real-omninav",
            "--real-step",
            "--scheduler-config", str(scheduler),
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

    if not (output / "metrics.json").is_file():
        raise FileNotFoundError("Step semantic micro did not produce metrics.json")
    gate = evaluate_step_semantic_quality(output)
    gate["launcher_return_code"] = return_code
    (output / "v19_step_semantic_quality_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    _write_rows(output / "semantic_subgoal_comparisons.csv", gate["comparisons"])
    _write_rows(output / "semantic_sequence_results.csv", gate["sequence_rows"])
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
