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

from isaac_vln_benchmark.semantic_value_screening import (
    SCREEN_MODES,
    evaluate_v20_screen,
    materialize_v20_screen_tasks,
)
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
        "# V20 Instrumented Semantic Value Screening",
        "",
        f"- gate: **{'PASS' if gate['pass'] else 'FAIL'}**",
    ]
    for mode, summary in gate["modes"].items():
        lines.append(
            f"- `{mode}`: success `{summary['success']}/15`, clean `{summary['clean']}/15`, "
            f"category `{json.dumps(summary['category_success'], sort_keys=True)}`"
        )
    evidence = gate["step_evidence"]
    lines.extend(
        [
            f"- Step vs baseline: success delta `{gate['success_delta']:+d}`, clean delta `{gate['clean_delta']:+d}`, "
            f"helped `{gate['helped']}`, hurt `{gate['hurt']}`",
            f"- Step responses: `{evidence['accepted_responses']}/{evidence['responses']}` accepted; fresh multimodal "
            f"`{evidence['fresh_multimodal_responses']}/{evidence['responses']}`",
            f"- Step p95 latency: `{evidence['step_latency_p95_sec']:.3f}s` (gate `<=7.0s`)",
            f"- gates: `{json.dumps(gate['gates'], sort_keys=True)}`",
            f"- 45-pair confirmation allowed: `{gate['paired_confirmation_allowed']}`",
            "- scope: instrumented markers with explicit numbered semantic stages; this does not prove open-vocabulary natural-language navigation.",
            "- local obstacle avoidance remains OmniNav-owned; Step never outputs motion or geometry.",
            "- full natural-language multimodal value remains UNPROVEN until a matching asset-rich task domain and 45-pair confirmation pass.",
            "- qualification_evidence: false; Sim2Real remains NOT READY FOR REAL ROBOT AUTONOMY.",
        ]
    )
    (output / "V20_SEMANTIC_VALUE_SCREENING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V20 matched semantic-executive value screen.")
    parser.add_argument("--output", default="")
    parser.add_argument("--analyze-existing", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=15)
    parser.add_argument("--isaac-host", default="10.100.120.111")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--dgx-host", default="10.100.100.128")
    parser.add_argument("--dgx-user", default="railgun")
    parser.add_argument("--dgx-password", default="")
    parser.add_argument("--plink", default=r"C:\Program Files\PuTTY\plink.exe")
    parser.add_argument("--omninav-startup-timeout-sec", type=float, default=180.0)
    args = parser.parse_args(argv)

    materialize_v20_screen_tasks(
        ROOT / "configs" / "generated" / "natural_language_navigation_v19_tasks.yaml",
        ROOT / "configs" / "generated" / "v20_semantic_value_screen_tasks.yaml",
    )
    config = ROOT / "configs" / "live_success_v20_semantic_value_screening.yaml"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else ROOT / "runs" / f"v20_semantic_value_screening_{stamp}"
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
            "--modes", *SCREEN_MODES,
            "--max-episodes", str(args.max_episodes),
            "--output", str(output),
            "--real-omninav",
            "--real-step",
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

    if not (output / "metrics.json").is_file():
        raise FileNotFoundError("V20 screening did not produce metrics.json")
    gate = evaluate_v20_screen(output)
    gate["launcher_return_code"] = return_code
    (output / "v20_semantic_value_screening_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    _write_rows(output / "paired_outcomes.csv", gate["paired"])
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
