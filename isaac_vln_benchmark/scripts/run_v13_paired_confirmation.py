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
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from isaac_vln_benchmark.v12_step_value_utils import BASELINE_MODE, ROUTE_STOP_MODE
from isaac_vln_benchmark.v13_confirmation_utils import evaluate_v13_confirmation
from run_v12_step_value_screening import monitor_dgx, sha256, write_dgx_summary, write_monitor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run randomized 45-pair OmniNav+Step confirmation.")
    parser.add_argument("--screen-run", default=str(ROOT / "runs" / "v12_step_value_screening_final_20260711"))
    parser.add_argument("--output", default="")
    parser.add_argument("--analyze-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isaac-host", default="10.100.120.111")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--dgx-host", default="10.100.100.128")
    parser.add_argument("--dgx-user", default="railgun")
    parser.add_argument("--dgx-password", default="")
    parser.add_argument("--plink", default=r"C:\Program Files\PuTTY\plink.exe")
    parser.add_argument("--scheduler-config", default="")
    args = parser.parse_args(argv)
    screen_run = Path(args.screen_run)
    screen_gate = json.loads((screen_run / "gate.json").read_text(encoding="utf-8"))
    if not bool(screen_gate.get("pass")) or screen_gate.get("confirmation") != "ALLOWED":
        raise SystemExit("V12 screening gate has not allowed paired confirmation")
    candidate_mode = ROUTE_STOP_MODE
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else ROOT / "runs" / f"v13_paired_confirmation_{stamp}"
    if not output.is_absolute():
        output = ROOT.parent / output if output.parts and output.parts[0] == ROOT.name else ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    config = ROOT / "configs" / "live_success_v13_paired_confirmation.yaml"
    if args.dry_run:
        print(json.dumps({"output": str(output), "candidate": candidate_mode, "pairs": 45, "episodes": 90}, indent=2))
        return 0

    rows: list[dict] = []
    stop = threading.Event()
    launcher_return_code = 0
    if not args.analyze_existing:
        monitor = threading.Thread(target=monitor_dgx, args=(stop, rows, args), daemon=True)
        monitor.start()
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_live_success_benchmark.py"),
            "--config", str(config),
            "--modes", BASELINE_MODE, candidate_mode,
            "--max-episodes", "90",
            "--output", str(output),
            "--real-omninav",
            "--real-step",
            "--isaac-host", args.isaac_host,
            "--isaac-user", args.isaac_user,
            "--isaac-password", args.isaac_password,
            "--dgx-user", args.dgx_user,
            "--dgx-password", args.dgx_password,
        ]
        if args.scheduler_config:
            command += ["--scheduler-config", args.scheduler_config]
        if args.isaac_hostkey:
            command += ["--isaac-hostkey", args.isaac_hostkey]
        try:
            launcher_return_code = subprocess.run(command, check=False).returncode
        finally:
            stop.set()
            monitor.join(timeout=15)
            write_monitor(output / "dgx_performance.csv", rows)
    if not (output / "metrics.json").is_file():
        raise SystemExit(f"confirmation metrics missing after launcher exit {launcher_return_code}")
    evaluation = evaluate_v13_confirmation(output, candidate_mode)
    write_dgx_summary(output)
    artifacts = ["config.yaml", "selected_tasks.yaml", "metrics.json", "summary.md", "gate.json", "paired_results.csv", "dgx_performance.csv"]
    manifest = {
        "schema_version": 1,
        "run_id": output.name,
        "screen_run": screen_run.name,
        "screen_gate_sha256": sha256(screen_run / "gate.json"),
        "candidate_mode": candidate_mode,
        "launcher_return_code": launcher_return_code,
        "gate": evaluation,
        "artifacts": [
            {"path": name, "sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in artifacts
            if (output / name).is_file()
        ],
    }
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "evaluation": evaluation}, indent=2, ensure_ascii=False))
    return 0 if evaluation["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
