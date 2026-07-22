#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.v11_value_utils import evaluate_v11


MODES = ["omninav_only_v11_matched", "omninav_forced_route_stop_oracle_v11"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def monitor_dgx(stop: threading.Event, rows: list[dict], args: argparse.Namespace) -> None:
    if not args.dgx_user or not args.dgx_password:
        return
    command = (
        "nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu "
        "--format=csv,noheader,nounits; printf 'SYS,'; cat /proc/loadavg; free -m | awk 'NR==2{print \"MEM,\"$3\",\"$2}'"
    )
    while not stop.is_set():
        started = time.time()
        try:
            output = subprocess.check_output(
                [
                    args.plink,
                    "-batch",
                    "-ssh",
                    "-pw",
                    args.dgx_password,
                    f"{args.dgx_user}@{args.dgx_host}",
                    command,
                ],
                text=True,
                timeout=10,
                stderr=subprocess.STDOUT,
            )
            gpu_lines = [line for line in output.splitlines() if line and line[0].isdigit()]
            sys_line = next((line for line in output.splitlines() if line.startswith("SYS,")), "")
            mem_line = next((line for line in output.splitlines() if line.startswith("MEM,")), "")
            load = sys_line.split(",", 1)[1].split()[0] if "," in sys_line else ""
            mem_parts = mem_line.split(",")
            for line in gpu_lines:
                values = [item.strip() for item in line.split(",")]
                if len(values) >= 7:
                    rows.append(
                        {
                            "timestamp": started,
                            "gpu_index": values[0],
                            "gpu_util_pct": values[1],
                            "memory_util_pct": values[2],
                            "memory_used_mib": values[3],
                            "memory_total_mib": values[4],
                            "power_w": values[5],
                            "temperature_c": values[6],
                            "load1": load,
                            "ram_used_mib": mem_parts[1] if len(mem_parts) > 2 else "",
                            "ram_total_mib": mem_parts[2] if len(mem_parts) > 2 else "",
                        }
                    )
        except Exception as exc:
            rows.append({"timestamp": started, "error": repr(exc)})
        stop.wait(5.0)


def write_monitor(path: Path, rows: list[dict]) -> None:
    headers = sorted({key for row in rows for key in row}) or ["timestamp", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(output: Path, evaluation: dict) -> None:
    names = [
        "config.yaml", "selected_tasks.yaml", "metrics.json", "summary.md", "v11_evaluation.json",
        "gate.json", "helped_hurt_no_effect.csv", "stale_attribution_v11.csv", "dgx_performance.csv",
    ]
    manifest = {
        "schema_version": 1,
        "run_id": output.name,
        "gate": evaluation,
        "artifacts": [
            {"path": name, "sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in names if (output / name).is_file()
        ],
    }
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run matched V11 OmniNav-only versus forced route/stop oracle.")
    parser.add_argument("--output", default="")
    parser.add_argument("--analyze-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isaac-host", default="10.100.120.111")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--dgx-host", default="10.100.100.128")
    parser.add_argument("--dgx-user", default="railgun")
    parser.add_argument("--dgx-password", default="spark")
    parser.add_argument("--plink", default=r"C:\Program Files\PuTTY\plink.exe")
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else ROOT / "runs" / f"v11_forced_oracle_upper_bound_{stamp}"
    if not output.is_absolute():
        output = ROOT.parent / output if output.parts and output.parts[0] == ROOT.name else ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    phase1_gate = ROOT / "runs" / "phase1_semantic_waypoint_brake_j_20260711" / "gate.json"
    phase1 = json.loads(phase1_gate.read_text(encoding="utf-8")) if phase1_gate.is_file() else {}
    if not phase1.get("pass"):
        raise SystemExit("Phase 1 semantic gate is not proven; V11 is not allowed")
    config = ROOT / "configs" / "live_success_v11_forced_oracle.yaml"
    if args.dry_run:
        print(json.dumps({"output": str(output), "config": str(config), "modes": MODES, "episodes": 30}, indent=2))
        return 0
    return_code = 0
    monitor_rows: list[dict] = []
    stop = threading.Event()
    monitor = None
    if not args.analyze_existing:
        monitor = threading.Thread(target=monitor_dgx, args=(stop, monitor_rows, args), daemon=True)
        monitor.start()
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_live_success_benchmark.py"),
            "--config", str(config),
            "--modes", *MODES,
            "--max-episodes", "30",
            "--output", str(output),
            "--real-omninav",
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
            monitor.join(timeout=15)
            write_monitor(output / "dgx_performance.csv", monitor_rows)
    evaluation = evaluate_v11(output)
    if not (output / "dgx_performance.csv").exists():
        write_monitor(output / "dgx_performance.csv", monitor_rows)
    write_manifest(output, evaluation)
    print(json.dumps({"output": str(output), "evaluation": evaluation}, indent=2, ensure_ascii=False))
    if return_code:
        return return_code
    return 0 if evaluation["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
