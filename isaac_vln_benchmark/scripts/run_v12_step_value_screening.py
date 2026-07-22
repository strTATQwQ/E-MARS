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
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.v12_step_value_utils import (
    BASELINE_MODE,
    ORACLE_MODE,
    ROUTE_MODE,
    ROUTE_STOP_MODE,
    SCREEN_MODES,
    STEP_ONLY_MODE,
    STOP_MODE,
    evaluate_v12_screening,
)


PREREQUISITES = {
    "v11": ROOT / "runs" / "v11_forced_oracle_upper_bound_final_20260711" / "gate.json",
    "step_route": ROOT / "runs" / "step_route_choice_real_v11_final2_20260711" / "metrics.json",
    "step_stop": ROOT / "runs" / "step_stop_verify_real_v11_final_20260711" / "metrics.json",
}
STEP_SCREEN_MODES = [ROUTE_MODE, STOP_MODE, ROUTE_STOP_MODE, STEP_ONLY_MODE]
CANONICAL_V11 = ROOT / "runs" / "v11_forced_oracle_upper_bound_final_20260711"


def require_prerequisites() -> dict[str, dict]:
    values: dict[str, dict] = {}
    for name, path in PREREQUISITES.items():
        if not path.is_file():
            raise SystemExit(f"missing prerequisite gate: {path}")
        values[name] = json.loads(path.read_text(encoding="utf-8"))
        if not bool(values[name].get("pass")):
            raise SystemExit(f"prerequisite gate did not pass: {name}")
    return values


def monitor_dgx(stop: threading.Event, rows: list[dict], args: argparse.Namespace) -> None:
    command = (
        "nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu "
        "--format=csv,noheader,nounits; printf 'SYS,'; cat /proc/loadavg; free -m | awk 'NR==2{print \"MEM,\"$3\",\"$2}'"
    )
    while not stop.is_set():
        started = time.time()
        try:
            output = subprocess.check_output(
                [args.plink, "-batch", "-ssh", "-pw", args.dgx_password, f"{args.dgx_user}@{args.dgx_host}", command],
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


def summarize_dgx_performance(path: Path) -> dict:
    rows = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    fields = [
        "gpu_util_pct",
        "memory_util_pct",
        "memory_used_mib",
        "power_w",
        "temperature_c",
        "load1",
        "ram_used_mib",
    ]
    summary = {"samples": len(rows), "errors": sum(1 for row in rows if row.get("error")), "metrics": {}}
    for field in fields:
        values = []
        for row in rows:
            try:
                values.append(float(row.get(field, "")))
            except (TypeError, ValueError):
                continue
        if values:
            ordered = sorted(values)
            rank = max(1, min(len(ordered), math.ceil(0.95 * len(ordered))))
            summary["metrics"][field] = {
                "mean": round(sum(values) / len(values), 3),
                "p95": round(ordered[rank - 1], 3),
                "max": round(max(values), 3),
            }
    return summary


def write_dgx_summary(output: Path) -> dict:
    summary = summarize_dgx_performance(output / "dgx_performance.csv")
    (output / "dgx_performance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = ["# DGX Performance Summary", "", f"- samples: {summary['samples']}", f"- monitor errors: {summary['errors']}", ""]
    for field, values in summary["metrics"].items():
        lines.append(f"- {field}: mean={values['mean']}, p95={values['p95']}, max={values['max']}")
    (output / "dgx_performance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reuse_canonical_v11_pair(output: Path) -> dict:
    current_metrics_path = output / "metrics.json"
    canonical_metrics_path = CANONICAL_V11 / "metrics.json"
    current = json.loads(current_metrics_path.read_text(encoding="utf-8"))
    canonical = json.loads(canonical_metrics_path.read_text(encoding="utf-8"))
    canonical_rows = [
        row for row in canonical.get("episodes", []) if row.get("mode") in {BASELINE_MODE, ORACLE_MODE}
    ]
    step_rows = [
        row for row in current.get("episodes", []) if row.get("mode") not in {BASELINE_MODE, ORACLE_MODE}
    ]
    current["episodes"] = canonical_rows + step_rows
    current["canonical_v11_pair_reused"] = True
    current_metrics_path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    selected_tasks = output / "selected_tasks.yaml"
    canonical_tasks = CANONICAL_V11 / "selected_tasks.yaml"
    audit = {
        "schema_version": 1,
        "canonical_run": CANONICAL_V11.name,
        "reused_modes": [BASELINE_MODE, ORACLE_MODE],
        "reused_episode_count": len(canonical_rows),
        "new_step_episode_count": len(step_rows),
        "canonical_gate_pass": bool(json.loads((CANONICAL_V11 / "gate.json").read_text(encoding="utf-8")).get("pass")),
        "canonical_metrics_sha256": sha256(canonical_metrics_path),
        "canonical_events_sha256": sha256(CANONICAL_V11 / "events.jsonl"),
        "canonical_tasks_sha256": sha256(canonical_tasks),
        "screening_tasks_sha256": sha256(selected_tasks),
        "task_manifest_identical": sha256(canonical_tasks) == sha256(selected_tasks),
        "event_storage": "canonical V11 events remain in canonical run; screening events.jsonl contains new Step modes only",
        "code_scope_audit": "V12 controller changes are activated only by controller_decision_source=step or semantic_stop_requires_step; reused V11 modes set neither flag",
    }
    (output / "canonical_v11_reuse_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if len(canonical_rows) != 30 or not audit["canonical_gate_pass"] or not audit["task_manifest_identical"]:
        raise SystemExit(f"canonical V11 reuse audit failed: {audit}")
    return audit


def write_manifest(output: Path, evaluation: dict, prerequisites: dict[str, dict]) -> None:
    names = [
        "config.yaml",
        "selected_tasks.yaml",
        "metrics.json",
        "summary.md",
        "v12_screening_evaluation.json",
        "gate.json",
        "helped_hurt_no_effect_v12.csv",
        "stale_attribution_v12.csv",
        "dgx_performance.csv",
        "remote_launcher_return_code.txt",
        "canonical_v11_reuse_audit.json",
        "dgx_performance_summary.json",
        "dgx_performance_summary.md",
    ]
    manifest = {
        "schema_version": 1,
        "run_id": output.name,
        "prerequisites": {name: bool(value.get("pass")) for name, value in prerequisites.items()},
        "gate": evaluation,
        "artifacts": [
            {"path": name, "sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in names
            if (output / name).is_file()
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def with_scheduler_config(command: list[str], scheduler_config: str) -> list[str]:
    if not scheduler_config:
        return command
    return [*command, "--scheduler-config", scheduler_config]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run matched real-Step screening for OmniNav route-stop value.")
    parser.add_argument("--output", default="")
    parser.add_argument("--analyze-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--rerun-v11-pair", action="store_true")
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
    prerequisites = require_prerequisites()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else ROOT / "runs" / f"v12_step_value_screening_{stamp}"
    if not output.is_absolute():
        output = ROOT.parent / output if output.parts and output.parts[0] == ROOT.name else ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    modes = args.modes or (SCREEN_MODES if args.rerun_v11_pair else STEP_SCREEN_MODES)
    max_episodes = args.max_episodes or (15 * len(modes))
    config = ROOT / "configs" / "live_success_v12_step_screening.yaml"
    if args.dry_run:
        print(json.dumps({"output": str(output), "config": str(config), "modes": modes, "episodes": max_episodes, "reuse_v11_pair": not args.rerun_v11_pair}, indent=2))
        return 0

    rows: list[dict] = []
    stop = threading.Event()
    monitor = None
    launcher_return_code = 0
    if not args.analyze_existing:
        monitor = threading.Thread(target=monitor_dgx, args=(stop, rows, args), daemon=True)
        monitor.start()
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_live_success_benchmark.py"),
            "--config", str(config),
            "--modes", *modes,
            "--max-episodes", str(max_episodes),
            "--output", str(output),
            "--real-omninav",
            "--real-step",
            "--isaac-host", args.isaac_host,
            "--isaac-user", args.isaac_user,
            "--isaac-password", args.isaac_password,
            "--dgx-user", args.dgx_user,
            "--dgx-password", args.dgx_password,
        ]
        command = with_scheduler_config(command, args.scheduler_config)
        if args.isaac_hostkey:
            command += ["--isaac-hostkey", args.isaac_hostkey]
        try:
            launcher_return_code = subprocess.run(command, check=False).returncode
        finally:
            stop.set()
            monitor.join(timeout=15)
            write_monitor(output / "dgx_performance.csv", rows)

    if not (output / "metrics.json").is_file():
        raise SystemExit(f"benchmark metrics missing after launcher exit {launcher_return_code}: {output}")
    if not args.rerun_v11_pair and not args.modes:
        reuse_canonical_v11_pair(output)
    evaluation = evaluate_v12_screening(output)
    if not (output / "dgx_performance.csv").is_file():
        write_monitor(output / "dgx_performance.csv", rows)
    write_dgx_summary(output)
    evaluation["launcher_return_code"] = launcher_return_code
    write_manifest(output, evaluation, prerequisites)
    print(json.dumps({"output": str(output), "evaluation": evaluation}, indent=2, ensure_ascii=False))
    return 0 if evaluation["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
