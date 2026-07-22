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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCH_PACKAGE = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
SCHEDULER_PACKAGE = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler"
for source_path in (BENCH_PACKAGE, SCHEDULER_PACKAGE):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from isaac_vln_benchmark.semantic_navigation_benchmark import (
    evaluate_semantic_navigation_run,
    load_task_set,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plink_command(args: argparse.Namespace, host: str, user: str, password: str, remote: str) -> list[str]:
    command = [args.plink, "-batch", "-ssh"]
    if host == args.isaac_host and args.isaac_hostkey:
        command += ["-hostkey", args.isaac_hostkey]
    return command + ["-pw", password, f"{user}@{host}", remote]


def ensure_dgx_omninav(args: argparse.Namespace) -> None:
    log = "/home/railgun/dgx-unitree/remote_dgx/logs/v18_omninav_model_client.log"
    config = (
        "/home/railgun/dgx-unitree/ros2_ws/install/omninav_step_scheduler/"
        "share/omninav_step_scheduler/config/scheduler_isaac_real_models.yaml"
    )
    remote = (
        "mkdir -p /home/railgun/dgx-unitree/remote_dgx/logs; "
        "if ! pgrep -f '[/]lib/omninav_step_scheduler/omninav_model_client_node' >/dev/null; then "
        "nohup setsid bash -lc 'source /opt/ros/jazzy/setup.bash && "
        "source /home/railgun/dgx-unitree/ros2_ws/install/setup.bash && "
        "export ROS_DOMAIN_ID=42 && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && "
        f"exec ros2 run omninav_step_scheduler omninav_model_client_node --ros-args -p config_file:={config}' "
        f"> {log} 2>&1 < /dev/null & "
        "else pgrep -f '[/]lib/omninav_step_scheduler/omninav_model_client_node' | head -1; fi"
    )
    subprocess.check_call(plink_command(args, args.dgx_host, args.dgx_user, args.dgx_password, remote))
    deadline = time.monotonic() + float(args.omninav_startup_timeout_sec)
    while time.monotonic() < deadline:
        probe = (
            "if ! pgrep -f '[/]lib/omninav_step_scheduler/omninav_model_client_node' >/dev/null; then echo PROCESS_EXITED; "
            f"elif grep -q 'OmniNav model loaded' {log} 2>/dev/null; then echo READY; "
            f"elif grep -q 'OmniNav model load failed' {log} 2>/dev/null; then echo LOAD_FAILED; "
            "else echo LOADING; fi"
        )
        state = subprocess.check_output(
            plink_command(args, args.dgx_host, args.dgx_user, args.dgx_password, probe),
            text=True,
            timeout=15,
        ).strip()
        if state == "READY":
            return
        if state in {"PROCESS_EXITED", "LOAD_FAILED"}:
            raise RuntimeError(f"DGX OmniNav startup failed: {state}")
        time.sleep(5.0)
    raise TimeoutError("DGX OmniNav did not become ready before startup timeout")


def monitor_dgx(stop: threading.Event, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    remote = (
        "nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu "
        "--format=csv,noheader,nounits; printf 'SYS,'; cat /proc/loadavg; "
        "free -m | awk 'NR==2{print \"MEM,\"$3\",\"$2}'"
    )
    while not stop.is_set():
        stamp = time.time()
        try:
            output = subprocess.check_output(
                plink_command(args, args.dgx_host, args.dgx_user, args.dgx_password, remote),
                text=True,
                timeout=15,
                stderr=subprocess.STDOUT,
            )
            gpu_lines = [line for line in output.splitlines() if line and line[0].isdigit()]
            sys_line = next((line for line in output.splitlines() if line.startswith("SYS,")), "")
            mem_line = next((line for line in output.splitlines() if line.startswith("MEM,")), "")
            load1 = sys_line.split(",", 1)[1].split()[0] if "," in sys_line else ""
            mem = mem_line.split(",")
            for gpu_line in gpu_lines:
                values = [value.strip() for value in gpu_line.split(",")]
                if len(values) >= 7:
                    rows.append(
                        {
                            "timestamp": stamp,
                            "gpu_index": values[0],
                            "gpu_util_pct": values[1],
                            "memory_util_pct": values[2],
                            "memory_used_mib": values[3],
                            "memory_total_mib": values[4],
                            "power_w": values[5],
                            "temperature_c": values[6],
                            "load1": load1,
                            "ram_used_mib": mem[1] if len(mem) > 2 else "",
                            "ram_total_mib": mem[2] if len(mem) > 2 else "",
                        }
                    )
        except Exception as exc:
            rows.append({"timestamp": stamp, "error": repr(exc)})
        stop.wait(5.0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row}) or ["timestamp", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_records(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for episode in metrics.get("episodes", []):
        records.append(
            {
                "task_id": episode.get("task_id"),
                "success": bool(episode.get("success", False)),
                "clean": bool(episode.get("clean_success", False)),
                "semantic_errors": int(episode.get("semantic_errors", 0) or 0),
                "collision": int(episode.get("num_collisions", 0) or 0),
                "runtime_stale": int(episode.get("runtime_stale", 0) or 0),
                "timebase_error": int(episode.get("timebase_error", 0) or 0),
                "stale_action_executed": int(episode.get("stale_action_executed", 0) or 0),
                "step_calls": 0,
                "fresh_multimodal_calls": 0,
                "real_omninav_calls": int(episode.get("num_omninav_real_image_calls", 0) or 0),
                "omninav_fallback_calls": int(episode.get("num_omninav_fallback_calls", 0) or 0),
                "failure_reason": episode.get("failure_reason"),
            }
        )
    return records


def audit_model_evidence(output: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    by_task: dict[str, dict[str, int]] = {}
    for calls_path in output.glob("*/*/model_calls.csv"):
        counts = {"real_image_calls": 0, "fallback_calls": 0, "total_omninav_calls": 0}
        with calls_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("model") or "") != "omninav":
                    continue
                source = str(row.get("source") or "")
                counts["total_omninav_calls"] += 1
                counts["real_image_calls"] += int(source == "omninav_model_client:ros_image")
                counts["fallback_calls"] += int("fallback" in source.lower())
        by_task[calls_path.parent.name] = counts
    for episode in metrics.get("episodes", []):
        counts = by_task.get(str(episode.get("task_id") or ""), {})
        episode["num_omninav_real_image_calls"] = int(counts.get("real_image_calls", 0))
        episode["num_omninav_fallback_calls"] = int(counts.get("fallback_calls", 0))
    audit = {
        "schema_version": 1,
        "episodes_audited": len(by_task),
        "episodes_with_real_image_calls": sum(value["real_image_calls"] > 0 for value in by_task.values()),
        "real_image_calls": sum(value["real_image_calls"] for value in by_task.values()),
        "fallback_calls": sum(value["fallback_calls"] for value in by_task.values()),
        "total_omninav_calls": sum(value["total_omninav_calls"] for value in by_task.values()),
        "pass": bool(by_task) and all(
            value["real_image_calls"] > 0 and value["fallback_calls"] == 0 for value in by_task.values()
        ),
        "by_task": by_task,
    }
    (output / "omninav_model_evidence_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


def write_episode_tables(output: Path, metrics: dict[str, Any], task_set: dict[str, Any]) -> None:
    categories = {str(task["task_id"]): str(task["category"]) for task in task_set["tasks"]}
    rows = []
    for episode in metrics.get("episodes", []):
        rows.append(
            {
                "task_id": episode.get("task_id"),
                "category": categories.get(str(episode.get("task_id") or ""), "unknown"),
                "success": bool(episode.get("success", False)),
                "failure_reason": episode.get("failure_reason") or "",
                "mission_time_sec": episode.get("mission_time_sec"),
                "path_length_m": episode.get("path_length_m"),
                "semantic_subgoals_completed": episode.get("semantic_subgoals_completed"),
                "semantic_subgoals_expected": episode.get("semantic_subgoals_expected"),
                "semantic_recovery_required": episode.get("semantic_recovery_required"),
                "semantic_recovery_observed": episode.get("semantic_recovery_observed"),
                "semantic_recovery_matched": episode.get("semantic_recovery_matched"),
                "semantic_errors": episode.get("semantic_errors"),
                "omninav_real_image_calls": episode.get("num_omninav_real_image_calls"),
                "omninav_fallback_calls": episode.get("num_omninav_fallback_calls"),
                "stale_discards": int(episode.get("num_stale_omninav_actions", 0) or 0)
                + int(episode.get("num_stale_step_results", 0) or 0),
                "runtime_stale": episode.get("runtime_stale"),
                "timebase_error": episode.get("timebase_error"),
                "stale_action_executed": episode.get("stale_action_executed"),
                "collision": episode.get("num_collisions"),
            }
        )
    fields = list(rows[0]) if rows else ["task_id"]
    with (output / "episode_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if not row["success"]]
    with (output / "failure_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)


def write_report(output: Path, gate: dict[str, Any], return_code: int) -> None:
    status = "PASS" if gate.get("pass") else "FAIL"
    lines = [
        "# V18 Forced Semantic Oracle Isaac Upper Bound",
        "",
        f"- gate: **{status}**",
        f"- launcher return code: `{return_code}`",
        f"- episodes: `{gate.get('episodes_scored', 0)}/30`",
        f"- success: `{gate.get('success', 0)}/30`",
        f"- clean: `{gate.get('clean', 0)}/30`",
        f"- category success: `{json.dumps(gate.get('category_success', {}), sort_keys=True)}`",
        f"- safety: `{json.dumps(gate.get('safety', {}), sort_keys=True)}`",
        f"- failure_top1: `{gate.get('failure_top1', 'unknown')}`",
        "- semantic decision owner: forced semantic oracle",
        "- local navigation and obstacle avoidance owner: real OmniNav",
        "- execution chain: stale_gate -> semantic_executive -> OmniNav -> primitive_executor -> safe_mux -> Isaac",
        "- scene evidence: instrumented semantic markers with oracle-only geometry judge",
        "- qualification_evidence: `false` (ideal-kinematic Isaac)",
        "- true multimodal OmniNav+Step value remains unproven; this run only tests the forced semantic upper bound.",
        "- Sim2Real: `NOT READY FOR REAL ROBOT AUTONOMY`",
    ]
    (output / "V18_FORCED_SEMANTIC_ORACLE_UPPER_BOUND_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_manifest(output: Path, gate: dict[str, Any]) -> None:
    artifacts = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json" or "_remote" in path.parts:
            continue
        if path.stat().st_size > 256 * 1024 * 1024:
            continue
        artifacts.append(
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": output.name,
        "mode": "forced_semantic_oracle",
        "gate": gate,
        "local_navigation_owner": "OmniNav",
        "direct_semantic_motion_output_allowed": False,
        "normal_execution_chain_required": True,
        "instrumented_semantic_markers": True,
        "qualification_evidence": False,
        "artifacts": artifacts,
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V18 real-OmniNav forced semantic oracle upper bound.")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "live_success_v18_forced_semantic_oracle.yaml"),
    )
    parser.add_argument("--max-episodes", type=int, default=30)
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

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else ROOT / "runs" / f"v18_forced_semantic_oracle_live_{stamp}"
    if not output.is_absolute():
        output = ROOT.parent / output if output.parts and output.parts[0] == ROOT.name else ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    monitor_rows: list[dict[str, Any]] = []
    return_code = 0
    if not args.analyze_existing:
        ensure_dgx_omninav(args)
        stop = threading.Event()
        monitor = threading.Thread(target=monitor_dgx, args=(stop, monitor_rows, args), daemon=True)
        monitor.start()
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_live_success_benchmark.py"),
            "--config",
            str(Path(args.config)),
            "--modes",
            "forced_semantic_oracle",
            "--max-episodes",
            str(args.max_episodes),
            "--output",
            str(output),
            "--real-omninav",
            "--scheduler-config",
            str(ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler" / "config" / "scheduler_isaac_real_models.yaml"),
            "--isaac-host",
            args.isaac_host,
            "--isaac-user",
            args.isaac_user,
            "--isaac-password",
            args.isaac_password,
            "--dgx-user",
            args.dgx_user,
            "--dgx-password",
            args.dgx_password,
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
        raise FileNotFoundError("live benchmark did not produce metrics.json and selected_tasks.yaml")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_evidence = audit_model_evidence(output, metrics)
    task_set = load_task_set(selected_tasks_path, require_full=False)
    write_episode_tables(output, metrics, task_set)
    records = normalized_records(metrics)
    gate = evaluate_semantic_navigation_run(records, task_set, mode="forced_semantic_oracle")
    gate.update(
        {
            "launcher_return_code": return_code,
            "real_omninav_required": True,
            "omninav_model_evidence_audit": model_evidence,
            "instrumented_semantic_markers": True,
            "geometry_judge_is_oracle_only": True,
            "qualification_evidence": False,
            "value_claim": "unproven",
            "sim2real": "NOT READY FOR REAL ROBOT AUTONOMY",
        }
    )
    (output / "forced_semantic_oracle_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(output, gate, return_code)
    if not (output / "dgx_performance.csv").exists():
        write_csv(output / "dgx_performance.csv", monitor_rows)
    write_manifest(output, gate)
    print(json.dumps({"output": str(output), "gate": gate}, indent=2, ensure_ascii=False))
    if return_code:
        return return_code
    return 0 if gate.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
