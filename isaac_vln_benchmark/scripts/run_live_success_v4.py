#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_metrics(path: str) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if p.is_dir():
        p = p / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forced-route-result", required=True)
    parser.add_argument("--forced-stop-result", required=True)
    parser.add_argument("--step-route-result", required=True)
    parser.add_argument("--step-stop-result", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "live_success_small_v4.yaml"))
    parser.add_argument("--output", default="")
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--isaac-host", default="")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--internnav-server", default="http://10.100.100.128:8087")
    parser.add_argument("--dgx-user", default="")
    parser.add_argument("--dgx-password", default="")
    args = parser.parse_args(argv)

    route = load_metrics(args.forced_route_result)
    stop = load_metrics(args.forced_stop_result)
    step_route = load_metrics(args.step_route_result)
    step_stop = load_metrics(args.step_stop_result)
    gates = {
        "forced_route_pass": bool(route.get("pass", False)),
        "forced_stop_pass": bool(stop.get("pass", False)),
        "step_route_pass": bool(step_route.get("pass", False)),
        "step_stop_pass": bool(step_stop.get("pass", False)),
    }
    gates_pass = all(gates.values())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else ROOT / "runs" / f"live_success_small_v4_guard_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    guard_metrics = {
        "benchmark": "live_success_small_v4_guard",
        **gates,
        "gates_pass": gates_pass,
        "forced_route_oracle_correct_branch_rate": route.get("forced_route_oracle_correct_branch_rate"),
        "forced_stop_oracle_accuracy": stop.get("forced_stop_oracle_accuracy"),
        "step_route_choice_correct_branch_rate": step_route.get("step_route_choice_correct_branch_rate"),
        "step_semantic_stop_accuracy": step_stop.get("step_semantic_stop_accuracy"),
        "visible_to_stop_latency_sec": step_stop.get("visible_to_stop_latency_sec", stop.get("visible_to_stop_latency_sec")),
    }
    (out_dir / "metrics.json").write_text(json.dumps(guard_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not gates_pass:
        (out_dir / "summary.md").write_text(
            "# live_success_small_v4 Guard\n\n"
            "- status: SKIPPED\n"
            "- reason: one or more v4 small experiment gates failed.\n"
            "- OmniNav+Step value remains unproven.\n",
            encoding="utf-8",
        )
        print(json.dumps(guard_metrics | {"status": "SKIPPED"}, indent=2, ensure_ascii=False))
        return 2
    if not args.execute_live and not args.mock_models:
        (out_dir / "summary.md").write_text(
            "# live_success_small_v4 Guard\n\n"
            "- status: READY_NOT_EXECUTED\n"
            "- reason: all small gates passed, but --execute-live was not provided.\n",
            encoding="utf-8",
        )
        print(json.dumps(guard_metrics | {"status": "READY_NOT_EXECUTED"}, indent=2, ensure_ascii=False))
        return 0

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_live_success_benchmark.py"),
        "--config",
        str(ROOT / "configs" / "live_success_small_v4.yaml"),
        "--modes",
        "omninav_only",
        "omninav_forced_route_stop_oracle",
        "step_route_stop_v4",
        "internnav_cma_baseline",
        "step_only_baseline",
        "--output",
        str(out_dir / "benchmark"),
    ]
    if args.isaac_host:
        cmd += ["--isaac-host", args.isaac_host, "--isaac-user", args.isaac_user, "--isaac-password", args.isaac_password]
    if args.isaac_hostkey:
        cmd += ["--isaac-hostkey", args.isaac_hostkey]
    if args.internnav_server:
        cmd += ["--internnav-server", args.internnav_server]
    if args.dgx_user:
        cmd += ["--dgx-user", args.dgx_user]
    if args.dgx_password:
        cmd += ["--dgx-password", args.dgx_password]
    if args.mock_models:
        cmd.append("--mock-models")
    result = subprocess.run(cmd, check=False)
    (out_dir / "summary.md").write_text(
        "# live_success_small_v4 Guard\n\n"
        f"- status: {'EXECUTED' if result.returncode == 0 else 'FAILED'}\n"
        f"- command_returncode: {result.returncode}\n"
        f"- benchmark_dir: {out_dir / 'benchmark'}\n",
        encoding="utf-8",
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
