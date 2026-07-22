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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_from_run(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return load_json(path / "metrics.json")
    return load_json(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-result", required=True)
    parser.add_argument("--stop-result", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "live_success_small_v3.yaml"))
    parser.add_argument("--output", default="")
    parser.add_argument("--isaac-host", default="")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--internnav-server", default="http://10.100.100.128:8087")
    parser.add_argument("--dgx-user", default="")
    parser.add_argument("--dgx-password", default="")
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--execute-live", action="store_true", help="run the live Isaac benchmark after gates pass")
    args = parser.parse_args(argv)

    route = metrics_from_run(Path(args.route_result))
    stop = metrics_from_run(Path(args.stop_result))
    gates_pass = bool(route.get("pass")) and bool(stop.get("pass"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else ROOT / "runs" / f"live_success_small_v3_guard_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    guard = {
        "route_choice_correct_branch_rate": route.get("route_choice_correct_branch_rate"),
        "semantic_stop_accuracy": stop.get("semantic_stop_accuracy"),
        "visible_to_stop_latency_sec": stop.get("visible_to_stop_latency_sec"),
        "gates_pass": gates_pass,
        "mock_models": bool(args.mock_models),
        "execute_live": bool(args.execute_live),
    }
    (out_dir / "metrics.json").write_text(json.dumps(guard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not gates_pass:
        (out_dir / "summary.md").write_text(
            "# live_success_small_v3 Guard\n\n"
            "- status: SKIPPED\n"
            "- reason: route_choice_v3 or semantic_stop_v3 gate did not pass.\n"
            "- Step value is not yet proven.\n",
            encoding="utf-8",
        )
        print(json.dumps(guard | {"status": "SKIPPED"}, indent=2, ensure_ascii=False))
        return 2

    if not args.execute_live and not args.mock_models:
        (out_dir / "summary.md").write_text(
            "# live_success_small_v3 Guard\n\n"
            "- status: READY_NOT_EXECUTED\n"
            "- reason: gates pass, but --execute-live was not provided.\n"
            "- no real Go2 motion interface was used.\n",
            encoding="utf-8",
        )
        print(json.dumps(guard | {"status": "READY_NOT_EXECUTED"}, indent=2, ensure_ascii=False))
        return 0

    runner = ROOT / "scripts" / "run_live_success_benchmark.py"
    cmd = [
        sys.executable,
        str(runner),
        "--config",
        str(ROOT / "configs" / "live_success_small_v3.yaml"),
        "--modes",
        "omninav_only",
        "omninav_route_stop_v3",
        "step_route_stop_v3",
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
    status = "EXECUTED" if result.returncode == 0 else "FAILED"
    (out_dir / "summary.md").write_text(
        "# live_success_small_v3 Guard\n\n"
        f"- status: {status}\n"
        f"- command_returncode: {result.returncode}\n"
        f"- benchmark_dir: {out_dir / 'benchmark'}\n",
        encoding="utf-8",
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
