#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.v7_value_utils import evaluate_golden_smoke, postprocess_v7_run, render_v7_summary


def add_live_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(ROOT / "configs" / "live_success_v7_golden_smoke.yaml"))
    parser.add_argument("--output", default="")
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--real-omninav", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--isaac-host", default="")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--internnav-server", default="http://10.100.100.128:8087")
    parser.add_argument("--dgx-user", default="")
    parser.add_argument("--dgx-password", default="")


def benchmark_cmd(args: argparse.Namespace, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_live_success_benchmark.py"),
        "--config",
        str(args.config),
        "--modes",
        "omninav_only_v6_golden",
        "--max-episodes",
        str(args.max_episodes),
        "--output",
        str(out_dir),
    ]
    if args.mock_models:
        cmd.append("--mock-models")
    if args.real_omninav:
        cmd.append("--real-omninav")
    if args.isaac_host and not args.mock_models:
        cmd += ["--isaac-host", args.isaac_host, "--isaac-user", args.isaac_user, "--isaac-password", args.isaac_password]
    if args.isaac_hostkey:
        cmd += ["--isaac-hostkey", args.isaac_hostkey]
    if args.internnav_server:
        cmd += ["--internnav-server", args.internnav_server]
    if args.dgx_user:
        cmd += ["--dgx-user", args.dgx_user]
    if args.dgx_password:
        cmd += ["--dgx-password", args.dgx_password]
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V7 OmniNav golden smoke before a V7 experiment.")
    add_live_args(parser)
    args = parser.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else ROOT / "runs" / f"v7_golden_smoke_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(benchmark_cmd(args, out_dir), check=False)
    metrics = postprocess_v7_run(out_dir, run_kind="golden_smoke")
    evaluation = evaluate_golden_smoke(metrics)
    (out_dir / "golden_smoke_result.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(render_v7_summary(out_dir, metrics, evaluation, title="V7 Golden Smoke"), encoding="utf-8")
    print(json.dumps({"output": str(out_dir), "evaluation": evaluation}, indent=2, ensure_ascii=False))
    if result.returncode != 0:
        return result.returncode
    return 0 if evaluation["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
