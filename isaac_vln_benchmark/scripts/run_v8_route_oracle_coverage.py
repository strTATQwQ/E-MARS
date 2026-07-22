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

from isaac_vln_benchmark.v8_coverage_utils import postprocess_route_coverage_run


def common_live_args(args: argparse.Namespace) -> list[str]:
    cmd: list[str] = []
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
    parser = argparse.ArgumentParser(description="Run V8 forced route oracle coverage diagnostic.")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-episodes", type=int, default=6)
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--real-omninav", action="store_true")
    parser.add_argument("--analyze-existing", action="store_true")
    parser.add_argument("--isaac-host", default="")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--internnav-server", default="http://10.100.100.128:8087")
    parser.add_argument("--dgx-user", default="")
    parser.add_argument("--dgx-password", default="")
    args = parser.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else ROOT / "runs" / f"v8_route_oracle_coverage_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.analyze_existing:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_live_success_benchmark.py"),
            "--config",
            str(ROOT / "configs" / "v8_route_oracle_coverage.yaml"),
            "--modes",
            "omninav_forced_route_stop_oracle_v8",
            "--max-episodes",
            str(args.max_episodes),
            "--output",
            str(out_dir),
        ] + common_live_args(args)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return result.returncode

    metrics = postprocess_route_coverage_run(out_dir, deterministic=False)
    summary = metrics.get("v8_route_coverage", {})
    print(json.dumps({"output": str(out_dir), "route_coverage": summary}, indent=2, ensure_ascii=False))
    return 0 if args.mock_models or summary.get("pass", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
