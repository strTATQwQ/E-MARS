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

from isaac_vln_benchmark.v10_geometry_utils import postprocess_branch_controller_run
from run_v8_route_oracle_coverage import common_live_args


def add_args(parser: argparse.ArgumentParser, default_episodes: int) -> None:
    parser.add_argument("--config", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-episodes", type=int, default=default_episodes)
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--real-omninav", action="store_true")
    parser.add_argument("--analyze-existing", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isaac-host", default="")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--internnav-server", default="http://10.100.100.128:8087")
    parser.add_argument("--dgx-user", default="")
    parser.add_argument("--dgx-password", default="")


def resolve_output(value: str, prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(value) if value else ROOT / "runs" / f"{prefix}_{stamp}"
    if not out.is_absolute():
        out = ROOT.parent / out if out.parts and out.parts[0] == ROOT.name else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V10 branch-entry geometric controller.")
    add_args(parser, 24)
    args = parser.parse_args(argv)
    out = resolve_output(args.output, "v10_branch_entry")
    config = Path(args.config) if args.config else ROOT / "configs" / "v10_branch_entry_controller.yaml"
    modes = ["v10_branch_yaw60_fwd08", "v10_branch_yaw75_fwd08", "v10_branch_yaw90_fwd08", "v10_branch_yaw90_fwd12"]
    if args.live and not args.mock_models and not args.isaac_host:
        args.isaac_host = "10.100.120.111"
    if args.dry_run:
        print(json.dumps({"dry_run": True, "output": str(out), "config": str(config), "modes": modes}, indent=2))
        return 0
    if not args.analyze_existing:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_live_success_benchmark.py"),
            "--config",
            str(config),
            "--modes",
            *modes,
            "--max-episodes",
            str(args.max_episodes),
            "--output",
            str(out),
        ] + common_live_args(args)
        rc = subprocess.run(cmd, check=False).returncode
        if rc:
            return int(rc)
    metrics = postprocess_branch_controller_run(out)
    summary = metrics.get("v10_branch_entry_controller", {})
    print(json.dumps({"output": str(out), "branch_entry": summary}, indent=2, ensure_ascii=False))
    return 0 if args.mock_models or summary.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
