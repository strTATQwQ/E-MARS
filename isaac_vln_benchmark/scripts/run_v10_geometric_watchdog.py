#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.v10_geometry_utils import postprocess_geometric_watchdog_run
from run_v10_branch_entry_controller import add_args, resolve_output
from run_v8_route_oracle_coverage import common_live_args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V10 staged geometric watchdog.")
    add_args(parser, 9)
    args = parser.parse_args(argv)
    out = resolve_output(args.output, "v10_geometric_watchdog")
    config = Path(args.config) if args.config else ROOT / "configs" / "v10_geometric_watchdog.yaml"
    modes = ["v10_geometric_watchdog"]
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
    metrics = postprocess_geometric_watchdog_run(out)
    print(json.dumps({"output": str(out), "geometric_watchdog": metrics.get("v10_geometric_watchdog", {})}, indent=2, ensure_ascii=False))
    return 0 if args.mock_models else 0


if __name__ == "__main__":
    raise SystemExit(main())
