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

from isaac_vln_benchmark.v9_progress_utils import postprocess_route_reachability_run
from run_v8_route_oracle_coverage import common_live_args


def add_common_args(parser: argparse.ArgumentParser, *, default_episodes: int) -> None:
    parser.add_argument("--output", default="")
    parser.add_argument("--max-episodes", type=int, default=default_episodes)
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--real-omninav", action="store_true")
    parser.add_argument("--analyze-existing", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--isaac-host", default="")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--internnav-server", default="http://10.100.100.128:8087")
    parser.add_argument("--dgx-user", default="")
    parser.add_argument("--dgx-password", default="")


def resolve_output(value: str, prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(value) if value else ROOT / "runs" / f"{prefix}_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_config(args: argparse.Namespace, *, out_dir: Path, config_name: str, modes: list[str]) -> int:
    if args.live and not args.mock_models and not args.isaac_host:
        args.isaac_host = "10.100.120.111"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_live_success_benchmark.py"),
        "--config",
        str(ROOT / "configs" / config_name),
        "--modes",
        *modes,
        "--max-episodes",
        str(args.max_episodes),
        "--output",
        str(out_dir),
    ] + common_live_args(args)
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V9 route trigger reachability probe.")
    add_common_args(parser, default_episodes=18)
    args = parser.parse_args(argv)

    out_dir = resolve_output(args.output, "v9_route_trigger_reachability")
    if not args.analyze_existing:
        rc = run_config(
            args,
            out_dir=out_dir,
            config_name="v9_route_trigger_reachability.yaml",
            modes=["v9_forced_forward_until_trigger", "v9_omninav_until_trigger", "v9_omninav_with_progress_assist"],
        )
        if rc != 0:
            return rc

    metrics = postprocess_route_reachability_run(out_dir)
    summary = metrics.get("v9_route_trigger_reachability", {})
    print(json.dumps({"output": str(out_dir), "route_trigger_reachability": summary}, indent=2, ensure_ascii=False))
    return 0 if args.mock_models or summary.get("pass", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
