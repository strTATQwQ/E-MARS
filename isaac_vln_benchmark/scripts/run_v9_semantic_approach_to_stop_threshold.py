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

from isaac_vln_benchmark.v9_progress_utils import postprocess_semantic_approach_run
from run_v9_route_trigger_reachability import add_common_args, common_live_args, resolve_output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V9 semantic approach-to-stop-threshold probe.")
    add_common_args(parser, default_episodes=9)
    args = parser.parse_args(argv)
    if args.live and not args.mock_models and not args.isaac_host:
        args.isaac_host = "10.100.120.111"

    out_dir = resolve_output(args.output, "v9_semantic_approach_to_stop_threshold")
    if not args.analyze_existing:
        modes = ["v9_semantic_forced_approach", "v9_semantic_omninav_approach", "v9_semantic_omninav_with_approach_assist"]
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_live_success_benchmark.py"),
            "--config",
            str(ROOT / "configs" / "v9_semantic_approach_to_stop_threshold.yaml"),
            "--modes",
            *modes,
            "--max-episodes",
            str(args.max_episodes),
            "--output",
            str(out_dir),
        ] + common_live_args(args)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return int(result.returncode)

    metrics = postprocess_semantic_approach_run(out_dir)
    summary = metrics.get("v9_semantic_approach_to_stop", {})
    print(json.dumps({"output": str(out_dir), "semantic_approach_to_stop": summary}, indent=2, ensure_ascii=False))
    return 0 if args.mock_models or summary.get("pass", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
