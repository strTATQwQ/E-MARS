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
from run_v8_route_oracle_coverage import common_live_args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V8 deterministic route injection diagnostic.")
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
    out_dir = Path(args.output) if args.output else ROOT / "runs" / f"v8_forced_route_injection_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.analyze_existing:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_live_success_benchmark.py"),
            "--config",
            str(ROOT / "configs" / "v8_forced_route_injection.yaml"),
            "--modes",
            "omninav_forced_route_injection_v8",
            "--max-episodes",
            str(args.max_episodes),
            "--output",
            str(out_dir),
        ] + common_live_args(args)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return result.returncode

    metrics = postprocess_route_coverage_run(out_dir, deterministic=True)
    summary = metrics.get("v8_route_coverage", {})
    print(json.dumps({"output": str(out_dir), "deterministic_route_injection": summary}, indent=2, ensure_ascii=False))
    return 0 if args.mock_models or summary.get("pass", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
