#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.config_loader import load_data
from isaac_vln_benchmark.reporting import load_run_metrics, write_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    metrics = load_run_metrics(run_dir)
    config = load_data(args.config) if args.config else load_data(run_dir / "config.yaml")
    path = write_summary(run_dir, metrics, config)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
