#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from isaac_vln_benchmark.compositional_progress_analysis import (
    analyze_compositional_run,
    write_compositional_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze V19 per-subgoal compositional progress.")
    parser.add_argument("--run", required=True, help="Directory containing per-task episode directories.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-ids", nargs="*", default=[])
    args = parser.parse_args()

    result = analyze_compositional_run(args.run, task_ids=args.task_ids)
    write_compositional_analysis(result, args.output)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
