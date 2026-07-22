#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.low_speed_qualification_utils import evaluate_low_speed_probe, write_low_speed_probe


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate actual Isaac motion under the 0.20 m/s qualification limit.")
    parser.add_argument("run_dir")
    parser.add_argument("--scheduler-config", required=True)
    args = parser.parse_args()
    output = Path(args.run_dir)
    result = evaluate_low_speed_probe(output, Path(args.scheduler_config))
    write_low_speed_probe(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
