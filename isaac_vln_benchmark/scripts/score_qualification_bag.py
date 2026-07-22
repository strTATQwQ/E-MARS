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

from isaac_vln_benchmark.bag_replay_utils import evaluate_bag_records, read_rosbag2, write_bag_replay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    result = evaluate_bag_records(read_rosbag2(Path(args.bag)), expected)
    result["bag_path"] = str(Path(args.bag))
    result["metrics_path"] = str(Path(args.metrics))
    write_bag_replay(Path(args.output), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
