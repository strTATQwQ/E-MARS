#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.bag_replay_utils import read_rosbag2
from isaac_vln_benchmark.decision_replay_utils import evaluate_decision_replay


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline-score the decision transport rosbag.")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    result = evaluate_decision_replay(read_rosbag2(Path(args.bag)), expected)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision_replay_gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
