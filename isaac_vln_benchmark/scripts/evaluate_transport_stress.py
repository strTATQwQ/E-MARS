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

from isaac_vln_benchmark.transport_stress_utils import evaluate_transport_profile, evaluate_transport_stress, write_transport_stress


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency", required=True)
    parser.add_argument("--packet-loss", required=True)
    parser.add_argument("--reset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    profiles = [
        evaluate_transport_profile(Path(args.latency), profile="latency", requested_delay_sec=0.20, requested_drop_rate=0.0),
        evaluate_transport_profile(Path(args.packet_loss), profile="packet_loss", requested_delay_sec=0.0, requested_drop_rate=1.0),
        evaluate_transport_profile(Path(args.reset), profile="reset", requested_delay_sec=1.0, requested_drop_rate=0.0, reset_expected=True),
    ]
    result = evaluate_transport_stress(profiles)
    write_transport_stress(Path(args.output), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
