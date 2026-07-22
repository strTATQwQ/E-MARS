#!/usr/bin/env python3
"""Materialize an opt-in T5 Nvblox runtime without starting ROS or CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "internvla_t4_sensors"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from internvla_t4_sensors.t5_nvblox_runtime import (  # noqa: E402
    T5NvbloxContractError,
    materialize_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("off", "shadow", "active_local_gt"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nav2-base", type=Path)
    parser.add_argument("--lane-namespace", choices=("/t5/lane_a", "/t5/lane_b"))
    args = parser.parse_args()
    try:
        payload = materialize_profile(
            ROOT,
            args.output_dir,
            args.mode,
            args.nav2_base,
            args.lane_namespace,
        )
    except (OSError, T5NvbloxContractError, ValueError) as exc:
        print(
            json.dumps({"schema_version": 1, "status": "FAIL", "error": str(exc)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
