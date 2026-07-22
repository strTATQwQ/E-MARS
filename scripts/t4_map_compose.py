#!/usr/bin/env python3
"""Render a fresh launch bundle for the managed completion companion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from t4_completion.map.composer import ComposeRequest, compose_bundle
from t4_completion.map.contract import DEFAULT_CONFIG_DIR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--nvblox-mode", choices=("shadow", "active"), default="shadow")
    parser.add_argument(
        "--nvblox-health", choices=("ready", "failed", "unknown"), default="unknown"
    )
    parser.add_argument("--target", default="isaac_simulation_only")
    parser.add_argument("--runtime-policy", default="completion_sim")
    args = parser.parse_args()
    try:
        payload = compose_bundle(
            args.output_dir,
            ComposeRequest(
                requested_mode=args.nvblox_mode,
                nvblox_health=args.nvblox_health,
                target=args.target,
                runtime_policy=args.runtime_policy,
            ),
            args.config_dir,
        )
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"schema_version": 1, "status": "FAIL", "error": str(exc)},
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
