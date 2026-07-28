#!/usr/bin/env python3
"""Validate coordinator-produced T4.3 localization Oracle artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from t4_completion.localization.validation import write_validation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=REPOSITORY_ROOT
        / "configs/completion_sim/localization/oracle_gate.json",
    )
    arguments = parser.parse_args()
    try:
        payload = write_validation(arguments.result_dir, arguments.gate_config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
