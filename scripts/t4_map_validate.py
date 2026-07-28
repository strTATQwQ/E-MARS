#!/usr/bin/env python3
"""Validate the owned completion map configuration without ROS or Isaac."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from t4_completion.map.composer import ComposeRequest, validation_report
from t4_completion.map.contract import (
    DEFAULT_CONFIG_DIR,
    audit_runtime_isolation,
    load_and_validate,
)
from t4_completion.map.static_map import grid_summary


def _emit(payload: object, output: Path = None) -> None:
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        configs = load_and_validate(args.config_dir)
        reports = {
            "default": validation_report(ComposeRequest(), configs),
            "active_ready": validation_report(
                ComposeRequest(requested_mode="active", nvblox_health="ready"),
                configs,
            ),
            "active_failed": validation_report(
                ComposeRequest(requested_mode="active", nvblox_health="failed"),
                configs,
            ),
        }
        payload = {
            "schema_version": 1,
            "status": "PASS",
            "offline_only": True,
            "config_dir": str(args.config_dir.resolve()),
            "runtime_isolation": audit_runtime_isolation(ROOT),
            "static_map": grid_summary(configs.smoke_map),
            "profiles": reports,
        }
        _emit(payload, args.output)
        return 0
    except Exception as exc:
        _emit({"schema_version": 1, "status": "FAIL", "error": str(exc)})
        return 2


if __name__ == "__main__":
    sys.exit(main())
