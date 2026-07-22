#!/usr/bin/env python3
"""Fail-closed entry point for T0.1 Isaac + Nav2 oracle evaluation.

The current recorded T0.0 result is BLOCKED, so this command must stop before
importing ROS or changing Isaac.  Once T0.0 is replaced by a passed result, the
script performs the runtime dependency checks and requires the dedicated ROS
episode backend; it never falls back to MP3D/R2R or a mock simulator.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omninav_nav2.gates import GateBlocked, require_passed_gate, write_blocked_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-status", default="results/omninav_t0/gate_status.json")
    parser.add_argument("--output", default="results/omninav_t0/t0_1_nav2_oracle")
    parser.add_argument("--episode-backend", default="omninav_nav2.ros_runtime:IsaacNav2OracleBackend")
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()
    try:
        require_passed_gate(args.gate_status, "T0.0")
    except GateBlocked as exc:
        path = write_blocked_result(args.output, gate="T0.1", required_gate="T0.0", reason=str(exc))
        print(f"T0.1 NOT RUN: {exc}; evidence={path}", file=sys.stderr)
        return 3
    if not args.manifest or not Path(args.manifest).is_file():
        raise SystemExit("a real Isaac OVON diagnostic/dataset manifest is required")
    module_name, separator, class_name = args.episode_backend.partition(":")
    if not separator:
        raise SystemExit("--episode-backend must be module:Class")
    try:
        backend_type = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"real Isaac/Nav2 backend is unavailable: {exc}") from exc
    backend = backend_type(output=Path(args.output))
    summary = backend.run_oracle(Path(args.manifest))
    if summary.get("mock") is not False or summary.get("real_isaac") is not True:
        raise SystemExit("backend did not prove real Isaac execution")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
