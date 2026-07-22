#!/usr/bin/env python3
"""Fail-closed T0.3 launcher for native OmniNav Slow/Fast + Nav2."""

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
    parser.add_argument("--output", default="results/omninav_t0/t0_3_full_fast_slow_nav2")
    parser.add_argument("--episode-backend", default="omninav_nav2.ros_runtime:FullFastSlowNav2Backend")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--phase", choices=("canary", "pilot", "formal"), default="canary")
    args = parser.parse_args()
    required = "T0.2" if args.phase in {"canary", "pilot"} else "T0.3"
    try:
        require_passed_gate(args.gate_status, required)
    except GateBlocked as exc:
        path = write_blocked_result(args.output, gate="T0.3", required_gate=required, reason=str(exc))
        print(f"T0.3 {args.phase} NOT RUN: {exc}; evidence={path}", file=sys.stderr)
        return 3
    if not args.manifest or not Path(args.manifest).is_file():
        raise SystemExit("a frozen OVON episode manifest is required")
    module_name, _, class_name = args.episode_backend.partition(":")
    try:
        backend_type = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"real Slow/Fast/Nav2 backend is unavailable: {exc}") from exc
    summary = backend_type(output=Path(args.output)).run(Path(args.manifest), phase=args.phase)
    if summary.get("mock") is not False or summary.get("real_isaac") is not True:
        raise SystemExit("backend did not prove real Isaac execution")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / f"{args.phase}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
