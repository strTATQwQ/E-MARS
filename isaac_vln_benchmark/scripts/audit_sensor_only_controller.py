#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from isaac_vln_benchmark.sensor_only_audit import audit_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Reject simulator-truth leakage in sensor-only controller evidence.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit_run(args.run_dir)
    output = args.output or args.run_dir / "sensor_only_oracle_leakage_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = output.with_suffix(".md")
    summary.write_text(
        "# Sensor-only controller audit\n\n"
        f"- gate: {'PASS' if result['pass'] else 'FAIL'}\n"
        f"- controller events: {result['controller_event_count']}\n"
        f"- oracle context leakage: {result['oracle_context_leakage']}\n"
        f"- actual viewport evidence: {result['actual_viewport_evidence_count']}\n"
        f"- failures: {', '.join(result['failures']) if result['failures'] else 'none'}\n"
        "- qualification_evidence: false\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
