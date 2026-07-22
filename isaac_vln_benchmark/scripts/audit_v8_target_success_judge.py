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

from isaac_vln_benchmark.v8_coverage_utils import audit_target_success_judge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit V8 target aliases and success judge target ids.")
    parser.add_argument("--tasks", default=str(ROOT / "configs" / "tasks.yaml"))
    parser.add_argument("--scenes", default=str(ROOT / "configs" / "scenes.yaml"))
    parser.add_argument("--output", default=str(ROOT / "runs" / "v8_target_success_judge_audit"))
    args = parser.parse_args(argv)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    result = audit_target_success_judge(Path(args.tasks), Path(args.scenes), output)
    print(json.dumps({"output": str(output), "target_success_judge_audit": result}, indent=2, ensure_ascii=False))
    return 0 if result.get("pass", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
