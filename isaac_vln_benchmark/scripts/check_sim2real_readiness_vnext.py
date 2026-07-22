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

from isaac_vln_benchmark.sim2real_vnext_utils import evaluate_sim2real_vnext, write_sim2real_vnext


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).is_file() else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the unified post-value Sim2Real qualification gate.")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--route-micro", default=str(ROOT / "runs" / "step_route_choice_real_v11_final2_20260711" / "metrics.json"))
    parser.add_argument("--stop-micro", default=str(ROOT / "runs" / "step_stop_verify_real_v11_final_20260711" / "metrics.json"))
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--stress", default="")
    parser.add_argument("--replay", default="")
    parser.add_argument("--output", default=str(ROOT / "runs" / "sim2real_gate_vnext_current"))
    args = parser.parse_args(argv)
    result = evaluate_sim2real_vnext(
        value_confirmation=load(args.confirmation),
        route_micro=load(args.route_micro),
        stop_micro=load(args.stop_micro),
        independent_sessions=[load(path) for path in args.session],
        stress=load(args.stress),
        replay=load(args.replay),
    )
    output = Path(args.output)
    write_sim2real_vnext(output, result)
    print(json.dumps({"output": str(output), "gate": result}, indent=2, ensure_ascii=False))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
