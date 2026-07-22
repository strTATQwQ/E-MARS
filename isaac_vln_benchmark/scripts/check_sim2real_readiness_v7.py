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

from isaac_vln_benchmark.config_loader import load_data
from isaac_vln_benchmark.v7_value_utils import evaluate_sim2real_v7, load_json, postprocess_v7_run, render_sim2real_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate conservative V7 sim2real readiness.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gate-config", default=str(ROOT / "configs" / "sim2real_gate_v7.yaml"))
    parser.add_argument("--output", default="")
    parser.add_argument("--postprocess", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    metrics = postprocess_v7_run(run_dir, run_kind="sim2real_gate_input") if args.postprocess else load_json(run_dir / "metrics.json")
    gate_path = Path(args.gate_config)
    if not gate_path.is_absolute() and not gate_path.exists():
        gate_path = ROOT / gate_path
    gate = load_data(gate_path)
    result = evaluate_sim2real_v7(metrics, gate)
    output = Path(args.output) if args.output else run_dir / "sim2real_gate_v7.md"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_sim2real_gate(result, metrics), encoding="utf-8")
    (run_dir / "sim2real_gate_v7.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "result": result}, indent=2, ensure_ascii=False))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
