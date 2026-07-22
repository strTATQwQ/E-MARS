#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.config_loader import load_data
from isaac_vln_benchmark.v8_coverage_utils import (
    analyze_stale_attribution_v8,
    evaluate_sim2real_v8,
    load_json,
    load_jsonl,
    render_sim2real_gate_v8,
)


def run_dir(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_run(path: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path is None:
        return {}, []
    metrics = load_json(path / "metrics.json")
    events = load_jsonl(path / "events.jsonl")
    return metrics, events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate conservative V8 sim2real readiness.")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--route-run-dir", default="")
    parser.add_argument("--stop-run-dir", default="")
    parser.add_argument("--gate-config", default=str(ROOT / "configs" / "sim2real_gate_v8.yaml"))
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    route_dir = run_dir(args.route_run_dir) or run_dir(args.run_dir)
    stop_dir = run_dir(args.stop_run_dir) or run_dir(args.run_dir)
    route_metrics, route_events = load_run(route_dir)
    stop_metrics, stop_events = load_run(stop_dir)
    events = route_events + ([] if stop_dir == route_dir else stop_events)
    combined: dict[str, Any] = {
        "output_dir": str(route_dir or stop_dir or ROOT),
        "episodes": route_metrics.get("episodes", []) + ([] if stop_dir == route_dir else stop_metrics.get("episodes", [])),
        "v8_route_coverage": route_metrics.get("v8_route_coverage", {}),
        "v8_semantic_stop_coverage": stop_metrics.get("v8_semantic_stop_coverage", {}),
        "v8_stale_attribution": analyze_stale_attribution_v8(events),
        "_events": events,
    }
    gate = load_data(args.gate_config)
    result = evaluate_sim2real_v8(combined, gate)
    output = Path(args.output) if args.output else (route_dir or stop_dir or ROOT / "runs") / "sim2real_gate_v8.md"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_sim2real_gate_v8(result, combined), encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "result": result}, indent=2, ensure_ascii=False))
    return 0 if result.get("ready", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
