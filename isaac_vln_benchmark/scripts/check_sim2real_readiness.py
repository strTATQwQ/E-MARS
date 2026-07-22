#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"))

from isaac_vln_benchmark.v3_benchmark_utils import evaluate_sim2real_readiness, parse_summary_metrics
from isaac_vln_benchmark.v4_benchmark_utils import evaluate_sim2real_readiness_v4


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_metrics(path: Path) -> dict[str, Any]:
    if path.is_dir():
        for name in ("sim2real_gate_input.json", "metrics.json"):
            candidate = path / name
            if candidate.exists():
                return json.loads(candidate.read_text(encoding="utf-8"))
        summary = path / "summary.md"
        if summary.exists():
            return parse_summary_metrics(summary)
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return parse_summary_metrics(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--route-result", default="")
    parser.add_argument("--stop-result", default="")
    parser.add_argument("--config", default=str(ROOT / "configs" / "sim2real_gate.yaml"))
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    gate = load_yaml(Path(args.config)).get("required", {})
    metrics = load_metrics(Path(args.input))
    if args.route_result:
        route = load_metrics(Path(args.route_result))
        metrics["route_choice_correct_branch_rate"] = route.get(
            "route_choice_correct_branch_rate", metrics.get("route_choice_correct_branch_rate")
        )
        metrics["mock_models"] = bool(metrics.get("mock_models", False) or route.get("mock_models", False))
    if args.stop_result:
        stop = load_metrics(Path(args.stop_result))
        metrics["semantic_stop_accuracy"] = stop.get("semantic_stop_accuracy", metrics.get("semantic_stop_accuracy"))
        metrics["visible_to_stop_latency_sec"] = stop.get(
            "visible_to_stop_latency_sec", metrics.get("visible_to_stop_latency_sec")
        )
        metrics["mock_models"] = bool(metrics.get("mock_models", False) or stop.get("mock_models", False))
    if any(key.startswith("forced_") or key.startswith("step_") for key in gate.keys()):
        result = evaluate_sim2real_readiness_v4(metrics, gate)
    else:
        result = evaluate_sim2real_readiness(metrics, gate)
    text = [
        f"# Sim2Real Readiness: {result['status']}",
        "",
        "## Metrics",
        json.dumps(metrics, indent=2, ensure_ascii=False),
        "",
        "## Failures",
    ]
    text.extend([f"- {item}" for item in result["failures"]] or ["- none"])
    text.append("")
    text.append("## Allowed Next Steps")
    text.extend([f"- {item}" for item in result["allowed_next_steps"]])
    text.append("")
    text.append("## Disallowed Next Steps")
    text.extend([f"- {item}" for item in result["disallowed_next_steps"]])
    rendered = "\n".join(text) + "\n"
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
