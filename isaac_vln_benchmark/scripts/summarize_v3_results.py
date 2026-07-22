#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_metrics(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if p.is_dir():
        p = p / "metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-result", default="")
    parser.add_argument("--stop-result", default="")
    parser.add_argument("--visual-result", default="")
    parser.add_argument("--live-result", default="")
    parser.add_argument("--gate-result", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    route = read_metrics(args.route_result)
    stop = read_metrics(args.stop_result)
    visual = read_metrics(args.visual_result)
    live = read_metrics(args.live_result)
    gate_text = Path(args.gate_result).read_text(encoding="utf-8") if args.gate_result and Path(args.gate_result).exists() else ""
    step_value_proven = bool(route.get("pass")) and bool(stop.get("pass")) and bool(live.get("best_success_count", 0) >= 8)
    lines = [
        "# V3 Benchmark Summary",
        "",
        "## V2 Baseline Recap",
        "- OmniNav-only remains the best clean baseline: 6/15 success, 5/15 clean.",
        "- Step+OmniNav is not proven better in v2: 6/15 success, 3/15 clean, longer mean time.",
        "- Step-only remains 0/15 and is not an executor.",
        "- InternNav is CmaAgent/system1/fallback_static_cma_tokens, not full InternVLA-N1.",
        "",
        "## Route-choice V3",
        f"- route_choice_correct_branch_rate: {route.get('route_choice_correct_branch_rate', 'n/a')}",
        f"- pass: {route.get('pass', 'n/a')}",
        "",
        "## Semantic-stop V3",
        f"- semantic_stop_accuracy: {stop.get('semantic_stop_accuracy', 'n/a')}",
        f"- visible_to_stop_latency_sec: {stop.get('visible_to_stop_latency_sec', 'n/a')}",
        f"- pass: {stop.get('pass', 'n/a')}",
        "",
        "## Visual Isaac Demo",
        f"- output: {visual.get('overlay_state_path', args.visual_result or 'n/a')}",
        "",
        "## Step Value Conclusion",
        f"- Step value proven: {step_value_proven}",
        "- Step value is not yet proven." if not step_value_proven else "- Step value has passed route/stop gates and full v3 target.",
        "",
        "## Sim2Real Gate",
        gate_text.strip() or "- gate not run",
        "",
        "## Failure_top1",
        "- timeout remains the v2 failure_top1 until live v3 proves otherwise.",
    ]
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
