#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.v5_timebase_utils import evaluate_sim2real_readiness_v5


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
            return parse_summary(summary)
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return parse_summary(path)


def parse_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    metrics: dict[str, Any] = {}
    patterns = {
        "timebase_error_count": r"timebase_error_count\s*[:=]\s*(\d+)",
        "episode_mismatch_count": r"episode_mismatch_count\s*[:=]\s*(\d+)",
        "missing_timestamp_count": r"missing_timestamp_count\s*[:=]\s*(\d+)",
        "stale_discard_count": r"stale_discard_count\s*[:=]\s*(\d+)",
        "visible_to_stop_latency_sec": r"visible_to_stop_latency_sec\s*[:=]\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            metrics[key] = float(value) if "." in value else int(value)
    return metrics


def render(result: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [
        f"# Sim2Real Readiness V5: {result['status']}",
        "",
        "## Metrics",
        json.dumps(metrics, indent=2, ensure_ascii=False),
        "",
        "## Failures",
    ]
    lines.extend([f"- {item}" for item in result["failures"]] or ["- none"])
    lines.extend(["", "## Allowed Next Steps"])
    lines.extend([f"- {item}" for item in result["allowed_next_steps"]])
    lines.extend(["", "## Disallowed Next Steps"])
    lines.extend([f"- {item}" for item in result["disallowed_next_steps"]])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "sim2real_gate_v5.yaml"))
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    gate = load_yaml(Path(args.config)).get("required", {})
    metrics = load_metrics(Path(args.input))
    result = evaluate_sim2real_readiness_v5(metrics, gate)
    text = render(result, metrics)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
