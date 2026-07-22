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

from isaac_vln_benchmark.v6_recovery_utils import evaluate_sim2real_readiness_v6


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_metrics(path: Path) -> dict[str, Any]:
    if path.is_dir():
        for name in ("sim2real_gate_input.json", "metrics.json", "out/metrics.json"):
            candidate = path / name
            if candidate.exists():
                return json.loads(candidate.read_text(encoding="utf-8"))
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render(result: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [
        f"# Sim2Real Readiness V6: {result['status']}",
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
    parser.add_argument("--config", default=str(ROOT / "configs" / "sim2real_gate_v6.yaml"))
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    metrics = load_metrics(Path(args.input))
    gate = load_yaml(Path(args.config)).get("required", {})
    result = evaluate_sim2real_readiness_v6(metrics, gate)
    text = render(result, metrics)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
