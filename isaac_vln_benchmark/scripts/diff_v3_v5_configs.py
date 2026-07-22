#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.v6_recovery_utils import config_diff_rows, write_csv


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def default_v3_path() -> Path:
    return ROOT.parent / "test_results" / "20260707_real_step_omninav_ros_client" / "dgx" / "scheduler_config.yaml"


def default_current_scheduler_path() -> Path:
    return ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler" / "config" / "scheduler_isaac_real_models.yaml"


def render_markdown(rows: list[dict[str, Any]], paths: dict[str, str]) -> str:
    lines = [
        "# V6 Config Diff",
        "",
        "## Inputs",
        "",
    ]
    lines.extend([f"- {name}: `{path}`" for name, path in paths.items()])
    high = [row for row in rows if str(row.get("risk", "")).startswith("high")]
    medium = [row for row in rows if str(row.get("risk", "")).startswith("medium")]
    lines.extend(["", "## High Risk Changes", ""])
    lines.extend([f"- {row['field']}: {row.get('risk')}" for row in high] or ["- none"])
    lines.extend(["", "## Medium Risk Changes", ""])
    lines.extend([f"- {row['field']}: {row.get('risk')}" for row in medium] or ["- none"])
    lines.extend(["", "## Table", "", "| field | risk | v3 | v5 | v6 |", "| --- | --- | --- | --- | --- |"])
    for row in rows:
        lines.append(
            f"| {row['field']} | {row.get('risk', '')} | "
            f"{_cell(row.get('v3'))} | {_cell(row.get('v5'))} | {_cell(row.get('v6'))} |"
        )
    return "\n".join(lines) + "\n"


def _cell(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    return text.replace("|", "\\|")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", default=str(default_v3_path()))
    parser.add_argument("--v5", default=str(default_current_scheduler_path()))
    parser.add_argument("--v6", default=str(default_current_scheduler_path()))
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    output = Path(args.output) if args.output else ROOT / "runs" / f"config_diff_v6_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    paths = {"v3": str(Path(args.v3)), "v5": str(Path(args.v5)), "v6": str(Path(args.v6))}
    configs = {name: load_yaml(Path(path)) for name, path in paths.items()}
    rows = config_diff_rows(configs)
    (output / "config_diff.json").write_text(json.dumps({"inputs": paths, "rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output / "config_diff.csv", rows)
    (output / "config_diff.md").write_text(render_markdown(rows, paths), encoding="utf-8")
    print(json.dumps({"output": str(output), "high_risk": [row["field"] for row in rows if str(row.get("risk", "")).startswith("high")]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
