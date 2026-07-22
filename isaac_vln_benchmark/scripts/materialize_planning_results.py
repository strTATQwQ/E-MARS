#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.perception_planning_suite import (
    evaluate_planning,
    planning_gate,
    planning_rows_from_micro,
)


def load(path: str) -> list[dict]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise SystemExit(f"expected JSON array: {path}")
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    headers = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize planning gate rows from genuine live micro artifacts.")
    parser.add_argument("--route-cases", required=True)
    parser.add_argument("--semantic-cases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    route = planning_rows_from_micro(load(args.route_cases))["route"]
    semantic = planning_rows_from_micro(load(args.semantic_cases))["semantic"]
    metrics = evaluate_planning(route, semantic)
    gate = planning_gate(metrics)
    (output / "route_planning_results.json").write_text(json.dumps(route, indent=2) + "\n", encoding="utf-8")
    (output / "semantic_planning_results.json").write_text(json.dumps(semantic, indent=2) + "\n", encoding="utf-8")
    (output / "planning_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "planning_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    failures = []
    for row in route:
        if not row["entered_correct_branch"]:
            failures.append({"episode_id": row["episode_id"], "reason": "wrong_or_missing_branch"})
    for row in semantic:
        if row["expected"] and not row["success"]:
            failures.append({"episode_id": row["episode_id"], "reason": "semantic_positive_failed"})
        if not row["expected"] and row["stop"]:
            failures.append({"episode_id": row["episode_id"], "reason": "semantic_false_stop"})
    (output / "failure_table.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "failure_table.csv", failures)
    trajectory_rows = []
    controller_rows = []
    for task_type, rows in (("route", route), ("semantic", semantic)):
        for row in rows:
            episode_id = row.get("episode_id")
            for index, point in enumerate(row.get("trajectory") or []):
                value = dict(point) if isinstance(point, dict) else {"value": point}
                trajectory_rows.append(
                    {"episode_id": episode_id, "task_type": task_type, "sample_index": index, **value}
                )
            for index, point in enumerate(row.get("controller_trace") or []):
                value = dict(point) if isinstance(point, dict) else {"value": point}
                controller_rows.append(
                    {"episode_id": episode_id, "task_type": task_type, "sample_index": index, **value}
                )
    write_csv(output / "trajectory.csv", trajectory_rows)
    write_csv(output / "controller_trace.csv", controller_rows)
    print(json.dumps(gate, indent=2))
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
