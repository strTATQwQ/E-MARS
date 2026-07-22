#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.perception_planning_suite import planning_rows_from_micro


def rate(value: int, total: int) -> float:
    return round(value / total, 6) if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Attribute route recognition and branch-entry failures by family and side.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    converted = planning_rows_from_micro(raw)["route"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(lambda: {"episodes": 0, "recognition_correct": 0, "branch_correct": 0})
    failures = []
    confusion = defaultdict(int)
    for source, row in zip(raw, converted):
        family = str(source.get("scene_family") or "unknown")
        side = str(row.get("expected") or "unknown")
        predicted = str(row.get("predicted") or "none")
        recognition = predicted == side
        branch = bool(row.get("entered_correct_branch"))
        for key in ("all", f"family:{family}", f"side:{side}", f"family_side:{family}:{side}"):
            groups[key]["episodes"] += 1
            groups[key]["recognition_correct"] += int(recognition)
            groups[key]["branch_correct"] += int(branch)
        confusion[(side, predicted)] += 1
        if not recognition or not branch:
            traces = list(source.get("controller_trace") or [])
            failures.append(
                {
                    "episode_id": row.get("episode_id"),
                    "family": family,
                    "expected": side,
                    "predicted": predicted,
                    "recognition_correct": recognition,
                    "entered_correct_branch": branch,
                    "last_phase": str(traces[-1].get("phase") or "") if traces else "no_trace",
                    "trace_count": len(traces),
                    "final_pose": json.dumps((source.get("full_trajectory") or source.get("trajectory") or [None])[-1]),
                    "failure": "recognition_and_branch" if not recognition and not branch else "recognition" if not recognition else "branch_entry",
                }
            )
    summary = {}
    for key, values in sorted(groups.items()):
        summary[key] = dict(values) | {
            "recognition_accuracy": rate(values["recognition_correct"], values["episodes"]),
            "branch_accuracy": rate(values["branch_correct"], values["episodes"]),
        }
    (output / "route_attribution.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "route_failure_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(failures[0]) if failures else ["empty"])
        writer.writeheader()
        writer.writerows(failures)
    confusion_rows = [
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in sorted(confusion.items())
    ]
    with (output / "route_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["expected", "predicted", "count"])
        writer.writeheader()
        writer.writerows(confusion_rows)
    print(json.dumps({"groups": summary, "failures": failures}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
