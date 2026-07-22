#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Attribute semantic recognition, track, stop, and judge failures.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    details = []
    groups = defaultdict(lambda: {"total": 0, "visible_correct": 0, "confirmed": 0, "stop": 0, "success": 0})
    for row in rows:
        expected = bool(row.get("expected"))
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        predicted = bool(response.get("target_visible"))
        track = response.get("track") if isinstance(response.get("track"), dict) else {}
        confirmed = bool(track.get("confirmed"))
        scoped = list(row.get("primitives_after_response") or [])
        stop = any(item.get("primitive") == "stop" for item in scoped)
        success = bool(row.get("semantic_success")) if expected else not stop
        target = str(row.get("target") or "unknown")
        for key in ("all", f"expected:{expected}", f"target:{target}"):
            groups[key]["total"] += 1
            groups[key]["visible_correct"] += int(predicted == expected)
            groups[key]["confirmed"] += int(confirmed)
            groups[key]["stop"] += int(stop)
            groups[key]["success"] += int(success)
        trajectory = list(row.get("full_trajectory") or row.get("trajectory") or [])
        steps = list(row.get("approach_steps") or [])
        decreasing = sum(float(item.get("distance_after", 999)) < float(item.get("distance_before", -999)) for item in steps)
        details.append(
            {
                "episode_id": row.get("episode_id"),
                "target": target,
                "expected": expected,
                "predicted_visible": predicted,
                "track_confirmed": confirmed,
                "track_hits": track.get("hits"),
                "stop_primitive": stop,
                "semantic_success": success,
                "distance_m_used": row.get("distance_m_used"),
                "threshold_to_safe_stop_sec": row.get("threshold_to_safe_stop_sec"),
                "coverage_to_stop_sec": row.get("coverage_to_stop_sec"),
                "approach_steps": len(steps),
                "decreasing_steps": decreasing,
                "final_pose": json.dumps(trajectory[-1] if trajectory else None),
                "failure": (
                    "none" if success else
                    "not_visible" if expected and not predicted else
                    "track_not_confirmed" if expected and not confirmed else
                    "stop_missing" if expected and not stop else
                    "judge_failed" if expected else "false_stop"
                ),
            }
        )
    with (output / "semantic_case_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]) if details else ["empty"])
        writer.writeheader()
        writer.writerows(details)
    result = {key: dict(value) for key, value in sorted(groups.items())}
    (output / "semantic_attribution.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"groups": result, "failures": [row for row in details if row["failure"] != "none"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
