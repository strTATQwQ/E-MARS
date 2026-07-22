#!/usr/bin/env python3
"""Find the complete progress log for one run and write safe episode evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from summarize_internnav_progress import (
        attach_official_metrics,
        metric_records,
        summarize,
    )
except ModuleNotFoundError:  # Package import used by offline tests.
    from scripts.summarize_internnav_progress import (
        attach_official_metrics,
        metric_records,
        summarize,
    )


def extract_complete(log_root: Path, expected: int) -> dict[str, object]:
    logs = list(log_root.rglob("*.log"))
    candidates = []
    for path in logs:
        payload = summarize(path)
        if (
            payload["expected_episode_count"] == expected
            and payload["completed_episode_count"] == expected
        ):
            candidates.append((path.stat().st_mtime_ns, payload))
    if not candidates:
        raise RuntimeError("no complete progress log found")
    candidates.sort(key=lambda item: item[0])
    payload = candidates[-1][1]

    exact_metrics: dict[str, dict[str, object]] = {}
    for path in logs:
        for key, record in metric_records(path).items():
            previous = exact_metrics.get(key)
            if previous is not None and previous != record:
                raise RuntimeError(f"conflicting evaluator metric logs for {key}")
            exact_metrics[key] = record
    for episode in payload["episodes"]:
        key = str(episode["trajectory_id"])
        metric = exact_metrics.get(key)
        if metric is not None:
            attach_official_metrics(episode, metric)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected", required=True, type=int)
    args = parser.parse_args()
    payload = extract_complete(args.log_root, args.expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
