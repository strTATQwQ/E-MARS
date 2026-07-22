#!/usr/bin/env python3
"""Finalize the fail-closed T1.2 Nav2 shadow gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-commands", type=int, default=120)
    arguments = parser.parse_args()
    summary = json.loads(arguments.summary.read_text(encoding="utf-8"))
    passing = (
        int(summary["command_count"]) == arguments.expected_commands
        and int(summary["trajectory_count"]) > 0
        and float(summary["tf_success_rate"]) == 1.0
        and int(summary["stale_execution_count"]) == 0
        and int(summary["cross_episode_pollution_count"]) == 0
        and float(summary["local_goal_valid_rate"]) >= 0.95
        and float(summary["direction_match_rate"]) >= 0.95
        and int(summary["map_missing_count"]) == 0
    )
    summary["status"] = "PASS" if passing else "FAIL"
    summary["expected_commands"] = arguments.expected_commands
    atomic_json(arguments.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if passing else 1)


if __name__ == "__main__":
    main()
