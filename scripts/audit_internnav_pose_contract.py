#!/usr/bin/env python3
"""Print pose-only summaries from the frozen replay without copying RGB-D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    arguments = parser.parse_args()
    records = [
        json.loads(line)
        for line in (arguments.replay_root / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: dict[int, list[dict]] = {}
    for record in records:
        generation = int(record["reset_generation"])
        selected.setdefault(generation, [])
        if not selected[generation] or int(record["sequence_id"]) == 0:
            selected[generation].append(record)
        elif int(record["sequence_id"]) > int(selected[generation][-1]["sequence_id"]):
            if len(selected[generation]) == 1:
                selected[generation].append(record)
            else:
                selected[generation][-1] = record
    output = []
    for generation, endpoints in sorted(selected.items()):
        for record in endpoints:
            with np.load(arguments.replay_root / record["file"], allow_pickle=False) as payload:
                output.append(
                    {
                        "reset_generation": generation,
                        "sequence_id": int(record["sequence_id"]),
                        "global_gps": np.asarray(payload["globalgps"]).astype(float).tolist(),
                        "global_rotation": np.asarray(payload["globalrotation"]).astype(float).tolist(),
                    }
                )
    print(json.dumps({"schema_version": 1, "pose_endpoints": output}, indent=2))


if __name__ == "__main__":
    main()
