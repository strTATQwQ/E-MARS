#!/usr/bin/env python3
"""Create the frozen-input manifest and formal T3.0 flash validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


FROZEN_FILES = (
    "result.json",
    "validation.json",
    "per_episode.json",
    "active_records.jsonl",
    "active_summary.json",
)


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_root", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected", type=int, default=20)
    args = parser.parse_args()
    control = args.control_root.resolve()
    root = args.result_dir.resolve()
    manifest_path = root / "t2_frozen_input_manifest.json"
    validation_path = root / "validation.json"
    if manifest_path.exists() or validation_path.exists():
        raise FileExistsError("formal postprocess outputs already exist")

    frozen = control / "results" / "t2_go2_nav2_pilot"
    entries = []
    for name in FROZEN_FILES:
        path = frozen / name
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            {
                "path": f"results/t2_go2_nav2_pilot/{name}",
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "mode": "go2_flash_nav2_sampled_baseline",
        "source_mutated": False,
        "files": entries,
    }

    result = load(root / "result.json")
    active = load(root / "active_summary.json")
    per_episode = load(root / "per_episode.json")
    if not all(isinstance(item, dict) for item in (result, active, per_episode)):
        raise RuntimeError("formal result summaries must be objects")
    metrics = result.get("val_unseen", result)
    records = [
        json.loads(line)
        for line in (root / "active_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    falls = sum(
        item["termination_reason"] == "fall" for item in per_episode["episodes"]
    )
    trajectory_records = [item for item in records if item.get("trajectory_valid")]
    path_hashes = [item.get("local_path_sha256") for item in trajectory_records]
    actions = Counter(
        int(item["selected_action"]) for item in records if "selected_action" in item
    )
    passing = (
        int(metrics.get("Count", metrics.get("length", 0))) == args.expected
        and per_episode["completed_episode_count"] == args.expected
        and falls == 0
        and active.get("status") == "FINISHED"
        and active.get("failure_count") == 0
        and active.get("execution_mode") == "sampled_flash"
        and bool(trajectory_records)
        and all(path_hashes)
        and len(actions) >= 2
    )
    validation = {
        "schema_version": 1,
        "status": "PASS" if passing else "FAIL",
        "mode": "go2_flash_nav2_sampled_baseline",
        "metrics": metrics,
        "warmup_and_episode_fall_count": falls,
        "trajectory_record_count": len(trajectory_records),
        "trajectory_hash_count": len(set(path_hashes)),
        "selected_action_distribution": dict(sorted(actions.items())),
        "active": active,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, indent=2, sort_keys=True))
    raise SystemExit(0 if passing else 1)


if __name__ == "__main__":
    main()
