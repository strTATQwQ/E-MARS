#!/usr/bin/env python3
"""Recover T3.0 postprocessing without rewriting a failed phase status."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from summarize_internnav_progress import summarize


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("progress_log_root", type=Path)
    parser.add_argument("--expected", type=int, default=20)
    parser.add_argument(
        "--reason",
        default="post-evaluation dependency absent from synchronized control tree",
    )
    args = parser.parse_args()

    root = args.result_dir.resolve()
    phase_path = root / "phase_status.json"
    phase = load(phase_path)
    if not isinstance(phase, dict):
        raise RuntimeError("phase status must be an object")
    exits = phase.get("exit_codes", {})
    if phase.get("status") != "FAIL" or exits.get("evaluator") != 0:
        raise RuntimeError("recovery is only valid for a failed phase with evaluator exit 0")

    outputs = (
        root / "per_episode_recovered.json",
        root / "validation_recovered.json",
        root / "postprocess_recovery.json",
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("recovery outputs already exist")

    candidates: list[tuple[int, Path, dict[str, object]]] = []
    for path in args.progress_log_root.resolve().rglob("*.log"):
        payload = summarize(path)
        if (
            payload["expected_episode_count"] == args.expected
            and payload["completed_episode_count"] == args.expected
        ):
            candidates.append((path.stat().st_mtime_ns, path, payload))
    if not candidates:
        raise RuntimeError("no complete progress log found")
    _, progress_path, per_episode = max(candidates, key=lambda item: item[0])

    result = load(root / "result.json")
    active = load(root / "active_summary.json")
    if not isinstance(result, dict) or not isinstance(active, dict):
        raise RuntimeError("result and active summary must be objects")
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
        "recovered_postprocess": True,
    }
    source_paths = (
        phase_path,
        root / "result.json",
        root / "active_summary.json",
        root / "active_records.jsonl",
        root / "go2_runtime_audit.jsonl",
        progress_path,
    )
    source_hashes = {
        (path.relative_to(root).as_posix() if path.is_relative_to(root) else progress_path.name): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in source_paths
    }
    recovery = {
        "schema_version": 1,
        "status": validation["status"],
        "original_phase_status_retained": True,
        "reason": args.reason,
        "progress_log": progress_path.relative_to(args.progress_log_root.resolve()).as_posix(),
        "source_hashes": source_hashes,
    }
    outputs[0].write_text(json.dumps(per_episode, indent=2, sort_keys=True) + "\n")
    outputs[1].write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    outputs[2].write_text(json.dumps(recovery, indent=2, sort_keys=True) + "\n")
    print(json.dumps(recovery, indent=2, sort_keys=True))
    raise SystemExit(0 if passing else 1)


if __name__ == "__main__":
    main()
