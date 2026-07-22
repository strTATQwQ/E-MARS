#!/usr/bin/env python3
"""Aggregate the same-episode T4.6 contribution ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED = {
    "full_system1_system2",
    "oracle_high_level_system1",
    "system2_oracle_local_path",
    "full_trajectory",
    "endpoint",
    "straight_line",
    "model_stop",
    "oracle_termination",
    "history_on",
    "history_off",
    "recovery_on",
    "recovery_off",
    "h1_view",
    "go2_view",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def variant_from_name(name: str) -> str | None:
    prefix = "t4_6_"
    marker = "_model_attempt_"
    if not name.startswith(prefix) or marker not in name:
        return None
    return name[len(prefix) : name.index(marker)]


def dataset_sha(input_manifest: dict[str, Any]) -> str:
    matches = [
        str(item["sha256"])
        for item in input_manifest.get("files", [])
        if item.get("label") == "dataset"
    ]
    if len(matches) != 1:
        raise RuntimeError("input manifest lacks exactly one dataset hash")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/internnav_t4"))
    parser.add_argument("--output", type=Path, default=Path("results/internnav_t4/t4_6_ablation.json"))
    args = parser.parse_args()
    runs: dict[str, dict[str, Any]] = {}
    for directory in sorted(args.results_root.glob("t4_6_*_model_attempt_*")):
        variant = variant_from_name(directory.name)
        if variant is None or not (directory / "result.json").is_file():
            continue
        if variant in runs:
            raise RuntimeError(f"multiple authoritative runs for {variant}")
        result = load(directory / "result.json")
        metrics = result.get("val_unseen", result)
        per_episode = load(directory / "per_episode.json")
        contract = load(directory / "t4_run_contract.json")
        active = load(directory / "active_summary.json")
        active_records = jsonl(directory / "active_records.jsonl")
        action_sources = [
            int(item["action_source"])
            for item in active_records
            if "action_source" in item and item.get("event") != "ablation_transform"
        ]
        lifecycle = (
            load(directory / "model_lifecycle_manifest.json")
            if (directory / "model_lifecycle_manifest.json").is_file()
            else None
        )
        termination_reasons = [
            str(item.get("termination_reason", ""))
            for item in per_episode.get("episodes", [])
        ]
        runs[variant] = {
            "source_directory": directory.as_posix(),
            "phase_status": load(directory / "phase_status.json").get("status"),
            "validation_status": load(directory / "validation.json").get("status"),
            "dataset_sha256": dataset_sha(load(directory / "input_manifest.json")),
            "episode_ids": [
                str(item.get("trajectory_id", ""))
                for item in per_episode.get("episodes", [])
            ],
            "metrics": metrics,
            "termination_reasons": termination_reasons,
            "stuck_rate": (
                termination_reasons.count("stuck") / len(termination_reasons)
                if termination_reasons
                else None
            ),
            "action_source_counts": {
                "system2": action_sources.count(1),
                "system1_new": action_sources.count(2),
                "system1_queue": action_sources.count(3),
                "unknown": action_sources.count(0),
            },
            "contract": contract,
            "model_lifecycle": lifecycle,
            "recovery": (
                load(directory / "recovery_metrics.json")
                if (directory / "recovery_metrics.json").is_file()
                else None
            ),
        }
    missing = sorted(REQUIRED - set(runs))
    datasets = {item["dataset_sha256"] for item in runs.values()}
    episode_sets = {tuple(item["episode_ids"]) for item in runs.values()}
    lifecycle_ok = all(
        item["model_lifecycle"] is not None
        and item["model_lifecycle"].get("history_mode")
        == item["contract"].get("required_model_history_mode")
        for item in runs.values()
    )
    complete_ok = all(
        item["phase_status"] == "PASS" and item["validation_status"] == "PASS"
        for item in runs.values()
    )
    status = (
        "PASS"
        if not missing
        and len(datasets) == 1
        and len(episode_sets) == 1
        and lifecycle_ok
        and complete_ok
        else "FAIL"
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "required_variants": sorted(REQUIRED),
        "missing_variants": missing,
        "same_dataset_hash": len(datasets) == 1,
        "same_episode_order": len(episode_sets) == 1,
        "model_history_lifecycle_matches_contract": lifecycle_ok,
        "all_runs_complete": complete_ok,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "runs": len(runs), "missing": missing}, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
