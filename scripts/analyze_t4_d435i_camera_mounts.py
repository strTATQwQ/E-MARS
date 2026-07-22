#!/usr/bin/env python3
"""Rank D435i mount trials without mixing them with the legacy generic-camera study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_t4_camera_ablation import ranking_key, summarize_run


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/internnav_t4/realsense_d435i_mount_ablation.json"),
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/internnav_t4")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/internnav_t4/t4_1_d435i_mount_ablation.json"),
    )
    parser.add_argument(
        "--frozen-camera",
        type=Path,
        default=Path("configs/internnav_t4/frozen_camera.json"),
    )
    args = parser.parse_args()
    config = load(args.config)
    candidates = {item["variant"]: item for item in config["candidates"]}
    runs: list[dict[str, Any]] = []
    for directory in sorted(path for path in args.results_root.iterdir() if path.is_dir()):
        summary = summarize_run(directory)
        if summary is None or summary["variant"] not in candidates:
            continue
        summary["geometric_prescreen"] = candidates[summary["variant"]]
        runs.append(summary)
    observed = [
        run
        for run in runs
        if run["split"] == "dev"
        and int((run["metrics"] or {}).get("Count") or 0) == 5
        and int((run["exit_codes"] or {}).get("preflight", 125)) == 0
        and int((run["exit_codes"] or {}).get("evaluator", 125)) == 0
    ]
    observed_by_variant: dict[str, list[dict[str, Any]]] = {}
    for run in observed:
        observed_by_variant.setdefault(run["variant"], []).append(run)
    duplicated_observed = sorted(
        name for name, values in observed_by_variant.items() if len(values) > 1
    )
    if duplicated_observed:
        raise RuntimeError(f"multiple observed D435i runs: {duplicated_observed}")
    eligible = [run for run in observed if run["evaluation_complete"]]
    eligible_by_variant: dict[str, list[dict[str, Any]]] = {}
    for run in eligible:
        eligible_by_variant.setdefault(run["variant"], []).append(run)
    duplicated = sorted(
        name for name, values in eligible_by_variant.items() if len(values) > 1
    )
    if duplicated:
        raise RuntimeError(f"multiple eligible D435i runs: {duplicated}")
    missing = sorted(set(candidates) - set(observed_by_variant))
    unsafe_or_invalid = sorted(set(observed_by_variant) - set(eligible_by_variant))
    feasible = [
        values[0]
        for name, values in eligible_by_variant.items()
        if candidates[name]["prescreen_feasible"]
    ]
    empirical = max(eligible, key=ranking_key) if eligible else None
    recommended = max(feasible, key=ranking_key) if feasible else None
    heldout = [run for run in runs if run["split"] == "heldout"]
    if len(heldout) > 1:
        raise RuntimeError("D435i held-out split was consumed more than once")
    frozen = load(args.frozen_camera) if args.frozen_camera.is_file() else None
    heldout_matches_frozen = bool(
        heldout
        and frozen
        and heldout[0]["variant"] == frozen.get("variant")
    )
    heldout_complete = bool(
        heldout and heldout[0]["evaluation_complete"] and heldout_matches_frozen
    )
    if missing:
        status = "INCOMPLETE"
    elif heldout_complete:
        status = "COMPLETE"
    else:
        status = "DEV_COMPLETE"
    payload = {
        "schema_version": 1,
        "camera_model": config["camera_model"],
        "status": status,
        "selection_policy": (
            "hardware/frustum prescreen, then lexicographic SR, OS, SPL, inverse "
            "NE, inverse p95 near-field proxy"
        ),
        "required_variants": list(candidates),
        "missing_variants": missing,
        "unsafe_or_invalid_variants": unsafe_or_invalid,
        "empirical_winner_without_hardware_constraints": (
            empirical["variant"] if empirical else None
        ),
        "recommended_feasible_variant": (
            recommended["variant"] if recommended else None
        ),
        "frozen_variant": frozen.get("variant") if frozen else None,
        "heldout_matches_frozen": heldout_matches_frozen if heldout else None,
        "heldout_evaluation_complete": heldout_complete if heldout else None,
        "heldout_result": heldout[0] if heldout else None,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )
    print(json.dumps({"status": payload["status"], "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
