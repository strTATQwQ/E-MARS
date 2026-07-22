#!/usr/bin/env python3
"""Aggregate T4.1 camera runs and enforce single-use held-out semantics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


VARIANT_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "current",
            "height_085",
            "height_100",
            "h1_equivalent",
            "pitch_20",
            "pitch_40",
            "hfov_80",
            "hfov_100",
            "custom",
        )
    )
}
REQUIRED_DEVELOPMENT_VARIANTS = tuple(
    name for name in VARIANT_ORDER if name != "custom"
)


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def audit_summary(path: Path) -> dict[str, Any]:
    near: list[float] = []
    roll: list[float] = []
    pitch: list[float] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                near.append(float(item["near_field_fraction"]))
                roll.append(abs(float(item["base_roll_rad"])))
                pitch.append(abs(float(item["base_pitch_rad"])))
    return {
        "sample_count": len(near),
        "near_field_fraction_mean": statistics.fmean(near) if near else None,
        "near_field_fraction_p95": percentile(near, 0.95),
        "absolute_base_roll_rad_p95": percentile(roll, 0.95),
        "absolute_base_pitch_rad_p95": percentile(pitch, 0.95),
    }


def parse_name(name: str) -> tuple[str, str, str] | None:
    prefix = "t4_1_camera_"
    marker = "_attempt_"
    if not name.startswith(prefix) or marker not in name:
        return None
    body, attempt = name[len(prefix) :].rsplit(marker, 1)
    split, variant = body.split("_", 1)
    if split not in {"dev", "heldout"}:
        return None
    return split, variant, attempt


def summarize_run(directory: Path) -> dict[str, Any] | None:
    parsed = parse_name(directory.name)
    required = [
        directory / "phase_status.json",
        directory / "result.json",
        directory / "per_episode.json",
        directory / "go2_asset_manifest.json",
    ]
    if parsed is None or any(not path.is_file() for path in required):
        return None
    split, variant, attempt = parsed
    phase = load(directory / "phase_status.json")
    result = load(directory / "result.json").get("val_unseen", {})
    episodes = load(directory / "per_episode.json")
    camera = load(directory / "go2_asset_manifest.json")
    controller = (
        load(directory / "controller_summary.json")
        if (directory / "controller_summary.json").is_file()
        else {}
    )
    validation = load(directory / "validation.json") if (directory / "validation.json").is_file() else {}
    expected = int(episodes.get("expected_episode_count", 0))
    exit_codes = phase.get("exit_codes") or {}
    evaluation_complete = bool(
        expected == 5
        and int(episodes.get("completed_episode_count", 0)) == expected
        and int(result.get("Count", 0)) == expected
        and int(exit_codes.get("preflight", 125)) == 0
        and int(exit_codes.get("evaluator", 125)) == 0
        and controller.get("status") == "FINISHED"
        and all(
            int(controller.get(name, 0)) == 0
            for name in (
                "nan_count",
                "fall_count",
                "physical_collision_count",
                "stale_or_identity_reject_count",
            )
        )
    )
    return {
        "split": split,
        "variant": variant,
        "attempt": attempt,
        "source_directory": directory.as_posix(),
        "phase_status": phase.get("status"),
        "validation_status": validation.get("status"),
        "exit_codes": exit_codes,
        "evaluation_complete": evaluation_complete,
        "metrics": {
            name: result.get(name)
            for name in ("Count", "SR", "OS", "SPL", "NE", "TL", "FR", "StR")
        },
        "termination_label_success_rate": (
            float(episodes["success_rate"]) if episodes.get("expected_episode_count") else None
        ),
        "termination_reasons": [
            {
                "ordinal": item["ordinal"],
                "trajectory_id": item["trajectory_id"],
                "reason": item["termination_reason"],
                "success": item["success"],
                "step_count": item["step_count"],
            }
            for item in episodes["episodes"]
        ],
        "camera": {
            key: camera.get(key)
            for key in (
                "wrapper_sha256",
                "camera_height_above_support_m",
                "camera_translation_from_base_m",
                "pitch_down_deg",
                "hfov_deg",
                "vfov_deg",
                "camera_model",
                "focal_length_mm",
                "mount",
                "posture_compensation",
            )
        },
        "camera_audit": audit_summary(directory / "camera_audit.jsonl"),
    }


def ranking_key(run: dict[str, Any]) -> tuple[float, ...]:
    metrics = run["metrics"]
    audit = run["camera_audit"]
    return (
        float(metrics.get("SR") or 0.0),
        float(metrics.get("OS") or 0.0),
        float(metrics.get("SPL") or 0.0),
        -float(metrics.get("NE") if metrics.get("NE") is not None else math.inf),
        -float(
            audit.get("near_field_fraction_p95")
            if audit.get("near_field_fraction_p95") is not None
            else math.inf
        ),
        -float(VARIANT_ORDER.get(run["variant"], 999)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/internnav_t4")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/internnav_t4/t4_1_camera_ablation.json"),
    )
    parser.add_argument("--frozen-camera", type=Path)
    args = parser.parse_args()
    runs = []
    for directory in sorted(path for path in args.results_root.iterdir() if path.is_dir()):
        summary = summarize_run(directory)
        if summary is not None:
            runs.append(summary)
    dev = [run for run in runs if run["split"] == "dev"]
    heldout = [run for run in runs if run["split"] == "heldout"]
    if len(heldout) > 1:
        raise RuntimeError("held-out camera split was consumed more than once")
    eligible = [
        run
        for run in dev
        if run["phase_status"] == "PASS"
        and all(value == 0 for value in (run["exit_codes"] or {}).values())
    ]
    eligible_variants = {run["variant"] for run in eligible}
    missing_development_variants = sorted(
        set(REQUIRED_DEVELOPMENT_VARIANTS) - eligible_variants,
        key=lambda name: VARIANT_ORDER[name],
    )
    development_complete = not missing_development_variants
    recommended = max(eligible, key=ranking_key) if eligible else None
    frozen = load(args.frozen_camera) if args.frozen_camera and args.frozen_camera.is_file() else None
    heldout_matches_frozen = None
    if heldout and frozen:
        heldout_matches_frozen = heldout[0]["variant"] == frozen.get("variant")
        if not heldout_matches_frozen:
            raise RuntimeError("held-out run does not use the frozen development winner")
    heldout_complete = bool(
        len(heldout) == 1
        and frozen
        and heldout_matches_frozen is True
        and heldout[0]["evaluation_complete"] is True
    )
    if development_complete and heldout_complete:
        status = "PASS"
    elif development_complete:
        status = "DEV_COMPLETE"
    else:
        status = "INCOMPLETE"
    payload = {
        "schema_version": 1,
        "status": status,
        "selection_policy": (
            "lexicographic SR, OS, SPL, inverse NE, inverse p95 near-field proxy; "
            "variant order breaks exact ties"
        ),
        "development_run_count": len(dev),
        "development_complete": development_complete,
        "required_development_variants": list(REQUIRED_DEVELOPMENT_VARIANTS),
        "missing_development_variants": missing_development_variants,
        "heldout_run_count": len(heldout),
        "heldout_single_use": len(heldout) <= 1,
        "heldout_matches_frozen": heldout_matches_frozen,
        "heldout_evaluation_complete": (
            heldout[0]["evaluation_complete"] if heldout else None
        ),
        "heldout_minimum_sr_0_4_met": (
            float(heldout[0]["metrics"].get("SR") or 0.0) >= 0.4
            if heldout
            else None
        ),
        "recommended_development_variant": (
            recommended["variant"] if recommended else None
        ),
        "runs": sorted(
            runs,
            key=lambda run: (
                run["split"],
                VARIANT_ORDER.get(run["variant"], 999),
                run["attempt"],
            ),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
