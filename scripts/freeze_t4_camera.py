#!/usr/bin/env python3
"""Freeze the development-selected T4.1 camera before held-out use."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("results/internnav_t4/t4_1_camera_ablation.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/internnav_t4/frozen_camera.json"),
    )
    args = parser.parse_args()
    raw = args.analysis.read_bytes()
    analysis = json.loads(raw)
    if analysis.get("status") != "DEV_COMPLETE":
        raise RuntimeError("all development variants must pass before freezing")
    if analysis.get("heldout_run_count") != 0:
        raise RuntimeError("camera must be frozen before the held-out split is consumed")
    variant = analysis.get("recommended_development_variant")
    matches = [
        run
        for run in analysis.get("runs", [])
        if run.get("split") == "dev"
        and run.get("variant") == variant
        and run.get("phase_status") == "PASS"
        and all(value == 0 for value in (run.get("exit_codes") or {}).values())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one authoritative development winner, got {len(matches)}")
    run = matches[0]
    selected = run["camera"]
    camera = {
        "height_above_support_m": selected["camera_height_above_support_m"],
        "translation_from_base_m": selected["camera_translation_from_base_m"],
        "pitch_down_deg": selected["pitch_down_deg"],
        "hfov_deg": selected["hfov_deg"],
        "focal_length_mm": selected["focal_length_mm"],
        "mount": selected["mount"],
        "posture_compensation": selected["posture_compensation"],
        "wrapper_sha256": selected["wrapper_sha256"],
    }
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_HELDOUT",
        "variant": variant,
        "development_attempt": run["attempt"],
        "selection_policy": analysis["selection_policy"],
        "development_metrics": run["metrics"],
        "camera": camera,
        "analysis_sha256": hashlib.sha256(raw).hexdigest(),
    }
    encoded = canonical_bytes(payload)
    if args.output.exists() and args.output.read_bytes() != encoded:
        raise RuntimeError(f"refusing to overwrite frozen camera: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(json.dumps({"status": "PASS", "variant": variant}, sort_keys=True))


if __name__ == "__main__":
    main()
