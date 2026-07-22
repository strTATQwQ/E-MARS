#!/usr/bin/env python3
"""Build the three permitted T4.2-R2 inflation-only candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "internnav_t4"
BASE = CONFIG_DIR / "nav2_nvblox_gt_pose.yaml"
RADII = {"0p35": "0.35", "0p32": "0.32", "0p30": "0.30"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    base = BASE.read_text(encoding="utf-8")
    if base.count("inflation_radius: 0.40") != 2:
        raise RuntimeError("expected exactly two frozen 0.40 m inflation radii")
    for frozen in (
        "robot_radius: 0.30",
        "footprint_padding: 0.02",
        "allow_unknown: true",
        "convert_to_binary_costmap: true",
    ):
        if frozen not in base:
            raise RuntimeError(f"frozen setting missing: {frozen}")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "base": {
            "path": str(BASE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(BASE.read_bytes()),
            "inflation_radius_m": 0.40,
        },
        "frozen": {
            "robot_radius_m": 0.30,
            "footprint_padding_m": 0.02,
            "allow_unknown": True,
            "convert_to_binary_costmap": True,
            "geometry": "unchanged",
            "collision_monitor": "unchanged",
            "reset_and_stale_safety": "unchanged",
        },
        "candidates": [],
    }
    for tag, radius in RADII.items():
        candidate = base.replace(
            "inflation_radius: 0.40", f"inflation_radius: {radius}"
        )
        path = CONFIG_DIR / f"t4_2_r2_inflation_{tag}.yaml"
        path.write_text(candidate, encoding="utf-8")
        manifest["candidates"].append(
            {
                "tag": tag,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path.read_bytes()),
                "single_parameter_diff": {
                    "local_and_global.inflation_radius_m": {
                        "from": 0.40,
                        "to": float(radius),
                    }
                },
            }
        )
    blocked_candidate = (
        "# STATUS: BLOCKED diagnostic candidate; not Gate-3 qualified.\n"
        "# No repaired inflation-only configuration passed T4.2-R2.\n"
        "# Do not use this file for Oracle regression or downstream T4 gates.\n"
        + base.replace("inflation_radius: 0.40", "inflation_radius: 0.30")
    )
    repaired_path = CONFIG_DIR / "t4_2_repaired.yaml"
    repaired_path.write_text(blocked_candidate, encoding="utf-8")
    manifest["delivery_config"] = {
        "path": str(repaired_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(repaired_path.read_bytes()),
        "status": "BLOCKED_NOT_GATE_QUALIFIED",
        "derived_from_candidate": "0p30",
        "reason": "all permitted inflation radii left the robot footprint unknown",
    }
    manifest_path = CONFIG_DIR / "t4_2_r2_candidates_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
