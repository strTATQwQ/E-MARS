#!/usr/bin/env python3
"""Enrich a T3 static-map manifest with dataset-declared initial yaw."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def normalize(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.manifest.read_bytes()
    dataset_bytes = args.dataset.read_bytes()
    manifest = json.loads(source_bytes)
    with gzip.open(args.dataset, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    by_identity = {
        (str(item["trajectory_id"]), str(item["episode_id"])): item
        for item in episodes
    }
    if len(by_identity) != len(episodes):
        raise RuntimeError("dataset episode identities are not unique")
    for generation in manifest["generations"]:
        identity = (
            str(generation["trajectory_id"]),
            str(generation["episode_id"]),
        )
        episode = by_identity.get(identity)
        if episode is None:
            raise RuntimeError(f"static manifest episode absent from dataset: {identity}")
        position = [float(value) for value in episode["start_position"]]
        rotation = [float(value) for value in episode["start_rotation"]]
        if len(position) != 3 or len(rotation) != 4:
            raise RuntimeError(f"invalid dataset start pose: {identity}")
        expected_xy = (position[0], -position[2])
        if math.hypot(
            expected_xy[0] - float(generation["start_map_xy"][0]),
            expected_xy[1] - float(generation["start_map_xy"][1]),
        ) > 1e-5:
            raise RuntimeError(f"start position transform mismatch: {identity}")
        # Habitat quaternion is xyzw with navigation yaw about +Y and camera
        # forward along -Z. Isaac map is [x,-z], whose zero yaw is +X.
        habitat_yaw = 2.0 * math.atan2(rotation[1], rotation[3])
        generation["start_map_yaw_rad"] = normalize(math.pi / 2.0 - habitat_yaw)
        generation["start_pose_source"] = "dataset_start_position_and_rotation"
    manifest["t4_truth_isolation"] = {
        "schema_version": 1,
        "status": "PASS",
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "runtime_ground_truth_pose_used_for_map_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "episode_count": len(manifest["generations"]),
                "output": args.output.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
