#!/usr/bin/env python3
"""Freeze ten short, single-floor, disjoint official episodes for Nav2 Oracle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from freeze_internnav_t0_episodes import load_official_dataset_utils


SOURCE_REVISION = "7b05993b21813c3787f2f7f604bfc22b80c48c8e"
SOURCE_SHA256 = "853673f8faadb26f883d57d71828cf895345c89fa594a7b918a78862318aeb31"


def key(item: dict) -> str:
    return f"{item['trajectory_id']}_{item['episode_id']}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internnav-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    source_file = arguments.source_root / "val_unseen" / "val_unseen.json.gz"
    if source_file.stat().st_size != 448607 or sha256(source_file) != SOURCE_SHA256:
        raise SystemExit("official val_unseen payload does not match pinned bytes")
    with gzip.open(source_file, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)["episodes"]
    originals = {key(item): item for item in raw}
    excluded: set[str] = set()
    for manifest_path in arguments.exclude_manifest:
        excluded.update(json.loads(manifest_path.read_text(encoding="utf-8"))["episode_keys"])
    load_data, skip_list = load_official_dataset_utils(arguments.internnav_root)
    filtered = load_data(
        str(arguments.source_root),
        "val_unseen",
        filter_same_trajectory=True,
        filter_stairs=True,
        dataset_type="mp3d",
        rank=0,
        world_size=1,
    )
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for scan, items in filtered.items():
        for filtered_item in items:
            item = originals[key(filtered_item)]
            heights = [float(point[1]) for point in item["reference_path"]]
            distance = float(item["info"]["geodesic_distance"])
            if (
                key(item) not in excluded
                and int(item["trajectory_id"]) not in skip_list
                and 3.2 <= distance <= 7.0
                and max(heights) - min(heights) <= 0.25
            ):
                by_scene[scan].append(item)
    eligible_scenes = [
        (scene, sorted(items, key=lambda item: (item["info"]["geodesic_distance"], key(item))))
        for scene, items in by_scene.items()
        if len(items) >= 10
    ]
    if not eligible_scenes:
        raise SystemExit("no single official scene has ten eligible short episodes")
    scene, candidates = sorted(eligible_scenes, key=lambda pair: (-len(pair[1]), pair[0]))[0]
    selected = candidates[:10]
    output_file = arguments.output_root / "val_unseen" / "val_unseen.json.gz"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"episodes": selected}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    output_file.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    manifest = {
        "schema_version": 1,
        "phase": "h1_nav2_oracle_10_short",
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "selection": "one fixed scene; official filter_stairs; single-floor; geodesic 3.2-7.0m; shortest ten; disjoint T0 canary/pilot/gate1",
        "episode_count": 10,
        "scene_id": scene,
        "episode_keys": [key(item) for item in selected],
        "geodesic_distances_m": [float(item["info"]["geodesic_distance"]) for item in selected],
        "overlay_sha256": sha256(output_file),
    }
    (arguments.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
