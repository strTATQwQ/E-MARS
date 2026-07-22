#!/usr/bin/env python3
"""Build the immutable official candidate pool for T3 Go2 clearance audit."""

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


def episode_key(item: dict) -> str:
    return f"{item['trajectory_id']}_{item['episode_id']}"


def start_key(item: dict) -> tuple[float, float, float]:
    return tuple(round(float(value), 6) for value in item["reference_path"][0])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internnav-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--minimum-geodesic-m", type=float, default=3.2)
    parser.add_argument("--maximum-geodesic-m", type=float, default=9.0)
    parser.add_argument("--episode-key", action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    if not 0.0 < arguments.minimum_geodesic_m < arguments.maximum_geodesic_m:
        raise SystemExit("invalid geodesic candidate range")
    source_file = arguments.source_root / "val_unseen" / "val_unseen.json.gz"
    if source_file.stat().st_size != 448607 or sha256(source_file) != SOURCE_SHA256:
        raise SystemExit("official val_unseen payload does not match pinned bytes")
    with gzip.open(source_file, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)["episodes"]
    originals = {episode_key(item): item for item in raw}
    excluded: set[str] = set()
    for manifest_path in arguments.exclude_manifest:
        excluded.update(json.loads(manifest_path.read_text(encoding="utf-8"))["episode_keys"])

    load_data, skip_list = load_official_dataset_utils(arguments.internnav_root)
    filtered = load_data(
        str(arguments.source_root), "val_unseen", filter_same_trajectory=True,
        filter_stairs=True, dataset_type="mp3d", rank=0, world_size=1,
    )
    by_scene: dict[str, list[dict]] = defaultdict(list)
    eligible_by_key: dict[str, dict] = {}
    for scene, items in filtered.items():
        used_starts: set[tuple[float, float, float]] = set()
        ranked = sorted(
            (originals[episode_key(item)] for item in items),
            key=lambda item: (float(item["info"]["geodesic_distance"]), episode_key(item)),
        )
        for item in ranked:
            heights = [float(point[1]) for point in item["reference_path"]]
            distance = float(item["info"]["geodesic_distance"])
            authored_start = start_key(item)
            if (
                episode_key(item) in excluded
                or int(item["trajectory_id"]) in skip_list
                or not arguments.minimum_geodesic_m
                <= distance
                <= arguments.maximum_geodesic_m
                or max(heights) - min(heights) > 0.25
            ):
                continue
            eligible_by_key[episode_key(item)] = item
            if arguments.episode_key:
                continue
            if authored_start in used_starts:
                continue
            used_starts.add(authored_start)
            by_scene[scene].append(item)

    if arguments.episode_key:
        if len(arguments.episode_key) < 1 or len(set(arguments.episode_key)) != len(
            arguments.episode_key
        ):
            raise SystemExit("explicit candidate keys must be unique")
        missing = [key for key in arguments.episode_key if key not in eligible_by_key]
        if missing:
            raise SystemExit(f"explicit episode keys fail official filters: {missing}")
        candidates = [eligible_by_key[key] for key in arguments.episode_key]
        selection_scope = "explicit official keys; duplicate authored starts retained"
    else:
        candidates = [item for scene in sorted(by_scene) for item in by_scene[scene]]
        selection_scope = "unique authored starts per scene"
    if len(candidates) < 10:
        raise SystemExit("fewer than ten official unique-start clearance candidates")
    output_file = arguments.output_root / "val_unseen" / "val_unseen.json.gz"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        {"episodes": candidates}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    output_file.write_bytes(gzip.compress(encoded, compresslevel=9, mtime=0))
    manifest = {
        "schema_version": 1,
        "phase": "t3_go2_clearance_candidate_pool",
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "selection": (
            "all scenes; official filter_stairs; single-floor; geodesic "
            f"{arguments.minimum_geodesic_m:.1f}-{arguments.maximum_geodesic_m:.1f}m; "
            f"{selection_scope}; disjoint T0 canary/pilot/gate1"
        ),
        "integrity": "official episode objects copied value-for-value; no field mutation",
        "episode_count": len(candidates),
        "minimum_geodesic_m": arguments.minimum_geodesic_m,
        "maximum_geodesic_m": arguments.maximum_geodesic_m,
        "scene_counts": {scene: len(items) for scene, items in sorted(by_scene.items())},
        "episode_keys": [episode_key(item) for item in candidates],
        "duplicate_authored_start_count": len(candidates)
        - len({start_key(item) for item in candidates}),
        "overlay_sha256": sha256(output_file),
    }
    (arguments.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
