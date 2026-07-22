#!/usr/bin/env python3
"""Freeze ten official short, geometry-clear episodes for physical Go2.

InternNav intentionally offsets duplicate spawn positions during dataset loading.
The default selection therefore requires unique authored starts. An explicit
ten-key freeze may retain a duplicate authored start when every selected key is
individually present in the pinned Matterport geometry audit; the manifest makes
that exception visible for reset-order validation. No trajectory point,
instruction, goal, or metric field is modified.
"""

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
    point = item["reference_path"][0]
    return tuple(round(float(value), 6) for value in point)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internnav-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--clearance-audit", type=Path, required=True)
    parser.add_argument("--minimum-prefix-clearance-m", type=float, default=0.45)
    parser.add_argument("--minimum-prefix-length-m", type=float, default=0.50)
    parser.add_argument("--minimum-geodesic-m", type=float, default=3.2)
    parser.add_argument("--maximum-geodesic-m", type=float, default=9.0)
    parser.add_argument("--episode-count", type=int, default=10)
    parser.add_argument("--episode-key", action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    if (
        arguments.minimum_prefix_clearance_m < 0.30
        or arguments.minimum_prefix_length_m <= 0.0
        or arguments.episode_count <= 0
        or not 0.0 < arguments.minimum_geodesic_m < arguments.maximum_geodesic_m
    ):
        raise SystemExit("invalid conservative Go2 clearance selection parameters")

    source_file = arguments.source_root / "val_unseen" / "val_unseen.json.gz"
    if source_file.stat().st_size != 448607 or sha256(source_file) != SOURCE_SHA256:
        raise SystemExit("official val_unseen payload does not match pinned bytes")
    with gzip.open(source_file, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)["episodes"]
    originals = {episode_key(item): item for item in raw}

    excluded: set[str] = set()
    for manifest_path in arguments.exclude_manifest:
        excluded.update(json.loads(manifest_path.read_text(encoding="utf-8"))["episode_keys"])

    clearance_payload = json.loads(arguments.clearance_audit.read_text(encoding="utf-8"))
    if clearance_payload.get("method") != "dense_reference_polyline_to_collision_height_matterport_vertices":
        raise SystemExit("unsupported Go2 clearance audit method")
    if float(clearance_payload.get("oracle_stop_distance_m", 0.0)) != 2.5:
        raise SystemExit("clearance audit does not use the frozen Oracle stop distance")
    clearance = {item["episode_key"]: item for item in clearance_payload.get("episodes", [])}
    scene_evidence = {item["scan"]: item for item in clearance_payload.get("scene_evidence", [])}

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
    for scene, items in filtered.items():
        for filtered_item in items:
            item = originals[episode_key(filtered_item)]
            heights = [float(point[1]) for point in item["reference_path"]]
            distance = float(item["info"]["geodesic_distance"])
            if (
                episode_key(item) not in excluded
                and int(item["trajectory_id"]) not in skip_list
                and arguments.minimum_geodesic_m
                <= distance
                <= arguments.maximum_geodesic_m
                and max(heights) - min(heights) <= 0.25
            ):
                by_scene[scene].append(item)

    selections: list[tuple[str, list[dict]]] = []
    geometry_clear_by_key: dict[str, dict] = {}
    for scene, items in by_scene.items():
        ranked = sorted(items, key=lambda item: (float(item["info"]["geodesic_distance"]), episode_key(item)))
        geometry_clear_all = []
        for item in ranked:
            evidence = clearance.get(episode_key(item))
            if evidence is None:
                continue
            oriented = evidence.get("required_prefix_oriented_geometry", {})
            if (
                float(evidence["start_clearance_m"]) >= arguments.minimum_prefix_clearance_m
                and float(evidence["required_prefix_minimum_clearance_m"])
                >= arguments.minimum_prefix_clearance_m
                and float(evidence["required_prefix_length_m"])
                >= arguments.minimum_prefix_length_m
                and float(oriented.get("nav2_radius_intrusion_fraction", 1.0)) == 0.0
                and float(
                    oriented.get("expanded_physical_body_intrusion_fraction", 1.0)
                ) == 0.0
                and float(
                    oriented.get("expanded_stop_polygon_intrusion_fraction", 1.0)
                ) == 0.0
            ):
                geometry_clear_all.append(item)
                geometry_clear_by_key[episode_key(item)] = item
        geometry_clear = []
        used_starts: set[tuple[float, float, float]] = set()
        for item in geometry_clear_all:
            authored_start = start_key(item)
            if authored_start in used_starts:
                continue
            used_starts.add(authored_start)
            geometry_clear.append(item)
        if geometry_clear:
            selections.append((scene, geometry_clear))
    candidate_by_key = {
        episode_key(item): item
        for _scene, items in selections
        for item in items
    }
    if arguments.episode_key:
        if (
            len(arguments.episode_key) != arguments.episode_count
            or len(set(arguments.episode_key)) != arguments.episode_count
        ):
            raise SystemExit(
                "explicit freeze requires --episode-count unique --episode-key values"
            )
        missing = [key for key in arguments.episode_key if key not in geometry_clear_by_key]
        if missing:
            raise SystemExit(f"explicit episode keys fail official/clearance filters: {missing}")
        selected = [geometry_clear_by_key[key] for key in arguments.episode_key]
        selected_scene = None
        selection_scope = "explicit frozen episode keys; duplicate authored starts retained"
        authored_start_policy = "explicit audited keys; reset-order behavior must be validated"
    else:
        single_scene = [
            pair for pair in selections if len(pair[1]) >= arguments.episode_count
        ]
        if single_scene:
            selected_scene, candidates = sorted(
                single_scene, key=lambda pair: (-len(pair[1]), pair[0])
            )[0]
            selected = candidates[: arguments.episode_count]
            selection_scope = "one fixed scene"
        else:
            candidates = [
                item
                for scene, items in sorted(selections)
                for item in items
            ]
            candidates.sort(
                key=lambda item: (
                    float(item["info"]["geodesic_distance"]),
                    str(item["scene_id"]),
                    episode_key(item),
                )
            )
            if len(candidates) < arguments.episode_count:
                raise SystemExit(
                    "fewer than --episode-count official episodes pass the physical "
                    "Go2 clearance audit"
                )
            selected = candidates[: arguments.episode_count]
            selected_scene = None
            selection_scope = "deterministic cross-scene fallback"
        authored_start_policy = "unique authored starts"
    output_file = arguments.output_root / "val_unseen" / "val_unseen.json.gz"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"episodes": selected}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    output_file.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))

    manifest = {
        "schema_version": 1,
        "phase": f"t3_go2_clear_{arguments.episode_count}_episodes",
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "selection": (
            f"{selection_scope}; official filter_stairs; single-floor; geodesic "
            f"{arguments.minimum_geodesic_m:.1f}-{arguments.maximum_geodesic_m:.1f}m; "
            f"{authored_start_policy}; required 2.5m-stop prefix; "
            f"scalar clearance >= {arguments.minimum_prefix_clearance_m:.2f}m; "
            "0.15m conservative sparse-mesh/body sweep margin over the 0.30m Nav2 radius; "
            "zero intrusion into Nav2 0.30m radius, physical body +0.03m, and "
            "Collision Monitor stop polygon +0.03m; "
            f"shortest {arguments.episode_count}; disjoint declared exclusion manifests"
        ),
        "integrity": "official episode objects copied value-for-value; no trajectory, instruction, goal, or metric mutation",
        "episode_count": arguments.episode_count,
        "scene_id": selected_scene,
        "scene_ids": sorted({str(item["scene_id"]) for item in selected}),
        "episode_keys": [episode_key(item) for item in selected],
        "start_keys": [list(start_key(item)) for item in selected],
        "duplicate_authored_start_count": len(selected)
        - len({start_key(item) for item in selected}),
        "authored_start_policy": authored_start_policy,
        "geodesic_distances_m": [float(item["info"]["geodesic_distance"]) for item in selected],
        "minimum_prefix_clearance_m": arguments.minimum_prefix_clearance_m,
        "minimum_prefix_length_m": arguments.minimum_prefix_length_m,
        "minimum_geodesic_m": arguments.minimum_geodesic_m,
        "maximum_geodesic_m": arguments.maximum_geodesic_m,
        "clearance_audit_sha256": sha256(arguments.clearance_audit),
        "selected_clearance_evidence": [clearance[episode_key(item)] for item in selected],
        "scene_geometry_evidence": [
            scene_evidence[scan]
            for scan in sorted(
                {
                    str(item["scene_id"]).replace("\\", "/").split("/")[-2]
                    for item in selected
                }
            )
            if scan in scene_evidence
        ],
        "overlay_sha256": sha256(output_file),
    }
    (arguments.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
