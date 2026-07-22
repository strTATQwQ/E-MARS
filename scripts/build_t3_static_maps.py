#!/usr/bin/env python3
"""Build deterministic per-episode MP3D occupancy maps for T3 Nav2 runs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
BUILDER_REVISION = "t3_static_map_v4"
PLY_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("tx", "<f4"), ("ty", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ply_vertices(path: Path) -> np.memmap:
    with path.open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = stream.readline()
            if not line:
                raise RuntimeError(f"truncated PLY header: {path}")
            header += line
    text = header.decode("ascii")
    if "format binary_little_endian 1.0" not in text:
        raise RuntimeError(f"unsupported PLY encoding: {path}")
    count = int(
        next(
            line.split()[2]
            for line in text.splitlines()
            if line.startswith("element vertex ")
        )
    )
    return np.memmap(
        path, dtype=PLY_VERTEX_DTYPE, mode="r", offset=len(header), shape=(count,)
    )


def scan_name(item: dict) -> str:
    parts = str(item["scene_id"]).replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"invalid scene_id: {item['scene_id']}")
    return parts[-2]


def reference_height(item: dict) -> float:
    return round(
        float(np.median([float(point[1]) for point in item["reference_path"]])), 4
    )


def navigation_points(item: dict) -> list[tuple[float, float]]:
    points = [
        (float(point[0]), -float(point[2])) for point in item["reference_path"]
    ]
    for goal in item.get("goals", []):
        position = goal.get("position")
        if isinstance(position, list) and len(position) >= 3:
            points.append((float(position[0]), -float(position[2])))
    return points


def required_prefix_samples(
    item: dict, *, spacing_m: float = 0.05, stop_distance_m: float = 2.5
) -> np.ndarray:
    raw = [
        np.asarray([float(point[0]), -float(point[2])], dtype=np.float64)
        for point in item["reference_path"]
    ]
    samples: list[np.ndarray] = []
    for first, second in zip(raw, raw[1:]):
        count = max(1, int(math.ceil(float(np.linalg.norm(second - first)) / spacing_m)))
        samples.extend(first + (second - first) * (index / count) for index in range(count))
    samples.append(raw[-1])
    goal_position = item.get("goals", [{}])[0].get("position", item["reference_path"][-1])
    goal = np.asarray(
        [float(goal_position[0]), -float(goal_position[2])], dtype=np.float64
    )
    array = np.asarray(samples, dtype=np.float64)
    reached = np.flatnonzero(np.linalg.norm(array - goal, axis=1) <= stop_distance_m)
    end = int(reached[0]) if len(reached) else len(array) - 1
    return array[: end + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resolution-m", type=float, default=0.05)
    parser.add_argument("--margin-m", type=float, default=4.0)
    parser.add_argument("--obstacle-z-min-offset-m", type=float, default=0.08)
    parser.add_argument("--obstacle-z-max-offset-m", type=float, default=1.50)
    parser.add_argument("--minimum-required-prefix-clearance-m", type=float, default=0.40)
    args = parser.parse_args()
    if (
        args.resolution_m <= 0.0
        or args.margin_m <= 0.0
        or args.minimum_required_prefix_clearance_m < 0.0
        or args.obstacle_z_max_offset_m <= args.obstacle_z_min_offset_m
    ):
        raise SystemExit("invalid static-map geometry parameters")

    dataset_sha = sha256(args.dataset)
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("builder_revision") == BUILDER_REVISION
            and existing.get("dataset_sha256") == dataset_sha
            and existing.get("required_prefix_minimum_clearance_gate_m")
            == args.minimum_required_prefix_clearance_m
        ):
            print(json.dumps(existing, indent=2, sort_keys=True))
            return
        raise RuntimeError(f"refusing to replace incompatible static-map cache: {manifest_path}")

    with gzip.open(args.dataset, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    if not episodes:
        raise RuntimeError("static-map dataset is empty")
    groups: dict[tuple[str, float], list[tuple[int, dict]]] = defaultdict(list)
    for generation, item in enumerate(episodes):
        groups[(scan_name(item), reference_height(item))].append((generation, item))

    args.output_root.mkdir(parents=True, exist_ok=True)
    map_entries: dict[str, dict] = {}
    generation_entries: list[dict] = []
    for (scan, floor_height), members in sorted(groups.items()):
        ply = args.scene_root / scan / "house_segmentations" / f"{scan}.ply"
        vertices = load_ply_vertices(ply)
        path_points = [point for _, item in members for point in navigation_points(item)]
        xs = np.asarray([point[0] for point in path_points], dtype=np.float64)
        ys = np.asarray([point[1] for point in path_points], dtype=np.float64)
        origin_x = math.floor((float(xs.min()) - args.margin_m) / args.resolution_m) * args.resolution_m
        origin_y = math.floor((float(ys.min()) - args.margin_m) / args.resolution_m) * args.resolution_m
        maximum_x = math.ceil((float(xs.max()) + args.margin_m) / args.resolution_m) * args.resolution_m
        maximum_y = math.ceil((float(ys.max()) + args.margin_m) / args.resolution_m) * args.resolution_m
        width = int(round((maximum_x - origin_x) / args.resolution_m))
        height = int(round((maximum_y - origin_y) / args.resolution_m))
        if width <= 0 or height <= 0 or width * height > 100_000_000:
            raise RuntimeError(f"invalid static-map dimensions for {scan}: {width}x{height}")
        mask = (
            (vertices["z"] >= floor_height + args.obstacle_z_min_offset_m)
            & (vertices["z"] <= floor_height + args.obstacle_z_max_offset_m)
            & (vertices["x"] >= origin_x)
            & (vertices["x"] < maximum_x)
            & (vertices["y"] >= origin_y)
            & (vertices["y"] < maximum_y)
        )
        surface_x = np.asarray(vertices["x"][mask], dtype=np.float64)
        surface_y = np.asarray(vertices["y"][mask], dtype=np.float64)
        grid = np.zeros((height, width), dtype=np.uint8)
        columns = np.floor((surface_x - origin_x) / args.resolution_m).astype(np.int64)
        rows = np.floor((surface_y - origin_y) / args.resolution_m).astype(np.int64)
        valid = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
        grid[rows[valid], columns[valid]] = 100
        occupied = grid > 0
        padded = np.pad(occupied, 1, mode="constant", constant_values=False)
        dilated = np.zeros_like(occupied)
        for row_offset in range(3):
            for column_offset in range(3):
                dilated |= padded[
                    row_offset : row_offset + height,
                    column_offset : column_offset + width,
                ]
        grid[dilated] = 100
        occupied_rows, occupied_columns = np.nonzero(grid == 100)
        occupied_x = origin_x + (occupied_columns.astype(np.float64) + 0.5) * args.resolution_m
        occupied_y = origin_y + (occupied_rows.astype(np.float64) + 0.5) * args.resolution_m
        if len(occupied_x) == 0:
            raise RuntimeError(f"static map has no occupied cells: {scan} at {floor_height}")
        map_key = f"{scan}_{floor_height:+.4f}"
        map_file = args.output_root / f"{map_key}.bin"
        map_file.write_bytes(grid.tobytes(order="C"))
        map_entry = {
            "key": map_key,
            "scan": scan,
            "reference_height_m": floor_height,
            "file": map_file.name,
            "sha256": sha256(map_file),
            "bytes": map_file.stat().st_size,
            "width": width,
            "height": height,
            "resolution_m": args.resolution_m,
            "origin_xy": [origin_x, origin_y],
            "occupied_cell_count": int(np.count_nonzero(grid == 100)),
            "source_collision_height_vertex_count": int(np.count_nonzero(mask)),
            "source_ply": str(ply.relative_to(args.scene_root)).replace("\\", "/"),
            "source_ply_sha256": sha256(ply),
        }
        map_entries[map_key] = map_entry
        for generation, item in members:
            prefix = required_prefix_samples(item)
            prefix_clearances = np.asarray(
                [
                    float(np.min(np.hypot(occupied_x - point[0], occupied_y - point[1])))
                    for point in prefix
                ],
                dtype=np.float64,
            )
            minimum_clearance = float(prefix_clearances.min())
            if minimum_clearance < args.minimum_required_prefix_clearance_m:
                raise RuntimeError(
                    "required static-map prefix clearance below "
                    f"{args.minimum_required_prefix_clearance_m:.2f}m for generation "
                    f"{generation}: {minimum_clearance:.3f}m"
                )
            generation_entries.append(
                {
                    "generation": generation,
                    "episode_id": str(item.get("episode_id", "")),
                    "trajectory_id": str(item.get("trajectory_id", "")),
                    "map_key": map_key,
                    "start_map_xy": [
                        float(item["reference_path"][0][0]),
                        -float(item["reference_path"][0][2]),
                    ],
                    "reference_height_m": floor_height,
                    "required_prefix_sample_count": int(len(prefix)),
                    "required_prefix_minimum_static_clearance_m": minimum_clearance,
                    "start_static_clearance_m": float(prefix_clearances[0]),
                }
            )
        del vertices, grid, surface_x, surface_y

    payload = {
        "schema_version": SCHEMA_VERSION,
        "builder_revision": BUILDER_REVISION,
        "method": "MP3D collision-height PLY vertices rasterized at 5cm and dilated one cell",
        "coordinate_transform": "InternNav [x,y,z] -> Isaac/map [x,-z]; PLY [x,y,z] already Isaac-aligned",
        "dataset": args.dataset.name,
        "dataset_sha256": dataset_sha,
        "episode_count": len(episodes),
        "map_count": len(map_entries),
        "resolution_m": args.resolution_m,
        "margin_m": args.margin_m,
        "obstacle_z_offsets_m": [
            args.obstacle_z_min_offset_m,
            args.obstacle_z_max_offset_m,
        ],
        "required_prefix_stop_distance_m": 2.5,
        "required_prefix_sample_spacing_m": 0.05,
        "required_prefix_minimum_clearance_gate_m": args.minimum_required_prefix_clearance_m,
        "maps": map_entries,
        "generations": sorted(generation_entries, key=lambda item: item["generation"]),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
