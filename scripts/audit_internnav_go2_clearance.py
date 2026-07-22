#!/usr/bin/env python3
"""Audit physical Go2 clearance along immutable InternNav reference paths.

The Matterport PLY is already in the Isaac frame (X/Y ground plane, Z up),
where an InternNav point ``[x, y, z]`` maps to ``[x, -z, y]``.  Floor and
ceiling returns are excluded, then a dense 5 cm sampling of every reference
polyline is queried against the remaining scene surface vertices.  This is a
conservative eligibility audit; it never edits an episode or its path.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


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
        next(line.split()[2] for line in text.splitlines() if line.startswith("element vertex "))
    )
    return np.memmap(path, dtype=PLY_VERTEX_DTYPE, mode="r", offset=len(header), shape=(count,))


def scan_name(item: dict) -> str:
    parts = str(item["scene_id"]).replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"invalid scene_id: {item['scene_id']}")
    return parts[-2]


def episode_key(item: dict) -> str:
    return f"{item['trajectory_id']}_{item['episode_id']}"


def world_polyline(item: dict, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    raw = item["reference_path"]
    output: list[tuple[float, float]] = []
    headings: list[float] = []
    for first, second in zip(raw, raw[1:]):
        a = np.asarray([float(first[0]), -float(first[2])], dtype=np.float64)
        b = np.asarray([float(second[0]), -float(second[2])], dtype=np.float64)
        distance = float(np.linalg.norm(b - a))
        heading = math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))
        count = max(1, int(math.ceil(distance / spacing)))
        for index in range(count):
            point = a + (b - a) * (index / count)
            output.append((float(point[0]), float(point[1])))
            headings.append(heading)
    last = raw[-1]
    output.append((float(last[0]), -float(last[2])))
    headings.append(headings[-1] if headings else 0.0)
    return np.asarray(output, dtype=np.float64), np.asarray(headings, dtype=np.float64)


def required_prefix(samples: np.ndarray, item: dict, stop_distance: float) -> np.ndarray:
    goal = item.get("goals", [{}])[0].get("position", item["reference_path"][-1])
    goal_xy = np.asarray([float(goal[0]), -float(goal[2])], dtype=np.float64)
    distances = np.linalg.norm(samples - goal_xy, axis=1)
    reached = np.flatnonzero(distances <= stop_distance)
    end = int(reached[0]) if len(reached) else len(samples) - 1
    return samples[: end + 1]


def oriented_intrusions(
    samples: np.ndarray,
    headings: np.ndarray,
    tree: cKDTree,
    surfaces: np.ndarray,
) -> dict[str, float | int]:
    neighborhoods = tree.query_ball_point(samples, r=0.65, workers=-1)
    stop_rows = expanded_stop_rows = body_rows = expanded_body_rows = circle_rows = 0
    stop_points = expanded_stop_points = body_points = expanded_body_points = circle_points = 0
    for sample, heading, indices in zip(samples, headings, neighborhoods):
        if not indices:
            continue
        delta = surfaces[np.asarray(indices, dtype=np.int64)].astype(np.float64) - sample
        cosine, sine = math.cos(float(heading)), math.sin(float(heading))
        forward = cosine * delta[:, 0] + sine * delta[:, 1]
        lateral = -sine * delta[:, 0] + cosine * delta[:, 1]
        stop = (forward >= -0.26) & (forward <= 0.38) & (np.abs(lateral) <= 0.26)
        expanded_stop = (
            (forward >= -0.29) & (forward <= 0.41) & (np.abs(lateral) <= 0.29)
        )
        body = (forward >= -0.255) & (forward <= 0.293) & (np.abs(lateral) <= 0.20)
        expanded_body = (
            (forward >= -0.285) & (forward <= 0.323) & (np.abs(lateral) <= 0.23)
        )
        circle = np.hypot(forward, lateral) <= 0.30
        stop_count = int(np.count_nonzero(stop))
        expanded_stop_count = int(np.count_nonzero(expanded_stop))
        body_count = int(np.count_nonzero(body))
        expanded_body_count = int(np.count_nonzero(expanded_body))
        circle_count = int(np.count_nonzero(circle))
        stop_rows += int(stop_count > 0)
        expanded_stop_rows += int(expanded_stop_count > 0)
        body_rows += int(body_count > 0)
        expanded_body_rows += int(expanded_body_count > 0)
        circle_rows += int(circle_count > 0)
        stop_points += stop_count
        expanded_stop_points += expanded_stop_count
        body_points += body_count
        expanded_body_points += expanded_body_count
        circle_points += circle_count
    denominator = max(1, len(samples))
    return {
        "stop_polygon_intrusion_sample_count": stop_rows,
        "stop_polygon_intrusion_fraction": stop_rows / denominator,
        "stop_polygon_intrusion_point_count": stop_points,
        "expanded_stop_polygon_intrusion_sample_count": expanded_stop_rows,
        "expanded_stop_polygon_intrusion_fraction": expanded_stop_rows / denominator,
        "expanded_stop_polygon_intrusion_point_count": expanded_stop_points,
        "physical_body_intrusion_sample_count": body_rows,
        "physical_body_intrusion_fraction": body_rows / denominator,
        "physical_body_intrusion_point_count": body_points,
        "expanded_physical_body_intrusion_sample_count": expanded_body_rows,
        "expanded_physical_body_intrusion_fraction": expanded_body_rows / denominator,
        "expanded_physical_body_intrusion_point_count": expanded_body_points,
        "nav2_radius_intrusion_sample_count": circle_rows,
        "nav2_radius_intrusion_fraction": circle_rows / denominator,
        "nav2_radius_intrusion_point_count": circle_points,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-spacing-m", type=float, default=0.05)
    parser.add_argument("--surface-z-lower-offset-m", type=float, default=-0.08)
    parser.add_argument("--surface-z-upper-offset-m", type=float, default=0.52)
    parser.add_argument("--oracle-stop-distance-m", type=float, default=2.5)
    arguments = parser.parse_args()
    if (
        arguments.sample_spacing_m <= 0.0
        or arguments.surface_z_upper_offset_m <= arguments.surface_z_lower_offset_m
        or arguments.oracle_stop_distance_m <= 0.0
    ):
        raise SystemExit("invalid clearance audit geometry parameters")

    with gzip.open(arguments.dataset, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in episodes:
        grouped[scan_name(item)].append(item)

    records: list[dict] = []
    scene_evidence: list[dict] = []
    for scan, items in sorted(grouped.items()):
        ply = arguments.scene_root / scan / "house_segmentations" / f"{scan}.ply"
        vertices = load_ply_vertices(ply)
        vertical = vertices["z"]
        by_height: dict[float, list[dict]] = defaultdict(list)
        for item in items:
            reference_height = float(
                np.median([float(point[1]) for point in item["reference_path"]])
            )
            by_height[round(reference_height, 4)].append(item)
        floor_band_evidence = []
        for reference_height, height_items in sorted(by_height.items()):
            surface_z_min = reference_height + arguments.surface_z_lower_offset_m
            surface_z_max = reference_height + arguments.surface_z_upper_offset_m
            mask = (vertical >= surface_z_min) & (vertical <= surface_z_max)
            surfaces = np.column_stack((vertices["x"][mask], vertices["y"][mask])).astype(
                np.float32, copy=False
            )
            if len(surfaces) == 0:
                raise RuntimeError(f"no collision-height vertices in {ply} at {reference_height}")
            tree = cKDTree(surfaces)
            floor_band_evidence.append(
                {
                    "reference_height_m": reference_height,
                    "surface_z_range_m": [surface_z_min, surface_z_max],
                    "collision_height_vertex_count": int(len(surfaces)),
                }
            )
            for item in height_items:
                samples, headings = world_polyline(item, arguments.sample_spacing_m)
                distances, indices = tree.query(samples, k=1, workers=-1)
                prefix = required_prefix(samples, item, arguments.oracle_stop_distance_m)
                prefix_headings = headings[: len(prefix)]
                prefix_distances, prefix_indices = tree.query(prefix, k=1, workers=-1)
                prefix_intrusions = oriented_intrusions(prefix, prefix_headings, tree, surfaces)
                minimum_index = int(np.argmin(distances))
                prefix_minimum_index = int(np.argmin(prefix_distances))
                nearest = surfaces[int(indices[minimum_index])]
                prefix_nearest = surfaces[int(prefix_indices[prefix_minimum_index])]
                prefix_length = float(
                    np.linalg.norm(np.diff(prefix, axis=0), axis=1).sum()
                    if len(prefix) > 1
                    else 0.0
                )
                records.append(
                    {
                    "episode_key": episode_key(item),
                    "scene": scan,
                    "reference_sample_count": int(len(samples)),
                    "minimum_clearance_m": float(distances[minimum_index]),
                    "p05_clearance_m": float(np.quantile(distances, 0.05)),
                    "median_clearance_m": float(np.median(distances)),
                    "start_clearance_m": float(distances[0]),
                    "required_prefix_sample_count": int(len(prefix)),
                    "required_prefix_length_m": prefix_length,
                    "required_prefix_minimum_clearance_m": float(
                        prefix_distances[prefix_minimum_index]
                    ),
                    "required_prefix_p05_clearance_m": float(
                        np.quantile(prefix_distances, 0.05)
                    ),
                    "required_prefix_fraction_below_0_38": float(
                        np.mean(prefix_distances < 0.38)
                    ),
                    "required_prefix_fraction_below_0_42": float(
                        np.mean(prefix_distances < 0.42)
                    ),
                    "required_prefix_fraction_below_0_45": float(
                        np.mean(prefix_distances < 0.45)
                    ),
                    "required_prefix_minimum_path_xy": prefix[
                        prefix_minimum_index
                    ].astype(float).tolist(),
                    "required_prefix_nearest_surface_xy": prefix_nearest.astype(float).tolist(),
                    "required_prefix_oriented_geometry": prefix_intrusions,
                    "fraction_below_0_30": float(np.mean(distances < 0.30)),
                    "fraction_below_0_35": float(np.mean(distances < 0.35)),
                    "fraction_below_0_38": float(np.mean(distances < 0.38)),
                    "fraction_below_0_42": float(np.mean(distances < 0.42)),
                    "minimum_path_xy": samples[minimum_index].astype(float).tolist(),
                    "nearest_surface_xy": nearest.astype(float).tolist(),
                        "geodesic_distance_m": float(item["info"]["geodesic_distance"]),
                        "reference_height_m": reference_height,
                        "surface_z_range_m": [surface_z_min, surface_z_max],
                    }
                )
            del tree, surfaces
        scene_evidence.append(
            {
                "scan": scan,
                "ply_path": f"{scan}/house_segmentations/{scan}.ply",
                "ply_bytes": ply.stat().st_size,
                "ply_sha256": sha256(ply),
                "total_vertex_count": int(len(vertices)),
                "floor_bands": floor_band_evidence,
            }
        )
        del vertices

    records.sort(key=lambda item: item["episode_key"])
    payload = {
        "schema_version": 1,
        "method": "dense_reference_polyline_to_collision_height_matterport_vertices",
        "dataset_path": arguments.dataset.name,
        "dataset_sha256": sha256(arguments.dataset),
        "sample_spacing_m": arguments.sample_spacing_m,
        "oracle_stop_distance_m": arguments.oracle_stop_distance_m,
        "surface_z_offsets_from_reference_height_m": [
            arguments.surface_z_lower_offset_m,
            arguments.surface_z_upper_offset_m,
        ],
        "coordinate_transform": "InternNav [x,y,z] -> Isaac ground [x,-z], PLY z is vertical",
        "oriented_envelopes_base_frame_m": {
            "physical_body": {"x": [-0.255, 0.293], "abs_y_max": 0.20},
            "physical_body_plus_0_03": {"x": [-0.285, 0.323], "abs_y_max": 0.23},
            "nav2_robot_radius": 0.30,
            "collision_monitor_stop_polygon": {"x": [-0.26, 0.38], "abs_y_max": 0.26},
            "collision_monitor_stop_polygon_plus_0_03": {
                "x": [-0.29, 0.41], "abs_y_max": 0.29
            },
        },
        "scene_evidence": scene_evidence,
        "episode_count": len(records),
        "episodes": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
