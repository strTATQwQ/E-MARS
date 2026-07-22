#!/usr/bin/env python3
"""Freeze the five-case T3 continuous-control diagnostic subset."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path


SCENARIOS = (
    (8, "straight_1", 93001, "Walk straight to diagnostic marker one.",
     [[100.0, 0.171628, -100.0], [100.0, 0.171628, -101.2]]),
    (7, "straight_2", 93002, "Walk straight to diagnostic marker two.",
     [[110.0, 0.171628, -100.0], [110.0, 0.171628, -101.8]]),
    (2, "turn_1", 93003, "Walk forward and turn right at the diagnostic corner.",
     [[120.0, 0.171628, -100.0], [120.0, 0.171628, -100.8], [120.8, 0.171628, -100.8]]),
    (5, "turn_2", 93004, "Walk forward and turn left at the diagnostic corner.",
     [[130.0, 0.171628, -100.0], [130.0, 0.171628, -100.8], [129.2, 0.171628, -100.8]]),
    (8, "doorway_1", 93005, "Walk straight through the diagnostic doorway.",
     [[140.0, 0.171628, -100.0], [140.0, 0.171628, -101.4]]),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instruction_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(instruction_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(instruction_text(item) for item in value)
    return ""


def geometry(episode: dict[str, object]) -> dict[str, object]:
    raw_path = episode["reference_path"]
    if not isinstance(raw_path, list) or len(raw_path) < 2:
        raise ValueError("reference path must contain at least two points")
    points = [(float(item[0]), float(item[2])) for item in raw_path]
    segments: list[tuple[float, float]] = []
    for first, second in zip(points, points[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = math.hypot(dx, dy)
        if length > 0.05:
            segments.append((math.atan2(dy, dx), length))
    path_length = sum(item[1] for item in segments)
    cumulative_turn = sum(
        abs((second[0] - first[0] + math.pi) % (2.0 * math.pi) - math.pi)
        for first, second in zip(segments, segments[1:])
    )
    direct = math.dist(points[0], points[-1])
    text = instruction_text(episode.get("instruction", {})).lower()
    return {
        "reference_point_count": len(points),
        "path_length_m": round(path_length, 6),
        "directness": round(direct / path_length, 6) if path_length else 0.0,
        "cumulative_turn_deg": round(math.degrees(cumulative_turn), 6),
        "doorway_keyword_present": any(
            word in text for word in ("door", "doorway", "entryway", "entrance")
        ),
    }


def write_gzip_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(encoded)


def bounded_reference_path(raw_path: list[list[float]]) -> list[list[float]]:
    """Return a deterministic metric prefix that fits the diagnostic budget."""
    bounded = [copy.deepcopy(raw_path[0])]
    remaining = DIAGNOSTIC_ROUTE_LENGTH_M
    for first, second in zip(raw_path, raw_path[1:]):
        dx = float(second[0]) - float(first[0])
        dz = float(second[2]) - float(first[2])
        segment = math.hypot(dx, dz)
        if segment <= 1e-9:
            continue
        if segment <= remaining + 1e-9:
            bounded.append(copy.deepcopy(second))
            remaining -= segment
            if remaining <= 1e-9:
                break
            continue
        ratio = remaining / segment
        bounded.append(
            [
                float(first[axis])
                + ratio * (float(second[axis]) - float(first[axis]))
                for axis in range(3)
            ]
        )
        remaining = 0.0
        break
    if len(bounded) < 2 or remaining > 1e-6:
        raise RuntimeError("diagnostic route is shorter than the bounded prefix")
    return bounded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    with gzip.open(args.source, "rt", encoding="utf-8") as stream:
        source_payload = json.load(stream)
    episodes = source_payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 10:
        raise RuntimeError("expected the frozen ten-episode T2 Oracle dataset")

    selected = []
    evidence = []
    for source_index, category, trajectory_id, text, reference_path in SCENARIOS:
        episode = copy.deepcopy(episodes[source_index])
        episode["episode_id"] = trajectory_id
        episode["trajectory_id"] = trajectory_id
        episode["start_position"] = copy.deepcopy(reference_path[0])
        episode["start_rotation"] = [0.0, 0.0, 0.0, 1.0]
        episode["reference_path"] = copy.deepcopy(reference_path)
        token_count = len(episode.get("instruction", {}).get("instruction_tokens", []))
        episode["instruction"] = {
            "instruction_text": text,
            "instruction_tokens": [0] * token_count,
        }
        goal = copy.deepcopy(episode["goals"][0])
        goal["position"] = copy.deepcopy(episode["reference_path"][-1])
        goal["radius"] = 0.5
        episode["goals"] = [goal]
        route_length = sum(
            math.hypot(second[0] - first[0], second[2] - first[2])
            for first, second in zip(reference_path, reference_path[1:])
        )
        episode.setdefault("info", {})["geodesic_distance"] = route_length
        route_transform = "synthetic_isolated_static_course_from_T2_schema_template"
        selected.append(episode)
        features = geometry(episode)
        if category.startswith("straight_") and features["cumulative_turn_deg"] > 15.0:
            raise RuntimeError("frozen straight diagnostic exceeds 15 degrees")
        if category.startswith("turn_") and features["cumulative_turn_deg"] < 45.0:
            raise RuntimeError("frozen turn diagnostic lost its turn geometry")
        if category == "doorway_1" and not features["doorway_keyword_present"]:
            raise RuntimeError("frozen doorway diagnostic no longer has doorway semantics")
        evidence.append(
            {
                "category": category,
                "source_index": source_index,
                "trajectory_id": str(episode.get("trajectory_id", "")),
                "route_transform": route_transform,
                **features,
            }
        )

    output = args.output_root / "val_unseen" / "val_unseen.json.gz"
    write_gzip_json(output, {"episodes": selected})
    manifest = {
        "schema_version": 1,
        "selection_policy": "fixed_isolated_static_course_two_straight_two_90deg_turns_one_1m_doorway",
        "source_episode_count": len(episodes),
        "selected_episode_count": len(selected),
        "source_sha256": sha256(args.source),
        "output_sha256": sha256(output),
        "scenarios": evidence,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
