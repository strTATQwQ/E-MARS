#!/usr/bin/env python3
"""Freeze disjoint five-episode T4.1 development and held-out splits."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def identity(episode: dict[str, Any]) -> str:
    trajectory = episode.get("trajectory_id", episode.get("trajectory", ""))
    episode_id = episode.get("episode_id", episode.get("episode", ""))
    return f"{trajectory}_{episode_id}"


def write_split(root: Path, episodes: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    target = root / "val_unseen/val_unseen.json.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"episodes": episodes}
    with gzip.GzipFile(filename=str(target), mode="wb", mtime=0) as raw:
        raw.write(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "episode_count": len(episodes),
        "episode_ids": [identity(item) for item in episodes],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--dev-output", required=True, type=Path)
    parser.add_argument("--heldout-output", required=True, type=Path)
    args = parser.parse_args()
    canary = read(args.canary)
    pilot = read(args.pilot)
    dev = list(canary["episodes"])
    if len(dev) != 5:
        raise RuntimeError(f"development source must contain 5 episodes, got {len(dev)}")
    dev_ids = {identity(item) for item in dev}
    heldout = [item for item in pilot["episodes"] if identity(item) not in dev_ids][:5]
    if len(heldout) != 5:
        raise RuntimeError("pilot source does not contain five disjoint held-out episodes")
    heldout_ids = {identity(item) for item in heldout}
    if dev_ids & heldout_ids:
        raise RuntimeError("development and held-out split overlap")
    result = {
        "schema_version": 1,
        "development": write_split(args.dev_output, dev, args.canary),
        "heldout": write_split(args.heldout_output, heldout, args.pilot),
        "overlap_count": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
