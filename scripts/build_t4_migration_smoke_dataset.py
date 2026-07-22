#!/usr/bin/env python3
"""Materialize a deterministic prefix for DGX-onboard functional runs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--episode-count", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.episode_count <= 20:
        raise ValueError("episode count must be between 1 and 20")
    source = args.source.resolve()
    output_root = args.output_root.resolve()
    output = output_root / "val_unseen" / "val_unseen.json.gz"
    manifest = output_root / "migration_smoke_dataset.json"
    if output_root.exists():
        raise FileExistsError(output_root)
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < args.episode_count:
        raise ValueError("source dataset has fewer episodes than requested")
    selected = dict(payload)
    selected_episodes = episodes[: args.episode_count]
    selected["episodes"] = selected_episodes
    output.parent.mkdir(parents=True)
    with gzip.GzipFile(filename=str(output), mode="wb", mtime=0) as stream:
        stream.write(
            json.dumps(selected, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
    result = {
        "schema_version": 1,
        "status": "PASS",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "episode_count": args.episode_count,
        "episode_ids": [
            str(episode.get("episode_id", episode.get("id", "")))
            for episode in selected_episodes
        ],
        "scene_ids": [str(episode.get("scene_id", "")) for episode in selected_episodes],
    }
    manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
