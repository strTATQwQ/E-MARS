#!/usr/bin/env python3
"""Create an immutable one-episode subset from an existing official dataset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.output_root.resolve() / "val_unseen" / "val_unseen.json.gz"
    if destination.exists():
        raise FileExistsError(destination)
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        episodes = payload["episodes"]
        if not episodes:
            raise RuntimeError("official source dataset is empty")
        output_payload = {**payload, "episodes": [episodes[0]]}
    elif isinstance(payload, list) and payload:
        output_payload = [payload[0]]
    else:
        raise RuntimeError("official source dataset has an unsupported shape")
    destination.parent.mkdir(parents=True, exist_ok=False)
    with gzip.GzipFile(destination, "wb", mtime=0) as compressed:
        compressed.write(json.dumps(output_payload, separators=(",", ":")).encode("utf-8"))
    output_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": output_sha,
        "episode_count": 1,
        "selection": "first_official_episode_for_engineering_smoke_only",
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.source_manifest is not None:
        source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
        scenarios = source_manifest.get("scenarios")
        if not isinstance(scenarios, list) or not scenarios:
            raise RuntimeError("obstacle manifest has no scenarios")
        obstacle_manifest = {
            **source_manifest,
            "scenario_plan": list(source_manifest.get("scenario_plan", []))[:1],
            "scenarios": [scenarios[0]],
            "selected_episode_count": 1,
            "selection_policy": "first_frozen_obstacle_oracle_episode_for_engineering_smoke_only",
            "source_manifest_sha256": hashlib.sha256(
                args.source_manifest.read_bytes()
            ).hexdigest(),
            "output_sha256": output_sha,
        }
        (args.output_root / "obstacle_manifest.json").write_text(
            json.dumps(obstacle_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
