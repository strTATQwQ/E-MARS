#!/usr/bin/env python3
"""Bind a staged completion_sim oracle dataset to the requested episode order."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re


SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")


def episode_key(episode: dict[str, object]) -> str:
    trajectory = episode.get("trajectory_id")
    episode_id = episode.get("episode_id")
    if trajectory is None or episode_id is None:
        raise ValueError("oracle episode requires trajectory_id and episode_id")
    return f"{trajectory}_{episode_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--episode-keys", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dataset = args.dataset.resolve(strict=True)
    if not dataset.is_file() or dataset.is_symlink():
        raise SystemExit("oracle dataset must be a regular non-symlink file")
    expected = args.episode_keys.split(",")
    if not expected or len(set(expected)) != len(expected):
        raise SystemExit("oracle episode keys must be non-empty and unique")
    if not all(SAFE_KEY.fullmatch(key) for key in expected):
        raise SystemExit("oracle episode key contains unsafe characters")
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list):
        raise SystemExit("oracle dataset has no episode list")
    observed = [episode_key(episode) for episode in episodes]
    if observed != expected:
        raise SystemExit(
            f"oracle dataset order mismatch: expected={expected!r} observed={observed!r}"
        )
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    output = args.output.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise SystemExit("oracle dataset binding output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": 1,
        "status": "PASS",
        "dataset": str(dataset),
        "dataset_sha256": digest,
        "episode_count": len(episodes),
        "episode_keys": observed,
        "termination_mode": "oracle_termination",
        "success_radius_m": 2.5,
        "credits_model_stop": False,
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
