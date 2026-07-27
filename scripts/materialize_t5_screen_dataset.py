#!/usr/bin/env python3
"""Create a deterministic screening view of a sealed T5 episode set."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re


SAFE_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_key(episode: dict) -> str:
    trajectory = episode.get("trajectory_id")
    episode_id = episode.get("episode_id")
    if trajectory is None or episode_id is None:
        raise ValueError("every screening episode needs trajectory_id and episode_id")
    return f"{trajectory}_{episode_id}"


def write_gzip_json(path: Path, value: object) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(encoded)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int, choices=(1, 3))
    parser.add_argument("--episode-key")
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-episode-keys", required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve(strict=True)
    source = source_root / "val_unseen/val_unseen.json.gz"
    if not source.is_file() or source.is_symlink():
        raise SystemExit("frozen source dataset is missing or is a symlink")
    if not SAFE_SHA256.fullmatch(args.expected_source_sha256):
        raise SystemExit("expected source SHA-256 is malformed")
    source_hash = sha256(source)
    if source_hash != args.expected_source_sha256:
        raise SystemExit("frozen source dataset SHA-256 mismatch")

    with gzip.open(source, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list) or len(episodes) not in {5, 10}:
        raise SystemExit("screen source must contain five or ten sealed episodes")
    source_keys = [episode_key(episode) for episode in episodes]
    expected_keys = args.expected_episode_keys.split(",")
    if (
        len(expected_keys) != len(episodes)
        or len(set(expected_keys)) != len(episodes)
    ):
        raise SystemExit(
            "expected episode keys must exactly name the sealed source episodes"
        )
    if source_keys != expected_keys:
        raise SystemExit("source episode order differs from the frozen input binding")
    if args.episode_key is not None:
        if args.count != 1:
            raise SystemExit("episode-key selection is only valid for count=1")
        if args.episode_key not in source_keys:
            raise SystemExit("episode-key is not present in the frozen input binding")
        selected_episodes = [episodes[source_keys.index(args.episode_key)]]
        selection = "frozen_episode_key"
    else:
        selected_episodes = episodes[: args.count]
        selection = "frozen_first_n"

    output_root = args.output_root.resolve(strict=False)
    if output_root.exists() or output_root.is_symlink():
        raise SystemExit("screen output root already exists")
    output = output_root / "val_unseen/val_unseen.json.gz"
    selected = dict(payload)
    selected["episodes"] = selected_episodes
    try:
        write_gzip_json(output, selected)
        audit = {
            "schema_version": 1,
            "status": "PASS",
            "selection": selection,
            "source_dataset": str(source),
            "source_sha256": source_hash,
            "source_episode_count": len(episodes),
            "source_episode_keys": source_keys,
            "output_dataset": str(output),
            "output_sha256": sha256(output),
            "selected_episode_count": args.count,
            "selected_episode_keys": [
                episode_key(episode) for episode in selected_episodes
            ],
        }
        audit_path = output_root / "screen_dataset_audit.json"
        audit_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except BaseException:
        if output_root.exists():
            import shutil

            shutil.rmtree(output_root)
        raise
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
