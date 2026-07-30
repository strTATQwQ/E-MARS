#!/usr/bin/env python3
"""Materialize an exact, ordered episode subset from a pinned source dataset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_key(value: dict) -> str:
    return f"{value['trajectory_id']}_{value['episode_id']}"


def deterministic_gzip(path: Path, value: object) -> None:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(encoded)
    os.replace(temporary, path)


def materialize(source_root: Path, manifest_path: Path, output_root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = manifest.get("episode_keys")
    if (
        manifest.get("status") != "FROZEN_FOR_EXECUTION"
        or manifest.get("episode_count") != 30
        or not isinstance(keys, list)
        or len(keys) != len(set(keys))
        or len(keys) != 30
    ):
        raise ValueError("paired-30 manifest is not a unique frozen 30-episode set")
    source = source_root / "val_unseen/val_unseen.json.gz"
    if sha256(source) != manifest.get("source_sha256"):
        raise ValueError("official source dataset hash differs from the manifest")
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        source_payload = json.load(stream)
    episodes = source_payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("official source dataset has no episode list")
    by_key = {episode_key(item): item for item in episodes}
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise ValueError(f"frozen episodes are absent from official source: {missing}")
    selected_payload = dict(source_payload)
    selected_payload["episodes"] = [by_key[key] for key in keys]
    output = output_root / "val_unseen/val_unseen.json.gz"
    if output_root.exists():
        if not output.is_file() or sha256(output) != manifest.get("dataset_sha256"):
            raise ValueError("existing paired-30 dataset does not match the manifest")
        selection = "verified_existing"
    else:
        deterministic_gzip(output, selected_payload)
        if sha256(output) != manifest.get("dataset_sha256"):
            raise ValueError("materialized paired-30 dataset hash differs from manifest")
        selection = "materialized"
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "selection": selection,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "dataset_sha256": sha256(output),
        "episode_count": len(keys),
        "episode_keys": keys,
    }
    audit_path = output_root / "materialization_audit.json"
    temporary = audit_path.with_name(f".{audit_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, audit_path)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    value = materialize(
        args.source_root.resolve(strict=True),
        args.manifest.resolve(strict=True),
        args.output_root.resolve(strict=False),
    )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
