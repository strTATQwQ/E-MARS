#!/usr/bin/env python3
"""Overlay a final-pilot deployment binding with one frozen paired-30 episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-binding", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-key", required=True)
    parser.add_argument("--lane", choices=("a", "b"), required=True)
    parser.add_argument("--static-map-manifest-path", required=True)
    parser.add_argument("--static-map-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.input_binding.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    keys = manifest.get("episode_keys")
    lane_sets = manifest.get("lane_sets") or {}
    dgx_root = str((binding.get("deployment_roots") or {}).get("dgx", ""))
    map_path = PurePosixPath(args.static_map_manifest_path)
    expected_map_parent = PurePosixPath(dgx_root) / "inputs"
    safe_map_path = (
        map_path.is_absolute()
        and ".." not in map_path.parts
        and map_path.name == "manifest.json"
        and expected_map_parent in map_path.parents
        and str(map_path) == args.static_map_manifest_path
    )
    checks = {
        "base_binding_pass": binding.get("status") == "PASS",
        "manifest_frozen": manifest.get("status") == "FROZEN_FOR_EXECUTION",
        "manifest_count": manifest.get("episode_count") == 30,
        "manifest_unique": isinstance(keys, list)
        and len(keys) == len(set(keys)) == 30,
        "lane_partition": isinstance(lane_sets.get("a"), list)
        and isinstance(lane_sets.get("b"), list)
        and len(lane_sets["a"]) == len(lane_sets["b"]) == 15
        and lane_sets["a"] + lane_sets["b"] == keys,
        "episode_in_lane": args.episode_key in lane_sets.get(args.lane, []),
        "execution_profile": binding.get("execution_profile") == "pilot-screen1",
        "static_map_path": bool(dgx_root) and safe_map_path,
        "static_map_sha256": re.fullmatch(
            r"[0-9a-f]{64}", args.static_map_manifest_sha256
        )
        is not None,
    }
    if not all(checks.values()):
        raise ValueError(f"paired-30 binding failed: {checks}")
    x86_root = binding["deployment_roots"]["x86"]
    binding.update(
        {
            "dataset_root": f"{x86_root}/inputs/paired30_frozen_v1",
            "dataset_sha256": manifest["dataset_sha256"],
            "episode_count": 30,
            "episode_keys": keys,
            "execution_episode_count": 1,
            "execution_episode_keys": [args.episode_key],
            "screen_episode_key": args.episode_key,
            "pair_set": "paired30",
            "source_lane": args.lane,
            "paired30_manifest": str(args.manifest),
            "paired30_manifest_sha256": sha256(args.manifest),
            "paired30_checks": checks,
            "static_map_manifest_path": args.static_map_manifest_path,
            "static_map_manifest_sha256": args.static_map_manifest_sha256,
        }
    )
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
