#!/usr/bin/env python3
"""Freeze disjoint Gate1/canary/pilot subsets from official val_unseen."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

SOURCE_REVISION = "7b05993b21813c3787f2f7f604bfc22b80c48c8e"
SOURCE_SHA256 = "853673f8faadb26f883d57d71828cf895345c89fa594a7b918a78862318aeb31"
PHASE_SLICES = {"gate1": (0, 1), "canary": (1, 6), "pilot": (6, 26)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(item: dict) -> str:
    return f"{item['trajectory_id']}_{item['episode_id']}"


def load_official_dataset_utils(internnav_root: Path):
    """Load the frozen utility without triggering simulator/server imports."""
    for package in ("internnav", "internnav.utils"):
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module
    common_log = types.ModuleType("internnav.utils.common_log_util")
    common_log.common_logger = logging.getLogger("internnav_t0_freeze")
    sys.modules[common_log.__name__] = common_log
    path = (
        internnav_root
        / "internnav"
        / "env"
        / "utils"
        / "episode_loader"
        / "dataset_utils.py"
    )
    spec = importlib.util.spec_from_file_location("internnav_t0_dataset_utils", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official dataset utility: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_data, module.skip_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internnav-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    load_data, skip_list = load_official_dataset_utils(args.internnav_root)

    source_file = args.source_root / "val_unseen" / "val_unseen.json.gz"
    if source_file.stat().st_size != 448607 or sha256(source_file) != SOURCE_SHA256:
        raise SystemExit("source val_unseen does not match the pinned official payload")
    with gzip.open(source_file, "rt", encoding="utf-8") as stream:
        source = json.load(stream)
    originals = {key(item): item for item in source["episodes"]}
    if len(originals) != len(source["episodes"]):
        raise SystemExit("source contains duplicate trajectory/episode keys")

    filtered = load_data(
        str(args.source_root),
        "val_unseen",
        filter_same_trajectory=True,
        filter_stairs=True,
        dataset_type="mp3d",
        rank=0,
        world_size=1,
    )
    # Deterministic round-robin over sorted official scenes gives scene diversity
    # without changing any task/filter semantics.
    selected = []
    eligible_by_scene = {
        scan: [item for item in filtered[scan] if item["trajectory_id"] not in skip_list]
        for scan in sorted(filtered)
    }
    for scene_index in range(max(map(len, eligible_by_scene.values()))):
        for scan in sorted(eligible_by_scene):
            if scene_index < len(eligible_by_scene[scan]):
                selected.append(originals[key(eligible_by_scene[scan][scene_index])])
            if len(selected) == PHASE_SLICES["pilot"][1]:
                break
        if len(selected) == PHASE_SLICES["pilot"][1]:
            break
    if len(selected) != PHASE_SLICES["pilot"][1]:
        raise SystemExit(
            f"only {len(selected)} eligible episodes remain across {len(filtered)} scenes; "
            "cannot freeze 26 episodes"
        )

    canary_keys = {key(item) for item in selected[slice(*PHASE_SLICES["canary"])]}
    for phase, bounds in PHASE_SLICES.items():
        episodes = selected[slice(*bounds)]
        phase_root = args.output_root / phase
        phase_file = phase_root / "val_unseen" / "val_unseen.json.gz"
        phase_file.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(
            {"episodes": episodes},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        phase_file.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))

        reloaded = load_data(
            str(phase_root),
            "val_unseen",
            filter_same_trajectory=True,
            filter_stairs=True,
            dataset_type="mp3d",
            rank=0,
            world_size=1,
        )
        phase_keys = [key(item) for item in episodes]
        post_filter_count = sum(len(items) for items in reloaded.values())
        if post_filter_count != len(episodes):
            raise SystemExit(f"{phase} lost episodes under the official loader")
        manifest = {
            "schema_version": 1,
            "phase": phase,
            "source_repo": "InternRobotics/InternData-N1",
            "source_revision": SOURCE_REVISION,
            "source_sha256": SOURCE_SHA256,
            "selection": "round-robin eligible episodes over sorted scenes; disjoint slices 1/5/20",
            "episode_count": len(episodes),
            "post_official_filter_episode_count": post_filter_count,
            "episode_keys": phase_keys,
            "scene_ids": [item["scene_id"] for item in episodes],
            "overlay_sha256": sha256(phase_file),
            "overlap_with_canary": sorted(set(phase_keys) & canary_keys) if phase == "pilot" else None,
        }
        (phase_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"PASS {phase} episodes={len(episodes)} "
            f"overlay_sha256={manifest['overlay_sha256']}"
        )


if __name__ == "__main__":
    main()
