#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    episodes = value.get("episodes") if isinstance(value, dict) else None
    if not isinstance(episodes, list):
        raise ValueError(f"{path} does not contain an episodes array")
    return [dict(row) for row in episodes]


def scene_id(episode: dict[str, Any]) -> str:
    value = str(episode.get("scene_id") or "")
    parts = value.replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"invalid episode scene_id={value!r}")
    return parts[-2]


def stable_rank(seed: int, split: str, episode: dict[str, Any]) -> str:
    material = ":".join(
        [
            str(seed),
            split,
            scene_id(episode),
            str(episode.get("episode_id")),
            str(episode.get("trajectory_id")),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def choose_episodes(
    episodes: list[dict[str, Any]],
    *,
    split: str,
    scenes: set[str],
    seed: int,
    total: int | None = None,
    per_scene: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        sid = scene_id(episode)
        if sid in scenes:
            grouped[sid].append(episode)
    selected: list[dict[str, Any]] = []
    if per_scene is not None:
        for sid in sorted(scenes):
            rows = sorted(grouped[sid], key=lambda row: stable_rank(seed, split, row))
            if len(rows) < per_scene:
                raise ValueError(f"scene {sid} has only {len(rows)} episodes, need {per_scene}")
            selected.extend(rows[:per_scene])
    else:
        rows = [row for sid in sorted(grouped) for row in grouped[sid]]
        rows.sort(key=lambda row: stable_rank(seed, split, row))
        selected = rows[:total]
    output = []
    for row in selected:
        value = dict(row)
        value["source_split"] = split
        value["scene_key"] = scene_id(row)
        value["benchmark_episode_id"] = f"{split}:{scene_id(row)}:{row.get('episode_id')}"
        value["selection_seed"] = seed
        output.append(value)
    return output


def inventory_scene(asset_root: Path, sid: str, connectivity_root: Path) -> dict[str, Any]:
    root = asset_root / sid
    metric = sorted(path for path in root.rglob("isaacsim_*.usd") if "non_metric" not in path.name)
    semantic_patterns = ("*.house", "*.ply", "*.semseg.json", "*.fsegs.json")
    return {
        "scene_id": sid,
        "present": root.is_dir(),
        "metric_usd": str(metric[0]) if metric else None,
        "metric_usd_count": len(metric),
        "obj_count": len(list(root.rglob("*.obj"))) if root.is_dir() else 0,
        "mtl_count": len(list(root.rglob("*.mtl"))) if root.is_dir() else 0,
        "texture_count": sum(len(list(root.rglob(pattern))) for pattern in ("*.jpg", "*.jpeg", "*.png")) if root.is_dir() else 0,
        "semantic_count": sum(len(list(root.rglob(pattern))) for pattern in semantic_patterns) if root.is_dir() else 0,
        "connectivity_path": str(connectivity_root / f"{sid}_connectivity.json"),
        "connectivity_present": (connectivity_root / f"{sid}_connectivity.json").is_file(),
    }


def generate_manifests(
    *,
    asset_root: Path,
    r2r_root: Path,
    connectivity_root: Path,
    output: Path,
    seed: int,
    formal_per_scene: int,
) -> dict[str, Any]:
    paths = {split: r2r_root / split / f"{split}.json.gz" for split in ("train", "val_seen", "val_unseen")}
    datasets = {split: load_dataset(path) for split, path in paths.items()}
    scenes = {split: {scene_id(row) for row in rows} for split, rows in datasets.items()}
    if scenes["val_unseen"] & (scenes["train"] | scenes["val_seen"]):
        raise ValueError("formal val_unseen scenes overlap development scenes")
    if len(scenes["val_unseen"]) < 10:
        raise ValueError("formal split must contain at least 10 held-out scenes")

    train_only = sorted(scenes["train"] - scenes["val_seen"])
    conversion_debug = train_only[:3]
    if len(conversion_debug) < 3:
        conversion_debug = sorted(scenes["train"])[:3]
    prompt_validation = sorted(scenes["val_seen"] - set(conversion_debug))[:5]
    calibration = sorted(scenes["train"] - set(conversion_debug) - set(prompt_validation))[:10]
    formal = sorted(scenes["val_unseen"])
    named_splits = {
        "conversion_debug_scenes": conversion_debug,
        "prompt_validation_scenes": prompt_validation,
        "nvfp4_calibration_scenes": calibration,
        "formal_test_scenes": formal,
    }
    all_named = [set(value) for value in named_splits.values()]
    for index, left in enumerate(all_named):
        for right in all_named[index + 1 :]:
            if left & right:
                raise ValueError(f"named scene-level splits overlap: {sorted(left & right)}")

    formal_episodes = choose_episodes(
        datasets["val_unseen"],
        split="val_unseen",
        scenes=set(formal),
        seed=seed,
        per_scene=formal_per_scene,
    )
    smoke_episodes = choose_episodes(
        datasets["val_seen"],
        split="val_seen",
        scenes=set(prompt_validation),
        seed=seed + 1,
        total=5,
    )
    pilot_episodes = choose_episodes(
        datasets["val_seen"],
        split="val_seen",
        scenes=set(prompt_validation),
        seed=seed + 2,
        total=30,
    )
    calibration_episodes = choose_episodes(
        datasets["train"],
        split="train",
        scenes=set(calibration),
        seed=seed + 3,
        total=64,
    )
    inventories = [inventory_scene(asset_root, sid, connectivity_root) for sid in sorted(set().union(*all_named))]
    missing = [row["scene_id"] for row in inventories if not row["present"] or not row["metric_usd"] or not row["connectivity_present"]]
    if missing:
        raise ValueError(f"selected scenes lack metric USD or connectivity: {missing}")

    source = {
        split: {
            "path": str(path),
            "sha256": sha256_file(path),
            "episodes": len(datasets[split]),
            "scenes": len(scenes[split]),
        }
        for split, path in paths.items()
    }
    audit = {
        "schema_version": 1,
        "selection_seed": seed,
        "asset_root": str(asset_root),
        "connectivity_root": str(connectivity_root),
        "coordinate_transform": {
            "habitat_xyz_to_isaac_xyz": ["x", "-z", "y"],
            "basis_rotation": "X+90deg",
            "status": "candidate_until_real_isaac_pose_validation",
        },
        "source_datasets": source,
        "scene_splits": named_splits,
        "scene_inventory": inventories,
        "episode_counts": {
            "smoke": len(smoke_episodes),
            "pilot": len(pilot_episodes),
            "formal": len(formal_episodes),
            "nvfp4_calibration": len(calibration_episodes),
        },
    }
    write_json(output / "data_audit.json", audit)
    write_json(output / "scene_splits.json", named_splits)
    for name, rows in (
        ("smoke_episodes", smoke_episodes),
        ("pilot_episodes", pilot_episodes),
        ("formal_episodes", formal_episodes),
        ("nvfp4_calibration_episodes", calibration_episodes),
    ):
        write_json(output / f"{name}.json", rows)
        write_jsonl(output / f"{name}.jsonl", rows)
    lock_files = sorted(output.glob("*.json")) + sorted(output.glob("*.jsonl"))
    lock = {
        "schema_version": 1,
        "selection_seed": seed,
        "files": {path.name: sha256_file(path) for path in lock_files if path.name != "manifest_lock.json"},
    }
    write_json(output / "manifest_lock.json", lock)
    return {"audit": audit, "lock": lock}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit MP3D/R2R data and freeze leak-free benchmark manifests.")
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--r2r-root", required=True)
    parser.add_argument("--connectivity-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--formal-per-scene", type=int, default=10)
    args = parser.parse_args()
    value = generate_manifests(
        asset_root=Path(args.asset_root),
        r2r_root=Path(args.r2r_root),
        connectivity_root=Path(args.connectivity_root),
        output=Path(args.output),
        seed=args.seed,
        formal_per_scene=args.formal_per_scene,
    )
    print(json.dumps({"episode_counts": value["audit"]["episode_counts"], "lock": value["lock"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
