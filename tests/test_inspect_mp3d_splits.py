from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.inspect_mp3d import generate_manifests


def write_dataset(path: Path, scenes: list[str], episodes_per_scene: int) -> None:
    rows = []
    episode_id = 0
    for scene in scenes:
        for _ in range(episodes_per_scene):
            episode_id += 1
            rows.append(
                {
                    "episode_id": episode_id,
                    "trajectory_id": episode_id,
                    "scene_id": f"mp3d/{scene}/{scene}.glb",
                    "start_position": [0, 0, 0],
                    "start_rotation": [0, 0, 0, 1],
                    "goals": [{"position": [1, 0, 0], "radius": 2}],
                    "instruction": {"instruction_text": "go"},
                    "reference_path": [[0, 0, 0], [1, 0, 0]],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": rows}, handle)


def test_generate_manifests_is_scene_disjoint_and_balanced(tmp_path: Path) -> None:
    train = [f"train_{index:02d}" for index in range(20)]
    val_seen = train[3:15]
    formal = [f"formal_{index:02d}" for index in range(11)]
    r2r = tmp_path / "r2r"
    write_dataset(r2r / "train/train.json.gz", train, 8)
    write_dataset(r2r / "val_seen/val_seen.json.gz", val_seen, 8)
    write_dataset(r2r / "val_unseen/val_unseen.json.gz", formal, 3)
    assets = tmp_path / "assets"
    connectivity = tmp_path / "connectivity"
    for scene in set(train + formal):
        mesh = assets / scene / "matterport_mesh" / "mesh"
        mesh.mkdir(parents=True)
        (mesh / "isaacsim_mesh.usd").write_bytes(b"usd")
        connectivity.mkdir(exist_ok=True)
        (connectivity / f"{scene}_connectivity.json").write_text("[]")
    output = tmp_path / "out"
    value = generate_manifests(
        asset_root=assets,
        r2r_root=r2r,
        connectivity_root=connectivity,
        output=output,
        seed=7,
        formal_per_scene=2,
    )
    splits = value["audit"]["scene_splits"]
    groups = [set(rows) for rows in splits.values()]
    assert all(not left & right for index, left in enumerate(groups) for right in groups[index + 1 :])
    formal_rows = json.loads((output / "formal_episodes.json").read_text())
    assert len(formal_rows) == 22
    assert len({row["scene_key"] for row in formal_rows}) == 11
    assert (output / "manifest_lock.json").is_file()
