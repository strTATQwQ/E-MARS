from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_t5_episode_order.py"
SPEC = importlib.util.spec_from_file_location("materialize_t5_episode_order", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture_from_frozen_manifest(tmp_path: Path, name: str):
    manifest = json.loads(
        (ROOT / "configs" / "internnav_t3" / name).read_text(encoding="utf-8")
    )
    raw_keys = manifest["episode_keys"]
    scene_by_key = {
        item["episode_key"]: item["scene"]
        for item in manifest["selected_clearance_evidence"]
    }
    assert set(scene_by_key) == set(raw_keys)
    episodes = []
    by_key = {}
    for key in raw_keys:
        trajectory_id, episode_id = key.split("_", 1)
        episode = {
            "trajectory_id": trajectory_id,
            "episode_id": episode_id,
            "scene_id": "mp3d/%s/%s.glb" % (scene_by_key[key], scene_by_key[key]),
        }
        episodes.append(episode)
        by_key[key] = episode
    dataset = tmp_path / name / "val_unseen.json.gz"
    dataset.parent.mkdir(parents=True)
    with gzip.open(dataset, "wt", encoding="utf-8") as stream:
        json.dump({"episodes": episodes}, stream)
    materialized = [
        key
        for scene in sorted(set(scene_by_key.values()))
        for key in raw_keys
        if scene_by_key[key] == scene
    ]
    path_key_data = {key: by_key[key] for key in materialized}
    return dataset, path_key_data


def test_frozen_five_materializes_actual_fresh_evaluator_order(tmp_path: Path) -> None:
    dataset, path_key_data = _fixture_from_frozen_manifest(
        tmp_path, "model_canary_episode_manifest.json"
    )
    payload = MODULE._payload_from_path_key_data(
        dataset, 5, path_key_data, {"fixture": "frozen-five"}
    )
    assert payload["ordered_episode_keys"] == [
        "2617_628",
        "1036_259",
        "2776_676",
        "5203_1339",
        "433_121",
    ]
    assert payload["ordered_episode_ids"] == ["628", "259", "676", "1339", "121"]


def test_frozen_twenty_uses_scene_materialization_not_reverse_gzip(
    tmp_path: Path,
) -> None:
    dataset, path_key_data = _fixture_from_frozen_manifest(
        tmp_path, "model_pilot_episode_manifest.json"
    )
    payload = MODULE._payload_from_path_key_data(
        dataset, 20, path_key_data, {"fixture": "frozen-twenty"}
    )
    assert payload["ordered_episode_ids"] == [
        "1765",
        "145",
        "1720",
        "1474",
        "703",
        "610",
        "1027",
        "625",
        "1657",
        "1417",
        "1561",
        "877",
        "976",
        "1003",
        "1741",
        "1255",
        "157",
        "448",
        "364",
        "1573",
    ]
    assert payload["ordered_episode_ids"] != list(
        reversed(
            [key.split("_", 1)[1] for key in payload["raw_episode_keys"]]
        )
    )


def test_materialized_order_rejects_filtered_or_missing_episode(tmp_path: Path) -> None:
    dataset, path_key_data = _fixture_from_frozen_manifest(
        tmp_path, "model_canary_episode_manifest.json"
    )
    path_key_data.pop(next(iter(path_key_data)))
    with pytest.raises(RuntimeError, match="exact upstream path-key"):
        MODULE._payload_from_path_key_data(
            dataset, 5, path_key_data, {"fixture": "missing"}
        )
