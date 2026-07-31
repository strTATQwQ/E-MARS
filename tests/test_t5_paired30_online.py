from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/internnav_t5/paired30_episode_manifest.json"
MATERIALIZER = ROOT / "scripts/materialize_t5_frozen_subset.py"
BINDER = ROOT / "scripts/bind_t5_paired30_screen.py"
COORDINATOR = ROOT / "coordination/run_t5_paired30_online.sh"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_frozen_paired30_contract_is_balanced_and_bounded() -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    keys = value["episode_keys"]
    assert value["status"] == "FROZEN_FOR_EXECUTION"
    assert len(keys) == len(set(keys)) == value["episode_count"] == 30
    assert value["lane_sets"]["a"] + value["lane_sets"]["b"] == keys
    assert len(value["lane_sets"]["a"]) == len(value["lane_sets"]["b"]) == 15
    assert value["per_execution_limits"] == {
        "wall_seconds_after_ready": 1800,
        "configured_max_step": 8000,
        "observed_evaluator_step_limit": 8100,
        "termination": "earlier_of_wall_or_physics_steps",
    }
    assert len(value["scene_ids"]) == len(set(value["scene_ids"])) == 5
    assert value["selection"]["static_grid_clearance_m"] == 0.25
    assert value["selection"]["duplicate_authored_start_count"] == 3
    assert set(value["selection"]["static_grid_corrections"]) == {
        "668_166",
        "1317_337",
        "3555_880",
        "4510_1129",
    }
    assert not set(value["selection"]["static_grid_corrections"]).intersection(keys)
    assert set(value["selection"]["static_grid_corrections"].values()).issubset(keys)


def test_materializer_copies_official_objects_in_frozen_order(tmp_path: Path) -> None:
    module = _module(MATERIALIZER, "paired30_materializer")
    source_root = tmp_path / "source"
    source = source_root / "val_unseen/val_unseen.json.gz"
    source.parent.mkdir(parents=True)
    episodes = [
        {"trajectory_id": index + 100, "episode_id": index, "payload": index}
        for index in range(35)
    ]
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        json.dump({"episodes": episodes}, stream)
    keys = [f"{index + 100}_{index}" for index in range(30)]
    expected_root = tmp_path / "expected"
    module.deterministic_gzip(
        expected_root / "val_unseen/val_unseen.json.gz",
        {"episodes": episodes[:30]},
    )
    expected = expected_root / "val_unseen/val_unseen.json.gz"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "FROZEN_FOR_EXECUTION",
                "episode_count": 30,
                "episode_keys": keys,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "dataset_sha256": hashlib.sha256(expected.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(MATERIALIZER),
            "--source-root",
            str(source_root),
            "--manifest",
            str(manifest),
            "--output-root",
            str(output_root),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    audit = json.loads(completed.stdout)
    assert audit["status"] == "PASS"
    assert audit["episode_keys"] == keys
    assert audit["dataset_sha256"] == hashlib.sha256(expected.read_bytes()).hexdigest()


def test_paired30_binding_reuses_deployment_but_replaces_dataset(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "status": "PASS",
                "execution_profile": "pilot-screen1",
                "deployment_roots": {
                    "dgx": "/home/railgun/deployment",
                    "x86": "/home/song/deployment",
                },
                "static_map_manifest_path": "/home/railgun/deployment/inputs/old/manifest.json",
                "static_map_manifest_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "binding.json"
    subprocess.run(
        [
            sys.executable,
            str(BINDER),
            "--input-binding",
            str(base),
            "--manifest",
            str(MANIFEST),
            "--episode-key",
            manifest["lane_sets"]["a"][3],
            "--lane",
            "a",
            "--static-map-manifest-path",
            "/home/railgun/deployment/inputs/paired30_run/static_maps/manifest.json",
            "--static-map-manifest-sha256",
            "a" * 64,
            "--output",
            str(output),
        ],
        check=True,
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["pair_set"] == "paired30"
    assert value["episode_keys"] == manifest["episode_keys"]
    assert value["execution_episode_keys"] == [manifest["lane_sets"]["a"][3]]
    assert value["dataset_root"] == "/home/song/deployment/inputs/paired30_frozen_v1"
    assert value["static_map_manifest_path"] == (
        "/home/railgun/deployment/inputs/paired30_run/static_maps/manifest.json"
    )
    assert value["static_map_manifest_sha256"] == "a" * 64
    assert value["paired30_checks"]["static_map_path"] is True


def test_fast_lane_accepts_a_fully_validated_paired30_binding() -> None:
    text = (ROOT / "coordination/run_t5_fast_lane_online.sh").read_text(
        encoding="utf-8"
    )
    assert 'profile=="pilot-screen1"' in text
    assert '(binding or {}).get("pair_set")=="paired30"' in text
    assert "all(value is True for value in paired30_checks.values())" in text
    assert "or paired30_binding" in text
    sensor_gate = (ROOT / "scripts" / "run_t4_sensor_gate.sh").read_text(
        encoding="utf-8"
    )
    assert (
        'INTERNVLA_T3_STATIC_CLEARANCE_GATE_M="${INTERNVLA_T3_STATIC_CLEARANCE_GATE_M:-0.30}"'
        in sensor_gate
    )


def test_coordinator_uses_two_balanced_rounds_and_hard_limits() -> None:
    text = COORDINATOR.read_text(encoding="utf-8")
    assert "for index in $(seq 0 14); do run_pair round1" in text
    assert "for index in $(seq 0 14); do run_pair round2" in text
    assert "INTERNNAV_T5_PILOT_MAX_STEP=8000" in text
    assert "INTERNNAV_T5_FAST_SCREEN_TIMEOUT_SEC=1800" in text
    assert "INTERNVLA_T3_STATIC_CLEARANCE_GATE_M=0.25" in text
    assert 'mkdir -p "$stage/cache" "$stage/maps"' in text
    assert 'mkdir -p "$stage/dataset"' not in text
    assert '"lane_a_result":sys.argv[9]' in text
    assert '"lane_b_arm":"internvla_step3"' in text
    assert "deadline=$((SECONDS + 60))" in text
    assert "kill -KILL" in text
