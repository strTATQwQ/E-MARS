from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import prepare_t5_step3_fixed5_contexts as prepare


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/internnav_t5/revc_four_camera_snapshot.json"
CONTRACT_SHA = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    result = tmp_path / "fast-lane-b-fixed5-fixture"
    evaluator = result / "remote/x86/evaluator"
    raw_keys = ["433_121", "5203_1339", "2776_676", "1036_259", "2617_628"]
    ordered_keys = list(reversed(raw_keys))
    ordered_ids = [key.rsplit("_", 1)[-1] for key in ordered_keys]
    episodes = []
    for index, key in enumerate(raw_keys):
        trajectory, episode_id = key.rsplit("_", 1)
        episodes.append(
            {
                "trajectory_id": trajectory,
                "episode_id": episode_id,
                "instruction": {"instruction_text": f"Navigate to target {episode_id}."},
                "start_position": [float(index), 0.1, -float(index)],
                "start_rotation": [0.0, 0.0, 0.0, 1.0],
            }
        )
    dataset = tmp_path / "dataset/val_unseen.json.gz"
    dataset.parent.mkdir(parents=True)
    with gzip.open(dataset, "wt", encoding="utf-8") as stream:
        json.dump({"episodes": episodes}, stream)
    dataset_sha = _sha(dataset)

    captures = []
    for index, episode_id in enumerate(ordered_ids):
        capture_id = f"capture-{index}"
        sidecar = evaluator / "revc_snapshots" / capture_id / "snapshot.json"
        _write(
            sidecar,
            {
                "episode_id": f"b::{episode_id}",
                "reset_generation": index,
                "sequence_id": index + 10,
            },
        )
        captures.append(
            {
                "capture_index": index,
                "ordered_episode_id": episode_id,
                "sidecar": f"revc_snapshots/{capture_id}/snapshot.json",
                "sidecar_sha256": _sha(sidecar),
                "request": {
                    "episode_id": f"b::{episode_id}",
                    "reset_generation": index,
                    "sequence_id": index + 10,
                },
            }
        )

    _write(
        result / "input_binding.json",
        {
            "status": "PASS",
            "lane": "b",
            "execution_profile": "fixed5",
            "isaac_sensor_profile": "lane_b_revc_fixed5_capture",
            "execution_episode_count": 5,
            "dataset_sha256": dataset_sha,
        },
    )
    _write(result / "fast_lane_summary.json", {"status": "PASS"})
    _write(
        result / "remote/x86/ordered_episode_manifest.json",
        {
            "status": "PASS",
            "dataset_sha256": dataset_sha,
            "ordered_episode_keys": ordered_keys,
            "ordered_episode_ids": ordered_ids,
        },
    )
    _write(
        evaluator / "revc_fixed5_capture.json",
        {
            "status": "PASS",
            "profile": "lane_b_revc_fixed5_capture",
            "lane": "b",
            "capture_count": 5,
            "same_render_tick_all": True,
            "contract_sha256": CONTRACT_SHA,
            "ordered_episode_ids": ordered_ids,
            "snapshots": captures,
        },
    )
    _write(
        result / "audits/revc_fixed5_scoped_pull.json",
        {"status": "PASS", "snapshot_count": 5, "png_count": 20},
    )
    return result, dataset


def test_prepares_natural_order_real_capture_contexts(tmp_path: Path) -> None:
    result, dataset = _fixture(tmp_path)
    output = result / "analysis/step3_fixed5_capture_contexts.json"
    receipt = prepare.prepare_contexts(
        result_root=result, dataset_file=dataset, output_path=output
    )
    assert receipt["status"] == "PASS"
    assert receipt["case_count"] == 5
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["evaluation_scope"] == "interface_screening_only"
    assert value["candidate_frontier_source"].endswith("not_live_nav2")
    assert [case["episode_key"] for case in value["cases"]] == [
        "2617_628",
        "1036_259",
        "2776_676",
        "5203_1339",
        "433_121",
    ]
    first = value["cases"][0]
    assert first["snapshot_id"] == "b::628::0::10"
    assert first["instruction"] == "Navigate to target 628."
    assert first["agent_pose"] == [4.0, 0.1, -4.0, 0.0, 0.0, 0.0, 1.0]
    assert [item["frontier_id"] for item in first["candidate_frontiers"]] == [0, 1, 2]
    assert (output.parent / first["source_snapshot_sidecar"]).resolve().is_file()


def test_rejects_dataset_or_capture_identity_drift(tmp_path: Path) -> None:
    result, dataset = _fixture(tmp_path)
    order = result / "remote/x86/ordered_episode_manifest.json"
    value = json.loads(order.read_text(encoding="utf-8"))
    value["dataset_sha256"] = "0" * 64
    _write(order, value)
    with pytest.raises(prepare.ContextPreparationError, match="dataset SHA"):
        prepare.prepare_contexts(
            result_root=result,
            dataset_file=dataset,
            output_path=result / "analysis/rejected.json",
        )

    result, dataset = _fixture(tmp_path / "identity")
    capture = result / "remote/x86/evaluator/revc_fixed5_capture.json"
    value = json.loads(capture.read_text(encoding="utf-8"))
    value["snapshots"][2]["request"]["episode_id"] = "b::wrong"
    _write(capture, value)
    with pytest.raises(prepare.ContextPreparationError, match="episode identity"):
        prepare.prepare_contexts(
            result_root=result,
            dataset_file=dataset,
            output_path=result / "analysis/rejected.json",
        )


def test_cli_help_names_capture_dataset_and_fresh_output() -> None:
    assert "strict=True" not in (
        ROOT / "scripts/prepare_t5_step3_fixed5_contexts.py"
    ).read_text(encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_t5_step3_fixed5_contexts.py"),
            "--help",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0
    assert "--result-root" in result.stdout
    assert "--dataset-file" in result.stdout
    assert "--output" in result.stdout
    assert "interface screening evidence" in result.stdout
