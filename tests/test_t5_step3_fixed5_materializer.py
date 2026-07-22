from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from scripts import materialize_t5_step3_frozen_fixed5 as materializer
from scripts import run_t5_step3_frozen_fixed5 as consumer
from slow_planner.lane_b import REV_C_VIEW_ORDER, snapshot_content_sha256


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/internnav_t5/revc_four_camera_snapshot.json"
CONTRACT_SHA = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_capture(root: Path, index: int) -> Path:
    capture_id = f"capture-{index}"
    evaluator = root / f"source-{index}" / "evaluator"
    snapshot_dir = evaluator / "revc_snapshots" / capture_id
    snapshot_dir.mkdir(parents=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    render_identity = {
        "referenceTimeDenominator": 1_000_000_000,
        "referenceTimeNumerator": (index + 1) * 1_000_000_000,
        "rendering_time": float(index + 1),
    }
    rows = []
    for view_index, (view_id, expected) in enumerate(
        zip(REV_C_VIEW_ORDER, contract["cameras"])
    ):
        image_path = snapshot_dir / f"{view_index:02d}_{view_id}.png"
        Image.new(
            "RGB", (640, 480), (index * 20, view_index * 30, 64)
        ).save(image_path, "PNG")
        rows.append(
            {
                "bytes": image_path.stat().st_size,
                "encoding": "png_rgb8",
                "frame_id": expected["frame_id"],
                "hfov_deg": 73.0,
                "identity": view_id,
                "order_index": view_index,
                "path": f"revc_snapshots/{capture_id}/{image_path.name}",
                "pitch_down_deg": 10.0,
                "position_F_M_mm": expected["position_F_M_mm"],
                "prim_path": expected["prim_path"],
                "render_identity": render_identity,
                "resolution": [640, 480],
                "sensor_name": expected["sensor_name"],
                "sha256": _sha(image_path),
                "yaw_deg": expected["yaw_deg"],
            }
        )
    raw = {
        "camera_order": list(REV_C_VIEW_ORDER),
        "cameras": rows,
        "contract_id": "internnav-t5-revc-four-camera-v1",
        "contract_sha256": CONTRACT_SHA,
        "cuvslam_stereo_is_separate": True,
        "episode_id": f"b::episode-{index}",
        "external_preview_max_hz": 1.0,
        "render_barrier_id": f"barrier-{index}",
        "render_identity_source": "replicator_annotator:ReferenceTime+SimulationManager",
        "render_identity_value": render_identity,
        "request_id": f"b::revc-test::{index}",
        "reset_generation": 0,
        "same_render_tick": True,
        "schema_version": 1,
        "scope": "completion_sim_only",
        "sequence_id": 0,
        "sim_stamp_after_ns": (index + 1) * 1_000_000_000,
        "sim_stamp_before_ns": (index + 1) * 1_000_000_000,
        "wall_time_unix": 1000.0 + index,
    }
    sidecar = snapshot_dir / "snapshot.json"
    sidecar.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return sidecar


def _case(index: int, sidecar: Path, context_dir: Path) -> dict[str, object]:
    return {
        "case_id": f"case-{index}",
        "episode_key": f"episode-key-{index}",
        "snapshot_id": f"b::episode-{index}::0::0",
        "source_snapshot_sidecar": str(sidecar.relative_to(context_dir)).replace(
            "\\", "/"
        ),
        "source_snapshot_sidecar_sha256": _sha(sidecar),
        "instruction": f"Find doorway {index}.",
        "candidate_frontiers": [
            {
                "frontier_id": index,
                "relative_xz": [0.1 * index, 1.0],
                "distance_m": 1.0,
                "bearing_deg": float(index),
            }
        ],
        "agent_pose": [float(index), 0.0, 0.0, 0.0],
        "visited_frontiers": [index] if index else [],
        "compact_history": [f"history-{index}"],
    }


def _contexts(tmp_path: Path) -> tuple[Path, list[Path]]:
    sidecars = [_raw_capture(tmp_path, index) for index in range(5)]
    contexts = {
        "schema_version": 1,
        "kind": materializer.CONTEXT_KIND,
        "frozen": True,
        "evaluation_scope": "interface_screening_only",
        "candidate_frontier_source":
            "deterministic_frozen_replay_triplet_not_live_nav2",
        "agent_pose_encoding":
            "dataset_start_position_xyz_plus_rotation_xyzw",
        "revc_contract_sha256": CONTRACT_SHA,
        "cases": [_case(index, sidecar, tmp_path) for index, sidecar in enumerate(sidecars)],
    }
    path = tmp_path / "contexts.json"
    path.write_text(json.dumps(contexts, indent=2) + "\n", encoding="utf-8")
    return path, sidecars


def test_materializes_real_fixed5_for_existing_consumer(tmp_path: Path) -> None:
    contexts_path, _ = _contexts(tmp_path)
    output = tmp_path / "bundle"
    receipt = materializer.materialize_bundle(
        contexts_path=contexts_path,
        revc_contract_path=CONTRACT,
        output_dir=output,
    )
    assert receipt["status"] == "PASS"
    assert receipt["case_count"] == 5
    assert receipt["jpeg_count"] == 20
    assert receipt["manifest_sha256"] == _sha(output / "manifest.json")

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version",
        "kind",
        "frozen",
        "evaluation_scope",
        "candidate_frontier_source",
        "agent_pose_encoding",
        "episode_count",
        "snapshot_sidecar",
        "snapshot_sidecar_sha256",
        "camera_root",
        "expected_config_sha256",
        "expected_extrinsic_sha256",
        "cases",
    }
    assert manifest["cases"][3]["instruction"] == "Find doorway 3."
    assert manifest["cases"][3]["candidate_frontiers"][0]["frontier_id"] == 3
    assert manifest["cases"][3]["agent_pose"] == [3.0, 0.0, 0.0, 0.0]
    assert manifest["cases"][3]["visited_frontiers"] == [3]
    assert manifest["cases"][3]["compact_history"] == ["history-3"]
    assert manifest["evaluation_scope"] == "interface_screening_only"

    (_, cases, sidecar_path, camera_root, config_sha, extrinsics) = consumer._load_manifest(
        output / "manifest.json", receipt["manifest_sha256"]
    )
    snapshots = consumer._load_sidecar(sidecar_path)
    assert len(cases) == len(snapshots) == 5
    assert config_sha == CONTRACT_SHA
    for case in cases:
        snapshot_id = case["snapshot_id"]
        record = snapshots[snapshot_id]
        assert snapshot_content_sha256(record) == record["snapshot_content_sha256"]
        snapshot = consumer._snapshot_from_record(
            record,
            camera_root=camera_root,
            expected_config_sha256=config_sha,
            expected_extrinsic_sha256=extrinsics,
        )
        assert len(snapshot.frames) == 4
        for frame in snapshot.frames:
            assert frame.image.width == 640
            assert frame.image.height == 480
            assert frame.image.jpeg.startswith(b"\xff\xd8")

    first = snapshots["b::episode-0::0::0"]
    expected_pose = [0.17, 0.051962, 0.2, math.radians(60.0), math.radians(-10.0)]
    assert first["snapshot_sim_stamp_s"] == pytest.approx(1.0)
    assert first["images"][0]["sim_stamp_s"] == pytest.approx(1.0)
    assert first["images"][0]["pose"] == pytest.approx(expected_pose)
    assert len({row["snapshot_content_sha256"] for row in snapshots.values()}) == 5


def test_fails_closed_when_one_real_snapshot_is_reused_as_five(tmp_path: Path) -> None:
    contexts_path, sidecars = _contexts(tmp_path)
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    repeated = str(sidecars[0].relative_to(tmp_path)).replace("\\", "/")
    repeated_sha = _sha(sidecars[0])
    for case in contexts["cases"]:
        case["source_snapshot_sidecar"] = repeated
        case["source_snapshot_sidecar_sha256"] = repeated_sha
    contexts_path.write_text(json.dumps(contexts), encoding="utf-8")
    with pytest.raises(materializer.MaterializationError, match="five distinct real"):
        materializer.materialize_bundle(
            contexts_path=contexts_path,
            revc_contract_path=CONTRACT,
            output_dir=tmp_path / "forbidden-bundle",
        )
    assert not (tmp_path / "forbidden-bundle").exists()


def test_fails_closed_on_missing_case_or_source_png_drift(tmp_path: Path) -> None:
    contexts_path, sidecars = _contexts(tmp_path)
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    contexts["cases"] = contexts["cases"][:1]
    contexts_path.write_text(json.dumps(contexts), encoding="utf-8")
    with pytest.raises(materializer.MaterializationError, match="exactly five"):
        materializer.materialize_bundle(
            contexts_path=contexts_path,
            revc_contract_path=CONTRACT,
            output_dir=tmp_path / "one-only",
        )

    contexts_path, sidecars = _contexts(tmp_path / "tamper")
    raw = json.loads(sidecars[2].read_text(encoding="utf-8"))
    image = sidecars[2].parent.parent.parent / raw["cameras"][0]["path"]
    image.write_bytes(b"not the frozen PNG")
    with pytest.raises(materializer.MaterializationError, match="source PNG SHA-256"):
        materializer.materialize_bundle(
            contexts_path=contexts_path,
            revc_contract_path=CONTRACT,
            output_dir=tmp_path / "tampered-bundle",
        )
    assert not (tmp_path / "tampered-bundle").exists()


def test_cli_help_documents_fresh_fail_closed_bundle() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/materialize_t5_step3_frozen_fixed5.py"),
            "--help",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0
    assert "--contexts" in result.stdout
    assert "--revc-contract" in result.stdout
    assert "--output" in result.stdout
    assert "one capture cannot be duplicated" in result.stdout
