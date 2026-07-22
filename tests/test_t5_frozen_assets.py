from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate_t5_frozen_assets import (
    static_map_audit,
    validate_checkpoint_files,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_t5_frozen_assets.py"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _checkpoint_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    internnav_root = tmp_path / "InternNav"
    checkpoint = internnav_root / "checkpoints" / "fixture-checkpoint"
    checkpoint.mkdir(parents=True)
    files = {
        "config.json": b'{"model":"fixture"}\n',
        "model.safetensors": b"fixed-size-model-bytes",
    }
    for name, value in files.items():
        (checkpoint / name).write_bytes(value)
    linked_path = (
        internnav_root / "checkpoints" / "depth_anything_v2_metric_hypersim_vits.pth"
    )
    linked_value = b"exact-linked-depth-checkpoint"
    linked_path.write_bytes(linked_value)
    content = {
        "schema_version": 1,
        "checkpoint_name": "fixture-checkpoint",
        "checkpoint_revision": "fixture-revision",
        "top_level_regular_file_set_strict": True,
        "files": [
            {"path": name, "bytes": len(value), "sha256": _sha(value)}
            for name, value in sorted(files.items())
        ],
        "linked_assets": [
            {
                "path_from_internnav_root": linked_path.relative_to(
                    internnav_root
                ).as_posix(),
                "bytes": len(linked_value),
                "sha256": _sha(linked_value),
            }
        ],
    }
    return internnav_root, checkpoint, content


def _checkpoint_status(
    content: dict, checkpoint: Path, internnav_root: Path
) -> tuple[bool, dict[str, bool]]:
    checks, _, _ = validate_checkpoint_files(content, checkpoint, internnav_root)
    return all(checks.values()), checks


def test_checkpoint_exact_content_and_linked_depth_pass(tmp_path: Path) -> None:
    internnav_root, checkpoint, content = _checkpoint_fixture(tmp_path)

    passed, checks = _checkpoint_status(content, checkpoint, internnav_root)

    assert passed
    assert checks["strict_top_level_file_set"] is True
    assert checks["all_checkpoint_files_content_match"] is True
    assert checks["all_linked_assets_content_match"] is True


def test_same_size_checkpoint_mutation_fails_sha256(tmp_path: Path) -> None:
    internnav_root, checkpoint, content = _checkpoint_fixture(tmp_path)
    model = checkpoint / "model.safetensors"
    original = model.read_bytes()
    model.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))

    passed, checks = _checkpoint_status(content, checkpoint, internnav_root)

    assert model.stat().st_size == len(original)
    assert passed is False
    assert checks["strict_top_level_file_set"] is True
    assert checks["all_checkpoint_files_content_match"] is False


def test_linked_depth_requires_exact_sha256_not_only_size(tmp_path: Path) -> None:
    internnav_root, checkpoint, content = _checkpoint_fixture(tmp_path)
    linked = (
        internnav_root / "checkpoints" / "depth_anything_v2_metric_hypersim_vits.pth"
    )
    original = linked.read_bytes()
    linked.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    passed, checks = _checkpoint_status(content, checkpoint, internnav_root)

    assert linked.stat().st_size == len(original)
    assert passed is False
    assert checks["all_linked_assets_content_match"] is False


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_checkpoint_strict_set_rejects_missing_and_extra_files(
    tmp_path: Path, mutation: str
) -> None:
    internnav_root, checkpoint, content = _checkpoint_fixture(tmp_path)
    if mutation == "missing":
        (checkpoint / "config.json").unlink()
    else:
        (checkpoint / "unlisted.bin").write_bytes(b"extra")

    passed, checks = _checkpoint_status(content, checkpoint, internnav_root)

    assert passed is False
    assert checks["strict_top_level_file_set"] is False


def test_checkpoint_manifest_rejects_duplicate_and_traversal_paths(
    tmp_path: Path,
) -> None:
    internnav_root, checkpoint, content = _checkpoint_fixture(tmp_path)
    duplicate = copy.deepcopy(content)
    duplicate["files"].append(copy.deepcopy(duplicate["files"][0]))
    passed, checks = _checkpoint_status(duplicate, checkpoint, internnav_root)
    assert passed is False
    assert checks["unique_checkpoint_file_paths"] is False

    duplicate_linked = copy.deepcopy(content)
    duplicate_linked["linked_assets"].append(
        copy.deepcopy(duplicate_linked["linked_assets"][0])
    )
    passed, checks = _checkpoint_status(duplicate_linked, checkpoint, internnav_root)
    assert passed is False
    assert checks["unique_linked_asset_paths"] is False

    traversal = copy.deepcopy(content)
    traversal["files"][0]["path"] = "../config.json"
    passed, checks = _checkpoint_status(traversal, checkpoint, internnav_root)
    assert passed is False
    assert checks["safe_top_level_file_names"] is False

    linked_traversal = copy.deepcopy(content)
    linked_traversal["linked_assets"][0][
        "path_from_internnav_root"
    ] = "../outside-depth.pth"
    (tmp_path / "outside-depth.pth").write_bytes(
        b"exact-linked-depth-checkpoint"
    )
    passed, checks = _checkpoint_status(linked_traversal, checkpoint, internnav_root)
    assert passed is False
    assert checks["safe_linked_asset_paths"] is False
    assert checks["all_linked_assets_content_match"] is False


def _static_map_fixture() -> tuple[dict, dict]:
    digests = {"scene-a": _sha(b"scene-a-ply"), "scene-b": _sha(b"scene-b-ply")}
    golden = {
        "scenes": {
            "d0_selected_scene_geometry": [
                {"scene": scene, "ply_sha256": digest}
                for scene, digest in digests.items()
            ]
        }
    }
    manifest = {
        "maps": {
            "scene-a_+0.0000": {
                "key": "scene-a_+0.0000",
                "scan": "scene-a",
                "source_ply": "scene-a/house_segmentations/scene-a.ply",
                "source_ply_sha256": digests["scene-a"],
            },
            "scene-b_+0.0000": {
                "key": "scene-b_+0.0000",
                "scan": "scene-b",
                "source_ply": "scene-b/house_segmentations/scene-b.ply",
                "source_ply_sha256": digests["scene-b"],
            },
        }
    }
    return golden, manifest


def test_static_map_requires_exact_scene_set_and_ply_hashes() -> None:
    golden, manifest = _static_map_fixture()

    assert static_map_audit(golden, manifest)["status"] == "PASS"

    tampered = copy.deepcopy(manifest)
    tampered["maps"]["scene-a_+0.0000"]["source_ply_sha256"] = _sha(
        b"same-scene-mutated-geometry"
    )
    audit = static_map_audit(golden, tampered)
    assert audit["status"] == "FAIL"
    assert audit["checks"]["exact_scene_set"] is True
    assert audit["checks"]["all_source_ply_sha256_match"] is False


@pytest.mark.parametrize("mutation", ["missing", "extra", "inconsistent", "traversal"])
def test_static_map_rejects_scene_drift_and_unsafe_paths(mutation: str) -> None:
    golden, manifest = _static_map_fixture()
    candidate = copy.deepcopy(manifest)
    if mutation == "missing":
        candidate["maps"].pop("scene-b_+0.0000")
    elif mutation == "extra":
        candidate["maps"]["scene-c_+0.0000"] = {
            "key": "scene-c_+0.0000",
            "scan": "scene-c",
            "source_ply": "scene-c/house_segmentations/scene-c.ply",
            "source_ply_sha256": _sha(b"scene-c-ply"),
        }
    elif mutation == "inconsistent":
        candidate["maps"]["scene-a_+1.0000"] = {
            "key": "scene-a_+1.0000",
            "scan": "scene-a",
            "source_ply": "scene-a/house_segmentations/scene-a.ply",
            "source_ply_sha256": _sha(b"different-scene-a-ply"),
        }
    else:
        candidate["maps"]["scene-a_+0.0000"]["source_ply"] = "../scene-a.ply"

    audit = static_map_audit(golden, candidate)

    assert audit["status"] == "FAIL"
    if mutation == "inconsistent":
        assert audit["checks"]["internally_consistent_scene_digests"] is False
    if mutation == "traversal":
        assert audit["checks"]["source_ply_paths_safe_and_canonical"] is False


def test_static_map_rejects_duplicate_golden_scene_entries() -> None:
    golden, manifest = _static_map_fixture()
    golden["scenes"]["d0_selected_scene_geometry"].append(
        copy.deepcopy(golden["scenes"]["d0_selected_scene_geometry"][0])
    )

    audit = static_map_audit(golden, manifest)

    assert audit["status"] == "FAIL"
    assert audit["checks"]["unique_golden_scenes"] is False


def test_cli_writes_fail_json_and_nonzero_for_invalid_or_duplicate_json(
    tmp_path: Path,
) -> None:
    golden, _ = _static_map_fixture()
    golden_path = tmp_path / "golden.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "audit.json"
    golden_path.write_text(json.dumps(golden), encoding="utf-8")
    manifest_path.write_text('{"maps": {}, "maps": {}}', encoding="utf-8")
    output.write_text('{"status":"PASS"}\n', encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "static-map",
            "--golden",
            str(golden_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    audit = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 1
    assert audit["status"] == "FAIL"
    assert audit["checks"] == {"input_and_audit_valid": False}
    assert audit["error_type"] == "ValueError"


def test_static_map_cli_passes_valid_fixture(tmp_path: Path) -> None:
    golden, manifest = _static_map_fixture()
    golden_path = tmp_path / "golden.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "audit.json"
    golden_path.write_text(json.dumps(golden), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "static-map",
            "--golden",
            str(golden_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
