from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from scripts.prepare_t5_lane_b_step3 import (
    MODEL_REVISION,
    TRANSFORMERS_VERSION,
    build_content_manifest,
    build_lan_rsync_command,
    compare_manifests,
    validate_content_manifest,
    verify_clean_bf16_load,
)


def materialize_fake_model(model, *, revision=MODEL_REVISION, metadata_text=None):
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "weights.bin").write_bytes(b"weights")
    metadata_dir = model / ".cache/huggingface/download"
    metadata_dir.mkdir(parents=True)
    content = metadata_text if metadata_text is not None else f"{revision}\netag\n0\n"
    (metadata_dir / "weights.bin.metadata").write_text(content, encoding="utf-8")


def test_artifact_manifest_is_deterministic_and_pins_runtime(tmp_path) -> None:
    model = tmp_path / "model"
    materialize_fake_model(model)
    first = build_content_manifest(model)
    second = build_content_manifest(model)
    assert first == second
    assert first["revision"] == MODEL_REVISION
    assert first["transformers_version"] == TRANSFORMERS_VERSION
    assert first["file_count"] == 3
    assert first["hf_metadata_provenance"]["metadata_count"] == 1
    assert first["hf_metadata_provenance"]["revision"] == MODEL_REVISION
    assert validate_content_manifest(first) == first


def test_manifest_comparison_fails_closed_on_changed_artifact(tmp_path) -> None:
    model = tmp_path / "model"
    materialize_fake_model(model)
    weights = model / "weights.bin"
    weights.write_bytes(b"before")
    expected = build_content_manifest(model)
    weights.write_bytes(b"after")
    actual = build_content_manifest(model)
    with pytest.raises(RuntimeError, match="differs from source manifest"):
        compare_manifests(expected, actual)


def test_manifest_rejects_revision_tampering(tmp_path) -> None:
    model = tmp_path / "model"
    materialize_fake_model(model)
    manifest = build_content_manifest(model)
    manifest["revision"] = "0" * 40
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_content_manifest(manifest)


def test_manifest_rejects_wrong_hugging_face_metadata_revision(tmp_path) -> None:
    model = tmp_path / "model"
    materialize_fake_model(model, revision="0" * 40)
    with pytest.raises(ValueError, match="WRONG_REVISION"):
        build_content_manifest(model)


@pytest.mark.parametrize("metadata_text", ["", "\n"])
def test_manifest_rejects_empty_hugging_face_metadata(tmp_path, metadata_text) -> None:
    model = tmp_path / "model"
    materialize_fake_model(model, metadata_text=metadata_text)
    with pytest.raises(ValueError, match="EMPTY_HF_METADATA"):
        build_content_manifest(model)


def test_manifest_rejects_missing_hugging_face_metadata(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    with pytest.raises(ValueError, match="MISSING_HF_METADATA"):
        build_content_manifest(model)


def test_lan_copy_command_uses_batch_ssh_and_contains_no_credentials(tmp_path) -> None:
    command = build_lan_rsync_command(
        source_user="railgun",
        source_host="10.100.100.128",
        source_path="/home/railgun/ai-stack/models/Step3-VL-10B",
        destination=tmp_path / "model",
    )
    rendered = json.dumps(command)
    assert command[0] == "rsync"
    assert "--copy-links" in command
    assert "BatchMode=yes" in rendered
    assert "StrictHostKeyChecking=yes" in rendered
    assert "password" not in rendered.lower()
    assert "token" not in rendered.lower()


def test_lan_copy_rejects_public_or_shell_unsafe_source(tmp_path) -> None:
    with pytest.raises(ValueError, match="private LAN"):
        build_lan_rsync_command(
            source_user="railgun",
            source_host="8.8.8.8",
            source_path="/model",
            destination=tmp_path,
        )
    with pytest.raises(ValueError, match="unsafe"):
        build_lan_rsync_command(
            source_user="railgun;bad",
            source_host="10.100.100.128",
            source_path="/model",
            destination=tmp_path,
        )
    with pytest.raises(ValueError, match="shell-safe"):
        build_lan_rsync_command(
            source_user="railgun",
            source_host="10.100.100.128",
            source_path="/model;touch-bad",
            destination=tmp_path,
        )


def test_clean_load_verifier_requires_frozen_revision_runtime_and_bf16(
    monkeypatch, tmp_path
) -> None:
    calls = {}

    class Planner:
        def health(self):
            return {
                "checkpoint_load_clean": True,
                "parameter_count": 10_171_750_144,
                "parameter_dtype_counts": {"torch.bfloat16": 10_171_750_144},
            }

        def close(self):
            calls["closed"] = True

    def fake_from_pretrained(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return Planner()

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(__version__=TRANSFORMERS_VERSION)
    )
    from slow_planner.step3_vl_10b import Step3VLSlowPlanner

    monkeypatch.setattr(Step3VLSlowPlanner, "from_pretrained", fake_from_pretrained)
    health = verify_clean_bf16_load(tmp_path)
    assert health["checkpoint_load_clean"] is True
    assert calls["kwargs"]["expected_transformers_version"] == TRANSFORMERS_VERSION
    assert calls["kwargs"]["require_clean_checkpoint_load"] is True
    assert calls["kwargs"]["require_all_parameters_bf16"] is True
    assert calls["closed"] is True
