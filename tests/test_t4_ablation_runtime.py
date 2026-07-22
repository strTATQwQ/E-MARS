from __future__ import annotations

import json
from pathlib import Path

import pytest

from t4_completion.ablation.contract import (
    ContractError,
    load_matrix,
    resolve_variant_configs,
    write_json,
)
from t4_completion.ablation.runtime import (
    DEFAULT_MATRIX,
    FACTOR_ORDER,
    load_runtime_variant,
    ordered_fields,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, variant_id: str) -> Path:
    matrix = load_matrix(DEFAULT_MATRIX)
    path = tmp_path / f"{variant_id}.json"
    write_json(path, resolve_variant_configs(matrix)[variant_id])
    return path


@pytest.mark.parametrize(
    ("variant_id", "factor", "expected"),
    [
        ("history_off", "history_mode", "off"),
        ("recovery_on", "recovery_mode", "on"),
        ("h1_view", "view_mode", "h1_view"),
        ("oracle_termination", "termination_mode", "oracle_termination"),
        ("endpoint", "trajectory_mode", "endpoint"),
        ("oracle_high_level_system1", "system_mode", "oracle_high_level_system1"),
    ],
)
def test_runtime_config_exposes_the_exact_frozen_factor(
    tmp_path: Path, variant_id: str, factor: str, expected: str
) -> None:
    summary = load_runtime_variant(_config(tmp_path, variant_id))
    assert summary["status"] == "RUNTIME_CONFIG_VALID"
    assert summary["runtime_policy"] == "completion_sim"
    assert summary["runtime_target"] == "isaac_simulation"
    assert summary["factors"][factor] == expected
    assert tuple(summary["factors"]) == FACTOR_ORDER
    assert len(ordered_fields(summary)) == 11


def test_runtime_config_rejects_a_mutated_generated_arm(tmp_path: Path) -> None:
    path = _config(tmp_path, "history_off")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["factors"]["history_mode"] = "on"
    write_json(path, payload)
    with pytest.raises(ContractError, match="does not match frozen arm"):
        load_runtime_variant(path)


def test_runtime_config_rejects_unknown_or_duplicate_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"variant_id":"x","variant_id":"y"}\n', encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate JSON key"):
        load_runtime_variant(path)


def test_online_launcher_consumes_config_and_requires_the_combined_lease() -> None:
    launcher = (ROOT / "scripts/run_t4_ablation.sh").read_text(encoding="utf-8")
    assert "GENERATED_VARIANT_CONFIG" in launcher
    assert 'INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" = "dgx+isaac"' in launcher
    assert 'INTERNNAV_RUNTIME_POLICY:-}" = "completion_sim"' in launcher
    assert 'INTERNNAV_SIMULATION_TARGET:-}" = "isaac"' in launcher
    assert "t4_ablation_runtime.py\" fields" in launcher
    assert 'INTERNVLA_T4_DGX_MODEL_CONFIG_SHA256:-}" = "$CONFIG_SHA256"' in launcher
    assert "INTERNVLA_T4_ENFORCE_RECOVERY_GATE=0" not in launcher


def test_completion_fast_path_is_static_lidar_gt_and_view_is_rgb_only() -> None:
    launcher = (ROOT / "scripts/run_t4_sensor_gate.sh").read_text(encoding="utf-8")
    assert "configs/completion_sim/map/nav2_static_lidar.yaml" in launcher
    assert "INTERNVLA_T4_MAP_SOURCE=static_map" in launcher
    assert "INTERNVLA_T4_POSE_SOURCE=ground_truth" in launcher
    assert 'INTERNVLA_T4_DEPTH_HEIGHT_M="$INTERNVLA_T4_CAMERA_HEIGHT_M"' not in launcher
    assert "INTERNVLA_T4_DEPTH_HEIGHT_OVERRIDE_M" in launcher
    assert '"semantic_rgb_view_scope":"internvla_rgb_only"' in launcher
    assert '"mapping_and_safety_geometry":"go2_frozen"' in launcher


def test_adapter_has_independent_factors_and_no_default_forward_action() -> None:
    adapter = (
        ROOT
        / "internvla_t4_recovery/internvla_t4_recovery/adapter_node.py"
    ).read_text(encoding="utf-8")
    compile(adapter, "adapter_node.py", "exec")
    assert "self.latest_system2_action: int | None = None" in adapter
    assert "self.latest_system2_action = 1" not in adapter
    for factor in (
        "system_mode",
        "trajectory_mode",
        "termination_mode",
        "history_mode",
        "recovery_mode",
        "view_mode",
    ):
        assert factor in adapter
    assert "no_real_system2_fallback" in adapter
    assert "oracle_continuation_unavailable" in adapter


def test_dgx_model_server_validates_the_same_generated_config() -> None:
    launcher = (ROOT / "scripts/run_t4_model_server.sh").read_text(encoding="utf-8")
    assert "t4_ablation_runtime.py\" fields" in launcher
    assert "model_variant_claim.json" in launcher
    assert 'INTERNVLA_T4_HISTORY_MODE="${FIELDS[6]}"' in launcher
    assert 'INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" = "dgx+isaac"' in launcher


def test_dgx_model_process_emits_the_required_lifecycle_manifest() -> None:
    model = (
        ROOT
        / "internvla_t4_recovery/internvla_t4_recovery/model_node.py"
    ).read_text(encoding="utf-8")
    compile(model, "model_node.py", "exec")
    assert '"model_lifecycle_manifest.json"' in model
    assert '"model_host": "dgx_spark"' in model
    assert '"history_mode": self.history_mode' in model
    assert 'path.open("x"' in model
