from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.t4_ablation_generate import main as generate_main
from scripts.t4_ablation_validate import main as validate_main
from t4_completion.ablation.contract import (
    BASELINE_FACTORS,
    COMPARISON_IDS,
    CONTROL_REPEAT_VARIANTS,
    ContractError,
    FACTOR_LEVELS,
    VARIANT_IDS,
    canonical_sha256,
    canonical_text_file_sha256,
    load_matrix,
    resolve_variant_configs,
    variant_config_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs/completion_sim/ablation/frozen_matrix_v1.json"


def test_checked_in_matrix_is_the_exact_frozen_14_arm_8_contrast_contract() -> None:
    matrix = load_matrix(MATRIX_PATH)
    assert tuple(item["variant_id"] for item in matrix["variants"]) == VARIANT_IDS
    assert tuple(item["comparison_id"] for item in matrix["comparisons"]) == COMPARISON_IDS
    assert matrix["baseline_factors"] == BASELINE_FACTORS
    assert matrix["runtime_policy"] == "completion_sim"
    assert matrix["target"] == "isaac_simulation_only"
    assert matrix["forbidden_targets"] == ["real_go2", "hardware_motion"]
    assert matrix["pairing"]["minimum_episode_count"] == 20
    for profile in matrix["runtime_profiles"].values():
        assert canonical_text_file_sha256(ROOT / profile["path"]) == profile["sha256"]


def test_runtime_profile_hash_is_checkout_line_ending_independent(tmp_path: Path) -> None:
    lf = tmp_path / "lf.yaml"
    crlf = tmp_path / "crlf.yaml"
    lf.write_bytes(b"runtime_policy: completion_sim\nvalue: true\n")
    crlf.write_bytes(b"runtime_policy: completion_sim\r\nvalue: true\r\n")
    assert canonical_text_file_sha256(lf) == canonical_text_file_sha256(crlf)


def test_every_factor_is_an_enum_and_each_contrast_changes_exactly_one_axis() -> None:
    matrix = load_matrix(MATRIX_PATH)
    variants = {item["variant_id"]: item for item in matrix["variants"]}
    for variant in variants.values():
        assert set(variant["factors"]) == set(FACTOR_LEVELS)
        for factor, level in variant["factors"].items():
            assert level in FACTOR_LEVELS[factor]
    for comparison in matrix["comparisons"]:
        candidate = variants[comparison["candidate_variant"]]["factors"]
        reference = variants[comparison["reference_variant"]]["factors"]
        changed = [key for key in FACTOR_LEVELS if candidate[key] != reference[key]]
        assert changed == [comparison["axis"]]


def test_repeated_controls_are_semantically_identical_but_configs_remain_named() -> None:
    matrix = load_matrix(MATRIX_PATH)
    configs = resolve_variant_configs(matrix)
    factor_hashes = {
        canonical_sha256(configs[variant_id]["factors"])
        for variant_id in CONTROL_REPEAT_VARIANTS
    }
    assert len(factor_hashes) == 1
    config_hashes = {
        canonical_sha256(configs[variant_id]) for variant_id in CONTROL_REPEAT_VARIANTS
    }
    assert len(config_hashes) == len(CONTROL_REPEAT_VARIANTS)
    assert all(config["strict_evidence_unchanged"] for config in configs.values())
    assert all(config["view_scope"] == "internvla_rgb_only" for config in configs.values())
    assert all(
        config["mapping_and_safety_geometry"] == "go2_frozen"
        for config in configs.values()
    )
    for config in configs.values():
        assert config["forbid_step_3_7"] is True
        assert config["forbid_nvfp4"] is True
        assert config["forbid_model_finetuning"] is True
        assert config["forbid_model_execution_in_worker"] is True


def test_oracle_termination_is_only_enabled_on_its_independent_arm() -> None:
    matrix = load_matrix(MATRIX_PATH)
    configs = resolve_variant_configs(matrix)
    enabled = [
        variant_id
        for variant_id, config in configs.items()
        if config["oracle_termination"]["enabled"]
    ]
    assert enabled == ["oracle_termination"]
    for variant_id in ("oracle_high_level_system1", "system2_oracle_local_path"):
        assert configs[variant_id]["factors"]["termination_mode"] == "model_stop"
        assert configs[variant_id]["oracle_termination"]["enabled"] is False


def test_frozen_json_comparison_rejects_bool_integer_coercion() -> None:
    matrix = load_matrix(MATRIX_PATH)
    from t4_completion.ablation.contract import validate_matrix, validate_variant_config

    mutated = deepcopy(matrix)
    mutated["completion_sim_invariants"]["strict_evidence_unchanged"] = 1
    with pytest.raises(ContractError, match="invariants"):
        validate_matrix(mutated)

    config = resolve_variant_configs(matrix)["history_on"]
    mutated_config = deepcopy(config)
    mutated_config["schema_version"] = True
    with pytest.raises(ContractError, match="does not match"):
        validate_variant_config(matrix, "history_on", mutated_config)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["variants"].pop(), "14-arm"),
        (
            lambda value: value["variants"][4]["factors"].update(
                {"history_mode": "off"}
            ),
            "14-arm",
        ),
        (
            lambda value: value["completion_sim_invariants"].update(
                {"strict_evidence_unchanged": False}
            ),
            "invariants",
        ),
        (
            lambda value: value.update({"target": "real_go2"}),
            "target",
        ),
    ],
)
def test_matrix_mutations_fail_closed(mutation, match: str) -> None:
    matrix = load_matrix(MATRIX_PATH)
    mutated = deepcopy(matrix)
    mutation(mutated)
    from t4_completion.ablation.contract import validate_matrix

    with pytest.raises(ContractError, match=match):
        validate_matrix(mutated)


def test_generator_is_byte_deterministic_and_validator_rejects_extra_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert generate_main(["--matrix", str(MATRIX_PATH), "--output-dir", str(first)]) == 0
    assert generate_main(["--matrix", str(MATRIX_PATH), "--output-dir", str(second)]) == 0
    first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert first_files == second_files
    assert {
        path.as_posix(): (first / path).read_bytes() for path in first_files
    } == {path.as_posix(): (second / path).read_bytes() for path in second_files}
    assert validate_main(
        ["--matrix", str(MATRIX_PATH), "configs", "--config-dir", str(first)]
    ) == 0
    (first / "unexpected.txt").write_text("not allowed\n", encoding="utf-8")
    assert validate_main(
        ["--matrix", str(MATRIX_PATH), "configs", "--config-dir", str(first)]
    ) == 2


def test_manifest_contains_no_time_environment_or_absolute_path_fields() -> None:
    matrix = load_matrix(MATRIX_PATH)
    manifest_text = json.dumps(variant_config_manifest(matrix), sort_keys=True)
    lowered = manifest_text.lower()
    assert "timestamp" not in lowered
    assert "created_at" not in lowered
    assert "environment" not in lowered
    assert str(ROOT).lower() not in lowered


def test_offline_modules_do_not_import_execution_or_network_stacks() -> None:
    forbidden_roots = {
        "subprocess",
        "socket",
        "paramiko",
        "requests",
        "rclpy",
        "isaacsim",
        "torch",
    }
    paths = list((ROOT / "t4_completion/ablation").glob("*.py")) + list(
        (ROOT / "scripts").glob("t4_ablation_*.py")
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & forbidden_roots), (path, imported & forbidden_roots)
