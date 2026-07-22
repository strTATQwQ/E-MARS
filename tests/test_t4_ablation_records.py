from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

import pytest

from t4_completion.ablation.contract import (
    ContractError,
    load_matrix,
    strict_json_loads,
)
from t4_completion.ablation.records import (
    build_synthetic_collection,
    build_synthetic_record_template,
    validate_episode_collection,
    validate_episode_record,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs/completion_sim/ablation/frozen_matrix_v1.json"


@pytest.fixture
def matrix():
    return load_matrix(MATRIX_PATH)


def _records(matrix) -> list[dict]:
    return build_synthetic_collection(matrix)


def test_complete_synthetic_collection_is_normalized_by_variant_and_ordinal(matrix) -> None:
    rows = _records(matrix)
    random.Random(11).shuffle(rows)
    collection = validate_episode_collection(rows, matrix)
    assert collection.evidence_kind == "synthetic"
    assert collection.execution_target == "offline_synthetic"
    assert collection.episode_count_per_variant == 20
    assert len(collection.records) == 280
    assert [row["episode_ordinal"] for row in collection.by_variant["history_on"]] == list(
        range(20)
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda row: row.update({"unknown": 1}), "keys mismatch"),
        (lambda row: row.update({"runtime_policy": "strict_evidence"}), "completion_sim"),
        (lambda row: row.update({"execution_target": "real_go2"}), "evidence_kind"),
        (lambda row: row["official_metrics"].update({"spl": float("nan")}), "finite"),
        (lambda row: row["safety"].update({"stale_command_count": 1}), "forbidden"),
        (lambda row: row["activation"].update({"visual_input_only": False}), "visual-input-only"),
    ],
)
def test_episode_record_schema_is_fail_closed(matrix, mutate, match: str) -> None:
    row = build_synthetic_record_template(matrix, "history_on", 0)
    mutate(row)
    with pytest.raises(ContractError, match=match):
        validate_episode_record(row, matrix)


def test_json_loader_rejects_duplicate_keys_and_nonfinite_constants() -> None:
    with pytest.raises(ContractError, match="duplicate"):
        strict_json_loads('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ContractError, match="non-finite"):
        strict_json_loads('{"value":NaN}')
    with pytest.raises(ContractError, match="non-finite"):
        strict_json_loads('{"value":Infinity}')
    with pytest.raises(ContractError, match="invalid"):
        strict_json_loads('{"value":' + "9" * 5000 + "}")


def test_collection_rejects_missing_extra_duplicate_and_mismatched_pairs(matrix) -> None:
    rows = _records(matrix)
    with pytest.raises(ContractError, match="misses variants"):
        validate_episode_collection(
            [row for row in rows if row["variant_id"] != "h1_view"], matrix
        )

    duplicated = rows + [deepcopy(rows[0])]
    with pytest.raises(ContractError, match="duplicate paired"):
        validate_episode_collection(duplicated, matrix)

    mismatched = deepcopy(rows)
    target = next(row for row in mismatched if row["variant_id"] == "history_off")
    target["seed"] += 1
    with pytest.raises(ContractError, match="same paired episode order"):
        validate_episode_collection(mismatched, matrix)

    too_short = [row for row in rows if row["episode_ordinal"] != 19]
    with pytest.raises(ContractError, match=">= 20"):
        validate_episode_collection(too_short, matrix)


def test_collection_binds_runner_commit_run_ids_and_unique_source_paths(matrix) -> None:
    rows = _records(matrix)
    mixed_commit = deepcopy(rows)
    mixed_commit[0]["provenance"]["runner_commit_sha"] = "d" * 40
    with pytest.raises(ContractError, match="runner commits"):
        validate_episode_collection(mixed_commit, matrix)

    mixed_run = deepcopy(rows)
    mixed_run[0]["provenance"]["run_id"] = "different-run"
    with pytest.raises(ContractError, match="mixes run IDs"):
        validate_episode_collection(mixed_run, matrix)

    duplicate_source = deepcopy(rows)
    duplicate_source[1]["provenance"]["source_result_path"] = duplicate_source[0][
        "provenance"
    ]["source_result_path"]
    with pytest.raises(ContractError, match="unique"):
        validate_episode_collection(duplicate_source, matrix)


def test_official_metric_identities_are_enforced(matrix) -> None:
    row = build_synthetic_record_template(matrix, "history_on", 0)
    row["official_metrics"].update({"sr": 0, "os": 1, "spl": 0.1})
    with pytest.raises(ContractError, match="spl cannot exceed sr"):
        validate_episode_record(row, matrix)

    row = build_synthetic_record_template(matrix, "history_on", 0)
    row["official_metrics"].update({"sr": 1, "os": 0, "spl": 0.5})
    with pytest.raises(ContractError, match="sr cannot exceed os"):
        validate_episode_record(row, matrix)

    row = build_synthetic_record_template(matrix, "history_on", 0)
    row["termination"].update(
        {
            "effective_source": "timeout",
            "reason": "timeout",
            "stop_stability": "not_observed",
            "model_stop_observed": False,
        }
    )
    with pytest.raises(ContractError, match="official SR"):
        validate_episode_record(row, matrix)

    row = build_synthetic_record_template(matrix, "oracle_termination", 0)
    row["official_metrics"]["ne_m"] = 2.0
    with pytest.raises(ContractError, match="navigation error"):
        validate_episode_record(row, matrix)

    row = build_synthetic_record_template(matrix, "history_on", 0)
    row["termination"].update({"effective_source": "stuck", "reason": "stuck"})
    row["official_metrics"].update({"sr": 0, "spl": 0.0})
    with pytest.raises(ContractError, match="stuck diagnostic"):
        validate_episode_record(row, matrix)

    row = build_synthetic_record_template(matrix, "history_on", 0)
    row["official_metrics"]["ne_m"] = 10**4000
    with pytest.raises(ContractError, match="representable"):
        validate_episode_record(row, matrix)


def test_model_stop_arm_cannot_claim_oracle_and_oracle_marker_is_complete(matrix) -> None:
    model = build_synthetic_record_template(matrix, "model_stop", 0)
    model["termination"].update(
        {
            "oracle_mode_enabled": True,
            "terminated_by_oracle": True,
            "oracle_reason": "stop_unstable_missing",
        }
    )
    with pytest.raises(ContractError, match="oracle_mode_enabled"):
        validate_episode_record(model, matrix)

    oracle = build_synthetic_record_template(matrix, "oracle_termination", 0)
    oracle["termination"]["oracle_distance_source"] = None
    with pytest.raises(ContractError, match="distance source"):
        validate_episode_record(oracle, matrix)

    oracle = build_synthetic_record_template(matrix, "oracle_termination", 0)
    oracle["termination"]["distance_at_termination_m"] = 3.0
    with pytest.raises(ContractError, match="exceeds"):
        validate_episode_record(oracle, matrix)

    oracle = build_synthetic_record_template(matrix, "oracle_termination", 0)
    oracle["termination"]["oracle_threshold_m"] = 3.0
    oracle["termination"]["distance_at_termination_m"] = 2.8
    with pytest.raises(ContractError, match="threshold value"):
        validate_episode_record(oracle, matrix)

    oracle = build_synthetic_record_template(matrix, "oracle_termination", 0)
    oracle["termination"].update(
        {
            "oracle_reason": "stop_unstable_premature",
            "stop_stability": "premature",
            "model_stop_observed": False,
            "model_stop_suppressed_count": 0,
        }
    )
    with pytest.raises(ContractError, match="STOP stability|observed STOP"):
        validate_episode_record(oracle, matrix)

    model = build_synthetic_record_template(matrix, "model_stop", 0)
    model["termination"].update(
        {
            "effective_source": "timeout",
            "reason": "timeout",
            "stop_stability": "premature",
            "model_stop_observed": False,
        }
    )
    with pytest.raises(ContractError, match="STOP stability"):
        validate_episode_record(model, matrix)


def test_system_oracle_arms_cannot_inherit_oracle_termination(matrix) -> None:
    for variant_id in ("oracle_high_level_system1", "system2_oracle_local_path"):
        row = build_synthetic_record_template(matrix, variant_id, 0)
        assert row["termination"]["configured_policy"] == "model_stop"
        row["termination"]["effective_source"] = "oracle_distance"
        with pytest.raises(ContractError, match="model_stop arm"):
            validate_episode_record(row, matrix)


def test_variant_activation_must_prove_the_selected_mutually_exclusive_path(matrix) -> None:
    row = build_synthetic_record_template(matrix, "system2_oracle_local_path", 0)
    row["activation"]["system1_count"] = 1
    with pytest.raises(ContractError, match="wrong system path"):
        validate_episode_record(row, matrix)

    row = build_synthetic_record_template(matrix, "oracle_high_level_system1", 0)
    row["diagnostics"]["latency_ms"]["system2"] = {
        "count": 1,
        "mean": 1.0,
        "p50": 1.0,
        "p95": 1.0,
        "max": 1.0,
    }
    with pytest.raises(ContractError, match="latency evidence"):
        validate_episode_record(row, matrix)

    row = build_synthetic_record_template(matrix, "history_off", 0)
    row["activation"]["history_frame_count"] = 1
    with pytest.raises(ContractError, match="history activation"):
        validate_episode_record(row, matrix)


def test_system_and_trajectory_activation_is_required_per_arm_not_per_episode(
    matrix,
) -> None:
    row = build_synthetic_record_template(matrix, "full_system1_system2", 0)
    row["activation"]["system1_count"] = 0
    row["diagnostics"]["latency_ms"]["system1"] = {
        "count": 0,
        "mean": None,
        "p50": None,
        "p95": None,
        "max": None,
    }
    validate_episode_record(row, matrix)

    rows = _records(matrix)
    for item in rows:
        if item["variant_id"] == "full_system1_system2":
            item["activation"]["system1_count"] = 0
            item["diagnostics"]["latency_ms"]["system1"] = {
                "count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "max": None,
            }
    with pytest.raises(ContractError, match="aggregate System1/System2"):
        validate_episode_collection(rows, matrix)

    history_collapse = _records(matrix)
    for item in history_collapse:
        if item["variant_id"] == "history_off":
            item["activation"]["system1_count"] = 0
            item["diagnostics"]["latency_ms"]["system1"] = {
                "count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "max": None,
            }
    validate_episode_collection(history_collapse, matrix)

    for item in history_collapse:
        if item["variant_id"] == "history_off":
            item["activation"]["system2_count"] = 0
            item["diagnostics"]["latency_ms"]["system2"] = {
                "count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "max": None,
            }
            item["diagnostics"]["latency_ms"]["end_to_end"] = {
                "count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "max": None,
            }
            item["warnings"] = ["recorder_partial_frame"]
    with pytest.raises(ContractError, match="aggregate full-system activation"):
        validate_episode_collection(history_collapse, matrix)

    row = build_synthetic_record_template(matrix, "endpoint", 0)
    row["activation"]["observed_trajectory_mode"] = "straight_line"
    with pytest.raises(ContractError, match="observed trajectory"):
        validate_episode_record(row, matrix)


def test_correlated_recorder_partial_frame_allows_explicit_missing_latency(matrix) -> None:
    row = build_synthetic_record_template(matrix, "full_system1_system2", 0)
    empty = {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    row["diagnostics"]["latency_ms"] = {
        "system1": dict(empty),
        "system2": dict(empty),
        "end_to_end": dict(empty),
    }
    row["activation"]["system1_count"] = 0
    row["activation"]["system2_count"] = 0
    with pytest.raises(ContractError, match="end_to_end latency"):
        validate_episode_record(row, matrix)
    row["warnings"] = ["recorder_partial_frame"]
    validate_episode_record(row, matrix)

    row = build_synthetic_record_template(matrix, "h1_view", 0)
    row["activation"]["observed_view_mode"] = "go2_view"
    with pytest.raises(ContractError, match="observed view"):
        validate_episode_record(row, matrix)
