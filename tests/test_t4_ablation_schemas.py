from __future__ import annotations

from pathlib import Path

import pytest

from t4_completion.ablation.contract import ContractError, load_json, load_matrix
from t4_completion.ablation.records import (
    build_synthetic_collection,
    build_synthetic_record_template,
)
from t4_completion.ablation.statistics import build_analysis_outputs


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/completion_sim/ablation"
MATRIX_PATH = CONFIG_DIR / "frozen_matrix_v1.json"


def test_all_declared_schemas_are_checked_in_strict_draft_2020_12_documents() -> None:
    matrix = load_matrix(MATRIX_PATH)
    for name, relative_path in matrix["schemas"].items():
        schema = load_json(ROOT / relative_path)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == relative_path
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert "authoritatively" in schema["$comment"]
        assert schema["required"], name
        assert set(schema["required"]) == set(schema["properties"]), name


def test_schema_documents_freeze_episode_and_aggregate_observability() -> None:
    episode = load_json(CONFIG_DIR / "episode_record_schema_v1.json")
    assert set(episode["properties"]["official_metrics"]["required"]) == {
        "sr",
        "os",
        "spl",
        "ne_m",
    }
    assert set(episode["properties"]["diagnostics"]["required"]) == {
        "stuck",
        "latency_ms",
    }
    termination_required = set(episode["properties"]["termination"]["required"])
    for key in (
        "terminated_by_oracle",
        "oracle_reason",
        "oracle_threshold_m",
        "oracle_threshold_source",
        "oracle_distance_source",
        "distance_at_termination_m",
        "model_stop_observed",
    ):
        assert key in termination_required

    aggregate = load_json(CONFIG_DIR / "aggregate_schema_v1.json")
    assert aggregate["properties"]["comparisons"]["minItems"] == 8
    assert aggregate["properties"]["metric_definitions"]["minItems"] == 11
    assert aggregate["properties"]["episode_count_per_variant"]["minimum"] == 20


def test_runtime_outputs_have_exact_top_level_keys_declared_by_schemas() -> None:
    matrix = load_matrix(MATRIX_PATH)
    rows = build_synthetic_collection(matrix)
    episode_level, aggregate = build_analysis_outputs(matrix, rows)
    episode_schema = load_json(CONFIG_DIR / "episode_level_schema_v1.json")
    aggregate_schema = load_json(CONFIG_DIR / "aggregate_schema_v1.json")
    assert set(episode_level) == set(episode_schema["required"])
    assert set(aggregate) == set(aggregate_schema["required"])
    assert episode_level["analysis_status"] == "SYNTHETIC_COMPLETE"
    assert aggregate["analysis_status"] == "SYNTHETIC_COMPLETE"


def test_boolean_is_rejected_where_integer_metric_is_required() -> None:
    matrix = load_matrix(MATRIX_PATH)
    row = build_synthetic_record_template(matrix, "history_on", 0)
    row["official_metrics"]["sr"] = True
    from t4_completion.ablation.records import validate_episode_record

    with pytest.raises(ContractError, match="integer"):
        validate_episode_record(row, matrix)
