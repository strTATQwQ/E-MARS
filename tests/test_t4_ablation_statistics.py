from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.t4_ablation_analyze import main as analyze_main
from scripts.t4_ablation_validate import main as validate_main
from t4_completion.ablation.contract import (
    ContractError,
    load_matrix,
)
from t4_completion.ablation.records import build_synthetic_collection
from t4_completion.ablation.statistics import (
    build_analysis_outputs,
    paired_bootstrap_ci,
    validate_analysis_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs/completion_sim/ablation/frozen_matrix_v1.json"


def _latency(value: float) -> dict[str, float | int]:
    return {
        "count": 5,
        "mean": value,
        "p50": value,
        "p95": value + 1.0,
        "max": value + 2.0,
    }


def _records(matrix) -> list[dict]:
    rows = build_synthetic_collection(matrix)
    for row in rows:
        if row["variant_id"] == "history_on":
            row["official_metrics"].update({"sr": 1, "os": 1, "spl": 0.8, "ne_m": 1.0})
            row["diagnostics"]["stuck"] = False
            row["diagnostics"]["latency_ms"]["end_to_end"] = _latency(30.0)
        elif row["variant_id"] == "history_off":
            row["official_metrics"].update({"sr": 0, "os": 0, "spl": 0.0, "ne_m": 3.0})
            row["diagnostics"]["stuck"] = True
            row["diagnostics"]["latency_ms"]["end_to_end"] = _latency(40.0)
            row["termination"].update(
                {
                    "effective_source": "stuck",
                    "reason": "stuck",
                    "stop_stability": "not_observed",
                    "model_stop_observed": False,
                }
            )
    return rows


def _comparison(aggregate: dict, comparison_id: str) -> dict:
    return next(
        item for item in aggregate["comparisons"] if item["comparison_id"] == comparison_id
    )


def test_paired_statistics_quantify_raw_and_benefit_oriented_contributions() -> None:
    matrix = load_matrix(MATRIX_PATH)
    episode_level, aggregate = build_analysis_outputs(matrix, _records(matrix))
    history = _comparison(aggregate, "history_on_vs_off")
    assert history["estimation_status"] == "ESTIMATED"
    assert history["paired_episode_count"] == 20

    sr = history["metrics"]["sr"]
    assert sr["raw_mean_delta"] == pytest.approx(1.0)
    assert sr["benefit_mean_delta"] == pytest.approx(1.0)
    assert sr["benefit_ci95"] == pytest.approx([1.0, 1.0])

    ne = history["metrics"]["ne_m"]
    assert ne["raw_mean_delta"] == pytest.approx(-2.0)
    assert ne["benefit_mean_delta"] == pytest.approx(2.0)
    assert ne["benefit_ci95"] == pytest.approx([2.0, 2.0])

    stuck = history["metrics"]["stuck"]
    assert stuck["raw_mean_delta"] == pytest.approx(-1.0)
    assert stuck["benefit_mean_delta"] == pytest.approx(1.0)

    latency = history["metrics"]["latency_end_to_end_mean_ms"]
    assert latency["raw_mean_delta"] == pytest.approx(-10.0)
    assert latency["benefit_mean_delta"] == pytest.approx(10.0)
    assert latency["candidate_wins"] == 20
    assert latency["reference_wins"] == 0

    episode_delta = next(
        item
        for item in episode_level["paired_contributions"]
        if item["comparison_id"] == "history_on_vs_off"
        and item["episode_ordinal"] == 0
    )
    assert episode_delta["metrics"]["ne_m"] == {
        "candidate_value": 1.0,
        "reference_value": 3.0,
        "raw_candidate_minus_reference": -2.0,
        "benefit_oriented_delta": 2.0,
    }


def test_statistics_are_independent_of_input_row_order() -> None:
    matrix = load_matrix(MATRIX_PATH)
    rows = _records(matrix)
    first = build_analysis_outputs(matrix, rows)
    random.Random(2026).shuffle(rows)
    second = build_analysis_outputs(matrix, rows)
    assert first == second


def test_bootstrap_is_seeded_and_constant_delta_interval_is_exact() -> None:
    first = paired_bootstrap_ci(
        [1.0, 2.0, 4.0, 8.0], replicates=200, confidence_level=0.95, seed=7
    )
    second = paired_bootstrap_ci(
        [1.0, 2.0, 4.0, 8.0], replicates=200, confidence_level=0.95, seed=7
    )
    assert first == second
    assert paired_bootstrap_ci(
        [2.0] * 20, replicates=2000, confidence_level=0.95, seed=9
    ) == [2.0, 2.0]
    with pytest.raises(ContractError, match="positive integer"):
        paired_bootstrap_ci([1.0], replicates=0, confidence_level=0.95, seed=1)
    huge = paired_bootstrap_ci(
        [1e308, 1e308], replicates=10, confidence_level=0.95, seed=1
    )
    assert huge == [1e308, 1e308]


def test_finite_extreme_metrics_do_not_overflow_aggregate_mean() -> None:
    matrix = load_matrix(MATRIX_PATH)
    rows = _records(matrix)
    for row in rows:
        if row["variant_id"] != "oracle_termination":
            row["official_metrics"]["ne_m"] = 1e308
    _, aggregate = build_analysis_outputs(matrix, rows)
    assert aggregate["variants"]["history_on"]["metrics"]["ne_m"]["mean"] == 1e308
    assert aggregate["variants"]["history_on"]["metrics"]["ne_m"]["median"] == 1e308


def test_stop_is_an_oracle_assisted_diagnostic_upper_bound() -> None:
    matrix = load_matrix(MATRIX_PATH)
    episode_level, aggregate = build_analysis_outputs(matrix, _records(matrix))
    stop = _comparison(aggregate, "model_stop_gap_to_oracle_termination")
    assert stop["effect_kind"] == "gap_to_oracle"
    assert stop["oracle_assisted"] is True
    assert stop["oracle_assisted_episode_count"] == 20
    assert stop["estimation_status"] == "ESTIMATED"
    assert "upper bound" in stop["interpretation"]
    stop_rows = [
        item
        for item in episode_level["paired_contributions"]
        if item["comparison_id"] == "model_stop_gap_to_oracle_termination"
    ]
    assert all(item["oracle_assisted_episode"] for item in stop_rows)


def test_no_oracle_use_or_recovery_trigger_is_explicitly_not_estimable() -> None:
    matrix = load_matrix(MATRIX_PATH)
    rows = _records(matrix)
    for row in rows:
        if row["variant_id"] == "recovery_on":
            row["activation"]["recovery_trigger_count"] = 0
        if row["variant_id"] == "oracle_termination":
            row["termination"].update(
                {
                    "effective_source": "timeout",
                    "reason": "timeout",
                    "stop_stability": "not_observed",
                    "terminated_by_oracle": False,
                    "oracle_reason": None,
                    "distance_at_termination_m": None,
                }
            )
            row["official_metrics"].update({"sr": 0, "spl": 0.0})
    _, aggregate = build_analysis_outputs(matrix, rows)
    recovery = _comparison(aggregate, "recovery_on_vs_off")
    stop = _comparison(aggregate, "model_stop_gap_to_oracle_termination")
    assert recovery["estimation_status"] == "NOT_ESTIMABLE_NO_RECOVERY_TRIGGER"
    assert stop["estimation_status"] == "NOT_ESTIMABLE_NO_ORACLE_TERMINATION"
    assert stop["oracle_assisted_episode_count"] == 0
    assert all(
        metric["estimation_status"] == "NOT_ESTIMABLE_NO_RECOVERY_TRIGGER"
        and metric["benefit_mean_delta"] is None
        and metric["paired_value_count"] == 0
        for metric in recovery["metrics"].values()
    )
    assert all(
        metric["estimation_status"] == "NOT_ESTIMABLE_NO_ORACLE_TERMINATION"
        and metric["benefit_mean_delta"] is None
        and metric["paired_value_count"] == 0
        for metric in stop["metrics"].values()
    )


def test_analysis_validator_recomputes_outputs_instead_of_trusting_aggregate() -> None:
    matrix = load_matrix(MATRIX_PATH)
    episode_level, aggregate = build_analysis_outputs(matrix, _records(matrix))
    validate_analysis_outputs(matrix, episode_level, aggregate)
    tampered = deepcopy(aggregate)
    tampered["comparisons"][0]["metrics"]["sr"]["benefit_mean_delta"] = 999.0
    with pytest.raises(ContractError, match="deterministic recomputation"):
        validate_analysis_outputs(matrix, episode_level, tampered)


def test_cli_writes_three_machine_readable_files_and_rejects_reuse(
    tmp_path: Path,
) -> None:
    matrix = load_matrix(MATRIX_PATH)
    rows = _records(matrix)
    input_path = tmp_path / "episodes.jsonl"
    input_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"
    args = [
        "--matrix",
        str(MATRIX_PATH),
        "--episodes",
        str(input_path),
        "--output-dir",
        str(output_dir),
    ]
    assert analyze_main(args) == 0
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "aggregate.json",
        "episode_level.json",
        "manifest.json",
    ]
    assert validate_main(
        [
            "--matrix",
            str(MATRIX_PATH),
            "analysis",
            "--analysis-dir",
            str(output_dir),
        ]
    ) == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["aggregate_sha256"] = "0" * 64
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert validate_main(
        [
            "--matrix",
            str(MATRIX_PATH),
            "analysis",
            "--analysis-dir",
            str(output_dir),
        ]
    ) == 2
    assert analyze_main(args) == 2
