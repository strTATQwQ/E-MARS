from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scripts.finalize_t4_migrated_ablation import finalize
from t4_completion.ablation.contract import VARIANT_IDS, load_matrix
from t4_completion.ablation.records import build_synthetic_collection


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/completion_sim/ablation/frozen_matrix_v1.json"


def test_finalize_combines_all_frozen_arms_and_builds_analysis(tmp_path: Path) -> None:
    matrix = load_matrix(MATRIX)
    records = build_synthetic_collection(matrix)
    for variant_id in VARIANT_IDS:
        arm = tmp_path / "arms" / variant_id
        arm.mkdir(parents=True)
        rows = [row for row in records if row["variant_id"] == variant_id]
        (arm / "episode_records.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (arm / f"migrated_ablation20_{variant_id}_summary.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "ablation_variant_id": variant_id,
                    "grant_id": f"synthetic-{variant_id}",
                    "expected_episode_count": 20,
                }
            ),
            encoding="utf-8",
        )
        (arm / "isaac-run").mkdir()
        (arm / "isaac-run/per_episode.json").write_text(
            json.dumps(
                {"episodes": [{"step_count": 1300} for _ in range(20)]}
            ),
            encoding="utf-8",
        )

    summary = finalize(tmp_path)

    assert summary["status"] == "PASS"
    assert summary["episode_count_per_variant"] == 20
    assert summary["total_episode_record_count"] == 280
    assert len(summary["comparisons"]) == 8
    assert summary["comparison_count"] == 8
    assert summary["estimable_comparison_count"] == 8
    assert summary["attribution_deviations"] == []
    assert summary["throughput_deviation"] == {
        "kind": "bounded_episode_step_threshold",
        "configured_max_step_threshold": 1200,
        "maximum_observed_step_count": 1300,
        "evaluator_checks_threshold_at_action_chunk_boundary": True,
        "applied_equally_to_all_arms": True,
        "pairing_or_factor_matrix_changed": False,
    }
    assert (tmp_path / "analysis/aggregate.json").is_file()
    assert (tmp_path / "episode_records.jsonl").is_file()


def test_finalize_reports_unactivated_frozen_stop_without_imputation(
    tmp_path: Path,
) -> None:
    matrix = load_matrix(MATRIX)
    records = build_synthetic_collection(matrix)
    changed = []
    for source in records:
        row = deepcopy(source)
        if row["variant_id"] == "oracle_termination":
            row["termination"].update(
                {
                    "effective_source": "timeout",
                    "reason": "timeout",
                    "stop_stability": "not_observed",
                    "model_stop_observed": False,
                    "model_stop_suppressed_count": 0,
                    "terminated_by_oracle": False,
                    "oracle_reason": None,
                    "distance_at_termination_m": None,
                }
            )
            row["official_metrics"].update({"sr": 0, "spl": 0.0})
        changed.append(row)

    for variant_id in VARIANT_IDS:
        arm = tmp_path / "arms" / variant_id
        arm.mkdir(parents=True)
        rows = [row for row in changed if row["variant_id"] == variant_id]
        (arm / "episode_records.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (arm / f"migrated_ablation20_{variant_id}_summary.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "ablation_variant_id": variant_id,
                    "grant_id": f"synthetic-{variant_id}",
                    "expected_episode_count": 20,
                }
            ),
            encoding="utf-8",
        )
        (arm / "isaac-run").mkdir()
        (arm / "isaac-run/per_episode.json").write_text(
            json.dumps({"episodes": [{"step_count": 100} for _ in range(20)]}),
            encoding="utf-8",
        )

    summary = finalize(tmp_path)

    assert summary["status"] == "PASS"
    assert summary["comparison_count"] == 8
    assert summary["estimable_comparison_count"] == 7
    assert summary["attribution_deviations"] == [
        {
            "comparison_id": "model_stop_gap_to_oracle_termination",
            "estimation_status": "NOT_ESTIMABLE_NO_ORACLE_TERMINATION",
            "kind": "frozen_intervention_not_activated",
            "disposition": "reported_not_imputed",
            "frozen_matrix_modified": False,
        }
    ]
