#!/usr/bin/env python3
"""Validate, combine and analyze all migrated T4.6 ablation arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from t4_completion.ablation.contract import (  # noqa: E402
    VARIANT_IDS,
    load_matrix,
    write_json,
)
from t4_completion.ablation.records import (  # noqa: E402
    load_episode_jsonl,
    validate_episode_collection,
)
from t4_completion.ablation.statistics import (  # noqa: E402
    analysis_manifest,
    build_analysis_outputs,
    validate_analysis_outputs,
)


MATRIX_PATH = ROOT / "configs/completion_sim/ablation/frozen_matrix_v1.json"

# The frozen statistics contract deliberately distinguishes a complete dataset
# from an activated intervention.  A STOP contrast cannot be estimated when no
# paired episode reaches the frozen oracle success radius.  That is a valid,
# auditable outcome of the frozen run rather than a reason to rewrite the radius
# after seeing the data.  Other non-estimable contrasts remain fatal here.
ALLOWED_ATTRIBUTION_DEVIATIONS = {
    "model_stop_gap_to_oracle_termination": "NOT_ESTIMABLE_NO_ORACLE_TERMINATION",
}


def finalize(result_dir: Path, *, matrix_path: Path = MATRIX_PATH) -> dict[str, object]:
    result_dir = result_dir.resolve()
    combined_path = result_dir / "episode_records.jsonl"
    analysis_dir = result_dir / "analysis"
    summary_path = result_dir / "migrated_ablation_matrix_summary.json"
    for path in (combined_path, analysis_dir, summary_path):
        if path.exists():
            raise FileExistsError(path)
    matrix = load_matrix(matrix_path)
    records = []
    arm_summaries = {}
    observed_step_counts = []
    for variant_id in VARIANT_IDS:
        arm = result_dir / "arms" / variant_id
        arm_records = load_episode_jsonl(arm / "episode_records.jsonl")
        if {record.get("variant_id") for record in arm_records} != {variant_id}:
            raise ValueError(f"arm identity mismatch: {variant_id}")
        records.extend(arm_records)
        summary = arm / f"migrated_ablation20_{variant_id}_summary.json"
        value = json.loads(summary.read_text(encoding="utf-8"))
        if value.get("status") != "PASS" or value.get("ablation_variant_id") != variant_id:
            raise ValueError(f"arm summary is not PASS: {variant_id}")
        per_episode = json.loads(
            (arm / "isaac-run" / "per_episode.json").read_text(encoding="utf-8")
        )
        steps = [
            int(row["step_count"])
            for row in per_episode.get("episodes", [])
            if isinstance(row, dict) and isinstance(row.get("step_count"), int)
        ]
        if len(steps) != 20:
            raise ValueError(f"arm step-count evidence is incomplete: {variant_id}")
        observed_step_counts.extend(steps)
        arm_summaries[variant_id] = {
            "status": "PASS",
            "grant_id": value.get("grant_id"),
            "expected_episode_count": value.get("expected_episode_count"),
            "configured_max_step_threshold": 1200,
            "maximum_observed_step_count": max(steps),
        }
    collection = validate_episode_collection(records, matrix)
    episode_level, aggregate = build_analysis_outputs(matrix, list(collection.records))
    validate_analysis_outputs(matrix, episode_level, aggregate)
    not_estimable = {
        comparison["comparison_id"]: comparison["estimation_status"]
        for comparison in aggregate["comparisons"]
        if comparison["estimation_status"] != "ESTIMATED"
    }
    unexpected_not_estimable = {
        comparison_id: status
        for comparison_id, status in not_estimable.items()
        if ALLOWED_ATTRIBUTION_DEVIATIONS.get(comparison_id) != status
    }
    if unexpected_not_estimable:
        raise ValueError(
            "frozen comparisons are not estimable from observed activation: "
            f"{unexpected_not_estimable}"
        )
    attribution_deviations = [
        {
            "comparison_id": comparison_id,
            "estimation_status": status,
            "kind": "frozen_intervention_not_activated",
            "disposition": "reported_not_imputed",
            "frozen_matrix_modified": False,
        }
        for comparison_id, status in sorted(not_estimable.items())
    ]
    combined_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in collection.records),
        encoding="utf-8",
    )
    analysis_dir.mkdir()
    write_json(analysis_dir / "episode_level.json", episode_level)
    write_json(analysis_dir / "aggregate.json", aggregate)
    write_json(analysis_dir / "manifest.json", analysis_manifest(episode_level, aggregate))
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "functional_status": "T4_6_FROZEN_ABLATIONS_COMPLETE",
        "episode_count_per_variant": collection.episode_count_per_variant,
        "total_episode_record_count": len(collection.records),
        "dataset_sha256": collection.dataset_sha256,
        "episode_manifest_sha256": collection.episode_manifest_sha256,
        "runner_commit_sha": collection.runner_commit_sha,
        "evidence_kind": collection.evidence_kind,
        "execution_target": collection.execution_target,
        "analysis_status": aggregate["analysis_status"],
        "comparison_count": len(aggregate["comparisons"]),
        "estimable_comparison_count": len(aggregate["comparisons"]) - len(not_estimable),
        "attribution_deviations": attribution_deviations,
        "arm_summaries": arm_summaries,
        "comparisons": aggregate["comparisons"],
        "throughput_deviation": {
            "kind": "bounded_episode_step_threshold",
            "configured_max_step_threshold": 1200,
            "maximum_observed_step_count": max(observed_step_counts),
            "evaluator_checks_threshold_at_action_chunk_boundary": True,
            "applied_equally_to_all_arms": True,
            "pairing_or_factor_matrix_changed": False,
        },
        "strict_evidence_modified": False,
        "real_go2_targeted": False,
    }
    write_json(summary_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    args = parser.parse_args()
    payload = finalize(args.result_dir, matrix_path=args.matrix)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "episode_count_per_variant": payload["episode_count_per_variant"],
                "total_episode_record_count": payload["total_episode_record_count"],
                "analysis_status": payload["analysis_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
