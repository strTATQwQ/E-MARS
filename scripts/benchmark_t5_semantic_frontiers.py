#!/usr/bin/env python3
"""Evaluate bounded semantic-frontier ranking without leaking oracle labels.

Input labels may be generated from a reference path and static map, but they
are kept outside ``FrontierFeatures`` and are consulted only after selection.
Consequently this script measures an offline counterfactual; it never claims a
live Nav2 or closed-loop navigation result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slow_planner.base import CandidateFrontier, OrderedImage, SlowPlannerRequest
from slow_planner.semantic_frontier import FrontierFeatures, SemanticFrontierScorer


KIND = "t5_semantic_frontier_offline_cases"
SCOPE = "ORACLE_MAP_OFFLINE_BENCHMARK"
SEMANTIC_STATUSES = frozenset({"SELECTION", "ABSTAIN", "MISSING", "STALE", "ILLEGAL"})


class BenchmarkError(ValueError):
    pass


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read benchmark manifest: {path}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError("benchmark manifest must be an object")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise BenchmarkError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise BenchmarkError(f"{name} must be finite")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _request(case: Mapping[str, Any]) -> SlowPlannerRequest:
    frontiers_raw = case.get("candidate_frontiers")
    if not isinstance(frontiers_raw, list) or not frontiers_raw:
        raise BenchmarkError("each case needs at least one legal candidate frontier")
    frontiers = tuple(CandidateFrontier.from_mapping(item) for item in frontiers_raw)
    episode_id = str(case.get("episode_id") or "")
    snapshot_id = str(case.get("snapshot_id") or "")
    if not episode_id or not snapshot_id:
        raise BenchmarkError("case episode_id/snapshot_id are required")
    return SlowPlannerRequest(
        episode_id=episode_id,
        snapshot_id=snapshot_id,
        instruction=str(case.get("instruction") or "offline benchmark instruction"),
        ordered_images=(OrderedImage("offline_redacted", (0.0,) * 7, b"offline", 1, 1),),
        candidate_frontiers=frontiers,
        agent_pose=(0.0, 0.0, 0.0),
        visited_frontiers=tuple(int(value) for value in case.get("visited_frontiers", [])),
    )


def _labels(case: Mapping[str, Any], legal_ids: set[int]) -> dict[int, dict[str, float]]:
    payload = case.get("labels")
    if not isinstance(payload, Mapping):
        raise BenchmarkError("case labels must be an object")
    if payload.get("usage") != "evaluation_only_not_scorer_input":
        raise BenchmarkError("oracle labels must declare evaluation-only isolation")
    rows = payload.get("frontiers")
    if not isinstance(rows, list):
        raise BenchmarkError("case labels.frontiers must be an array")
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise BenchmarkError("frontier label must be an object")
        frontier_id = int(row.get("frontier_id"))
        if frontier_id in result:
            raise BenchmarkError("frontier labels repeat an ID")
        result[frontier_id] = {
            "progress_m": _finite(row.get("progress_m"), "progress_m"),
            "remaining_geodesic_m": _finite(
                row.get("remaining_geodesic_m"), "remaining_geodesic_m"
            ),
        }
    if set(result) != legal_ids:
        raise BenchmarkError("labels must cover exactly the legal frontier IDs")
    return result


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "case_count": 0,
            "positive_progress_rate": None,
            "mean_progress_m": None,
            "mean_geodesic_regret_m": None,
        }
    return {
        "case_count": len(rows),
        "positive_progress_rate": sum(row["positive"] for row in rows) / len(rows),
        "mean_progress_m": sum(row["progress_m"] for row in rows) / len(rows),
        "mean_geodesic_regret_m": sum(row["regret_m"] for row in rows) / len(rows),
    }


def run_benchmark(
    manifest_path: Path,
    output_dir: Path,
    *,
    minimum_labeled_cases: int = 5,
    minimum_semantic_cases: int = 5,
    minimum_hit_rate_gain: float = 0.10,
    minimum_regret_reduction: float = 0.20,
) -> dict[str, Any]:
    manifest = _load_object(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != KIND:
        raise BenchmarkError("unsupported benchmark schema/kind")
    if manifest.get("evaluation_scope") != SCOPE:
        raise BenchmarkError("benchmark must be explicitly oracle-map offline")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkError("benchmark needs at least one case")
    if output_dir.exists():
        raise BenchmarkError(f"output directory already exists: {output_dir}")

    scorer = SemanticFrontierScorer()
    result_rows: list[dict[str, Any]] = []
    scorer_metrics: list[dict[str, float]] = []
    geometry_metrics: list[dict[str, float]] = []
    random_metrics: list[dict[str, float]] = []
    semantic_scorer_metrics: list[dict[str, float]] = []
    semantic_random_metrics: list[dict[str, float]] = []
    semantic_fallback_metrics: list[dict[str, float]] = []
    status_counts = {name: 0 for name in sorted(SEMANTIC_STATUSES)}

    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise BenchmarkError(f"case[{index}] must be an object")
        request = _request(case)
        features_raw = case.get("features")
        if not isinstance(features_raw, list):
            raise BenchmarkError("case features must be an array")
        features = tuple(FrontierFeatures.from_mapping(item) for item in features_raw)
        legal_ids = {frontier.frontier_id for frontier in request.candidate_frontiers}
        labels = _labels(case, legal_ids)
        semantic_status = str(case.get("semantic_status") or "MISSING")
        if semantic_status not in SEMANTIC_STATUSES:
            raise BenchmarkError(f"unsupported semantic_status={semantic_status!r}")
        status_counts[semantic_status] += 1
        semantic_abstain = semantic_status != "SELECTION"
        if semantic_status == "SELECTION" and not any(
            feature.semantic_relevance is not None for feature in features
        ):
            raise BenchmarkError("SELECTION requires bounded semantic relevance")

        outcome = scorer.rank(
            request,
            features,
            semantic_abstain=semantic_abstain,
            semantic_source=str(case.get("semantic_source") or "semantic_model"),
        )
        geometry_features = tuple(
            FrontierFeatures(
                frontier_id=feature.frontier_id,
                semantic_relevance=None,
                information_gain=feature.information_gain,
                path_cost_m=feature.path_cost_m,
                revisit_count=feature.revisit_count,
                stuck_penalty=feature.stuck_penalty,
            )
            for feature in features
        )
        geometry = scorer.rank(request, geometry_features, semantic_abstain=True)
        best_remaining = min(label["remaining_geodesic_m"] for label in labels.values())

        def measured(frontier_id: int) -> dict[str, float]:
            label = labels[frontier_id]
            return {
                "positive": 1.0 if label["progress_m"] > 0.0 else 0.0,
                "progress_m": label["progress_m"],
                "regret_m": label["remaining_geodesic_m"] - best_remaining,
            }

        selected_id = int(outcome.decision.frontier_id)
        geometry_id = int(geometry.decision.frontier_id)
        scorer_row = measured(selected_id)
        geometry_row = measured(geometry_id)
        random_row = {
            "positive": sum(
                1.0 if label["progress_m"] > 0.0 else 0.0 for label in labels.values()
            )
            / len(labels),
            "progress_m": sum(label["progress_m"] for label in labels.values()) / len(labels),
            "regret_m": sum(
                label["remaining_geodesic_m"] - best_remaining for label in labels.values()
            )
            / len(labels),
        }
        scorer_metrics.append(scorer_row)
        geometry_metrics.append(geometry_row)
        random_metrics.append(random_row)
        if outcome.source != "frozen_fallback":
            semantic_scorer_metrics.append(scorer_row)
            semantic_random_metrics.append(random_row)
            semantic_fallback_metrics.append(geometry_row)
        result_rows.append(
            {
                "schema_version": 1,
                "kind": "t5_semantic_frontier_offline_result",
                "case_index": index,
                "episode_id": request.episode_id,
                "snapshot_id": request.snapshot_id,
                "semantic_status": semantic_status,
                "semantic_source": case.get("semantic_source"),
                "selection_source": outcome.source,
                "selected_frontier_id": selected_id,
                "geometry_frontier_id": geometry_id,
                "legal_frontier_ids": sorted(legal_ids),
                "selected_metrics": scorer_row,
                "geometry_metrics": geometry_row,
                "random_expected_metrics": random_row,
                "scores": [score.to_mapping() for score in outcome.scores],
                "oracle_labels_used_by_scorer": False,
                "motion_authority": "none",
                "terminal_stop_authority": "none",
            }
        )

    scorer_aggregate = _aggregate(scorer_metrics)
    geometry_aggregate = _aggregate(geometry_metrics)
    random_aggregate = _aggregate(random_metrics)
    semantic_aggregate = _aggregate(semantic_scorer_metrics)
    semantic_random_aggregate = _aggregate(semantic_random_metrics)
    semantic_fallback_aggregate = _aggregate(semantic_fallback_metrics)
    hit_gain = None
    regret_reduction_vs_random = None
    regret_reduction_vs_frozen = None
    if semantic_scorer_metrics:
        hit_gain = float(semantic_aggregate["positive_progress_rate"]) - float(
            semantic_random_aggregate["positive_progress_rate"]
        )
        random_regret = float(semantic_random_aggregate["mean_geodesic_regret_m"])
        semantic_regret = float(semantic_aggregate["mean_geodesic_regret_m"])
        regret_reduction_vs_random = (
            (random_regret - semantic_regret) / random_regret
            if random_regret > 0.0
            else 0.0
        )
        fallback_regret = float(
            semantic_fallback_aggregate["mean_geodesic_regret_m"]
        )
        regret_reduction_vs_frozen = (
            (fallback_regret - semantic_regret) / fallback_regret
            if fallback_regret > 0.0
            else 0.0
        )

    blocker = ""
    status = "FAIL"
    if len(result_rows) < minimum_labeled_cases:
        status = "BLOCKED_DATA"
        blocker = "INSUFFICIENT_ORACLE_LABELS"
    elif len(semantic_scorer_metrics) < minimum_semantic_cases:
        status = "BLOCKED_DATA"
        blocker = "INSUFFICIENT_SEMANTIC_COVERAGE"
    elif (hit_gain or 0.0) >= minimum_hit_rate_gain or (
        regret_reduction_vs_frozen or 0.0
    ) >= minimum_regret_reduction:
        status = "PASS"

    summary = {
        "schema_version": 1,
        "kind": "t5_semantic_frontier_offline_summary",
        "status": status,
        "blocker": blocker,
        "evaluation_scope": SCOPE,
        "candidate_frontier_source": manifest.get("candidate_frontier_source"),
        "source_is_live_nav2": bool(manifest.get("source_is_live_nav2", False)),
        "navigation_effect_claim_eligible": False,
        "case_count": len(result_rows),
        "semantic_used_case_count": len(semantic_scorer_metrics),
        "semantic_status_counts": status_counts,
        "scorer": scorer_aggregate,
        "frozen_fallback_baseline": geometry_aggregate,
        "random_expected_baseline": random_aggregate,
        "semantic_used_subset": semantic_aggregate,
        "semantic_used_random_expected": semantic_random_aggregate,
        "semantic_used_frozen_fallback": semantic_fallback_aggregate,
        "semantic_positive_progress_rate_gain_vs_random": hit_gain,
        "semantic_geodesic_regret_reduction_vs_random": regret_reduction_vs_random,
        "semantic_geodesic_regret_reduction_vs_frozen_fallback": regret_reduction_vs_frozen,
        "thresholds": {
            "minimum_labeled_cases": minimum_labeled_cases,
            "minimum_semantic_cases": minimum_semantic_cases,
            "minimum_hit_rate_gain": minimum_hit_rate_gain,
            "minimum_regret_reduction": minimum_regret_reduction,
        },
        "oracle_labels_used_by_scorer": False,
        "illegal_frontier_execution_count": 0,
        "stale_response_execution_count": 0,
        "motion_authority": "none",
        "terminal_stop_authority": "none",
    }
    output_dir.mkdir(parents=True)
    _append_jsonl(output_dir / "results.jsonl", result_rows)
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-labeled-cases", type=int, default=5)
    parser.add_argument("--minimum-semantic-cases", type=int, default=5)
    arguments = parser.parse_args()
    summary = run_benchmark(
        arguments.manifest.resolve(),
        arguments.output.resolve(),
        minimum_labeled_cases=arguments.minimum_labeled_cases,
        minimum_semantic_cases=arguments.minimum_semantic_cases,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] in {"PASS", "BLOCKED_DATA"} else 75


if __name__ == "__main__":
    raise SystemExit(main())
