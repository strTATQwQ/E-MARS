from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict
import inspect
from pathlib import Path
import tarfile

from scripts.benchmark_t5_semantic_frontiers import run_benchmark
from scripts.materialize_t5_semantic_frontier_fixed5 import (
    _legal_frontier_geometry,
    _semantic_row,
    materialize,
)
from slow_planner.semantic_frontier import SemanticFrontierScorer, SemanticFrontierWeights


def case(index: int, *, semantic_status: str = "SELECTION") -> dict:
    semantic = [0.0, 0.0, 1.0] if semantic_status == "SELECTION" else [None, None, None]
    return {
        "episode_id": f"episode-{index}",
        "snapshot_id": f"b::episode-{index}::{index}::0",
        "instruction": "find the kitchen",
        "candidate_frontiers": [
            {"frontier_id": 0, "relative_xz": [-0.8, 1.2], "distance_m": 1.44, "bearing_deg": -33.7},
            {"frontier_id": 1, "relative_xz": [0.0, 1.8], "distance_m": 1.8, "bearing_deg": 0.0},
            {"frontier_id": 2, "relative_xz": [0.8, 1.2], "distance_m": 1.44, "bearing_deg": 33.7},
        ],
        "features": [
            {
                "frontier_id": frontier_id,
                "semantic_relevance": semantic[frontier_id],
                "information_gain": 0.5,
                "path_cost_m": 1.0 + frontier_id,
                "revisit_count": 0,
                "stuck_penalty": 0.0,
            }
            for frontier_id in range(3)
        ],
        "labels": {
            "source": "oracle_static_map_astar",
            "usage": "evaluation_only_not_scorer_input",
            "frontiers": [
                {"frontier_id": 0, "progress_m": -0.5, "remaining_geodesic_m": 4.5},
                {"frontier_id": 1, "progress_m": -0.1, "remaining_geodesic_m": 4.1},
                {"frontier_id": 2, "progress_m": 0.8, "remaining_geodesic_m": 3.2},
            ],
        },
        "semantic_status": semantic_status,
        "semantic_source": "fixture_semantic_model",
    }


def manifest(cases: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "kind": "t5_semantic_frontier_offline_cases",
        "evaluation_scope": "ORACLE_MAP_OFFLINE_BENCHMARK",
        "candidate_frontier_source": "fixture_not_live_nav2",
        "source_is_live_nav2": False,
        "cases": cases,
    }


def test_checked_in_gate_config_matches_scorer_and_benchmark_defaults():
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/internnav_t5/semantic_frontier_offline_gate.json").read_text(
            encoding="utf-8"
        )
    )
    scorer = SemanticFrontierScorer()
    assert config["scorer"]["weights"] == asdict(SemanticFrontierWeights())
    assert config["scorer"]["semantic_min_confidence"] == scorer.semantic_min_confidence
    defaults = inspect.signature(run_benchmark).parameters
    promotion = config["promotion"]
    assert promotion["minimum_labeled_cases"] == defaults["minimum_labeled_cases"].default
    assert promotion["minimum_semantic_cases"] == defaults["minimum_semantic_cases"].default
    assert promotion["positive_progress_rate_gain_vs_random"] == defaults[
        "minimum_hit_rate_gain"
    ].default
    assert promotion["geodesic_regret_reduction_vs_frozen_fallback"] == defaults[
        "minimum_regret_reduction"
    ].default
    assert config["scorer"]["missing_stale_or_abstained_semantics"] == (
        "slow_planner.base.deterministic_fallback"
    )
    assert config["scorer"]["oracle_reference_path_or_goal_as_feature"] is False
    assert config["authority"] == {
        "legal_current_frontier_ids_only": True,
        "cmd_vel": False,
        "terminal_stop": False,
        "coordinate_goal": False,
        "stale_or_illegal_execution": False,
    }
    assert config["limitations"] == {
        "frozen_replay_triplet_is_live_nav2": False,
        "offline_pass_is_navigation_effect_evidence": False,
        "online_canary_allowed_before_offline_pass": False,
    }


def test_legal_frontier_binding_has_no_oracle_input_boundary():
    parameters = set(inspect.signature(_legal_frontier_geometry).parameters)
    assert not parameters.intersection(
        {"reference_path", "reference_cells", "goal", "goal_cell", "labels"}
    )
    grid = bytes([0] * 25)
    legal, rejections = _legal_frontier_geometry(
        [
            {
                "frontier_id": 7,
                "bearing_deg": 0.0,
                "distance_m": 1.0,
                "relative_xz": [0.0, 1.0],
            }
        ],
        start_xy=(2.0, 2.0),
        heading=0.0,
        map_entry={"resolution_m": 1.0, "origin_xy": [0.0, 0.0]},
        grid=grid,
        height=5,
        width=5,
        start_cell=(2, 2),
        resolution=1.0,
    )
    assert [item["frontier"]["frontier_id"] for item in legal] == [7]
    assert rejections == []


def test_internvla_semantics_bind_episode_reset_and_sequence():
    replay_case = {
        "episode_key": "1036_259",
        "snapshot_id": "b::259::3::7",
        "candidate_frontiers": [
            {"frontier_id": 0, "bearing_deg": -30.0},
            {"frontier_id": 1, "bearing_deg": 0.0},
            {"frontier_id": 2, "bearing_deg": 30.0},
        ],
    }
    wrong_reset = {
        "episode_id": "b::259",
        "reset_generation": 2,
        "sequence_id": 7,
        "model_discrete_action": 2,
    }
    exact = {
        "episode_id": "b::259",
        "reset_generation": 3,
        "sequence_id": 7,
        "model_discrete_action": 3,
    }
    status, source, relevance, provenance = _semantic_row(
        replay_case,
        {0, 1, 2},
        {},
        {"1036_259": [wrong_reset, exact]},
        "internvla",
    )
    assert (status, source) == ("SELECTION", "internvla")
    assert relevance == {0: 0.0, 1: 0.0, 2: 1.0}
    assert provenance["source_reset_generation"] == 3

    stale_status, _, stale_relevance, stale_provenance = _semantic_row(
        replay_case,
        {0, 1, 2},
        {},
        {"1036_259": [wrong_reset]},
        "internvla",
    )
    assert stale_status == "STALE"
    assert stale_relevance == {0: None, 1: None, 2: None}
    assert stale_provenance["required_identity"] == {
        "episode_id": "259",
        "reset_generation": 3,
        "sequence_id": 7,
    }


def test_oracle_label_failure_excludes_whole_case_without_changing_legal_set(tmp_path):
    dataset = tmp_path / "dataset.json.gz"
    with gzip.open(dataset, "wt", encoding="utf-8") as stream:
        json.dump(
            {
                "episodes": [
                    {
                        "trajectory_id": 2617,
                        "episode_id": 628,
                        "reference_path": [
                            [1.5, 0.0, -1.5],
                            [3.5, 0.0, -3.5],
                        ],
                        "instruction": {"instruction_text": "find the room"},
                    }
                ]
            },
            stream,
        )

    replay_manifest = {
        "evaluation_scope": "interface_screening_only",
        "candidate_frontier_source": "deterministic_frozen_replay_triplet_not_live_nav2",
        "agent_pose_encoding": "dataset_start_position_xyz_plus_rotation_xyzw",
        "cases": [
            {
                "episode_key": "2617_628",
                "snapshot_id": "b::628::0::0",
                "agent_pose": [1.5, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0],
                "candidate_frontiers": [
                    {
                        "frontier_id": 4,
                        "bearing_deg": 0.0,
                        "distance_m": 1.0,
                        "relative_xz": [0.0, 1.0],
                    }
                ],
            }
        ],
    }
    replay_json = tmp_path / "manifest.json"
    replay_json.write_text(json.dumps(replay_manifest), encoding="utf-8")
    replay_archive = tmp_path / "replay.tar.gz"
    with tarfile.open(replay_archive, "w:gz") as bundle:
        bundle.add(replay_json, arcname="manifest.json")

    step3_results = tmp_path / "step3.jsonl"
    step3_results.write_text(
        json.dumps(
            {
                "episode_key": "2617_628",
                "snapshot_id": "b::628::0::0",
                "decision": {"abstain": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    internvla_records = tmp_path / "internvla.jsonl"
    internvla_records.write_text(
        json.dumps(
            {
                "episode_id": "b::628",
                "reset_generation": 0,
                "sequence_id": 0,
                "model_discrete_action": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Start (1,1) and candidate endpoint (2,1) share one component. The
    # oracle-only goal (3,3) is a separate free component.
    grid_values = [100] * 25
    for row, column in ((1, 1), (2, 1), (3, 3)):
        grid_values[row * 5 + column] = 0
    grid = bytes(grid_values)
    map_root = tmp_path / "maps"
    map_root.mkdir()
    grid_path = map_root / "map.bin"
    grid_path.write_bytes(grid)
    map_manifest = {
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "generations": [{"trajectory_id": 2617, "episode_id": 628, "map_key": "m"}],
        "maps": {
            "m": {
                "file": "map.bin",
                "sha256": hashlib.sha256(grid).hexdigest(),
                "height": 5,
                "width": 5,
                "resolution_m": 1.0,
                "origin_xy": [0.0, 0.0],
            }
        },
    }
    map_manifest_path = tmp_path / "map_manifest.json"
    map_manifest_path.write_text(json.dumps(map_manifest), encoding="utf-8")

    payload = materialize(
        dataset=dataset,
        replay_archive=replay_archive,
        step3_results=step3_results,
        internvla_records=internvla_records,
        map_manifest_path=map_manifest_path,
        map_root=map_root,
        semantic_source="step3_then_internvla",
        output=tmp_path / "output.json",
    )
    assert payload["case_count"] == 0
    assert payload["excluded_case_count"] == 1
    assert payload["case_exclusions"] == [
        {
            "episode_key": "2617_628",
            "legal_frontier_ids": [4],
            "candidate_rejections": [],
            "reason": "ORACLE_START_TO_GOAL_LABEL_UNAVAILABLE",
        }
    ]


def test_offline_gate_passes_only_on_semantic_subset_without_oracle_leak(tmp_path):
    source = tmp_path / "cases.json"
    source.write_text(json.dumps(manifest([case(index) for index in range(5)])), encoding="utf-8")
    summary = run_benchmark(source, tmp_path / "out")
    assert summary["status"] == "PASS"
    assert summary["semantic_used_case_count"] == 5
    assert summary["semantic_positive_progress_rate_gain_vs_random"] > 0.10
    assert summary["semantic_geodesic_regret_reduction_vs_frozen_fallback"] > 0.20
    assert summary["oracle_labels_used_by_scorer"] is False
    assert summary["navigation_effect_claim_eligible"] is False
    rows = [json.loads(line) for line in (tmp_path / "out/results.jsonl").read_text().splitlines()]
    assert all(row["selected_frontier_id"] == 2 for row in rows)
    assert all(row["oracle_labels_used_by_scorer"] is False for row in rows)


def test_realistic_one_of_five_semantic_coverage_is_blocked_data(tmp_path):
    cases = [case(0)] + [case(index, semantic_status="ABSTAIN") for index in range(1, 5)]
    source = tmp_path / "cases.json"
    source.write_text(json.dumps(manifest(cases)), encoding="utf-8")
    summary = run_benchmark(source, tmp_path / "out")
    assert summary["status"] == "BLOCKED_DATA"
    assert summary["blocker"] == "INSUFFICIENT_SEMANTIC_COVERAGE"
    assert summary["semantic_used_case_count"] == 1
    assert summary["semantic_status_counts"]["ABSTAIN"] == 4
    assert summary["illegal_frontier_execution_count"] == 0
    assert summary["stale_response_execution_count"] == 0


def test_zero_fallback_regret_is_not_reported_as_semantic_improvement(tmp_path):
    only = case(0)
    only["labels"]["frontiers"] = [
        {"frontier_id": 0, "progress_m": 0.1, "remaining_geodesic_m": 3.0},
        {"frontier_id": 1, "progress_m": 0.0, "remaining_geodesic_m": 3.4},
        {"frontier_id": 2, "progress_m": 0.1, "remaining_geodesic_m": 3.0},
    ]
    source = tmp_path / "cases.json"
    source.write_text(json.dumps(manifest([only])), encoding="utf-8")
    summary = run_benchmark(
        source,
        tmp_path / "out",
        minimum_labeled_cases=1,
        minimum_semantic_cases=1,
    )
    assert summary["semantic_geodesic_regret_reduction_vs_frozen_fallback"] == 0.0
