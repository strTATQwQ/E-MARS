from __future__ import annotations

import json

from slow_planner.candidate_set_v2 import PRIVATE_DECISION_KIND
from slow_planner.candidate_set_v2_replay import audit_candidate_set_v2_replay
from slow_planner.lane_b import LaneBSnapshotIdentity
from slow_planner.live_frontier_capture import Pose2D
from slow_planner.reachable_viewpoint import (
    ReachableViewpoint,
    build_reachable_viewpoint_mapping,
)


def _source(sequence: int) -> dict:
    identity = LaneBSnapshotIdentity("628", 0, sequence)
    return build_reachable_viewpoint_mapping(
        source_node="/t5/lane_b/reachable_viewpoint_source",
        identity=identity,
        ros_episode_id="b::628",
        captured_sim_time_s=10.0 + sequence,
        valid_until_sim_time_s=12.0 + sequence,
        robot_pose_in_map_frame=Pose2D(0.0, 0.0, 0.0),
        candidates=(
            ReachableViewpoint(0, (0.0, 1.0), (1.0, 0.0), 1.0, 1.0, 0.0, "odom"),
            ReachableViewpoint(1, (1.0, 1.0), (1.0, -1.0), 2**0.5, 1.5, 45.0, "odom"),
        ),
    )


def _decision(source: dict, candidate_id: int = 1) -> dict:
    return {
        "protocol_version": 2,
        "kind": PRIVATE_DECISION_KIND,
        "lane_id": "b",
        "episode_id": source["episode_id"],
        "reset_id": source["reset_id"],
        "sequence_id": source["sequence_id"],
        "snapshot_id": source["snapshot_id"],
        "source_artifact": f"results/lane_b/{source['sequence_id']}.json",
        "source_candidate_set_sha256": source["candidate_set_sha256"],
        "grid_frame_id": source["grid_frame_id"],
        "decision": "select_viewpoint",
        "candidate_id": candidate_id,
        "confidence": 0.8,
    }


def test_real_multi_candidate_sources_and_bound_decisions_pass_minimal_replay(tmp_path):
    for sequence in range(2):
        source = _source(sequence)
        (tmp_path / f"source-{sequence}.json").write_text(json.dumps(source), encoding="utf-8")
        (tmp_path / f"decision-{sequence}.json").write_text(
            json.dumps(_decision(source)), encoding="utf-8"
        )

    result = audit_candidate_set_v2_replay(
        [tmp_path], minimum_multi_candidate_snapshots=2
    )

    assert result["status"] == "OFFLINE_SHADOW_REPLAY_PASS"
    assert result["counts"]["identity_bound_replays"] == 2
    assert result["counts"]["nontrivial_vs_geometry_baseline"] == 2
    assert result["authority"]["publish_navigation_goal"] is False


def test_missing_real_candidate_sets_is_explicit_blocked_data(tmp_path):
    status = {
        "schema_version": 1,
        "kind": "t5_live_frontier_capture_runtime_status",
        "status": "BLOCKED",
        "blocker_code": "NO_CURRENT_LEGAL_FRONTIERS",
        "snapshot_id": "b::628::0::13",
        "snapshot_ready_count": 0,
    }
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")

    result = audit_candidate_set_v2_replay([tmp_path])

    assert result["status"] == "BLOCKED_DATA"
    assert result["counts"]["valid_multi_candidate_snapshots"] == 0
    assert result["blocker_codes"] == [
        "INSUFFICIENT_REAL_MULTI_CANDIDATE_SNAPSHOTS",
        "INSUFFICIENT_IDENTITY_BOUND_RANKING_DECISIONS",
    ]
    assert result["archived_runtime_blockers"][0]["blocker_code"] == "NO_CURRENT_LEGAL_FRONTIERS"
    assert result["authority"]["online_resources_used"] is False
