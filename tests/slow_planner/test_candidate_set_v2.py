from __future__ import annotations

import inspect
import hashlib
import json
import math
from pathlib import Path

import pytest

from slow_planner.candidate_set_v2 import (
    ABSTAIN,
    PRIVATE_DECISION_KIND,
    PRIVATE_PROTOCOL_VERSION,
    SELECT_VIEWPOINT,
    CandidateSetV2Error,
    candidate_set_v2_compatibility_receipt,
    materialize_candidate_set_v2_shadow,
    parse_candidate_set_v2_decision,
    validate_candidate_set_v2_resolution,
)
from slow_planner.lane_b import (
    FROZEN_INTERNVLA_FALLBACK_CANDIDATE,
    LaneBSnapshotIdentity,
)
from slow_planner.live_frontier_capture import Pose2D
from slow_planner.reachable_viewpoint import (
    ReachableViewpoint,
    build_reachable_viewpoint_mapping,
    viewpoint_content_sha256,
)


IDENTITY = LaneBSnapshotIdentity("628", 4, 17)
ARTIFACT = "results/internnav_t5/lane_b/candidate_set.json"


def candidate_set(
    *, identity: LaneBSnapshotIdentity = IDENTITY, distance_m: float = 2.0
) -> dict:
    candidates = (
        ReachableViewpoint(
            candidate_id=3,
            relative_xz=(0.0, distance_m),
            grid_xy=(2.25, 4.75),
            distance_m=distance_m,
            geodesic_distance_m=2.25,
            bearing_deg=-18.4,
            candidate_xy_frame="odom",
        ),
        ReachableViewpoint(
            candidate_id=7,
            relative_xz=(0.0, 2.4),
            grid_xy=(3.5, 5.0),
            distance_m=2.4,
            geodesic_distance_m=2.5,
            bearing_deg=23.2,
            candidate_xy_frame="odom",
        ),
    )
    return build_reachable_viewpoint_mapping(
        source_node="/t5/lane_b/reachable_viewpoint_source",
        identity=identity,
        ros_episode_id=f"b::{identity.episode_id}",
        captured_sim_time_s=10.0,
        valid_until_sim_time_s=12.5,
        robot_pose_in_map_frame=Pose2D(1.0, 2.0, 0.25),
        candidates=candidates,
    )


def decision_mapping(source: dict, **changes: object) -> dict:
    value = {
        "protocol_version": PRIVATE_PROTOCOL_VERSION,
        "kind": PRIVATE_DECISION_KIND,
        "lane_id": "b",
        "episode_id": source["episode_id"],
        "reset_id": source["reset_id"],
        "sequence_id": source["sequence_id"],
        "snapshot_id": source["snapshot_id"],
        "source_artifact": ARTIFACT,
        "source_candidate_set_sha256": source["candidate_set_sha256"],
        "grid_frame_id": source["grid_frame_id"],
        "decision": SELECT_VIEWPOINT,
        "candidate_id": 3,
        "confidence": 0.8,
    }
    value.update(changes)
    return value


def resolve(source: dict, text: str | None, **changes: object) -> dict:
    arguments = {
        "candidate_set": source,
        "decision_text": text,
        "expected_identity": IDENTITY,
        "expected_source_artifact": ARTIFACT,
        "expected_candidate_set_sha256": source["candidate_set_sha256"],
        "expected_grid_frame_id": "odom",
        "now_sim_time_s": 11.0,
        "maximum_goal_distance_m": 5.0,
    }
    arguments.update(changes)
    return materialize_candidate_set_v2_shadow(**arguments)


def test_select_viewpoint_materializes_only_a_bounded_inert_nav2_proposal() -> None:
    source = candidate_set()
    result = resolve(source, json.dumps(decision_mapping(source)))

    assert result["decision"] == SELECT_VIEWPOINT
    assert result["candidate_id"] == 3
    assert result["snapshot_id"] == "b::628::4::17"
    assert result["source_artifact"] == ARTIFACT
    assert result["source_candidate_set_sha256"] == source["candidate_set_sha256"]
    assert result["proposal"] == {
        "kind": "bounded_nav2_goal_proposal",
        "frame_id": "base_link",
        "position_xy": [2.0, 0.0],
        "distance_m": 2.0,
        "valid_until_sim_time_s": 12.5,
        "proposal_only": True,
    }
    assert result["fallback"]["used"] is False
    assert result["authority"] == {
        "mode": "proposal_only",
        "publish_navigation_goal": False,
        "publish_cmd_vel": False,
        "publish_terminal_stop": False,
    }
    assert "raw_text" not in json.dumps(result)
    validate_candidate_set_v2_resolution(result)


def test_abstain_and_timeout_deterministically_fallback_to_frozen_internvla() -> None:
    source = candidate_set()
    abstain = decision_mapping(
        source, decision=ABSTAIN, candidate_id=None, confidence=0.2
    )
    first = resolve(source, json.dumps(abstain))
    second = resolve(source, json.dumps(abstain))
    timeout = resolve(source, None, failure_reason="timeout")

    assert first == second
    assert first["decision"] == ABSTAIN
    assert first["proposal"] is None
    assert first["fallback"] == {
        "used": True,
        "mode": "shadow_only",
        "candidate": FROZEN_INTERNVLA_FALLBACK_CANDIDATE,
        "reason": "MODEL_ABSTAIN",
    }
    assert timeout["fallback"]["reason"] == "timeout"
    assert timeout["fallback"]["candidate"] == FROZEN_INTERNVLA_FALLBACK_CANDIDATE
    validate_candidate_set_v2_resolution(timeout)

    private_failure = resolve(source, None, failure_reason="hidden reasoning with spaces")
    assert private_failure["fallback"]["reason"] == "timeout"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"candidate_id": 99}, "UNKNOWN_CANDIDATE"),
        ({"source_artifact": "results/other.json"}, "SOURCE_ARTIFACT_MISMATCH"),
        ({"source_candidate_set_sha256": "0" * 64}, "DECISION_DIGEST_MISMATCH"),
        ({"grid_frame_id": "map"}, "DECISION_FRAME_MISMATCH"),
        ({"sequence_id": 16, "snapshot_id": "b::628::4::16"}, "STALE_OR_CROSS_RESET_DECISION"),
    ],
)
def test_unknown_candidate_and_identity_frame_digest_mismatches_fail_to_shadow(
    changes: dict[str, object], reason: str
) -> None:
    source = candidate_set()
    result = resolve(source, json.dumps(decision_mapping(source, **changes)))
    assert result["decision"] == ABSTAIN
    assert result["proposal"] is None
    assert result["fallback"]["reason"] == reason


def test_cross_reset_stale_future_and_source_digest_mismatch_are_rejected() -> None:
    other_source = candidate_set(identity=LaneBSnapshotIdentity("628", 5, 0))
    cross_reset = resolve(
        other_source,
        json.dumps(decision_mapping(other_source)),
        expected_candidate_set_sha256=other_source["candidate_set_sha256"],
    )
    assert cross_reset["fallback"]["reason"] == "STALE_OR_CROSS_RESET_SOURCE"

    source = candidate_set()
    stale = resolve(source, json.dumps(decision_mapping(source)), now_sim_time_s=12.5001)
    future = resolve(source, json.dumps(decision_mapping(source)), now_sim_time_s=9.9999)
    wrong_digest = resolve(
        source,
        json.dumps(decision_mapping(source)),
        expected_candidate_set_sha256="f" * 64,
    )
    assert stale["fallback"]["reason"] == "STALE_OR_FUTURE_SOURCE"
    assert future["fallback"]["reason"] == "STALE_OR_FUTURE_SOURCE"
    assert wrong_digest["fallback"]["reason"] == "SOURCE_DIGEST_MISMATCH"


def test_nonfinite_tampered_and_out_of_bound_coordinates_fail_closed() -> None:
    source = candidate_set()
    tampered = json.loads(json.dumps(source))
    tampered["candidates"][0]["grid_xy"][0] = math.nan
    invalid = resolve(tampered, json.dumps(decision_mapping(source)))
    assert invalid["fallback"]["reason"] == "INVALID_CANDIDATE_SET"

    far_source = candidate_set(distance_m=5.01)
    far = resolve(far_source, json.dumps(decision_mapping(far_source)))
    assert far["fallback"]["reason"] == "CANDIDATE_OUT_OF_BOUNDS"

    inconsistent = candidate_set()
    inconsistent["candidates"][0]["relative_xz"] = [0.0, 4.9]
    inconsistent["candidate_set_sha256"] = viewpoint_content_sha256(inconsistent)
    inconsistent_result = resolve(
        inconsistent, json.dumps(decision_mapping(inconsistent))
    )
    assert inconsistent_result["fallback"]["reason"] == "CANDIDATE_OUT_OF_BOUNDS"

    # A grid coordinate cannot acquire authority: v1 has no same-grid robot
    # pose for every frame, so the proposal is derived from bounded base_link
    # relative_xz instead.
    huge_grid = candidate_set()
    huge_grid["candidates"][0]["grid_xy"] = [1e6, -1e6]
    huge_grid["candidate_set_sha256"] = viewpoint_content_sha256(huge_grid)
    bounded = resolve(huge_grid, json.dumps(decision_mapping(huge_grid)))
    assert bounded["decision"] == SELECT_VIEWPOINT
    assert bounded["proposal"]["frame_id"] == "base_link"
    assert bounded["proposal"]["position_xy"] == [2.0, 0.0]

    with pytest.raises(CandidateSetV2Error) as bound:
        resolve(
            source,
            json.dumps(decision_mapping(source)),
            maximum_goal_distance_m=5.01,
        )
    assert bound.value.code == "INVALID_BOUND"


@pytest.mark.parametrize(
    "raw",
    [
        "reasoning first {}",
        "{\"protocol_version\":2} trailing",
        "{\"protocol_version\":2,\"protocol_version\":2}",
        json.dumps({**decision_mapping(candidate_set()), "raw_text": "hidden"}),
        json.dumps(decision_mapping(candidate_set(), candidate_id=True)),
        json.dumps(decision_mapping(candidate_set(), source_artifact="C:/tmp/set.json")),
        json.dumps(decision_mapping(candidate_set(), source_artifact="C:drive-relative.json")),
    ],
)
def test_private_parser_rejects_prose_cot_unknown_duplicate_and_bool_id(raw: str) -> None:
    with pytest.raises(CandidateSetV2Error):
        parse_candidate_set_v2_decision(raw)


def test_invalid_decision_text_becomes_deterministic_shadow_fallback() -> None:
    source = candidate_set()
    first = resolve(source, "not-json")
    second = resolve(source, "not-json")
    assert first == second
    assert first["fallback"]["reason"] == "INVALID_DECISION"
    assert first["proposal"] is None


def test_resolution_validator_rejects_authority_or_content_tampering() -> None:
    source = candidate_set()
    result = resolve(source, json.dumps(decision_mapping(source)))
    authority = json.loads(json.dumps(result))
    authority["authority"]["publish_navigation_goal"] = True
    with pytest.raises(CandidateSetV2Error) as forbidden:
        validate_candidate_set_v2_resolution(authority)
    assert forbidden.value.code == "FORBIDDEN_AUTHORITY"

    coordinate = json.loads(json.dumps(result))
    coordinate["proposal"]["position_xy"][0] += 0.1
    with pytest.raises(CandidateSetV2Error) as digest:
        validate_candidate_set_v2_resolution(coordinate)
    assert digest.value.code == "INVALID_RESOLUTION"

    # A self-consistent digest does not excuse semantically unbounded output.
    unbounded = json.loads(json.dumps(result))
    unbounded["proposal"]["position_xy"] = [1e6, -1e6]
    content = {
        key: unbounded[key] for key in unbounded if key != "resolution_sha256"
    }
    unbounded["resolution_sha256"] = hashlib.sha256(
        json.dumps(
            content, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(CandidateSetV2Error) as semantic_bound:
        validate_candidate_set_v2_resolution(unbounded)
    assert semantic_bound.value.code == "INVALID_RESOLUTION"


def test_receipt_and_api_prove_private_shadow_only_scope() -> None:
    receipt = candidate_set_v2_compatibility_receipt()
    assert receipt["status"] == "OFFLINE_SHADOW_READY"
    assert receipt["existing_slow_planner_v1_changed"] is False
    assert receipt["reachable_viewpoint_relabelled_as_frontier"] is False
    assert receipt["shared_ros_schema_changed"] is False
    assert receipt["online_runtime_wired"] is False
    assert receipt["authority"]["publish_navigation_goal"] is False
    assert set(inspect.signature(materialize_candidate_set_v2_shadow).parameters).isdisjoint(
        {"goal_publisher", "cmd_vel_publisher", "stop_publisher", "nav2_client"}
    )


def test_candidate_set_source_digest_is_canonical_and_bound() -> None:
    source = candidate_set()
    assert source["candidate_set_sha256"] == viewpoint_content_sha256(source)
    parsed = parse_candidate_set_v2_decision(json.dumps(decision_mapping(source)))
    assert parsed.source_candidate_set_sha256 == source["candidate_set_sha256"]


def test_machine_readable_config_and_receipt_keep_authority_and_v1_frozen() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/internnav_t5/lane_b_candidate_set_v2_shadow.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = json.loads(
        (root / "reports/t5_lane_b_candidate_set_v2_compatibility.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["online_runtime_wired"] is False
    assert config["input_candidate_type"] == "reachable_free_space_viewpoint"
    assert config["authority"]["publish_navigation_goal"] is False
    assert config["frozen_contracts"]["slow_planner_v1_changed"] is False
    assert config["frozen_contracts"]["reachable_viewpoint_relabelled_as_frontier"] is False
    assert receipt["status"] == "OFFLINE_SHADOW_READY_ONLINE_NOT_RUN"
    assert receipt["compatibility"]["runtime_runner_added"] is False
    assert receipt["promotion"]["eligible_for_online_control"] is False
