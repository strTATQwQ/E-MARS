from __future__ import annotations

import json
from dataclasses import replace

import pytest

from slow_planner.base import CandidateFrontier, OrderedImage, PlannerDecision
from slow_planner.lane_b import (
    LaneBSnapshotIdentity,
    TimedRevCImage,
    validate_rev_c_snapshot,
)
from slow_planner.live_frontier import (
    LIVE_FRONTIER_AGENT_POSE_ENCODING,
    LIVE_FRONTIER_GEOMETRY_FRAME,
    LIVE_FRONTIER_KIND,
    BoundedFrontierAdvice,
    LiveBoundedFrontierArbiter,
    LiveFrontierContractError,
    LiveNav2FrontierFile,
    LiveNav2FrontierSnapshot,
    adjudicate_live_bounded_advice,
    build_live_bounded_advisor_request,
    live_frontier_content_sha256,
)


HASH = "a" * 64


def source_mapping(
    *,
    snapshot_id: str = "b::episode-7::2::9",
    captured: float = 10.0,
    valid_until: float = 20.0,
    frontiers: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    identity = LaneBSnapshotIdentity.parse(snapshot_id)
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": LIVE_FRONTIER_KIND,
        "source_node": "/b/nav2_live_frontier_provider",
        "ros_episode_id": f"b::{identity.episode_id}",
        "episode_id": identity.episode_id,
        "reset_id": identity.reset_id,
        "sequence_id": identity.sequence_id,
        "snapshot_id": identity.snapshot_id,
        "captured_sim_time_s": captured,
        "valid_until_sim_time_s": valid_until,
        "geometry_frame": LIVE_FRONTIER_GEOMETRY_FRAME,
        "agent_pose_encoding": LIVE_FRONTIER_AGENT_POSE_ENCODING,
        "agent_pose": [1.0, 2.0, 0.25],
        "candidate_frontiers": frontiers
        if frontiers is not None
        else [
            {
                "frontier_id": 3,
                "relative_xz": [-0.4, 1.2],
                "distance_m": 1.2649,
                "bearing_deg": -18.435,
            },
            {
                "frontier_id": 8,
                "relative_xz": [0.7, 1.0],
                "distance_m": 1.2207,
                "bearing_deg": 34.992,
            },
        ],
        "frontier_set_sha256": "",
    }
    value["frontier_set_sha256"] = live_frontier_content_sha256(value)
    return value


def camera_snapshot(snapshot_id: str = "b::episode-7::2::9"):
    frames = tuple(
        TimedRevCImage(
            image=OrderedImage(view, (0.0,) * 7, f"jpeg-{view}".encode(), 640, 480),
            sim_stamp_s=10.0,
            extrinsic_sha256=HASH,
            source_frame_id=f"{view}_optical",
        )
        for view in ("front_left", "front", "front_right", "rear")
    )
    return validate_rev_c_snapshot(
        episode_id=LaneBSnapshotIdentity.parse(snapshot_id).episode_id,
        snapshot_id=snapshot_id,
        snapshot_sim_stamp_s=10.0,
        frames=frames,
        config_sha256=HASH,
        expected_config_sha256=HASH,
        expected_extrinsic_sha256={frame.image.view_id: HASH for frame in frames},
        max_frame_age_s=0.5,
        max_inter_camera_skew_s=0.2,
    )


def selected(request, frontier_id: int = 3) -> PlannerDecision:
    return PlannerDecision(
        episode_id=request.episode_id,
        snapshot_id=request.snapshot_id,
        decision="select_frontier",
        frontier_id=frontier_id,
        target_relative_xz=None,
        confidence=0.8,
        raw_text="redacted by service boundary",
    )


def test_real_source_contract_binds_unchanged_v1_request_and_returns_only_id() -> None:
    source = LiveNav2FrontierSnapshot.from_mapping(source_mapping())
    request = build_live_bounded_advisor_request(
        camera_snapshot(),
        source,
        instruction="find the kitchen",
        sim_now_s=10.1,
        max_frontier_age_s=5.0,
    )
    assert request.candidate_frontiers == source.candidate_frontiers
    assert request.agent_pose == (1.0, 2.0, 0.25)

    result = adjudicate_live_bounded_advice(
        request,
        selected(request),
        submitted_frontiers=source,
        current_frontiers=source,
        current_identity=source.identity,
        safe_stop_active=False,
        sim_now_s=10.2,
        max_frontier_age_s=5.0,
    )
    assert result.decision == "select_frontier"
    assert result.frontier_id == 3
    public = result.to_mapping()
    assert public["motion_authority"] == "none"
    assert public["terminal_stop_authority"] == "none"
    serialized = json.dumps(public)
    assert "relative_xz" not in serialized
    assert "cmd_vel" not in serialized
    assert '"stop"' not in serialized


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda value: value.update(kind="frozen_replay"),
            "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
        ),
        (
            lambda value: value.update(snapshot_id="b::episode-7::3::9"),
            "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
        ),
        (
            lambda value: value.update(frontier_set_sha256="0" * 64),
            "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
        ),
    ],
)
def test_invalid_or_replay_frontier_sources_fail_closed(mutation, code: str) -> None:
    value = source_mapping()
    mutation(value)
    with pytest.raises(LiveFrontierContractError) as raised:
        LiveNav2FrontierSnapshot.from_mapping(value)
    assert raised.value.code == code


def test_missing_live_source_has_machine_readable_blocker(tmp_path) -> None:
    with pytest.raises(LiveFrontierContractError) as raised:
        LiveNav2FrontierFile(tmp_path / "not-published.json").read()
    assert raised.value.code == "MISSING_LIVE_NAV2_FRONTIER_SOURCE"


def test_empty_live_frontier_set_abstains_before_model_request() -> None:
    source = LiveNav2FrontierSnapshot.from_mapping(source_mapping(frontiers=[]))
    with pytest.raises(LiveFrontierContractError) as raised:
        build_live_bounded_advisor_request(
            camera_snapshot(),
            source,
            instruction="find the kitchen",
            sim_now_s=10.1,
            max_frontier_age_s=5.0,
        )
    assert raised.value.code == "NO_CURRENT_LEGAL_FRONTIERS"


def test_consumer_rechecks_identity_hash_and_sim_freshness() -> None:
    source = LiveNav2FrontierSnapshot.from_mapping(source_mapping())
    request = build_live_bounded_advisor_request(
        camera_snapshot(),
        source,
        instruction="find the kitchen",
        sim_now_s=10.1,
        max_frontier_age_s=5.0,
    )
    changed_mapping = source_mapping(
        frontiers=[
            {
                "frontier_id": 3,
                "relative_xz": [0.4, 1.0],
                "distance_m": 1.077,
                "bearing_deg": 21.8,
            }
        ]
    )
    changed = LiveNav2FrontierSnapshot.from_mapping(changed_mapping)
    result = adjudicate_live_bounded_advice(
        request,
        selected(request),
        submitted_frontiers=source,
        current_frontiers=changed,
        current_identity=source.identity,
        safe_stop_active=False,
        sim_now_s=10.2,
        max_frontier_age_s=5.0,
    )
    assert result.decision == "abstain"
    assert result.frontier_id is None
    assert result.reason == "stale_live_nav2_frontier_source"

    stale = adjudicate_live_bounded_advice(
        request,
        selected(request),
        submitted_frontiers=source,
        current_frontiers=source,
        current_identity=source.identity,
        safe_stop_active=False,
        sim_now_s=16.0,
        max_frontier_age_s=5.0,
    )
    assert stale.decision == "abstain"
    assert stale.reason == "stale_live_nav2_frontier_source"


def test_arbiter_rereads_source_file_before_consuming(tmp_path) -> None:
    source_path = tmp_path / "current.json"
    original_mapping = source_mapping()
    source_path.write_text(json.dumps(original_mapping), encoding="utf-8")
    arbiter = LiveBoundedFrontierArbiter(
        LiveNav2FrontierFile(source_path), max_frontier_age_s=5.0
    )
    submission = arbiter.prepare(
        camera_snapshot(),
        instruction="find the kitchen",
        sim_now_s=10.1,
    )

    changed_mapping = source_mapping(
        frontiers=[
            {
                "frontier_id": 8,
                "relative_xz": [0.7, 1.0],
                "distance_m": 1.2207,
                "bearing_deg": 34.992,
            }
        ]
    )
    source_path.write_text(json.dumps(changed_mapping), encoding="utf-8")
    result = arbiter.consume(
        submission,
        selected(submission.request),
        current_identity=submission.submitted_frontiers.identity,
        safe_stop_active=False,
        sim_now_s=10.2,
    )
    assert result.decision == "abstain"
    assert result.reason == "stale_live_nav2_frontier_source"


def test_stale_or_illegal_step3_output_cannot_be_consumed() -> None:
    source = LiveNav2FrontierSnapshot.from_mapping(source_mapping())
    request = build_live_bounded_advisor_request(
        camera_snapshot(),
        source,
        instruction="find the kitchen",
        sim_now_s=10.1,
        max_frontier_age_s=5.0,
    )
    stale_decision = replace(selected(request), snapshot_id="b::episode-7::2::8")
    stale = adjudicate_live_bounded_advice(
        request,
        stale_decision,
        submitted_frontiers=source,
        current_frontiers=source,
        current_identity=source.identity,
        safe_stop_active=False,
        sim_now_s=10.2,
        max_frontier_age_s=5.0,
    )
    assert stale.decision == "abstain"
    assert stale.frontier_id is None

    illegal = adjudicate_live_bounded_advice(
        request,
        selected(request, 99),
        submitted_frontiers=source,
        current_frontiers=source,
        current_identity=source.identity,
        safe_stop_active=False,
        sim_now_s=10.2,
        max_frontier_age_s=5.0,
    )
    assert illegal.decision == "abstain"
    assert illegal.frontier_id is None


def test_authoritative_identity_advance_or_safe_stop_forces_abstain() -> None:
    source = LiveNav2FrontierSnapshot.from_mapping(source_mapping())
    request = build_live_bounded_advisor_request(
        camera_snapshot(),
        source,
        instruction="find the kitchen",
        sim_now_s=10.1,
        max_frontier_age_s=5.0,
    )
    advanced = adjudicate_live_bounded_advice(
        request,
        selected(request),
        submitted_frontiers=source,
        current_frontiers=source,
        current_identity=LaneBSnapshotIdentity("episode-7", 2, 10),
        safe_stop_active=False,
        sim_now_s=10.2,
        max_frontier_age_s=5.0,
    )
    stopped = adjudicate_live_bounded_advice(
        request,
        selected(request),
        submitted_frontiers=source,
        current_frontiers=source,
        current_identity=source.identity,
        safe_stop_active=True,
        sim_now_s=10.2,
        max_frontier_age_s=5.0,
    )
    assert advanced.decision == stopped.decision == "abstain"
    assert advanced.reason == "stale_live_nav2_frontier_source"
    assert stopped.reason == "safe_stop_active"


def test_bounded_advice_rejects_coordinate_and_stop_payloads() -> None:
    with pytest.raises(TypeError):
        BoundedFrontierAdvice(
            episode_id="episode-7",
            snapshot_id="b::episode-7::2::9",
            decision="select_frontier",
            frontier_id=3,
            reason="ok",
            frontier_set_sha256="a" * 64,
            target_relative_xz=(1.0, 1.0),
        )
