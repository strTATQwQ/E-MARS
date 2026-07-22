from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from slow_planner.base import (
    CandidateFrontier,
    OrderedImage,
    PlannerDecision,
    PlannerMetrics,
    StructuredPlannerDecision,
)
from slow_planner.lane_b import (
    REV_C_VIEW_ORDER,
    LaneBContractError,
    LaneBDecisionSidecar,
    LaneBIntent,
    LaneBPlannerAdapter,
    LaneBPlannerMode,
    LaneBSnapshotAdvance,
    LaneBSnapshotIdentity,
    LaneBSnapshotSequenceTracker,
    LaneBSnapshotSidecar,
    TimedRevCImage,
    build_lane_b_request,
    build_lane_b_snapshot_id,
    frontend_decision_record,
    parse_lane_b_snapshot_id,
    validate_rev_c_snapshot,
)


HASH = "a" * 64
CONFIG_HASH = "b" * 64


def frames(
    *, stamps: tuple[float, ...] = (9.9, 9.9, 9.85, 9.9)
) -> tuple[TimedRevCImage, ...]:
    return tuple(
        TimedRevCImage(
            image=OrderedImage(
                view_id=view_id,
                pose=(float(index), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                jpeg=f"jpeg-{view_id}".encode(),
                width=640,
                height=480,
            ),
            sim_stamp_s=stamps[index],
            extrinsic_sha256=HASH,
            source_frame_id=f"lane_b/{view_id}_optical",
        )
        for index, view_id in enumerate(REV_C_VIEW_ORDER)
    )


def snapshot(
    *,
    episode_id="episode-7",
    reset_id=2,
    sequence_id=11,
    frame_values=None,
    config_sha256=CONFIG_HASH,
    expected_config_sha256=CONFIG_HASH,
    expected_extrinsic_sha256=None,
):
    return validate_rev_c_snapshot(
        episode_id=episode_id,
        snapshot_id=build_lane_b_snapshot_id(episode_id, reset_id, sequence_id),
        snapshot_sim_stamp_s=10.0,
        frames=frames() if frame_values is None else frame_values,
        config_sha256=config_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_extrinsic_sha256=(
            {view_id: HASH for view_id in REV_C_VIEW_ORDER}
            if expected_extrinsic_sha256 is None
            else expected_extrinsic_sha256
        ),
        max_frame_age_s=0.5,
        max_inter_camera_skew_s=0.2,
    )


def request():
    return build_lane_b_request(
        snapshot(),
        instruction="Move to the table.",
        candidate_frontiers=(
            CandidateFrontier(3, (1.0, 0.2), 1.02, 11.3),
            CandidateFrontier(8, (-0.5, 1.0), 1.12, -26.6),
        ),
        agent_pose=(0.0, 0.0, 0.0),
        visited_frontiers=(8,),
        wall_timestamp_s=100.0,
    )


def decision(
    kind: str,
    *,
    frontier_id=None,
    target=None,
    snapshot_id=None,
    fallback_used=False,
    fallback_reason="",
):
    active_request = request()
    return PlannerDecision(
        episode_id=active_request.episode_id,
        snapshot_id=snapshot_id or active_request.snapshot_id,
        decision=kind,
        frontier_id=frontier_id,
        target_relative_xz=target,
        confidence=0.8,
        raw_text="private model generation",
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )


def test_lane_b_snapshot_identity_is_strict_and_round_trips() -> None:
    value = build_lane_b_snapshot_id("episode-7", 2, 11)
    assert value == "b::episode-7::2::11"
    parsed = parse_lane_b_snapshot_id(value)
    assert (parsed.episode_id, parsed.reset_id, parsed.sequence_id) == (
        "episode-7",
        2,
        11,
    )
    for invalid in (
        "a::episode-7::2::11",
        "b::episode-7::-1::11",
        "b::episode-7::02::11",
        "b::episode::extra::2::11",
    ):
        with pytest.raises(LaneBContractError):
            parse_lane_b_snapshot_id(invalid)


def test_snapshot_sequence_tracker_rejects_regression_and_cross_episode_reopen() -> (
    None
):
    tracker = LaneBSnapshotSequenceTracker()
    assert tracker.observe("b::episode-1::0::0") is LaneBSnapshotAdvance.NEW
    assert (
        tracker.observe(LaneBSnapshotIdentity("episode-1", 0, 0))
        is LaneBSnapshotAdvance.IDEMPOTENT_REPLAY
    )
    assert tracker.observe("b::episode-1::0::2") is LaneBSnapshotAdvance.NEW
    with pytest.raises(LaneBContractError, match="sequence_id regressed"):
        tracker.observe("b::episode-1::0::1")
    with pytest.raises(LaneBContractError, match="must begin at sequence_id=0"):
        tracker.observe("b::episode-1::1::1")
    assert tracker.observe("b::episode-1::1::0") is LaneBSnapshotAdvance.NEW
    with pytest.raises(LaneBContractError, match="reset_id regressed"):
        tracker.observe("b::episode-1::0::3")
    with pytest.raises(LaneBContractError, match="begin at sequence_id=0"):
        tracker.observe("b::episode-2::2::1")
    with pytest.raises(LaneBContractError, match="global reset generation"):
        tracker.observe("b::episode-2::1::0")
    with pytest.raises(LaneBContractError, match="global reset generation"):
        tracker.observe("b::episode-2::0::0")
    assert tracker.observe("b::episode-2::2::0") is LaneBSnapshotAdvance.NEW
    assert tracker.observe("b::episode-3::4::0") is LaneBSnapshotAdvance.NEW
    with pytest.raises(LaneBContractError, match="cannot be reopened"):
        tracker.observe("b::episode-1::5::0")


def test_rev_c_snapshot_requires_fixed_order_resolution_and_sim_freshness() -> None:
    value = snapshot()
    assert tuple(image.view_id for image in value.ordered_images) == REV_C_VIEW_ORDER
    assert value.inter_camera_skew_s == pytest.approx(0.05)
    with pytest.raises(LaneBContractError, match="ordered_images"):
        validate_rev_c_snapshot(
            episode_id="episode-7",
            snapshot_id="b::episode-7::2::11",
            snapshot_sim_stamp_s=10.0,
            frames=tuple(reversed(frames())),
            config_sha256=CONFIG_HASH,
            expected_config_sha256=CONFIG_HASH,
            expected_extrinsic_sha256={view_id: HASH for view_id in REV_C_VIEW_ORDER},
            max_frame_age_s=0.5,
            max_inter_camera_skew_s=0.2,
        )
    with pytest.raises(LaneBContractError, match="exceeds max_frame_age"):
        validate_rev_c_snapshot(
            episode_id="episode-7",
            snapshot_id="b::episode-7::2::11",
            snapshot_sim_stamp_s=10.0,
            frames=frames(stamps=(9.0, 9.9, 9.9, 9.9)),
            config_sha256=CONFIG_HASH,
            expected_config_sha256=CONFIG_HASH,
            expected_extrinsic_sha256={view_id: HASH for view_id in REV_C_VIEW_ORDER},
            max_frame_age_s=0.5,
            max_inter_camera_skew_s=2.0,
        )


def test_rev_c_snapshot_requires_exact_frozen_config_and_all_extrinsic_hashes() -> None:
    with pytest.raises(LaneBContractError, match="config hash mismatch"):
        snapshot(expected_config_sha256="c" * 64)
    with pytest.raises(LaneBContractError, match="extrinsic hashes are required"):
        validate_rev_c_snapshot(
            episode_id="episode-7",
            snapshot_id=build_lane_b_snapshot_id("episode-7", 2, 11),
            snapshot_sim_stamp_s=10.0,
            frames=frames(),
            config_sha256=CONFIG_HASH,
            expected_config_sha256=CONFIG_HASH,
            expected_extrinsic_sha256=None,  # type: ignore[arg-type]
            max_frame_age_s=0.5,
            max_inter_camera_skew_s=0.2,
        )
    with pytest.raises(LaneBContractError, match="cover exactly"):
        snapshot(expected_extrinsic_sha256={"front": HASH})
    expected = {view_id: HASH for view_id in REV_C_VIEW_ORDER}
    expected["rear"] = "c" * 64
    with pytest.raises(LaneBContractError, match="extrinsic hash mismatch"):
        snapshot(expected_extrinsic_sha256=expected)


def test_sidecar_is_append_only_metadata_and_mirrors_four_jpegs(tmp_path) -> None:
    writer = LaneBSnapshotSidecar(
        tmp_path / "snapshots.jsonl", camera_dir=tmp_path / "cameras"
    )
    record = writer.append(snapshot())
    stored = json.loads((tmp_path / "snapshots.jsonl").read_text(encoding="utf-8"))
    assert stored["snapshot_id"] == "b::episode-7::2::11"
    assert stored["view_order"] == list(REV_C_VIEW_ORDER)
    assert "jpeg" not in json.dumps(stored).lower().replace("jpeg_sha256", "").replace(
        "jpeg_path", ""
    )
    assert record == stored
    for row in stored["images"]:
        assert (tmp_path / "cameras") in Path(row["jpeg_path"]).parents
        assert Path(row["jpeg_path"]).parts[-4:-1] == (
            "episode-7",
            "reset-2",
            "sequence-11",
        )
        assert Path(row["jpeg_path"]).read_bytes() == f"jpeg-{row['view_id']}".encode()


def test_sidecar_snapshot_id_replay_is_idempotent_and_conflict_cannot_overwrite(
    tmp_path,
) -> None:
    sidecar_path = tmp_path / "snapshots.jsonl"
    camera_dir = tmp_path / "cameras"
    writer = LaneBSnapshotSidecar(sidecar_path, camera_dir=camera_dir)
    original = writer.append(snapshot())
    replay = writer.append(snapshot())
    assert replay == original
    assert len(sidecar_path.read_text(encoding="utf-8").splitlines()) == 1

    changed_frames = list(frames())
    changed_frames[0] = replace(
        changed_frames[0],
        image=replace(changed_frames[0].image, jpeg=b"conflicting-jpeg"),
    )
    with pytest.raises(LaneBContractError, match="snapshot_id reuse conflict"):
        writer.append(snapshot(frame_values=tuple(changed_frames)))
    assert len(sidecar_path.read_text(encoding="utf-8").splitlines()) == 1
    assert Path(original["images"][0]["jpeg_path"]).read_bytes() == b"jpeg-front_left"

    reopened = LaneBSnapshotSidecar(sidecar_path, camera_dir=camera_dir)
    assert reopened.append(snapshot()) == original


def test_sidecar_reloads_real_global_reset_generations_across_episodes(
    tmp_path,
) -> None:
    sidecar_path = tmp_path / "snapshots.jsonl"
    camera_dir = tmp_path / "cameras"
    writer = LaneBSnapshotSidecar(sidecar_path, camera_dir=camera_dir)
    first = snapshot(episode_id="628", reset_id=0, sequence_id=0)
    second = snapshot(episode_id="259", reset_id=1, sequence_id=0)
    writer.append(first)
    expected = writer.append(second)

    reopened = LaneBSnapshotSidecar(sidecar_path, camera_dir=camera_dir)
    assert reopened.append(second) == expected
    assert len(sidecar_path.read_text(encoding="utf-8").splitlines()) == 2


def test_bounded_advisor_can_only_select_current_frontier_or_request_internvla_fallback() -> (
    None
):
    adapter = LaneBPlannerAdapter(LaneBPlannerMode.BOUNDED_ADVISOR)
    selected = adapter.resolve(request(), decision("select_frontier", frontier_id=3))
    assert selected.intent is LaneBIntent.FRONTIER_ADVICE
    assert selected.frontier_id == 3

    target = adapter.resolve(request(), decision("target_found", target=(0.2, 0.4)))
    assert target.intent is LaneBIntent.INTERNVLA_FALLBACK_REQUIRED
    assert target.frontier_id is None
    assert target.target_relative_xz is None
    assert target.requires_internvla_fallback is True
    assert target.fallback_used is True
    assert target.fallback_reason == "bounded_advisor_forbids_relative_target"

    abstained = adapter.resolve(request(), decision("abstain"))
    assert abstained.intent is LaneBIntent.INTERNVLA_FALLBACK_REQUIRED
    assert abstained.frontier_id is None
    assert abstained.fallback_reason == "step3_abstain"


def test_direct_high_level_target_found_is_only_a_safe_hold_candidate() -> None:
    adapter = LaneBPlannerAdapter(LaneBPlannerMode.DIRECT_HIGH_LEVEL)
    selected = adapter.resolve(request(), decision("select_frontier", frontier_id=3))
    assert selected.intent is LaneBIntent.FRONTIER_GOAL_CANDIDATE
    assert selected.frontier_id == 3

    outcome = adapter.resolve(request(), decision("target_found", target=(0.2, 0.4)))
    public = outcome.to_public_mapping()
    assert outcome.intent is LaneBIntent.RELATIVE_TARGET_SAFE_HOLD_CANDIDATE
    assert outcome.requires_arrival_confirmation is True
    assert public["target_relative_xz"] == [0.2, 0.4]
    assert public["motion_authority"] == "none"
    serialized = json.dumps(public).lower()
    assert "cmd_vel" not in serialized
    assert '"stop"' not in serialized


def test_direct_relative_target_is_nonzero_and_bounded_by_current_frontiers() -> None:
    adapter = LaneBPlannerAdapter(LaneBPlannerMode.DIRECT_HIGH_LEVEL)
    active_request = request()
    boundary = max(
        frontier.distance_m for frontier in active_request.candidate_frontiers
    )

    accepted = adapter.resolve(
        active_request,
        decision("target_found", target=(0.0, boundary)),
    )
    assert accepted.intent is LaneBIntent.RELATIVE_TARGET_SAFE_HOLD_CANDIDATE

    cases = (
        ((0.0, 0.0), "relative_target_zero", active_request),
        ((1e300, 0.0), "relative_target_out_of_range", active_request),
        (
            (0.1, 0.0),
            "relative_target_without_frontiers",
            replace(active_request, candidate_frontiers=()),
        ),
    )
    for target, reason, candidate_request in cases:
        rejected = adapter.resolve(
            candidate_request,
            decision("target_found", target=target),
        )
        assert rejected.intent is LaneBIntent.DIRECT_SAFE_STOP_REQUIRED
        assert rejected.target_relative_xz is None
        assert rejected.frontier_id is None
        assert rejected.fallback_used is False
        assert rejected.requires_internvla_fallback is False
        assert rejected.requires_safe_stop is True
        assert rejected.safe_stop_reason == reason


def test_stale_direct_response_requires_safe_stop_without_fallback() -> None:
    adapter = LaneBPlannerAdapter(LaneBPlannerMode.DIRECT_HIGH_LEVEL)
    stale = adapter.resolve(
        request(),
        decision("select_frontier", frontier_id=3, snapshot_id="b::episode-7::2::10"),
    )
    assert stale.intent is LaneBIntent.DIRECT_SAFE_STOP_REQUIRED
    assert stale.frontier_id is None
    assert stale.target_relative_xz is None
    assert stale.requires_internvla_fallback is False
    assert stale.requires_safe_stop is True
    assert stale.fallback_used is False
    assert stale.safe_stop_reason == "stale_or_mismatched_response"


def test_direct_service_failure_requires_coordinator_safe_stop_without_fallback() -> (
    None
):
    adapter = LaneBPlannerAdapter(LaneBPlannerMode.DIRECT_HIGH_LEVEL)
    outcome = adapter.resolve_failure(request(), "timeout")
    public = outcome.to_public_mapping()
    assert outcome.intent is LaneBIntent.DIRECT_SAFE_STOP_REQUIRED
    assert outcome.frontier_id is None
    assert outcome.target_relative_xz is None
    assert outcome.fallback_used is False
    assert outcome.requires_safe_stop is True
    assert outcome.safe_stop_reason == "timeout"
    assert public["fallback_owner"] is None
    assert public["fallback_candidate"] is None
    assert public["motion_authority"] == "none"


def test_structured_step3_summary_reaches_readonly_outcome_without_raw_text() -> None:
    active_request = request()
    model_decision = StructuredPlannerDecision(
        episode_id=active_request.episode_id,
        snapshot_id=active_request.snapshot_id,
        decision="select_frontier",
        frontier_id=3,
        target_relative_xz=None,
        confidence=0.8,
        raw_text="private generation",
        scene_summary="open hallway ahead",
        target_evidence=("doorway visible",),
        blocked_directions=("rear",),
        recommended_frontier=3,
        target_found=False,
        abstain=False,
    )
    public = LaneBPlannerAdapter(LaneBPlannerMode.BOUNDED_ADVISOR).resolve(
        active_request, model_decision
    ).to_public_mapping()
    assert public["scene_summary"] == "open hallway ahead"
    assert public["target_evidence"] == ["doorway visible"]
    assert public["blocked_directions"] == ["rear"]
    assert public["recommended_frontier"] == 3
    assert "private generation" not in json.dumps(public)


def test_step3_internal_or_invalid_frontier_fallback_is_never_executed() -> None:
    adapter = LaneBPlannerAdapter(LaneBPlannerMode.DIRECT_HIGH_LEVEL)
    internal = adapter.resolve(
        request(),
        decision(
            "select_frontier",
            frontier_id=3,
            fallback_used=True,
            fallback_reason="schema_failure:bad",
        ),
    )
    invalid = adapter.resolve(request(), decision("select_frontier", frontier_id=99))
    assert internal.intent is LaneBIntent.DIRECT_SAFE_STOP_REQUIRED
    assert internal.safe_stop_reason == "step3_internal_fallback"
    assert invalid.intent is LaneBIntent.DIRECT_SAFE_STOP_REQUIRED
    assert invalid.safe_stop_reason == "frontier_not_current"
    assert not internal.fallback_used and not invalid.fallback_used
    assert internal.frontier_id is invalid.frontier_id is None


def test_frontend_record_omits_raw_model_text() -> None:
    outcome = LaneBPlannerAdapter(LaneBPlannerMode.BOUNDED_ADVISOR).resolve(
        request(), decision("select_frontier", frontier_id=3)
    )
    record = frontend_decision_record(
        outcome,
        PlannerMetrics(end_to_end_ms=321.0, model_variant="step3_vl_10b_bf16"),
    )
    serialized = json.dumps(record)
    assert "private model generation" not in serialized
    assert "raw_text" not in serialized

    mapping_record = frontend_decision_record(
        outcome,
        {"end_to_end_ms": 12.0, "raw_text": "private", "reasoning": "private"},
    )
    assert mapping_record["metrics"] == {"end_to_end_ms": 12.0}


def test_decision_sidecar_writes_only_redacted_structured_outcome(tmp_path) -> None:
    outcome = LaneBPlannerAdapter(LaneBPlannerMode.BOUNDED_ADVISOR).resolve(
        request(), decision("select_frontier", frontier_id=3)
    )
    writer = LaneBDecisionSidecar(tmp_path / "decisions.jsonl")
    record = writer.append(outcome, {"end_to_end_ms": 17.0})
    stored = json.loads((tmp_path / "decisions.jsonl").read_text(encoding="utf-8"))
    assert stored == record
    assert stored["decision"]["intent"] == "frontier_advice"
    assert "raw_text" not in json.dumps(stored)


def test_request_keeps_frozen_protocol_v1_and_wall_timestamp_separate() -> None:
    value = request()
    assert value.protocol_version == 1
    assert value.timestamp == 100.0
    assert value.snapshot_id == "b::episode-7::2::11"
    assert not hasattr(value.ordered_images[0], "sim_stamp_s")
