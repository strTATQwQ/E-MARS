from __future__ import annotations

import math
from pathlib import Path

import pytest

from t4_completion.localization.config import SelectorPolicy, load_config
from t4_completion.localization.contracts import (
    CUVSLAM,
    ISAAC_GT,
    LIDAR_IMU,
    Pose,
    PoseSample,
)
from t4_completion.localization.selector import LocalizationSelector


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/completion_sim/localization/selector.json"


def completion_selector() -> LocalizationSelector:
    config = load_config(
        CONFIG,
        expected_runtime_policy="completion_sim",
        expected_runtime_target="isaac_simulation",
    )
    return LocalizationSelector(policy=config.policy, sources=config.sources)


def strict_selector() -> LocalizationSelector:
    config = load_config(
        CONFIG,
        expected_runtime_policy="completion_sim",
        expected_runtime_target="isaac_simulation",
    )
    return LocalizationSelector(
        policy=SelectorPolicy.strict_evidence(), sources=config.sources
    )


def yaw_quaternion(degrees: float) -> tuple[float, float, float, float]:
    radians = math.radians(degrees)
    return (0.0, 0.0, math.sin(radians / 2.0), math.cos(radians / 2.0))


def sample(
    source: str,
    *,
    stamp_ns: int,
    received_ns: int,
    sequence_id: int = 0,
    generation: int = 0,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw_degrees: float = 0.0,
    backend_ready: bool = True,
    tracking: bool = True,
) -> PoseSample:
    parents = {
        CUVSLAM: "cuvslam_odom",
        LIDAR_IMU: "lidar_imu_odom",
        ISAAC_GT: "isaac_world",
    }
    return PoseSample(
        source=source,
        generation=generation,
        sequence_id=sequence_id,
        stamp_ns=stamp_ns,
        received_monotonic_ns=received_ns,
        parent_frame=parents[source],
        child_frame="base_link",
        pose=Pose(translation, yaw_quaternion(yaw_degrees)),
        backend_ready=backend_ready,
        tracking=tracking,
        health_reason="tracking" if tracking else "tracking_lost",
    )


def test_priority_truth_table_prefers_cuvslam_then_lidar_imu() -> None:
    selector = completion_selector()
    assert selector.ingest(sample(CUVSLAM, stamp_ns=100, received_ns=1_000))[0]
    assert selector.ingest(sample(LIDAR_IMU, stamp_ns=100, received_ns=1_000))[0]
    decision = selector.decide(1_000, 100)
    assert decision.selected_source == CUVSLAM
    assert decision.switch_reason == "INITIAL_PREFERRED_HEALTHY"
    assert decision.odometry_deferred is False
    assert decision.deviation is None

    alternate = completion_selector()
    assert alternate.ingest(
        sample(
            CUVSLAM,
            stamp_ns=100,
            received_ns=1_000,
            tracking=False,
        )
    )[0]
    assert alternate.ingest(sample(LIDAR_IMU, stamp_ns=100, received_ns=1_000))[0]
    decision = alternate.decide(1_000, 100)
    assert decision.selected_source == LIDAR_IMU
    assert decision.switch_reason == "INITIAL_ALTERNATE_HEALTHY"
    assert decision.source_health[CUVSLAM].state == "INVALID"
    assert decision.odometry_deferred is False


def test_dual_sensor_failure_explicitly_falls_back_only_in_completion_sim() -> None:
    selector = completion_selector()
    assert selector.ingest(sample(ISAAC_GT, stamp_ns=100, received_ns=1_000))[0]
    decision = selector.decide(1_000, 100)
    assert decision.selected_source == ISAAC_GT
    assert decision.switch_reason == "ALL_SENSOR_ODOMETRY_DEFERRED"
    assert decision.odometry_deferred is True
    assert decision.deviation is not None
    assert decision.deviation["severity"] == "WARN"
    assert decision.deviation["code"] == "ODOMETRY_DEFERRED_ISAAC_GT_FALLBACK"
    assert decision.deviation["sensor_odometry_status"] == "STRICT_EXTENSION_PENDING"
    assert set(decision.deviation["sensor_sources"]) == {CUVSLAM, LIDAR_IMU}

    strict = strict_selector()
    assert strict.ingest(sample(ISAAC_GT, stamp_ns=100, received_ns=1_000))[0]
    strict_decision = strict.decide(1_000, 100)
    assert strict_decision.selected_source is None
    assert strict_decision.output is None
    assert strict_decision.switch_reason == (
        "STRICT_SENSOR_ODOMETRY_UNAVAILABLE_FAIL_CLOSED"
    )
    assert strict_decision.deviation is None


def test_gt_unhealthy_fails_closed_without_republishing_last_pose() -> None:
    selector = completion_selector()
    selector.ingest(sample(ISAAC_GT, stamp_ns=100, received_ns=1_000))
    first = selector.decide(1_000, 100)
    assert first.output is not None
    stale = selector.decide(5_000_001_001, 100)
    assert stale.output is None
    assert stale.selected_source is None
    assert stale.switch_event is True
    assert stale.switch_reason == "GT_UNAVAILABLE_FAIL_CLOSED"

    # Recovery is treated as a real source transition and re-anchors to the
    # last emitted pose instead of silently resuming a potentially jumped raw
    # frame.
    selector.ingest(
        sample(
            ISAAC_GT,
            stamp_ns=200,
            received_ns=5_100_000_000,
            sequence_id=1,
            translation=(100.0, 0.0, 0.0),
        )
    )
    recovered = selector.decide(5_100_000_000, 200)
    assert recovered.switch_event is True
    assert recovered.output is not None
    assert recovered.output.pose.translation == pytest.approx(
        first.output.pose.translation, abs=1e-9
    )


def test_source_timeout_boundary_is_inclusive_then_stale() -> None:
    selector = completion_selector()
    selector.ingest(sample(CUVSLAM, stamp_ns=100, received_ns=1_000))
    exactly = selector.health_snapshot(5_000_001_000, 100)
    assert exactly[CUVSLAM].state == "HEALTHY"
    after = selector.health_snapshot(5_000_001_001, 100)
    assert after[CUVSLAM].state == "STALE"


def test_sim_stamp_age_and_future_stamp_are_fail_closed() -> None:
    selector = completion_selector()
    selector.ingest(sample(CUVSLAM, stamp_ns=100, received_ns=1_000))
    exact = selector.health_snapshot(1_000, 5_000_000_100)
    assert exact[CUVSLAM].state == "HEALTHY"
    old = selector.health_snapshot(1_000, 5_000_000_101)
    assert old[CUVSLAM].state == "STALE"
    assert old[CUVSLAM].reason == "sample_stamp_timeout_exceeded"

    future = completion_selector()
    future.ingest(sample(CUVSLAM, stamp_ns=2_000_001, received_ns=1_000))
    health = future.health_snapshot(1_000, 1_000_000)
    assert health[CUVSLAM].state == "INVALID"
    assert health[CUVSLAM].reason == "future_sample_stamp"


def test_health_snapshot_is_pure_and_decision_time_cannot_regress() -> None:
    selector = completion_selector()
    selector.ingest(sample(CUVSLAM, stamp_ns=100, received_ns=1_000))
    first = selector.health_snapshot(1_000, 100)
    second = selector.health_snapshot(1_000, 100)
    assert first[CUVSLAM].consecutive_healthy_decisions == 0
    assert second[CUVSLAM].consecutive_healthy_decisions == 0
    decided = selector.decide(1_000, 100)
    assert decided.source_health[CUVSLAM].consecutive_healthy_decisions == 1
    with pytest.raises(ValueError, match="monotonic_time_regression"):
        selector.decide(999, 100)
    with pytest.raises(ValueError, match="sim_time_regression"):
        selector.decide(1_001, 99)


def test_old_recovery_candidate_does_not_block_fresh_active_gt() -> None:
    selector = completion_selector()
    selector.ingest(sample(ISAAC_GT, stamp_ns=200, received_ns=1_000))
    assert selector.decide(1_000, 200).selected_source == ISAAC_GT
    selector.ingest(sample(CUVSLAM, stamp_ns=100, received_ns=2_000))
    selector.ingest(
        sample(
            ISAAC_GT,
            stamp_ns=300,
            received_ns=2_000,
            sequence_id=1,
            translation=(1.0, 0.0, 0.0),
        )
    )
    decision = selector.decide(2_000, 300)
    assert decision.selected_source == ISAAC_GT
    assert decision.output is not None
    assert decision.output.stamp_ns == 300


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"generation": 1}, "generation_mismatch"),
        ({"parent_frame": "odom"}, "unexpected_parent_frame"),
        ({"child_frame": "wrong"}, "unexpected_child_frame"),
        ({"stamp_ns": 0}, "nonpositive_stamp"),
        ({"translation": (math.nan, 0.0, 0.0)}, "nonfinite_pose_or_twist"),
    ],
)
def test_invalid_samples_fail_closed(mutation: dict[str, object], reason: str) -> None:
    base = {
        "source": CUVSLAM,
        "generation": 0,
        "sequence_id": 0,
        "stamp_ns": 100,
        "received_monotonic_ns": 1_000,
        "parent_frame": "cuvslam_odom",
        "child_frame": "base_link",
        "translation": (0.0, 0.0, 0.0),
    }
    base.update(mutation)
    candidate = PoseSample(
        source=str(base["source"]),
        generation=int(base["generation"]),
        sequence_id=int(base["sequence_id"]),
        stamp_ns=int(base["stamp_ns"]),
        received_monotonic_ns=int(base["received_monotonic_ns"]),
        parent_frame=str(base["parent_frame"]),
        child_frame=str(base["child_frame"]),
        pose=Pose(base["translation"], (0.0, 0.0, 0.0, 1.0)),  # type: ignore[arg-type]
    )
    accepted, actual_reason = completion_selector().ingest(candidate)
    assert accepted is False
    assert actual_reason == reason


def test_replay_and_sequence_regression_invalidate_source() -> None:
    selector = completion_selector()
    assert selector.ingest(sample(CUVSLAM, stamp_ns=100, received_ns=1_000))[0]
    accepted, reason = selector.ingest(
        sample(CUVSLAM, stamp_ns=100, received_ns=2_000, sequence_id=1)
    )
    assert accepted is False
    assert reason == "non_monotonic_stamp"
    assert selector.health_snapshot(2_000, 100)[CUVSLAM].state == "INVALID"


def test_generation_reset_clears_samples_selection_and_alignment() -> None:
    selector = completion_selector()
    selector.ingest(sample(CUVSLAM, stamp_ns=100, received_ns=1_000))
    assert selector.decide(1_000, 100).output is not None
    selector.reset_generation(1)
    assert selector.last_output is None
    assert selector.selected_source is None
    old = sample(
        CUVSLAM,
        stamp_ns=200,
        received_ns=2_000,
        generation=0,
        sequence_id=1,
    )
    assert selector.ingest(old) == (False, "generation_mismatch")
    assert selector.decide(2_000, 200).output is None
    assert selector.summary()["current_output_available"] is False


def test_generation_reset_allows_sim_clock_restart_but_not_monotonic_regression() -> None:
    selector = completion_selector()
    selector.ingest(sample(CUVSLAM, stamp_ns=100, received_ns=1_000))
    assert selector.decide(1_000, 100).output is not None
    selector.reset_generation(1)
    # A timer may run once against the old clock epoch before /clock resets.
    assert selector.decide(1_500, 100).output is None
    selector.ingest(
        sample(
            CUVSLAM,
            stamp_ns=10,
            received_ns=2_000,
            generation=1,
            sequence_id=0,
        )
    )
    assert selector.decide(2_000, 10).output is not None
    with pytest.raises(ValueError, match="monotonic_time_regression"):
        selector.decide(2_000, 11)
    with pytest.raises(ValueError, match="monotonic_time_regression"):
        selector.decide(1_999, 11)


def test_core_rejects_boolean_identity_and_malformed_health_types() -> None:
    selector = completion_selector()
    candidate = sample(CUVSLAM, stamp_ns=100, received_ns=1_000)
    malformed_sequence = PoseSample(
        source=candidate.source,
        generation=candidate.generation,
        sequence_id=True,  # type: ignore[arg-type]
        stamp_ns=candidate.stamp_ns,
        received_monotonic_ns=candidate.received_monotonic_ns,
        parent_frame=candidate.parent_frame,
        child_frame=candidate.child_frame,
        pose=candidate.pose,
    )
    assert selector.ingest(malformed_sequence) == (
        False,
        "sequence_id_must_be_integer",
    )
    malformed_health = PoseSample(
        source=candidate.source,
        generation=candidate.generation,
        sequence_id=0,
        stamp_ns=101,
        received_monotonic_ns=1_001,
        parent_frame=candidate.parent_frame,
        child_frame=candidate.child_frame,
        pose=candidate.pose,
        backend_ready="false",  # type: ignore[arg-type]
    )
    assert selector.ingest(malformed_health) == (
        False,
        "source_health_must_be_boolean",
    )
    with pytest.raises(ValueError, match="decision_time_must_be_integer"):
        selector.decide(True, 100)  # type: ignore[arg-type]
