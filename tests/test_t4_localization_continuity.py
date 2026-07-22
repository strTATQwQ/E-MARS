from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from t4_completion.localization.cli import main
from t4_completion.localization.config import load_config
from t4_completion.localization.contracts import (
    CANONICAL_CHILD_FRAME,
    CANONICAL_PARENT_FRAME,
    CUVSLAM,
    ISAAC_GT,
    LIDAR_IMU,
    Pose,
    PoseSample,
)
from t4_completion.localization.evidence import LocalizationEvidenceWriter
from t4_completion.localization.selector import LocalizationSelector


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/completion_sim/localization/selector.json"


def selector() -> LocalizationSelector:
    config = load_config(
        CONFIG,
        expected_runtime_policy="completion_sim",
        expected_runtime_target="isaac_simulation",
    )
    return LocalizationSelector(policy=config.policy, sources=config.sources)


def yaw(degrees: float) -> tuple[float, float, float, float]:
    radians = math.radians(degrees)
    return (0.0, 0.0, math.sin(radians / 2), math.cos(radians / 2))


def observation(
    source: str,
    stamp: int,
    received: int,
    sequence: int,
    xyz: tuple[float, float, float],
    yaw_degrees: float,
) -> PoseSample:
    frame = {
        CUVSLAM: "cuvslam_odom",
        LIDAR_IMU: "lidar_imu_odom",
        ISAAC_GT: "isaac_world",
    }[source]
    return PoseSample(
        source=source,
        generation=0,
        sequence_id=sequence,
        stamp_ns=stamp,
        received_monotonic_ns=received,
        parent_frame=frame,
        child_frame="base_link",
        pose=Pose(xyz, yaw(yaw_degrees)),
    )


def test_switch_alignment_has_no_jump_and_then_propagates_motion() -> None:
    runtime = selector()
    runtime.ingest(observation(CUVSLAM, 100, 1_000, 0, (2.0, 3.0, 0.5), 179.0))
    first = runtime.decide(1_000, 100)
    assert first.output is not None

    # cuVSLAM is stale at this decision; the new source has a very different
    # origin and crosses the +179/-179 quaternion sign boundary.
    runtime.ingest(
        observation(
            LIDAR_IMU,
            200,
            6_000_001_001,
            0,
            (100.0, -40.0, 7.0),
            -179.0,
        )
    )
    switched = runtime.decide(6_000_001_001, 200)
    assert switched.output is not None
    assert switched.selected_source == LIDAR_IMU
    assert switched.switch_event is True
    assert switched.switch_reason == "ACTIVE_SOURCE_UNHEALTHY"
    assert switched.switch_translation_jump_m == pytest.approx(0.0, abs=1e-9)
    assert switched.switch_rotation_jump_rad == pytest.approx(0.0, abs=1e-9)
    assert switched.output.pose.translation == pytest.approx(
        first.output.pose.translation, abs=1e-9
    )

    runtime.ingest(
        observation(
            LIDAR_IMU,
            300,
            6_100_000_000,
            1,
            (101.0, -40.0, 7.0),
            -179.0,
        )
    )
    moved = runtime.decide(6_100_000_000, 300)
    assert moved.output is not None
    assert moved.switch_event is False
    assert moved.output.pose.translation_distance(switched.output.pose) == pytest.approx(
        1.0, abs=1e-9
    )


def test_full_se3_alignment_reports_machine_zero_rotation_jump() -> None:
    previous = Pose(
        (4.5, -8.0, 2.25),
        (0.20203050891044216, -0.3030457633656632, 0.4040610178208843, 0.8384266128511549),
    ).normalized()
    incoming = Pose(
        (-91.0, 37.5, 12.0),
        (-0.5144957554275265, 0.1028991510855053, 0.2057983021710106, 0.8252871893281489),
    ).normalized()
    aligned = previous.compose(incoming.inverse()).compose(incoming)
    assert aligned.translation_distance(previous) <= 1e-12
    assert aligned.rotation_distance_rad(previous) <= 1e-12


def test_pose_and_tf_are_one_canonical_value_with_exact_stamp_and_frames() -> None:
    runtime = selector()
    runtime.ingest(observation(CUVSLAM, 123456789, 1_000, 0, (1, 2, 3), 15))
    decision = runtime.decide(1_000, 123456789)
    assert decision.output is not None
    output = decision.output.to_dict()
    assert output["stamp_ns"] == 123456789
    assert output["parent_frame"] == CANONICAL_PARENT_FRAME
    assert output["child_frame"] == CANONICAL_CHILD_FRAME
    assert output["odometry_topic"] == "/odom"
    assert output["tf_topic"] == "/tf"
    # The ROS adapter consumes this single object for both messages; there is
    # no second pose calculation or wall-clock restamp field in the contract.
    assert set(output["pose"]) == {"translation_xyz", "quaternion_xyzw"}


def test_recovery_from_gt_waits_for_hysteresis_then_switches_continuously() -> None:
    runtime = selector()
    runtime.ingest(observation(ISAAC_GT, 100, 1_000, 0, (5, 5, 0), 0))
    fallback = runtime.decide(1_000, 100)
    assert fallback.selected_source == ISAAC_GT

    runtime.ingest(observation(CUVSLAM, 200, 2_000, 0, (0, 0, 0), 0))
    waiting = runtime.decide(2_000, 200)
    assert waiting.switch_event is False
    assert runtime.selected_source == ISAAC_GT

    recovered = runtime.decide(1_000_002_000, 200)
    assert recovered.output is not None
    assert recovered.selected_source == CUVSLAM
    assert recovered.switch_reason == "RECOVERED_AFTER_HYSTERESIS"
    assert recovered.switch_translation_jump_m == pytest.approx(0.0, abs=1e-9)
    assert recovered.odometry_deferred is False
    assert recovered.deviation is None
    assert runtime.summary()["sensor_odometry_status"] == "STRICT_EXTENSION_PENDING"


def test_evidence_writer_refuses_reuse_and_writes_structured_deviation(
    tmp_path: Path,
) -> None:
    runtime = selector()
    runtime.ingest(observation(ISAAC_GT, 100, 1_000, 0, (0, 0, 0), 0))
    decision = runtime.decide(1_000, 100)
    result_dir = tmp_path / "fresh"
    writer = LocalizationEvidenceWriter(
        result_dir,
        record_filename="records.jsonl",
        summary_filename="summary.json",
    )
    writer.append(decision)
    writer.close(runtime.summary(), status="PASS_WITH_DEVIATION")
    records = [json.loads(line) for line in (result_dir / "records.jsonl").read_text().splitlines()]
    assert records[0]["odometry_deferred"] is True
    assert records[0]["deviation"]["severity"] == "WARN"
    deviations = json.loads((result_dir / "localization_deviations.json").read_text())
    assert deviations["status"] == "WARN"
    assert deviations["deviation_count"] == 1
    with pytest.raises(FileExistsError):
        LocalizationEvidenceWriter(
            result_dir,
            record_filename="records.jsonl",
            summary_filename="summary.json",
        )


def test_dependency_free_replay_cli_is_minimally_runnable(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "sample",
                        "source": ISAAC_GT,
                        "generation": 0,
                        "sequence_id": 0,
                        "stamp_ns": 100,
                        "received_monotonic_ns": 1_000,
                        "parent_frame": "isaac_world",
                        "child_frame": "base_link",
                        "translation_xyz": [0, 0, 0],
                        "quaternion_xyzw": [0, 0, 0, 1],
                    }
                ),
                json.dumps(
                    {"event": "tick", "monotonic_ns": 1_000, "sim_time_ns": 100}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result_dir = tmp_path / "replay-result"
    exit_code = main(
        [
            "replay",
            "--config",
            str(CONFIG),
            "--runtime-policy",
            "completion_sim",
            "--runtime-target",
            "isaac_simulation",
            "--events",
            str(events),
            "--result-dir",
            str(result_dir),
        ]
    )
    assert exit_code == 0
    summary = json.loads((result_dir / "localization_summary.json").read_text())
    assert summary["status"] == "PASS_WITH_DEVIATION"
    assert summary["selector"]["sensor_odometry_status"] == "STRICT_EXTENSION_PENDING"


def test_replay_final_stale_source_is_fail_closed(tmp_path: Path) -> None:
    events = tmp_path / "stale-events.jsonl"
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "sample",
                        "source": ISAAC_GT,
                        "generation": 0,
                        "sequence_id": 0,
                        "stamp_ns": 100,
                        "received_monotonic_ns": 1_000,
                        "parent_frame": "isaac_world",
                        "child_frame": "base_link",
                        "translation_xyz": [0, 0, 0],
                        "quaternion_xyzw": [0, 0, 0, 1],
                    }
                ),
                json.dumps(
                    {"event": "tick", "monotonic_ns": 1_000, "sim_time_ns": 100}
                ),
                json.dumps(
                    {
                        "event": "tick",
                        "monotonic_ns": 5_000_001_001,
                        "sim_time_ns": 100,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result_dir = tmp_path / "stale-result"
    exit_code = main(
        [
            "replay",
            "--config",
            str(CONFIG),
            "--runtime-policy",
            "completion_sim",
            "--runtime-target",
            "isaac_simulation",
            "--events",
            str(events),
            "--result-dir",
            str(result_dir),
        ]
    )
    assert exit_code == 2
    summary = json.loads((result_dir / "localization_summary.json").read_text())
    assert summary["status"] == "FAIL"
    assert summary["selector"]["current_output_available"] is False
