from __future__ import annotations

from dataclasses import asdict
import inspect
import json
from pathlib import Path

import pytest

from slow_planner.base import CandidateFrontier
from slow_planner.lane_b import LaneBSnapshotIdentity
from slow_planner.live_frontier import LiveNav2FrontierSnapshot
from slow_planner.live_frontier_capture import (
    AtomicLiveFrontierWriter,
    CaptureIdentity,
    FrontierCaptureConfig,
    LiveFrontierCaptureCoordinator,
    LiveFrontierCaptureError,
    OccupancyGrid2D,
    Pose2D,
    build_live_frontier_mapping,
    extract_reachable_frontiers,
    resolve_lane_topic,
)


SNAPSHOT_KEYS = {
    "schema_version",
    "kind",
    "source_node",
    "ros_episode_id",
    "episode_id",
    "reset_id",
    "sequence_id",
    "snapshot_id",
    "captured_sim_time_s",
    "valid_until_sim_time_s",
    "geometry_frame",
    "agent_pose_encoding",
    "agent_pose",
    "candidate_frontiers",
    "frontier_set_sha256",
}


def grid(rows: list[str], *, stamp: float = 10.0) -> OccupancyGrid2D:
    values = {"#": 100, ".": 0, "?": -1}
    assert rows and len({len(row) for row in rows}) == 1
    return OccupancyGrid2D(
        width=len(rows[0]),
        height=len(rows),
        resolution_m=1.0,
        origin_x_m=0.0,
        origin_y_m=0.0,
        origin_yaw_rad=0.0,
        frame_id="odom",
        stamp_sim_time_s=stamp,
        data=tuple(values[cell] for row in rows for cell in row),
    )


def identity(
    *, reset: int = 0, sequence: int = 0, captured: float = 10.0
) -> CaptureIdentity:
    lane = LaneBSnapshotIdentity("episode-7", reset, sequence)
    return CaptureIdentity(
        identity=lane,
        ros_episode_id="b::episode-7",
        captured_sim_time_s=captured,
        valid_until_sim_time_s=captured + 2.0,
    )


def corridor(*, stamp: float = 10.0) -> OccupancyGrid2D:
    return grid(
        [
            "#######",
            "#.....?",
            "#.....?",
            "#######",
        ],
        stamp=stamp,
    )


def test_unknown_boundary_clusters_to_reachable_centroid_representative() -> None:
    frontiers = extract_reachable_frontiers(
        corridor(),
        robot_pose_in_grid_frame=Pose2D(1.5, 1.5, 0.0),
    )
    assert len(frontiers) == 1
    assert frontiers[0].frontier_id == 0
    assert frontiers[0].relative_xz == (0.0, 4.0)
    assert frontiers[0].distance_m == 4.0
    assert frontiers[0].bearing_deg == 0.0


def test_occupied_partition_excludes_unreachable_unknown_boundary() -> None:
    separated = grid(
        [
            "#########",
            "#...#...?",
            "#...#...?",
            "#########",
        ]
    )
    assert (
        extract_reachable_frontiers(
            separated,
            robot_pose_in_grid_frame=Pose2D(1.5, 1.5, 0.0),
        )
        == ()
    )


def test_diagonal_corner_cut_does_not_create_false_connectivity() -> None:
    corner_blocked = grid(
        [
            "#####",
            "#.###",
            "##.?#",
            "##.?#",
            "#####",
        ]
    )
    assert (
        extract_reachable_frontiers(
            corner_blocked,
            robot_pose_in_grid_frame=Pose2D(1.5, 1.5, 0.0),
        )
        == ()
    )


def test_extractor_has_no_oracle_goal_instruction_or_reference_input() -> None:
    parameters = set(inspect.signature(extract_reachable_frontiers).parameters)
    assert parameters == {"grid", "robot_pose_in_grid_frame", "config"}
    forbidden = {"goal", "reference_path", "instruction", "success", "model"}
    assert not parameters.intersection(forbidden)


def test_snapshot_mapping_uses_only_frozen_schema_and_content_hash() -> None:
    current_identity = identity()
    candidate = CandidateFrontier(0, (0.0, 2.0), 2.0, 0.0)
    first = build_live_frontier_mapping(
        source_node="/b/nav2_live_frontier_capture",
        identity=current_identity,
        robot_pose_in_map_frame=Pose2D(1.0, 2.0, 0.25),
        candidate_frontiers=(candidate,),
        valid_until_sim_time_s=11.0,
    )
    assert set(first) == SNAPSHOT_KEYS
    parsed = LiveNav2FrontierSnapshot.from_mapping(first)
    assert parsed.identity.snapshot_id == "b::episode-7::0::0"
    assert parsed.candidate_frontiers == (candidate,)

    changed = build_live_frontier_mapping(
        source_node="/b/nav2_live_frontier_capture",
        identity=current_identity,
        robot_pose_in_map_frame=Pose2D(1.0, 2.0, 0.25),
        candidate_frontiers=(CandidateFrontier(0, (0.1, 2.0), 2.002498439, 2.862405226),),
        valid_until_sim_time_s=11.0,
    )
    assert changed["frontier_set_sha256"] != first["frontier_set_sha256"]


def test_reset_clears_previous_atomic_snapshot_before_new_generation(tmp_path) -> None:
    output = tmp_path / "current.json"
    coordinator = LiveFrontierCaptureCoordinator(
        AtomicLiveFrontierWriter(output),
        source_node="/b/nav2_live_frontier_capture",
    )
    payload = coordinator.capture(
        identity=identity(),
        grid=corridor(),
        robot_pose_in_grid_frame=Pose2D(1.5, 1.5, 0.0),
        robot_pose_in_map_frame=Pose2D(4.0, 5.0, 0.0),
        odometry_sim_time_s=10.0,
        sim_now_s=10.1,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload

    coordinator.observe_identity(LaneBSnapshotIdentity("episode-8", 1, 0))
    assert not output.exists()


def test_missing_identity_clears_file_but_retains_reset_monotonicity(tmp_path) -> None:
    output = tmp_path / "current.json"
    coordinator = LiveFrontierCaptureCoordinator(
        AtomicLiveFrontierWriter(output),
        source_node="/b/nav2_live_frontier_capture",
    )
    coordinator.capture(
        identity=identity(reset=2),
        grid=corridor(),
        robot_pose_in_grid_frame=Pose2D(1.5, 1.5, 0.0),
        robot_pose_in_map_frame=Pose2D(4.0, 5.0, 0.0),
        odometry_sim_time_s=10.0,
        sim_now_s=10.1,
    )
    coordinator.clear_for_missing_identity()
    assert not output.exists()
    with pytest.raises(LiveFrontierCaptureError) as regressed:
        coordinator.observe_identity(LaneBSnapshotIdentity("episode-7", 1, 0))
    assert regressed.value.code == "STALE_CAPTURE_IDENTITY"


def test_stale_input_and_sequence_regression_publish_nothing(tmp_path) -> None:
    output = tmp_path / "current.json"
    coordinator = LiveFrontierCaptureCoordinator(
        AtomicLiveFrontierWriter(output),
        source_node="/b/nav2_live_frontier_capture",
    )
    with pytest.raises(LiveFrontierCaptureError) as stale:
        coordinator.capture(
            identity=identity(captured=5.0),
            grid=corridor(stamp=5.0),
            robot_pose_in_grid_frame=Pose2D(1.5, 1.5, 0.0),
            robot_pose_in_map_frame=Pose2D(4.0, 5.0, 0.0),
            odometry_sim_time_s=5.0,
            sim_now_s=10.0,
        )
    assert stale.value.code == "STALE_METADATA"
    assert not output.exists()

    coordinator.observe_identity(LaneBSnapshotIdentity("episode-7", 0, 2))
    with pytest.raises(LiveFrontierCaptureError) as regressed:
        coordinator.observe_identity(LaneBSnapshotIdentity("episode-7", 0, 1))
    assert regressed.value.code == "STALE_CAPTURE_IDENTITY"
    assert not output.exists()


def test_expiry_removes_snapshot_even_for_same_identity(tmp_path) -> None:
    output = tmp_path / "current.json"
    config = FrontierCaptureConfig(snapshot_ttl_s=0.5)
    coordinator = LiveFrontierCaptureCoordinator(
        AtomicLiveFrontierWriter(output),
        source_node="/b/nav2_live_frontier_capture",
        config=config,
    )
    coordinator.capture(
        identity=identity(),
        grid=corridor(),
        robot_pose_in_grid_frame=Pose2D(1.5, 1.5, 0.0),
        robot_pose_in_map_frame=Pose2D(4.0, 5.0, 0.0),
        odometry_sim_time_s=10.0,
        sim_now_s=10.1,
    )
    assert output.exists()
    coordinator.expire(sim_now_s=10.6)
    assert not output.exists()


@pytest.mark.parametrize(
    ("namespace", "topic", "expected"),
    [
        ("/b", "odom", "/b/odom"),
        ("/lane_b", "local_costmap/costmap", "/lane_b/local_costmap/costmap"),
    ],
)
def test_topics_are_confined_to_lane_namespace(
    namespace: str, topic: str, expected: str
) -> None:
    assert resolve_lane_topic(namespace, topic) == expected


@pytest.mark.parametrize(
    ("namespace", "topic"),
    [
        ("/", "odom"),
        ("b", "odom"),
        ("/b", "/odom"),
        ("/b", "../odom"),
    ],
)
def test_root_or_escaping_topics_fail_closed(namespace: str, topic: str) -> None:
    with pytest.raises(LiveFrontierCaptureError) as raised:
        resolve_lane_topic(namespace, topic)
    assert raised.value.code == "INVALID_NAMESPACE"


def test_checked_in_config_freezes_algorithm_and_zero_authority() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/internnav_t5/live_frontier_capture.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["extraction"] == asdict(FrontierCaptureConfig())
    assert config["topics"] == {
        "metadata": "internvla/observation/metadata",
        "odometry": "odom",
        "local_costmap": "local_costmap/costmap",
        "global_costmap": "global_costmap/costmap",
    }
    assert config["authority"] == {
        "capture_only": True,
        "ros_publishers": 0,
        "model_requests": False,
        "navigation_goal": False,
        "cmd_vel": False,
        "terminal_stop": False,
    }
    assert all(value is False for value in config["oracle_inputs"].values())


def test_ros_wrapper_is_subscription_only_and_lane_scoped() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/t5_live_frontier_snapshot_node.py").read_text(
        encoding="utf-8"
    )
    assert "create_subscription" in source
    assert "create_publisher" not in source
    assert "ActionClient" not in source
    assert "resolve_lane_topic" in source
    assert '"/tf:=tf"' in source
    assert '"/tf_static:=tf_static"' in source
    assert '"use_sim_time:=true"' in source
