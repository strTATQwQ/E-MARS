from __future__ import annotations

import inspect
import json
import math

import pytest

from slow_planner.lane_b import LaneBSnapshotIdentity
from slow_planner.live_frontier_capture import OccupancyGrid2D, Pose2D
from slow_planner.reachable_viewpoint import (
    REACHABLE_VIEWPOINT_CANDIDATE_TYPE,
    REACHABLE_VIEWPOINT_KIND,
    REACHABLE_VIEWPOINT_SOURCE,
    SLOW_PLANNER_V1_BLOCKER,
    ReachableViewpoint,
    ReachableViewpointConfig,
    ReachableViewpointError,
    build_reachable_viewpoint_mapping,
    extract_reachable_viewpoints,
    slow_planner_v1_compatibility,
    validate_reachable_viewpoint_mapping,
)


def grid(
    rows: list[str],
    *,
    resolution: float = 0.5,
    frame_id: str = "map",
    origin_x_m: float = 0.0,
    origin_y_m: float = 0.0,
    origin_yaw_rad: float = 0.0,
) -> OccupancyGrid2D:
    values = {"#": 100, ".": 0, "?": -1}
    return OccupancyGrid2D(
        width=len(rows[0]),
        height=len(rows),
        resolution_m=resolution,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        origin_yaw_rad=origin_yaw_rad,
        frame_id=frame_id,
        stamp_sim_time_s=10.0,
        data=tuple(values[cell] for row in rows for cell in row),
    )


def known_room() -> OccupancyGrid2D:
    return grid(
        [
            "###############",
            "#.............#",
            "#.............#",
            "#.............#",
            "#.............#",
            "#.............#",
            "#.............#",
            "#.............#",
            "###############",
        ]
    )


def test_fully_known_free_map_yields_diverse_reachable_viewpoints() -> None:
    result = extract_reachable_viewpoints(
        known_room(),
        robot_pose_in_grid_frame=Pose2D(3.75, 2.25, 0.0),
    )
    assert len(result) >= 6
    assert [item.candidate_id for item in result] == list(range(len(result)))
    assert all(item.candidate_type == REACHABLE_VIEWPOINT_CANDIDATE_TYPE for item in result)
    assert all(item.source == REACHABLE_VIEWPOINT_SOURCE for item in result)
    assert all(0.75 <= item.distance_m <= 5.0 for item in result)
    assert len({round(item.bearing_deg / 45.0) for item in result}) >= 6


def test_obstacles_and_unknown_are_never_crossed_or_selected() -> None:
    separated = grid(
        [
            "#############",
            "#.....#.....#",
            "#.....#..?..#",
            "#.....#.....#",
            "#############",
        ]
    )
    result = extract_reachable_viewpoints(
        separated,
        robot_pose_in_grid_frame=Pose2D(1.25, 1.25, 0.0),
        config=ReachableViewpointConfig(
            minimum_distance_m=0.5,
            preferred_distance_m=1.5,
            maximum_distance_m=3.0,
        ),
    )
    assert result
    assert all(candidate.grid_xy[0] < 3.0 for candidate in result)
    assert all(separated.value(separated.world_to_cell(*candidate.grid_xy)) == 0 for candidate in result)


def test_nonidentity_odom_grid_coordinates_keep_explicit_frame() -> None:
    odom_grid = grid(
        [
            "#########",
            "#.......#",
            "#.......#",
            "#.......#",
            "#########",
        ],
        frame_id="odom",
        origin_x_m=10.0,
        origin_y_m=-4.0,
        origin_yaw_rad=math.pi / 2.0,
    )
    pose_xy = odom_grid.cell_to_world((2, 4))
    candidates = extract_reachable_viewpoints(
        odom_grid,
        robot_pose_in_grid_frame=Pose2D(*pose_xy, math.pi / 2.0),
        config=ReachableViewpointConfig(
            minimum_distance_m=0.5,
            preferred_distance_m=1.0,
            maximum_distance_m=2.0,
        ),
    )
    assert candidates
    assert all(candidate.candidate_xy_frame == "odom" for candidate in candidates)
    assert all(
        odom_grid.value(odom_grid.world_to_cell(*candidate.grid_xy)) == 0
        for candidate in candidates
    )

    payload = build_reachable_viewpoint_mapping(
        source_node="/t5/lane_b/reachable_viewpoint_source",
        identity=LaneBSnapshotIdentity("628", 0, 13),
        ros_episode_id="b::628",
        captured_sim_time_s=10.0,
        valid_until_sim_time_s=12.5,
        robot_pose_in_map_frame=Pose2D(1.0, 2.0, 0.25),
        candidates=candidates,
    )
    assert payload["grid_frame_id"] == "odom"
    assert all(row["candidate_xy_frame"] == "odom" for row in payload["candidates"])
    assert all("map_xy" not in row for row in payload["candidates"])
    validate_reachable_viewpoint_mapping(payload)


def test_diagonal_corner_cut_is_not_reachable() -> None:
    corner = grid(["#####", "#.###", "##..#", "#####"])
    with pytest.raises(ReachableViewpointError) as captured:
        extract_reachable_viewpoints(
            corner,
            robot_pose_in_grid_frame=Pose2D(0.75, 0.75, 0.0),
            config=ReachableViewpointConfig(
                minimum_distance_m=0.25,
                preferred_distance_m=0.5,
                maximum_distance_m=2.0,
            ),
        )
    assert captured.value.code == "NO_REACHABLE_VIEWPOINT_CANDIDATES"


def test_extractor_accepts_no_oracle_instruction_goal_or_model() -> None:
    parameters = set(inspect.signature(extract_reachable_viewpoints).parameters)
    assert parameters == {"grid", "robot_pose_in_grid_frame", "config"}
    assert not parameters.intersection({"goal", "instruction", "reference_path", "success", "model"})


def test_private_mapping_names_viewpoints_and_refuses_v1_frontier_conversion() -> None:
    candidates = extract_reachable_viewpoints(
        known_room(), robot_pose_in_grid_frame=Pose2D(3.75, 2.25, 0.0)
    )
    payload = build_reachable_viewpoint_mapping(
        source_node="/t5/lane_b/reachable_viewpoint_source",
        identity=LaneBSnapshotIdentity("628", 0, 13),
        ros_episode_id="b::628",
        captured_sim_time_s=10.0,
        valid_until_sim_time_s=12.5,
        robot_pose_in_map_frame=Pose2D(1.0, 2.0, 0.25),
        candidates=candidates,
    )
    validate_reachable_viewpoint_mapping(payload)
    assert payload["kind"] == REACHABLE_VIEWPOINT_KIND
    assert payload["candidate_type"] == "reachable_free_space_viewpoint"
    assert "candidate_frontiers" not in payload
    assert all("frontier_id" not in item for item in payload["candidates"])
    assert not any(payload["authority"].values())
    assert not any(payload["oracle_inputs"].values())
    compatibility = payload["slow_planner_v1_compatibility"]
    assert compatibility["status"] == "BLOCKED"
    assert compatibility["blocker_code"] == SLOW_PLANNER_V1_BLOCKER
    assert compatibility["conversion_to_candidate_frontiers_allowed"] is False

    changed = json.loads(json.dumps(payload))
    changed["candidates"][0]["grid_xy"][0] += 0.1
    with pytest.raises(ReachableViewpointError) as tampered:
        validate_reachable_viewpoint_mapping(changed)
    assert tampered.value.code == "INVALID_VIEWPOINT_SOURCE"


def test_frozen_v1_compatibility_receipt_is_deterministic() -> None:
    first = slow_planner_v1_compatibility()
    second = slow_planner_v1_compatibility()
    assert first == second
    assert first["protocol_version"] == 1
    assert first["online_control_allowed"] is False


@pytest.mark.parametrize("bad_id", [True, False, 1.0, "1", None])
def test_candidate_id_rejects_bool_and_every_non_integer(bad_id: object) -> None:
    with pytest.raises(ReachableViewpointError) as captured:
        ReachableViewpoint(
            candidate_id=bad_id,  # type: ignore[arg-type]
            relative_xz=(0.0, 1.0),
            grid_xy=(1.0, 2.0),
            distance_m=1.0,
            geodesic_distance_m=1.0,
            bearing_deg=0.0,
            candidate_xy_frame="map",
        )
    assert captured.value.code == "INVALID_VIEWPOINT_CANDIDATE"
