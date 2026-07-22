from __future__ import annotations

import json
import math

from omninav_nav2.adapter import (
    AdapterConfig,
    ExecutionState,
    FastWaypoint,
    GridCostmapChecker,
    MapBounds,
    OmniNavNav2Adapter,
    Pose2D,
)


def checker(data=None):
    return GridCostmapChecker(
        width=7,
        height=7,
        resolution_m=1.0,
        origin_x=-3.0,
        origin_y=-3.0,
        data=[0] * 49 if data is None else data,
    )


def waypoint(now=10.0, sequence=1, subgoal="frontier-1", heading=(0.0, 1.0), xy=(1.0, 0.0)):
    return FastWaypoint.from_action_head(
        episode_id="episode-1",
        subgoal_id=subgoal,
        sequence_id=sequence,
        stamp_s=now,
        waypoint_xy_m=xy,
        heading_sin_cos=heading,
    )


def test_action_head_uses_norm_and_heading_then_transforms_to_map():
    wp = waypoint(heading=(1.0, 0.0), xy=(0.6, 0.8))
    assert math.isclose(wp.forward_m, 0.0, abs_tol=1e-9)
    assert math.isclose(wp.left_m, 1.0)
    pose = Pose2D(2.0, 3.0, math.pi / 2.0, 10.0)
    goal = OmniNavNav2Adapter.transform_to_map(pose, wp)
    assert math.isclose(goal.x, 1.0)
    assert math.isclose(goal.y, 3.0)
    assert math.isclose(goal.yaw, math.pi, abs_tol=1e-9)


def test_rejects_stale_non_monotonic_and_out_of_bounds():
    adapter = OmniNavNav2Adapter(
        AdapterConfig(bounds=MapBounds(-2, 2, -2, 2)), checker(), clock=lambda: 10.0
    )
    pose = Pose2D(0, 0, 0, 10.0)
    assert adapter.consider(waypoint(now=8.0), pose, ExecutionState()).reason == "stale_waypoint"
    first = adapter.consider(waypoint(sequence=2), pose, ExecutionState())
    assert first.send_goal
    assert adapter.consider(waypoint(sequence=2), pose, ExecutionState()).reason == "non_monotonic_sequence"
    far = waypoint(sequence=3, xy=(3.1, 0.0))
    assert adapter.consider(far, pose, ExecutionState()).reason == "waypoint_too_far"


def test_costmap_checks_connectivity_not_only_goal_cell():
    data = [0] * 49
    for row in range(7):
        data[row * 7 + 3] = 100
    grid = checker(data)
    result = grid.check(Pose2D(-1, 0, 0, 0), Pose2D(1, 0, 0, 0))
    assert result.reachable is False
    assert result.reason == "no_costmap_path"


def test_only_resends_for_reached_invalid_path_or_changed_subgoal(tmp_path):
    log = tmp_path / "adapter.jsonl"
    adapter = OmniNavNav2Adapter(AdapterConfig(), checker(), audit_log=log, clock=lambda: 10.0)
    pose = Pose2D(0, 0, 0, 10.0)
    initial = adapter.consider(waypoint(sequence=1), pose, ExecutionState())
    assert initial.send_goal and initial.update_reason == "initial_goal"
    held = adapter.consider(waypoint(sequence=2), pose, ExecutionState(path_valid=True))
    assert held.accepted and not held.send_goal and held.reason == "active_goal_held"
    invalid = adapter.consider(waypoint(sequence=3), pose, ExecutionState(path_valid=False))
    assert invalid.send_goal and invalid.update_reason == "path_invalid"
    changed = adapter.consider(
        waypoint(sequence=4, subgoal="frontier-2"), pose, ExecutionState(path_valid=True)
    )
    assert changed.send_goal and changed.update_reason == "subgoal_changed"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["original_waypoint"]["raw_xy_m"] == [1.0, 0.0]
    assert rows[0]["transformed_goal"]["frame_id"] == "map"


def test_semantic_direction_is_rejected_not_rotated():
    adapter = OmniNavNav2Adapter(
        AdapterConfig(max_semantic_deviation_deg=30), checker(), clock=lambda: 10.0
    )
    result = adapter.consider(
        waypoint(heading=(1.0, 0.0)),
        Pose2D(0, 0, 0, 10.0),
        ExecutionState(),
        semantic_bearing_rad=0.0,
    )
    assert result.reason == "semantic_direction_violation"
