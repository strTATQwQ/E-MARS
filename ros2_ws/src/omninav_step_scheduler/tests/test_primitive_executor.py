from types import SimpleNamespace

from omninav_step_scheduler.primitive_executor_node import (
    blocked_turn_lock_action,
    is_escape_twist,
    lane_centering_yaw_rate,
    low_speed_turn_command,
    motion_blocked_by_safety,
)


def test_lane_centering_turns_back_from_negative_y_drift():
    rate = lane_centering_yaw_rate([1.5, -1.0, -0.77], max_yaw_rate=0.28)
    assert rate == 0.28


def test_lane_centering_is_small_near_centerline():
    rate = lane_centering_yaw_rate([1.5, 0.0, 0.02], max_yaw_rate=0.28)
    assert -0.02 < rate < 0.0


def test_local_obstacle_blocks_forward_but_allows_escape_turns():
    blocked_safety = SimpleNamespace(local_costmap_clear=False, estop=False, robot_fallen_or_unstable=False)
    forward = _twist(linear_x=0.2)
    turn = _twist(angular_z=0.2)
    back = _twist(linear_x=-0.1)

    assert not is_escape_twist(forward)
    assert is_escape_twist(turn)
    assert is_escape_twist(back)
    assert motion_blocked_by_safety(forward, blocked_safety)
    assert not motion_blocked_by_safety(turn, blocked_safety)
    assert not motion_blocked_by_safety(back, blocked_safety)


def test_low_speed_turn_uses_bounded_crawl_only_on_clear_path():
    assert low_speed_turn_command(1.0, local_costmap_clear=True) == (0.08, 0.20)
    assert low_speed_turn_command(-1.0, local_costmap_clear=True) == (0.08, -0.20)
    assert low_speed_turn_command(1.0, local_costmap_clear=False) == (-0.08, 0.20)


def test_blocked_turn_lock_prevents_rapid_direction_thrash():
    kind, direction, locked_until, active = blocked_turn_lock_action(
        "turn_left",
        local_costmap_clear=False,
        lock_direction=0.0,
        locked_until=0.0,
        timestamp=10.0,
    )
    assert (kind, direction, locked_until, active) == ("turn_left", 1.0, 14.0, True)

    kind, direction, locked_until, active = blocked_turn_lock_action(
        "turn_right",
        local_costmap_clear=False,
        lock_direction=direction,
        locked_until=locked_until,
        timestamp=10.2,
    )
    assert (kind, direction, locked_until, active) == ("turn_left", 1.0, 14.0, True)

    kind, direction, locked_until, active = blocked_turn_lock_action(
        "move_forward",
        local_costmap_clear=False,
        lock_direction=direction,
        locked_until=locked_until,
        timestamp=11.0,
    )
    assert (kind, direction, locked_until, active) == ("turn_left", 1.0, 14.0, True)


def test_blocked_turn_lock_never_overrides_stop_and_releases_when_clear():
    assert blocked_turn_lock_action(
        "stop",
        local_costmap_clear=False,
        lock_direction=-1.0,
        locked_until=12.5,
        timestamp=11.0,
    ) == ("stop", -1.0, 12.5, True)
    assert blocked_turn_lock_action(
        "move_forward",
        local_costmap_clear=True,
        lock_direction=-1.0,
        locked_until=12.5,
        timestamp=11.0,
    ) == ("turn_right", -1.0, 12.5, True)
    assert blocked_turn_lock_action(
        "move_forward",
        local_costmap_clear=True,
        lock_direction=-1.0,
        locked_until=12.5,
        timestamp=13.0,
    ) == ("move_forward", 0.0, 0.0, False)


def test_blocked_forward_starts_deterministic_pure_turn_escape():
    assert blocked_turn_lock_action(
        "move_forward",
        local_costmap_clear=False,
        lock_direction=0.0,
        locked_until=0.0,
        timestamp=20.0,
        escape_direction=-1.0,
    ) == ("turn_right", -1.0, 24.0, True)


def _twist(*, linear_x: float = 0.0, linear_y: float = 0.0, angular_z: float = 0.0):
    return SimpleNamespace(
        linear=SimpleNamespace(x=linear_x, y=linear_y, z=0.0),
        angular=SimpleNamespace(x=0.0, y=0.0, z=angular_z),
    )
