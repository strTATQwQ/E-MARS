from omninav_step_scheduler.stale_gate import clock_domains_comparable


def test_wall_and_ros_system_are_comparable():
    assert clock_domains_comparable("wall", "ros_system")


def test_ros_sim_is_not_comparable_to_wall():
    assert not clock_domains_comparable("ros_sim", "wall")


def test_unknown_clock_domain_is_not_comparable():
    assert not clock_domains_comparable("unknown", "wall")
