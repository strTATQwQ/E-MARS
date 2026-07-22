import math

import pytest

from slow_benchmark.control import heading_head_step_vector


def test_heading_head_controls_direction_and_waypoint_controls_radius():
    forward, left, heading = heading_head_step_vector((0.3, 0.4), (1.0, 0.0))
    assert heading == pytest.approx(math.pi / 2.0)
    assert forward == pytest.approx(0.0, abs=1.0e-9)
    assert left == pytest.approx(0.5)
