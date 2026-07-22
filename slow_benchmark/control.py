from __future__ import annotations

import math
from typing import Sequence


def heading_head_step_vector(
    waypoint: Sequence[float], heading_sin_cos: Sequence[float]
) -> tuple[float, float, float]:
    """Reproduce OmniNav's first-step radius plus heading-head direction.

    Returns local (forward, left, heading_radians).  The waypoint is already
    rescaled to metric units by the adapter; upstream uses only its norm for
    distance and obtains direction independently from atan2(sin, cos).
    """
    if len(waypoint) != 2 or len(heading_sin_cos) != 2:
        raise ValueError("waypoint and heading_sin_cos must each contain two values")
    radius = math.hypot(float(waypoint[0]), float(waypoint[1]))
    heading = math.atan2(float(heading_sin_cos[0]), float(heading_sin_cos[1]))
    return radius * math.cos(heading), radius * math.sin(heading), heading
