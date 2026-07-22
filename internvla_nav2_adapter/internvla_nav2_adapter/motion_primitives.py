"""Pure geometry for T5/T4 System2 navigation primitives."""

from __future__ import annotations

import math


ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3


def system2_local_poses(
    action: int,
    *,
    t5_completion_sim: bool,
    forward_step_m: float,
    sample_count: int = 9,
) -> list[tuple[float, float, float]]:
    """Return local ``(x, y, yaw_delta)`` samples for one System2 action."""

    if action not in {ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT}:
        raise ValueError("unsupported System2 action")
    if not math.isfinite(forward_step_m) or forward_step_m <= 0.0:
        raise ValueError("forward step must be finite and positive")
    if sample_count < 2:
        raise ValueError("System2 primitive requires at least two samples")
    samples = [index / float(sample_count - 1) for index in range(sample_count)]
    if action == ACTION_FORWARD:
        return [(forward_step_m * value, 0.0, 0.0) for value in samples]

    sign = 1.0 if action == ACTION_LEFT else -1.0
    total = sign * math.radians(15.0)
    if t5_completion_sim:
        return [(0.0, 0.0, total * value) for value in samples]

    radius = 0.45
    output = []
    for value in samples:
        theta = total * value
        output.append(
            (
                radius * math.sin(abs(theta)),
                sign * radius * (1.0 - math.cos(theta)),
                theta,
            )
        )
    return output
