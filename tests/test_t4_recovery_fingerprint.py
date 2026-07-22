from __future__ import annotations

import math

import pytest

from t4_completion.recovery import trajectory_signature


def signature(points: list[tuple[float, float]]):
    return trajectory_signature(points, resolution_m=0.05, maximum_points=64)


def test_signature_is_stable_across_density_and_translation() -> None:
    sparse = signature([(0.0, 0.0), (1.0, 0.0)])
    dense_shifted = signature(
        [(5.01, -2.01), (5.25, -2.0), (5.5, -2.0), (5.75, -2.0), (6.0, -2.0)]
    )
    assert sparse.shape_sha256 == dense_shifted.shape_sha256
    assert sparse.absolute_sha256 != dense_shifted.absolute_sha256
    assert sparse.length_m == pytest.approx(1.0)


def test_different_route_shape_has_a_different_shape_signature() -> None:
    straight = signature([(0.0, 0.0), (1.0, 0.0)])
    turn = signature([(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)])
    assert straight.shape_sha256 != turn.shape_sha256


@pytest.mark.parametrize(
    "points",
    [
        [],
        [(0.0, 0.0)],
        [(0.0, 0.0), (0.001, 0.0)],
        [(0.0, 0.0), (math.nan, 1.0)],
        [(0.0, 0.0), (math.inf, 1.0)],
        [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0)],
    ],
)
def test_invalid_or_degenerate_trajectory_is_rejected(points) -> None:
    with pytest.raises(ValueError):
        signature(points)


def test_trajectory_point_count_is_hard_bounded() -> None:
    with pytest.raises(ValueError, match="maximum_points"):
        trajectory_signature(
            [(float(index), 0.0) for index in range(5)],
            resolution_m=0.05,
            maximum_points=4,
        )
