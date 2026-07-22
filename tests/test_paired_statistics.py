import pytest

from scripts.paired_slow_benchmark_statistics import paired_bootstrap


def test_paired_bootstrap_is_deterministic_and_preserves_constant_difference():
    first = paired_bootstrap([2.0, 2.0, 2.0], samples=100, seed=7)
    second = paired_bootstrap([2.0, 2.0, 2.0], samples=100, seed=7)
    assert first == second
    assert first["mean_difference"] == pytest.approx(2.0)
    assert first["ci95"] == pytest.approx([2.0, 2.0])
