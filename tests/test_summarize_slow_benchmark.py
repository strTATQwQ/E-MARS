import pytest

from scripts.summarize_slow_benchmark_run import distribution, percentile


def test_percentile_interpolates_and_distribution_counts():
    assert percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)
    result = distribution([1.0, 2.0, 3.0])
    assert result["count"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert result["p50"] == pytest.approx(2.0)
