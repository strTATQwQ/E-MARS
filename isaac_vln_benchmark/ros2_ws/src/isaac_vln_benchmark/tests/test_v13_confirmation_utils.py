from isaac_vln_benchmark.v13_confirmation_utils import paired_bootstrap_ci


def test_paired_bootstrap_ci_excludes_zero_for_consistent_gain():
    result = paired_bootstrap_ci([1.0] * 12 + [0.0] * 33, samples=5000, seed=7)
    assert result["pairs"] == 45
    assert result["mean"] > 0
    assert result["ci95_lower"] > 0


def test_paired_bootstrap_ci_includes_zero_for_weak_gain():
    result = paired_bootstrap_ci([1.0] + [0.0] * 44, samples=5000, seed=7)
    assert result["mean"] > 0
    assert result["ci95_lower"] == 0
