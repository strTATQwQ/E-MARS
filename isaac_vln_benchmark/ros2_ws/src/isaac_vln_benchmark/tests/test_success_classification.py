from isaac_vln_benchmark.metrics import classify_success


def test_clean_success_classification():
    result = classify_success(reached=True)
    assert result["success"] is True
    assert result["clean_success"] is True
    assert result["recovered_success"] is False
    assert result["success_class"] == "clean_success"


def test_recovered_success_classification():
    result = classify_success(reached=True, recovery_override_count=2)
    assert result["success"] is True
    assert result["clean_success"] is False
    assert result["recovered_success"] is True
    assert result["success_class"] == "recovered_success"


def test_timeout_failure_classification():
    result = classify_success(reached=False, failure_reason="timeout")
    assert result["success"] is False
    assert result["success_class"] == "failure"
    assert result["failure_reason"] == "timeout"
