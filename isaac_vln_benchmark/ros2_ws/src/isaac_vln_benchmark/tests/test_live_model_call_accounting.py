from isaac_vln_benchmark.benchmark_runner import LiveRosBenchmarkRunner


def test_step_http_metrics_event_is_accounted_in_milliseconds():
    call = LiveRosBenchmarkRunner._model_call_from_payload(
        "step",
        {
            "request_id": "step-1",
            "result": "accepted",
            "latency_s": 4.17093,
            "multimodal": True,
        },
    )
    assert call["model"] == "step"
    assert call["latency_ms"] == 4170.93
    assert call["multimodal"] is True


def test_model_call_deduplicates_same_request_across_topics():
    runner = object.__new__(LiveRosBenchmarkRunner)
    runner.model_calls = []
    call = {"model": "step", "request_id": "step-1", "latency_ms": 1000.0}
    runner._record_model_call(call)
    runner._record_model_call(call | {"latency_ms": 2000.0})
    assert runner.model_calls == [call]
