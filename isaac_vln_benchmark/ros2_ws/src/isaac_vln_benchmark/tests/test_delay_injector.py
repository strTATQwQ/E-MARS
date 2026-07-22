from isaac_vln_benchmark.metrics import DelayInjectorCore
from isaac_vln_benchmark.delay_injector_node import delay_audit_payload


def test_delay_injector_delays_and_releases():
    injector = DelayInjectorCore(delay_sec=1.0, drop_rate=0.0, seed=1)
    assert injector.enqueue("/topic", {"x": 1}, now=0.0)
    assert injector.drain(0.5) == []
    ready = injector.drain(1.1)
    assert len(ready) == 1
    assert ready[0]["topic"] == "/topic"
    assert ready[0]["delay_sec"] >= 1.0


def test_delay_injector_drop():
    injector = DelayInjectorCore(delay_sec=0.0, drop_rate=1.0, seed=1)
    assert not injector.enqueue("/topic", {"x": 1}, now=0.0)
    assert injector.drain(1.0) == []
    assert injector.records[0]["dropped"]


def test_delay_audit_keeps_request_identity_without_model_payload():
    audit = delay_audit_payload(
        {
            "topic": "/raw",
            "message": '{"request_id":"step_7","episode_id":"ep1","role":"route_choice"}',
            "dropped": True,
            "delay_sec": None,
        },
        "packet_loss",
    )

    assert audit["request_id"] == "step_7"
    assert audit["episode_id"] == "ep1"
    assert "message" not in audit
