import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from omninav_cosmos.action_head import ActionHeadConfig, NavigationActionHead
from omninav_cosmos.backbones.cosmos_qwen3vl import CosmosQwen3VLAdapter


class FakeModel:
    class Base:
        rope_deltas = object()

    model = Base()


def test_untrained_action_head_is_rejected_by_default():
    head = NavigationActionHead(ActionHeadConfig(hidden_size=8, attention_heads=2))
    with pytest.raises(RuntimeError, match="trained Cosmos"):
        CosmosQwen3VLAdapter(
            model=FakeModel(),
            processor=object(),
            action_head=head,
            model_fingerprint="fingerprint",
        )


def test_explicit_interface_smoke_can_use_deterministic_untrained_head():
    head = NavigationActionHead(ActionHeadConfig(hidden_size=8, attention_heads=2))
    adapter = CosmosQwen3VLAdapter(
        model=FakeModel(),
        processor=object(),
        action_head=head,
        model_fingerprint="fingerprint",
        model_variant="qwen3-vl-dev",
        allow_untrained_action_head=True,
    )
    adapter.reset_episode("ep-a")
    assert adapter.health()["action_head_trained"] is False
    assert adapter.health()["model_variant"] == "qwen3-vl-dev"
    assert FakeModel.model.rope_deltas is None
