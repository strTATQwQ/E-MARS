import pytest


torch = pytest.importorskip("torch")

from omninav_cosmos.action_head import ActionHeadConfig, NavigationActionHead


def test_action_head_shapes_and_waypoint_cumsum():
    config = ActionHeadConfig(hidden_size=16, waypoint_count=5, attention_heads=4, arrive_count=5)
    head = NavigationActionHead(config)
    hidden = torch.randn(2, 7, 16)
    output = head(hidden, torch.ones(2, 7, dtype=torch.bool))
    assert output.waypoints.shape == (2, 5, 2)
    assert output.heading_sin_cos.shape == (2, 5, 2)
    assert output.arrive_logits.shape == (2, 5)
    assert output.confidence.shape == (2,)
    assert torch.all((output.confidence >= 0) & (output.confidence <= 1))


def test_heading_normalization_is_optional_for_legacy_compatibility():
    head = NavigationActionHead(
        ActionHeadConfig(hidden_size=8, attention_heads=2, normalize_heading=True)
    )
    output = head(torch.randn(1, 3, 8))
    norms = torch.linalg.vector_norm(output.heading_sin_cos, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_legacy_loader_accepts_released_single_layer_keys():
    config = ActionHeadConfig(
        hidden_size=8,
        waypoint_count=5,
        attention_heads=2,
        arrive_count=5,
        predict_confidence=True,
    )
    source = NavigationActionHead(config)
    target = NavigationActionHead(config)
    legacy = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith("confidence_predictor")
    }
    missing, unexpected = target.load_legacy_state_dict(legacy)
    assert all(key.startswith("confidence_predictor") for key in missing)
    assert unexpected == []
    assert torch.equal(source.query_action, target.query_action)

