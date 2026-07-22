from __future__ import annotations

import json
from pathlib import Path

import pytest

from omninav_cosmos.backtest import (
    BacktestValidationError,
    ExpectedModel,
    REQUIRED_EPISODE_FIELDS,
    manifest_episodes,
    validate_episode_records,
    validate_seed_manifest,
    validate_service_health,
)


ROOT = Path(__file__).resolve().parents[2]


def test_committed_seed_manifest_meets_formal_minimums():
    manifest = json.loads((ROOT / "configs" / "isaac" / "seeds.json").read_text(encoding="utf-8"))
    summary = validate_seed_manifest(manifest)
    assert summary["smoke_episodes"] == 5
    assert summary["formal_episodes"] == 130
    assert all(value == 20 for value in summary["obstacle_counts"].values())
    assert summary["semantic_episodes"] == 30
    assert summary["complex_semantic_episodes"] == 20


def test_service_health_rejects_untrained_development_head():
    expected = ExpectedModel("cosmos-reason2-8b-omninav", "bf16")
    with pytest.raises(BacktestValidationError, match="action_head_trained"):
        validate_service_health(
            {
                "ready": True,
                "model_variant": "qwen3-vl-8b-dev-untrained-action-head",
                "precision_mode": "bf16",
                "action_head_trained": False,
            },
            expected,
        )


def test_episode_validator_requires_real_headless_camera_and_all_metrics():
    manifest = json.loads((ROOT / "configs" / "isaac" / "seeds.json").read_text(encoding="utf-8"))
    expected_row = manifest_episodes(manifest, "smoke")[0]
    record = {field: 0 for field in REQUIRED_EPISODE_FIELDS}
    record.update(
        episode_key=expected_row["episode_key"],
        random_seed=expected_row["random_seed"],
        mock=False,
        use_isaac=True,
        live_closed_loop=True,
        isaac_headless=True,
        camera_source="real_isaac_render_product_udp_15012",
        action_head_trained=True,
        fallback_count=0,
        num_omninav_real_image_calls=3,
        model_variant="qwen2.5-vl-omninav-legacy",
        precision_mode="bf16",
    )
    summary = validate_episode_records(
        [record],
        [expected_row],
        ExpectedModel("qwen2.5-vl-omninav-legacy", "bf16"),
    )
    assert summary["episodes"] == 1
    bad = dict(record, camera_source="synthetic_adapter_camera")
    with pytest.raises(BacktestValidationError, match="camera source"):
        validate_episode_records(
            [bad],
            [expected_row],
            ExpectedModel("qwen2.5-vl-omninav-legacy", "bf16"),
        )
