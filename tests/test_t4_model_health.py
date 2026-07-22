from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "internvla_ros2"))

from internvla_ros2.identity import CHECKPOINT_REVISION, MODEL_REVISION  # noqa: E402
from internvla_ros2.model_health import validate_uninitialized_model_health  # noqa: E402


def snapshot() -> dict[str, object]:
    return {
        "status_code": 0,
        "status_message": "model preloaded; awaiting initialize",
        "initialized": False,
        "lifecycle_state": 0,
        "episode_id": "",
        "reset_generation": 0,
        "last_sequence_id": 0,
        "model_revision": MODEL_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
    }


def test_model_health_requires_preloaded_episode_free_frozen_identity() -> None:
    payload = validate_uninitialized_model_health(snapshot())
    assert payload["status"] == "PASS"
    assert payload["model_revision"] == MODEL_REVISION

    for field, bad in (
        ("initialized", True),
        ("lifecycle_state", 2),
        ("episode_id", "stale-episode"),
        ("reset_generation", 1),
        ("model_revision", "wrong"),
        ("checkpoint_revision", "wrong"),
    ):
        value = snapshot()
        value[field] = bad
        with pytest.raises(ValueError, match="model health identity mismatch"):
            validate_uninitialized_model_health(value)


def test_functional_model_launcher_requires_combined_lease_and_fresh_result() -> None:
    launcher = (ROOT / "scripts" / "run_t4_model_server.sh").read_text(
        encoding="utf-8"
    )
    client = (
        ROOT / "internvla_ros2" / "internvla_ros2" / "client_node.py"
    ).read_text(encoding="utf-8")
    probe = (ROOT / "scripts" / "t4_model_health_probe.py").read_text(
        encoding="utf-8"
    )
    assert "INTERNVLA_T4_FUNCTIONAL_MODEL" in launcher
    assert 'INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" = "dgx+isaac"' in launcher
    assert 'test ! -e "$RESULT_DIR"' in launcher
    assert "validate_uninitialized_model_health" in client
    assert '"model_health_start": self.model_health_start' in client
    assert '"/internvla/health"' in probe
    assert '"/internvla/step"' in probe
