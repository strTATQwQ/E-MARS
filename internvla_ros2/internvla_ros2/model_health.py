"""Pure validation for a freshly preloaded DGX model service."""

from __future__ import annotations

from typing import Any, Mapping

from .identity import CHECKPOINT_REVISION, MODEL_REVISION


EXPECTED_UNINITIALIZED_MODEL_HEALTH = {
    "status_code": 0,
    "initialized": False,
    "lifecycle_state": 0,
    "episode_id": "",
    "reset_generation": 0,
    "last_sequence_id": 0,
    "model_revision": MODEL_REVISION,
    "checkpoint_revision": CHECKPOINT_REVISION,
}


def validate_uninitialized_model_health(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized snapshot or reject a stale/wrong model service."""

    observed = {
        "status_code": int(value.get("status_code", -1)),
        "initialized": value.get("initialized"),
        "lifecycle_state": int(value.get("lifecycle_state", -1)),
        "episode_id": str(value.get("episode_id", "")),
        "reset_generation": int(value.get("reset_generation", -1)),
        "last_sequence_id": int(value.get("last_sequence_id", -1)),
        "model_revision": str(value.get("model_revision", "")),
        "checkpoint_revision": str(value.get("checkpoint_revision", "")),
    }
    if observed != EXPECTED_UNINITIALIZED_MODEL_HEALTH:
        mismatches = {
            key: {
                "expected": expected,
                "observed": observed.get(key),
            }
            for key, expected in EXPECTED_UNINITIALIZED_MODEL_HEALTH.items()
            if observed.get(key) != expected
        }
        raise ValueError(f"DGX model health identity mismatch: {mismatches}")
    return {
        "schema_version": 1,
        "status": "PASS",
        **observed,
        "status_message": str(value.get("status_message", "")),
    }
