"""Runtime-facing validation for one frozen T4.6 ablation arm.

This module is deliberately offline and standard-library-only.  It turns the
reviewed generated config into a small, exact factor vector for coordinator
launchers; it never starts ROS, Isaac, DGX, SSH, or a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import (
    ContractError,
    RUNTIME_PROFILES,
    canonical_sha256,
    canonical_text_file_sha256,
    load_json,
    load_matrix,
    validate_variant_config,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / "configs/completion_sim/ablation/frozen_matrix_v1.json"
FACTOR_ORDER = (
    "system_mode",
    "trajectory_mode",
    "termination_mode",
    "history_mode",
    "recovery_mode",
    "view_mode",
)
FIELD_ORDER = (
    "variant_id",
    "config_sha256",
    "matrix_sha256",
    *FACTOR_ORDER,
    "completion_sim_profile_sha256",
    "strict_evidence_profile_sha256",
)


def load_runtime_variant(
    config_path: Path,
    *,
    matrix_path: Path = DEFAULT_MATRIX,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Validate one generated arm and return its exact launch-time fields."""

    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    matrix_path = matrix_path.resolve()
    if not config_path.is_file():
        raise ContractError(f"variant config is not a file: {config_path}")
    if not matrix_path.is_file():
        raise ContractError(f"matrix is not a file: {matrix_path}")

    matrix = load_matrix(matrix_path)
    config = load_json(config_path)
    if not isinstance(config, dict) or not isinstance(config.get("variant_id"), str):
        raise ContractError("variant config lacks a string variant_id")
    validate_variant_config(matrix, config["variant_id"], config)

    profile_hashes: dict[str, str] = {}
    for name, frozen in RUNTIME_PROFILES.items():
        path = repository_root / frozen["path"]
        actual = canonical_text_file_sha256(path)
        if actual != frozen["sha256"]:
            raise ContractError(f"runtime profile content changed: {name}")
        profile_hashes[name] = actual

    factors = config["factors"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNTIME_CONFIG_VALID",
        "runtime_policy": "completion_sim",
        "runtime_target": "isaac_simulation",
        "variant_id": config["variant_id"],
        "config_sha256": canonical_sha256(config),
        "matrix_sha256": canonical_sha256(matrix),
        "factors": {name: factors[name] for name in FACTOR_ORDER},
        "runtime_profile_sha256": profile_hashes,
        "resource_use": "none",
    }
    return result


def ordered_fields(summary: dict[str, Any]) -> tuple[str, ...]:
    """Return newline-safe scalar fields consumed by the Bash launcher."""

    if summary.get("status") != "RUNTIME_CONFIG_VALID":
        raise ContractError("runtime summary is not valid")
    factors = summary.get("factors")
    profiles = summary.get("runtime_profile_sha256")
    if not isinstance(factors, dict) or not isinstance(profiles, dict):
        raise ContractError("runtime summary factors or profiles are malformed")
    values = (
        summary.get("variant_id"),
        summary.get("config_sha256"),
        summary.get("matrix_sha256"),
        *(factors.get(name) for name in FACTOR_ORDER),
        profiles.get("completion_sim"),
        profiles.get("strict_evidence"),
    )
    if any(
        not isinstance(value, str)
        or not value
        or "\n" in value
        or "\r" in value
        for value in values
    ):
        raise ContractError("runtime field is not a nonempty newline-safe string")
    return tuple(values)
