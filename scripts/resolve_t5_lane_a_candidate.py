#!/usr/bin/env python3
"""Resolve one preregistered Lane-A candidate without shell evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


MANIFEST_RELATIVE = Path("configs/internnav_t5/lane_a_candidates/manifest.json")
FAMILY_CONTRACT = (
    (
        "action_observation_recovery",
        "configs/internnav_t5/lane_a_candidates/action_observation_recovery.json",
        ("a0_action_gate_only", "a1_action_gate_recovery_a"),
        frozenset({"INTERNNAV_T5_CANDIDATE_PROFILE"}),
    ),
    (
        "camera_history_alignment",
        "configs/internnav_t5/lane_a_candidates/camera_history_alignment.json",
        ("b0_go2_history_off", "b1_go2_history_on"),
        frozenset({"INTERNVLA_T4_VIEW_MODE", "INTERNVLA_T4_HISTORY_MODE"}),
    ),
    (
        "trajectory_horizon_refresh",
        "configs/internnav_t5/lane_a_candidates/trajectory_horizon_refresh.json",
        (
            "c0_upstream_mean",
            "c1_depth_geometry_rerank",
            "c2_rerank_short_fresh",
        ),
        frozenset(
            {
                "INTERNVLA_T5_TRAJECTORY_RERANK",
                "INTERNVLA_T4_PROGRESS_HORIZON_SEC",
                "INTERNVLA_T4_REFRESH_DISTANCE_M",
                "INTERNVLA_T4_REFRESH_TIME_SEC",
                "INTERNVLA_T4_TRAJECTORY_VALIDITY_SEC",
            }
        ),
    ),
)
ALLOWED_VALUES = {
    "INTERNNAV_T5_CANDIDATE_PROFILE": frozenset({"baseline", "recovery_a"}),
    "INTERNVLA_T4_VIEW_MODE": frozenset({"go2_view"}),
    "INTERNVLA_T4_HISTORY_MODE": frozenset({"off", "on"}),
    "INTERNVLA_T5_TRAJECTORY_RERANK": frozenset({"0", "1"}),
    "INTERNVLA_T4_PROGRESS_HORIZON_SEC": frozenset({"4.0", "5.0"}),
    "INTERNVLA_T4_REFRESH_DISTANCE_M": frozenset({"0.20", "0.30"}),
    "INTERNVLA_T4_REFRESH_TIME_SEC": frozenset({"1.5", "2.0"}),
    "INTERNVLA_T4_TRAJECTORY_VALIDITY_SEC": frozenset({"3.0", "5.0"}),
}
COMPOSITE_RE = re.compile(r"^a[01](?:\+b[01](?:\+c[012])?)?$")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _code_bundle_sha256(root: Path, relative_paths: list[str]) -> str:
    """Hash the analyzer's framed path/content stream with canonical LF text."""
    digest = hashlib.sha256()
    for relative in relative_paths:
        unresolved = root / relative
        if unresolved.is_symlink():
            raise ValueError(f"code bundle path is not a regular file: {relative}")
        path = unresolved.resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"code bundle path escapes root: {relative}") from error
        if not path.is_file():
            raise ValueError(f"code bundle path is not a regular file: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        # Git archives and Linux deployments contain LF.  A Windows checkout
        # may materialize the same tracked text as CRLF; bind the semantic Git
        # content so local and remote resolution remain identical.
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256(value: object, label: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _seal_resolution(value: dict[str, Any]) -> dict[str, Any]:
    if "resolution_sha256" in value:
        raise ValueError("resolution payload is already sealed")
    result = dict(value)
    provenance = result["provenance"]
    result["canonical_binding"] = {
        "candidate_selector": result["candidate_selector"],
        "effective_candidate_profile": result["effective_candidate_profile"],
        "candidate_manifest": result["candidate_manifest"],
        "selected_configs": result.get("selected_configs", []),
        "registered_config_sha256_by_family": result[
            "registered_config_sha256_by_family"
        ],
        "code_bundle_sha256": provenance["code_bundle_sha256"],
        "fixed_episode_count": provenance["fixed_episode_count"],
        "fixed_episode_keys": provenance["fixed_episode_keys"],
        "predecessor_candidate_ids": provenance["predecessor_candidate_ids"],
    }
    result["resolution_sha256"] = _canonical_sha256(result)
    return result


def _safe_config_path(root: Path, relative: str, expected: str) -> Path:
    if relative != expected:
        raise ValueError(f"candidate config path drift: {relative!r}")
    unresolved = root / relative
    if unresolved.is_symlink():
        raise ValueError(f"candidate config must not be a symlink: {relative!r}")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"candidate config escapes root: {relative!r}") from error
    if not path.is_file():
        raise ValueError(f"candidate config must be a regular non-symlink: {relative!r}")
    return path


def _validate_override(key: object, value: object) -> tuple[str, str]:
    if not isinstance(key, str) or key not in ALLOWED_VALUES:
        raise ValueError(f"candidate override key is not allowed: {key!r}")
    if not isinstance(value, str) or value not in ALLOWED_VALUES[key]:
        raise ValueError(f"candidate override value is not allowed: {key}={value!r}")
    return key, value


def _load_registry(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    unresolved_manifest = root / MANIFEST_RELATIVE
    if unresolved_manifest.is_symlink():
        raise ValueError("Lane-A candidate manifest must not be a symlink")
    manifest_path = unresolved_manifest.resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as error:
        raise ValueError("candidate manifest escapes root") from error
    if not manifest_path.is_file():
        raise ValueError("Lane-A candidate manifest is missing or is a symlink")
    manifest = _load_object(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("lane") != "a":
        raise ValueError("invalid Lane-A candidate manifest identity")
    fixed = manifest.get("fixed_evaluation")
    if not isinstance(fixed, dict):
        raise ValueError("candidate manifest is missing fixed_evaluation")
    episode_keys = fixed.get("episode_keys")
    episode_count = fixed.get("episode_count")
    if (
        not isinstance(episode_count, int)
        or isinstance(episode_count, bool)
        or episode_count != 5
        or not isinstance(episode_keys, list)
        or len(episode_keys) != episode_count
        or len(set(map(str, episode_keys))) != episode_count
        or any(not isinstance(item, str) or not item for item in episode_keys)
    ):
        raise ValueError("candidate fixed episode keys/count are invalid")
    provenance = manifest.get("provenance_contract")
    if not isinstance(provenance, dict):
        raise ValueError("candidate manifest is missing provenance_contract")
    required_fields = provenance.get("result_required_fields")
    if required_fields != [
        "predecessor_candidate_ids",
        "code_bundle_sha256",
        "candidate_config_sha256",
        "candidate_manifest_sha256",
    ]:
        raise ValueError("candidate predecessor provenance contract drift")
    code_paths = provenance.get("code_paths")
    if (
        not isinstance(code_paths, list)
        or not code_paths
        or len(set(map(str, code_paths))) != len(code_paths)
        or any(not isinstance(item, str) or not item for item in code_paths)
    ):
        raise ValueError("candidate code_paths contract is invalid")
    expected_code_bundle = _sha256(
        provenance.get("code_bundle_sha256"), "code_bundle_sha256"
    )
    actual_code_bundle = _code_bundle_sha256(root, code_paths)
    if actual_code_bundle != expected_code_bundle:
        raise ValueError(
            "code bundle changed after manifest registration: "
            f"expected={expected_code_bundle} observed={actual_code_bundle}"
        )
    rounds = manifest.get("successive_halving", {}).get("rounds")
    if not isinstance(rounds, list) or [
        item.get("cumulative_episode_count") if isinstance(item, dict) else None
        for item in rounds
    ] != [1, 3, 5]:
        raise ValueError("candidate manifest must retain screen1/screen3/fixed5")

    declared_families = manifest.get("candidate_families")
    if not isinstance(declared_families, list) or len(declared_families) != 3:
        raise ValueError("candidate manifest must declare exactly three families")
    registry: dict[str, dict[str, Any]] = {}
    if len(declared_families) != len(FAMILY_CONTRACT):
        raise ValueError("candidate family count does not match the frozen contract")
    # Python 3.8 on the x86/WSL coordinator predates zip(strict=True).
    for declared, contract in zip(declared_families, FAMILY_CONTRACT):
        family_id, expected_path, expected_ids, family_keys = contract
        if not isinstance(declared, dict) or declared.get("family_id") != family_id:
            raise ValueError(f"candidate family order/identity drift: {family_id}")
        if declared.get("candidate_ids") != list(expected_ids):
            raise ValueError(f"candidate ID set drift: {family_id}")
        path = _safe_config_path(root, declared.get("config"), expected_path)
        config = _load_object(path)
        if config.get("family_id") != family_id:
            raise ValueError(f"candidate config family mismatch: {family_id}")
        digest = _canonical_sha256(config)
        if declared.get("config_sha256") != digest:
            raise ValueError(f"candidate config canonical SHA mismatch: {family_id}")
        variants = config.get("variants")
        if not isinstance(variants, list) or [
            item.get("candidate_id") if isinstance(item, dict) else None
            for item in variants
        ] != list(expected_ids):
            raise ValueError(f"candidate variants do not match manifest: {family_id}")
        for variant in variants:
            candidate_id = variant["candidate_id"]
            alias = candidate_id.split("_", 1)[0]
            if alias in registry:
                raise ValueError(f"duplicate candidate alias: {alias}")
            overrides = variant.get("runtime_overrides")
            if not isinstance(overrides, dict) or not overrides:
                raise ValueError(f"candidate overrides missing: {candidate_id}")
            if set(overrides) != family_keys:
                raise ValueError(f"candidate crosses family override scope: {candidate_id}")
            checked = dict(_validate_override(key, value) for key, value in overrides.items())
            registry[alias] = {
                "alias": alias,
                "candidate_id": candidate_id,
                "family_id": family_id,
                "config": expected_path,
                "config_sha256": digest,
                "runtime_overrides": checked,
            }
    return manifest, registry


def resolve(root: Path, selector: str) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("candidate root must be a directory")
    if selector in {"baseline", "recovery_a"}:
        overrides = {"INTERNNAV_T5_CANDIDATE_PROFILE": selector}
        return _seal_resolution({
            "schema_version": 2,
            "status": "PASS",
            "candidate_selector": selector,
            "selector_kind": "legacy",
            "selected_candidate_ids": [],
            "selected_families": [],
            "runtime_overrides": overrides,
            "effective_candidate_profile": selector,
            "candidate_manifest": None,
            "registered_config_sha256_by_family": {},
            "provenance": {
                "code_bundle_sha256": None,
                "fixed_episode_count": None,
                "fixed_episode_keys": None,
                "predecessor_candidate_ids": [],
            },
            "successive_halving_episode_counts": [1, 3, 5],
        })
    if COMPOSITE_RE.fullmatch(selector) is None:
        raise ValueError(f"invalid candidate selector: {selector!r}")
    if selector.startswith("a0+") and "+c" in selector:
        raise ValueError("trajectory family requires recovery-enabled a1 predecessor")

    manifest, registry = _load_registry(root)
    aliases = selector.split("+")
    expected_prefixes = ("a", "b", "c")[: len(aliases)]
    if tuple(alias[0] for alias in aliases) != expected_prefixes:
        raise ValueError("candidate family dependency order is invalid")

    selected = []
    merged: dict[str, str] = {}
    for alias in aliases:
        candidate = registry.get(alias)
        if candidate is None:
            raise ValueError(f"candidate alias is not preregistered: {alias}")
        for key, value in candidate["runtime_overrides"].items():
            if key in merged and merged[key] != value:
                raise ValueError(f"conflicting candidate override: {key}")
            merged[key] = value
        selected.append(candidate)
    merged.setdefault("INTERNNAV_T5_CANDIDATE_PROFILE", "baseline")
    for key, value in merged.items():
        _validate_override(key, value)
    fixed = manifest["fixed_evaluation"]
    provenance = manifest["provenance_contract"]
    selected_candidate_ids = [item["candidate_id"] for item in selected]
    registered_configs = {
        family_id: registry[expected_ids[0].split("_", 1)[0]]["config_sha256"]
        for family_id, _path, expected_ids, _keys in FAMILY_CONTRACT
    }
    return _seal_resolution({
        "schema_version": 2,
        "status": "PASS",
        "candidate_selector": selector,
        "selector_kind": "preregistered_composite",
        "selected_candidate_ids": selected_candidate_ids,
        "selected_families": [item["family_id"] for item in selected],
        "selected_configs": [
            {"path": item["config"], "canonical_sha256": item["config_sha256"]}
            for item in selected
        ],
        "runtime_overrides": dict(sorted(merged.items())),
        "effective_candidate_profile": merged["INTERNNAV_T5_CANDIDATE_PROFILE"],
        "candidate_manifest": {
            "path": MANIFEST_RELATIVE.as_posix(),
            "canonical_sha256": _canonical_sha256(manifest),
        },
        "registered_config_sha256_by_family": registered_configs,
        "provenance": {
            "code_bundle_sha256": provenance["code_bundle_sha256"],
            "fixed_episode_count": fixed["episode_count"],
            "fixed_episode_keys": fixed["episode_keys"],
            "predecessor_candidate_ids": selected_candidate_ids[:-1],
        },
        "successive_halving_episode_counts": [1, 3, 5],
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("env", "json", "none"), default="json")
    arguments = parser.parse_args()
    try:
        result = resolve(arguments.root, arguments.selector)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        # pathlib.Path.write_text gained the ``newline`` keyword after the
        # Python 3.8 runtime used by the x86/WSL coordinator.  Open the file
        # explicitly so candidate materialization stays byte-identical across
        # the coordinator and both newer DGX runtimes.
        with arguments.output.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if arguments.format == "env":
        for key, value in result["runtime_overrides"].items():
            print(f"{key}={value}")
    elif arguments.format == "json":
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
