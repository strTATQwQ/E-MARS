#!/usr/bin/env python3
"""Resolve a fail-closed final-pilot lane input from a portable asset receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_REMOTE = re.compile(r"^/[A-Za-z0-9._/-]+$")
EXPECTED_PROFILE = {
    "candidate_profile": "a1+b1+c1",
    "rtf_ablation_profile": "navigation_fast",
    "isaac_sensor_profile": "baseline",
    "strict_extension_profile": "off",
    "nvblox_mode": "off",
    "run_mode": "model",
}
WP03_STOP_SHADOW_PROFILE = {
    "candidate_profile": "recovery_a",
    "rtf_ablation_profile": "navigation_fast",
    "isaac_sensor_profile": "dual_lane_wp03_stop_shadow",
    "strict_extension_profile": "off",
    "nvblox_mode": "off",
    "run_mode": "model",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _checks_pass(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(item is True for item in value.values())
    )


def _receipt_scope(receipt: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    has_scope = "prepare_scope" in receipt
    has_lanes = "prepared_lanes" in receipt
    if not has_scope and not has_lanes:
        return "dual", ("a", "b")
    contract = (receipt.get("prepare_scope"), receipt.get("prepared_lanes"))
    if contract == ("dual", ["a", "b"]):
        return "dual", ("a", "b")
    if contract == ("lane-a", ["a"]):
        return "lane-a", ("a",)
    raise ValueError("final-pilot preparation receipt has an invalid prepare scope")


def _resolved_relative(parent: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} is not a portable relative path")
    candidate = (parent / relative).resolve()
    try:
        candidate.relative_to(parent.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the preparation root") from error
    return candidate


def resolve_binding(
    *,
    prepare_root: Path,
    lane: str,
    code_sha: str,
    candidate_resolution_path: Path,
    runtime_profile: dict[str, str],
    output: Path,
    execution_profile: str = "final10",
    episode_key: str | None = None,
    evaluation_arm: str = "internvla_only",
) -> dict[str, Any]:
    if lane not in {"a", "b"}:
        raise ValueError("lane must be a or b")
    if SHA40.fullmatch(code_sha) is None:
        raise ValueError("code SHA must be a full lowercase Git SHA")
    if evaluation_arm not in {"internvla_only", "internvla_step3"}:
        raise ValueError("evaluation arm must be internvla_only or internvla_step3")
    prepare_root = prepare_root.resolve()
    receipt_path = prepare_root / "final_pilot_prepare_receipt.json"
    receipt = _object(receipt_path, "final-pilot preparation receipt")
    prepare_scope, prepared_lanes = _receipt_scope(receipt)
    if lane not in prepared_lanes:
        raise ValueError(
            f"Lane {lane} was not prepared by the {prepare_scope} final-pilot receipt"
        )
    candidate = _object(candidate_resolution_path.resolve(), "candidate resolution")
    split = receipt.get("split") if isinstance(receipt.get("split"), dict) else {}
    lanes = split.get("lanes") if isinstance(split.get("lanes"), dict) else {}
    lane_receipt = lanes.get(lane) if isinstance(lanes.get(lane), dict) else {}
    lane_episode_keys = lane_receipt.get("episode_keys")
    if execution_profile == "final10":
        if episode_key is not None:
            raise ValueError("final10 does not accept an episode key")
        execution_episode_keys = lane_episode_keys
    elif execution_profile == "pilot-screen1":
        if not isinstance(episode_key, str) or re.fullmatch(
            r"[A-Za-z0-9_.-]+", episode_key
        ) is None:
            raise ValueError("pilot-screen1 requires one safe episode key")
        if not isinstance(lane_episode_keys, list) or episode_key not in lane_episode_keys:
            raise ValueError(
                "pilot-screen1 episode key is not in the frozen lane manifest"
            )
        execution_episode_keys = [episode_key]
    else:
        raise ValueError("execution profile must be final10 or pilot-screen1")
    roots = receipt.get("deployment_roots")
    roots = roots if isinstance(roots, dict) else {}
    maps = receipt.get("static_maps")
    maps = maps if isinstance(maps, dict) else {}
    split_audit_path = _resolved_relative(
        prepare_root, split.get("audit_relative_path"), "split audit"
    )
    lane_dataset_path = _resolved_relative(
        prepare_root, lane_receipt.get("dataset_relative_path"), "lane dataset"
    )
    map_manifest_path = _resolved_relative(
        prepare_root, maps.get("manifest_relative_path"), "map manifest"
    )
    declared_resolution_sha = candidate.get("resolution_sha256")
    unsigned_candidate = dict(candidate)
    unsigned_candidate.pop("resolution_sha256", None)
    candidate_binding = candidate.get("canonical_binding")
    provenance = candidate.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    is_wp03_stop_shadow = runtime_profile == WP03_STOP_SHADOW_PROFILE
    expected_dgx_key = "dgx_a" if lane == "a" else "dgx_b"
    expected_x86_key = "x86_a" if lane == "a" else "x86_b"
    expected_root_keys = {
        root_key
        for active_lane in prepared_lanes
        for root_key in (f"dgx_{active_lane}", f"x86_{active_lane}")
    }
    expected_remote_map = lane_receipt.get("remote_map_manifest")
    expected_remote_dataset = lane_receipt.get("remote_dataset_root")
    remote_receipts = receipt.get("remote_receipts")
    remote_receipts = remote_receipts if isinstance(remote_receipts, dict) else {}
    remote_dgx_maps = remote_receipts.get("dgx_maps")
    remote_dgx_maps = remote_dgx_maps if isinstance(remote_dgx_maps, dict) else {}
    source_receipts = {
        "final_pilot_prepare_receipt_sha256": _sha256(receipt_path),
        "split_audit_sha256": _sha256(split_audit_path),
        "lane_dataset_sha256": _sha256(lane_dataset_path),
        "static_map_manifest_sha256": _sha256(map_manifest_path),
        "base_prepare_input_sha256": str(
            (receipt.get("base_prepare") or {}).get("input_sha256", "")
        ),
        "base_prepare_summary_sha256": str(
            (receipt.get("base_prepare") or {}).get("summary_sha256", "")
        ),
        "base_prepare_final_sha256": str(
            (receipt.get("base_prepare") or {}).get("final_sha256", "")
        ),
        "remote_x86_prepare_receipt_sha256": str(
            (remote_receipts.get("x86") or {}).get("sha256", "")
        ),
        "remote_dgx_map_receipt_sha256": str(
            (remote_dgx_maps.get(lane) or {}).get("sha256", "")
        ),
    }
    checks = {
        "prepare_receipt_pass": receipt.get("status") == "PASS"
        and _checks_pass(receipt.get("checks")),
        "prepare_scope_exact": prepare_scope in {"dual", "lane-a"}
        and lane in prepared_lanes,
        "prepare_exact_code": receipt.get("code_ref_sha") == code_sha,
        # The preparation receipt freezes the disjoint A10/B10 assets.  The
        # WP-03 shadow run intentionally reuses only those assets while keeping
        # its navigation/termination overlay explicit in this derived binding.
        "runtime_profile_exact": (
            runtime_profile == EXPECTED_PROFILE
            or runtime_profile == WP03_STOP_SHADOW_PROFILE
        )
        and receipt.get("final_profile") == EXPECTED_PROFILE,
        "candidate_resolution_pass": candidate.get("schema_version") == 2
        and candidate.get("status") == "PASS"
        and candidate.get("candidate_selector")
        == runtime_profile["candidate_profile"],
        "candidate_resolution_self_hash": SHA256.fullmatch(
            str(declared_resolution_sha or "")
        )
        is not None
        and declared_resolution_sha == _canonical_sha256(unsigned_candidate),
        "candidate_binding_present": isinstance(candidate_binding, dict)
        and candidate_binding.get("candidate_selector")
        == runtime_profile["candidate_profile"]
        and (
            candidate_binding.get("fixed_episode_count") == 5
            or (
                is_wp03_stop_shadow
                and candidate.get("selector_kind") == "legacy"
                and candidate_binding.get("effective_candidate_profile")
                    == "recovery_a"
                and candidate_binding.get("fixed_episode_count") is None
            )
        ),
        "candidate_selection_evidence_stays_fixed5": (
            provenance.get("fixed_episode_count") == 5
            and isinstance(provenance.get("fixed_episode_keys"), list)
            and len(provenance["fixed_episode_keys"]) == 5
        ) or (
            is_wp03_stop_shadow
            and candidate.get("selector_kind") == "legacy"
            and provenance.get("fixed_episode_count") is None
            and provenance.get("fixed_episode_keys") is None
        ),
        "split_audit_sha_binding": _sha256(split_audit_path)
        == split.get("audit_sha256"),
        "lane_dataset_file_binding": _sha256(lane_dataset_path)
        == lane_receipt.get("dataset_sha256")
        and lane_receipt.get("episode_count") == 10
        and isinstance(lane_episode_keys, list)
        and len(lane_episode_keys) == 10
        and len(set(lane_episode_keys)) == 10,
        "execution_selection_exact": (
            execution_profile == "final10"
            and episode_key is None
            and execution_episode_keys == lane_episode_keys
        ) or (
            execution_profile == "pilot-screen1"
            and isinstance(lane_episode_keys, list)
            and episode_key in lane_episode_keys
            and execution_episode_keys == [episode_key]
        ),
        "static_map_file_binding": _sha256(map_manifest_path)
        == maps.get("manifest_sha256")
        and maps.get("episode_count") == 20
        and isinstance(maps.get("scene_ids"), list)
        and len(maps["scene_ids"]) == 5
        and len(set(maps["scene_ids"])) == 5,
        "deployment_roots_exact": set(roots) == expected_root_keys
        and all(isinstance(value, str) and SAFE_REMOTE.fullmatch(value) for value in roots.values()),
        "remote_receipts_scope_exact": set(remote_dgx_maps)
        == set(prepared_lanes),
        "remote_dataset_root_safe": isinstance(expected_remote_dataset, str)
        and SAFE_REMOTE.fullmatch(expected_remote_dataset) is not None
        and expected_remote_dataset.startswith(str(roots.get(expected_x86_key, "")) + "/"),
        "remote_map_manifest_safe": isinstance(expected_remote_map, str)
        and SAFE_REMOTE.fullmatch(expected_remote_map) is not None
        and expected_remote_map.startswith(str(roots.get(expected_dgx_key, "")) + "/")
        and expected_remote_map.endswith("/manifest.json"),
        "all_source_receipts_sha256": all(
            SHA256.fullmatch(value) is not None for value in source_receipts.values()
        ),
    }
    payload = {
        "schema_version": 2,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "lane": lane,
        "prepare_scope": prepare_scope,
        "prepared_lanes": list(prepared_lanes),
        "code_ref_sha": code_sha,
        "candidate_profile": runtime_profile["candidate_profile"],
        "candidate_resolution_sha256": declared_resolution_sha,
        "candidate_resolution_file_sha256": _sha256(candidate_resolution_path),
        "candidate_binding": candidate_binding,
        "source_final_profile": receipt.get("final_profile"),
        "wp03_stop_shadow_overlay": is_wp03_stop_shadow,
        "rtf_ablation_profile": runtime_profile["rtf_ablation_profile"],
        "isaac_sensor_profile": runtime_profile["isaac_sensor_profile"],
        "strict_extension_profile": runtime_profile["strict_extension_profile"],
        "nvblox_mode": runtime_profile["nvblox_mode"],
        "run_mode": runtime_profile["run_mode"],
        "source_execution_profile": "final10",
        "execution_profile": execution_profile,
        "execution_episode_count": len(execution_episode_keys or []),
        "execution_episode_keys": execution_episode_keys,
        "evaluation_arm": evaluation_arm,
        "pair_set": "paired10_a" if lane == "a" else "paired10_b",
        "screen_episode_key": episode_key,
        "deployment_roots": {
            "dgx": roots.get(expected_dgx_key),
            "x86": roots.get(expected_x86_key),
        },
        "dataset_root": expected_remote_dataset,
        "dataset_sha256": lane_receipt.get("dataset_sha256"),
        "static_map_manifest_sha256": maps.get("manifest_sha256"),
        "static_map_manifest_path": expected_remote_map,
        "episode_count": lane_receipt.get("episode_count"),
        "episode_keys": lane_episode_keys,
        "split_audit_sha256": split.get("audit_sha256"),
        "source_receipts": source_receipts,
        "path_contract": {
            "prepare_receipt": "portable_relative_local_paths",
            "runtime_inputs": "immutable_remote_deployment_paths",
        },
        "checks": checks,
    }
    output = output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ValueError("binding output already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    if payload["status"] != "PASS":
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"final-pilot lane binding failed: {failed}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-root", required=True, type=Path)
    parser.add_argument("--lane", required=True, choices=("a", "b"))
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--candidate-resolution", required=True, type=Path)
    parser.add_argument("--candidate-profile", required=True)
    parser.add_argument("--rtf-ablation-profile", required=True)
    parser.add_argument("--isaac-sensor-profile", required=True)
    parser.add_argument("--strict-extension-profile", required=True)
    parser.add_argument("--nvblox-mode", required=True)
    parser.add_argument("--run-mode", required=True)
    parser.add_argument(
        "--evaluation-arm",
        default="internvla_only",
        choices=("internvla_only", "internvla_step3"),
    )
    parser.add_argument(
        "--execution-profile",
        choices=("final10", "pilot-screen1"),
        default="final10",
    )
    parser.add_argument("--episode-key")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    runtime_profile = {
        "candidate_profile": arguments.candidate_profile,
        "rtf_ablation_profile": arguments.rtf_ablation_profile,
        "isaac_sensor_profile": arguments.isaac_sensor_profile,
        "strict_extension_profile": arguments.strict_extension_profile,
        "nvblox_mode": arguments.nvblox_mode,
        "run_mode": arguments.run_mode,
    }
    try:
        payload = resolve_binding(
            prepare_root=arguments.prepare_root,
            lane=arguments.lane,
            code_sha=arguments.code_sha,
            candidate_resolution_path=arguments.candidate_resolution,
            runtime_profile=runtime_profile,
            output=arguments.output,
            execution_profile=arguments.execution_profile,
            episode_key=arguments.episode_key,
            evaluation_arm=arguments.evaluation_arm,
        )
    except (OSError, ValueError) as error:
        print(f"final-pilot lane binding failed: {error}")
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
