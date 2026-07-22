#!/usr/bin/env python3
"""Seal the independently prepared T5 A10/B10 pilot assets.

The online preparation wrapper does the host I/O.  This module performs the
portable, fail-closed content checks and writes the receipt consumed by the
two lane runners.  All local paths in the receipt are relative to the receipt
directory so the complete result directory can be relocated before final
analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_MANIFEST = (
    ROOT / "configs" / "internnav_t5" / "t5_24h_execution_manifest.json"
)
PILOT_MANIFEST = (
    ROOT / "configs" / "internnav_t3" / "model_pilot_episode_manifest.json"
)
EXPECTED_SOURCE_SHA256 = (
    "f09b11004d15d579620ba2b2769dcbf863efe4cf04d9e404f071f66244463f8b"
)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_REMOTE = re.compile(r"^/[A-Za-z0-9._/-]+$")
FINAL_PROFILE = {
    "candidate_profile": "a1+b1+c1",
    "rtf_ablation_profile": "navigation_fast",
    "isaac_sensor_profile": "baseline",
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


def _prepare_scope(
    base_input: dict[str, Any],
    base_summary: dict[str, Any],
    base_final: dict[str, Any],
) -> tuple[str, tuple[str, ...]]:
    records = (base_input, base_summary, base_final)
    if all(
        "prepare_scope" not in record and "prepared_lanes" not in record
        for record in records
    ):
        return "dual", ("a", "b")
    contracts = [
        (record.get("prepare_scope"), record.get("prepared_lanes"))
        for record in records
    ]
    if all(contract == ("dual", ["a", "b"]) for contract in contracts):
        return "dual", ("a", "b")
    if all(contract == ("lane-a", ["a"]) for contract in contracts):
        return "lane-a", ("a",)
    raise ValueError("base preparation scope receipts disagree or are unsupported")


def _relative(path: Path, base: Path, label: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(base.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must be inside the receipt directory") from error
    if not relative or relative.startswith("../"):
        raise ValueError(f"{label} is not a portable relative path")
    return relative


def _remote_under(path: str, root: str, label: str) -> bool:
    if SAFE_REMOTE.fullmatch(path) is None or SAFE_REMOTE.fullmatch(root) is None:
        return False
    candidate = PurePosixPath(path)
    parent = PurePosixPath(root)
    return candidate != parent and parent in candidate.parents and ".." not in candidate.parts


def _episode_key(item: dict[str, Any]) -> str:
    trajectory = item.get("trajectory_id")
    episode = item.get("episode_id")
    if trajectory is None or episode is None:
        raise ValueError("map generation lacks trajectory_id or episode_id")
    return f"{trajectory}_{episode}"


def _map_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("maps", {})
    if isinstance(raw, dict):
        values = list(raw.values())
    elif isinstance(raw, list):
        values = raw
    else:
        return []
    return [item for item in values if isinstance(item, dict)]


def _archive_contract(archive: Path, map_dir: Path, manifest: dict[str, Any]) -> bool:
    if archive.is_symlink() or not archive.is_file():
        return False
    expected = {"manifest.json"}
    map_records = _map_records(manifest)
    if len(map_records) != manifest.get("map_count"):
        return False
    for item in map_records:
        relative = item.get("file")
        if not isinstance(relative, str):
            return False
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
            return False
        expected.add(relative)
        local = map_dir / relative
        if local.is_symlink() or not local.is_file():
            return False
    try:
        with tarfile.open(archive, "r:gz") as stream:
            members = [item for item in stream.getmembers() if item.isfile()]
            names = {PurePosixPath(item.name).as_posix().lstrip("./") for item in members}
            if names != expected:
                return False
            if any(
                item.issym()
                or item.islnk()
                or PurePosixPath(item.name).is_absolute()
                or ".." in PurePosixPath(item.name).parts
                for item in stream.getmembers()
            ):
                return False
            archived = stream.extractfile(next(item for item in members if PurePosixPath(item.name).as_posix().lstrip("./") == "manifest.json"))
            return archived is not None and hashlib.sha256(archived.read()).hexdigest() == _sha256(map_dir / "manifest.json")
    except (OSError, tarfile.TarError, StopIteration):
        return False


def build_receipt(
    *,
    code_sha: str,
    base_prepare_root: Path,
    split_audit_path: Path,
    map_dir: Path,
    map_archive: Path,
    remote_x86_receipt_path: Path,
    remote_lane_map_receipt_paths: dict[str, Path],
    remote_lane_dataset_roots: dict[str, str],
    remote_lane_map_manifests: dict[str, str],
    output: Path,
) -> dict[str, Any]:
    if SHA40.fullmatch(code_sha) is None:
        raise ValueError("code SHA must be a full lowercase Git SHA")
    output = output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ValueError("receipt output already exists")
    base_prepare_root = base_prepare_root.resolve()
    split_audit_path = split_audit_path.resolve()
    map_dir = map_dir.resolve()
    map_archive = map_archive.resolve()
    remote_x86_receipt_path = remote_x86_receipt_path.resolve()

    base_summary_path = base_prepare_root / "d0_prepare_summary.json"
    base_final_path = base_prepare_root / "d0_prepare_final_summary.json"
    base_input_path = base_prepare_root / "fast_prepare_input.json"
    base_summary = _object(base_summary_path, "base preparation summary")
    base_final = _object(base_final_path, "base preparation final summary")
    base_input = _object(base_input_path, "base preparation input")
    prepare_scope, active_lanes = _prepare_scope(
        base_input, base_summary, base_final
    )
    split = _object(split_audit_path, "final-pilot split audit")
    map_manifest_path = map_dir / "manifest.json"
    map_manifest = _object(map_manifest_path, "final-pilot static-map manifest")
    remote_x86 = _object(remote_x86_receipt_path, "remote x86 asset receipt")
    missing_remote_inputs = {
        lane
        for lane in active_lanes
        if lane not in remote_lane_map_receipt_paths
        or lane not in remote_lane_dataset_roots
        or lane not in remote_lane_map_manifests
    }
    if missing_remote_inputs:
        raise ValueError(
            "remote inputs missing for active lanes: "
            + ", ".join(sorted(missing_remote_inputs))
        )
    remote_maps = {
        lane: _object(
            remote_lane_map_receipt_paths[lane].resolve(),
            f"remote Lane {lane} map receipt",
        )
        for lane in active_lanes
    }
    execution = _object(EXECUTION_MANIFEST, "T5 execution manifest")
    pilot = _object(PILOT_MANIFEST, "frozen pilot manifest")

    final_evaluation = execution.get("final_evaluation")
    authored = (
        final_evaluation.get("pilot")
        if isinstance(final_evaluation, dict)
        else None
    )
    authored = authored if isinstance(authored, dict) else {}
    expected_keys = pilot.get("episode_keys")
    expected_scenes = [
        str(value).replace("\\", "/").split("/")[-2]
        for value in pilot.get("scene_ids", [])
        if isinstance(value, str) and len(str(value).replace("\\", "/").split("/")) >= 2
    ]
    split_source = split.get("source") if isinstance(split.get("source"), dict) else {}
    split_lanes = split.get("lanes") if isinstance(split.get("lanes"), dict) else {}
    roots = base_final.get("deployment_roots")
    roots = roots if isinstance(roots, dict) else {}
    required_root_keys = {
        root_key
        for lane in active_lanes
        for root_key in (f"dgx_{lane}", f"x86_{lane}")
    }
    allowed_root_keys = required_root_keys | {"x86_prepare"}
    execution_roots = {key: roots.get(key) for key in sorted(required_root_keys)}
    lane_keys = {
        "a": authored.get("lane_a_episode_keys"),
        "b": authored.get("lane_b_episode_keys"),
    }
    lane_records: dict[str, dict[str, Any]] = {}
    lane_file_checks: dict[str, bool] = {}
    remote_lane_path_checks: dict[str, bool] = {}
    for lane in ("a", "b"):
        raw = split_lanes.get(lane)
        raw = raw if isinstance(raw, dict) else {}
        path = Path(str(raw.get("output_dataset", ""))).resolve()
        digest = _sha256(path) if path.is_file() and not path.is_symlink() else None
        root_key = "x86_a" if lane == "a" else "x86_b"
        dgx_key = "dgx_a" if lane == "a" else "dgx_b"
        lane_file_checks[lane] = (
            digest == raw.get("output_sha256")
            and raw.get("episode_count") == 10
            and raw.get("episode_keys") == lane_keys[lane]
        )
        lane_record = {
            "dataset_relative_path": _relative(path, output.parent, f"Lane {lane} dataset"),
            "dataset_sha256": raw.get("output_sha256"),
            "episode_count": raw.get("episode_count"),
            "episode_keys": raw.get("episode_keys"),
        }
        if lane in active_lanes:
            lane_records[lane] = lane_record
            dataset_remote = remote_lane_dataset_roots[lane]
            map_remote = remote_lane_map_manifests[lane]
            remote_lane_path_checks[lane] = (
                _remote_under(
                    dataset_remote,
                    str(roots.get(root_key, "")),
                    f"Lane {lane} dataset",
                )
                and dataset_remote.endswith("/val_unseen") is False
                and _remote_under(
                    map_remote,
                    str(roots.get(dgx_key, "")),
                    f"Lane {lane} map",
                )
                and map_remote.endswith("/manifest.json")
            )
            lane_record.update(
                {
                    "remote_dataset_root": dataset_remote,
                    "remote_map_manifest": map_remote,
                }
            )

    source_path = Path(str(split_source.get("dataset", ""))).resolve()
    source_file_sha = (
        _sha256(source_path)
        if source_path.is_file() and not source_path.is_symlink()
        else None
    )
    generations = map_manifest.get("generations")
    maps = _map_records(map_manifest)
    generation_keys = (
        [_episode_key(item) for item in generations]
        if isinstance(generations, list) and all(isinstance(item, dict) for item in generations)
        else []
    )
    map_scans = sorted({str(item.get("scan")) for item in maps})
    checks = {
        "base_prepare_summary_pass": base_summary.get("status") == "PASS"
        and _checks_pass(base_summary.get("checks")),
        "base_prepare_final_pass": base_final.get("status") == "PASS"
        and _checks_pass(base_final.get("checks")),
        "base_prepare_exact_code": base_summary.get("code_ref_sha") == code_sha
        and base_final.get("code_ref_sha") == code_sha
        and base_input.get("code_ref_sha") == code_sha,
        "base_prepare_scope_exact": prepare_scope in {"dual", "lane-a"}
        and list(active_lanes)
        == (["a"] if prepare_scope == "lane-a" else ["a", "b"]),
        "base_prepare_summary_binding": base_final.get("preparation_summary_sha256")
        == _sha256(base_summary_path),
        "base_prepare_deployment_roots": required_root_keys.issubset(roots)
        and set(roots).issubset(allowed_root_keys)
        and all(isinstance(value, str) and SAFE_REMOTE.fullmatch(value) for value in roots.values()),
        "split_audit_pass": split.get("status") == "PASS"
        and _checks_pass(split.get("checks")),
        "split_source_exact": split_source.get("sha256") == EXPECTED_SOURCE_SHA256
        and source_file_sha == EXPECTED_SOURCE_SHA256
        and split_source.get("episode_count") == 20
        and split_source.get("episode_keys") == expected_keys,
        "split_lane_files_exact": all(lane_file_checks.values()),
        "active_lane_remote_paths_exact": all(
            remote_lane_path_checks.get(lane) is True for lane in active_lanes
        ),
        "split_lane_keys_disjoint": isinstance(lane_keys["a"], list)
        and isinstance(lane_keys["b"], list)
        and set(lane_keys["a"]).isdisjoint(lane_keys["b"])
        and lane_keys["a"] + lane_keys["b"] == expected_keys,
        "map_manifest_full20": map_manifest.get("dataset_sha256")
        == EXPECTED_SOURCE_SHA256
        and map_manifest.get("episode_count") == 20
        and generation_keys == expected_keys,
        "map_manifest_five_scenes": len(expected_scenes) == 5
        and len(set(expected_scenes)) == 5
        and map_scans == sorted(expected_scenes),
        "map_truth_isolated": isinstance(map_manifest.get("t4_truth_isolation"), dict)
        and map_manifest["t4_truth_isolation"].get("status") == "PASS"
        and map_manifest["t4_truth_isolation"].get("dataset_sha256")
        == EXPECTED_SOURCE_SHA256
        and map_manifest["t4_truth_isolation"].get(
            "runtime_ground_truth_pose_used_for_map_selection"
        )
        is False,
        "map_archive_exact": _archive_contract(map_archive, map_dir, map_manifest),
        "remote_x86_assets_exact": remote_x86.get("status") == "PASS"
        and remote_x86.get("code_ref_sha") == code_sha
        and remote_x86.get("source_sha256") == EXPECTED_SOURCE_SHA256
        and isinstance(remote_x86.get("lane_dataset_sha256"), dict)
        and set(remote_x86["lane_dataset_sha256"]) == set(active_lanes)
        and all(
            remote_x86["lane_dataset_sha256"].get(lane)
            == lane_records[lane]["dataset_sha256"]
            for lane in active_lanes
        )
        and isinstance(remote_x86.get("lane_dataset_roots"), dict)
        and set(remote_x86["lane_dataset_roots"]) == set(active_lanes)
        and all(
            remote_x86["lane_dataset_roots"].get(lane)
            == remote_lane_dataset_roots[lane]
            for lane in active_lanes
        )
        and remote_x86.get("map_manifest_sha256") == _sha256(map_manifest_path),
        "remote_dgx_maps_exact": set(remote_maps) == set(active_lanes)
        and all(
            remote_maps[lane].get("status") == "PASS"
            and remote_maps[lane].get("lane") == lane
            and remote_maps[lane].get("code_ref_sha") == code_sha
            and remote_maps[lane].get("manifest_path")
            == remote_lane_map_manifests[lane]
            and remote_maps[lane].get("manifest_sha256")
            == _sha256(map_manifest_path)
            for lane in active_lanes
        ),
        "final_profile_frozen": FINAL_PROFILE == {
            "candidate_profile": "a1+b1+c1",
            "rtf_ablation_profile": "navigation_fast",
            "isaac_sensor_profile": "baseline",
            "strict_extension_profile": "off",
            "nvblox_mode": "off",
            "run_mode": "model",
        },
    }
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": "t5_final_pilot_asset_prepare",
        "code_ref_sha": code_sha,
        "prepare_scope": prepare_scope,
        "prepared_lanes": list(active_lanes),
        "base_prepare": {
            "relative_path": Path(os.path.relpath(base_prepare_root, output.parent)).as_posix(),
            "input_sha256": _sha256(base_input_path),
            "summary_sha256": _sha256(base_summary_path),
            "final_sha256": _sha256(base_final_path),
        },
        "deployment_roots": execution_roots,
        "split": {
            "audit_relative_path": _relative(split_audit_path, output.parent, "split audit"),
            "audit_sha256": _sha256(split_audit_path),
            "source_relative_path": _relative(source_path, output.parent, "source dataset"),
            "source_sha256": EXPECTED_SOURCE_SHA256,
            "episode_count": 20,
            "episode_keys": expected_keys,
            "lanes": {lane: lane_records[lane] for lane in active_lanes},
        },
        "static_maps": {
            "manifest_relative_path": _relative(map_manifest_path, output.parent, "map manifest"),
            "manifest_sha256": _sha256(map_manifest_path),
            "archive_relative_path": _relative(map_archive, output.parent, "map archive"),
            "archive_sha256": _sha256(map_archive),
            "episode_count": map_manifest.get("episode_count"),
            "scene_ids": sorted(expected_scenes),
            "map_count": map_manifest.get("map_count"),
        },
        "remote_receipts": {
            "x86": {
                "relative_path": _relative(remote_x86_receipt_path, output.parent, "remote x86 receipt"),
                "sha256": _sha256(remote_x86_receipt_path),
            },
            "dgx_maps": {
                lane: {
                    "relative_path": _relative(remote_lane_map_receipt_paths[lane], output.parent, f"remote Lane {lane} map receipt"),
                    "sha256": _sha256(remote_lane_map_receipt_paths[lane]),
                }
                for lane in active_lanes
            },
        },
        "final_profile": FINAL_PROFILE,
        "path_contract": {
            "base": "receipt_parent",
            "local_paths_are_relative": True,
            "remote_paths_are_immutable_deployment_paths": True,
        },
        "checks": checks,
    }
    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    try:
        with os.fdopen(temporary_descriptor, "w", encoding="utf-8", newline="\n") as stream:
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
        raise ValueError(f"final-pilot preparation receipt failed: {failed}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--base-prepare-root", required=True, type=Path)
    parser.add_argument("--split-audit", required=True, type=Path)
    parser.add_argument("--map-dir", required=True, type=Path)
    parser.add_argument("--map-archive", required=True, type=Path)
    parser.add_argument("--remote-x86-receipt", required=True, type=Path)
    parser.add_argument("--remote-lane-a-map-receipt", required=True, type=Path)
    parser.add_argument("--remote-lane-b-map-receipt", type=Path)
    parser.add_argument("--remote-lane-a-dataset-root", required=True)
    parser.add_argument("--remote-lane-b-dataset-root")
    parser.add_argument("--remote-lane-a-map-manifest", required=True)
    parser.add_argument("--remote-lane-b-map-manifest")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        payload = build_receipt(
            code_sha=arguments.code_sha,
            base_prepare_root=arguments.base_prepare_root,
            split_audit_path=arguments.split_audit,
            map_dir=arguments.map_dir,
            map_archive=arguments.map_archive,
            remote_x86_receipt_path=arguments.remote_x86_receipt,
            remote_lane_map_receipt_paths={
                lane: value
                for lane, value in {
                    "a": arguments.remote_lane_a_map_receipt,
                    "b": arguments.remote_lane_b_map_receipt,
                }.items()
                if value is not None
            },
            remote_lane_dataset_roots={
                lane: value
                for lane, value in {
                    "a": arguments.remote_lane_a_dataset_root,
                    "b": arguments.remote_lane_b_dataset_root,
                }.items()
                if value is not None
            },
            remote_lane_map_manifests={
                lane: value
                for lane, value in {
                    "a": arguments.remote_lane_a_map_manifest,
                    "b": arguments.remote_lane_b_map_manifest,
                }.items()
                if value is not None
            },
            output=arguments.output,
        )
    except (OSError, ValueError) as error:
        print(f"final-pilot preparation receipt failed: {error}")
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
