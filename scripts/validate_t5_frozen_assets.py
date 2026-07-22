#!/usr/bin/env python3
"""Fail-closed content validation for the T5 Golden model and D0 scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_byte_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_plain_name(value: object) -> bool:
    """Return true only for a single portable top-level file/scene name."""

    return (
        isinstance(value, str)
        and bool(value)
        and "\x00" not in value
        and "/" not in value
        and "\\" not in value
        and value not in {".", ".."}
        and PurePosixPath(value).name == value
    )


def _safe_relative_path(value: object) -> PurePosixPath | None:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return None
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    if relative.as_posix() != value:
        return None
    return relative


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _has_symlink_component(root: Path, relative: PurePosixPath) -> bool:
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _manifest_entries(
    value: object, *, path_key: str
) -> tuple[list[dict[str, Any]], bool, bool]:
    """Return dict entries plus shape and unique-path validity."""

    if not isinstance(value, list) or not value:
        return [], False, False
    entries: list[dict[str, Any]] = []
    paths: list[object] = []
    shape_valid = True
    for raw in value:
        if not isinstance(raw, dict):
            shape_valid = False
            continue
        entries.append(raw)
        paths.append(raw.get(path_key))
    strings = [item for item in paths if isinstance(item, str)]
    unique = len(strings) == len(paths) and len(strings) == len(set(strings))
    return entries, shape_valid, unique


def validate_checkpoint_files(
    content: dict[str, Any], checkpoint: Path, internnav_root: Path
) -> tuple[dict[str, bool], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the exact checkpoint file set and every content digest.

    The routine deliberately does not trust byte size as a content identity and
    never follows a manifest-provided path outside ``internnav_root``.
    """

    file_entries, files_shape_valid, unique_files = _manifest_entries(
        content.get("files"), path_key="path"
    )
    safe_names = files_shape_valid and all(
        _is_plain_name(item.get("path")) for item in file_entries
    )
    file_metadata_valid = files_shape_valid and all(
        _is_byte_count(item.get("bytes")) and _is_sha256(item.get("sha256"))
        for item in file_entries
    )
    expected_files: dict[str, dict[str, Any]] = {}
    if unique_files and safe_names:
        expected_files = {str(item["path"]): item for item in file_entries}

    actual_files: set[str] = set()
    checkpoint_readable = checkpoint.is_dir() and not checkpoint.is_symlink()
    if checkpoint_readable:
        # A symlink is included in the set so that it cannot masquerade as an
        # expected regular file. Directories are outside the stated top-level
        # regular-file contract.
        actual_files = {
            path.name
            for path in checkpoint.iterdir()
            if path.is_file() or path.is_symlink()
        }

    file_audits: list[dict[str, Any]] = []
    for relative, expected in sorted(expected_files.items()):
        path = checkpoint / relative
        regular = checkpoint_readable and path.is_file() and not path.is_symlink()
        actual_bytes = path.stat().st_size if regular else None
        actual_sha = sha256_file(path) if regular else None
        expected_bytes = expected.get("bytes")
        expected_sha = expected.get("sha256")
        file_audits.append(
            {
                "path": relative,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "regular_file_without_symlink": regular,
                "match": bool(
                    regular
                    and _is_byte_count(expected_bytes)
                    and _is_sha256(expected_sha)
                    and actual_bytes == expected_bytes
                    and actual_sha == expected_sha
                ),
            }
        )

    linked_entries, linked_shape_valid, unique_linked = _manifest_entries(
        content.get("linked_assets"), path_key="path_from_internnav_root"
    )
    linked_paths = [
        _safe_relative_path(item.get("path_from_internnav_root"))
        for item in linked_entries
    ]
    safe_linked_paths = linked_shape_valid and all(path is not None for path in linked_paths)
    linked_metadata_valid = linked_shape_valid and all(
        _is_byte_count(item.get("bytes")) and _is_sha256(item.get("sha256"))
        for item in linked_entries
    )
    linked_audits: list[dict[str, Any]] = []
    root_resolved = internnav_root.resolve()
    for expected, relative in zip(linked_entries, linked_paths):
        display_path = expected.get("path_from_internnav_root")
        inside_root = False
        no_symlink = False
        path: Path | None = None
        if relative is not None:
            candidate = internnav_root.joinpath(*relative.parts)
            resolved = candidate.resolve()
            inside_root = _is_within(resolved, root_resolved)
            no_symlink = not _has_symlink_component(internnav_root, relative)
            if inside_root:
                path = candidate
        regular = bool(
            path is not None and no_symlink and path.is_file() and not path.is_symlink()
        )
        actual_bytes = path.stat().st_size if regular and path is not None else None
        actual_sha = sha256_file(path) if regular and path is not None else None
        expected_bytes = expected.get("bytes")
        expected_sha = expected.get("sha256")
        linked_audits.append(
            {
                "path_from_internnav_root": display_path,
                "inside_internnav_root": inside_root,
                "no_symlink_components": no_symlink,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "regular_file_without_symlink": regular,
                "match": bool(
                    regular
                    and _is_byte_count(expected_bytes)
                    and _is_sha256(expected_sha)
                    and actual_bytes == expected_bytes
                    and actual_sha == expected_sha
                ),
            }
        )

    checks = {
        "checkpoint_manifest_files_shape": files_shape_valid,
        "unique_checkpoint_file_paths": unique_files,
        "safe_top_level_file_names": safe_names,
        "checkpoint_file_metadata_valid": file_metadata_valid,
        "strict_top_level_file_set_contract": content.get(
            "top_level_regular_file_set_strict"
        )
        is True,
        "strict_top_level_file_set": bool(expected_files)
        and checkpoint_readable
        and actual_files == set(expected_files),
        "all_checkpoint_files_content_match": bool(file_audits)
        and len(file_audits) == len(file_entries)
        and all(item["match"] for item in file_audits),
        "linked_asset_manifest_shape": linked_shape_valid,
        "unique_linked_asset_paths": unique_linked,
        "safe_linked_asset_paths": safe_linked_paths,
        "linked_asset_metadata_valid": linked_metadata_valid,
        "all_linked_assets_content_match": bool(linked_audits)
        and len(linked_audits) == len(linked_entries)
        and all(item["match"] for item in linked_audits),
    }
    return checks, file_audits, linked_audits


def _source_identity(
    internnav_root: Path, golden_model: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(internnav_root), *arguments],
            text=True,
            stderr=subprocess.PIPE,
        )

    revision = git("rev-parse", "HEAD").strip()
    tree = git("rev-parse", "HEAD^{tree}").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    raw_submodules = git("submodule", "status", "--recursive")
    submodules: dict[str, str] = {}
    markers_clean = True
    for line in raw_submodules.splitlines():
        if not line:
            continue
        markers_clean &= line[0] == " "
        fields = line[1:].split()
        if len(fields) < 2 or fields[1] in submodules:
            markers_clean = False
            continue
        submodules[fields[1]] = fields[0]
    checks = {
        "model_revision": revision == golden_model.get("internnav_revision"),
        "source_tree_sha": tree == golden_model.get("internnav_tree_sha"),
        "worktree_clean": golden_model.get("internnav_worktree_clean_required")
        is True
        and not status,
        "submodule_markers_clean": markers_clean,
        "submodule_revisions": submodules
        == golden_model.get("internnav_submodules"),
    }
    return checks, {
        "model_revision": revision,
        "internnav_tree_sha": tree,
        "internnav_submodules": submodules,
        "internnav_worktree_status_entry_count": len(status.splitlines()),
    }


def checkpoint_audit(
    golden: dict[str, Any], content: dict[str, Any], internnav_root: Path
) -> dict[str, Any]:
    model = golden.get("model")
    if not isinstance(model, dict):
        raise ValueError("golden model contract must be an object")
    checkpoint_name = model.get("checkpoint_name")
    safe_checkpoint_name = _is_plain_name(checkpoint_name)
    checkpoint = (
        internnav_root / "checkpoints" / str(checkpoint_name)
        if safe_checkpoint_name
        else internnav_root / "checkpoints" / "__invalid_checkpoint_name__"
    )
    content_sha = canonical_json_sha256(content)
    source_checks, source = _source_identity(internnav_root, model)
    file_checks, files, linked = validate_checkpoint_files(
        content, checkpoint, internnav_root
    )
    checks = {
        **source_checks,
        "safe_checkpoint_name": safe_checkpoint_name,
        "checkpoint_directory": checkpoint.is_dir() and not checkpoint.is_symlink(),
        "content_manifest_schema_version": content.get("schema_version") == 1,
        "content_manifest_path": model.get("checkpoint_content_manifest")
        == "configs/internnav_t5/checkpoint_content_manifest.json",
        "content_manifest_sha256": content_sha
        == model.get("checkpoint_content_manifest_canonical_sha256"),
        "checkpoint_name": content.get("checkpoint_name") == checkpoint_name,
        "checkpoint_revision": content.get("checkpoint_revision")
        == model.get("checkpoint_revision"),
        **file_checks,
    }
    return {
        "schema_version": 2,
        "status": "PASS" if all(checks.values()) else "FAIL",
        **source,
        "checkpoint_revision": content.get("checkpoint_revision"),
        "checkpoint_content_manifest_canonical_sha256": content_sha,
        "expected_runtime_weight_inventory_sha256": model.get(
            "weight_inventory_sha256"
        ),
        "model_loaded": False,
        "checks": checks,
        "checkpoint_files": files,
        "linked_assets": linked,
    }


def static_map_audit(
    golden: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    scenes = golden.get("scenes")
    expected_entries = (
        scenes.get("d0_selected_scene_geometry") if isinstance(scenes, dict) else None
    )
    expected_shape = isinstance(expected_entries, list) and bool(expected_entries)
    expected: dict[str, str] = {}
    expected_names: list[str] = []
    expected_metadata_valid = expected_shape
    if expected_shape:
        for item in expected_entries:
            if not isinstance(item, dict):
                expected_metadata_valid = False
                continue
            scene, digest = item.get("scene"), item.get("ply_sha256")
            expected_metadata_valid &= _is_plain_name(scene) and _is_sha256(digest)
            if isinstance(scene, str) and isinstance(digest, str):
                expected_names.append(scene)
                expected[scene] = digest
    unique_expected_scenes = (
        expected_shape
        and len(expected_names) == len(expected_entries)
        and len(expected_names) == len(set(expected_names))
    )

    maps = manifest.get("maps")
    maps_shape = isinstance(maps, dict) and bool(maps)
    observed: dict[str, str] = {}
    map_entries_valid = maps_shape
    internally_consistent = maps_shape
    source_paths_safe = maps_shape
    if isinstance(maps, dict):
        for map_key, raw in maps.items():
            if not isinstance(map_key, str) or not map_key or not isinstance(raw, dict):
                map_entries_valid = False
                internally_consistent = False
                source_paths_safe = False
                continue
            scene = raw.get("scan")
            digest = raw.get("source_ply_sha256")
            source_ply = raw.get("source_ply")
            entry_key = raw.get("key")
            metadata_valid = (
                _is_plain_name(scene)
                and _is_sha256(digest)
                and entry_key == map_key
            )
            map_entries_valid &= metadata_valid
            expected_source_path = (
                f"{scene}/house_segmentations/{scene}.ply"
                if _is_plain_name(scene)
                else None
            )
            source_path = _safe_relative_path(source_ply)
            source_paths_safe &= bool(
                source_path is not None and source_path.as_posix() == expected_source_path
            )
            if not isinstance(scene, str) or not isinstance(digest, str):
                internally_consistent = False
                continue
            if scene in observed and observed[scene] != digest:
                internally_consistent = False
            observed[scene] = digest
    checks = {
        "golden_scene_entries_present": expected_shape,
        "golden_scene_metadata_valid": expected_metadata_valid,
        "unique_golden_scenes": unique_expected_scenes,
        "map_entries_present": maps_shape,
        "map_entries_valid": map_entries_valid,
        "source_ply_paths_safe_and_canonical": source_paths_safe,
        "internally_consistent_scene_digests": internally_consistent,
        "exact_scene_set": bool(expected)
        and set(observed) == set(expected)
        and unique_expected_scenes,
        "all_source_ply_sha256_match": bool(expected) and observed == expected,
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "expected_scene_geometry": expected,
        "observed_scene_geometry": observed,
        "checks": checks,
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_result(output: Path, result: dict[str, Any]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return 0 if result.get("status") == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("--golden", type=Path, required=True)
    checkpoint.add_argument("--content-manifest", type=Path, required=True)
    checkpoint.add_argument("--internnav-root", type=Path, required=True)
    checkpoint.add_argument("--output", type=Path, required=True)
    static_map = subparsers.add_parser("static-map")
    static_map.add_argument("--golden", type=Path, required=True)
    static_map.add_argument("--manifest", type=Path, required=True)
    static_map.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        golden = _load_json_object(args.golden)
        if args.command == "checkpoint":
            content = _load_json_object(args.content_manifest)
            result = checkpoint_audit(golden, content, args.internnav_root)
        else:
            manifest = _load_json_object(args.manifest)
            result = static_map_audit(golden, manifest)
    except Exception as error:
        result = {
            "schema_version": 1,
            "status": "FAIL",
            "checks": {"input_and_audit_valid": False},
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return _write_result(args.output, result)


if __name__ == "__main__":
    raise SystemExit(main())
