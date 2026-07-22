#!/usr/bin/env python3
"""Build and verify a Git-object-exact T4 functional deployment payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


FUNCTIONAL_ROOTS = (
    "coordination",
    "sensor_runtime",
    "scripts",
    "go2_sensor_bridge",
    "t4_completion",
    "configs",
    "internvla_ros2_msgs",
    "internvla_ros2",
    "internvla_nav2_adapter",
    "internvla_go2_controller",
    "internvla_t4_sensors",
    "internvla_t4_recovery",
)

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ROOT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def _validate_roots(roots: tuple[str, ...]) -> None:
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("payload roots must be nonempty and unique")
    if any(_SAFE_ROOT.fullmatch(root) is None for root in roots):
        raise ValueError("payload root is not a safe top-level name")


def _git_tree(
    git_dir: Path, ref_sha: str, roots: tuple[str, ...]
) -> dict[str, tuple[str, bool]]:
    _validate_roots(roots)
    if _SHA1.fullmatch(ref_sha) is None:
        raise ValueError("ref_sha must be a full SHA-1")
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            f"--git-dir={git_dir}",
            "ls-tree",
            "-r",
            "-z",
            ref_sha,
            "--",
            *roots,
        ],
        check=True,
        capture_output=True,
    )
    expected: dict[str, tuple[str, bool]] = {}
    covered: set[str] = set()
    for record in completed.stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise RuntimeError("functional payload contains a non-regular Git entry")
        path = raw_path.decode("utf-8")
        if path in expected:
            raise RuntimeError(f"duplicate Git path: {path}")
        expected[path] = (object_id, mode == "100755")
        covered.add(path.split("/", 1)[0])
    missing_roots = sorted(set(roots) - covered)
    if missing_roots:
        raise RuntimeError(f"payload roots missing from Git tree: {missing_roots}")
    return expected


def _git_blob_id(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _archive_tree(archive_path: Path) -> dict[str, tuple[str, bool]]:
    observed: dict[str, tuple[str, bool]] = {}
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"payload contains a non-regular member: {member.name}"
                )
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise RuntimeError(f"payload contains an unsafe path: {member.name}")
            normalized = path.as_posix()
            if normalized in observed:
                raise RuntimeError(f"payload contains a duplicate member: {normalized}")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"payload member is unreadable: {normalized}")
            observed[normalized] = (
                _git_blob_id(stream.read()),
                bool(member.mode & 0o111),
            )
    return observed


def _compare(
    expected: dict[str, tuple[str, bool]],
    observed: dict[str, tuple[str, bool]],
) -> None:
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    mismatched = sorted(
        path
        for path in set(expected) & set(observed)
        if expected[path] != observed[path]
    )
    if missing or extra or mismatched:
        raise RuntimeError(
            "payload/ref mismatch "
            f"missing={missing} extra={extra} mismatched={mismatched}"
        )


def build_payload(
    git_dir: Path,
    ref_sha: str,
    archive_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if archive_path.exists() or manifest_path.exists():
        raise FileExistsError("payload outputs must be fresh")
    expected = _git_tree(git_dir, ref_sha, FUNCTIONAL_ROOTS)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            f"--git-dir={git_dir}",
            "archive",
            "--format=tar",
            f"--output={archive_path}",
            ref_sha,
            "--",
            *FUNCTIONAL_ROOTS,
        ],
        check=True,
    )
    _compare(expected, _archive_tree(archive_path))
    files = [
        {"path": path, "git_blob_id": value[0], "executable": value[1]}
        for path, value in sorted(expected.items())
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "FUNCTIONAL_PAYLOAD_READY",
        "ref_sha": ref_sha,
        "runtime_policy": "completion_sim",
        "runtime_target": "isaac_simulation",
        "model_host": "dgx_spark_only",
        "strict_evidence_modified": False,
        "real_go2_targeted": False,
        "roots": list(FUNCTIONAL_ROOTS),
        "path_count": len(files),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "files": files,
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest_stream:
        manifest_stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _manifest_expected(
    manifest_path: Path, expected_ref: str
) -> tuple[dict[str, Any], dict[str, tuple[str, bool]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported functional payload manifest")
    if manifest.get("status") != "FUNCTIONAL_PAYLOAD_READY":
        raise RuntimeError("functional payload is not ready")
    if manifest.get("ref_sha") != expected_ref or _SHA1.fullmatch(expected_ref) is None:
        raise RuntimeError("functional payload ref mismatch")
    if tuple(manifest.get("roots", [])) != FUNCTIONAL_ROOTS:
        raise RuntimeError("functional payload roots changed")
    if manifest.get("runtime_policy") != "completion_sim":
        raise RuntimeError("functional payload runtime policy changed")
    if manifest.get("model_host") != "dgx_spark_only":
        raise RuntimeError("functional payload model host changed")
    if manifest.get("strict_evidence_modified") is not False:
        raise RuntimeError("strict evidence mutation was claimed")
    if manifest.get("real_go2_targeted") is not False:
        raise RuntimeError("real Go2 cannot consume this payload")
    expected: dict[str, tuple[str, bool]] = {}
    records = manifest.get("files")
    if not isinstance(records, list):
        raise RuntimeError("functional payload file list is missing")
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("invalid functional payload file record")
        path_text = str(record.get("path", ""))
        path = PurePosixPath(path_text)
        if (
            not path_text
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in path_text
            or path.parts[0] not in FUNCTIONAL_ROOTS
        ):
            raise RuntimeError(f"unsafe manifest path: {path_text}")
        object_id = str(record.get("git_blob_id", ""))
        if _SHA1.fullmatch(object_id) is None:
            raise RuntimeError(f"invalid Git blob ID: {path_text}")
        if not isinstance(record.get("executable"), bool):
            raise RuntimeError(f"invalid executable claim: {path_text}")
        if path_text in expected:
            raise RuntimeError(f"duplicate manifest path: {path_text}")
        expected[path_text] = (object_id, bool(record["executable"]))
    if int(manifest.get("path_count", -1)) != len(expected):
        raise RuntimeError("functional payload path count changed")
    return manifest, expected


def _deployed_tree(root: Path) -> dict[str, tuple[str, bool]]:
    observed: dict[str, tuple[str, bool]] = {}
    root_real = root.resolve(strict=True)
    for payload_root in FUNCTIONAL_ROOTS:
        source = root / payload_root
        if not source.is_dir() or source.is_symlink():
            raise RuntimeError(f"deployed root is missing or unsafe: {payload_root}")
        for directory, directory_names, file_names in os.walk(source):
            directory_path = Path(directory)
            for name in tuple(directory_names):
                candidate = directory_path / name
                if candidate.is_symlink():
                    raise RuntimeError(f"deployed tree contains symlink: {candidate}")
            for name in file_names:
                candidate = directory_path / name
                metadata = candidate.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    raise RuntimeError(
                        f"deployed tree contains non-regular file: {candidate}"
                    )
                resolved = candidate.resolve(strict=True)
                if os.path.commonpath((str(root_real), str(resolved))) != str(root_real):
                    raise RuntimeError(f"deployed path escaped root: {candidate}")
                relative = candidate.relative_to(root).as_posix()
                observed[relative] = (
                    _git_blob_id(candidate.read_bytes()),
                    bool(metadata.st_mode & 0o111),
                )
    return observed


def verify_tree(root: Path, manifest_path: Path, expected_ref: str) -> int:
    _, expected = _manifest_expected(manifest_path, expected_ref)
    _compare(expected, _deployed_tree(root))
    return len(expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--git-dir", type=Path, required=True)
    build.add_argument("--ref-sha", required=True)
    build.add_argument("--archive", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    verify = subparsers.add_parser("verify-tree")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-ref", required=True)
    args = parser.parse_args()
    if args.command == "build":
        manifest = build_payload(
            args.git_dir.resolve(), args.ref_sha, args.archive, args.manifest
        )
        print(
            f"FUNCTIONAL_PAYLOAD_OK paths={manifest['path_count']} "
            f"ref={manifest['ref_sha']}"
        )
        return 0
    count = verify_tree(args.root, args.manifest, args.expected_ref)
    print(f"FUNCTIONAL_TREE_OK paths={count} ref={args.expected_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
