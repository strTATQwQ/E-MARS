#!/usr/bin/env python3
"""Verify that a deployment tar is an exact raw-byte view of one Git tree."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tarfile
from pathlib import Path


PAYLOAD_ROOTS = (
    "sensor_runtime",
    "scripts",
    "go2_sensor_bridge",
    "coordination",
    "t4_completion",
    "configs",
)


def _git_tree(git_dir: Path, ref_sha: str) -> dict[str, tuple[str, bool]]:
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
            *PAYLOAD_ROOTS,
        ],
        check=True,
        capture_output=True,
    )
    expected: dict[str, tuple[str, bool]] = {}
    for record in completed.stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        if object_type != "blob":
            continue
        expected[raw_path.decode("utf-8")] = (
            object_id,
            mode == "100755",
        )
    return expected


def verify_archive(git_dir: Path, ref_sha: str, archive_path: Path) -> int:
    expected = _git_tree(git_dir, ref_sha)
    observed: dict[str, tuple[str, bool]] = {}
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"payload contains a non-regular member: {member.name}"
                )
            if member.name in observed:
                raise RuntimeError(f"payload contains a duplicate member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"payload member is unreadable: {member.name}")
            data = stream.read()
            header = f"blob {len(data)}\0".encode("ascii")
            observed[member.name] = (
                hashlib.sha1(header + data).hexdigest(),
                bool(member.mode & 0o111),
            )

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
    return len(expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-dir", type=Path, required=True)
    parser.add_argument("--ref-sha", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    count = verify_archive(args.git_dir, args.ref_sha, args.archive)
    print(f"PAYLOAD_REF_BLOBS_OK paths={count} ref={args.ref_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
