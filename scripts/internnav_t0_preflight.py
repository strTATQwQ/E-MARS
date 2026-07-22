#!/usr/bin/env python3
"""Fail-closed preflight shared by the official InternNav T0 launchers."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path


COMMIT = "1d8d078aa9031a4a02a1ae05844d49a1768a10e4"
SUBMODULES = {
    "internnav/model/basemodel/LongCLIP": "3966af9ae9331666309a22128468b734db4672a7",
    "third_party/diffusion-policy": "5ba07ac6661db573af695b419a7947ecb704690f",
}
EPISODE_COUNTS = {"gate1": 1, "canary": 5, "pilot": 20}


def fail(message: str) -> None:
    raise SystemExit(f"T0 preflight failed: {message}")


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode:
        fail(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_status(path: Path, required: str) -> dict:
    if not path.is_file():
        fail(f"required status file is absent: {path}")
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("status") != "PASS":
        fail(f"{path.name} is {status.get('status', 'UNKNOWN')}, not PASS")
    if status.get("gate") != required:
        fail(f"{path.name} does not attest {required}")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("server", "eval"), required=True)
    parser.add_argument("--phase", choices=tuple(EPISODE_COUNTS))
    parser.add_argument("--internnav-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()

    root = args.internnav_root.resolve()
    control = args.control_root.resolve()
    load_status(control / "results/internnav_t0/gate0_status.json", "gate0")

    if git(root, "rev-parse", "HEAD") != COMMIT:
        fail("InternNav HEAD is not the frozen 0.3.1 commit")
    if git(root, "status", "--porcelain"):
        fail("frozen InternNav worktree is dirty")
    for relative, expected in SUBMODULES.items():
        if git(root / relative, "rev-parse", "HEAD") != expected:
            fail(f"submodule {relative} is not at its frozen commit")

    if args.role == "server":
        print("T0 server preflight PASS")
        return

    if args.phase is None or args.dataset_root is None:
        fail("eval role requires --phase and --dataset-root")
    prerequisite = {"gate1": "gate0", "canary": "gate1", "pilot": "canary"}[args.phase]
    if prerequisite != "gate0":
        load_status(
            control / f"results/internnav_t0/{prerequisite}_status.json",
            prerequisite,
        )

    episode_file = args.dataset_root / "val_unseen" / "val_unseen.json.gz"
    manifest_file = args.dataset_root / "manifest.json"
    if not episode_file.is_file() or not manifest_file.is_file():
        fail(f"{args.phase} episode overlay or manifest is absent")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    with gzip.open(episode_file, "rt", encoding="utf-8") as stream:
        raw_count = len(json.load(stream)["episodes"])
    expected_count = EPISODE_COUNTS[args.phase]
    if raw_count != expected_count or manifest.get("episode_count") != expected_count:
        fail(f"{args.phase} overlay does not contain exactly {expected_count} episodes")
    episode_keys = manifest.get("episode_keys", [])
    if len(episode_keys) != expected_count or len(set(episode_keys)) != expected_count:
        fail(f"{args.phase} manifest lacks {expected_count} unique frozen episode keys")
    if manifest.get("post_official_filter_episode_count") != expected_count:
        fail(f"{args.phase} overlay was not attested after official filters")
    if manifest.get("source_revision") != "7b05993b21813c3787f2f7f604bfc22b80c48c8e":
        fail("episode manifest is not pinned to InternData-N1 revision")
    if manifest.get("source_sha256") != "853673f8faadb26f883d57d71828cf895345c89fa594a7b918a78862318aeb31":
        fail("episode manifest source hash does not match official val_unseen")
    if manifest.get("overlay_sha256") != sha256(episode_file):
        fail("episode overlay hash differs from its manifest")
    if args.phase == "pilot" and manifest.get("overlap_with_canary") != []:
        fail("pilot manifest does not attest zero overlap with canary")
    print(f"T0 {args.phase} preflight PASS")


if __name__ == "__main__":
    main()
