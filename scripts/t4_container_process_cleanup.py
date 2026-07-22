#!/usr/bin/env python3
"""Gracefully stop every trial-owned process in the dedicated T4 container."""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path


BASE_TARGET_TOKENS = (
    "/workspaces/isaac/install/",
    "/opt/ros/jazzy/lib/nav2_",
    "/opt/ros/jazzy/lib/opennav_",
    "/opt/ros/jazzy/lib/nvblox_ros/",
    "/opt/ros/jazzy/bin/ros2 launch nav2_bringup",
    "/opt/ros/jazzy/bin/ros2 run ",
)


def target_tokens(control_root: Path) -> tuple[str, ...]:
    return BASE_TARGET_TOKENS + (
        f"{control_root.as_posix()}/t4_completion/map/warn_relay.py",
    )


def ancestors(pid: int) -> set[int]:
    values = {pid}
    while pid > 1:
        try:
            fields = (Path("/proc") / str(pid) / "stat").read_text().split()
            pid = int(fields[3])
        except (FileNotFoundError, IndexError, ValueError):
            break
        values.add(pid)
    return values


def targets(excluded: set[int], tokens: tuple[str, ...]) -> list[int]:
    values: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded or pid <= 1:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(token in command for token in tokens):
            values.append(pid)
    return sorted(values, reverse=True)


def send(pids: list[int], signum: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, signum)
        except (PermissionError, ProcessLookupError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.control_root.is_absolute()
        or not args.control_root.is_dir()
        or args.control_root.is_symlink()
        or args.control_root.resolve() != args.control_root
    ):
        raise RuntimeError("container cleanup control root is unsafe")
    tokens = target_tokens(args.control_root)
    excluded = ancestors(os.getpid())
    initial = targets(excluded, tokens)
    send(initial, signal.SIGINT)
    deadline = time.monotonic() + 6.0
    remaining = initial
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = targets(excluded, tokens)
    if remaining:
        send(remaining, signal.SIGTERM)
        time.sleep(1.0)
        remaining = targets(excluded, tokens)
    if remaining:
        send(remaining, signal.SIGKILL)
    print(
        f"trial_processes_initial={len(initial)} "
        f"forced_kill={len(remaining)}"
    )


if __name__ == "__main__":
    main()
