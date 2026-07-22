#!/usr/bin/env python3
"""Run one preparation command and reap its same-PGID descendants on TERM."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

from .processes import live_group_members


def _members_except_self(pgid: int) -> list[dict[str, object]]:
    return [item for item in live_group_members(pgid) if int(item["pid"]) != os.getpid()]


def _terminate_members(
    pgid: int, *, term_timeout_sec: float = 2.0
) -> list[dict[str, object]]:
    members = _members_except_self(pgid)
    for item in members:
        try:
            os.kill(int(item["pid"]), signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + term_timeout_sec
    members = _members_except_self(pgid)
    while members and time.monotonic() < deadline:
        time.sleep(0.02)
        members = _members_except_self(pgid)
    for item in members:
        try:
            os.kill(int(item["pid"]), signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 2.0
    members = _members_except_self(pgid)
    while members and time.monotonic() < deadline:
        time.sleep(0.02)
        members = _members_except_self(pgid)
    return members


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if os.name != "posix" or not command:
        raise RuntimeError("managed command requires POSIX and a command")
    if os.getpid() != os.getpgid(0):
        raise RuntimeError("managed command must be its process-group leader")
    received: int | None = None

    def interrupted(signum: int, _frame: object) -> None:
        nonlocal received
        received = signum

    for handled in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(handled, interrupted)
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    pgid = os.getpgid(0)
    term_timeout_sec = (
        8.0
        if os.environ.get("INTERNNAV_RUNTIME_POLICY") == "completion_sim"
        else 2.0
    )
    while received is None:
        code = child.poll()
        if code is not None:
            residual = _members_except_self(pgid)
            if residual:
                _terminate_members(pgid, term_timeout_sec=term_timeout_sec)
                return 2
            return int(code)
        time.sleep(0.02)
    residual = _terminate_members(pgid, term_timeout_sec=term_timeout_sec)
    code: int | None = None
    try:
        code = child.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass
    if residual:
        return 2
    # During coordinated registry cleanup the command itself receives TERM and
    # is expected to finish its own cleanup with zero.  Preserve that proof;
    # owner-death cases still return a signal-style status when no clean child
    # exit was observed.
    return 0 if code == 0 else 128 + int(received)


if __name__ == "__main__":
    raise SystemExit(main())
