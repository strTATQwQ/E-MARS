#!/usr/bin/env python3
"""Restricted SSH stdio bridge for the remote Isaac lease holder.

The credential-bearing connect helper is deliberately repository-external.
This adapter accepts the ssh-shaped argv produced by with_resource_lease.sh,
verifies the fixed Isaac target and connected peer, and keeps one Paramiko
channel alive until the local lease controller writes RELEASE on stdin.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import BinaryIO


EXIT_USAGE = 64
EXIT_UNAVAILABLE = 69
EXPECTED_TARGET = "song@10.100.120.111"
EXPECTED_PEER = "10.100.120.111"
EXPECTED_PORT = "22"
EXPECTED_OPTIONS = {
    "ConnectTimeout=8",
    "ServerAliveInterval=5",
    "ServerAliveCountMax=2",
    "ExitOnForwardFailure=yes",
}


class UsageError(ValueError):
    """Raised for an ssh-shaped invocation outside the allowed contract."""


def _parse(argv: list[str]) -> tuple[Path, str, str]:
    if len(argv) < 4 or argv[0] != "--connect-helper":
        raise UsageError("--connect-helper PATH is required")
    helper = Path(argv[1])
    rest = argv[2:]
    port: str | None = None
    no_pty = False
    options: set[str] = set()
    positionals: list[str] = []
    index = 0
    while index < len(rest):
        value = rest[index]
        if value == "-T":
            if no_pty:
                raise UsageError("duplicate -T")
            no_pty = True
            index += 1
        elif value in {"-p", "-o"}:
            if index + 1 >= len(rest):
                raise UsageError(f"{value} requires a value")
            option_value = rest[index + 1]
            if value == "-p":
                if port is not None:
                    raise UsageError("duplicate SSH port")
                port = option_value
            elif option_value in options:
                raise UsageError("duplicate SSH option")
            else:
                options.add(option_value)
            index += 2
        elif value.startswith("-"):
            raise UsageError(f"unsupported SSH option: {value}")
        else:
            positionals.extend(rest[index:])
            break
    if port != EXPECTED_PORT:
        raise UsageError("only Isaac SSH port 22 is permitted")
    if not no_pty:
        raise UsageError("non-PTY mode is required")
    if options != EXPECTED_OPTIONS:
        raise UsageError("SSH holder options do not match the frozen contract")
    if len(positionals) != 2:
        raise UsageError("exactly one target and one remote command are required")
    target, remote_command = positionals
    if target != EXPECTED_TARGET:
        raise UsageError("only the fixed Isaac lease target is permitted")
    if (
        "/tmp/internnav_isaac.lock" not in remote_command
        or "lease-holder" not in remote_command
        or "'isaac'" not in remote_command
    ):
        raise UsageError("remote command is not an Isaac lease holder")
    if not helper.is_file():
        raise UsageError("connect helper is unavailable")
    return helper, target, remote_command


def _load_helper(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("internnav_external_connect_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("connect helper could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "connect", None)):
        raise RuntimeError("connect helper does not export connect()")
    return module


def _write_stream(stream: BinaryIO, data: bytes) -> None:
    stream.write(data)
    stream.flush()


def _bridge_stdin(channel: object, errors: list[BaseException]) -> None:
    try:
        stdin_fd = sys.stdin.fileno()
        while True:
            # os.read returns as soon as control bytes are available.  A
            # BufferedReader.read(size) can wait for EOF/full size, while the
            # child may inherit another writer for the lease FIFO; RELEASE
            # must never depend on that EOF.
            chunk = os.read(stdin_fd, 65536)
            if not chunk:
                break
            channel.sendall(chunk)  # type: ignore[attr-defined]
        try:
            channel.shutdown_write()  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass
    except BaseException as exc:  # daemon thread reports failure to main loop
        errors.append(exc)
        try:
            channel.close()  # type: ignore[attr-defined]
        except BaseException:
            pass


def run(argv: list[str]) -> int:
    try:
        helper_path, _target, remote_command = _parse(argv)
    except UsageError as exc:
        print(f"lease SSH bridge: {exc}", file=sys.stderr)
        return EXIT_USAGE

    client = None
    channel = None
    try:
        helper = _load_helper(helper_path)
        client = helper.connect()
        transport = client.get_transport()
        if transport is None or not transport.is_active():
            raise RuntimeError("connect helper returned an inactive transport")
        peer = transport.getpeername()
        peer_ip = peer[0] if isinstance(peer, (tuple, list)) and peer else None
        if peer_ip != EXPECTED_PEER:
            raise RuntimeError("connected peer is not the fixed Isaac host")
        transport.set_keepalive(5)

        channel = transport.open_session(timeout=8)
        channel.exec_command(remote_command)

        def close_connection(_signum: int, _frame: object) -> None:
            try:
                channel.close()
            finally:
                client.close()

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, close_connection)

        stdin_errors: list[BaseException] = []
        stdin_thread = threading.Thread(
            target=_bridge_stdin, args=(channel, stdin_errors), daemon=True
        )
        stdin_thread.start()

        while True:
            made_progress = False
            while channel.recv_ready():
                data = channel.recv(65536)
                if not data:
                    break
                _write_stream(sys.stdout.buffer, data)
                made_progress = True
            while channel.recv_stderr_ready():
                data = channel.recv_stderr(65536)
                if not data:
                    break
                _write_stream(sys.stderr.buffer, data)
                made_progress = True
            if stdin_errors:
                raise RuntimeError("lease control stream failed") from stdin_errors[0]
            if (
                channel.exit_status_ready()
                and not channel.recv_ready()
                and not channel.recv_stderr_ready()
            ):
                break
            if not made_progress:
                time.sleep(0.02)
        return int(channel.recv_exit_status())
    except BaseException as exc:
        # Do not stringify exceptions from credential-bearing helpers: some
        # authentication stacks include connection details in exception text.
        print(f"lease SSH bridge: {type(exc).__name__}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    finally:
        if channel is not None:
            try:
                channel.close()
            except BaseException:
                pass
        if client is not None:
            try:
                client.close()
            except BaseException:
                pass


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
