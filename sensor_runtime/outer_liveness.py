"""Kernel-backed outer-owner liveness shared across a bind mount."""

from __future__ import annotations

import errno
import json
import os
import time
from pathlib import Path
from typing import Any

from .atomic import atomic_write_json


class OuterOwnerGone(RuntimeError):
    """Raised when the inner process can acquire the outer-owner lock."""


def _fcntl() -> Any:
    if os.name != "posix":
        raise RuntimeError("outer-owner flock liveness requires POSIX")
    import fcntl

    return fcntl


class OuterAliveLock:
    """Hold one nonblocking exclusive flock for the complete inner lifetime."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self._stream: Any | None = None
        self._evidence: dict[str, Any] | None = None
        self._released = False

    def acquire(self) -> dict[str, Any]:
        if self._stream is not None or self.path.exists():
            raise RuntimeError("outer liveness lock was already claimed")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("x+b", buffering=0)
        try:
            _fcntl().flock(stream.fileno(), _fcntl().LOCK_EX | _fcntl().LOCK_NB)
            stat = os.fstat(stream.fileno())
            evidence = {
                "schema_version": 1,
                "status": "HELD",
                "mechanism": "flock_exclusive_nonblocking",
                "path": str(self.path),
                "owner_pid": os.getpid(),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "acquired_monotonic_ns": time.monotonic_ns(),
                "release_after_inner_probe": True,
            }
            stream.write((json.dumps(evidence, sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            raise
        self._stream = stream
        self._evidence = evidence
        return dict(evidence)

    def check(self) -> None:
        if self._stream is None or self._stream.closed:
            raise RuntimeError("outer liveness flock is not held")
        stat = os.fstat(self._stream.fileno())
        if self._evidence is None or (
            int(stat.st_dev), int(stat.st_ino)
        ) != (int(self._evidence["device"]), int(self._evidence["inode"])):
            raise RuntimeError("outer liveness flock identity changed")

    def release(
        self,
        evidence_path: Path,
        *,
        inner_probe_completed: bool,
        reason: str,
    ) -> dict[str, Any]:
        if self._released:
            raise RuntimeError("outer liveness flock was released more than once")
        if self._stream is None or self._evidence is None:
            raise RuntimeError("outer liveness flock was never acquired")
        stream, self._stream = self._stream, None
        try:
            _fcntl().flock(stream.fileno(), _fcntl().LOCK_UN)
        finally:
            stream.close()
        self._released = True
        payload = {
            "schema_version": 1,
            "status": "RELEASED",
            "mechanism": self._evidence["mechanism"],
            "path": self._evidence["path"],
            "device": self._evidence["device"],
            "inode": self._evidence["inode"],
            "released_monotonic_ns": time.monotonic_ns(),
            "inner_supervisor_zero_probe_completed": bool(inner_probe_completed),
            "reason": reason,
        }
        atomic_write_json(evidence_path, payload)
        return payload


def probe_outer_alive(path: Path) -> dict[str, Any]:
    """Prove another open-file description still owns the exclusive flock.

    A successful exclusive acquisition is a failure signal: it proves that
    the outer owner died or released the lock before inner cleanup completed.
    """

    target = Path(path).resolve()
    stream = target.open("r+b", buffering=0)
    try:
        stat = os.fstat(stream.fileno())
        try:
            _fcntl().flock(stream.fileno(), _fcntl().LOCK_EX | _fcntl().LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            return {
                "status": "OUTER_ALIVE",
                "mechanism": "flock_exclusive_nonblocking",
                "path": str(target),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "checked_monotonic_ns": time.monotonic_ns(),
            }
        else:
            _fcntl().flock(stream.fileno(), _fcntl().LOCK_UN)
            raise OuterOwnerGone("outer liveness flock became acquirable")
    finally:
        stream.close()
