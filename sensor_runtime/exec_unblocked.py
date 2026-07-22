#!/usr/bin/env python3
"""Exec trampoline that removes the parent's ledger-window signal mask."""

from __future__ import annotations

import os
import ctypes
import signal
import sys


def main() -> None:
    if (
        os.name != "posix"
        or len(sys.argv) < 5
        or sys.argv[1] != "--expected-parent-pid"
        or sys.argv[3] != "--"
    ):
        raise SystemExit(
            "usage: exec_unblocked.py --expected-parent-pid PID -- command [args ...]"
        )
    expected_parent = int(sys.argv[2])
    if sys.platform != "linux":
        raise RuntimeError("parent-death binding requires Linux prctl")
    libc = ctypes.CDLL(None, use_errno=True)
    PR_SET_PDEATHSIG = 1
    if libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # Close the race where the parent died before prctl was installed.
    if os.getppid() != expected_parent:
        raise SystemExit("registry parent died before parent-death binding")
    signal.pthread_sigmask(
        signal.SIG_UNBLOCK,
        tuple(
            dict.fromkeys(
                (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM))
            )
        ),
    )
    os.execvpe(sys.argv[4], sys.argv[4:], os.environ)


if __name__ == "__main__":
    main()
