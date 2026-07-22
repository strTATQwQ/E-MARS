#!/usr/bin/env python3
"""POSIX-only inner-like fixture for cross-boundary flock owner death."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .atomic import atomic_write_json
from .outer_liveness import OuterOwnerGone, probe_outer_alive
from .processes import ProcessRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    registry = ProcessRegistry(args.result_dir / "inner_lifecycle")
    registry.start(
        "fixture_inner_role",
        [sys.executable, "-m", "sensor_runtime.pdeath_fixture_child"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    registry.start_monitor()
    atomic_write_json(
        args.result_dir / "monitor_ready.json",
        {"status": "READY", "outer_probe": probe_outer_alive(args.lock)},
    )
    detected = False
    while not detected:
        registry.check("fixture_outer_liveness")
        try:
            probe_outer_alive(args.lock)
        except OuterOwnerGone as exc:
            registry.record_failure(stage="outer_liveness", reason=str(exc))
            detected = True
        time.sleep(0.02)
    try:
        cleanup = registry.cleanup()
    except BaseException:
        cleanup = json.loads(registry.cleanup_path.read_text(encoding="utf-8"))
    atomic_write_json(
        args.result_dir / "liveness_cleanup.json",
        {
            "status": "PASS"
            if detected
            and cleanup.get("residual_cleanup_confirmed") is True
            and int(cleanup.get("pid_count", -1)) == 0
            and int(cleanup.get("pgid_count", -1)) == 0
            else "FAIL",
            "outer_owner_gone_detected": detected,
            "cleanup": cleanup,
        },
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
