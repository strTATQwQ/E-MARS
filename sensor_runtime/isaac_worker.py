#!/usr/bin/env python3
"""Lease-internal standalone Isaac worker; intentionally never calls eval.py."""

from __future__ import annotations

import argparse
import os
import signal
import time
import traceback
from pathlib import Path

from .atomic import atomic_write_json
from .contract import CONTRACT_SHA256, PROFILES
from .isaac_eula import ISAAC_RUNTIME_PREFLIGHT_FILENAME, build_runtime_preflight
from .isaac_experience import (
    ISAAC_STARTUP_READY_FILENAME,
    build_startup_ready,
    prepare_frozen_experience,
)
from .runtime_policy import policy_for_session_profile, require_policy
from .wire import LatestBatchEmitter
from .workload import BoundedModelFreeWorkload


def main() -> int:
    worker_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--wrapper-usd", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    runtime_policy = policy_for_session_profile(args.profile)
    environment_policy = require_policy(
        os.environ.get("INTERNNAV_RUNTIME_POLICY", "strict_evidence")
    )
    if environment_policy.name != runtime_policy.name:
        raise RuntimeError("session profile and runtime policy environment disagree")
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    for handled in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(handled, request_stop)

    emitter = LatestBatchEmitter(
        args.socket,
        send_timeout_sec=runtime_policy.sensor_wire_send_timeout_sec,
    )
    app = None
    backend = None
    exit_code = 0
    failure_recorded = False
    try:
        runtime_preflight = build_runtime_preflight(
            os.environ,
            stdin_target=os.readlink("/proc/self/fd/0"),
            stdin_is_tty=os.isatty(0),
        )
        atomic_write_json(result_dir / ISAAC_RUNTIME_PREFLIGHT_FILENAME, runtime_preflight)
        import isaacsim
        from isaacsim import SimulationApp

        experience, launch_config, startup_policy = prepare_frozen_experience(
            isaacsim
        )
        app = SimulationApp(launch_config, experience=str(experience))
        simulation_app_ready = time.monotonic()
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.sensors.camera")
        enable_extension("isaacsim.sensors.experimental.physics")
        app.update()
        extensions_ready = time.monotonic()
        atomic_write_json(
            result_dir / ISAAC_STARTUP_READY_FILENAME,
            build_startup_ready(
                policy=startup_policy,
                simulation_app_elapsed_sec=simulation_app_ready - worker_started,
                extensions_ready_elapsed_sec=extensions_ready - worker_started,
            ),
        )
        from .isaac_backend import IsaacModelFreeBackend

        backend = IsaacModelFreeBackend(
            args.wrapper_usd.resolve(),
            render_stamp_deviation_tolerance_ns=(
                runtime_policy.render_stamp_deviation_tolerance_ns
            ),
            render_resync_limit=runtime_policy.render_resync_limit,
        )
        emitter.start()
        workload = BoundedModelFreeWorkload(backend, emitter, PROFILES[args.profile])
        summary = workload.run_bounded(lambda: stop)
        emitter_status = emitter.flush()
        atomic_write_json(
            result_dir / "producer_completion.json",
            {
                "schema_version": 2,
                "status": "BOUNDED_DURATION_REACHED",
                "profile": args.profile,
                "runtime_policy": runtime_policy.as_dict(),
                "contract_sha256": CONTRACT_SHA256,
                "navigation_evaluator_used": False,
                "generation": summary.generation,
                "active_reset_count": summary.active_reset_count,
                "capture_count": summary.capture_count,
                "safe_stop_step_count": summary.safe_stop_step_count,
                "started_monotonic_ns": summary.started_monotonic_ns,
                "bounded_end_monotonic_ns": summary.bounded_end_monotonic_ns,
                "elapsed_sec": summary.elapsed_sec,
                "emitter": emitter_status,
                "backend_runtime": backend.runtime_evidence(),
            },
        )
        freeze_request = result_dir / "snapshot_freeze.request"
        workload.hold_until_freeze(lambda: stop, freeze_request.is_file)
        emitter_status = emitter.flush()
        last = emitter_status["last_sent"]
        if not isinstance(last, tuple) or len(last) != 2:
            raise RuntimeError("producer snapshot lacks a last sent identity")
        atomic_write_json(
            result_dir / "producer_snapshot_ack.json",
            {
                "schema_version": 2,
                "status": "PASS",
                "generation": int(last[0]),
                "sequence": int(last[1]),
                "capture_count": workload.capture_count,
                "safe_stop_step_count": workload._last_safe_count,
                "emitter": emitter_status,
                "backend_runtime": backend.runtime_evidence(),
            },
        )
        workload.safe_stop_until(lambda: stop)
    except BaseException as exc:
        failure_recorded = True
        exit_code = 2
        atomic_write_json(
            result_dir / "isaac_worker_failure.json",
            {
                "schema_version": 2,
                "status": "FAIL",
                "type": type(exc).__name__,
                "reason": str(exc),
                "traceback": traceback.format_exc().splitlines()[-40:],
            },
        )
    finally:
        cleanup_errors = []
        for name, action in (
            ("emitter", emitter.close),
            ("backend", lambda: None if backend is None else backend.close()),
            (
                "simulation_app",
                lambda: None if app is None or not app.is_running() else app.close(),
            ),
        ):
            try:
                action()
            except BaseException as exc:
                cleanup_errors.append(f"{name}: {type(exc).__name__}: {exc}")
        atomic_write_json(
            result_dir / "isaac_worker_cleanup.json",
            {
                "schema_version": 2,
                "status": "PASS" if not cleanup_errors else "FAIL",
                "actions_attempted": ["emitter", "backend", "simulation_app"],
                "errors": cleanup_errors,
            },
        )
        if cleanup_errors:
            exit_code = 2
            if not failure_recorded:
                atomic_write_json(
                    result_dir / "isaac_worker_failure.json",
                    {
                        "schema_version": 2,
                        "status": "FAIL",
                        "type": "CleanupError",
                        "reason": "; ".join(cleanup_errors),
                        "traceback": [],
                    },
                )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
