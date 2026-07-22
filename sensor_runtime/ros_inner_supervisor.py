#!/usr/bin/env python3
"""Container-internal owner of the real bridge and sidecar PID/PGID lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .atomic import atomic_write_json
from .graph_handshake import (
    INNER_HANDSHAKE_ARTIFACTS,
    validate_inner_handshake_documents,
)
from .outer_liveness import OuterOwnerGone, probe_outer_alive
from .processes import ProcessRegistry, linux_start_ticks
from .pythonpath_policy import (
    require_frozen_setup_sha256,
    validate_child_pythonpath,
)
from .runtime_policy import require_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    control_root = args.control_root.resolve()
    result_dir = args.result_dir.resolve()
    runtime_policy = require_policy(
        os.environ.get("INTERNNAV_RUNTIME_POLICY", "strict_evidence")
    )
    session_profile = os.environ.get("INTERNNAV_SESSION_PROFILE", "")
    map_module = os.environ.get("INTERNNAV_T4_MAP_COMPANION_MODULE", "")
    map_enabled = session_profile == "completion_sim_map"
    if map_enabled:
        if runtime_policy.name != "completion_sim":
            raise RuntimeError("completion_sim_map requires completion_sim policy")
        if map_module != "t4_completion.map.companion":
            raise RuntimeError("completion_sim_map companion module is not frozen")
        if os.environ.get("INTERNNAV_SIMULATION_TARGET") != "isaac":
            raise RuntimeError("completion_sim_map rejects non-Isaac targets")
    elif map_module:
        raise RuntimeError("map companion is forbidden outside completion_sim_map")
    lifecycle_dir = result_dir / "inner_lifecycle"
    lifecycle_dir.mkdir(exist_ok=False)
    stop_request = result_dir / "inner_stop.request"
    inner_ready = result_dir / "inner_ready.json"
    inner_cleanup = result_dir / "inner_cleanup.json"
    outer_lock = result_dir / "outer_alive.lock"
    signaled: int | None = None
    supervisor_pid = os.getpid()
    supervisor_pgid = os.getpgid(0)
    if supervisor_pid != supervisor_pgid:
        raise RuntimeError("inner supervisor requires an isolated setsid process group")
    supervisor_start_ticks = linux_start_ticks(supervisor_pid)
    if supervisor_start_ticks is None:
        raise RuntimeError("cannot bind inner supervisor to a Linux process start identity")
    child_pythonpath = validate_child_pythonpath(
        control_root, os.environ.get("PYTHONPATH", "")
    )
    setup_pythonpath_sha256 = require_frozen_setup_sha256(
        os.environ.get("INTERNNAV_FROZEN_SETUP_PYTHONPATH_SHA256")
    )
    initial_outer_probe = probe_outer_alive(outer_lock)
    atomic_write_json(
        result_dir / "inner_supervisor_identity.json",
        {
            "schema_version": 3,
            "pid": supervisor_pid,
            "pgid": supervisor_pgid,
            "linux_start_ticks": supervisor_start_ticks,
            "isolated_process_group": True,
            "child_pythonpath": child_pythonpath,
            "setup_pythonpath_sha256": setup_pythonpath_sha256,
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
            "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
                "outer_liveness": initial_outer_probe,
                "runtime_policy": runtime_policy.as_dict(),
            "outer_liveness_watchdog_clock": "STEADY_MONOTONIC",
            "started_wall_unix": time.time(),
        },
    )

    def on_signal(signum: int, _frame: object) -> None:
        nonlocal signaled
        signaled = signum

    for handled in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(handled, on_signal)

    registry = ProcessRegistry(
        lifecycle_dir,
        socket_paths=[args.socket],
        term_timeout_sec=(
            10.0
            if map_enabled
            else 5.0
            if runtime_policy.name == "completion_sim"
            else 2.0
        ),
    )
    logs = result_dir / "logs"
    logs.mkdir(exist_ok=True)
    sidecar_out = (logs / "ros_sidecar.log").open("xb")
    bridge_out = (logs / "go2_sensor_bridge.log").open("xb")
    recorder_out = (logs / "downstream_recorder.log").open("xb")
    map_out = (
        (logs / "t4_map_companion.log").open("xb") if map_enabled else None
    )
    status = "FAIL"
    failure = ""

    def require_outer_alive(stage: str) -> None:
        try:
            probe_outer_alive(outer_lock)
        except OuterOwnerGone as exc:
            registry.record_failure(
                stage=stage, reason=f"outer_owner_flock_released: {exc}"
            )
            raise

    try:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.environ["PYTHONPATH"]
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        require_outer_alive("inner_before_first_spawn")
        registry.start(
            "go2_sensor_bridge",
            [
                sys.executable,
                str(control_root / "go2_sensor_bridge/go2_sensor_bridge/bridge_node.py"),
                "--ros-args",
                "-p",
                f"result_dir:={result_dir / 'bridge'}",
                "-p",
                "use_sim_time:=true",
                "-p",
                f"sensor_timeout_sec:={runtime_policy.bridge_timeout_sec}",
                "-p",
                f"runtime_policy:={runtime_policy.name}",
            ],
            stdout=bridge_out,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        registry.start(
            "sensor_ros_sidecar",
            [
                sys.executable,
                "-m",
                "sensor_runtime.ros_sidecar",
                "--socket",
                str(args.socket),
                "--result-dir",
                str(result_dir),
                "--ros-args",
                "-p",
                "use_sim_time:=true",
            ],
            stdout=sidecar_out,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        registry.start(
            "downstream_recorder",
            [
                sys.executable,
                "-m",
                "sensor_runtime.downstream_recorder",
                "--result-dir",
                str(result_dir / "downstream"),
                "--runtime-policy",
                runtime_policy.name,
                "--ros-args",
                "-p",
                "use_sim_time:=true",
            ],
            stdout=recorder_out,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        if map_enabled:
            if map_out is None:
                raise RuntimeError("map companion log was not created")
            environment["INTERNNAV_T4_MAP_COMPANION_ACK"] = "1"
            registry.start(
                "t4_map_companion",
                [
                    sys.executable,
                    "-m",
                    map_module,
                    "--result-dir",
                    str(result_dir),
                ],
                stdout=map_out,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        registry.start_monitor()
        deadline = time.monotonic() + 5.0
        handshake: dict[str, object] | None = None
        while True:
            if stop_request.exists():
                raise InterruptedError("outer requested stop before inner readiness")
            require_outer_alive("inner_pre_ready_outer_liveness")
            registry.check("inner_pre_ready")
            paths = {name: result_dir / name for name in INNER_HANDSHAKE_ARTIFACTS}
            if args.socket.exists() and all(path.is_file() for path in paths.values()):
                documents = {
                    name: json.loads(path.read_text(encoding="utf-8"))
                    for name, path in paths.items()
                }
                handshake = validate_inner_handshake_documents(documents)
                if not handshake["ready"]:
                    raise RuntimeError(
                        f"inner graph handshake artifacts are invalid: {handshake['errors']}"
                    )
                break
            if time.monotonic() >= deadline:
                missing = [name for name, path in paths.items() if not path.is_file()]
                raise TimeoutError(
                    "ROS graph handshake did not complete within 5 seconds; "
                    f"socket={args.socket.exists()} missing={missing}"
                )
            time.sleep(0.02)
        registry.check("inner_ready")
        atomic_write_json(
            inner_ready,
            {
                "schema_version": 2,
                "status": "READY",
                "roles": [record.as_dict() for record in registry.records],
                "socket_path": str(args.socket),
                "graph_handshake": handshake,
                "first_downstream_identity_required": [0, 0],
                "session_profile": session_profile,
                "map_companion_managed": map_enabled,
                "ready_wall_unix": time.time(),
                "outer_docker_client_is_not_cleanup_evidence": True,
            },
        )
        while True:
            if stop_request.exists():
                break
            require_outer_alive("inner_running_outer_liveness")
            registry.check("inner_running")
            if signaled is not None:
                raise InterruptedError(f"inner supervisor received signal {signaled}")
            time.sleep(0.02)
        registry.stop_boundary_snapshot(
            result_dir / "inner_stop_boundary.json", "inner_stop_boundary"
        )
        status = "PASS"
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        registry.record_failure(stage="inner_supervisor", reason=failure)
    finally:
        cleanup: dict[str, object] = {}
        cleanup_error = ""
        try:
            cleanup = registry.cleanup()
        except BaseException as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
            try:
                cleanup = json.loads((lifecycle_dir / "cleanup.json").read_text(encoding="utf-8"))
            except BaseException:
                cleanup = {"cleanup_confirmed": False, "pid_count": -1, "pgid_count": -1, "socket_count": -1}
        close_errors: list[str] = []
        for name, stream in (
            ("sidecar_log", sidecar_out),
            ("bridge_log", bridge_out),
            ("recorder_log", recorder_out),
            ("map_companion_log", map_out),
        ):
            if stream is None:
                continue
            try:
                stream.close()
            except BaseException as exc:
                close_errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if cleanup_error or close_errors:
            failure = "; ".join(filter(None, (failure, cleanup_error, *close_errors)))
            status = "FAIL"
        child_ok = cleanup.get("cleanup_confirmed") is True and not close_errors
        # This process cannot truthfully prove its own PID/PGID zero.  A
        # separate container-side recovery/probe must replace this artifact
        # after the docker exec client observes supervisor exit.
        atomic_write_json(
            inner_cleanup,
            {
                "schema_version": 3,
                "status": "AWAITING_SUPERVISOR_EXIT" if child_ok else "FAIL",
                "cleanup_confirmed": False,
                "child_cleanup_confirmed": child_ok,
                "bridge_sidecar_pid_count": int(cleanup.get("pid_count", -1)),
                "bridge_sidecar_pgid_count": int(cleanup.get("pgid_count", -1)),
                "supervisor_pid_count": 1,
                "supervisor_pgid_count": 1,
                "pid_count": int(cleanup.get("pid_count", -1)) + 1,
                "pgid_count": int(cleanup.get("pgid_count", -1)) + 1,
                "sensor_socket_count": int(cleanup.get("socket_count", -1)),
                "inner_ledger": "inner_lifecycle/process_ledger.json",
                "inner_cleanup_detail": "inner_lifecycle/cleanup.json",
                "supervisor_identity": "inner_supervisor_identity.json",
                "failure": failure or None,
            },
        )
    return 0 if status == "PASS" and not failure else 2


if __name__ == "__main__":
    raise SystemExit(main())
