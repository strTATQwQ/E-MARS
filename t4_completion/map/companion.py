"""Managed ROS companion for the coordinator-owned F1 online session.

The module is inert unless a shared 01R supervisor explicitly acknowledges it.
All child processes remain in the companion's inherited process group so the
shared ProcessRegistry can account for and clean the complete subtree.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .composer import ComposeRequest, compose_bundle
from .contract import DEFAULT_CONFIG_DIR, ContractError


SMOKE_DURATION_SEC = 60.0
TOPIC_TIMEOUT_SEC = 10.0
REQUIRED_TOPICS = (
    "/map",
    "/go2/lidar/points_base",
    "/local_costmap/costmap",
    "/global_costmap/costmap",
    "/cmd_vel_safe",
    "/internvla/stop",
)


def require_managed_boundary(result_dir: Path) -> Path:
    if os.name != "posix":
        raise RuntimeError("completion map companion requires a POSIX Isaac host")
    if os.environ.get("INTERNNAV_RUNTIME_POLICY") != "completion_sim":
        raise RuntimeError("completion map companion requires completion_sim")
    if os.environ.get("INTERNNAV_SIMULATION_TARGET") != "isaac":
        raise RuntimeError("completion map companion rejects non-Isaac targets")
    if os.environ.get("INTERNNAV_T4_MAP_COMPANION_ACK") != "1":
        raise RuntimeError("shared sensor supervisor did not acknowledge map companion")
    if os.getpid() != os.getpgrp():
        raise RuntimeError("map companion must be the shared registry process-group leader")
    session_value = os.environ.get("RESULT_DIR")
    if not session_value:
        raise RuntimeError("shared sensor session RESULT_DIR is absent")
    session_dir = Path(session_value).resolve()
    requested = result_dir.resolve()
    if requested != session_dir:
        raise RuntimeError("map companion result directory differs from sensor session")
    if not session_dir.is_dir():
        raise RuntimeError("sensor session result directory is missing")
    return session_dir


def _replace_tokens(argv: List[str], bundle: Path, result: Path) -> List[str]:
    return [
        item.replace("{bundle}", str(bundle)).replace("{result}", str(result))
        for item in argv
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes the complete inode while preserving the
        # create-once contract: an existing final artifact raises instead of
        # being silently overwritten.
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _probe_topic(topic: str, log_path: Path, environment: Mapping[str, str]) -> bool:
    command = [
        "timeout",
        str(int(TOPIC_TIMEOUT_SEC)),
        "ros2",
        "topic",
        "echo",
        topic,
        "--once",
    ]
    if topic == "/go2/lidar/points_base":
        command.extend(
            ["--qos-reliability", "best_effort", "--qos-durability", "volatile"]
        )
    elif topic == "/map":
        command.extend(
            ["--qos-reliability", "reliable", "--qos-durability", "transient_local"]
        )
    with log_path.open("xb") as stream:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=dict(environment),
            check=False,
        )
    if completed.returncode != 0 or log_path.stat().st_size == 0:
        return False
    if topic == "/internvla/stop":
        return any(
            line.strip() == "data: true"
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        )
    return True


def _process_group_members(group_id: int) -> List[int]:
    members: List[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("/proc is required for descendant cleanup proof")
    for item in proc.iterdir():
        if not item.name.isdigit():
            continue
        try:
            raw = (item / "stat").read_text(encoding="ascii")
            tail = raw[raw.rfind(")") + 2 :].split()
            process_group = int(tail[2])
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
        pid = int(item.name)
        if process_group == group_id and pid != os.getpid():
            members.append(pid)
    return sorted(members)


def _clean_group_descendants(group_id: int) -> List[int]:
    for sig, timeout_sec in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 2.0)):
        members = _process_group_members(group_id)
        if not members:
            return []
        for pid in members:
            try:
                # Recheck membership immediately before targeting the PID.
                if pid in _process_group_members(group_id):
                    os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if not _process_group_members(group_id):
                return []
            time.sleep(0.05)
    return _process_group_members(group_id)


def run(result_dir: Path) -> int:
    session_dir = require_managed_boundary(result_dir)
    requested_mode = os.environ.get("INTERNNAV_T4_MAP_NVBLOX_MODE", "shadow")
    health = os.environ.get("INTERNNAV_T4_MAP_NVBLOX_HEALTH", "unknown")
    config_dir = Path(
        os.environ.get("INTERNNAV_T4_MAP_CONFIG_DIR", str(DEFAULT_CONFIG_DIR))
    )
    request = ComposeRequest(requested_mode=requested_mode, nvblox_health=health)
    bundle = session_dir / "runtime" / "map_bundle"
    compose_bundle(bundle, request, config_dir)
    plan = json.loads((bundle / "launch_plan.json").read_text(encoding="utf-8"))
    map_dir = session_dir / "map"
    map_dir.mkdir(parents=True, exist_ok=False)
    logs = map_dir / "logs"
    logs.mkdir()
    ledger_path = map_dir / "companion_ledger.json"
    runtime_validation_path = map_dir / "runtime_validation.json"
    validation_path = map_dir / "smoke_validation.json"
    cleanup_path = map_dir / "companion_cleanup.json"

    environment = os.environ.copy()
    environment.update(plan["launch_environment"])
    environment["ROS2CLI_NO_DAEMON"] = "1"
    environment["INTERNNAV_T4_MAP_COMPANION_ACK"] = "1"
    processes: Dict[str, subprocess.Popen[bytes]] = {}
    streams: Dict[str, Any] = {}
    stop_requested = False

    def on_signal(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    for signum in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)):
        signal.signal(signum, on_signal)

    started = time.monotonic()
    failure = ""
    fallback_event: Optional[Mapping[str, Any]] = None
    topic_checks: Dict[str, bool] = {}
    runtime_ready = False
    runtime_started: Optional[float] = None
    try:
        for process in plan["processes"]:
            role = str(process["role"])
            log_stream = (logs / "{}.log".format(role)).open("xb")
            streams[role] = log_stream
            child = subprocess.Popen(
                _replace_tokens(list(process["argv"]), bundle, session_dir),
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                env=environment,
                close_fds=True,
                # Do not start a new session: the shared supervisor owns this
                # companion PGID and must also own every descendant.
                start_new_session=False,
            )
            processes[role] = child
        _write_json(
            ledger_path,
            {
                "schema_version": 1,
                "status": "RUNNING",
                "companion_pid": os.getpid(),
                "companion_pgid": os.getpgrp(),
                "children": [
                    {"role": role, "pid": child.pid, "required": bool(next(
                        item["required"] for item in plan["processes"] if item["role"] == role
                    ))}
                    for role, child in processes.items()
                ],
            },
        )
        # Give lifecycle nodes a bounded construction window, then prove that
        # actual topics (not ready files) exist and carry at least one message.
        construction_deadline = time.monotonic() + TOPIC_TIMEOUT_SEC
        while time.monotonic() < construction_deadline and not stop_requested:
            required_dead = [
                item["role"]
                for item in plan["processes"]
                if item["required"] and processes[item["role"]].poll() is not None
            ]
            if required_dead:
                raise RuntimeError("required map role exited: {}".format(required_dead))
            time.sleep(0.1)
        for topic in REQUIRED_TOPICS:
            safe_name = topic.strip("/").replace("/", "_") or "root"
            topic_checks[topic] = _probe_topic(
                topic, logs / "probe_{}.log".format(safe_name), environment
            )
        if not all(topic_checks.values()):
            raise RuntimeError("required map smoke topic probe failed")

        runtime_started = time.monotonic()
        deadline = runtime_started + SMOKE_DURATION_SEC
        while not stop_requested:
            for process in plan["processes"]:
                role = process["role"]
                child = processes[role]
                if child.poll() is None:
                    continue
                if role == "nvblox_mapper" and process["required"]:
                    completed = subprocess.run(
                        list(plan["active_runtime_fallback"]["command"]),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        check=False,
                        text=True,
                    )
                    if completed.returncode != 0:
                        raise RuntimeError("active Nvblox fallback command failed")
                    fallback_event = {
                        "from": "active",
                        "to": "shadow",
                        "child_exit_code": child.returncode,
                        "core_nav_sha256": plan["core_nav_sha256"],
                    }
                    process["required"] = False
                elif process["required"]:
                    raise RuntimeError("required map role exited: {}".format(role))
            if not runtime_ready and time.monotonic() >= deadline:
                runtime_ready = True
                _write_json(
                    runtime_validation_path,
                    {
                        "schema_version": 1,
                        "status": "PASS",
                        "runtime_policy": "completion_sim",
                        "profile": "completion_sim_map",
                        "target": "isaac_simulation_only",
                        "duration_sec": time.monotonic() - runtime_started,
                        "startup_elapsed_sec": runtime_started - started,
                        "requested_duration_sec": SMOKE_DURATION_SEC,
                        "topic_checks": topic_checks,
                        "effective_nvblox_mode": plan["effective_mode"],
                        "nvblox_process_started": "nvblox_mapper" in processes,
                        "fallback_event": fallback_event,
                        "core_nav_sha256": plan["core_nav_sha256"],
                        "strict_evidence_modified": False,
                        "continues_until_shared_stop": True,
                    },
                )
            time.sleep(0.1)
    except Exception as exc:
        failure = str(exc)
    finally:
        for child in reversed(list(processes.values())):
            if child.poll() is None:
                child.terminate()
        cleanup_deadline = time.monotonic() + 5.0
        for child in reversed(list(processes.values())):
            remaining = max(0.0, cleanup_deadline - time.monotonic())
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2.0)
        for stream in streams.values():
            stream.close()

    residual = [role for role, child in processes.items() if child.poll() is None]
    residual_group_pids = _clean_group_descendants(os.getpgrp())
    cleanup_status = not residual and not residual_group_pids
    _write_json(
        cleanup_path,
        {
            "schema_version": 1,
            "status": "PASS" if cleanup_status else "FAIL",
            "companion_pid": os.getpid(),
            "companion_pgid": os.getpgrp(),
            "child_roles_alive": residual,
            "descendant_pids_alive": residual_group_pids,
            "owned_socket_count": 0,
        },
    )
    status = (
        "PASS"
        if not failure
        and runtime_ready
        and cleanup_status
        and all(topic_checks.values())
        and runtime_started is not None
        and time.monotonic() - runtime_started >= SMOKE_DURATION_SEC
        else "FAIL"
    )
    _write_json(
        validation_path,
        {
            "schema_version": 1,
            "status": status,
            "runtime_policy": "completion_sim",
            "target": "isaac_simulation_only",
            "duration_sec": time.monotonic() - started,
            "runtime_duration_sec": (
                None
                if runtime_started is None
                else time.monotonic() - runtime_started
            ),
            "requested_duration_sec": SMOKE_DURATION_SEC,
            "runtime_validation_written": runtime_ready,
            "topic_checks": topic_checks,
            "fallback_event": fallback_event,
            "nvblox_process_started": "nvblox_mapper" in processes,
            "failure": failure or None,
            "child_exit_codes": {
                role: child.returncode for role, child in processes.items()
            },
            "residual_roles": residual,
            "residual_group_pids": residual_group_pids,
            "core_nav_sha256": plan["core_nav_sha256"],
            "strict_evidence_modified": False,
        },
    )
    return 0 if status == "PASS" else 2


def main(argv: List[str] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.result_dir)
    except (ContractError, Exception) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
