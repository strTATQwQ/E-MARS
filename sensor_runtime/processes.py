"""Durable required-process supervision and measured PGID/socket cleanup."""

from __future__ import annotations

import os
import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .atomic import atomic_write_json


SIGHUP = getattr(signal, "SIGHUP", signal.SIGTERM)
HANDLED_SIGNALS = tuple(dict.fromkeys((signal.SIGINT, signal.SIGTERM, SIGHUP)))


class RequiredProcessExited(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessRecord:
    role: str
    pid: int
    pgid: int
    required: bool
    expected_long_running: bool
    started_wall_unix: float
    started_monotonic_ns: int
    linux_start_ticks: int
    command_summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "pid": self.pid,
            "pgid": self.pgid,
            "required": self.required,
            "expected_long_running": self.expected_long_running,
            "started_wall_unix": self.started_wall_unix,
            "started_monotonic_ns": self.started_monotonic_ns,
            "linux_start_ticks": self.linux_start_ticks,
            "command_summary": self.command_summary,
        }


def _summary(argv: Sequence[str]) -> str:
    """Return a deliberately sparse summary that cannot expose argv secrets."""

    executable = Path(argv[0]).name if argv else "missing"
    entrypoint = ""
    for value in argv[1:3]:
        if value.endswith((".py", ".sh")) or value == "-m":
            entrypoint = Path(value).name
            break
    return f"{executable} {entrypoint}".strip()


def live_group_members(pgid: int) -> list[dict[str, Any]]:
    """Measure non-zombie members of one process group using a fresh ps read."""

    completed = subprocess.run(
        ["ps", "-eo", "pid=,pgid=,stat="],
        check=True,
        capture_output=True,
        text=True,
        # The observer must never join the PGID it is measuring.  Otherwise a
        # sufficiently slow ps can report itself as a live workload residual.
        start_new_session=True,
    )
    members: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        pid_text, group_text, state = fields[:3]
        if int(group_text) == int(pgid) and not state.startswith("Z"):
            pid = int(pid_text)
            members.append(
                {
                    "pid": pid,
                    "pgid": int(group_text),
                    "state": state,
                    "linux_start_ticks": linux_start_ticks(pid),
                }
            )
    return members


def linux_start_ticks(pid: int) -> int | None:
    """Read Linux /proc starttime (clock ticks since boot) for reuse binding."""

    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    closing = text.rfind(")")
    if closing < 0:
        raise RuntimeError(f"malformed /proc/{pid}/stat")
    fields_after_comm = text[closing + 2 :].split()
    if len(fields_after_comm) <= 19:
        raise RuntimeError(f"short /proc/{pid}/stat")
    return int(fields_after_comm[19])


def _assert_leader_identity(record: ProcessRecord) -> None:
    observed = linux_start_ticks(record.pid)
    if observed is not None and observed != record.linux_start_ticks:
        raise RuntimeError(
            f"refusing reused PID/PGID for {record.role}: "
            f"start {observed} != {record.linux_start_ticks}"
        )


def socket_listener_count(path: Path) -> int:
    """Count kernel UNIX-socket entries for an exact path.

    Counting every exact entry is stricter than counting listeners alone and
    catches an unlinked/path-only false positive after cleanup.
    """

    table = Path("/proc/net/unix")
    if not table.is_file():
        raise RuntimeError("/proc/net/unix is required for socket-zero evidence")
    target = str(path)
    count = 0
    for line in table.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 8 and fields[7] == target:
            count += 1
    return count


def terminate_bound_process_group(
    *,
    role: str,
    pid: int,
    pgid: int,
    linux_start_identity: int,
    term_timeout_sec: float = 2.0,
    kill_timeout_sec: float = 2.0,
) -> dict[str, Any]:
    """Terminate one externally recorded group without risking PID reuse."""

    if min(pid, pgid, linux_start_identity) <= 0:
        raise ValueError("process identity values must all be positive")
    if pid == os.getpid() or pgid == os.getpgid(0):
        raise RuntimeError("refusing to terminate the recovery probe itself")
    observed_start = linux_start_ticks(pid)
    before = live_group_members(pgid)
    if observed_start is not None and observed_start != linux_start_identity:
        return {
            "role": role,
            "pid": pid,
            "pgid": pgid,
            "linux_start_ticks": linux_start_identity,
            "observed_start_ticks": observed_start,
            "identity_verified": False,
            "identity_error": "recorded PID was reused",
            "live_before": before,
            "live_after_term": before,
            "kill_sent": False,
            "live_after_kill": before,
        }
    if before:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + term_timeout_sec
    after_term = live_group_members(pgid)
    while after_term and time.monotonic() < deadline:
        time.sleep(0.05)
        after_term = live_group_members(pgid)
    if after_term:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + kill_timeout_sec
    after_kill = live_group_members(pgid)
    while after_kill and time.monotonic() < deadline:
        time.sleep(0.05)
        after_kill = live_group_members(pgid)
    return {
        "role": role,
        "pid": pid,
        "pgid": pgid,
        "linux_start_ticks": linux_start_identity,
        "observed_start_ticks": observed_start,
        "identity_verified": True,
        "identity_error": None,
        "live_before": before,
        "live_after_term": after_term,
        "kill_sent": bool(after_term),
        "live_after_kill": after_kill,
    }


class ProcessRegistry:
    """Spawn in isolated sessions, persist immediately, and monitor per role."""

    def __init__(
        self,
        result_dir: Path,
        socket_paths: Iterable[Path] = (),
        *,
        term_timeout_sec: float = 2.0,
        allowed_required_exit_codes: Iterable[int] = (0,),
    ) -> None:
        if os.name != "posix":
            raise RuntimeError("online process supervision requires a POSIX host")
        if term_timeout_sec <= 0.0:
            raise ValueError("TERM cleanup timeout must be positive")
        allowed = frozenset(int(value) for value in allowed_required_exit_codes)
        if 0 not in allowed:
            raise ValueError("required exit-code policy must include zero")
        self.result_dir = result_dir.resolve()
        self.ledger_path = self.result_dir / "process_ledger.json"
        self.failure_path = self.result_dir / "first_failure.json"
        self.cleanup_path = self.result_dir / "cleanup.json"
        self.socket_paths = tuple(Path(value) for value in socket_paths)
        self.term_timeout_sec = float(term_timeout_sec)
        self.allowed_required_exit_codes = allowed
        self._records: list[ProcessRecord] = []
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.RLock()
        self._failure = threading.Event()
        self._failure_writer_fault = ""
        self._monitor_stop = threading.Event()
        self._cleanup_started = False
        self._monitor_thread: threading.Thread | None = None
        atomic_write_json(
            self.ledger_path,
            {"schema_version": 2, "status": "EMPTY", "processes": []},
        )

    @property
    def records(self) -> tuple[ProcessRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def process(self, role: str) -> subprocess.Popen[bytes]:
        with self._lock:
            return self._processes[role]

    def _persist_ledger(self) -> None:
        atomic_write_json(
            self.ledger_path,
            {
                "schema_version": 2,
                "status": "ACTIVE",
                "updated_wall_unix": time.time(),
                "processes": [record.as_dict() for record in self._records],
            },
        )

    def start(
        self,
        role: str,
        argv: Sequence[str],
        *,
        stdout: Any,
        stderr: Any,
        env: Mapping[str, str] | None = None,
        required: bool = True,
        expected_long_running: bool = True,
        cwd: Path | None = None,
    ) -> subprocess.Popen[bytes]:
        """Start and ledger a group while handled signals are blocked.

        The helper does not return the process until the atomic ledger replace
        has completed, closing the former shell-variable/signal window.
        """

        if not role or role in self._processes:
            raise ValueError(f"duplicate or empty process role: {role!r}")
        if not argv:
            raise ValueError("process argv is required")
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, HANDLED_SIGNALS)
        process: subprocess.Popen[bytes] | None = None
        try:
            trampoline = Path(__file__).with_name("exec_unblocked.py")
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(trampoline),
                    "--expected-parent-pid",
                    str(os.getpid()),
                    "--",
                    *argv,
                ],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=None if env is None else dict(env),
                cwd=None if cwd is None else str(cwd),
                start_new_session=True,
            )
            pgid = os.getpgid(process.pid)
            if pgid != process.pid:
                raise RuntimeError(f"{role} did not receive an isolated PGID")
            start_ticks = linux_start_ticks(process.pid)
            if start_ticks is None:
                raise RuntimeError(f"{role} exited before its Linux start identity was captured")
            record = ProcessRecord(
                role=role,
                pid=process.pid,
                pgid=pgid,
                required=required,
                expected_long_running=expected_long_running,
                started_wall_unix=time.time(),
                started_monotonic_ns=time.monotonic_ns(),
                linux_start_ticks=start_ticks,
                command_summary=_summary(argv),
            )
            with self._lock:
                self._records.append(record)
                self._processes[role] = process
                self._persist_ledger()
        except BaseException as spawn_exc:
            if process is not None:
                emergency_pgid = process.pid
                try:
                    emergency_pgid = os.getpgid(process.pid)
                except ProcessLookupError:
                    pass
                before = live_group_members(emergency_pgid)
                if before:
                    try:
                        os.killpg(emergency_pgid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                after_term = self._wait_empty(emergency_pgid, 1.0)
                if after_term:
                    try:
                        os.killpg(emergency_pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                after_kill = self._wait_empty(emergency_pgid, 2.0)
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
                emergency = {
                    "schema_version": 2,
                    "status": "PASS" if not after_kill else "FAIL",
                    "role": role,
                    "pid": process.pid,
                    "pgid": emergency_pgid,
                    "linux_start_ticks": linux_start_ticks(process.pid),
                    "reason": f"{type(spawn_exc).__name__}: {spawn_exc}",
                    "live_before": before,
                    "live_after_term": after_term,
                    "live_after_kill": after_kill,
                }
                atomic_write_json(self.result_dir / "spawn_failure_recovery.json", emergency)
                if after_kill:
                    raise RuntimeError("spawn failure left live process-group members") from spawn_exc
            raise
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        return process

    def record_failure(self, *, stage: str, reason: str, role: str | None = None, exit_code: int | None = None) -> None:
        with self._lock:
            # In-memory failure is authoritative for control flow.  Evidence
            # persistence is attempted after fail-closed state is latched.
            self._failure.set()
            if self.failure_path.exists():
                return
            try:
                atomic_write_json(
                    self.failure_path,
                    {
                        "schema_version": 2,
                        "status": "FAIL",
                        "stage": stage,
                        "reason": reason,
                        "role": role,
                        "exit_code": exit_code,
                        "detected_wall_unix": time.time(),
                        "detected_monotonic_ns": time.monotonic_ns(),
                    },
                )
            except BaseException as exc:
                self._failure_writer_fault = f"{type(exc).__name__}: {exc}"

    def _monitor(self) -> None:
        while not self._monitor_stop.wait(0.02):
            with self._lock:
                if self._cleanup_started:
                    return
                items = list(self._processes.items())
                records = {record.role: record for record in self._records}
            for role, process in items:
                record = records[role]
                if not (record.required and record.expected_long_running):
                    continue
                code = process.poll()
                if code is not None:
                    self.record_failure(
                        stage="required_process_monitor",
                        reason="required_process_exited",
                        role=role,
                        exit_code=code,
                    )
                    return

    def start_monitor(self) -> None:
        if self._monitor_thread is not None:
            raise RuntimeError("process monitor already started")
        self._monitor_thread = threading.Thread(
            target=self._monitor,
            name="required-process-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def check(self, stage: str) -> None:
        if self._failure.is_set():
            suffix = (
                f"; first-failure evidence write failed: {self._failure_writer_fault}"
                if self._failure_writer_fault
                else ""
            )
            raise RequiredProcessExited(f"required process failed during {stage}{suffix}")
        if (
            self._monitor_thread is not None
            and not self._cleanup_started
            and not self._monitor_thread.is_alive()
        ):
            self.record_failure(stage=stage, reason="required_process_monitor_died")
            raise RequiredProcessExited(f"required process monitor died during {stage}")
        # Synchronous per-role poll prevents a caller from outrunning the 20 ms monitor.
        with self._lock:
            items = list(self._processes.items())
            records = {record.role: record for record in self._records}
        for role, process in items:
            record = records[role]
            if record.required and record.expected_long_running:
                code = process.poll()
                if code is not None:
                    self.record_failure(
                        stage=stage,
                        reason="required_process_exited",
                        role=role,
                        exit_code=code,
                    )
                    raise RequiredProcessExited(f"{role} exited with {code} during {stage}")

    def wait_preparation(self, role: str, timeout_sec: float) -> None:
        process = self._processes[role]
        try:
            code = process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            self.record_failure(stage="preparation", reason="preparation_timeout", role=role)
            raise
        if code != 0:
            self.record_failure(
                stage="preparation",
                reason="preparation_process_failed",
                role=role,
                exit_code=code,
            )
            raise RequiredProcessExited(f"{role} preparation exited with {code}")

    @staticmethod
    def _wait_empty(pgid: int, timeout_sec: float) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout_sec
        members = live_group_members(pgid)
        while members and time.monotonic() < deadline:
            time.sleep(0.05)
            members = live_group_members(pgid)
        return members

    def cleanup(self) -> dict[str, Any]:
        """TERM/KILL every ledger PGID and confirm fresh PID/PGID/socket zero."""

        self.begin_cleanup()
        groups: list[dict[str, Any]] = []
        cleanup_errors: list[str] = []
        for record in reversed(self.records):
            row_errors: list[str] = []
            identity_verified = True
            try:
                _assert_leader_identity(record)
                identity_error = None
            except BaseException as exc:
                identity_verified = False
                identity_error = f"{type(exc).__name__}: {exc}"
                row_errors.append(identity_error)
            try:
                before = live_group_members(record.pgid)
            except BaseException as exc:
                before = [{"measurement_error": f"{type(exc).__name__}: {exc}"}]
                row_errors.append(f"live_before: {type(exc).__name__}: {exc}")
            if identity_verified and before:
                try:
                    os.killpg(record.pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except BaseException as exc:
                    row_errors.append(f"term: {type(exc).__name__}: {exc}")
            try:
                after_term = self._wait_empty(record.pgid, self.term_timeout_sec)
            except BaseException as exc:
                after_term = [{"measurement_error": f"{type(exc).__name__}: {exc}"}]
                row_errors.append(f"after_term: {type(exc).__name__}: {exc}")
            kill_sent = bool(after_term)
            if identity_verified and after_term:
                try:
                    os.killpg(record.pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except BaseException as exc:
                    row_errors.append(f"kill: {type(exc).__name__}: {exc}")
            try:
                after_kill = self._wait_empty(record.pgid, 2.0)
            except BaseException as exc:
                after_kill = [{"measurement_error": f"{type(exc).__name__}: {exc}"}]
                row_errors.append(f"after_kill: {type(exc).__name__}: {exc}")
            process = self._processes.get(record.role)
            exit_code: int | None = None
            if process is not None:
                try:
                    exit_code = process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    row_errors.append("leader wait timed out")
                except BaseException as exc:
                    row_errors.append(f"leader_wait: {type(exc).__name__}: {exc}")
            clean_exit = exit_code in self.allowed_required_exit_codes
            if record.required and not clean_exit:
                row_errors.append(
                    f"required role {record.role} did not exit cleanly: {exit_code}"
                )
            groups.append(
                {
                    "role": record.role,
                    "pid": record.pid,
                    "pgid": record.pgid,
                    "identity_verified": identity_verified,
                    "identity_error": identity_error,
                    "live_before": before,
                    "live_after_term": after_term,
                    "kill_sent": kill_sent,
                    "live_after_kill": after_kill,
                    "exit_code": exit_code,
                    "clean_exit": clean_exit,
                    "errors": row_errors,
                }
            )
            cleanup_errors.extend(f"{record.role}: {value}" for value in row_errors)

        sockets: list[dict[str, Any]] = []
        for path in self.socket_paths:
            try:
                entries_before_unlink = socket_listener_count(path)
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                entries_after_unlink = socket_listener_count(path)
                sockets.append(
                    {
                        "path": str(path),
                        "entries_before_unlink": entries_before_unlink,
                        "entries_after_unlink": entries_after_unlink,
                        "path_exists_after": path.exists(),
                        "error": None,
                    }
                )
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                cleanup_errors.append(f"socket {path}: {error}")
                sockets.append(
                    {
                        "path": str(path),
                        "entries_before_unlink": -1,
                        "entries_after_unlink": -1,
                        "path_exists_after": path.exists(),
                        "error": error,
                    }
                )

        live_after = [item for item in groups if item["live_after_kill"]]
        bad_sockets = [
            item
            for item in sockets
            if item["entries_after_unlink"] != 0 or item["path_exists_after"] or item["error"]
        ]
        payload = {
            "schema_version": 3,
            "term_timeout_sec": self.term_timeout_sec,
            "allowed_required_exit_codes": sorted(self.allowed_required_exit_codes),
            "status": "PASS" if not live_after and not bad_sockets and not cleanup_errors else "FAIL",
            "cleanup_confirmed": not live_after and not bad_sockets and not cleanup_errors,
            "residual_cleanup_confirmed": not live_after and not bad_sockets,
            "errors": cleanup_errors,
            "pid_count": sum(len(item["live_after_kill"]) for item in groups),
            "pgid_count": sum(bool(item["live_after_kill"]) for item in groups),
            "socket_count": sum(item["entries_after_unlink"] for item in sockets),
            "groups": groups,
            "sockets": sockets,
            "measured_wall_unix": time.time(),
        }
        atomic_write_json(self.cleanup_path, payload)
        if payload["status"] != "PASS":
            raise RuntimeError("measured PGID/socket cleanup failed")
        return payload

    def begin_cleanup(self) -> None:
        """Stop early-exit monitoring before a coordinated inner shutdown."""

        with self._lock:
            self._cleanup_started = True
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(1.0)

    def stop_boundary_snapshot(self, path: Path, stage: str) -> dict[str, Any]:
        """Synchronously prove all required roles alive before expected stop."""

        self.check(stage)
        records = {record.role: record for record in self.records}
        roles: list[dict[str, Any]] = []
        for role, process in sorted(self._processes.items()):
            record = records[role]
            if not (record.required and record.expected_long_running):
                continue
            code = process.poll()
            roles.append(
                {
                    "role": role,
                    "pid": record.pid,
                    "pgid": record.pgid,
                    "linux_start_ticks": record.linux_start_ticks,
                    "alive": code is None,
                    "exit_code": code,
                }
            )
        payload = {
            "schema_version": 2,
            "status": "PASS",
            "stage": stage,
            "first_failure_absent": not self.failure_path.exists(),
            "roles": roles,
            "measured_monotonic_ns": time.monotonic_ns(),
        }
        if not roles or not payload["first_failure_absent"] or not all(row["alive"] for row in roles):
            payload["status"] = "FAIL"
            atomic_write_json(path, payload)
            raise RequiredProcessExited(f"required role failed at {stage}")
        atomic_write_json(path, payload)
        return payload


def cleanup_persisted_ledger(
    lifecycle_dir: Path,
    socket_paths: Iterable[Path],
    output_path: Path,
    *,
    expected_roles: set[str],
) -> dict[str, Any]:
    """Container recovery path when the original inner supervisor is gone."""

    ledger = json.loads((lifecycle_dir / "process_ledger.json").read_text(encoding="utf-8"))
    records = [ProcessRecord(**item) for item in ledger.get("processes", [])]
    observed_roles = {record.role for record in records}
    duplicate_roles = sorted(
        role for role in observed_roles if sum(item.role == role for item in records) != 1
    )
    contract_errors: list[str] = []
    if observed_roles != expected_roles or duplicate_roles or len(records) != len(expected_roles):
        contract_errors.append("persisted ledger roles are missing, duplicate, or unexpected")
    by_role: dict[str, list[ProcessRecord]] = {}
    for record in records:
        by_role.setdefault(record.role, []).append(record)
    safe_records: list[ProcessRecord] = []
    for role, role_records in by_role.items():
        unique: dict[tuple[int, int, int], ProcessRecord] = {}
        for item in role_records:
            unique.setdefault((item.pid, item.pgid, item.linux_start_ticks), item)
        safe_records.extend(unique.values())
        if len(unique) > 1:
            contract_errors.append(f"conflicting duplicate ledger records for role {role}")
    groups: list[dict[str, Any]] = []
    cleanup_errors: list[str] = list(contract_errors)
    for record in reversed(safe_records):
        try:
            row = terminate_bound_process_group(
                role=record.role,
                pid=record.pid,
                pgid=record.pgid,
                linux_start_identity=record.linux_start_ticks,
            )
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            cleanup_errors.append(f"{record.role}: {error}")
            try:
                after = live_group_members(record.pgid)
            except BaseException as measure_exc:
                after = [{"measurement_error": f"{type(measure_exc).__name__}: {measure_exc}"}]
            row = {
                "role": record.role,
                "pid": record.pid,
                "pgid": record.pgid,
                "identity_verified": False,
                "identity_error": error,
                "live_before": after,
                "live_after_term": after,
                "kill_sent": False,
                "live_after_kill": after,
            }
        row.update({"exit_code": None, "clean_exit": False, "leader_exit_observed": False})
        groups.append(row)
    sockets = []
    for value in socket_paths:
        path = Path(value)
        try:
            before = socket_listener_count(path)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            after = socket_listener_count(path)
            sockets.append(
                {
                    "path": str(path),
                    "entries_before_unlink": before,
                    "entries_after_unlink": after,
                    "path_exists_after": path.exists(),
                    "error": None,
                }
            )
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            cleanup_errors.append(f"socket {path}: {error}")
            sockets.append(
                {
                    "path": str(path),
                    "entries_before_unlink": -1,
                    "entries_after_unlink": -1,
                    "path_exists_after": path.exists(),
                    "error": error,
                }
            )
    failed_groups = [item for item in groups if item["live_after_kill"]]
    failed_sockets = [
        item
        for item in sockets
        if item["entries_after_unlink"] or item["path_exists_after"] or item["error"]
    ]
    residual_cleanup_confirmed = not failed_groups and not failed_sockets and not cleanup_errors
    exit_errors = [
        f"persisted recovery cannot observe clean leader exit for required role {item.role}"
        for item in safe_records
        if item.required
    ]
    payload = {
        "schema_version": 3,
        "status": "PASS" if residual_cleanup_confirmed and not exit_errors else "FAIL",
        "cleanup_confirmed": residual_cleanup_confirmed and not exit_errors,
        "residual_cleanup_confirmed": residual_cleanup_confirmed,
        "errors": cleanup_errors + exit_errors,
        "pid_count": sum(len(item["live_after_kill"]) for item in groups),
        "pgid_count": sum(bool(item["live_after_kill"]) for item in groups),
        "socket_count": sum(item["entries_after_unlink"] for item in sockets),
        "recovered_from_persisted_inner_ledger": True,
        "expected_roles": sorted(expected_roles),
        "observed_roles": sorted(observed_roles),
        "duplicate_roles": duplicate_roles,
        "ledger_contract_errors": contract_errors,
        "groups": groups,
        "sockets": sockets,
    }
    atomic_write_json(output_path, payload)
    return payload
