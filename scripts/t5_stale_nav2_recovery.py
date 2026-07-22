#!/usr/bin/env python3
"""Evidence-bound, DGX_A-only recovery for stale T5 Nav2 process groups.

The production entry point transfers this file and the frozen structured
process auditor inline.  The remote host therefore does not need to trust an
old deployment tree.  Only three evidence-bound session leaders are eligible
for signalling: one ``ros2 launch nav2_bringup`` leader and the two explicit
``ros2 run nav2_lifecycle_manager`` leaders used by Lane A.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import posixpath
import re
import signal
import stat
import subprocess
import sys
import time
import types
from typing import Any, Callable, Iterable, Sequence


EXIT_FAIL_CLOSED = 75
ROLE = "dgx-a"
INTERNAL_ROLE = "dgx_a"
LANE_NAMESPACE = "/t5/lane_a"
OLD_ROOT_PREFIX = "/home/railgun/internnav-t1-t2/.t5-deployments/"
FIXED_OLD_ROOT = (
    "/home/railgun/internnav-t1-t2/.t5-deployments/"
    "t5d0020260719t135307-78cc4b1e7b1b-lane-a"
)
FIXED_MANIFEST_SHA256 = (
    "60e193f5ce3ca1060d72c65de74505b2ba5790cc10aa12386d0dfb8938232b85"
)
FIXED_SOURCE_CONFLICT_SHA256 = (
    "c57de9a1d838a66ca41323e09e7108da64a3736ca7d3d9ccdcdb9818baa784ed"
)
FIXED_LEADERS = {
    "nav2_bringup": {
        "pid": 247273,
        "ppid": 1,
        "pgid": 247273,
        "sid": 247273,
        "starttime": 1182700,
        "argv_sha256": "ca5e2b53ed8e43f1c6ddc0329ba50382641c0f5dc85ada16f54efdf05774acbf",
    },
    "lifecycle_navigation": {
        "pid": 247274,
        "ppid": 1,
        "pgid": 247274,
        "sid": 247274,
        "starttime": 1182700,
        "argv_sha256": "1369fc544b3c05491eb1a67237fd1b3445e5e2fbfcf1d0d7178fa63ea4aa93a4",
    },
    "lifecycle_collision": {
        "pid": 247275,
        "ppid": 1,
        "pgid": 247275,
        "sid": 247275,
        "starttime": 1182700,
        "argv_sha256": "aca8b7425d4d67ba121d53ed7123e8363701b4eb7604ca820e7868c528d004f8",
    },
}
OLD_ROOT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,239}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
T5_PORTS = (25137, 25138, 25139, 25140, 25141, 25239, 25240, 25241)
SIG_TERM = getattr(signal, "SIGTERM", 15)
SIG_KILL = getattr(signal, "SIGKILL", 9)
LEADER_ROLES = {
    "nav2_bringup",
    "lifecycle_navigation",
    "lifecycle_collision",
}
LIFECYCLE_NODE_NAMES = {
    "lifecycle_navigation": "lifecycle_manager_navigation_t4_dgx",
    "lifecycle_collision": "lifecycle_manager_collision_t4_dgx",
}
EXTRA_COMPUTE_IDENTITIES = {
    "docking_server",
    "lifecycle_manager",
    "route_server",
}


class RecoveryError(RuntimeError):
    """A condition that must leave the DGX quarantine marker armed."""


def validate_old_root(value: str) -> str:
    if not isinstance(value, str) or not value.startswith(OLD_ROOT_PREFIX):
        raise RecoveryError("old deployment root is outside the DGX_A prefix")
    if any(character in value for character in ("\0", "\r", "\n", "|")):
        raise RecoveryError("old deployment root contains an unsafe delimiter")
    normalized = posixpath.normpath(value)
    # DGX/WSL recovery hosts may still use Python 3.8. The prefix was proven
    # immediately above, so slicing avoids a newer string helper.
    name = normalized[len(OLD_ROOT_PREFIX) :]
    if (
        normalized != value
        or not name
        or "/" in name
        or OLD_ROOT_NAME_RE.fullmatch(name) is None
    ):
        raise RecoveryError("old deployment root is not one exact deployment")
    if normalized != FIXED_OLD_ROOT:
        raise RecoveryError("old deployment root is not the fixed attempt-10 root")
    return normalized


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _argv_sha256(raw: bytes) -> str:
    return _sha256(raw)


def _validate_sha(value: str, label: str) -> str:
    normalized = value.lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise RecoveryError(f"{label} is not a SHA-256")
    return normalized


def validate_evidence(
    raw: bytes, expected_sha256: str, old_root: str
) -> dict[str, Any]:
    expected_sha256 = _validate_sha(expected_sha256, "evidence SHA-256")
    if expected_sha256 != FIXED_MANIFEST_SHA256:
        raise RecoveryError("evidence SHA-256 is not the fixed recovery manifest")
    if not hmac.compare_digest(_sha256(raw), expected_sha256):
        raise RecoveryError("evidence SHA-256 mismatch")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryError("evidence is not valid UTF-8 JSON") from exc
    required = {
        "schema_version",
        "role",
        "deployment_root",
        "lane_namespace",
        "source_conflict_sha256",
        "captured_unix",
        "leaders",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RecoveryError("evidence top-level schema is not exact")
    if value["schema_version"] != 1 or value["role"] != ROLE:
        raise RecoveryError("evidence role/schema mismatch")
    if value["deployment_root"] != old_root:
        raise RecoveryError("evidence deployment root mismatch")
    if value["lane_namespace"] != LANE_NAMESPACE:
        raise RecoveryError("evidence lane namespace mismatch")
    source_conflict_sha256 = _validate_sha(
        value["source_conflict_sha256"], "source conflict SHA-256"
    )
    if source_conflict_sha256 != FIXED_SOURCE_CONFLICT_SHA256:
        raise RecoveryError("source conflict SHA-256 is not attempt-11 evidence")
    if not isinstance(value["captured_unix"], (int, float)) or isinstance(
        value["captured_unix"], bool
    ):
        raise RecoveryError("evidence captured_unix is invalid")
    leaders = value["leaders"]
    if not isinstance(leaders, list) or len(leaders) != 3:
        raise RecoveryError("evidence must contain exactly three leaders")
    leader_keys = {
        "role",
        "pid",
        "ppid",
        "pgid",
        "sid",
        "starttime",
        "state",
        "argv",
        "argv_sha256",
    }
    observed_roles: set[str] = set()
    observed_pids: set[int] = set()
    for leader in leaders:
        if not isinstance(leader, dict) or set(leader) != leader_keys:
            raise RecoveryError("leader evidence schema is not exact")
        leader_role = leader["role"]
        if leader_role not in LEADER_ROLES or leader_role in observed_roles:
            raise RecoveryError("leader role set is not exact")
        observed_roles.add(leader_role)
        for key in ("pid", "ppid", "pgid", "sid", "starttime"):
            if not isinstance(leader[key], int) or isinstance(leader[key], bool):
                raise RecoveryError(f"leader {key} must be an integer")
        pid = leader["pid"]
        if pid <= 1 or pid in observed_pids:
            raise RecoveryError("leader PID set is unsafe")
        observed_pids.add(pid)
        if leader["pgid"] != pid or leader["sid"] != pid:
            raise RecoveryError("leader is not PID=PGID=SID")
        if leader["ppid"] < 1 or leader["starttime"] <= 0:
            raise RecoveryError("leader ancestry/starttime is invalid")
        if leader["state"] not in {"R", "S", "D", "I"}:
            raise RecoveryError("leader evidence is not live")
        argv = leader["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or "\0" in item for item in argv)
        ):
            raise RecoveryError("leader argv is invalid")
        raw_argv = b"\0".join(item.encode("utf-8") for item in argv) + b"\0"
        if not hmac.compare_digest(
            _argv_sha256(raw_argv),
            _validate_sha(leader["argv_sha256"], "leader argv SHA-256"),
        ):
            raise RecoveryError("leader argv and argv SHA-256 disagree")
        fixed = FIXED_LEADERS[leader_role]
        if any(leader[key] != fixed[key] for key in fixed):
            raise RecoveryError(f"leader {leader_role} is not the fixed attempt-10 identity")
    if observed_roles != LEADER_ROLES:
        raise RecoveryError("leader role set is incomplete")
    return value


def _read_stable_file(path: Path, *, require_readonly: bool) -> tuple[bytes, os.stat_result]:
    if not path.is_absolute():
        raise RecoveryError("evidence path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RecoveryError(f"cannot open evidence safely: {type(exc).__name__}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RecoveryError("evidence must be a single-link regular file")
        if require_readonly and before.st_mode & 0o222:
            raise RecoveryError("archived evidence must be read-only")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read()
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise RecoveryError("evidence changed while being read")
        return raw, after
    finally:
        os.close(descriptor)


def stage_evidence(
    source: Path, destination: Path, expected_sha256: str, old_root: str
) -> dict[str, Any]:
    raw, source_stat = _read_stable_file(source, require_readonly=False)
    validate_evidence(raw, expected_sha256, validate_old_root(old_root))
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        raise RecoveryError("archived evidence destination must be new and absolute")
    descriptor = None
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        if os.name == "posix":
            parent_fd = os.open(
                destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    except OSError as exc:
        raise RecoveryError(f"cannot archive evidence atomically: {type(exc).__name__}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    archived, archived_stat = _read_stable_file(destination, require_readonly=True)
    if archived != raw:
        raise RecoveryError("archived evidence bytes changed")
    return {
        "schema_version": 1,
        "status": "PASS",
        "action": "STAGE_EVIDENCE",
        "source": str(source),
        "archive": str(destination),
        "sha256": _sha256(raw),
        "source_identity": [source_stat.st_dev, source_stat.st_ino],
        "archive_identity": [archived_stat.st_dev, archived_stat.st_ino],
        "archive_read_only": archived_stat.st_mode & 0o222 == 0,
    }


def verify_archived_evidence(
    path: Path, expected_sha256: str, old_root: str
) -> dict[str, Any]:
    raw, metadata = _read_stable_file(path, require_readonly=True)
    validate_evidence(raw, expected_sha256, validate_old_root(old_root))
    return {
        "schema_version": 1,
        "status": "PASS",
        "action": "VERIFY_EVIDENCE",
        "path": str(path),
        "sha256": _sha256(raw),
        "identity": [metadata.st_dev, metadata.st_ino],
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "read_only": metadata.st_mode & 0o222 == 0,
    }


def _decode_base64(value: str, label: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise RecoveryError(f"invalid base64 {label}") from exc


def _parse_stat(raw: str) -> dict[str, Any]:
    close = raw.rfind(")")
    if close < 1:
        raise RecoveryError("malformed proc stat")
    try:
        pid = int(raw[: raw.index(" ")])
        fields = raw[close + 2 :].split()
        return {
            "pid": pid,
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "starttime": int(fields[19]),
        }
    except (ValueError, IndexError) as exc:
        raise RecoveryError("malformed proc stat fields") from exc


def proc_snapshot(proc_root: Path = Path("/proc")) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not proc_root.is_dir():
        return [], ["proc_root_missing"]
    try:
        entries = sorted(proc_root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        return [], [f"proc_root:{type(exc).__name__}"]
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            first = _parse_stat((entry / "stat").read_text(encoding="utf-8"))
            raw_argv = (entry / "cmdline").read_bytes()
            status = (entry / "status").read_text(encoding="utf-8")
            second = _parse_stat((entry / "stat").read_text(encoding="utf-8"))
            if first != second:
                raise RecoveryError("process identity changed during proc read")
            uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
            uid = int(uid_line.split()[1])
            argv = [
                item.decode("utf-8")
                for item in raw_argv.split(b"\0")
                if item
            ]
            rows.append(
                {
                    **first,
                    "uid": uid,
                    "argv": argv,
                    "raw_argv_sha256": _argv_sha256(raw_argv),
                }
            )
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError, StopIteration, ValueError, RecoveryError) as exc:
            errors.append(f"pid={entry.name}:{type(exc).__name__}")
    return rows, errors


def _old_root_associated(argv: Sequence[str], old_root: str) -> bool:
    for token in argv:
        start = token.find(old_root)
        while start >= 0:
            end = start + len(old_root)
            left_ok = start == 0 or token[start - 1] in "=:,"
            right_ok = end == len(token) or token[end] == "/"
            if left_ok and right_ok:
                return True
            start = token.find(old_root, start + 1)
    return False


def _load_auditor(raw: bytes) -> types.ModuleType:
    module = types.ModuleType("t5_process_identity_audit_frozen")
    try:
        exec(compile(raw, "<frozen-t5-process-identity-audit>", "exec"), module.__dict__)
    except Exception as exc:  # noqa: BLE001 - a frozen audit load error is fatal.
        raise RecoveryError(f"cannot load structured auditor: {type(exc).__name__}") from exc
    if not callable(getattr(module, "process_identities", None)) or not callable(
        getattr(module, "forbidden_reason", None)
    ):
        raise RecoveryError("structured auditor API mismatch")
    return module


def _exact_ros_identity(
    row: dict[str, Any], leader: dict[str, Any], auditor: types.ModuleType, old_root: str
) -> None:
    if any(row[key] != leader[key] for key in ("pid", "ppid", "pgid", "sid", "starttime")):
        raise RecoveryError(f"{leader['role']} proc identity changed")
    if row["state"] in {"Z", "X", "x"}:
        raise RecoveryError(f"{leader['role']} leader is not live")
    if row["argv"] != leader["argv"] or not hmac.compare_digest(
        row["raw_argv_sha256"], leader["argv_sha256"].lower()
    ):
        raise RecoveryError(f"{leader['role']} argv changed")
    identities = auditor.process_identities(row["argv"])
    role = leader["role"]
    if role == "nav2_bringup":
        required = {
            "ros2-launch-package:nav2_bringup",
            "ros2-launch-file:bringup_launch.py",
        }
        if not required.issubset(identities):
            raise RecoveryError("Nav2 launch identity mismatch")
        if f"namespace:={LANE_NAMESPACE}" not in row["argv"]:
            raise RecoveryError("Nav2 launch namespace mismatch")
        if not _old_root_associated(row["argv"], old_root):
            raise RecoveryError("Nav2 launch is not associated with the exact old root")
    else:
        required = {
            "ros2-run-package:nav2_lifecycle_manager",
            "ros2-run-node:lifecycle_manager",
        }
        if not required.issubset(identities):
            raise RecoveryError(f"{role} ROS identity mismatch")
        if f"__ns:={LANE_NAMESPACE}" not in row["argv"]:
            raise RecoveryError(f"{role} namespace mismatch")
        if f"__node:={LIFECYCLE_NODE_NAMES[role]}" not in row["argv"]:
            raise RecoveryError(f"{role} node identity mismatch")


def _verify_group(
    rows: Sequence[dict[str, Any]], leader: dict[str, Any], expected_uid: int
) -> list[dict[str, Any]]:
    group = [row for row in rows if row["pgid"] == leader["pgid"]]
    if not group:
        raise RecoveryError(f"empty target PGID for {leader['role']}")
    for member in group:
        if member["sid"] != leader["sid"] or member["uid"] != expected_uid:
            raise RecoveryError(f"mixed session/owner in PGID for {leader['role']}")
        if member["starttime"] < leader["starttime"]:
            raise RecoveryError(f"pre-existing member in PGID for {leader['role']}")
    return group


def _scan_ports(ports: Iterable[int]) -> tuple[bool, list[str], str | None]:
    try:
        value = subprocess.run(
            ["ss", "-H", "-lntup"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, [], type(exc).__name__
    selected = [
        line
        for line in value.stdout.splitlines()
        if any(f":{port} " in line for port in ports)
    ]
    return value.returncode == 0 and not selected, selected, (
        None if value.returncode == 0 else f"ss_rc_{value.returncode}"
    )


def recover_stale_nav2(
    evidence: dict[str, Any],
    old_root: str,
    auditor: types.ModuleType,
    *,
    snapshot_provider: Callable[[], tuple[list[dict[str, Any]], list[str]]] = proc_snapshot,
    port_scanner: Callable[[Iterable[int]], tuple[bool, list[str], str | None]] = _scan_ports,
    signaler: Callable[[int, int], None] | None = None,
    expected_uid: int | None = None,
    term_timeout: float = 20.0,
    kill_timeout: float = 10.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if not (0 <= term_timeout <= 60 and 0 <= kill_timeout <= 30):
        raise RecoveryError("recovery timeout is outside the bounded contract")
    expected_uid = os.geteuid() if expected_uid is None else expected_uid
    if signaler is None:
        signaler = getattr(os, "killpg", None)
        if signaler is None:
            raise RecoveryError("process-group signalling is unavailable")

    def complete_snapshot(label: str) -> list[dict[str, Any]]:
        # /proc is inherently racy while unrelated short-lived processes come
        # and go. Retry a bounded number of complete scans; a persistent read
        # error still fails closed before signalling or certifying cleanup.
        observations: list[list[str]] = []
        for attempt in range(5):
            rows, errors = snapshot_provider()
            if not errors:
                return rows
            observations.append(list(errors))
            if attempt < 4:
                sleeper(0.05)
        raise RecoveryError(
            f"{label} /proc audit incomplete after 5 attempts: {observations[-1]}"
        )

    leaders = {leader["role"]: leader for leader in evidence["leaders"]}
    initial_rows = complete_snapshot("initial")
    by_pid = {row["pid"]: row for row in initial_rows}
    initial_groups: dict[int, list[dict[str, Any]]] = {}
    for role in sorted(LEADER_ROLES):
        leader = leaders[role]
        row = by_pid.get(leader["pid"])
        if row is None:
            raise RecoveryError(f"evidence leader absent: {role}")
        if row["uid"] != expected_uid:
            raise RecoveryError(f"evidence leader owner mismatch: {role}")
        _exact_ros_identity(row, leader, auditor, old_root)
        initial_groups[leader["pgid"]] = _verify_group(
            initial_rows, leader, expected_uid
        )
    if len(initial_groups) != 3:
        raise RecoveryError("target PGIDs are not distinct")

    signals: list[dict[str, Any]] = []
    active_groups = set(initial_groups)

    def send(requested: int, groups: set[int]) -> None:
        current_rows = complete_snapshot("immediate pre-signal")
        current_by_pid = {row["pid"]: row for row in current_rows}
        for group in sorted(groups):
            leader = next(item for item in leaders.values() if item["pgid"] == group)
            members = [row for row in current_rows if row["pgid"] == group]
            if not members:
                active_groups.discard(group)
                signals.append(
                    {
                        "pgid": group,
                        "signal": "SIGTERM" if requested == SIG_TERM else "SIGKILL",
                        "already_absent": True,
                    }
                )
                continue
            current_leader = current_by_pid.get(leader["pid"])
            if current_leader is not None:
                _exact_ros_identity(current_leader, leader, auditor, old_root)
            _verify_group(current_rows, leader, expected_uid)
            try:
                signaler(group, requested)
            except ProcessLookupError:
                active_groups.discard(group)
                signals.append(
                    {
                        "pgid": group,
                        "signal": "SIGTERM" if requested == SIG_TERM else "SIGKILL",
                        "already_absent": True,
                    }
                )
            else:
                signals.append(
                    {
                        "pgid": group,
                        "signal": "SIGTERM" if requested == SIG_TERM else "SIGKILL",
                    }
                )

    def wait_for_groups(timeout: float) -> set[int]:
        deadline = monotonic() + timeout
        while True:
            rows = complete_snapshot("bounded wait")
            present = {row["pgid"] for row in rows}
            active_groups.intersection_update(present)
            if not active_groups or monotonic() >= deadline:
                return set(active_groups)
            sleeper(min(0.2, max(0.0, deadline - monotonic())))

    term_started = monotonic()
    send(SIG_TERM, set(active_groups))
    residual_after_term = wait_for_groups(term_timeout)
    term_elapsed = monotonic() - term_started
    kill_started = monotonic()
    if residual_after_term:
        send(SIG_KILL, residual_after_term)
        wait_for_groups(kill_timeout)
    kill_elapsed = monotonic() - kill_started

    final_rows = complete_snapshot("final")
    final_errors: list[str] = []
    original_pids = {leader["pid"] for leader in leaders.values()}
    original_pgids = {leader["pgid"] for leader in leaders.values()}
    old_root_rows = [
        row for row in final_rows if _old_root_associated(row["argv"], old_root)
    ]
    compute_matches: list[dict[str, Any]] = []
    for row in final_rows:
        identities = auditor.process_identities(row["argv"])
        reason = auditor.forbidden_reason(identities)
        if reason is None:
            reason = next(
                (identity for identity in sorted(identities) if identity in EXTRA_COMPUTE_IDENTITIES),
                None,
            )
        if reason is not None:
            compute_matches.append(
                {
                    "pid": row["pid"],
                    "pgid": row["pgid"],
                    "sid": row["sid"],
                    "reason": reason,
                    "identities": sorted(identities),
                }
            )
    ports_clean, port_residual, port_error = port_scanner(T5_PORTS)
    checks = {
        "evidence_contract_exact": True,
        "leaders_exactly_verified": True,
        "target_pgids_exactly_three": len(initial_groups) == 3,
        "complete_session_groups_targeted": all(
            initial_groups[group] for group in initial_groups
        ),
        "term_wait_bounded": term_elapsed <= term_timeout + 1.0,
        "kill_wait_bounded": kill_elapsed <= kill_timeout + 1.0,
        "old_deployment_argv_zero": not old_root_rows,
        "structured_compute_audit_readable": not final_errors,
        "structured_compute_audit_zero": not compute_matches,
        "t5_port_audit_readable": port_error is None,
        "t5_ports_zero": ports_clean,
        "original_pids_zero": not any(row["pid"] in original_pids for row in final_rows),
        "original_pgids_zero": not any(row["pgid"] in original_pgids for row in final_rows),
    }
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "action": "RECOVER_STALE_DGX_A_NAV2",
        "role": ROLE,
        "deployment_root": old_root,
        "lane_namespace": LANE_NAMESPACE,
        "workload_started": False,
        "signals": signals,
        "initial_leaders": [
            {
                key: leader[key]
                for key in ("role", "pid", "ppid", "pgid", "sid", "starttime", "state")
            }
            for leader in evidence["leaders"]
        ],
        "initial_group_members": {
            str(group): [
                {
                    "pid": row["pid"],
                    "pgid": row["pgid"],
                    "sid": row["sid"],
                    "state": row["state"],
                    "identities": sorted(auditor.process_identities(row["argv"])),
                }
                for row in members
            ]
            for group, members in sorted(initial_groups.items())
        },
        "residual_after_term_pgids": sorted(residual_after_term),
        "old_root_residual": [row["pid"] for row in old_root_rows],
        "structured_compute_matches": compute_matches,
        "structured_compute_errors": final_errors,
        "port_residual": port_residual,
        "port_error": port_error,
        "checks": checks,
        "recorded_unix": time.time(),
    }
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage-evidence")
    stage.add_argument("source", type=Path)
    stage.add_argument("destination", type=Path)
    stage.add_argument("expected_sha256")
    stage.add_argument("old_root")
    verify = subparsers.add_parser("verify-evidence")
    verify.add_argument("path", type=Path)
    verify.add_argument("expected_sha256")
    verify.add_argument("old_root")
    recover = subparsers.add_parser("recover")
    recover.add_argument("old_root_b64")
    recover.add_argument("evidence_b64")
    recover.add_argument("evidence_sha256")
    recover.add_argument("auditor_b64")
    recover.add_argument("term_timeout", type=float)
    recover.add_argument("kill_timeout", type=float)
    args = parser.parse_args(argv)
    try:
        if args.command == "stage-evidence":
            payload = stage_evidence(
                args.source, args.destination, args.expected_sha256, args.old_root
            )
        elif args.command == "verify-evidence":
            payload = verify_archived_evidence(
                args.path, args.expected_sha256, args.old_root
            )
        else:
            old_root = validate_old_root(
                _decode_base64(args.old_root_b64, "old root").decode("utf-8")
            )
            evidence_raw = _decode_base64(args.evidence_b64, "evidence")
            evidence = validate_evidence(
                evidence_raw, args.evidence_sha256, old_root
            )
            auditor = _load_auditor(_decode_base64(args.auditor_b64, "auditor"))
            payload = recover_stale_nav2(
                evidence,
                old_root,
                auditor,
                term_timeout=args.term_timeout,
                kill_timeout=args.kill_timeout,
            )
        _print(payload)
        return 0 if payload.get("status") == "PASS" else EXIT_FAIL_CLOSED
    except (RecoveryError, OSError, UnicodeError, ValueError, TypeError) as exc:
        payload = {
            "schema_version": 1,
            "status": "FAIL",
            "action": args.command.upper().replace("-", "_"),
            "role": ROLE,
            "workload_started": False,
            "checks": {"recovery_contract_complete": False},
            "error": str(exc),
            "recorded_unix": time.time(),
        }
        _print(payload)
        return EXIT_FAIL_CLOSED


if __name__ == "__main__":
    raise SystemExit(main())
