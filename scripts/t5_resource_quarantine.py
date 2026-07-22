#!/usr/bin/env python3
"""Fail-closed T5 host quarantine marker and recovery operations.

This file is intentionally self-contained.  Coordinators base64-transfer its
exact bytes to a resource host, so recovery does not trust a possibly partial
remote deployment.  A marker is removed only when its run ownership still
matches and the relevant cleanup/audit contract passes.
"""

from __future__ import annotations

import argparse
import base64
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
from typing import Any, Callable, Iterable, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - production recovery is Linux-only.
    fcntl = None  # type: ignore[assignment]


EXIT_FAIL_CLOSED = 75
PRODUCTION_MARKERS = {
    "dgx_a": Path("/tmp/internnav_dgx.quarantine"),
    "dgx_b": Path("/tmp/internnav_dgx.quarantine"),
    "x86": Path("/tmp/internnav_isaac.quarantine"),
    "x86_gpu0": Path("/tmp/internnav_isaac_gpu0.quarantine"),
    "x86_gpu1": Path("/tmp/internnav_isaac_gpu1.quarantine"),
}
ALLOWED_REASONS = {"d0_prepare_in_progress", "t5_online_stage_in_progress"}
ALLOWED_ROOT_PREFIXES = {
    "dgx_a": "/home/railgun/internnav-t1-t2/.t5-deployments/",
    "dgx_b": "/home/rail/internnav-t1-t2/.t5-deployments/",
    "x86": "/home/song/internnav-t1-t2/.t5-deployments/",
    "x86_gpu0": "/home/song/internnav-t1-t2/.t5-deployments/",
    "x86_gpu1": "/home/song/internnav-t1-t2/.t5-deployments/",
}
T5_PORTS = (25137, 25138, 25139, 25140, 25141, 25239, 25240, 25241)
X86_CONTAINERS = ("internnav_t5_isaac_a", "internnav_t5_isaac_b")
X86_ROLE_CONTAINERS = {
    "x86": X86_CONTAINERS,
    "x86_gpu0": ("internnav_t5_isaac_a",),
    "x86_gpu1": ("internnav_t5_isaac_b",),
}
X86_ROLE_PORTS = {
    "x86": T5_PORTS,
    "x86_gpu0": (25137, 25139, 25140, 25141),
    "x86_gpu1": (25138, 25239, 25240, 25241),
}
X86_HEALTH_SOCKETS = (
    Path("/tmp/internnav_t5_a_ipc/isaac_health.sock"),
    Path("/tmp/internnav_t5_b_ipc/isaac_health.sock"),
)
X86_RUNTIME_LOCKS = (
    Path("/tmp/internnav_t5_isaac_a_runtime.lock"),
    Path("/tmp/internnav_t5_isaac_b_runtime.lock"),
    Path("/tmp/internnav_t5_isaac_shared_assets.lock"),
)
X86_ROLE_HEALTH_SOCKETS = {
    "x86": X86_HEALTH_SOCKETS,
    "x86_gpu0": (X86_HEALTH_SOCKETS[0],),
    "x86_gpu1": (X86_HEALTH_SOCKETS[1],),
}
X86_ROLE_RUNTIME_LOCKS = {
    "x86": X86_RUNTIME_LOCKS,
    "x86_gpu0": (X86_RUNTIME_LOCKS[0],),
    "x86_gpu1": (X86_RUNTIME_LOCKS[1],),
}
RUN_TAG_RE = re.compile(r"^[A-Za-z0-9._:-]{1,240}$")
ARMED_AT_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
MARKER_KEYS = {
    "schema_version",
    "state",
    "reason",
    "role",
    "run_tag",
    "scope_roots",
    "armed_at",
}


class QuarantineError(RuntimeError):
    """A condition that must leave the host quarantined."""


def _validate_marker_path(marker: Path, role: str, *, production: bool) -> None:
    if role not in PRODUCTION_MARKERS:
        raise QuarantineError(f"unsupported quarantine role: {role}")
    if production and marker != PRODUCTION_MARKERS[role]:
        raise QuarantineError(f"unexpected production marker for {role}: {marker}")


def _validate_identity(reason: str, role: str, run_tag: str) -> None:
    if reason not in ALLOWED_REASONS:
        raise QuarantineError(f"unsupported quarantine reason: {reason}")
    if role not in PRODUCTION_MARKERS:
        raise QuarantineError(f"unsupported quarantine role: {role}")
    if RUN_TAG_RE.fullmatch(run_tag) is None:
        raise QuarantineError("invalid quarantine run_tag")


def _validate_roots(
    roots: Sequence[str], role: str, *, production: bool
) -> tuple[str, ...]:
    if not roots:
        raise QuarantineError("quarantine scope_roots cannot be empty")
    normalized: list[str] = []
    for root in roots:
        if not isinstance(root, str) or not root.startswith("/"):
            raise QuarantineError("quarantine scope root must be absolute")
        if any(character in root for character in ("\n", "\r", "\0", "|")):
            raise QuarantineError("quarantine scope root contains a delimiter")
        value = posixpath.normpath(root)
        if production and not value.startswith(ALLOWED_ROOT_PREFIXES[role]):
            raise QuarantineError(f"scope root is outside {role} deployment prefix")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise QuarantineError("duplicate quarantine scope root")
    return tuple(normalized)


def _parse_marker(
    marker: Path, *, require_private_permissions: bool = True
) -> tuple[dict[str, str], bytes, tuple[int, int]]:
    try:
        metadata = marker.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise QuarantineError("quarantine marker must be a regular non-symlink file")
        if require_private_permissions and os.name == "posix" and metadata.st_mode & 0o077:
            raise QuarantineError("quarantine marker permissions are not private")
        if require_private_permissions and os.name == "posix":
            if metadata.st_uid != os.geteuid() or metadata.st_nlink != 1:
                raise QuarantineError("quarantine marker owner/link count is unsafe")
        raw = marker.read_bytes()
    except OSError as exc:
        raise QuarantineError(f"cannot read quarantine marker: {exc}") from exc
    values: dict[str, str] = {}
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise QuarantineError("quarantine marker is not UTF-8") from exc
    for line in lines:
        if not line or "=" not in line:
            raise QuarantineError("malformed quarantine marker line")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise QuarantineError("duplicate/empty quarantine marker key")
        if key not in MARKER_KEYS:
            raise QuarantineError(f"unknown quarantine marker key: {key}")
        values[key] = value
    if set(values) != MARKER_KEYS:
        missing = sorted(MARKER_KEYS - set(values))
        raise QuarantineError(f"quarantine marker key set mismatch; missing={missing}")
    if values["schema_version"] != "1":
        raise QuarantineError("unsupported quarantine marker schema_version")
    armed_at = values["armed_at"]
    if ARMED_AT_RE.fullmatch(armed_at) is None:
        raise QuarantineError("invalid quarantine marker armed_at")
    try:
        parsed_armed_at = time.strptime(armed_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise QuarantineError("invalid quarantine marker armed_at") from exc
    if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed_armed_at) != armed_at:
        raise QuarantineError("invalid quarantine marker armed_at")
    return values, raw, (metadata.st_dev, metadata.st_ino)


def _owned_marker(
    marker: Path,
    *,
    reason: str,
    role: str,
    run_tag: str,
    roots: Sequence[str],
    production: bool,
) -> tuple[dict[str, str], bytes, tuple[int, int], tuple[str, ...]]:
    _validate_marker_path(marker, role, production=production)
    _validate_identity(reason, role, run_tag)
    expected_roots = _validate_roots(roots, role, production=production)
    values, raw, identity = _parse_marker(
        marker, require_private_permissions=production
    )
    marker_roots = tuple(filter(None, values.get("scope_roots", "").split("|")))
    checks = {
        "state": values.get("state") == "DIRTY",
        "reason": values.get("reason") == reason,
        "role": values.get("role") == role,
        "run_tag": values.get("run_tag") == run_tag,
        "scope_roots": marker_roots == expected_roots,
    }
    if not all(checks.values()):
        raise QuarantineError(f"quarantine ownership mismatch: {checks}")
    return values, raw, identity, expected_roots


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path.parent, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def arm_marker(
    marker: Path,
    *,
    reason: str,
    role: str,
    run_tag: str,
    roots: Sequence[str],
    production: bool = True,
) -> dict[str, Any]:
    _validate_marker_path(marker, role, production=production)
    _validate_identity(reason, role, run_tag)
    normalized = _validate_roots(roots, role, production=production)
    payload = {
        "schema_version": "1",
        "state": "DIRTY",
        "reason": reason,
        "role": role,
        "run_tag": run_tag,
        "scope_roots": "|".join(normalized),
        "armed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    data = "".join(f"{key}={value}\n" for key, value in payload.items()).encode()
    descriptor = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(marker, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_parent(marker)
    except OSError as exc:
        raise QuarantineError(f"cannot atomically arm quarantine: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return {
        "schema_version": 1,
        "status": "PASS",
        "action": "ARM",
        "marker": str(marker),
        "reason": reason,
        "role": role,
        "run_tag": run_tag,
        "scope_roots": list(normalized),
        "marker_present": marker.is_file(),
    }


def observe_owned_marker(
    marker: Path,
    *,
    reason: str,
    role: str,
    run_tag: str,
    roots: Sequence[str],
    production: bool = True,
) -> dict[str, Any]:
    """Prove exact ownership of an existing marker without mutating it."""
    import hashlib

    values, raw, identity, normalized = _owned_marker(
        marker,
        reason=reason,
        role=role,
        run_tag=run_tag,
        roots=roots,
        production=production,
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "action": "OBSERVE_OWNED",
        "marker": str(marker),
        "marker_present": os.path.lexists(marker),
        "reason": reason,
        "role": role,
        "run_tag": run_tag,
        "scope_roots": list(normalized),
        "marker_identity": list(identity),
        "marker_sha256": hashlib.sha256(raw).hexdigest(),
        "armed_at": values["armed_at"],
        "workload_started": False,
    }


def _passing_cleanup_receipt(raw: bytes) -> tuple[dict[str, Any], str]:
    import hashlib

    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QuarantineError("cleanup receipt is not valid JSON") from exc
    checks = receipt.get("checks") if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get("status") != "PASS"
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise QuarantineError("cleanup receipt does not prove all checks PASS")
    return receipt, hashlib.sha256(raw).hexdigest()


def clear_owned_marker(
    marker: Path,
    *,
    reason: str,
    role: str,
    run_tag: str,
    roots: Sequence[str],
    cleanup_receipt: bytes,
    production: bool = True,
) -> dict[str, Any]:
    _, original, identity, normalized = _owned_marker(
        marker,
        reason=reason,
        role=role,
        run_tag=run_tag,
        roots=roots,
        production=production,
    )
    _, receipt_sha256 = _passing_cleanup_receipt(cleanup_receipt)
    current = marker.lstat()
    if (current.st_dev, current.st_ino) != identity or marker.read_bytes() != original:
        raise QuarantineError("quarantine marker changed during owned clear")
    marker.unlink()
    _fsync_parent(marker)
    if marker.exists():
        raise QuarantineError("quarantine marker still exists after owned clear")
    return {
        "schema_version": 1,
        "status": "PASS",
        "action": "OWNED_CLEAR",
        "marker": str(marker),
        "role": role,
        "run_tag": run_tag,
        "scope_roots": list(normalized),
        "cleanup_receipt_sha256": receipt_sha256,
        "marker_absent": True,
    }


def _process_table() -> list[dict[str, Any]]:
    ancestors: set[int] = set()
    cursor = os.getpid()
    while cursor > 1 and cursor not in ancestors:
        ancestors.add(cursor)
        try:
            status_lines = Path(f"/proc/{cursor}/status").read_text().splitlines()
            cursor = int(
                next(line.split()[1] for line in status_lines if line.startswith("PPid:"))
            )
        except (OSError, StopIteration, ValueError):
            break
    table: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        pid = int(entry.name)
        try:
            stat_raw = (entry / "stat").read_text(encoding="utf-8")
            close_paren = stat_raw.rfind(")")
            if close_paren < 0:
                raise ValueError("missing comm terminator")
            stat_tail = stat_raw[close_paren + 2 :].split()
            # stat_tail starts at field 3 (state).  Keep all identity fields
            # from one procfs read so a mixed-PGID recovery can detect PID
            # reuse or an exec/session boundary before signalling the group.
            ppid = int(stat_tail[1])
            pgid = int(stat_tail[2])
            sid = int(stat_tail[3])
            starttime = int(stat_tail[19])
            identity_readable = True
        except (OSError, ValueError, IndexError):
            try:
                # Preserve an otherwise visible group member as unreadable.
                # Omitting it could make a mixed group look fully owned.
                pgid = os.getpgid(pid)
            except (OSError, ProcessLookupError):
                continue
            ppid = None
            sid = None
            starttime = None
            identity_readable = False
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
                .strip()
            )
            command_readable = True
        except OSError:
            # Keep an unreadable member in the table.  Omitting it could make
            # a mixed process group appear fully deployment-owned.
            command = ""
            command_readable = False
        table.append(
            {
                "pid": pid,
                "ppid": ppid,
                "pgid": pgid,
                "sid": sid,
                "starttime": starttime,
                "command": command,
                "command_readable": command_readable,
                "identity_readable": identity_readable,
            }
        )
    return table


def _command_has_run_root(command: str, root: str) -> bool:
    """Match an argv/path root, never an adjacent path-prefix lookalike."""

    # cmdline NUL separators are rendered as spaces by _process_table.  A root
    # may also appear as an assignment value inside an argv element.  The
    # leading boundary rejects a root embedded in another path; the trailing
    # boundary accepts a descendant path but rejects /root-copy, /root_copy,
    # /root.old, and alphanumeric suffixes.
    pattern = re.compile(
        rf"(?:^|[\s=:'\"(,\[{{]){re.escape(root)}(?=$|[/\s:'\"),;\]}}])"
    )
    return pattern.search(command) is not None


def _scoped(table: Sequence[dict[str, Any]], roots: Sequence[str]) -> list[dict[str, Any]]:
    return [
        row
        for row in table
        if any(
            _command_has_run_root(str(row.get("command", "")), root)
            for root in roots
        )
    ]


_PROCESS_IDENTITY_FIELDS = ("pid", "ppid", "pgid", "sid", "starttime")


def _process_identity(row: dict[str, Any]) -> tuple[int, int, int, int, int] | None:
    """Return the immutable recovery identity, or None for incomplete rows."""

    values: list[int] = []
    for field in _PROCESS_IDENTITY_FIELDS:
        value = row.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        values.append(value)
    pid, ppid, pgid, sid, starttime = values
    if pid <= 1 or ppid < 0 or pgid <= 1 or sid <= 0 or starttime <= 0:
        return None
    return pid, ppid, pgid, sid, starttime


def _is_descendant_of(
    pid: int, leader_pid: int, rows_by_pid: dict[int, dict[str, Any]]
) -> bool:
    """Prove the current procfs ancestry reaches leader_pid without a gap."""

    cursor = pid
    visited: set[int] = set()
    while cursor != leader_pid:
        if cursor in visited:
            return False
        visited.add(cursor)
        row = rows_by_pid.get(cursor)
        identity = _process_identity(row) if row is not None else None
        if identity is None:
            return False
        parent = identity[1]
        if parent <= 1:
            return False
        cursor = parent
    return True


def _authorize_mixed_descendant_group(
    group: int,
    roots: Sequence[str],
    initial_members: Sequence[dict[str, Any]],
    current_members: Sequence[dict[str, Any]],
    current_table: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Authorize only a run-root leader and its native same-session children.

    Native ROS/Nav2 executables commonly drop the deployment root from argv
    after the root-scoped ``ros2 launch`` leader starts them.  This narrow
    exception remains fail-closed: the leader identity must survive the fresh
    pre-signal read, every member must be inspectable and in that leader's
    session, and every member's live ancestry must reach the leader.
    """

    error_prefix = f"refusing to signal mixed-scope PGID {group}"
    initial_leaders = [row for row in initial_members if row.get("pid") == group]
    current_leaders = [row for row in current_members if row.get("pid") == group]
    if len(initial_leaders) != 1 or len(current_leaders) != 1:
        raise QuarantineError(f"{error_prefix}: missing unique group leader")
    initial_leader = initial_leaders[0]
    current_leader = current_leaders[0]
    if initial_leader not in _scoped((initial_leader,), roots):
        raise QuarantineError(f"{error_prefix}: group leader is outside run root")
    if current_leader not in _scoped((current_leader,), roots):
        raise QuarantineError(f"{error_prefix}: group leader left run root")
    initial_identity = _process_identity(initial_leader)
    current_identity = _process_identity(current_leader)
    if initial_identity is None or current_identity is None:
        raise QuarantineError(f"{error_prefix}: group leader identity is unreadable")
    # PPID is ancestry evidence for the live tree, not immutable process
    # identity.  An SSH/session leader can be reparented to init between the
    # discovery and the immediate pre-signal read without changing the owned
    # PID/PGID/SID/starttime or leaving the frozen run root.
    initial_stable_identity = (
        initial_identity[0],
        initial_identity[2],
        initial_identity[3],
        initial_identity[4],
    )
    current_stable_identity = (
        current_identity[0],
        current_identity[2],
        current_identity[3],
        current_identity[4],
    )
    if initial_stable_identity != current_stable_identity:
        raise QuarantineError(f"{error_prefix}: group leader identity changed")
    if current_identity[0] != group or current_identity[2] != group:
        raise QuarantineError(f"{error_prefix}: invalid group leader identity")

    leader_sid = current_identity[3]
    rows_by_pid: dict[int, dict[str, Any]] = {}
    for row in current_table:
        pid = row.get("pid")
        if isinstance(pid, int) and not isinstance(pid, bool):
            # Duplicate procfs identities cannot be safely interpreted.
            if pid in rows_by_pid:
                raise QuarantineError(f"{error_prefix}: duplicate process identity")
            rows_by_pid[pid] = row
    member_identities: list[dict[str, int]] = []
    for member in current_members:
        identity = _process_identity(member)
        if member.get("command_readable") is not True:
            raise QuarantineError(f"{error_prefix}: member command is unreadable")
        if identity is None or member.get("identity_readable", True) is not True:
            raise QuarantineError(f"{error_prefix}: member identity is unreadable")
        pid, _ppid, pgid, sid, _starttime = identity
        if pgid != group or sid != leader_sid:
            raise QuarantineError(f"{error_prefix}: member session/group mismatch")
        if not _is_descendant_of(pid, group, rows_by_pid):
            raise QuarantineError(f"{error_prefix}: member ancestry is unrelated")
        member_identities.append(
            {
                field: value
                for field, value in zip(_PROCESS_IDENTITY_FIELDS, identity)
            }
        )
    return {
        "leader_identity": {
            field: value
            for field, value in zip(_PROCESS_IDENTITY_FIELDS, current_identity)
        },
        "member_identities": sorted(member_identities, key=lambda row: row["pid"]),
    }


def _frozen_mixed_group_survivors(
    group: int,
    authorization: dict[str, Any],
    current_table: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Revalidate the exact initially-authorized group after TERM.

    The group leader may exit first and Linux may reparent its children, so
    PPID is retained in the audit snapshot but is not required to remain
    unchanged.  PID/PGID/SID/starttime are stable process identity.  A reused
    PID, a member moving groups/sessions, an unreadable survivor, or any new
    member in the PGID makes the fallback fail closed.
    """

    error_prefix = f"refusing frozen mixed-scope PGID {group} fallback"
    frozen_rows = authorization.get("member_identities")
    if not isinstance(frozen_rows, list) or not frozen_rows:
        raise QuarantineError(f"{error_prefix}: missing frozen members")
    frozen_by_pid: dict[int, dict[str, Any]] = {}
    for row in frozen_rows:
        if not isinstance(row, dict):
            raise QuarantineError(f"{error_prefix}: invalid frozen member")
        identity = _process_identity(row)
        if identity is None or identity[2] != group:
            raise QuarantineError(f"{error_prefix}: invalid frozen identity")
        if identity[0] in frozen_by_pid:
            raise QuarantineError(f"{error_prefix}: duplicate frozen PID")
        frozen_by_pid[identity[0]] = row

    current_by_pid: dict[int, dict[str, Any]] = {}
    for row in current_table:
        pid = row.get("pid")
        if isinstance(pid, int) and not isinstance(pid, bool):
            if pid in current_by_pid:
                raise QuarantineError(f"{error_prefix}: duplicate current PID")
            current_by_pid[pid] = row

    survivors: list[dict[str, Any]] = []
    for pid, frozen in frozen_by_pid.items():
        current = current_by_pid.get(pid)
        if current is None:
            continue
        identity = _process_identity(current)
        if current.get("command_readable") is not True:
            raise QuarantineError(f"{error_prefix}: survivor command is unreadable")
        if identity is None or current.get("identity_readable", True) is not True:
            raise QuarantineError(f"{error_prefix}: survivor identity is unreadable")
        stable_identity = (identity[0], identity[2], identity[3], identity[4])
        frozen_identity = (
            frozen["pid"],
            frozen["pgid"],
            frozen["sid"],
            frozen["starttime"],
        )
        if stable_identity != frozen_identity:
            raise QuarantineError(f"{error_prefix}: survivor identity changed")
        survivors.append(current)

    for member in current_table:
        pgid = member.get("pgid")
        if pgid == group and member.get("pid") not in frozen_by_pid:
            raise QuarantineError(f"{error_prefix}: new group member appeared")
    return survivors


def _claimed_pgids_from_ledgers(
    roots: Sequence[str],
) -> tuple[set[int], list[str]]:
    claims: set[int] = set()
    errors: list[str] = []
    paths: set[Path] = set()
    for root_value in roots:
        results = Path(root_value) / "results"
        if not results.is_dir():
            continue
        try:
            paths.update(results.rglob("*.supervisor.json"))
            paths.update(results.rglob("pid_ledger.jsonl"))
            paths.update(results.rglob("*ledger.txt"))
        except OSError as exc:
            errors.append(f"{results}:{type(exc).__name__}")
    for path in sorted(paths):
        try:
            if path.name == "pid_ledger.jsonl":
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                absent = {
                    (row.get("scope", "host"), row.get("component"), row.get("pid"))
                    for row in rows
                    if row.get("event") == "verified_absent"
                }
                for row in rows:
                    key = (
                        row.get("scope", "host"),
                        row.get("component"),
                        row.get("pid"),
                    )
                    value = row.get("pgid")
                    if (
                        row.get("event") == "started"
                        and key not in absent
                        and row.get("scope", "host") == "host"
                        and isinstance(value, int)
                        and value > 1
                    ):
                        claims.add(value)
            elif path.suffix == ".json":
                value = json.loads(path.read_text(encoding="utf-8"))
                pgid = value.get("pgid") if isinstance(value, dict) else None
                if isinstance(pgid, int) and pgid > 1:
                    claims.add(pgid)
            else:
                values = {}
                for line in path.read_text(encoding="utf-8").splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        values[key] = value
                pgid = int(values.get("pgid", "0"))
                if pgid > 1:
                    claims.add(pgid)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{path}:{type(exc).__name__}")
    return claims, errors


def _signal_scoped_groups(
    roots: Sequence[str],
    *,
    process_provider: Callable[[], list[dict[str, Any]]],
    claimed_pgids: set[int],
    term_timeout: float,
    kill_timeout: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[int],
    dict[int, list[dict[str, Any]]],
]:
    signals: list[dict[str, Any]] = []
    mixed_authorizations: dict[int, dict[str, Any]] = {}

    def signal_groups(
        requested_signal: signal.Signals,
        discovery_table: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        table = (
            list(discovery_table)
            if discovery_table is not None
            else process_provider()
        )
        rows = _scoped(table, roots)
        groups = {int(row["pgid"]) for row in rows if int(row["pgid"]) > 1}
        for group in claimed_pgids:
            members = [row for row in table if int(row["pgid"]) == group]
            if members and any(row in rows for row in members):
                groups.add(group)
        for group, authorization in mixed_authorizations.items():
            if _frozen_mixed_group_survivors(group, authorization, table):
                groups.add(group)
        for group in sorted(groups):
            # Re-read the process table immediately before every signal.  It is
            # not sufficient that one member is deployment-scoped: killpg()
            # affects every member, so a mixed group must remain quarantined
            # for operator recovery rather than risk killing an unrelated job.
            current = process_provider()
            members = [row for row in current if int(row["pgid"]) == group]
            scoped_members = _scoped(members, roots)
            initial_members = [row for row in table if int(row["pgid"]) == group]
            authorization = mixed_authorizations.get(group)
            if authorization is not None:
                members = _frozen_mixed_group_survivors(
                    group, authorization, current
                )
                if not members:
                    signals.append(
                        {"pgid": group, "signal": requested_signal.name, "absent": True}
                    )
                    continue
                scoped_members = _scoped(members, roots)
            if not members:
                signals.append(
                    {"pgid": group, "signal": requested_signal.name, "absent": True}
                )
                continue
            if not scoped_members and authorization is None:
                raise QuarantineError(f"refusing to signal unassociated PGID {group}")
            if authorization is None and len(scoped_members) != len(members):
                authorization = _authorize_mixed_descendant_group(
                    group,
                    roots,
                    initial_members,
                    members,
                    current,
                )
                mixed_authorizations[group] = authorization
            try:
                os.killpg(group, requested_signal)
                record: dict[str, Any] = {
                    "pgid": group,
                    "signal": requested_signal.name,
                }
                if authorization is not None:
                    record["mixed_scope_descendants_authorized"] = True
                    record["leader_identity"] = authorization["leader_identity"]
                    record["frozen_member_identities"] = authorization[
                        "member_identities"
                    ]
                    if not scoped_members:
                        record["leader_absent_frozen_survivor_fallback"] = True
                signals.append(record)
            except ProcessLookupError:
                signals.append({"pgid": group, "signal": requested_signal.name, "absent": True})

    def currently_absent(table: Sequence[dict[str, Any]]) -> bool:
        for group, authorization in mixed_authorizations.items():
            if _frozen_mixed_group_survivors(group, authorization, table):
                return False
        scoped_rows = _scoped(table, roots)
        associated_claims = {
            group
            for group in claimed_pgids
            if any(int(row["pgid"]) == group and row in scoped_rows for row in table)
        }
        return not scoped_rows and not any(
            int(row["pgid"]) in associated_claims for row in table
        )

    def wait_absent(timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            table = process_provider()
            if currently_absent(table):
                return True
            time.sleep(0.2)
        return currently_absent(process_provider())

    initial_table = process_provider()
    if _scoped(initial_table, roots):
        # Preserve the discovery snapshot through the immediate fresh read.
        # Otherwise a leader exiting between two independent discovery calls
        # could erase the only run-root association while native children live.
        signal_groups(signal.SIGTERM, initial_table)
        if not wait_absent(term_timeout):
            signal_groups(signal.SIGKILL)
            wait_absent(kill_timeout)
    final_table = process_provider()
    final_scoped = _scoped(final_table, roots)
    # Frozen native descendants remain deployment-scoped even after their
    # leader's argv/root evidence disappears.  This also prevents an
    # unclaimed PGID that survives SIGKILL (for example, an unreaped process)
    # from being omitted from the final marker-clear decision.
    for group, authorization in mixed_authorizations.items():
        for survivor in _frozen_mixed_group_survivors(
            group, authorization, final_table
        ):
            if survivor not in final_scoped:
                final_scoped.append(survivor)
    unassociated_claims: list[int] = []
    residual_claimed: dict[int, list[dict[str, Any]]] = {}
    for group in sorted(claimed_pgids):
        members = [row for row in final_table if int(row["pgid"]) == group]
        if not members:
            continue
        residual_claimed[group] = members
        if not any(row in final_scoped for row in members):
            unassociated_claims.append(group)
    return signals, final_scoped, unassociated_claims, residual_claimed


def _scan_t5_ports(ports: Iterable[int]) -> tuple[bool, list[str]]:
    try:
        completed = subprocess.run(
            ["ss", "-H", "-lntup"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, ["ss audit unavailable"]
    selected = [
        line
        for line in completed.stdout.splitlines()
        if any(f":{port} " in line for port in ports)
    ]
    return completed.returncode == 0 and not selected, selected


def _recover_x86_containers(
    roots: Sequence[str], names: Sequence[str] = X86_CONTAINERS
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        inspect = subprocess.run(
            ["docker", "inspect", name], text=True, capture_output=True, check=False
        )
        if inspect.returncode != 0:
            detail = f"{inspect.stdout}\n{inspect.stderr}".lower()
            if "no such object" in detail or "no such container" in detail:
                rows.append({"name": name, "present": False, "stopped": True})
                continue
            raise QuarantineError(f"cannot prove container absence for {name}")
        try:
            value = json.loads(inspect.stdout)[0]
        except (json.JSONDecodeError, IndexError, TypeError) as exc:
            raise QuarantineError(f"cannot decode container inspect for {name}") from exc
        labels = value.get("Config", {}).get("Labels") or {}
        deployment_root = labels.get("internnav.t5.deployment_root")
        running = value.get("State", {}).get("Running") is True
        owned = deployment_root in roots
        if not owned:
            rows.append(
                {
                    "name": name,
                    "present": True,
                    "running": running,
                    "pid": value.get("State", {}).get("Pid"),
                    "owned": False,
                    "deployment_root": deployment_root,
                    "stopped": False,
                }
            )
            continue
        if running:
            # ``recover`` has a one-line JSON stdout protocol.  Docker prints
            # the stopped container name on stdout, so keep that diagnostic
            # out of the machine-readable receipt.
            subprocess.run(
                ["docker", "stop", "-t", "15", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            current = json.loads(
                subprocess.run(
                    ["docker", "inspect", name],
                    text=True,
                    capture_output=True,
                    check=True,
                ).stdout
            )[0]
            if current.get("State", {}).get("Running") is True:
                subprocess.run(["docker", "kill", name], check=False)
        final = json.loads(
            subprocess.run(
                ["docker", "inspect", name],
                text=True,
                capture_output=True,
                check=True,
            ).stdout
        )[0]
        stopped = (
            final.get("State", {}).get("Running") is False
            and int(final.get("State", {}).get("Pid", -1)) == 0
        )
        rows.append(
            {
                "name": name,
                "present": True,
                "running": final.get("State", {}).get("Running"),
                "pid": final.get("State", {}).get("Pid"),
                "owned": owned,
                "deployment_root": deployment_root,
                "stopped": stopped,
            }
        )
    return all(
        row.get("present") is False
        or (row.get("owned") is True and row.get("stopped") is True)
        for row in rows
    ), rows


def _clean_health_sockets(paths: Sequence[Path]) -> tuple[bool, list[str]]:
    residual: list[str] = []
    for path in paths:
        try:
            if os.path.lexists(os.fspath(path)):
                mode = path.lstat().st_mode
                if not stat.S_ISSOCK(mode):
                    residual.append(f"{path}:not-a-socket")
                    continue
                path.unlink()
            if os.path.lexists(os.fspath(path)):
                residual.append(str(path))
        except OSError as exc:
            residual.append(f"{path}:{type(exc).__name__}")
    return not residual, residual


def _runtime_locks_free(paths: Sequence[Path]) -> tuple[bool, list[str]]:
    if fcntl is None:
        return False, ["fcntl unavailable"]
    busy: list[str] = []
    for path in paths:
        if not os.path.lexists(os.fspath(path)):
            continue
        try:
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                busy.append(f"{path}:unsafe-file-type")
                continue
            with path.open("a+") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except (OSError, BlockingIOError):
            busy.append(str(path))
    return not busy, busy


def _derive_prepare_roots(role: str, run_tag: str) -> tuple[str, ...]:
    prefix = ALLOWED_ROOT_PREFIXES[role]
    if role == "dgx_a":
        return (f"{prefix}{run_tag}-lane-a",)
    if role == "dgx_b":
        return (f"{prefix}{run_tag}-lane-b",)
    if role == "x86":
        return (
        f"{prefix}{run_tag}-isaac-prepare",
        f"{prefix}{run_tag}-isaac-a",
        f"{prefix}{run_tag}-isaac-b",
        )
    raise QuarantineError("legacy prepare marker cannot use a per-GPU role")


def recover_marker(
    marker: Path,
    *,
    role: str,
    production: bool = True,
    process_provider: Callable[[], list[dict[str, Any]]] = _process_table,
    ledger_provider: Callable[
        [Sequence[str]], tuple[set[int], list[str]]
    ] = _claimed_pgids_from_ledgers,
    port_scanner: Callable[[Iterable[int]], tuple[bool, list[str]]] = _scan_t5_ports,
    container_recoverer: Callable[[Sequence[str]], tuple[bool, list[dict[str, Any]]]] = _recover_x86_containers,
    socket_cleaner: Callable[[Sequence[Path]], tuple[bool, list[str]]] = _clean_health_sockets,
    lock_checker: Callable[[Sequence[Path]], tuple[bool, list[str]]] = _runtime_locks_free,
    term_timeout: float = 20,
    kill_timeout: float = 10,
) -> dict[str, Any]:
    _validate_marker_path(marker, role, production=production)
    # Path.exists() follows symlinks and reports a dangling symlink as absent.
    # A quarantine path of any filesystem type must instead be parsed and
    # rejected unless it is the exact private regular marker contract.
    if not os.path.lexists(os.fspath(marker)):
        return {
            "schema_version": 1,
            "status": "PASS",
            "action": "RECOVER",
            "role": role,
            "marker": str(marker),
            "already_clear": True,
            "workload_started": False,
        }
    values, original, identity = _parse_marker(
        marker, require_private_permissions=production
    )
    reason = values.get("reason", "")
    run_tag = values.get("run_tag", "")
    marker_role = values.get("role", "")
    _validate_identity(reason, marker_role, run_tag)
    if values.get("state") != "DIRTY" or marker_role != role:
        raise QuarantineError("recovery marker ownership/state mismatch")
    encoded_roots = tuple(filter(None, values.get("scope_roots", "").split("|")))
    if not encoded_roots and reason == "d0_prepare_in_progress":
        encoded_roots = _derive_prepare_roots(role, run_tag)
    roots = _validate_roots(encoded_roots, role, production=production)

    claimed_pgids, ledger_errors = ledger_provider(roots)
    (
        signals,
        residual_processes,
        unassociated_claimed_pgids,
        residual_claimed_pgids,
    ) = _signal_scoped_groups(
        roots,
        process_provider=process_provider,
        claimed_pgids=claimed_pgids,
        term_timeout=term_timeout,
        kill_timeout=kill_timeout,
    )
    if role in X86_ROLE_CONTAINERS:
        if container_recoverer is _recover_x86_containers:
            containers_clean, containers = _recover_x86_containers(
                roots, X86_ROLE_CONTAINERS[role]
            )
        else:
            containers_clean, containers = container_recoverer(roots)
        sockets_clean, socket_residual = socket_cleaner(
            X86_ROLE_HEALTH_SOCKETS[role]
        )
        locks_clean, busy_locks = lock_checker(X86_ROLE_RUNTIME_LOCKS[role])
    else:
        containers_clean, containers = True, []
        sockets_clean, socket_residual = True, []
        locks_clean, busy_locks = True, []
    ports_clean, port_residual = port_scanner(X86_ROLE_PORTS.get(role, T5_PORTS))
    checks = {
        "marker_owned_and_dirty": True,
        "scoped_processes_absent": not residual_processes,
        "runtime_ledgers_readable": not ledger_errors,
        "claimed_pgids_absent": not residual_claimed_pgids,
        "reused_or_unassociated_claimed_pgids_absent": not unassociated_claimed_pgids,
        "owned_containers_absent_or_stopped": containers_clean,
        "health_sockets_absent": sockets_clean,
        "runtime_locks_free": locks_clean,
        "t5_ports_absent": ports_clean,
        "marker_unchanged_during_recovery": (
            (marker.lstat().st_dev, marker.lstat().st_ino) == identity
            and marker.read_bytes() == original
        ),
    }
    if all(checks.values()):
        marker.unlink()
        _fsync_parent(marker)
    checks["marker_removed_only_after_full_audit"] = (
        not marker.exists() if all(checks.values()) else marker.exists()
    )
    status_value = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "schema_version": 1,
        "status": status_value,
        "action": "RECOVER",
        "role": role,
        "marker": str(marker),
        "reason": reason,
        "run_tag": run_tag,
        "scope_roots": list(roots),
        "workload_started": False,
        "signals": signals,
        "claimed_pgids": sorted(claimed_pgids),
        "ledger_errors": ledger_errors,
        "unassociated_claimed_pgids": unassociated_claimed_pgids,
        "residual_claimed_pgids": residual_claimed_pgids,
        "residual_processes": residual_processes,
        "containers": containers,
        "socket_residual": socket_residual,
        "busy_runtime_locks": busy_locks,
        "port_residual": port_residual,
        "checks": checks,
        "recorded_unix": time.time(),
    }
    if status_value != "PASS":
        raise QuarantineError(json.dumps(payload, sort_keys=True))
    return payload


def _decode_roots(value: str) -> list[str]:
    try:
        decoded = base64.b64decode(value, validate=True).decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise QuarantineError("invalid base64 scope roots") from exc
    return decoded.split("|") if decoded else []


def _json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    arm = subparsers.add_parser("arm")
    observe = subparsers.add_parser("observe-owned")
    clear = subparsers.add_parser("owned-clear")
    recover = subparsers.add_parser("recover")
    for command in (arm, observe, clear):
        command.add_argument("marker", type=Path)
        command.add_argument("reason")
        command.add_argument("role")
        command.add_argument("run_tag")
        command.add_argument("roots_b64")
    clear.add_argument("cleanup_receipt_b64")
    recover.add_argument("marker", type=Path)
    recover.add_argument("role")
    args = parser.parse_args(argv)
    try:
        if args.command == "arm":
            payload = arm_marker(
                args.marker,
                reason=args.reason,
                role=args.role,
                run_tag=args.run_tag,
                roots=_decode_roots(args.roots_b64),
            )
        elif args.command == "observe-owned":
            payload = observe_owned_marker(
                args.marker,
                reason=args.reason,
                role=args.role,
                run_tag=args.run_tag,
                roots=_decode_roots(args.roots_b64),
            )
        elif args.command == "owned-clear":
            try:
                receipt = base64.b64decode(args.cleanup_receipt_b64, validate=True)
            except ValueError as exc:
                raise QuarantineError("invalid base64 cleanup receipt") from exc
            payload = clear_owned_marker(
                args.marker,
                reason=args.reason,
                role=args.role,
                run_tag=args.run_tag,
                roots=_decode_roots(args.roots_b64),
                cleanup_receipt=receipt,
            )
        else:
            payload = recover_marker(args.marker, role=args.role)
        _json_print(payload)
        return 0
    except (
        QuarantineError,
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ) as exc:
        _json_print(
            {
                "schema_version": 1,
                "status": "FAIL",
                "action": args.command.upper().replace("-", "_"),
                "error": str(exc),
                "marker_preserved": os.path.lexists(
                    os.fspath(getattr(args, "marker", Path("/nonexistent")))
                ),
            }
        )
        return EXIT_FAIL_CLOSED


if __name__ == "__main__":
    raise SystemExit(main())
