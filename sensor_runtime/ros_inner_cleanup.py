#!/usr/bin/env python3
"""Container-side fallback cleanup from the durable inner ledger."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .atomic import atomic_write_json
from .contract import PROFILES
from .processes import cleanup_persisted_ledger, terminate_bound_process_group


BASE_EXPECTED_ROLES = frozenset(
    {"go2_sensor_bridge", "sensor_ros_sidecar", "downstream_recorder"}
)


def expected_roles_for_profile(profile: str) -> set[str]:
    if profile not in PROFILES:
        raise RuntimeError("inner cleanup requires a frozen session profile")
    expected = set(BASE_EXPECTED_ROLES)
    if profile == "completion_sim_map":
        expected.add("t4_map_companion")
    return expected


def validate_supervisor_identity(identity: object) -> tuple[int, int, int]:
    if not isinstance(identity, dict):
        raise RuntimeError("inner supervisor identity is absent or malformed")
    if identity.get("isolated_process_group") is not True:
        raise RuntimeError("inner supervisor identity lacks isolated-process-group proof")
    try:
        pid = int(identity["pid"])
        pgid = int(identity["pgid"])
        start_ticks = int(identity["linux_start_ticks"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("inner supervisor identity is absent or malformed") from exc
    if min(pid, pgid, start_ticks) <= 0 or pid != pgid:
        raise RuntimeError("inner supervisor identity is not an isolated PID/PGID")
    return pid, pgid, start_ticks


def validate_required_clean_exits(
    payload: object, expected_roles: set[str]
) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise RuntimeError("normal child cleanup detail is absent or malformed")
    groups = payload.get("groups")
    if (
        payload.get("status") != "PASS"
        or payload.get("cleanup_confirmed") is not True
        or not isinstance(groups, list)
    ):
        raise RuntimeError("normal child cleanup did not complete successfully")
    selected = [item for item in groups if isinstance(item, Mapping) and item.get("role") in expected_roles]
    roles = [str(item["role"]) for item in selected]
    if len(selected) != len(expected_roles) or set(roles) != expected_roles or len(roles) != len(set(roles)):
        raise RuntimeError("normal child cleanup required-role exit set is incomplete")
    if any(item.get("clean_exit") is not True or int(item.get("exit_code", -1)) != 0 for item in selected):
        raise RuntimeError("normal child cleanup contains a nonzero required-role exit")
    return [
        {"role": str(item["role"]), "exit_code": 0, "clean_exit": True}
        for item in sorted(selected, key=lambda value: str(value["role"]))
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    expected_roles = expected_roles_for_profile(
        os.environ.get("INTERNNAV_SESSION_PROFILE", "")
    )
    errors: list[str] = []
    identity_values: tuple[int, int, int] | None = None
    try:
        identity = json.loads(
            (result_dir / "inner_supervisor_identity.json").read_text(encoding="utf-8")
        )
        identity_values = validate_supervisor_identity(identity)
    except BaseException as exc:
        errors.append(f"supervisor_identity: {type(exc).__name__}: {exc}")

    detail: dict[str, Any]
    try:
        detail = cleanup_persisted_ledger(
            result_dir / "inner_lifecycle",
            [args.socket],
            result_dir / "inner_lifecycle/cleanup_recovery.json",
            expected_roles=expected_roles,
        )
        if detail.get("residual_cleanup_confirmed") is not True:
            errors.append("child persisted-ledger residual cleanup was not confirmed")
    except BaseException as exc:
        errors.append(f"child_cleanup: {type(exc).__name__}: {exc}")
        detail = {"pid_count": -1, "pgid_count": -1, "socket_count": -1, "residual_cleanup_confirmed": False}

    required_role_exits: list[dict[str, Any]] = []
    try:
        normal_detail = json.loads(
            (result_dir / "inner_lifecycle/cleanup.json").read_text(encoding="utf-8")
        )
        required_role_exits = validate_required_clean_exits(normal_detail, expected_roles)
    except BaseException as exc:
        errors.append(f"child_clean_exit: {type(exc).__name__}: {exc}")

    supervisor: dict[str, Any]
    if identity_values is None:
        supervisor = {
            "identity_verified": False,
            "identity_error": "identity unavailable",
            "live_after_kill": [{"unknown": True}],
        }
    else:
        try:
            supervisor = terminate_bound_process_group(
                role="inner_supervisor",
                pid=identity_values[0],
                pgid=identity_values[1],
                linux_start_identity=identity_values[2],
            )
        except BaseException as exc:
            errors.append(f"supervisor_cleanup: {type(exc).__name__}: {exc}")
            supervisor = {
                "identity_verified": False,
                "identity_error": f"{type(exc).__name__}: {exc}",
                "live_after_kill": [{"unknown": True}],
            }
    supervisor_pid_count = len(supervisor["live_after_kill"])
    supervisor_pgid_count = int(bool(supervisor["live_after_kill"]))
    combined_ok = (
        detail.get("residual_cleanup_confirmed") is True
        and len(required_role_exits) == len(expected_roles)
        and supervisor["identity_verified"]
        and supervisor_pid_count == 0
        and supervisor_pgid_count == 0
        and not errors
    )
    atomic_write_json(
        result_dir / "inner_cleanup.json",
        {
            "schema_version": 2,
            "status": "PASS" if combined_ok else "FAIL",
            "cleanup_confirmed": combined_ok,
            "child_cleanup_confirmed": len(required_role_exits) == len(expected_roles),
            "bridge_sidecar_pid_count": int(detail.get("pid_count", -1)),
            "bridge_sidecar_pgid_count": int(detail.get("pgid_count", -1)),
            "supervisor_pid_count": supervisor_pid_count,
            "supervisor_pgid_count": supervisor_pgid_count,
            "pid_count": int(detail.get("pid_count", -1)) + supervisor_pid_count,
            "pgid_count": int(detail.get("pgid_count", -1)) + supervisor_pgid_count,
            "sensor_socket_count": int(detail.get("socket_count", -1)),
            "inner_ledger": "inner_lifecycle/process_ledger.json",
            "inner_cleanup_detail": "inner_lifecycle/cleanup_recovery.json",
            "recovery_used": True,
            "supervisor_cleanup": supervisor,
            "required_role_exits": required_role_exits,
            "errors": errors,
        },
    )
    return 0 if combined_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
