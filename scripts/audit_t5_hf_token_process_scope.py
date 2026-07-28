#!/usr/bin/env python3
"""Prove an exact HF credential is confined to the T5 model process group."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _identity(process_dir: Path) -> dict[str, object]:
    status = process_dir.joinpath("status").read_text(
        encoding="utf-8", errors="replace"
    )
    real_uid = None
    for line in status.splitlines():
        if line.startswith("Uid:"):
            real_uid = int(line.split()[1])
            break
    if real_uid is None:
        raise ValueError("process status has no real UID")
    stat = process_dir.joinpath("stat").read_text(
        encoding="utf-8", errors="replace"
    )
    close = stat.rfind(")")
    if close < 0:
        raise ValueError("malformed process stat")
    fields = stat[close + 1 :].split()
    if len(fields) < 3:
        raise ValueError("truncated process stat")
    return {
        "pid": int(process_dir.name),
        "real_uid": real_uid,
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
    }


def _descendant_state(
    pid: int, ancestor: int, identities: dict[int, dict]
) -> bool | None:
    """Return True/False only when ancestry is provable; None means unknown."""

    seen: set[int] = set()
    current = pid
    while current > 0:
        if current == ancestor:
            return True
        if current == 1:
            return False
        if current in seen:
            return None
        seen.add(current)
        row = identities.get(current)
        if row is None:
            return None
        current = int(row["ppid"])
    return False


def _is_descendant(pid: int, ancestor: int, identities: dict[int, dict]) -> bool:
    return _descendant_state(pid, ancestor, identities) is True


def audit(
    *,
    proc_root: Path,
    secret: bytes,
    model_pid: int,
    model_pgid: int,
    parent_pid: int,
    onboard_pid: int,
    evaluator_pid: int,
    effective_uid: int | None = None,
) -> dict[str, object]:
    if not secret:
        raise ValueError("secret input is empty")
    if b"\x00" in secret or b"\n" in secret or b"\r" in secret:
        raise ValueError("secret input contains a forbidden delimiter")

    identities: dict[int, dict] = {}
    vanished_identity_race_count = 0
    live_identity_unreadable_count = 0
    for process_dir in proc_root.iterdir():
        if not process_dir.is_dir() or not process_dir.name.isdigit():
            continue
        try:
            row = _identity(process_dir)
        except (FileNotFoundError, ProcessLookupError):
            if process_dir.exists():
                live_identity_unreadable_count += 1
            else:
                vanished_identity_race_count += 1
            continue
        except (PermissionError, OSError, ValueError):
            live_identity_unreadable_count += 1
            continue
        identities[int(row["pid"])] = row

    if effective_uid is None:
        if hasattr(os, "geteuid"):
            effective_uid = os.geteuid()
        elif parent_pid in identities:
            effective_uid = int(identities[parent_pid]["real_uid"])
        else:
            raise RuntimeError("cannot determine effective UID for process audit")
    matching: list[dict[str, object]] = []
    same_uid_scanned = 0
    same_uid_environ_unreadable = 0
    foreign_environ_unreadable = 0
    unreadable_allowed_model_scope = 0
    unreadable_noninheriting = 0
    unreadable_unapproved = 0
    vanished_environ_race_count = 0
    unreadable_processes: list[dict[str, object]] = []
    for pid, row in sorted(identities.items()):
        same_uid = int(row["real_uid"]) == effective_uid
        try:
            environ = proc_root.joinpath(str(pid), "environ").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            if proc_root.joinpath(str(pid)).exists():
                unreadable_unapproved += 1
                unreadable_processes.append(
                    {
                        "pid": pid,
                        "ppid": int(row["ppid"]),
                        "pgid": int(row["pgid"]),
                        "same_effective_uid": same_uid,
                        "parent_subtree": None,
                        "model_subtree": None,
                        "model_process_group": None,
                        "classification": "live_identity_changed_during_environ_scan",
                    }
                )
            else:
                vanished_environ_race_count += 1
            continue
        except (PermissionError, OSError):
            if same_uid:
                same_uid_environ_unreadable += 1
            else:
                foreign_environ_unreadable += 1
            model_state = _descendant_state(pid, model_pid, identities)
            model_subtree = model_state is True
            model_process_group = int(row["pgid"]) == model_pgid
            parent_state = _descendant_state(pid, parent_pid, identities)
            if model_subtree and model_process_group:
                classification = "allowed_model_scope"
                unreadable_allowed_model_scope += 1
            elif parent_state is False:
                # Environment inheritance is the audited leak path.  A
                # non-dumpable process outside the only process tree that
                # received the secret cannot inherit it from this run.
                classification = "outside_secret_inheritance_tree"
                unreadable_noninheriting += 1
            else:
                classification = "unapproved_secret_inheriting_scope"
                unreadable_unapproved += 1
            unreadable_processes.append(
                {
                    "pid": pid,
                    "ppid": int(row["ppid"]),
                    "pgid": int(row["pgid"]),
                    "same_effective_uid": same_uid,
                    "parent_subtree": parent_state,
                    "model_subtree": model_state,
                    "model_process_group": model_process_group,
                    "classification": classification,
                }
            )
            continue
        if same_uid:
            same_uid_scanned += 1
        if secret not in environ:
            continue
        descendant = _is_descendant(pid, model_pid, identities)
        same_pgid = int(row["pgid"]) == model_pgid
        matching.append(
            {
                "pid": pid,
                "ppid": int(row["ppid"]),
                "pgid": int(row["pgid"]),
                "model_subtree": descendant,
                "model_process_group": same_pgid,
                "allowed": descendant and same_pgid,
            }
        )

    matching_pids = {int(row["pid"]) for row in matching}
    disallowed = [row for row in matching if not bool(row["allowed"])]
    checks = {
        "model_root_present": model_pid in identities,
        "model_pgid_matches_live_root": model_pid in identities
        and int(identities[model_pid]["pgid"]) == model_pgid,
        "at_least_one_exact_secret_match": len(matching) >= 1,
        "all_matches_in_model_pgid_and_subtree": not disallowed,
        "parent_has_zero_matches": parent_pid not in matching_pids,
        "onboard_has_zero_matches": onboard_pid not in matching_pids,
        "evaluator_has_zero_matches": evaluator_pid not in matching_pids,
        "other_processes_have_zero_matches": not disallowed,
        "all_secret_inheriting_environs_readable_or_allowed_model_scope": (
            unreadable_unapproved == 0
        ),
        "all_live_process_identities_readable": live_identity_unreadable_count == 0,
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": (
            "all_readable_proc_environs_with_fail_closed_secret_inheritance_tree_coverage"
        ),
        "model_pid": model_pid,
        "model_pgid": model_pgid,
        "parent_pid": parent_pid,
        "onboard_pid": onboard_pid,
        "evaluator_pid": evaluator_pid,
        "exact_secret_match_count": len(matching),
        "allowed_model_match_count": len(matching) - len(disallowed),
        "disallowed_match_count": len(disallowed),
        "parent_match_count": int(parent_pid in matching_pids),
        "onboard_match_count": int(onboard_pid in matching_pids),
        "evaluator_match_count": int(evaluator_pid in matching_pids),
        "same_uid_environ_scanned_count": same_uid_scanned,
        "same_uid_environ_unreadable_count": same_uid_environ_unreadable,
        "foreign_environ_unreadable_count": foreign_environ_unreadable,
        "unreadable_allowed_model_scope_count": unreadable_allowed_model_scope,
        "unreadable_outside_secret_inheritance_tree_count": unreadable_noninheriting,
        "unreadable_unapproved_secret_inheriting_scope_count": unreadable_unapproved,
        "unreadable_processes": unreadable_processes,
        "identity_race_or_unreadable_count": (
            vanished_identity_race_count
            + live_identity_unreadable_count
            + vanished_environ_race_count
        ),
        "vanished_identity_race_count": vanished_identity_race_count,
        "vanished_environ_race_count": vanished_environ_race_count,
        "live_identity_unreadable_count": live_identity_unreadable_count,
        "matching_processes": matching,
        "checks": checks,
        "secret_value_recorded": False,
        "secret_digest_recorded": False,
        "recorded_unix": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--secret-fd", type=int, required=True)
    parser.add_argument("--model-pid", type=int, required=True)
    parser.add_argument("--model-pgid", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--onboard-pid", type=int, required=True)
    parser.add_argument("--evaluator-pid", type=int, required=True)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with os.fdopen(args.secret_fd, "rb", closefd=True) as stream:
        secret = stream.read().rstrip(b"\r\n")
    result = audit(
        proc_root=args.proc_root,
        secret=secret,
        model_pid=args.model_pid,
        model_pgid=args.model_pgid,
        parent_pid=args.parent_pid,
        onboard_pid=args.onboard_pid,
        evaluator_pid=args.evaluator_pid,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
