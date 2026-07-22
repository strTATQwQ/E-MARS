import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "scripts" / "t5_resource_quarantine.py"
COMMON_PATH = ROOT / "scripts" / "t5_quarantine_common.sh"
RECOVERY_PATH = ROOT / "scripts" / "recover_t5_resource_quarantine.sh"
LEASE_PATH = ROOT / "scripts" / "with_resource_lease.sh"

SPEC = importlib.util.spec_from_file_location("t5_resource_quarantine", GUARD_PATH)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def _pass_receipt() -> bytes:
    return json.dumps(
        {"schema_version": 1, "status": "PASS", "checks": {"zero": True}}
    ).encode()


def test_exclusive_private_arm_and_owned_receipt_clear(tmp_path: Path) -> None:
    marker = tmp_path / "quarantine"
    value = guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="d0_1:test",
        roots=["/tmp/t5-fixture-a"],
        production=False,
    )
    assert value["status"] == "PASS"
    if os.name == "posix":
        assert marker.stat().st_mode & 0o777 == 0o600
    original = marker.read_bytes()
    observed = guard.observe_owned_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="d0_1:test",
        roots=["/tmp/t5-fixture-a"],
        production=False,
    )
    assert observed["status"] == "PASS"
    assert observed["action"] == "OBSERVE_OWNED"
    assert observed["workload_started"] is False
    assert marker.read_bytes() == original
    with pytest.raises(guard.QuarantineError, match="atomically arm"):
        guard.arm_marker(
            marker,
            reason="t5_online_stage_in_progress",
            role="dgx_a",
            run_tag="other",
            roots=["/tmp/t5-fixture-a"],
            production=False,
        )
    assert marker.read_bytes() == original
    cleared = guard.clear_owned_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="d0_1:test",
        roots=["/tmp/t5-fixture-a"],
        cleanup_receipt=_pass_receipt(),
        production=False,
    )
    assert cleared["status"] == "PASS"
    assert not marker.exists()


@pytest.mark.parametrize(
    "receipt",
    [
        {"status": "FAIL", "checks": {"zero": True}},
        {"status": "PASS", "checks": {"zero": False}},
        {"status": "PASS", "checks": {}},
    ],
)
def test_failed_cleanup_receipt_never_clears_marker(
    tmp_path: Path, receipt: dict
) -> None:
    marker = tmp_path / "quarantine"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_b",
        run_tag="d0_2:test",
        roots=["/tmp/t5-fixture-b"],
        production=False,
    )
    with pytest.raises(guard.QuarantineError, match="does not prove"):
        guard.clear_owned_marker(
            marker,
            reason="t5_online_stage_in_progress",
            role="dgx_b",
            run_tag="d0_2:test",
            roots=["/tmp/t5-fixture-b"],
            cleanup_receipt=json.dumps(receipt).encode(),
            production=False,
        )
    assert marker.is_file()


def test_wrong_run_tag_and_symlink_are_never_unlinked(tmp_path: Path) -> None:
    marker = tmp_path / "quarantine"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="owned",
        roots=["/tmp/t5-fixture"],
        production=False,
    )
    with pytest.raises(guard.QuarantineError, match="ownership mismatch"):
        guard.clear_owned_marker(
            marker,
            reason="t5_online_stage_in_progress",
            role="dgx_a",
            run_tag="foreign",
            roots=["/tmp/t5-fixture"],
            cleanup_receipt=_pass_receipt(),
            production=False,
        )
    assert marker.is_file()
    marker.unlink()
    target = tmp_path / "target"
    target.write_text("state=DIRTY\n", encoding="utf-8")
    marker.symlink_to(target)
    with pytest.raises(guard.QuarantineError, match="non-symlink"):
        guard.recover_marker(marker, role="dgx_a", production=False)
    assert marker.is_symlink()
    assert target.read_text(encoding="utf-8") == "state=DIRTY\n"


def test_dangling_quarantine_symlink_is_not_reported_already_clear(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "quarantine"
    marker.symlink_to(tmp_path / "missing-target")
    with pytest.raises(guard.QuarantineError, match="non-symlink"):
        guard.recover_marker(marker, role="dgx_a", production=False)
    assert marker.is_symlink()


@pytest.mark.parametrize(
    ("rewrite", "error"),
    [
        (lambda raw: raw.replace("schema_version=1\n", ""), "key set mismatch"),
        (
            lambda raw: raw.replace("schema_version=1", "schema_version=2"),
            "schema_version",
        ),
        (lambda raw: raw.replace("armed_at=", "missing_armed_at="), "unknown"),
        (
            lambda raw: re.sub(r"armed_at=.*", "armed_at=2026-99-99T99:99:99Z", raw),
            "armed_at",
        ),
    ],
)
def test_marker_schema_is_exact_and_invalid_marker_is_preserved(
    tmp_path: Path, rewrite, error: str
) -> None:
    marker = tmp_path / "quarantine"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="schema-test",
        roots=["/tmp/t5-schema-test"],
        production=False,
    )
    marker.write_text(rewrite(marker.read_text(encoding="utf-8")), encoding="utf-8")
    with pytest.raises(guard.QuarantineError, match=error):
        guard.recover_marker(marker, role="dgx_a", production=False)
    assert marker.is_file()


def test_recovery_positive_audits_then_clears_without_workload(tmp_path: Path) -> None:
    marker = tmp_path / "quarantine"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="recover:test",
        roots=["/tmp/t5-recovery-scope"],
        production=False,
    )
    value = guard.recover_marker(
        marker,
        role="dgx_a",
        production=False,
        process_provider=lambda: [],
        port_scanner=lambda _ports: (True, []),
    )
    assert value["status"] == "PASS"
    assert value["workload_started"] is False
    assert all(value["checks"].values())
    assert not marker.exists()


def test_recovery_residual_port_preserves_marker(tmp_path: Path) -> None:
    marker = tmp_path / "quarantine"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="recover:blocked",
        roots=["/tmp/t5-recovery-scope"],
        production=False,
    )
    with pytest.raises(guard.QuarantineError, match="t5_ports_absent"):
        guard.recover_marker(
            marker,
            role="dgx_a",
            production=False,
            process_provider=lambda: [],
            port_scanner=lambda _ports: (False, ["LISTEN 0 1 *:25137"]),
        )
    assert marker.is_file()


def test_recovery_never_signals_or_clears_a_reused_unassociated_ledger_pgid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "quarantine"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="recover:reused-pgid",
        roots=["/tmp/t5-owned-scope"],
        production=False,
    )
    signalled: list[tuple[int, object]] = []
    monkeypatch.setattr(
        guard.os,
        "killpg",
        lambda pgid, requested: signalled.append((pgid, requested)),
        raising=False,
    )
    unrelated = [{"pid": 99991, "pgid": 424242, "command": "/usr/bin/unrelated"}]
    with pytest.raises(guard.QuarantineError, match="unassociated_claimed_pgids"):
        guard.recover_marker(
            marker,
            role="dgx_a",
            production=False,
            process_provider=lambda: unrelated,
            ledger_provider=lambda _roots: ({424242}, []),
            port_scanner=lambda _ports: (True, []),
        )
    assert signalled == []
    assert marker.is_file()


def test_recovery_never_signals_or_clears_a_mixed_scope_pgid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "quarantine"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="recover:mixed-pgid",
        roots=["/tmp/t5-owned-scope"],
        production=False,
    )
    signalled: list[tuple[int, object]] = []
    monkeypatch.setattr(
        guard.os,
        "killpg",
        lambda pgid, requested: signalled.append((pgid, requested)),
        raising=False,
    )
    mixed = [
        {
            "pid": 99991,
            "pgid": 424242,
            "command": "/tmp/t5-owned-scope/bin/runtime",
        },
        {"pid": 99992, "pgid": 424242, "command": "/usr/bin/unrelated"},
    ]
    with pytest.raises(guard.QuarantineError, match="mixed-scope PGID"):
        guard.recover_marker(
            marker,
            role="dgx_a",
            production=False,
            process_provider=lambda: mixed,
            ledger_provider=lambda _roots: ({424242}, []),
            port_scanner=lambda _ports: (True, []),
        )
    assert signalled == []
    assert marker.is_file()


def _process_row(
    pid: int,
    *,
    ppid: int,
    pgid: int,
    sid: int,
    starttime: int,
    command: str,
    command_readable: bool = True,
) -> dict:
    return {
        "pid": pid,
        "ppid": ppid,
        "pgid": pgid,
        "sid": sid,
        "starttime": starttime,
        "command": command,
        "command_readable": command_readable,
        "identity_readable": True,
    }


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/tmp/t5-owned-scope", True),
        ("/tmp/t5-owned-scope/child", True),
        ("params_file:=/tmp/t5-owned-scope/nav2.yaml", True),
        ("--arg=/tmp/t5-owned-scope/config.json", True),
        ("python /tmp/t5-owned-scope/launch.py", True),
        ("/tmp/t5-owned-scope-copy/launch.py", False),
        ("/tmp/t5-owned-scope_old/launch.py", False),
        ("/tmp/t5-owned-scope配置/launch.py", False),
        ("/tmp/t5-owned-scope$copy/launch.py", False),
        ("/tmp/t5-owned-scope+copy/launch.py", False),
        ("prefix$/tmp/t5-owned-scope/launch.py", False),
        ("/tmp/prefix/tmp/t5-owned-scope/launch.py", False),
    ],
)
def test_run_root_matching_requires_argv_or_path_boundary(
    command: str, expected: bool
) -> None:
    root = "/tmp/t5-owned-scope"
    assert guard._command_has_run_root(command, root) is expected


def test_recovery_signals_root_scoped_leader_and_native_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "quarantine"
    root = "/tmp/t5-owned-scope"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="recover:native-descendants",
        roots=[root],
        production=False,
    )
    group = 424242
    rows = [
        _process_row(
            group,
            ppid=1,
            pgid=group,
            sid=group,
            starttime=100,
            command=f"/usr/bin/python3 {root}/launch.py",
        ),
        _process_row(
            group + 1,
            ppid=group,
            pgid=group,
            sid=group,
            starttime=101,
            command="/opt/ros/humble/lib/controller_server",
        ),
        _process_row(
            group + 2,
            ppid=group + 1,
            pgid=group,
            sid=group,
            starttime=102,
            command="/opt/ros/humble/lib/planner_server",
        ),
    ]
    current = list(rows)
    signalled: list[tuple[int, object]] = []

    def fake_killpg(pgid: int, requested: object) -> None:
        nonlocal current
        signalled.append((pgid, requested))
        current = []

    monkeypatch.setattr(guard.os, "killpg", fake_killpg, raising=False)
    value = guard.recover_marker(
        marker,
        role="dgx_a",
        production=False,
        process_provider=lambda: list(current),
        ledger_provider=lambda _roots: ({group}, []),
        port_scanner=lambda _ports: (True, []),
        term_timeout=0.01,
        kill_timeout=0.01,
    )
    assert signalled == [(group, guard.signal.SIGTERM)]
    assert value["status"] == "PASS"
    assert value["signals"][0]["mixed_scope_descendants_authorized"] is True
    assert value["signals"][0]["leader_identity"] == {
        "pid": group,
        "ppid": 1,
        "pgid": group,
        "sid": group,
        "starttime": 100,
    }
    assert not marker.exists()


def test_recovery_kills_frozen_native_survivors_after_term_exits_leader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kill_signal = getattr(guard.signal, "SIGKILL", None)
    if kill_signal is None:  # Windows unit runner; production recovery is Linux-only.
        kill_signal = type("KillSignal", (), {"name": "SIGKILL"})()
        monkeypatch.setattr(guard.signal, "SIGKILL", kill_signal, raising=False)
    marker = tmp_path / "quarantine"
    root = "/tmp/t5-owned-scope"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="recover:term-leader-first",
        roots=[root],
        production=False,
    )
    group = 424242
    initial = [
        _process_row(
            group,
            ppid=1,
            pgid=group,
            sid=group,
            starttime=100,
            command=f"/usr/bin/python3 {root}/launch.py",
        ),
        _process_row(
            group + 1,
            ppid=group,
            pgid=group,
            sid=group,
            starttime=101,
            command="/opt/ros/humble/lib/controller_server",
        ),
        _process_row(
            group + 2,
            ppid=group + 1,
            pgid=group,
            sid=group,
            starttime=102,
            command="/opt/ros/humble/lib/planner_server",
        ),
    ]
    current = [dict(row) for row in initial]
    signalled: list[tuple[int, object]] = []

    def fake_killpg(pgid: int, requested: object) -> None:
        nonlocal current
        signalled.append((pgid, requested))
        if requested == guard.signal.SIGTERM:
            # The root-scoped leader and one child exit.  The remaining native
            # process is reparented but retains its stable process identity.
            survivor = dict(initial[2])
            survivor["ppid"] = 1
            current = [survivor]
        else:
            current = []

    monkeypatch.setattr(guard.os, "killpg", fake_killpg, raising=False)
    value = guard.recover_marker(
        marker,
        role="dgx_a",
        production=False,
        process_provider=lambda: [dict(row) for row in current],
        # This Nav2 PGID was found through its run-root leader and is
        # deliberately absent from the top-level ledger claims.
        ledger_provider=lambda _roots: (set(), []),
        port_scanner=lambda _ports: (True, []),
        term_timeout=0,
        kill_timeout=0,
    )
    assert signalled == [
        (group, guard.signal.SIGTERM),
        (group, kill_signal),
    ]
    assert value["status"] == "PASS"
    assert value["signals"][1]["leader_absent_frozen_survivor_fallback"] is True
    assert [
        row["pid"] for row in value["signals"][1]["frozen_member_identities"]
    ] == [group, group + 1, group + 2]
    assert not marker.exists()


@pytest.mark.parametrize("mutation", ("new_member", "pgid", "sid", "starttime"))
def test_recovery_refuses_frozen_fallback_on_new_or_changed_survivor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    marker = tmp_path / "quarantine"
    root = "/tmp/t5-owned-scope"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag=f"recover:fallback-{mutation}",
        roots=[root],
        production=False,
    )
    group = 424242
    initial = [
        _process_row(
            group,
            ppid=1,
            pgid=group,
            sid=group,
            starttime=100,
            command=f"/usr/bin/python3 {root}/launch.py",
        ),
        _process_row(
            group + 1,
            ppid=group,
            pgid=group,
            sid=group,
            starttime=101,
            command="/opt/ros/humble/lib/controller_server",
        ),
    ]
    current = [dict(row) for row in initial]
    signalled: list[tuple[int, object]] = []

    def fake_killpg(pgid: int, requested: object) -> None:
        nonlocal current
        signalled.append((pgid, requested))
        if requested != guard.signal.SIGTERM:
            pytest.fail("unsafe SIGKILL after frozen member-set drift")
        survivor = dict(initial[1])
        survivor["ppid"] = 1
        current = [survivor]
        if mutation == "new_member":
            current.append(
                _process_row(
                    group + 99,
                    ppid=1,
                    pgid=group,
                    sid=group,
                    starttime=999,
                    command="/opt/ros/humble/lib/unrelated",
                )
            )
        else:
            current[0][mutation] += 77

    monkeypatch.setattr(guard.os, "killpg", fake_killpg, raising=False)
    with pytest.raises(guard.QuarantineError, match="frozen mixed-scope PGID"):
        guard.recover_marker(
            marker,
            role="dgx_a",
            production=False,
            process_provider=lambda: [dict(row) for row in current],
            ledger_provider=lambda _roots: (set(), []),
            port_scanner=lambda _ports: (True, []),
            term_timeout=0,
            kill_timeout=0,
        )
    assert signalled == [(group, guard.signal.SIGTERM)]
    assert marker.is_file()


def test_recovery_keeps_quarantine_if_leader_exits_before_first_fresh_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "quarantine"
    root = "/tmp/t5-owned-scope"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="recover:leader-race",
        roots=[root],
        production=False,
    )
    group = 424242
    leader = _process_row(
        group,
        ppid=1,
        pgid=group,
        sid=group,
        starttime=100,
        command=f"/usr/bin/python3 {root}/launch.py",
    )
    native = _process_row(
        group + 1,
        ppid=1,
        pgid=group,
        sid=group,
        starttime=101,
        command="/opt/ros/humble/lib/controller_server",
    )
    calls = 0

    def process_provider():
        nonlocal calls
        calls += 1
        return [dict(row) for row in ([leader, native] if calls == 1 else [native])]

    signalled: list[tuple[int, object]] = []
    monkeypatch.setattr(
        guard.os,
        "killpg",
        lambda pgid, requested: signalled.append((pgid, requested)),
        raising=False,
    )
    with pytest.raises(guard.QuarantineError, match="unassociated PGID"):
        guard.recover_marker(
            marker,
            role="dgx_a",
            production=False,
            process_provider=process_provider,
            ledger_provider=lambda _roots: (set(), []),
            port_scanner=lambda _ports: (True, []),
            term_timeout=0,
            kill_timeout=0,
        )
    assert signalled == []
    assert marker.is_file()


def test_recovery_keeps_quarantine_if_unclaimed_frozen_survivor_resists_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kill_signal = getattr(guard.signal, "SIGKILL", None)
    if kill_signal is None:  # Windows unit runner; production recovery is Linux-only.
        kill_signal = type("KillSignal", (), {"name": "SIGKILL"})()
        monkeypatch.setattr(guard.signal, "SIGKILL", kill_signal, raising=False)
    marker = tmp_path / "quarantine"
    root = "/tmp/t5-owned-scope"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="recover:kill-survivor",
        roots=[root],
        production=False,
    )
    group = 424242
    leader = _process_row(
        group,
        ppid=1,
        pgid=group,
        sid=group,
        starttime=100,
        command=f"/usr/bin/python3 {root}/launch.py",
    )
    child = _process_row(
        group + 1,
        ppid=group,
        pgid=group,
        sid=group,
        starttime=101,
        command="/opt/ros/humble/lib/controller_server",
    )
    current = [leader, child]
    signalled: list[tuple[int, object]] = []

    def fake_killpg(pgid: int, requested: object) -> None:
        nonlocal current
        signalled.append((pgid, requested))
        survivor = dict(child)
        survivor["ppid"] = 1
        current = [survivor]

    monkeypatch.setattr(guard.os, "killpg", fake_killpg, raising=False)
    with pytest.raises(guard.QuarantineError, match="scoped_processes_absent"):
        guard.recover_marker(
            marker,
            role="dgx_a",
            production=False,
            process_provider=lambda: [dict(row) for row in current],
            ledger_provider=lambda _roots: (set(), []),
            port_scanner=lambda _ports: (True, []),
            term_timeout=0,
            kill_timeout=0,
        )
    assert signalled == [
        (group, guard.signal.SIGTERM),
        (group, kill_signal),
    ]
    assert marker.is_file()


@pytest.mark.parametrize(
    "mutation",
    (
        "leader_outside_root",
        "leader_adjacent_prefix",
        "unreadable_command",
        "mixed_sid",
        "unrelated_ancestry",
    ),
)
def test_recovery_rejects_unproven_native_descendant_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    marker = tmp_path / "quarantine"
    root = "/tmp/t5-owned-scope"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag=f"recover:reject-{mutation}",
        roots=[root],
        production=False,
    )
    group = 424242
    leader_command = f"/usr/bin/python3 {root}/launch.py"
    child_command = "/opt/ros/humble/lib/controller_server"
    rows = [
        _process_row(
            group,
            ppid=1,
            pgid=group,
            sid=group,
            starttime=100,
            command=leader_command,
        ),
        _process_row(
            group + 1,
            ppid=group,
            pgid=group,
            sid=group,
            starttime=101,
            command=child_command,
        ),
    ]
    if mutation == "leader_outside_root":
        # Keep the group discoverable through a scoped child, but require the
        # actual PID==PGID leader to carry the run root.
        rows[0]["command"] = "/usr/bin/ros2 launch foreign"
        rows[1]["command"] = f"{root}/native_child"
    elif mutation == "leader_adjacent_prefix":
        rows[0]["command"] = f"/usr/bin/python3 {root}-copy/launch.py"
        rows[1]["command"] = f"{root}/native_child"
    elif mutation == "unreadable_command":
        rows[1]["command_readable"] = False
    elif mutation == "mixed_sid":
        rows[1]["sid"] = group + 99
    elif mutation == "unrelated_ancestry":
        rows[1]["ppid"] = 1

    signalled: list[tuple[int, object]] = []
    monkeypatch.setattr(
        guard.os,
        "killpg",
        lambda pgid, requested: signalled.append((pgid, requested)),
        raising=False,
    )
    with pytest.raises(guard.QuarantineError, match="mixed-scope PGID"):
        guard.recover_marker(
            marker,
            role="dgx_a",
            production=False,
            process_provider=lambda: list(rows),
            ledger_provider=lambda _roots: ({group}, []),
            port_scanner=lambda _ports: (True, []),
            term_timeout=0.01,
            kill_timeout=0.01,
        )
    assert signalled == []
    assert marker.is_file()


@pytest.mark.parametrize("identity_field", ("sid", "starttime"))
def test_recovery_rejects_mixed_group_when_leader_identity_changes_on_fresh_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_field: str,
) -> None:
    marker = tmp_path / "quarantine"
    root = "/tmp/t5-owned-scope"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="recover:leader-reused",
        roots=[root],
        production=False,
    )
    group = 424242
    initial = [
        _process_row(
            group,
            ppid=1,
            pgid=group,
            sid=group,
            starttime=100,
            command=f"/usr/bin/python3 {root}/launch.py",
        ),
        _process_row(
            group + 1,
            ppid=group,
            pgid=group,
            sid=group,
            starttime=101,
            command="/opt/ros/humble/lib/controller_server",
        ),
    ]
    changed = [dict(row) for row in initial]
    changed[0][identity_field] += 999
    calls = 0

    def process_provider():
        nonlocal calls
        calls += 1
        # The preserved discovery snapshot is followed by an immediate fresh
        # pre-signal read, which must catch every leader identity mutation.
        return [dict(row) for row in (initial if calls == 1 else changed)]

    signalled: list[tuple[int, object]] = []
    monkeypatch.setattr(
        guard.os,
        "killpg",
        lambda pgid, requested: signalled.append((pgid, requested)),
        raising=False,
    )
    with pytest.raises(guard.QuarantineError, match="group leader identity changed"):
        guard.recover_marker(
            marker,
            role="dgx_a",
            production=False,
            process_provider=process_provider,
            ledger_provider=lambda _roots: ({group}, []),
            port_scanner=lambda _ports: (True, []),
            term_timeout=0.01,
            kill_timeout=0.01,
        )
    assert signalled == []
    assert marker.is_file()


def test_recovery_accepts_run_root_leader_reparented_before_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "quarantine"
    root = "/tmp/t5-owned-scope"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="recover:leader-reparented",
        roots=[root],
        production=False,
    )
    group = 424242
    initial = [
        _process_row(
            group,
            ppid=999,
            pgid=group,
            sid=group,
            starttime=100,
            command=f"/usr/bin/python3 {root}/launch.py",
        ),
        _process_row(
            group + 1,
            ppid=group,
            pgid=group,
            sid=group,
            starttime=101,
            command="/opt/ros/humble/lib/controller_server",
        ),
    ]
    reparented = [dict(row) for row in initial]
    reparented[0]["ppid"] = 1
    calls = 0

    def process_provider():
        nonlocal calls
        calls += 1
        if calls == 1:
            return [dict(row) for row in initial]
        if calls == 2:
            return [dict(row) for row in reparented]
        return []

    signalled: list[tuple[int, object]] = []
    monkeypatch.setattr(
        guard.os,
        "killpg",
        lambda pgid, requested: signalled.append((pgid, requested)),
        raising=False,
    )
    receipt = guard.recover_marker(
        marker,
        role="dgx_a",
        production=False,
        process_provider=process_provider,
        ledger_provider=lambda _roots: ({group}, []),
        port_scanner=lambda _ports: (True, []),
        term_timeout=0.01,
        kill_timeout=0.01,
    )
    assert receipt["status"] == "PASS"
    assert signalled and signalled[0][0] == group
    assert not marker.exists()


def test_stopped_foreign_same_name_container_keeps_x86_quarantined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    value = [
        {
            "Config": {
                "Labels": {"internnav.t5.deployment_root": "/tmp/foreign-root"}
            },
            "State": {"Running": False, "Pid": 0},
        }
    ]

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(value), stderr="")

    monkeypatch.setattr(guard.subprocess, "run", fake_run)
    clean, rows = guard._recover_x86_containers(
        ["/tmp/owned-root"], ("internnav_t5_isaac_a",)
    )
    assert clean is False
    assert rows[0]["owned"] is False
    assert rows[0]["stopped"] is False
    assert all(call[:2] == ["docker", "inspect"] for call in calls)


def test_container_inspect_error_is_not_treated_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="permission denied contacting docker daemon"
        )

    monkeypatch.setattr(guard.subprocess, "run", fake_run)
    with pytest.raises(guard.QuarantineError, match="cannot prove container absence"):
        guard._recover_x86_containers(
            ["/tmp/owned-root"], ("internnav_t5_isaac_a",)
        )


def test_owned_running_container_stop_does_not_pollute_json_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = {
        "Config": {
            "Labels": {"internnav.t5.deployment_root": "/tmp/owned-root"}
        },
        "State": {"Running": True, "Pid": 123},
    }
    stopped = {
        "Config": running["Config"],
        "State": {"Running": False, "Pid": 0},
    }
    inspect_count = 0
    stop_kwargs: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        nonlocal inspect_count
        if argv[:2] == ["docker", "stop"]:
            stop_kwargs.update(kwargs)
            return subprocess.CompletedProcess(argv, 0, stdout=None, stderr="")
        inspect_count += 1
        value = running if inspect_count == 1 else stopped
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps([value]), stderr=""
        )

    monkeypatch.setattr(guard.subprocess, "run", fake_run)
    clean, rows = guard._recover_x86_containers(
        ["/tmp/owned-root"], ("internnav_t5_isaac_a",)
    )
    assert clean is True
    assert rows[0]["stopped"] is True
    assert stop_kwargs["stdout"] is subprocess.DEVNULL


def test_dangling_health_socket_symlink_is_residual(tmp_path: Path) -> None:
    path = tmp_path / "health.sock"
    path.symlink_to(tmp_path / "missing.sock")
    clean, residual = guard._clean_health_sockets((path,))
    assert clean is False
    assert residual == [f"{path}:not-a-socket"]
    assert path.is_symlink()


def test_per_gpu_markers_are_independent_and_can_be_armed_concurrently(
    tmp_path: Path,
) -> None:
    gpu0 = tmp_path / "gpu0.quarantine"
    gpu1 = tmp_path / "gpu1.quarantine"
    guard.arm_marker(
        gpu0,
        reason="t5_online_stage_in_progress",
        role="x86_gpu0",
        run_tag="lane-a",
        roots=["/tmp/t5-lane-a"],
        production=False,
    )
    guard.arm_marker(
        gpu1,
        reason="t5_online_stage_in_progress",
        role="x86_gpu1",
        run_tag="lane-b",
        roots=["/tmp/t5-lane-b"],
        production=False,
    )
    assert gpu0.is_file() and gpu1.is_file()
    with pytest.raises(guard.QuarantineError, match="does not prove"):
        guard.clear_owned_marker(
            gpu0,
            reason="t5_online_stage_in_progress",
            role="x86_gpu0",
            run_tag="lane-a",
            roots=["/tmp/t5-lane-a"],
            cleanup_receipt=json.dumps(
                {"schema_version": 1, "status": "FAIL", "checks": {"zero": False}}
            ).encode(),
            production=False,
        )
    assert gpu0.is_file() and gpu1.is_file()
    guard.clear_owned_marker(
        gpu1,
        reason="t5_online_stage_in_progress",
        role="x86_gpu1",
        run_tag="lane-b",
        roots=["/tmp/t5-lane-b"],
        cleanup_receipt=_pass_receipt(),
        production=False,
    )
    assert gpu0.is_file()  # Lane A failure cannot keep a clean Lane B quarantined.
    assert not gpu1.exists()
    guard.clear_owned_marker(
        gpu0,
        reason="t5_online_stage_in_progress",
        role="x86_gpu0",
        run_tag="lane-a",
        roots=["/tmp/t5-lane-a"],
        cleanup_receipt=_pass_receipt(),
        production=False,
    )
    assert not gpu0.exists()


def test_recovery_entry_and_lease_mode_cannot_launch_arbitrary_workload() -> None:
    recovery = RECOVERY_PATH.read_text(encoding="utf-8")
    lease = LEASE_PATH.read_text(encoding="utf-8")
    common = COMMON_PATH.read_text(encoding="utf-8")
    assert "recover-dgx-a" in lease
    assert "recover-dgx-b" in lease
    assert "recover-isaac" in lease
    assert "recover-isaac-gpu0" in lease
    assert "recover-isaac-gpu1" in lease
    assert "recover-all" in lease
    assert "recovery-only mode cannot run an arbitrary command" in lease
    assert 'command_argv[1]}" == "$recovery_entry"' in lease
    assert "RECOVERY_ONLY resource=%s" in lease
    assert "/tmp/internnav_isaac.quarantine|/tmp/internnav_isaac_gpu0.quarantine" in lease
    assert "/tmp/internnav_isaac.quarantine|/tmp/internnav_isaac_gpu1.quarantine" in lease
    assert "t5_quarantine_recover" in common
    assert "t5_quarantine_owned_observe" in common
    assert "workload_started" in recovery
    for forbidden in (
        "run_t5_dgx_lane.sh",
        "run_t5_distributed_isaac.sh",
        "docker start",
        "HF_TOKEN",
        ".env.local",
        "T5_DUAL_LANE_BOARD",
        "golden_bundle",
        "finalize_t5_d0",
    ):
        assert forbidden not in recovery


def test_only_guard_unlinks_quarantine_markers() -> None:
    common = COMMON_PATH.read_text(encoding="utf-8")
    recovery = RECOVERY_PATH.read_text(encoding="utf-8")
    assert re.search(r"\brm\s+[^\n]*quarantine", common) is None
    assert re.search(r"\brm\s+[^\n]*quarantine", recovery) is None
    assert "marker.unlink()" in GUARD_PATH.read_text(encoding="utf-8")
