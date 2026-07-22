import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "scripts" / "t5_stale_nav2_recovery.py"
ENTRY_PATH = ROOT / "scripts" / "recover_t5_stale_dgx_nav2.sh"
LEASE_PATH = ROOT / "scripts" / "with_resource_lease.sh"
AUDITOR_PATH = ROOT / "scripts" / "t5_process_identity_audit.py"
GUARD_PATH = ROOT / "scripts" / "t5_resource_quarantine.py"
MANIFEST_PATH = (
    ROOT / "configs" / "internnav_t5" / "attempt10_stale_nav2_recovery_manifest.json"
)
OLD_ROOT = (
    "/home/railgun/internnav-t1-t2/.t5-deployments/"
    "t5d0020260719t135307-78cc4b1e7b1b-lane-a"
)


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


engine = _import(ENGINE_PATH, "t5_stale_nav2_recovery")
auditor = _import(AUDITOR_PATH, "t5_stale_nav2_auditor")
guard = _import(GUARD_PATH, "t5_stale_nav2_guard")


def _raw_argv(argv: list[str]) -> bytes:
    return b"\0".join(value.encode() for value in argv) + b"\0"


def _leader(role: str, pid: int, argv: list[str]) -> dict:
    return {
        "role": role,
        "pid": pid,
        "ppid": 1,
        "pgid": pid,
        "sid": pid,
        "starttime": 1182700,
        "state": "S",
        "argv": argv,
        "argv_sha256": hashlib.sha256(_raw_argv(argv)).hexdigest(),
    }


def _evidence() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _evidence_raw(value: dict | None = None) -> tuple[bytes, str]:
    raw = (
        MANIFEST_PATH.read_bytes()
        if value is None
        else json.dumps(value, sort_keys=True).encode()
    )
    return raw, hashlib.sha256(raw).hexdigest()


def _row(leader: dict, *, uid: int = 1000) -> dict:
    return {
        **{key: leader[key] for key in ("pid", "ppid", "pgid", "sid", "starttime", "state")},
        "uid": uid,
        "argv": list(leader["argv"]),
        "raw_argv_sha256": leader["argv_sha256"],
    }


def _child(pid: int, pgid: int, executable: str) -> dict:
    argv = [
        f"/opt/ros/jazzy/lib/{executable}/{executable}",
        "--ros-args",
        "-r",
        "__ns:=/t5/lane_a",
    ]
    return {
        "pid": pid,
        "ppid": pgid,
        "pgid": pgid,
        "sid": pgid,
        "starttime": 1182701,
        "state": "S",
        "uid": 1000,
        "argv": argv,
        "raw_argv_sha256": hashlib.sha256(_raw_argv(argv)).hexdigest(),
    }


def _initial_rows(value: dict | None = None) -> list[dict]:
    evidence = value or _evidence()
    leaders = {row["role"]: row for row in evidence["leaders"]}
    return [
        *[_row(row) for row in evidence["leaders"]],
        _child(247300, leaders["nav2_bringup"]["pgid"], "controller_server"),
        # The docking child is not a separate target.  It is deliberately in
        # the launch leader's complete process group.
        _child(247301, leaders["nav2_bringup"]["pgid"], "docking_server"),
        _child(247302, leaders["lifecycle_navigation"]["pgid"], "lifecycle_manager"),
        _child(247303, leaders["lifecycle_collision"]["pgid"], "lifecycle_manager"),
    ]


class MutableProcesses:
    def __init__(self, rows: list[dict], *, require_kill: bool = False):
        self.rows = rows
        self.require_kill = require_kill
        self.signals: list[tuple[int, int]] = []
        self.now = 0.0

    def snapshot(self):
        return ([dict(row) for row in self.rows], [])

    def send(self, pgid: int, requested: int):
        self.signals.append((pgid, requested))
        if requested == engine.SIG_KILL or not self.require_kill:
            self.rows = [row for row in self.rows if row["pgid"] != pgid]

    def monotonic(self):
        return self.now

    def sleep(self, duration: float):
        self.now += duration


class OneShotProcRace:
    def __init__(self, state: MutableProcesses, fail_on_call: int):
        self.state = state
        self.fail_on_call = fail_on_call
        self.calls = 0

    def snapshot(self):
        self.calls += 1
        if self.calls == self.fail_on_call:
            return ([], ["pid=999999:RecoveryError"])
        return self.state.snapshot()


def _run(state: MutableProcesses, evidence: dict | None = None):
    value = evidence or _evidence()
    return engine.recover_stale_nav2(
        value,
        OLD_ROOT,
        auditor,
        snapshot_provider=state.snapshot,
        port_scanner=lambda _ports: (True, [], None),
        signaler=state.send,
        expected_uid=1000,
        term_timeout=1.0,
        kill_timeout=1.0,
        monotonic=state.monotonic,
        sleeper=state.sleep,
    )


def test_exact_schema_and_read_only_archive_are_sha_bound(tmp_path: Path) -> None:
    raw, digest = _evidence_raw()
    source = tmp_path / "source.json"
    source.write_bytes(raw)
    archive = tmp_path / "archive.json"
    staged = engine.stage_evidence(source, archive, digest, OLD_ROOT)
    assert staged["status"] == "PASS"
    assert staged["archive_read_only"] is True
    assert archive.read_bytes() == raw
    verified = engine.verify_archived_evidence(archive, digest, OLD_ROOT)
    assert verified["status"] == "PASS"
    assert verified["read_only"] is True
    with pytest.raises(engine.RecoveryError, match="SHA-256"):
        engine.verify_archived_evidence(archive, "0" * 64, OLD_ROOT)


def test_recovery_is_immutably_bound_to_attempt10_manifest() -> None:
    raw = MANIFEST_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    value = engine.validate_evidence(raw, digest, OLD_ROOT)
    assert digest == engine.FIXED_MANIFEST_SHA256
    assert value["source_conflict_sha256"] == engine.FIXED_SOURCE_CONFLICT_SHA256
    assert value["deployment_root"] == engine.FIXED_OLD_ROOT
    for leader in value["leaders"]:
        fixed = engine.FIXED_LEADERS[leader["role"]]
        assert all(leader[key] == expected for key, expected in fixed.items())

    alternate = json.loads(raw)
    alternate["source_conflict_sha256"] = "a" * 64
    alternate_raw = json.dumps(alternate, sort_keys=True).encode()
    with pytest.raises(engine.RecoveryError):
        engine.validate_evidence(
            alternate_raw, hashlib.sha256(alternate_raw).hexdigest(), OLD_ROOT
        )


@pytest.mark.parametrize(
    "root",
    [
        "/tmp/deployment",
        "/home/rail/internnav-t1-t2/.t5-deployments/lane-a",
        "/home/railgun/internnav-t1-t2/.t5-deployments/../foreign",
        "/home/railgun/internnav-t1-t2/.t5-deployments/a/nested",
    ],
)
def test_only_one_absolute_old_dgx_a_deployment_root_is_accepted(root: str) -> None:
    with pytest.raises(engine.RecoveryError):
        engine.validate_old_root(root)


def test_recovery_engine_keeps_python38_compatible_string_handling() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    assert ".removeprefix(" not in source
    assert "normalized[len(OLD_ROOT_PREFIX) :]" in source


def test_term_cleans_only_three_exact_session_groups_including_docking_child() -> None:
    evidence = _evidence()
    state = MutableProcesses(_initial_rows(evidence))
    receipt = _run(state, evidence)
    target_pgids = {row["pgid"] for row in evidence["leaders"]}
    assert receipt["status"] == "PASS"
    assert all(receipt["checks"].values())
    assert state.rows == []
    assert set(state.signals) == {(pgid, engine.SIG_TERM) for pgid in target_pgids}
    nav_members = receipt["initial_group_members"]["247273"]
    assert any("docking_server" in row["identities"] for row in nav_members)
    assert all(row["pgid"] in target_pgids for row in receipt["signals"])


def test_term_timeout_escalates_only_continuously_present_exact_groups() -> None:
    evidence = _evidence()
    state = MutableProcesses(_initial_rows(evidence), require_kill=True)
    receipt = _run(state, evidence)
    target_pgids = {row["pgid"] for row in evidence["leaders"]}
    assert receipt["status"] == "PASS"
    assert {pgid for pgid, sig in state.signals if sig == engine.SIG_TERM} == target_pgids
    assert {pgid for pgid, sig in state.signals if sig == engine.SIG_KILL} == target_pgids
    assert receipt["residual_after_term_pgids"] == sorted(target_pgids)


def test_transient_proc_race_is_retried_before_any_signal() -> None:
    evidence = _evidence()
    state = MutableProcesses(_initial_rows(evidence))
    flaky = OneShotProcRace(state, fail_on_call=2)
    receipt = engine.recover_stale_nav2(
        evidence,
        OLD_ROOT,
        auditor,
        snapshot_provider=flaky.snapshot,
        port_scanner=lambda _ports: (True, [], None),
        signaler=state.send,
        expected_uid=1000,
        term_timeout=1.0,
        kill_timeout=1.0,
        monotonic=state.monotonic,
        sleeper=state.sleep,
    )
    assert receipt["status"] == "PASS"
    assert flaky.calls >= 3
    assert len(state.signals) == 3


@pytest.mark.parametrize("mutation", ["starttime", "argv", "namespace", "mixed_sid"])
def test_identity_or_session_drift_refuses_every_signal(mutation: str) -> None:
    evidence = _evidence()
    rows = _initial_rows(evidence)
    if mutation == "starttime":
        rows[0]["starttime"] += 1
    elif mutation == "argv":
        rows[0]["argv"] = [*rows[0]["argv"], "unexpected:=true"]
    elif mutation == "namespace":
        index = rows[1]["argv"].index("__ns:=/t5/lane_a")
        rows[1]["argv"][index] = "__ns:=/t5/lane_b"
        rows[1]["raw_argv_sha256"] = hashlib.sha256(
            _raw_argv(rows[1]["argv"])
        ).hexdigest()
    else:
        rows[-1]["sid"] = 999999
    state = MutableProcesses(rows)
    with pytest.raises(engine.RecoveryError):
        _run(state, evidence)
    assert state.signals == []


def test_unrelated_structured_compute_or_t5_port_keeps_receipt_failed() -> None:
    evidence = _evidence()
    model_argv = ["python3", "-m", "internvla_ros2.model_node"]
    unrelated_model = {
        "pid": 300001,
        "ppid": 1,
        "pgid": 300001,
        "sid": 300001,
        "starttime": 1182800,
        "state": "S",
        "uid": 1000,
        "argv": model_argv,
        "raw_argv_sha256": hashlib.sha256(_raw_argv(model_argv)).hexdigest(),
    }
    state = MutableProcesses([*_initial_rows(evidence), unrelated_model])
    receipt = _run(state, evidence)
    assert receipt["status"] == "FAIL"
    assert receipt["checks"]["structured_compute_audit_zero"] is False
    assert receipt["structured_compute_matches"][0]["pid"] == 300001

    state = MutableProcesses(_initial_rows(evidence))
    receipt = engine.recover_stale_nav2(
        evidence,
        OLD_ROOT,
        auditor,
        snapshot_provider=state.snapshot,
        port_scanner=lambda _ports: (False, ["LISTEN *:25137"], None),
        signaler=state.send,
        expected_uid=1000,
        term_timeout=0,
        kill_timeout=0,
        monotonic=state.monotonic,
        sleeper=state.sleep,
    )
    assert receipt["status"] == "FAIL"
    assert receipt["checks"]["t5_ports_zero"] is False


def test_failed_receipt_cannot_clear_owned_marker(tmp_path: Path) -> None:
    marker = tmp_path / "quarantine"
    guard.arm_marker(
        marker,
        reason="t5_online_stage_in_progress",
        role="dgx_a",
        run_tag="stale_nav2:test",
        roots=[OLD_ROOT],
        production=False,
    )
    failed = {
        "schema_version": 1,
        "status": "FAIL",
        "checks": {"structured_compute_audit_zero": False},
    }
    with pytest.raises(guard.QuarantineError, match="does not prove"):
        guard.clear_owned_marker(
            marker,
            reason="t5_online_stage_in_progress",
            role="dgx_a",
            run_tag="stale_nav2:test",
            roots=[OLD_ROOT],
            cleanup_receipt=json.dumps(failed).encode(),
            production=False,
        )
    assert marker.is_file()


def test_fixed_lease_entry_and_shell_have_no_substring_kill_escape_hatch() -> None:
    entry = ENTRY_PATH.read_text(encoding="utf-8")
    lease = LEASE_PATH.read_text(encoding="utf-8")
    assert "recover-dgx-a" in entry
    assert "attempt10_stale_nav2_recovery_manifest.json" in entry
    assert engine.FIXED_MANIFEST_SHA256 in entry
    assert engine.FIXED_MANIFEST_SHA256 in lease
    assert engine.FIXED_OLD_ROOT in entry
    assert engine.FIXED_OLD_ROOT in lease
    assert engine.FIXED_SOURCE_CONFLICT_SHA256 in ENGINE_PATH.read_text(encoding="utf-8")
    assert 'test "$role" = dgx-a' in entry
    assert "stale-dgx-recovery-" in entry
    assert "t5_quarantine_arm" in entry
    assert "t5_quarantine_owned_observe" in entry
    assert "stale-dgx-recovery-attempt10-c3f2d1f" in entry
    assert "t5_quarantine_owned_clear" in entry
    assert entry.index("t5_quarantine_arm") < entry.index("recover '$old_root_b64'")
    assert entry.index("remote_receipt_all_true") < entry.index(
        '"owned_clear_after_pass_receipt"'
    )
    assert "recover_t5_stale_dgx_nav2.sh" in lease
    assert '"$mode" == recover-dgx-a' in lease
    assert "recovery-only mode cannot run an arbitrary command" in lease
    combined = entry + "\n" + ENGINE_PATH.read_text(encoding="utf-8")
    for forbidden in ("pgrep", "pkill", "killall"):
        assert forbidden not in combined


def test_recovery_lease_rejects_an_arbitrary_command_before_ssh(tmp_path: Path) -> None:
    # Validation is local and runs before any SSH holder can be launched.
    completed = subprocess.run(
        [
            "wsl",
            "bash",
            str(LEASE_PATH).replace("C:\\", "/mnt/c/").replace("\\", "/"),
            "recover-dgx-a",
            "--task",
            "fixture",
            "--log-dir",
            str(tmp_path).replace("C:\\", "/mnt/c/").replace("\\", "/"),
            "--",
            "bash",
            "-c",
            "true",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 64
    assert "cannot run an arbitrary command" in completed.stderr
