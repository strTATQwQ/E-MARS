from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import sensor_runtime.processes as processes
from sensor_runtime.processes import (
    ProcessRecord,
    ProcessRegistry,
    RequiredProcessExited,
    cleanup_persisted_ledger,
    linux_start_ticks,
    live_group_members,
)


pytestmark = pytest.mark.skipif(os.name != "posix", reason="Linux /proc and process groups required")


def _devnull():
    return open(os.devnull, "wb")


def test_setsid_wait_holds_forking_client_and_propagates_child_exit() -> None:
    setsid = shutil.which("setsid")
    assert setsid is not None
    started = time.monotonic()
    completed = subprocess.run(
        [
            setsid,
            "--wait",
            sys.executable,
            "-c",
            "import sys,time; time.sleep(0.15); sys.exit(7)",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=5.0,
        start_new_session=True,
    )
    assert completed.returncode == 7, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    assert time.monotonic() - started >= 0.12


def test_managed_command_observer_cannot_self_contaminate_target_group(
    tmp_path: Path,
) -> None:
    real_ps = shutil.which("ps")
    assert real_ps is not None
    shim = tmp_path / "ps"
    shim.write_text(
        "#!/bin/sh\n"
        "sleep 0.15\n"
        'exec "$REAL_PS" "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    environment = os.environ.copy()
    environment["REAL_PS"] = real_ps
    environment["PATH"] = f"{tmp_path}{os.pathsep}{environment['PATH']}"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sensor_runtime.managed_command",
            "--",
            "/bin/true",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10.0,
        start_new_session=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_child_exec_unblocks_handled_signals_and_term_exits_gracefully(tmp_path: Path) -> None:
    registry = ProcessRegistry(tmp_path)
    code = (
        "import json,signal,sys,time; "
        "blocked=signal.pthread_sigmask(signal.SIG_BLOCK, []); "
        "print(json.dumps(sorted(int(x) for x in blocked)), flush=True); "
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
        "time.sleep(30)"
    )
    process = registry.start("mask", [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    blocked = json.loads(process.stdout.readline().decode())
    assert not {int(signal.SIGINT), int(signal.SIGTERM), int(signal.SIGHUP)}.intersection(blocked)
    os.killpg(process.pid, signal.SIGTERM)
    assert process.wait(timeout=2.0) == 0
    assert registry.cleanup()["cleanup_confirmed"] is True


def test_completion_registry_accepts_configured_sigterm_status_with_zero_residuals(
    tmp_path: Path,
) -> None:
    registry = ProcessRegistry(
        tmp_path,
        term_timeout_sec=3.0,
        allowed_required_exit_codes={0, 143},
    )
    code = (
        "import signal,sys,time; "
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(143)); "
        "time.sleep(30)"
    )
    with _devnull() as output:
        registry.start(
            "completion",
            [sys.executable, "-c", code],
            stdout=output,
            stderr=output,
        )
        cleanup = registry.cleanup()
    assert cleanup["cleanup_confirmed"] is True
    assert cleanup["residual_cleanup_confirmed"] is True
    assert cleanup["allowed_required_exit_codes"] == [0, 143]
    assert cleanup["groups"][0]["exit_code"] == 143
    assert cleanup["groups"][0]["clean_exit"] is True


def test_per_role_early_exit_writes_first_failure_immediately(tmp_path: Path) -> None:
    with _devnull() as output:
        registry = ProcessRegistry(tmp_path)
        registry.start("required", [sys.executable, "-c", "raise SystemExit(7)"], stdout=output, stderr=output)
        registry.start("other", [sys.executable, "-c", "import time; time.sleep(30)"], stdout=output, stderr=output)
        registry.start_monitor()
        deadline = time.monotonic() + 2.0
        while not registry.failure_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        failure = json.loads(registry.failure_path.read_text())
        assert (failure["role"], failure["exit_code"]) == ("required", 7)
        with pytest.raises(RuntimeError, match="cleanup"):
            registry.cleanup()
        cleanup = json.loads(registry.cleanup_path.read_text())
        assert cleanup["pid_count"] == cleanup["pgid_count"] == 0


def test_failure_evidence_write_fault_still_latches_and_check_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _devnull() as output:
        registry = ProcessRegistry(tmp_path)
        registry.start("required", [sys.executable, "-c", "raise SystemExit(3)"], stdout=output, stderr=output)
        original = processes.atomic_write_json

        def broken(path: Path, payload: dict) -> None:
            if path == registry.failure_path:
                raise OSError("disk fault")
            original(path, payload)

        monkeypatch.setattr(processes, "atomic_write_json", broken)
        registry.start_monitor()
        time.sleep(0.1)
        with pytest.raises(RequiredProcessExited, match="evidence write failed"):
            registry.check("test")
        with pytest.raises(RuntimeError, match="cleanup"):
            registry.cleanup()


def test_monitor_thread_death_is_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _devnull() as output:
        registry = ProcessRegistry(tmp_path)
        registry.start("required", [sys.executable, "-c", "import time; time.sleep(30)"], stdout=output, stderr=output)
        monkeypatch.setattr(registry, "_monitor", lambda: None)
        registry.start_monitor()
        registry._monitor_thread.join(1.0)  # type: ignore[union-attr]
        with pytest.raises(RequiredProcessExited, match="monitor died"):
            registry.check("test")
        with pytest.raises(RuntimeError, match="cleanup"):
            registry.cleanup()


def test_cleanup_kills_descendant_when_group_leader_already_exited(tmp_path: Path) -> None:
    with _devnull() as output:
        registry = ProcessRegistry(tmp_path)
        process = registry.start(
            "leader",
            [sys.executable, "-c", "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); sys.exit(0)"],
            stdout=output,
            stderr=output,
            expected_long_running=False,
        )
        assert process.wait(timeout=2.0) == 0
        assert live_group_members(process.pid)
        cleanup = registry.cleanup()
        assert cleanup["pid_count"] == cleanup["pgid_count"] == 0


def test_ledger_write_failure_recovers_spawned_group_and_descendant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _devnull() as output:
        registry = ProcessRegistry(tmp_path)

        def fail_after_child_can_spawn() -> None:
            time.sleep(0.15)
            raise OSError("ledger rename failed")

        monkeypatch.setattr(registry, "_persist_ledger", fail_after_child_can_spawn)
        with pytest.raises(OSError, match="ledger"):
            registry.start(
                "broken",
                [sys.executable, "-c", "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); time.sleep(30)"],
                stdout=output,
                stderr=output,
            )
        recovery = json.loads((tmp_path / "spawn_failure_recovery.json").read_text())
        assert recovery["status"] == "PASS"
        assert recovery["live_after_kill"] == []


def _record(role: str, process: subprocess.Popen[bytes]) -> ProcessRecord:
    ticks = linux_start_ticks(process.pid)
    assert ticks is not None
    return ProcessRecord(role, process.pid, process.pid, True, True, time.time(), time.monotonic_ns(), ticks, "python")


def _direct_sleep() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def test_partial_persisted_ledger_still_cleans_recorded_group_but_fails_contract(tmp_path: Path) -> None:
    lifecycle = tmp_path / "life"
    lifecycle.mkdir()
    process = _direct_sleep()
    try:
        (lifecycle / "process_ledger.json").write_text(json.dumps({"processes": [_record("bridge", process).as_dict()]}))
        payload = cleanup_persisted_ledger(lifecycle, [], tmp_path / "cleanup.json", expected_roles={"bridge", "sidecar", "recorder"})
        assert payload["status"] == "FAIL" and payload["pid_count"] == payload["pgid_count"] == 0
        assert live_group_members(process.pid) == []
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_conflicting_duplicate_role_cleans_both_unique_groups_and_reports_fail(tmp_path: Path) -> None:
    lifecycle = tmp_path / "life"
    lifecycle.mkdir()
    first, second = _direct_sleep(), _direct_sleep()
    try:
        records = [_record("bridge", first).as_dict(), _record("bridge", second).as_dict()]
        (lifecycle / "process_ledger.json").write_text(json.dumps({"processes": records}))
        payload = cleanup_persisted_ledger(lifecycle, [], tmp_path / "cleanup.json", expected_roles={"bridge"})
        assert payload["status"] == "FAIL"
        assert payload["pid_count"] == payload["pgid_count"] == 0
        assert len(payload["groups"]) == 2
        assert live_group_members(first.pid) == live_group_members(second.pid) == []
    finally:
        for process in (first, second):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
