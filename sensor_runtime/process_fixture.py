#!/usr/bin/env python3
"""Stdlib-only POSIX lifecycle fixture for hosts without pytest installed."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .isaac_eula import (
    FROZEN_ISAAC_RUNTIME_PREFLIGHT,
    apply_frozen_isaac_eula_environment,
)
from .processes import (
    ProcessRecord,
    ProcessRegistry,
    cleanup_persisted_ledger,
    linux_start_ticks,
    live_group_members,
    terminate_bound_process_group,
)
from .outer_liveness import OuterAliveLock, probe_outer_alive


def _record(role: str, process: subprocess.Popen[bytes]) -> ProcessRecord:
    ticks = linux_start_ticks(process.pid)
    assert ticks is not None
    return ProcessRecord(
        role, process.pid, process.pid, True, True,
        time.time(), time.monotonic_ns(), ticks, "python",
    )


def _sleep_group() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def main() -> int:
    if os.name != "posix":
        raise RuntimeError("POSIX fixture requires Linux")
    checks: list[str] = []
    with tempfile.TemporaryDirectory(prefix="01r-process-fixture-") as value:
        root = Path(value)

        # A deliberately slow ps must not be able to observe itself inside the
        # workload PGID when managed_command checks for residual descendants.
        real_ps = shutil.which("ps")
        assert real_ps is not None
        observer_dir = root / "observer"
        observer_dir.mkdir()
        observer = observer_dir / "ps"
        observer.write_text(
            "#!/bin/sh\n"
            "sleep 0.15\n"
            'exec "$REAL_PS" "$@"\n',
            encoding="utf-8",
        )
        observer.chmod(0o755)
        observer_env = os.environ.copy()
        observer_env["REAL_PS"] = real_ps
        observer_env["PATH"] = (
            f"{observer_dir}{os.pathsep}{observer_env['PATH']}"
        )
        observer_env["PYTHONDONTWRITEBYTECODE"] = "1"
        managed_clean = subprocess.run(
            [
                sys.executable,
                "-m",
                "sensor_runtime.managed_command",
                "--",
                "/bin/true",
            ],
            env=observer_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10.0,
            start_new_session=True,
        )
        assert managed_clean.returncode == 0, managed_clean.stderr.decode(
            "utf-8", errors="replace"
        )
        checks.append("managed_command_observer_group_isolation")

        # docker exec can present setsid with an existing session/group
        # leader.  GNU setsid must wait for its forked supervisor and
        # propagate the real exit status instead of returning cleanly early.
        setsid = shutil.which("setsid")
        assert setsid is not None
        setsid_started = time.monotonic()
        setsid_wait = subprocess.run(
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
        assert setsid_wait.returncode == 7, setsid_wait.stderr.decode(
            "utf-8", errors="replace"
        )
        assert time.monotonic() - setsid_started >= 0.12
        checks.append("setsid_wait_holds_and_propagates_exit")

        registry = ProcessRegistry(root / "mask")
        code = (
            "import json,signal,sys,time; "
            "print(json.dumps(sorted(int(x) for x in signal.pthread_sigmask(signal.SIG_BLOCK, []))),flush=True); "
            "signal.signal(signal.SIGTERM,lambda *_:sys.exit(0)); time.sleep(30)"
        )
        process = registry.start(
            "mask", [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        blocked = set(json.loads(process.stdout.readline().decode()))
        assert not blocked.intersection({int(signal.SIGINT), int(signal.SIGTERM), int(signal.SIGHUP)})
        os.killpg(process.pid, signal.SIGTERM)
        assert process.wait(timeout=2.0) == 0
        assert registry.cleanup()["cleanup_confirmed"]
        checks.append("child_signal_mask_and_term")

        completion_registry = ProcessRegistry(
            root / "completion_sigterm",
            term_timeout_sec=3.0,
            allowed_required_exit_codes={0, 143},
        )
        completion_code = (
            "import signal,sys,time; "
            "signal.signal(signal.SIGTERM,lambda *_:sys.exit(143)); "
            "print('ready',flush=True); time.sleep(30)"
        )
        completion_process = completion_registry.start(
            "completion",
            [sys.executable, "-c", completion_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert completion_process.stdout is not None
        assert completion_process.stdout.readline().decode().strip() == "ready"
        completion_cleanup = completion_registry.cleanup()
        assert completion_cleanup["cleanup_confirmed"] is True
        assert completion_cleanup["residual_cleanup_confirmed"] is True
        assert completion_cleanup["groups"][0]["exit_code"] == 143
        assert completion_cleanup["groups"][0]["clean_exit"] is True
        checks.append("completion_sigterm_status_with_zero_residuals")

        # Exercise the actual ProcessRegistry exec boundary: the child must
        # receive the normalized EULA environment and a real /dev/null stdin.
        eula_environment = os.environ.copy()
        apply_frozen_isaac_eula_environment(eula_environment)
        eula_registry = ProcessRegistry(root / "isaac_eula")
        eula_code = (
            "import json,os; "
            "from sensor_runtime.isaac_eula import build_runtime_preflight; "
            "print(json.dumps(build_runtime_preflight(os.environ,"
            "stdin_target=os.readlink('/proc/self/fd/0'),"
            "stdin_is_tty=os.isatty(0)),sort_keys=True),flush=True)"
        )
        eula_process = eula_registry.start(
            "isaac_eula_preflight",
            [sys.executable, "-c", eula_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=eula_environment,
            required=False,
            expected_long_running=False,
        )
        eula_stdout, eula_stderr = eula_process.communicate(timeout=3.0)
        assert eula_process.returncode == 0, eula_stderr.decode(
            "utf-8", errors="replace"
        )
        assert json.loads(eula_stdout) == dict(FROZEN_ISAAC_RUNTIME_PREFLIGHT)
        assert eula_registry.cleanup()["cleanup_confirmed"]
        checks.append("isaac_eula_environment_and_devnull_exec_boundary")

        normal_lock = OuterAliveLock(root / "normal_outer_alive.lock")
        normal_evidence = normal_lock.acquire()
        normal_probe = probe_outer_alive(root / "normal_outer_alive.lock")
        assert (normal_probe["device"], normal_probe["inode"]) == (
            normal_evidence["device"], normal_evidence["inode"]
        )
        normal_release = normal_lock.release(
            root / "normal_outer_liveness_release.json",
            inner_probe_completed=True,
            reason="inner_cleanup_probe_completed",
        )
        assert normal_release["inner_supervisor_zero_probe_completed"] is True
        try:
            normal_lock.release(
                root / "duplicate_release.json",
                inner_probe_completed=True,
                reason="duplicate",
            )
        except RuntimeError as exc:
            assert "more than once" in str(exc)
        else:
            raise AssertionError("duplicate outer liveness release was accepted")
        checks.append("outer_liveness_release_exactly_once_after_probe")

        lifecycle = root / "duplicates"
        lifecycle.mkdir()
        first, second = _sleep_group(), _sleep_group()
        try:
            (lifecycle / "process_ledger.json").write_text(
                json.dumps({"processes": [_record("bridge", first).as_dict(), _record("bridge", second).as_dict()]}),
                encoding="utf-8",
            )
            payload = cleanup_persisted_ledger(
                lifecycle, [], root / "duplicate_cleanup.json", expected_roles={"bridge"}
            )
            assert payload["status"] == "FAIL"
            assert payload["pid_count"] == payload["pgid_count"] == payload["socket_count"] == 0
            assert len(payload["groups"]) == 2
            assert live_group_members(first.pid) == live_group_members(second.pid) == []
        finally:
            for child in (first, second):
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        checks.append("conflicting_duplicate_groups_both_zero_and_fail")

        current_ticks = linux_start_ticks(os.getpid())
        assert current_ticks is not None
        try:
            terminate_bound_process_group(
                role="nonisolated_fixture",
                pid=os.getpid(),
                pgid=os.getpgid(0),
                linux_start_identity=current_ticks,
            )
        except RuntimeError as exc:
            assert "recovery probe itself" in str(exc)
        else:
            raise AssertionError("non-isolated recovery group was not rejected")
        isolated = _sleep_group()
        isolated_ticks = linux_start_ticks(isolated.pid)
        assert isolated_ticks is not None and isolated.pid == os.getpgid(isolated.pid)
        isolated_cleanup = terminate_bound_process_group(
            role="isolated_supervisor_fixture",
            pid=isolated.pid,
            pgid=isolated.pid,
            linux_start_identity=isolated_ticks,
        )
        assert isolated_cleanup["identity_verified"] is True
        assert isolated_cleanup["live_after_kill"] == []
        checks.append("supervisor_group_isolation_and_zero")

        broken = ProcessRegistry(root / "spawn_failure")

        def fail_ledger() -> None:
            time.sleep(0.15)
            raise OSError("fixture ledger failure")

        broken._persist_ledger = fail_ledger  # type: ignore[method-assign]
        try:
            broken.start(
                "broken",
                [sys.executable, "-c", "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); time.sleep(30)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass
        else:
            raise AssertionError("ledger persistence failure was not propagated")
        recovery = json.loads((root / "spawn_failure/spawn_failure_recovery.json").read_text())
        assert recovery["status"] == "PASS" and recovery["live_after_kill"] == []
        checks.append("ledger_failure_descendant_recovery_zero")

        owner_dir = root / "pdeath_owner"
        owner_code = (
            "import subprocess,sys,time; from pathlib import Path; "
            "from sensor_runtime.processes import ProcessRegistry; "
            f"root=Path({str(owner_dir)!r}); root.mkdir(); registry=ProcessRegistry(root); "
            "registry.start('pdeath_role',[sys.executable,'-m','sensor_runtime.pdeath_fixture_child'],"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
            "(root/'owner_ready').write_text('ready'); time.sleep(30)"
        )
        owner = subprocess.Popen(
            [sys.executable, "-c", owner_code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        ready_deadline = time.monotonic() + 3.0
        while not (owner_dir / "owner_ready").exists() and time.monotonic() < ready_deadline:
            time.sleep(0.01)
        assert (owner_dir / "owner_ready").exists()
        ledger = json.loads((owner_dir / "process_ledger.json").read_text())
        role_pgid = int(ledger["processes"][0]["pgid"])
        assert live_group_members(role_pgid)
        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=2.0)
        zero_deadline = time.monotonic() + 4.0
        while live_group_members(role_pgid) and time.monotonic() < zero_deadline:
            time.sleep(0.05)
        assert live_group_members(role_pgid) == []
        checks.append("parent_death_term_child_descendant_group_zero")

        # A direct child across the simulated daemon/container boundary does
        # not inherit ProcessRegistry's parent-death binding.  The kernel
        # flock release on owner SIGKILL must make the inner-like supervisor
        # clean its own child/descendant process group.
        liveness_result = root / "flock_owner_death"
        liveness_result.mkdir()
        owner_code = (
            "import subprocess,sys,time; from pathlib import Path; "
            "from sensor_runtime.outer_liveness import OuterAliveLock; "
            f"root=Path({str(liveness_result)!r}); lock=OuterAliveLock(root/'outer_alive.lock'); lock.acquire(); "
            "subprocess.Popen([sys.executable,'-m','sensor_runtime.liveness_fixture_supervisor',"
            "'--lock',str(root/'outer_alive.lock'),'--result-dir',str(root)],"
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True); "
            "time.sleep(30)"
        )
        owner = subprocess.Popen(
            [sys.executable, "-c", owner_code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + 4.0
        while not (liveness_result / "monitor_ready.json").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (liveness_result / "monitor_ready.json").exists()
        inner_ledger = json.loads(
            (liveness_result / "inner_lifecycle/process_ledger.json").read_text()
        )
        inner_pgid = int(inner_ledger["processes"][0]["pgid"])
        assert live_group_members(inner_pgid)
        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=2.0)
        deadline = time.monotonic() + 5.0
        while not (liveness_result / "liveness_cleanup.json").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        liveness_cleanup = json.loads(
            (liveness_result / "liveness_cleanup.json").read_text()
        )
        assert liveness_cleanup["status"] == "PASS"
        assert liveness_cleanup["outer_owner_gone_detected"] is True
        assert live_group_members(inner_pgid) == []
        checks.append("bind_mount_flock_owner_death_inner_group_zero")

        # Asset preparation runs below a managed group leader.  Parent-death
        # TERM to that leader must explicitly reap its same-PGID descendants.
        managed_dir = root / "managed_asset_owner"
        managed_code = (
            "import subprocess,sys,time; from pathlib import Path; "
            "from sensor_runtime.processes import ProcessRegistry; "
            f"root=Path({str(managed_dir)!r}); root.mkdir(); registry=ProcessRegistry(root); "
            "registry.start('asset_builder',[sys.executable,'-m','sensor_runtime.managed_command','--',"
            "sys.executable,'-c',\"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); time.sleep(30)\"],"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,required=False,expected_long_running=False); "
            "(root/'owner_ready').write_text('ready'); time.sleep(30)"
        )
        managed_owner = subprocess.Popen(
            [sys.executable, "-c", managed_code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + 3.0
        while not (managed_dir / "owner_ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        managed_ledger = json.loads((managed_dir / "process_ledger.json").read_text())
        managed_pgid = int(managed_ledger["processes"][0]["pgid"])
        deadline = time.monotonic() + 3.0
        while len(live_group_members(managed_pgid)) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(live_group_members(managed_pgid)) >= 2
        os.kill(managed_owner.pid, signal.SIGKILL)
        managed_owner.wait(timeout=2.0)
        deadline = time.monotonic() + 5.0
        while live_group_members(managed_pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert live_group_members(managed_pgid) == []
        checks.append("managed_asset_parent_death_descendant_group_zero")

    print(json.dumps({"status": "PASS", "checks": checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
