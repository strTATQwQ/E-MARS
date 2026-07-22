import json
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_lease_wrapper_exposes_bounded_fail_closed_cleanup_contract():
    text = (ROOT / "scripts" / "with_resource_lease.sh").read_text(encoding="utf-8")
    assert "--cleanup-timeout" in text
    assert "--kill-wait-timeout" in text
    assert "drain_wrapped_group" in text
    assert "wrapped_process_group_absent_before_lock_release" in text
    assert "wrapped_group_is_alive" in text
    assert "command_left_descendants" in text
    assert "lease_cleanup_receipt.json" in text
    assert "/tmp/internnav_dgx.quarantine" in text
    assert "/tmp/internnav_isaac.quarantine" in text
    assert "/tmp/internnav_isaac_gpu0.quarantine" in text
    assert "/tmp/internnav_isaac_gpu1.quarantine" in text
    assert "QUARANTINED resource=%s" in text
    assert '[[ -e "$quarantine_file" || -L "$quarantine_file" ]]' in text
    assert text.index('for quarantine_file in "${quarantine_candidates[@]}"') < text.index(
        "write_metadata HELD"
    )
    assert text.index('drain_wrapped_group "lease_lost:$lease_lost"') < text.index(
        "state=LEASE_LOST"
    )


@pytest.mark.skipif(shutil.which("wsl") is None, reason="WSL is required for flock fixture")
def test_persistent_remote_quarantine_denies_a_new_heavy_lease():
    fixture = ROOT / ".codex-tmp" / "lease-quarantine-fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    fake_ssh = fixture / "fake-ssh.sh"
    workload = fixture / "workload.sh"
    workload_marker = fixture / "workload-started"
    log_dir = fixture / "logs"

    def wsl(path: Path) -> str:
        return subprocess.run(
            ["wsl", "wslpath", "-a", str(path)], text=True,
            capture_output=True, check=True,
        ).stdout.strip()

    quarantine = "/tmp/internnav_dgx.quarantine"
    exists = subprocess.run(["wsl", "test", "-e", quarantine], capture_output=True)
    if exists.returncode == 0:
        pytest.skip("preserving an existing operator quarantine marker")
    if workload_marker.exists():
        workload_marker.unlink()
    if log_dir.exists():
        shutil.rmtree(log_dir)
    fake_ssh.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nexec bash -c "${!#}"\n',
        encoding="utf-8", newline="\n",
    )
    workload.write_text(
        f'#!/usr/bin/env bash\nprintf started >{wsl(workload_marker)!r}\n',
        encoding="utf-8", newline="\n",
    )
    fake_ssh.chmod(0o755)
    workload.chmod(0o755)
    subprocess.run(
        ["wsl", "bash", "-c", f"umask 077; printf 'state=DIRTY\\n' >{quarantine}"],
        check=True,
    )
    try:
        completed = subprocess.run(
            ["wsl", "env", f"LEASE_SSH_BIN={wsl(fake_ssh)}", "bash",
             wsl(ROOT / "scripts" / "with_resource_lease.sh"), "dgx-a",
             "--task", "fixture-quarantined", "--log-dir", wsl(log_dir),
             "--acquire-timeout", "3", "--", wsl(workload)],
            text=True, capture_output=True, timeout=10,
        )
    finally:
        subprocess.run(["wsl", "rm", "-f", quarantine], check=True)
    assert completed.returncode == 73, completed.stderr
    assert "QUARANTINED resource=dgx_a" in completed.stderr
    assert not workload_marker.exists()


@pytest.mark.skipif(shutil.which("wsl") is None, reason="WSL is required for flock fixture")
def test_gpu0_quarantine_does_not_block_independent_gpu1_lane():
    fixture = ROOT / ".codex-tmp" / "lease-per-gpu-quarantine-fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    fake_ssh = fixture / "fake-ssh.sh"
    workload = fixture / "workload.sh"
    lane_a_marker = fixture / "lane-a-started"
    lane_b_marker = fixture / "lane-b-started"

    def wsl(path: Path) -> str:
        return subprocess.run(
            ["wsl", "wslpath", "-a", str(path)],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    quarantine_paths = (
        "/tmp/internnav_isaac.quarantine",
        "/tmp/internnav_isaac_gpu0.quarantine",
        "/tmp/internnav_isaac_gpu1.quarantine",
        "/tmp/internnav_dgx.quarantine",
    )
    for path in quarantine_paths:
        if subprocess.run(["wsl", "test", "-e", path]).returncode == 0:
            pytest.skip(f"preserving existing operator marker {path}")
    for path in (lane_a_marker, lane_b_marker):
        path.unlink(missing_ok=True)
    fake_ssh.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nexec bash -c "${!#}"\n',
        encoding="utf-8",
        newline="\n",
    )
    workload.write_text(
        '#!/usr/bin/env bash\nprintf started >"$1"\n',
        encoding="utf-8",
        newline="\n",
    )
    fake_ssh.chmod(0o755)
    workload.chmod(0o755)
    gpu0 = "/tmp/internnav_isaac_gpu0.quarantine"
    subprocess.run(
        ["wsl", "bash", "-c", f"umask 077; printf 'state=DIRTY\\n' >{gpu0}"],
        check=True,
    )
    try:
        base = [
            "wsl", "env", f"LEASE_SSH_BIN={wsl(fake_ssh)}", "bash",
            wsl(ROOT / "scripts" / "with_resource_lease.sh"),
        ]
        lane_a = subprocess.run(
            base + ["lane-a", "--task", "gpu0-blocked", "--log-dir",
                    wsl(fixture / "lane-a-logs"), "--acquire-timeout", "3", "--",
                    wsl(workload), wsl(lane_a_marker)],
            text=True, capture_output=True, timeout=15,
        )
        lane_b = subprocess.run(
            base + ["lane-b", "--task", "gpu1-independent", "--log-dir",
                    wsl(fixture / "lane-b-logs"), "--acquire-timeout", "3", "--",
                    wsl(workload), wsl(lane_b_marker)],
            text=True, capture_output=True, timeout=15,
        )
    finally:
        subprocess.run(["wsl", "rm", "-f", gpu0], check=True)
    assert lane_a.returncode == 73, lane_a.stderr
    assert "internnav_isaac_gpu0.quarantine" in lane_a.stderr
    assert not lane_a_marker.exists()
    assert lane_b.returncode == 0, lane_b.stderr
    assert lane_b_marker.read_text(encoding="utf-8") == "started"


@pytest.mark.skipif(shutil.which("wsl") is None, reason="WSL is required for shell fixture")
def test_recovery_mode_rejects_an_arbitrary_command_before_ssh():
    fixture = ROOT / ".codex-tmp" / "lease-recovery-deny-fixture"
    completed = subprocess.run(
        [
            "wsl", "bash", subprocess.run(
                ["wsl", "wslpath", "-a", str(ROOT / "scripts" / "with_resource_lease.sh")],
                text=True, capture_output=True, check=True,
            ).stdout.strip(),
            "recover-isaac-gpu0", "--task", "must-not-run", "--log-dir",
            subprocess.run(
                ["wsl", "wslpath", "-a", str(fixture)],
                text=True, capture_output=True, check=True,
            ).stdout.strip(),
            "--", "bash", "-c", "printf unsafe",
        ],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 64
    assert "recovery-only mode requires the fixed recovery entry" in completed.stderr


@pytest.mark.skipif(shutil.which("wsl") is None, reason="WSL is required for flock fixture")
def test_holder_loss_allows_graceful_cleanup_before_remaining_lock_release():
    fixture = ROOT / ".codex-tmp" / "lease-loss-fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    fake_ssh = fixture / "fake-ssh.sh"
    workload = fixture / "workload.py"
    marker = fixture / "cleanup-complete"
    ready_marker = fixture / "workload-ready"
    release_marker = fixture / "remaining-lock-released-after-cleanup"
    log_dir = fixture / "logs"

    def wsl(path: Path) -> str:
        value = subprocess.run(
            ["wsl", "wslpath", "-a", str(path)],
            text=True,
            capture_output=True,
            check=True,
        )
        return value.stdout.strip()

    if marker.exists():
        marker.unlink()
    if ready_marker.exists():
        ready_marker.unlink()
    if release_marker.exists():
        release_marker.unlink()
    if log_dir.exists():
        shutil.rmtree(log_dir)
    fake_ssh.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            remote_command="${!#}"
            if [[ "$remote_command" == *"'isaac_gpu0'"* ]]; then
              bash -c "$remote_command" &
              child=$!
              for _ in $(seq 1 100); do
                test -f "$FIXTURE_READY_MARKER" && break
                sleep 0.1
              done
              test -f "$FIXTURE_READY_MARKER"
              kill -TERM "$child" 2>/dev/null || true
              wait "$child" 2>/dev/null || true
              exit 75
            fi
            bash -c "$remote_command"
            rc=$?
            test -f "$FIXTURE_CLEANUP_MARKER"
            printf verified >"$FIXTURE_RELEASE_MARKER"
            exit "$rc"
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    workload.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import signal
            import time
            from pathlib import Path

            def cleanup(signum, frame):
                time.sleep(2)
                Path({wsl(marker)!r}).write_text("complete", encoding="utf-8")
                raise SystemExit(143)

            signal.signal(signal.SIGTERM, cleanup)
            signal.signal(signal.SIGHUP, cleanup)
            Path({wsl(ready_marker)!r}).write_text("ready", encoding="utf-8")
            while True:
                time.sleep(0.2)
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    fake_ssh.chmod(0o755)
    workload.chmod(0o755)

    command = [
        "wsl", "env", f"LEASE_SSH_BIN={wsl(fake_ssh)}",
        f"FIXTURE_CLEANUP_MARKER={wsl(marker)}",
        f"FIXTURE_READY_MARKER={wsl(ready_marker)}",
        f"FIXTURE_RELEASE_MARKER={wsl(release_marker)}",
        "bash", wsl(ROOT / "scripts" / "with_resource_lease.sh"),
        "lane-a", "--task", "fixture-holder-loss", "--log-dir", wsl(log_dir),
        "--acquire-timeout", "5", "--cleanup-timeout", "5",
        "--kill-wait-timeout", "2", "--", wsl(workload),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=20)
    assert completed.returncode == 75, completed.stderr
    assert marker.read_text(encoding="utf-8") == "complete"
    assert release_marker.read_text(encoding="utf-8") == "verified"
    receipt = json.loads((log_dir / "lease_cleanup_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["graceful_cleanup_completed"] is True
    assert receipt["kill_escalated"] is False
    assert receipt["wrapped_process_group_absent_before_lock_release"] is True
    assert receipt["resources_still_held_before_release"] == ["dgx_a"]
    metadata = (log_dir / "lease_metadata.txt").read_text(encoding="utf-8")
    assert "state=LEASE_LOST" in metadata
    assert (log_dir / "holder_dgx_a.stdout.log").is_file()


@pytest.mark.skipif(shutil.which("wsl") is None, reason="WSL is required for setsid fixture")
def test_successful_leader_cannot_release_lock_with_live_group_descendant():
    fixture = ROOT / ".codex-tmp" / "lease-descendant-fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    fake_ssh = fixture / "fake-ssh.sh"
    workload = fixture / "workload.py"
    child_pid_path = fixture / "child.pid"
    log_dir = fixture / "logs"
    if child_pid_path.exists():
        child_pid_path.unlink()
    if log_dir.exists():
        shutil.rmtree(log_dir)

    def wsl(path: Path) -> str:
        return subprocess.run(
            ["wsl", "wslpath", "-a", str(path)], text=True,
            capture_output=True, check=True,
        ).stdout.strip()

    fake_ssh.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nexec bash -c "${!#}"\n',
        encoding="utf-8", newline="\n",
    )
    workload.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import os
            import signal
            import time
            from pathlib import Path

            def raise_exit():
                raise SystemExit(143)

            child = os.fork()
            if child == 0:
                signal.signal(signal.SIGTERM, lambda *_: raise_exit())
                while True:
                    time.sleep(0.2)
            Path({wsl(child_pid_path)!r}).write_text(str(child), encoding="utf-8")
            os._exit(0)
            """
        ),
        encoding="utf-8", newline="\n",
    )
    fake_ssh.chmod(0o755)
    workload.chmod(0o755)
    completed = subprocess.run(
        ["wsl", "env", f"LEASE_SSH_BIN={wsl(fake_ssh)}", "bash",
         wsl(ROOT / "scripts" / "with_resource_lease.sh"), "dgx-a",
         "--task", "fixture-descendant", "--log-dir", wsl(log_dir),
         "--acquire-timeout", "5", "--cleanup-timeout", "3",
         "--kill-wait-timeout", "2", "--", wsl(workload)],
        text=True, capture_output=True, timeout=20,
    )
    assert completed.returncode == 75, completed.stderr
    child_pid = child_pid_path.read_text(encoding="utf-8")
    absent = subprocess.run(["wsl", "kill", "-0", child_pid], capture_output=True)
    assert absent.returncode != 0
    receipt = json.loads((log_dir / "lease_cleanup_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["reason"] == "command_left_descendants"
