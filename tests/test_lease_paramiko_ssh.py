#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "coordination" / "lease_paramiko_ssh.py"
REMOTE_COMMAND = (
    "bash -c holder lease-holder '/tmp/internnav_isaac.lock' 'isaac' payload"
)


FAKE_HELPER = r'''
import os

class Channel:
    def __init__(self):
        self.output = bytearray(b"ACQUIRED resource=isaac host=fake pid=42\n")
        self.control = bytearray()
        self.done = False
    def exec_command(self, command):
        with open(os.environ["LEASE_FAKE_COMMAND"], "w", encoding="utf-8") as stream:
            stream.write(command)
    def sendall(self, data):
        self.control.extend(data)
        if b"RELEASE\n" in self.control and not self.done:
            self.output.extend(b"RELEASED resource=isaac\n")
            self.done = True
    def shutdown_write(self):
        pass
    def recv_ready(self):
        return bool(self.output)
    def recv(self, size):
        data = bytes(self.output[:size])
        del self.output[:size]
        return data
    def recv_stderr_ready(self):
        return False
    def recv_stderr(self, size):
        return b""
    def exit_status_ready(self):
        return self.done
    def recv_exit_status(self):
        return int(os.environ.get("LEASE_FAKE_RC", "0"))
    def close(self):
        pass

class Transport:
    def __init__(self):
        self.channel = Channel()
    def is_active(self):
        return True
    def getpeername(self):
        return (os.environ.get("LEASE_FAKE_PEER", "10.100.120.111"), 22)
    def set_keepalive(self, interval):
        with open(os.environ["LEASE_FAKE_KEEPALIVE"], "w", encoding="utf-8") as stream:
            stream.write(str(interval))
    def open_session(self, timeout=None):
        return self.channel

class Client:
    def __init__(self):
        self.transport = Transport()
    def get_transport(self):
        return self.transport
    def close(self):
        pass

def connect():
    return Client()
'''


class LeaseParamikoSshTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="internnav-lease-bridge-")
        self.temp = Path(self.temp_dir.name)
        self.helper = self.temp / "external_connect.py"
        self.helper.write_text(textwrap.dedent(FAKE_HELPER), encoding="utf-8")
        self.command_record = self.temp / "command.txt"
        self.keepalive_record = self.temp / "keepalive.txt"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def invoke(
        self,
        *,
        target: str = "song@10.100.120.111",
        peer: str | None = None,
        rc: int = 0,
        keep_stdin_open: bool = False,
    ) -> subprocess.CompletedProcess[bytes]:
        environment = os.environ.copy()
        environment["LEASE_FAKE_COMMAND"] = str(self.command_record)
        environment["LEASE_FAKE_KEEPALIVE"] = str(self.keepalive_record)
        environment["LEASE_FAKE_RC"] = str(rc)
        if peer is not None:
            environment["LEASE_FAKE_PEER"] = peer
        argv = [
            sys.executable,
            str(ADAPTER),
            "--connect-helper",
            str(self.helper),
            "-T",
            "-p",
            "22",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            "-o",
            "ExitOnForwardFailure=yes",
            target,
            REMOTE_COMMAND,
        ]
        if not keep_stdin_open:
            return subprocess.run(
                argv,
                input=b"RELEASE\n",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                timeout=5,
                check=False,
            )
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        try:
            process.stdin.write(b"RELEASE\n")
            process.stdin.flush()
            returncode = process.wait(timeout=5)
            process.stdin.close()
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            process.stdout.close()
            process.stderr.close()
            return subprocess.CompletedProcess(
                argv,
                returncode,
                stdout,
                stderr,
            )
        except BaseException:
            process.kill()
            process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()
            raise

    def test_bridges_control_and_output(self) -> None:
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b"ACQUIRED resource=isaac", result.stdout)
        self.assertIn(b"RELEASED resource=isaac", result.stdout)
        self.assertEqual(self.command_record.read_text(encoding="utf-8"), REMOTE_COMMAND)
        self.assertEqual(self.keepalive_record.read_text(encoding="utf-8"), "5")

    def test_rejects_wrong_target_before_connect(self) -> None:
        result = self.invoke(target="song@10.100.120.112")
        self.assertEqual(result.returncode, 64)
        self.assertFalse(self.command_record.exists())

    def test_rejects_wrong_connected_peer(self) -> None:
        result = self.invoke(peer="10.100.120.112")
        self.assertEqual(result.returncode, 69)
        self.assertFalse(self.command_record.exists())

    def test_propagates_remote_exit_status(self) -> None:
        result = self.invoke(rc=23)
        self.assertEqual(result.returncode, 23)

    def test_release_does_not_require_control_eof(self) -> None:
        result = self.invoke(keep_stdin_open=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b"RELEASED resource=isaac", result.stdout)


if __name__ == "__main__":
    unittest.main()
