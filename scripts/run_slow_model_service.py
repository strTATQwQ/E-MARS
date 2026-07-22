#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def health_request(endpoint: str, timeout_ms: int) -> dict:
    import zmq

    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(endpoint)
    try:
        socket.send_json({"type": "health"})
        return dict(socket.recv_json())
    finally:
        socket.close(linger=0)


def terminate(pid: int, timeout_s: float = 30.0) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(0.25)
    raise RuntimeError(f"existing service pid={pid} did not exit after SIGTERM")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start one slow model service and record cold-start evidence.")
    parser.add_argument("--python", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:8200")
    parser.add_argument("--replace-pid", type=int)
    parser.add_argument("--ready-timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if args.replace_pid:
        terminate(args.replace_pid)
    config = Path(args.config).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "server.log"
    # Preserve the venv launcher path instead of resolving its symlink to the system interpreter.
    command = [str(Path(args.python).absolute()), "-m", "slow_planner.serve", "--config", str(config)]
    started = time.time()
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd(),
            env=dict(os.environ),
            start_new_session=True,
        )
    health = None
    last_error = ""
    deadline = time.monotonic() + args.ready_timeout_seconds
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            last_error = f"service exited with returncode={returncode}"
            break
        try:
            health = health_request(args.endpoint, 750)
            if health.get("ready"):
                break
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"
        time.sleep(0.25)
    ready_at = time.time() if health and health.get("ready") else None
    report = {
        "schema_version": 1,
        "pid": process.pid,
        "command": command,
        "started_at": started,
        "ready_at": ready_at,
        "cold_start_to_health_seconds": ready_at - started if ready_at is not None else None,
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "health": health,
        "last_poll_error": "" if health else last_error,
        "log_path": str(log_path),
    }
    (output / "cold_start.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "server.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if health and health.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
