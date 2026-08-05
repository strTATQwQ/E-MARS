#!/usr/bin/env python3
"""Drive the frozen T5 completion_sim fault schedule over lane-scoped SSH."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "internvla_ros2"))

from internvla_ros2.fault_injection import (
    FAULT_PROFILE,
    load_fault_plan,
    read_event_records,
)


SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "ServerAliveInterval=5",
    "-o",
    "ServerAliveCountMax=3",
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _validate_remote_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not all(part.replace("-", "").replace("_", "").replace(".", "").isalnum()
                   for part in path.parts[1:])
    ):
        raise ValueError(f"unsafe remote path: {value!r}")
    return str(path)


class Remote:
    def __init__(self, target: str):
        self.target = target

    def run(self, arguments: list[str], *, timeout: float = 20.0) -> str:
        command = " ".join(shlex.quote(item) for item in arguments)
        completed = subprocess.run(
            ["ssh", *SSH_OPTIONS, self.target, command],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"remote command failed on {self.target}: rc={completed.returncode} "
                f"stderr={completed.stderr[-1000:]!r}"
            )
        return completed.stdout

    def read_text(self, path: str, *, missing_ok: bool = False) -> str:
        path = _validate_remote_path(path)
        script = "import pathlib,sys;p=pathlib.Path(sys.argv[1]);" + (
            "print(p.read_text(encoding='utf-8'),end='') if p.is_file() else None"
            if missing_ok
            else "print(p.read_text(encoding='utf-8'),end='')"
        )
        return self.run(["python3", "-c", script, path])

    def read_json(self, path: str) -> dict[str, Any]:
        value = json.loads(self.read_text(path))
        if not isinstance(value, dict):
            raise RuntimeError(f"remote JSON is not an object: {path}")
        return value

    def write_json(self, path: str, value: dict[str, Any]) -> None:
        path = _validate_remote_path(path)
        encoded = base64.b64encode(
            (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
        ).decode("ascii")
        script = (
            "import base64,os,pathlib,sys;"
            "p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);"
            "t=p.with_name('.'+p.name+'.tmp');"
            "t.write_bytes(base64.b64decode(sys.argv[2]));os.replace(t,p)"
        )
        self.run(["python3", "-c", script, path, encoded])


class Director:
    def __init__(self, arguments: argparse.Namespace):
        self.lane = arguments.lane
        self.plan = load_fault_plan(arguments.config.resolve())
        self.output_dir = arguments.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.timeline_path = self.output_dir / "fault_injection_timeline.jsonl"
        self.summary_path = self.output_dir / "fault_injection_summary.json"
        self.dgx_run = _validate_remote_path(arguments.dgx_run)
        self.x86_run = _validate_remote_path(arguments.x86_run)
        self.dgx = Remote(
            "railgun@10.100.100.128" if self.lane == "a" else "rail@10.100.120.122"
        )
        self.x86 = Remote("song@10.100.120.123")
        self.wall_timeout = float(arguments.wall_liveness_timeout_sec)
        if not 300.0 <= self.wall_timeout <= 8400.0:
            raise ValueError(
                "fault actuator wall liveness timeout must be 300-8400 seconds"
            )
        self.revision = 0
        self.anchor_sim_ns = 0
        self.last_sim_ns = 0
        self.results: list[dict[str, Any]] = []

    def _record(self, event: dict[str, Any], phase: str, **extra: Any) -> None:
        row = {
            "schema_version": 1,
            "profile": FAULT_PROFILE,
            "lane": self.lane,
            "event_id": event["event_id"],
            "kind": event["kind"],
            "phase": phase,
            "wall_unix": time.time(),
            **extra,
        }
        with self.timeline_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def _clock_ns(self) -> int:
        value = self.x86.read_json(f"{self.x86_run}/health/clock_live.json")
        stamp = value.get("last_clock_ns")
        if (
            value.get("status") != "RUNNING"
            or isinstance(stamp, bool)
            or not isinstance(stamp, int)
            or stamp <= 0
            or value.get("regression_count") != 0
            or value.get("invalid_count") != 0
        ):
            raise RuntimeError("x86 simulation clock is not a valid running authority")
        if stamp < self.last_sim_ns:
            raise RuntimeError("fault director observed a simulation clock regression")
        self.last_sim_ns = stamp
        return stamp

    def _wait_until_sim(self, target_ns: int) -> int:
        deadline = time.monotonic() + self.wall_timeout
        while time.monotonic() < deadline:
            observed = self._clock_ns()
            if observed >= target_ns:
                return observed
            time.sleep(1.0)
        raise TimeoutError("simulation clock did not reach the next fault boundary")

    def _state(self, event: dict[str, Any] | None, sim_ns: int) -> dict[str, Any]:
        self.revision += 1
        active = [] if event is None else [
            {"event_id": event["event_id"], "kind": event["kind"]}
        ]
        return {
            "schema_version": 1,
            "profile": FAULT_PROFILE,
            "lane": self.lane,
            "revision": self.revision,
            "observed_sim_ns": int(sim_ns),
            "active": active,
        }

    def _write_state(
        self, event: dict[str, Any] | None, sim_ns: int, *, target: str
    ) -> None:
        state = self._state(event, sim_ns)
        if target in {"dgx", "both"}:
            self.dgx.write_json(f"{self.dgx_run}/fault_control/state.json", state)
        if target in {"x86", "both"}:
            self.x86.write_json(f"{self.x86_run}/fault_control/state.json", state)

    @staticmethod
    def _event_seen(
        text: str, event_id: str, *, component: str, phase: str
    ) -> bool:
        try:
            records = read_event_records(text.splitlines())
        except (ValueError, json.JSONDecodeError):
            return False
        return any(
            row.get("event_id") == event_id
            and row.get("component") == component
            and row.get("phase") == phase
            for row in records
        )

    def _wait_event(
        self, remote: Remote, path: str, event: dict[str, Any], component: str, phase: str
    ) -> None:
        deadline = time.monotonic() + self.wall_timeout
        while time.monotonic() < deadline:
            text = remote.read_text(path, missing_ok=True)
            if self._event_seen(
                text, str(event["event_id"]), component=component, phase=phase
            ):
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"fault event acknowledgement timed out: {event['event_id']} {component} {phase}"
        )

    def _request_dgx_restart(self, event: dict[str, Any], sim_ns: int) -> dict[str, Any]:
        request = {
            "schema_version": 1,
            "profile": FAULT_PROFILE,
            "lane": self.lane,
            "event_id": event["event_id"],
            "action": event["kind"],
            "requested_sim_ns": int(sim_ns),
        }
        request_path = f"{self.dgx_run}/fault_control/requests/{event['event_id']}.json"
        ack_path = f"{self.dgx_run}/fault_control/acks/{event['event_id']}.json"
        self.dgx.write_json(request_path, request)
        deadline = time.monotonic() + self.wall_timeout
        while time.monotonic() < deadline:
            try:
                ack = self.dgx.read_json(ack_path)
            except RuntimeError:
                time.sleep(1.0)
                continue
            if ack.get("event_id") != event["event_id"]:
                raise RuntimeError("DGX fault acknowledgement identity mismatch")
            if (
                ack.get("profile") != FAULT_PROFILE
                or ack.get("lane") != self.lane
                or ack.get("action") != event["kind"]
                or ack.get("status") != "PASS"
                or ack.get("safe_stop_confirmed") is not True
                or ack.get("residual_after_stop") != 0
                or ack.get("session_continuity_confirmed") is not True
                or not isinstance(ack.get("session_sha256"), str)
                or len(ack["session_sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in ack["session_sha256"])
                or isinstance(ack.get("new_pid"), bool)
                or not isinstance(ack.get("new_pid"), int)
                or ack["new_pid"] <= 1
            ):
                raise RuntimeError(f"DGX fault action failed: {ack}")
            return ack
        raise TimeoutError(f"DGX restart acknowledgement timed out: {event['event_id']}")

    def _run_event(self, event: dict[str, Any]) -> dict[str, Any]:
        start_target = self.anchor_sim_ns + int(
            float(event["start_sim_offset_sec"]) * 1e9
        )
        observed_start = self._wait_until_sim(start_target)
        self._record(event, "started", observed_sim_ns=observed_start)
        kind = str(event["kind"])
        result: dict[str, Any] = {"event_id": event["event_id"], "kind": kind}
        if kind == "model_request_timeout":
            self._write_state(event, observed_start, target="both")
            self._wait_event(
                self.dgx,
                f"{self.dgx_run}/fault_control/events.jsonl",
                event,
                "dgx_model",
                "injected_timeout",
            )
            self._wait_event(
                self.x86,
                f"{self.x86_run}/fault_control/events.jsonl",
                event,
                "model_agent",
                "expected_timeout_safe_stop",
            )
            self._write_state(None, self._clock_ns(), target="both")
        elif kind in {"model_service_restart", "dgx_ros_node_restart"}:
            self._write_state(event, observed_start, target="both")
            self._wait_event(
                self.x86,
                f"{self.x86_run}/fault_control/events.jsonl",
                event,
                "model_agent",
                "maintenance_wait_started",
            )
            result["ack"] = self._request_dgx_restart(event, observed_start)
            self._write_state(None, self._clock_ns(), target="both")
            self._wait_event(
                self.x86,
                f"{self.x86_run}/fault_control/events.jsonl",
                event,
                "model_agent",
                "maintenance_wait_finished",
            )
        elif kind == "episode_reset":
            self._write_state(event, observed_start, target="x86")
            self._wait_event(
                self.x86,
                f"{self.x86_run}/fault_control/events.jsonl",
                event,
                "model_agent",
                "reset_completed",
            )
            self._write_state(None, self._clock_ns(), target="x86")
        else:
            self._write_state(event, observed_start, target="x86")
            self._wait_event(
                self.x86,
                f"{self.x86_run}/fault_control/events.jsonl",
                event,
                "isaac_controller",
                "observed_active",
            )
            end_target = start_target + int(float(event["duration_sim_sec"]) * 1e9)
            observed_end = self._wait_until_sim(end_target)
            self._write_state(None, observed_end, target="x86")
            self._wait_event(
                self.x86,
                f"{self.x86_run}/fault_control/events.jsonl",
                event,
                "isaac_controller",
                "observed_recovered",
            )
        recovered_sim_ns = self._clock_ns()
        result.update(status="PASS", recovered_sim_ns=recovered_sim_ns)
        self._record(event, "recovered", observed_sim_ns=recovered_sim_ns)
        return result

    def run(self) -> dict[str, Any]:
        status = "FAIL"
        fatal_error: str | None = None
        controls_cleared = False
        try:
            self.anchor_sim_ns = self._clock_ns()
            self._write_state(None, self.anchor_sim_ns, target="both")
            for event in self.plan["events"]:
                result = self._run_event(event)
                self.results.append(result)
            status = "PASS"
        except BaseException as exc:
            fatal_error = repr(exc)[:2048]
        finally:
            controls_cleared = True
            for remote, run_root in ((self.dgx, self.dgx_run), (self.x86, self.x86_run)):
                try:
                    self.revision += 1
                    state_path = f"{run_root}/fault_control/state.json"
                    cleared = {
                        "schema_version": 1,
                        "profile": FAULT_PROFILE,
                        "lane": self.lane,
                        "revision": self.revision,
                        "observed_sim_ns": max(0, self.last_sim_ns),
                        "active": [],
                    }
                    remote.write_json(state_path, cleared)
                    if remote.read_json(state_path) != cleared:
                        raise RuntimeError("fault control clear readback mismatch")
                except BaseException:
                    controls_cleared = False
                    status = "FAIL"
        checks = {
            "all_six_events_recovered": status == "PASS"
            and len(self.results) == 6
            and all(item.get("status") == "PASS" for item in self.results),
            "schedule_uses_sim_time": self.plan.get("schedule_timebase") == "sim_time",
            "wall_only_liveness": self.plan.get("wall_time_scope")
            == "actuator_and_process_liveness_only",
            "control_cleared_after_run": controls_cleared,
        }
        summary = {
            "schema_version": 1,
            "profile": FAULT_PROFILE,
            "status": "PASS" if all(checks.values()) else "FAIL",
            "lane": self.lane,
            "anchor_sim_ns": self.anchor_sim_ns,
            "event_results": self.results,
            "timeline_path": self.timeline_path.name,
            "fatal_error": fatal_error,
            "checks": checks,
            "finished_unix": time.time(),
        }
        _atomic_json(self.summary_path, summary)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--config", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--lane", choices=("a", "b"), required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--dgx-run", required=True)
    run.add_argument("--x86-run", required=True)
    run.add_argument("--wall-liveness-timeout-sec", type=float, default=7200.0)
    arguments = parser.parse_args()
    if arguments.command == "validate":
        value = load_fault_plan(arguments.config.resolve())
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    summary = Director(arguments).run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "PASS" else 75)


if __name__ == "__main__":
    main()
