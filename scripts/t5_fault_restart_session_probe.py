#!/usr/bin/env python3
"""Capture or verify the quiesced model identity used by T5 fault restarts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import rclpy
from internvla_ros2.fault_injection import (
    FAULT_RESTART_ACTIONS,
    build_fault_restart_session,
    load_fault_restart_session,
)
from internvla_ros2_msgs.action import Step
from internvla_ros2_msgs.srv import Health
from rclpy.action import ActionClient


def _health_snapshot(timeout_sec: float) -> dict[str, Any]:
    rclpy.init()
    node = rclpy.create_node(f"internvla_t5_fault_session_probe_{os.getpid()}")
    health = node.create_client(Health, "/internvla/health")
    step = ActionClient(node, Step, "/internvla/step")
    started = time.monotonic()
    try:
        deadline = started + timeout_sec
        while time.monotonic() < deadline:
            if health.wait_for_service(timeout_sec=0.1) and step.wait_for_server(
                timeout_sec=0.1
            ):
                break
        else:
            raise TimeoutError("typed model health/action graph was not ready")
        request = Health.Request()
        request.protocol_version = 1
        future = health.call_async(request)
        rclpy.spin_until_future_complete(
            node, future, timeout_sec=max(0.1, deadline - time.monotonic())
        )
        response = future.result()
        if response is None:
            raise TimeoutError("typed model health response timed out")
        return {
            "status_code": int(response.status_code),
            "status_message": str(response.status_message),
            "initialized": bool(response.initialized),
            "lifecycle_state": int(response.lifecycle_state),
            "episode_id": str(response.episode_id),
            "reset_generation": int(response.reset_generation),
            "last_sequence_id": int(response.last_sequence_id),
            "model_revision": str(response.model_revision),
            "checkpoint_revision": str(response.checkpoint_revision),
            "step_action_ready": True,
            "probe_duration_sec": time.monotonic() - started,
        }
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _write_fresh(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _client_identity(path: Path, health: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = value.get("fault_restart_session") if isinstance(value, dict) else None
    if not isinstance(identity, dict):
        raise RuntimeError("client summary lacks the fault restart session")
    episode_id = identity.get("episode_id")
    reset_generation = identity.get("reset_generation")
    next_sequence_id = identity.get("next_sequence_id")
    last_sequence_id = identity.get("last_committed_sequence")
    if (
        value.get("status") != "RUNNING"
        or not isinstance(episode_id, str)
        or not episode_id
        or isinstance(reset_generation, bool)
        or not isinstance(reset_generation, int)
        or reset_generation < 0
        or isinstance(next_sequence_id, bool)
        or not isinstance(next_sequence_id, int)
        or next_sequence_id < 0
        or isinstance(last_sequence_id, bool)
        or not isinstance(last_sequence_id, int)
        or last_sequence_id < -1
        or next_sequence_id != last_sequence_id + 1
        or health.get("episode_id") != episode_id
        or health.get("reset_generation") != reset_generation
        or health.get("last_sequence_id") != max(0, last_sequence_id)
    ):
        raise RuntimeError("client/model fault restart identity mismatch")
    return {
        "episode_id": episode_id,
        "reset_generation": reset_generation,
        "next_sequence_id": next_sequence_id,
        "last_sequence_id": last_sequence_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--output", required=True, type=Path)
    capture.add_argument("--lane", required=True, choices=("a", "b"))
    capture.add_argument("--event-id", required=True)
    capture.add_argument(
        "--action", required=True, choices=tuple(sorted(FAULT_RESTART_ACTIONS))
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--session", required=True, type=Path)
    verify.add_argument("--output", required=True, type=Path)
    verify.add_argument("--lane", required=True, choices=("a", "b"))
    for command in (capture, verify):
        command.add_argument("--client-summary", required=True, type=Path)
        command.add_argument("--timeout-sec", type=float, default=5.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be fresh")
    if not 0.1 <= args.timeout_sec <= 30.0:
        parser.error("timeout must be in [0.1, 30] seconds")

    health = _health_snapshot(args.timeout_sec)
    client_identity = _client_identity(args.client_summary, health)
    if args.command == "capture":
        health["last_sequence_id"] = client_identity["last_sequence_id"]
        payload = build_fault_restart_session(
            health,
            lane=args.lane,
            event_id=args.event_id,
            action=args.action,
        )
    else:
        session = load_fault_restart_session(
            args.session, expected_lane=args.lane
        )
        identity_fields = (
            "status_code",
            "initialized",
            "lifecycle_state",
            "episode_id",
            "reset_generation",
            "model_revision",
            "checkpoint_revision",
        )
        mismatches = {
            field: {"expected": session[field], "observed": health.get(field)}
            for field in identity_fields
            if health.get(field) != session[field]
        }
        if health.get("last_sequence_id") != max(0, session["last_sequence_id"]):
            mismatches["last_sequence_id"] = {
                "expected": max(0, session["last_sequence_id"]),
                "observed": health.get("last_sequence_id"),
            }
        for field in ("episode_id", "reset_generation", "last_sequence_id"):
            if client_identity[field] != session[field]:
                mismatches[f"client_{field}"] = {
                    "expected": session[field],
                    "observed": client_identity[field],
                }
        if mismatches:
            raise RuntimeError(f"fault restart session continuity mismatch: {mismatches}")
        payload = {
            "schema_version": 1,
            "profile": session["profile"],
            "status": "PASS",
            "lane": args.lane,
            "event_id": session["event_id"],
            "action": session["action"],
            "session_sha256": hashlib.sha256(args.session.read_bytes()).hexdigest(),
            "session_continuity_confirmed": True,
            "health": health,
            "recorded_unix": time.time(),
        }
    _write_fresh(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
