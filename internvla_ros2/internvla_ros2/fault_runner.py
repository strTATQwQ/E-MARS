"""Deterministic cross-host protocol and fail-safe fault gate."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from internvla_ros2_msgs.action import Step
from internvla_ros2_msgs.srv import Reset

from .client_node import ClientFailure, ClientRuntime, _assign_time, _future_result
from .protocol import (
    STATUS_CANCELED,
    STATUS_INTERNAL_ERROR,
    STATUS_INVALID_REQUEST,
    STATUS_OBSERVATION_MISSING,
    STATUS_OK,
    STATUS_RESET_MISMATCH,
    STATUS_STALE,
    STATUS_TIMEOUT,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640, 1), dtype=np.float32),
    )


def _step_kwargs(rgb: np.ndarray, depth: np.ndarray) -> dict[str, Any]:
    return {
        "rgb": rgb,
        "depth": depth,
        "instruction": "walk forward",
        "instruction_tokens": [1, 2],
        "global_gps": [0.0, 0.0, 0.0],
        "global_rotation": [0.0, 0.0, 0.0, 1.0],
    }


def _raw_goal(
    runtime: ClientRuntime,
    *,
    episode_id: str,
    generation: int,
    sequence: int,
    request_id: str,
    deadline_offset_sec: float,
) -> Step.Result:
    node = runtime.node
    now_ns = node.get_clock().now().nanoseconds
    stamp = node.get_clock().now().to_msg()
    goal = Step.Goal()
    goal.header.stamp = stamp
    goal.header.frame_id = "internvla_camera"
    goal.protocol_version = 1
    goal.episode_id = episode_id
    goal.reset_generation = generation
    goal.sequence_id = sequence
    goal.request_id = request_id
    goal.observation_digest = "0" * 64
    goal.client_wall_monotonic_ns = time.monotonic_ns()
    goal.sim_stamp = stamp
    goal.observation_stamp = stamp
    _assign_time(goal.deadline, now_ns + int(deadline_offset_sec * 1e9))
    _assign_time(goal.valid_until, now_ns + int(max(deadline_offset_sec, 0.0) * 1e9) + 1_000_000_000)
    handle = _future_result(node.step_client.send_goal_async(goal), 5.0, "raw fault goal")
    if not handle.accepted:
        raise RuntimeError("raw fault goal was rejected before typed status response")
    return _future_result(handle.get_result_async(), 8.0, "raw fault result").result


def _basic(runtime: ClientRuntime) -> dict[str, Any]:
    node = runtime.node
    rgb, depth = _arrays()
    node.initialize("fault-basic-0")
    common = _step_kwargs(rgb, depth)
    first = node.step_arrays(**common, sequence_id=0, request_id="basic:0:0")
    duplicate = node.step_arrays(
        **common,
        sequence_id=0,
        request_id="basic:0:0",
        commit_sequence=False,
    )

    collision_status = None
    changed = rgb.copy()
    changed[0, 0, 0] = 1
    try:
        node.step_arrays(
            **_step_kwargs(changed, depth),
            sequence_id=0,
            request_id="basic:0:0",
            commit_sequence=False,
        )
    except ClientFailure as exc:
        collision_status = exc.status_code
    safe_after_collision = node.safe_stop

    second = node.step_arrays(**common, sequence_id=1, request_id="basic:0:1")
    missing = _raw_goal(
        runtime,
        episode_id=node.episode_id,
        generation=node.reset_generation,
        sequence=2,
        request_id="basic:missing",
        deadline_offset_sec=4.0,
    )
    expired = _raw_goal(
        runtime,
        episode_id=node.episode_id,
        generation=node.reset_generation,
        sequence=2,
        request_id="basic:expired",
        deadline_offset_sec=-0.1,
    )
    old_episode = node.episode_id
    old_generation = node.reset_generation
    reset = node.reset("fault-basic-1")
    stale = _raw_goal(
        runtime,
        episode_id=old_episode,
        generation=old_generation,
        sequence=2,
        request_id="basic:stale",
        deadline_offset_sec=2.0,
    )
    after_reset = node.step_arrays(**common, sequence_id=0, request_id="basic:1:0")

    bad_reset = Reset.Request()
    bad_reset.protocol_version = 1
    bad_reset.next_episode_id = "must-not-apply"
    bad_reset.expected_reset_generation = 0
    bad_reset.reset_barrier_sequence_id = 0
    bad_reset_response = _future_result(
        node.reset_client.call_async(bad_reset), 5.0, "mismatched reset"
    )
    health = node.health()
    checks = {
        "first_ok": first["status_code"] == STATUS_OK and not first["replayed"],
        "duplicate_idempotent": duplicate["replayed"]
        and duplicate["discrete_action"] == first["discrete_action"]
        and duplicate["local_path"] == first["local_path"],
        "collision_rejected": collision_status == STATUS_INVALID_REQUEST,
        "collision_safe_stop": safe_after_collision,
        "sequence_recovers": second["sequence_id"] == 1,
        "missing_observation_typed": int(missing.status_code) == STATUS_OBSERVATION_MISSING,
        "expired_deadline_typed": int(expired.status_code) == STATUS_TIMEOUT,
        "reset_advanced_once": int(reset["reset_generation"]) == 1,
        "stale_generation_rejected": int(stale.status_code)
        in (STATUS_STALE, STATUS_RESET_MISMATCH),
        "post_reset_sequence_zero": after_reset["sequence_id"] == 0
        and after_reset["reset_generation"] == 1,
        "mismatched_reset_rejected": int(bad_reset_response.status_code)
        == STATUS_RESET_MISMATCH,
        "mismatched_reset_not_applied": health["reset_generation"] == 1
        and health["episode_id"] == "fault-basic-1",
    }
    return {
        "checks": checks,
        "observed_status": {
            "collision": collision_status,
            "missing_observation": int(missing.status_code),
            "expired_deadline": int(expired.status_code),
            "stale_generation": int(stale.status_code),
            "mismatched_reset": int(bad_reset_response.status_code),
        },
        "health_after": health,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def _timeout(runtime: ClientRuntime) -> dict[str, Any]:
    node = runtime.node
    rgb, depth = _arrays()
    node.initialize("fault-timeout-0")
    caught_status = None
    try:
        node.step_arrays(
            **_step_kwargs(rgb, depth),
            deadline_sec=0.5,
            validity_sec=1.0,
        )
    except ClientFailure as exc:
        caught_status = exc.status_code
    time.sleep(3.0)
    health = node.health()
    checks = {
        "client_rejected_late_result": caught_status in (STATUS_TIMEOUT, STATUS_CANCELED),
        "client_safe_stop": node.safe_stop,
        "generation_invalidated": health["reset_generation"] == 1,
        "model_not_committed": health["last_sequence_id"] == 0,
        "model_reports_safe_status": health["status_code"]
        in (STATUS_STALE, STATUS_TIMEOUT, STATUS_CANCELED),
    }
    return {
        "checks": checks,
        "caught_status": caught_status,
        "health_after": health,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def _loss(runtime: ClientRuntime, ready_file: Path, wait_sec: float) -> dict[str, Any]:
    node = runtime.node
    rgb, depth = _arrays()
    node.initialize("fault-loss-0")
    node.step_arrays(**_step_kwargs(rgb, depth))
    _atomic_json(ready_file, {"schema_version": 1, "ready_for_model_termination": True})
    time.sleep(wait_sec)
    caught_status = None
    try:
        node.step_arrays(**_step_kwargs(rgb, depth), deadline_sec=2.0, validity_sec=3.0)
    except ClientFailure as exc:
        caught_status = exc.status_code
    checks = {
        "model_loss_detected": caught_status in (STATUS_TIMEOUT, STATUS_INTERNAL_ERROR),
        "client_safe_stop": node.safe_stop,
        "sequence_not_committed": node.last_committed_sequence == 0,
    }
    return {
        "checks": checks,
        "caught_status": caught_status,
        "client_status_message": node.last_status_message,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("basic", "timeout", "loss"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--loss-wait-sec", type=float, default=8.0)
    arguments, ros_arguments = parser.parse_known_args()
    rclpy.init(args=ros_arguments)
    runtime = ClientRuntime()
    report: dict[str, Any] = {"schema_version": 1, "mode": arguments.mode, "status": "FAIL"}
    try:
        runtime.start()
        if arguments.mode == "basic":
            report.update(_basic(runtime))
        elif arguments.mode == "timeout":
            report.update(_timeout(runtime))
        else:
            if arguments.ready_file is None:
                raise ValueError("--ready-file is required for loss mode")
            report.update(_loss(runtime, arguments.ready_file.resolve(), arguments.loss_wait_sec))
    except BaseException as exc:
        report["fatal_error"] = repr(exc)[:2048]
    finally:
        runtime.stop()
    _atomic_json(arguments.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
