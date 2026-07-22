"""Deterministic offline exercise for both recovery parameter candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import RecoveryConfig, load_recovery_config
from .state_machine import Observation, RecoveryAction, RecoveryMachine, RecoveryState


OLD_TRAJECTORY = ((0.0, 0.0), (1.0, 0.0))
FRESH_TRAJECTORY = ((0.0, 0.0), (0.0, 0.5), (0.0, 1.0))


def _ack(machine: RecoveryMachine, now: float) -> tuple[float, list[RecoveryAction]]:
    command = machine.pending_command
    if command is None:
        raise AssertionError("expected a pending recovery command")
    details: dict[str, object] = {}
    if command.action == RecoveryAction.SCAN:
        details["measured_max_linear_mps"] = 0.0
        details["measured_max_angular_rps"] = (
            machine.config.scan_angular_speed_rps
        )
    elif command.action == RecoveryAction.SAFE_RETREAT:
        details["measured_max_linear_mps"] = (
            machine.config.retreat_linear_speed_mps
        )
        details["measured_max_angular_rps"] = 0.0
    if command.action in {RecoveryAction.SCAN, RecoveryAction.SAFE_RETREAT}:
        details.update(
            {
                "safety_observation_monotonic_sec": now,
                "simulation_estop_preemption_count": 0,
                "safety_gate_violation_count": 0,
                "velocity_chain": "cmd_vel_safe",
            }
        )
    emitted = machine.acknowledge(
        operation_id=command.operation_id,
        episode_id=machine.episode_id,
        reset_generation=machine.reset_generation,
        recovery_id=machine.recovery_id,
        success=True,
        monotonic_sec=now,
        details=details,
    )
    return now + 0.05, [item.action for item in emitted]


def run_profile_smoke(config: RecoveryConfig) -> dict[str, Any]:
    machine = RecoveryMachine(config)
    machine.begin_episode(f"offline-{config.profile_id}", 1, 0.0)
    now = 0.0
    first_command = None
    for index in range(100):
        emitted = machine.observe(
            Observation(
                episode_id=machine.episode_id,
                reset_generation=1,
                goal_id="goal-1",
                monotonic_sec=now,
                x_m=0.0,
                y_m=0.0,
                goal_distance_m=5.0,
                goal_active=True,
                motion_requested=True,
                safety_clear=True,
                sim_estop_ready=True,
                sim_estop_active=False,
                rear_clearance_m=0.50,
                plan_points=OLD_TRAJECTORY if index == 0 else None,
                plan_update=index == 0,
            )
        )
        if emitted:
            first_command = emitted[0]
            break
        now += 0.25
    if first_command is None or first_command.action != RecoveryAction.CANCEL_GOAL:
        raise AssertionError("no-progress did not issue cancel_goal")
    now += 0.05

    expected_prefix = [
        RecoveryAction.CANCEL_GOAL,
        RecoveryAction.CLEAR_LOCAL_CACHE,
        RecoveryAction.SCAN,
        RecoveryAction.REQUEST_REPLAN,
    ]
    observed_prefix = [first_command.action]
    while machine.pending_command is not None:
        action = machine.pending_command.action
        now, emitted = _ack(machine, now)
        observed_prefix.extend(emitted)
        if action == RecoveryAction.REQUEST_REPLAN:
            break
    if observed_prefix[: len(expected_prefix)] != expected_prefix:
        raise AssertionError(f"unexpected recovery prefix: {observed_prefix}")
    if machine.state != RecoveryState.WAIT_FRESH_PLAN:
        raise AssertionError("machine did not wait for a replanned trajectory")

    # The same failed trajectory must be rejected before the execution barrier.
    for _ in range(config.maximum_plan_rejections_per_stage):
        confirmed_replan = next(
            event["operation_id"]
            for event in reversed(machine.events())
            if event["event"] == "action_confirmed"
            and event["action"] == RecoveryAction.REQUEST_REPLAN.value
        )
        machine.submit_replanned_trajectory(
            request_operation_id=confirmed_replan,
            episode_id=machine.episode_id,
            reset_generation=machine.reset_generation,
            recovery_id=machine.recovery_id,
            points=OLD_TRAJECTORY,
            monotonic_sec=now,
        )
        now += 0.05
        if machine.pending_command is not None and (
            machine.pending_command.action == RecoveryAction.REQUEST_REPLAN
        ):
            now, _ = _ack(machine, now)

    if machine.pending_command is None or (
        machine.pending_command.action != RecoveryAction.SAFE_RETREAT
    ):
        raise AssertionError("old-plan budget did not select one bounded retreat")
    while machine.pending_command is not None:
        action = machine.pending_command.action
        now, _ = _ack(machine, now)
        if action == RecoveryAction.REQUEST_REPLAN:
            break
    if machine.state != RecoveryState.WAIT_FRESH_PLAN:
        raise AssertionError("post-retreat scan did not reach replan wait")
    confirmed_replan = next(
        event["operation_id"]
        for event in reversed(machine.events())
        if event["event"] == "action_confirmed"
        and event["action"] == RecoveryAction.REQUEST_REPLAN.value
    )
    emitted = machine.submit_replanned_trajectory(
        request_operation_id=confirmed_replan,
        episode_id=machine.episode_id,
        reset_generation=machine.reset_generation,
        recovery_id=machine.recovery_id,
        points=FRESH_TRAJECTORY,
        monotonic_sec=now,
    )
    now += 0.05
    if [item.action for item in emitted] != [RecoveryAction.ACCEPT_FRESH_PLAN]:
        raise AssertionError("novel plan did not reach the fresh-command barrier")
    now, _ = _ack(machine, now)
    if machine.state != RecoveryState.COOLDOWN:
        raise AssertionError("successful recovery did not enter cooldown")
    machine.tick(
        monotonic_sec=now + config.cooldown_sec,
        safety_clear=True,
        sim_estop_ready=True,
        sim_estop_active=False,
        rear_clearance_m=0.50,
    )
    if machine.state != RecoveryState.MONITORING:
        raise AssertionError("bounded cooldown did not complete")

    statistics = machine.event_statistics()
    required_actions = {
        action.value
        for action in (
            RecoveryAction.CANCEL_GOAL,
            RecoveryAction.CLEAR_LOCAL_CACHE,
            RecoveryAction.SCAN,
            RecoveryAction.REQUEST_REPLAN,
            RecoveryAction.SAFE_RETREAT,
            RecoveryAction.ACCEPT_FRESH_PLAN,
        )
    }
    if not required_actions.issubset(statistics["action_counts"]):
        raise AssertionError("smoke did not exercise every recovery action")
    if statistics["old_trajectory_execution_count"] != 0:
        raise AssertionError("an old trajectory crossed the execution barrier")
    if statistics["terminal_safe_stop_count"] != 0:
        raise AssertionError("happy-path smoke ended in a terminal stop")
    if statistics["commands_this_recovery"] > config.maximum_commands_per_recovery:
        raise AssertionError("command budget was exceeded")
    statistics["status"] = "PASS"
    statistics["event_stream_sha256"] = hashlib.sha256(
        json.dumps(
            machine.events(), separators=(",", ":"), sort_keys=True, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return statistics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the T4.5 deterministic offline recovery smoke."
    )
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    profiles = [run_profile_smoke(load_recovery_config(path)) for path in args.configs]
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(item["status"] == "PASS" for item in profiles) else "FAIL",
        "online_resources_used": [],
        "profiles": profiles,
    }
    rendered = json.dumps(
        payload,
        indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        if not args.output.parent.is_dir():
            raise FileNotFoundError(args.output.parent)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
