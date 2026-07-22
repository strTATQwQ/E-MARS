from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from t4_completion.recovery import (
    Observation,
    RecoveryAction,
    RecoveryMachine,
    RecoveryState,
    load_recovery_config,
    trajectory_signature,
)


ROOT = Path(__file__).resolve().parents[1]
OLD_A = ((0.0, 0.0), (1.0, 0.0))
OLD_B = ((0.0, 0.0), (0.0, 1.0))
FRESH = ((0.0, 0.0), (0.5, 0.5), (1.0, 0.5))


def config(profile: str = "A"):
    return load_recovery_config(
        ROOT / f"configs/completion_sim/recovery/profile_{profile.lower()}.json"
    )


def observation(
    now: float,
    *,
    x: float = 0.0,
    y: float = 0.0,
    goal_distance: float = 5.0,
    goal_active: bool = True,
    motion_requested: bool = True,
    estop_ready: bool = True,
    estop_active: bool = False,
    safety_clear: bool = True,
    rear_clearance: float | None = 0.5,
    plan=None,
    plan_update: bool = False,
    episode: str = "episode",
    generation: int = 1,
    goal: str = "goal",
) -> Observation:
    return Observation(
        episode_id=episode,
        reset_generation=generation,
        goal_id=goal,
        monotonic_sec=now,
        x_m=x,
        y_m=y,
        goal_distance_m=goal_distance,
        goal_active=goal_active,
        motion_requested=motion_requested,
        safety_clear=safety_clear,
        sim_estop_ready=estop_ready,
        sim_estop_active=estop_active,
        rear_clearance_m=rear_clearance,
        plan_points=plan,
        plan_update=plan_update,
    )


def machine_with_episode(profile: str = "A") -> RecoveryMachine:
    machine = RecoveryMachine(config(profile))
    machine.begin_episode("episode", 1, 0.0)
    return machine


def trigger_no_progress(machine: RecoveryMachine, start: float = 0.0):
    now = start
    first = True
    for _ in range(100):
        commands = machine.observe(
            observation(now, plan=OLD_A if first else None, plan_update=first)
        )
        first = False
        if commands:
            assert [item.action for item in commands] == [RecoveryAction.CANCEL_GOAL]
            return now, commands[0]
        now += 0.25
    raise AssertionError("no-progress did not trigger")


def ack(machine: RecoveryMachine, now: float, *, success: bool = True):
    pending = machine.pending_command
    assert pending is not None
    details = {}
    if pending.action == RecoveryAction.SCAN:
        details["measured_max_linear_mps"] = 0.0
        details["measured_max_angular_rps"] = machine.config.scan_angular_speed_rps
    elif pending.action == RecoveryAction.SAFE_RETREAT:
        details["measured_max_linear_mps"] = machine.config.retreat_linear_speed_mps
        details["measured_max_angular_rps"] = 0.0
    if pending.action in {RecoveryAction.SCAN, RecoveryAction.SAFE_RETREAT}:
        details.update(
            {
                "safety_observation_monotonic_sec": now,
                "simulation_estop_preemption_count": 0,
                "safety_gate_violation_count": 0,
                "velocity_chain": "cmd_vel_safe",
            }
        )
    commands = machine.acknowledge(
        operation_id=pending.operation_id,
        episode_id=machine.episode_id,
        reset_generation=machine.reset_generation,
        recovery_id=machine.recovery_id,
        success=success,
        monotonic_sec=now,
        details=details,
    )
    return pending, commands


def drive_to_plan_wait(machine: RecoveryMachine, now: float) -> tuple[float, str]:
    actions = []
    while machine.pending_command is not None:
        pending, commands = ack(machine, now)
        actions.append(pending.action)
        now += 0.05
        if pending.action == RecoveryAction.REQUEST_REPLAN:
            assert machine.state == RecoveryState.WAIT_FRESH_PLAN
            return now, pending.operation_id
        assert len(commands) == 1
    raise AssertionError(f"replan wait was not reached; actions={actions}")


def submit(machine: RecoveryMachine, request_id: str, points, now: float):
    return machine.submit_replanned_trajectory(
        request_operation_id=request_id,
        episode_id=machine.episode_id,
        reset_generation=machine.reset_generation,
        recovery_id=machine.recovery_id,
        points=points,
        monotonic_sec=now,
    )


def test_no_progress_runs_confirmed_sequence_and_finite_cooldown() -> None:
    machine = machine_with_episode()
    now, _ = trigger_no_progress(machine)
    assert machine.pending_command.action == RecoveryAction.CANCEL_GOAL
    # Repeated ticks cannot advance past cancellation without its exact ACK.
    assert machine.tick(
        monotonic_sec=now + 0.01,
        safety_clear=True,
        sim_estop_ready=True,
        sim_estop_active=False,
        rear_clearance_m=0.5,
    ) == []
    now, request_id = drive_to_plan_wait(machine, now + 0.05)
    commands = submit(machine, request_id, FRESH, now)
    assert [item.action for item in commands] == [RecoveryAction.ACCEPT_FRESH_PLAN]
    pending, _ = ack(machine, now + 0.05)
    assert pending.action == RecoveryAction.ACCEPT_FRESH_PLAN
    assert machine.state == RecoveryState.COOLDOWN
    machine.tick(
        monotonic_sec=now + 0.05 + machine.config.cooldown_sec,
        safety_clear=True,
        sim_estop_ready=True,
        sim_estop_active=False,
        rear_clearance_m=0.5,
    )
    assert machine.state == RecoveryState.MONITORING
    stats = machine.event_statistics()
    assert stats["recoveries_completed"] == 1
    assert stats["old_trajectory_execution_count"] == 0


def test_emitted_command_and_nested_arguments_are_deeply_immutable() -> None:
    machine = machine_with_episode()
    now, cancel = trigger_no_progress(machine)
    with pytest.raises(TypeError):
        cancel.arguments["disable_motion_atomically"] = False  # type: ignore[index]
    assert machine.pending_command is not None
    assert machine.pending_command.arguments["disable_motion_atomically"] is True

    _, clear_commands = ack(machine, now + 0.05)
    clear = clear_commands[0]
    assert isinstance(clear.arguments["clear"], tuple)
    with pytest.raises(AttributeError):
        clear.arguments["clear"].append("simulation_estop_latch")

    _, scan_commands = ack(machine, now + 0.10)
    scan = scan_commands[0]
    with pytest.raises(TypeError):
        scan.arguments["maximum_angular_speed_rps"] = 99.0  # type: ignore[index]
    assert (
        machine.pending_command.arguments["maximum_angular_speed_rps"]
        == machine.config.scan_angular_speed_rps
    )


def test_normal_progress_and_inactive_goal_do_not_trigger() -> None:
    machine = machine_with_episode()
    for index in range(30):
        now = index * 0.25
        assert machine.observe(
            observation(
                now,
                x=0.04 * index,
                goal_distance=5.0 - 0.04 * index,
                plan=OLD_A if index == 0 else None,
                plan_update=index == 0,
            )
        ) == []
    assert machine.state == RecoveryState.MONITORING

    idle = machine_with_episode()
    for index in range(30):
        assert idle.observe(
            observation(index * 0.25, goal_active=False, motion_requested=False)
        ) == []
    assert idle.event_statistics()["recoveries_started"] == 0


def test_gapped_observations_do_not_form_a_false_progress_window() -> None:
    machine = machine_with_episode("B")
    for index, now in enumerate((0.0, 0.1, 10.0, 10.1)):
        assert machine.observe(
            observation(
                now,
                plan=OLD_A if index == 0 else None,
                plan_update=index == 0,
            )
        ) == []
    assert machine.event_statistics()["recoveries_started"] == 0
    assert machine.event_statistics()["event_counts"]["observation_gap_reset"] == 1


def test_old_plan_reuse_remains_detectable_after_a_long_delay() -> None:
    machine = machine_with_episode("B")
    machine.observe(observation(0.0, plan=OLD_A, plan_update=True))
    emitted = machine.observe(observation(10.0, plan=OLD_A, plan_update=True))
    assert [item.action for item in emitted] == [RecoveryAction.CANCEL_GOAL]


def test_pose_loop_is_detected_before_no_progress_horizon() -> None:
    machine = machine_with_episode()
    route = [
        (0.0, 0.0),
        (0.2, 0.0),
        (0.2, 0.2),
        (0.0, 0.2),
        (0.0, 0.0),
        (0.2, 0.0),
        (0.2, 0.2),
        (0.0, 0.2),
        (0.0, 0.0),
    ]
    emitted = []
    for index, (x, y) in enumerate(route):
        emitted = machine.observe(
            observation(
                index * 0.2,
                x=x,
                y=y,
                plan=OLD_A if index == 0 else None,
                plan_update=index == 0,
            )
        )
        if emitted:
            break
    assert [item.action for item in emitted] == [RecoveryAction.CANCEL_GOAL]
    assert machine.event_statistics()["trigger_counts"] == {"pose_loop": 1}


def test_a_b_plan_cycle_marks_both_trajectories_taboo() -> None:
    machine = machine_with_episode("B")
    emitted = []
    for index, plan in enumerate((OLD_A, OLD_B, OLD_A, OLD_B)):
        emitted = machine.observe(
            observation(index * 0.1, plan=plan, plan_update=True)
        )
        if emitted:
            break
    assert [item.action for item in emitted] == [RecoveryAction.CANCEL_GOAL]
    assert machine.event_statistics()["trigger_counts"] == {"trajectory_cycle": 1}
    for plan in (OLD_A, OLD_B):
        fingerprint = trajectory_signature(
            plan,
            resolution_m=machine.config.trajectory_resolution_m,
            maximum_points=machine.config.trajectory_max_points,
        )
        assert set(fingerprint.tokens).issubset(machine.taboo_tokens)


def test_three_trajectory_cycle_is_detected_without_unbounded_repetition() -> None:
    machine = machine_with_episode("B")
    old_c = ((0.0, 0.0), (-1.0, 0.0))
    emitted = []
    for index, plan in enumerate((OLD_A, OLD_B, old_c, OLD_A, OLD_B, old_c)):
        emitted = machine.observe(
            observation(index * 0.1, plan=plan, plan_update=True)
        )
        if emitted:
            break
    assert [item.action for item in emitted] == [RecoveryAction.CANCEL_GOAL]
    assert machine.event_statistics()["trigger_counts"] == {"trajectory_cycle": 1}


def test_unbounded_old_plan_stream_terminates_without_execution() -> None:
    machine = machine_with_episode()
    now, _ = trigger_no_progress(machine)
    now, request_id = drive_to_plan_wait(machine, now + 0.05)
    steps = 0
    while machine.state != RecoveryState.TERMINAL_SAFE_STOP and steps < 50:
        steps += 1
        if machine.state == RecoveryState.WAIT_FRESH_PLAN:
            submit(machine, request_id, OLD_A, now)
            now += 0.05
        if machine.pending_command is not None:
            pending, _ = ack(machine, now)
            now += 0.05
            if pending.action == RecoveryAction.REQUEST_REPLAN:
                request_id = pending.operation_id
    assert steps < 50
    assert machine.state == RecoveryState.TERMINAL_SAFE_STOP
    stats = machine.event_statistics()
    assert stats["old_trajectory_execution_count"] == 0
    assert stats["commands_this_recovery"] <= stats["maximum_commands_per_recovery"]
    assert stats["terminal_safe_stop_count"] == 1


def test_action_retry_is_bounded_and_cancel_failure_fails_safe() -> None:
    machine = machine_with_episode()
    now, _ = trigger_no_progress(machine)
    for attempt in range(machine.config.maximum_action_attempts):
        _, commands = ack(machine, now + 0.05, success=False)
        now += 0.1
        if attempt + 1 < machine.config.maximum_action_attempts:
            assert commands == []
            commands = machine.tick(
                monotonic_sec=now + machine.config.retry_backoff_sec,
                safety_clear=True,
                sim_estop_ready=True,
                sim_estop_active=False,
                rear_clearance_m=0.5,
            )
            assert [item.action for item in commands] == [RecoveryAction.CANCEL_GOAL]
            now += machine.config.retry_backoff_sec
    assert machine.state == RecoveryState.TERMINAL_SAFE_STOP
    assert machine.event_statistics()["retry_counts"] == {"cancel_goal": 1}


def test_estop_preempts_scan_and_latches_stop() -> None:
    machine = machine_with_episode()
    now, _ = trigger_no_progress(machine)
    # Confirm cancel and clear so scan is in flight.
    _, _ = ack(machine, now + 0.05)
    _, scan_commands = ack(machine, now + 0.10)
    assert [item.action for item in scan_commands] == [RecoveryAction.SCAN]
    commands = machine.tick(
        monotonic_sec=now + 0.15,
        safety_clear=True,
        sim_estop_ready=True,
        sim_estop_active=True,
        rear_clearance_m=0.5,
    )
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]
    assert machine.state == RecoveryState.TERMINAL_SAFE_STOP


def test_measured_speed_violation_fails_safe() -> None:
    machine = machine_with_episode()
    now, _ = trigger_no_progress(machine)
    ack(machine, now + 0.05)
    ack(machine, now + 0.10)
    scan = machine.pending_command
    assert scan is not None and scan.action == RecoveryAction.SCAN
    commands = machine.acknowledge(
        operation_id=scan.operation_id,
        episode_id=machine.episode_id,
        reset_generation=machine.reset_generation,
        recovery_id=machine.recovery_id,
        success=True,
        monotonic_sec=now + 0.15,
        details={
            "measured_max_linear_mps": 0.0,
            "measured_max_angular_rps": machine.config.scan_angular_speed_rps
            + 0.01,
            "safety_observation_monotonic_sec": now + 0.15,
            "simulation_estop_preemption_count": 0,
            "safety_gate_violation_count": 0,
            "velocity_chain": "cmd_vel_safe",
        },
    )
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]
    assert machine.event_statistics()["speed_cap_violation_count"] == 1


def test_motion_ack_requires_explicit_measured_velocity_evidence() -> None:
    machine = machine_with_episode()
    now, _ = trigger_no_progress(machine)
    ack(machine, now + 0.05)
    ack(machine, now + 0.10)
    scan = machine.pending_command
    assert scan is not None and scan.action == RecoveryAction.SCAN
    commands = machine.acknowledge(
        operation_id=scan.operation_id,
        episode_id=machine.episode_id,
        reset_generation=machine.reset_generation,
        recovery_id=machine.recovery_id,
        success=True,
        monotonic_sec=now + 0.15,
        details={},
    )
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]
    assert machine.event_statistics()["missing_motion_evidence_count"] == 1


def test_failed_motion_ack_still_validates_speed_and_never_retries() -> None:
    machine = machine_with_episode()
    now, _ = trigger_no_progress(machine)
    ack(machine, now + 0.05)
    ack(machine, now + 0.10)
    scan = machine.pending_command
    assert scan is not None and scan.action == RecoveryAction.SCAN
    commands = machine.acknowledge(
        operation_id=scan.operation_id,
        episode_id=machine.episode_id,
        reset_generation=machine.reset_generation,
        recovery_id=machine.recovery_id,
        success=False,
        monotonic_sec=now + 0.15,
        details={
            "measured_max_linear_mps": 0.0,
            "measured_max_angular_rps": machine.config.scan_angular_speed_rps
            + 0.01,
            "safety_observation_monotonic_sec": now + 0.15,
            "simulation_estop_preemption_count": 0,
            "safety_gate_violation_count": 0,
            "velocity_chain": "cmd_vel_safe",
        },
    )
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]
    stats = machine.event_statistics()
    assert stats["speed_cap_violation_count"] == 1
    assert stats["action_counts"]["scan"] == 1
    assert "scan" not in stats["retry_counts"]


def test_rejected_motion_with_valid_evidence_fails_closed_without_retry() -> None:
    machine = machine_with_episode()
    now, _ = trigger_no_progress(machine)
    ack(machine, now + 0.05)
    ack(machine, now + 0.10)
    scan = machine.pending_command
    assert scan is not None and scan.action == RecoveryAction.SCAN
    commands = machine.acknowledge(
        operation_id=scan.operation_id,
        episode_id=machine.episode_id,
        reset_generation=machine.reset_generation,
        recovery_id=machine.recovery_id,
        success=False,
        monotonic_sec=now + 0.15,
        details={
            "measured_max_linear_mps": 0.0,
            "measured_max_angular_rps": machine.config.scan_angular_speed_rps,
            "safety_observation_monotonic_sec": now + 0.15,
            "simulation_estop_preemption_count": 0,
            "safety_gate_violation_count": 0,
            "velocity_chain": "cmd_vel_safe",
        },
    )
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]
    assert machine.state == RecoveryState.TERMINAL_SAFE_STOP
    stats = machine.event_statistics()
    assert stats["action_counts"]["scan"] == 1
    assert "scan" not in stats["retry_counts"]


def test_motion_timeout_fails_closed_without_reissuing_physical_action() -> None:
    machine = machine_with_episode()
    now, _ = trigger_no_progress(machine)
    ack(machine, now + 0.05)
    ack(machine, now + 0.10)
    scan = machine.pending_command
    assert scan is not None and scan.action == RecoveryAction.SCAN
    commands = machine.tick(
        monotonic_sec=scan.deadline_monotonic_sec + 0.01,
        safety_clear=True,
        sim_estop_ready=True,
        sim_estop_active=False,
        rear_clearance_m=0.5,
    )
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]
    stats = machine.event_statistics()
    assert stats["action_counts"]["scan"] == 1
    assert "scan" not in stats["retry_counts"]


def test_cached_safety_snapshot_cannot_start_scan_after_freshness_deadline() -> None:
    machine = machine_with_episode()
    now, _ = trigger_no_progress(machine)
    ack(machine, now + 0.05)
    clear = machine.pending_command
    assert clear is not None and clear.action == RecoveryAction.CLEAR_LOCAL_CACHE
    commands = machine.acknowledge(
        operation_id=clear.operation_id,
        episode_id=machine.episode_id,
        reset_generation=machine.reset_generation,
        recovery_id=machine.recovery_id,
        success=True,
        monotonic_sec=now + machine.config.safety_freshness_timeout_sec + 0.01,
        details={},
    )
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]
    assert machine.state == RecoveryState.TERMINAL_SAFE_STOP


def test_late_replanned_trajectory_never_reaches_accept_barrier() -> None:
    machine = machine_with_episode()
    now, _ = trigger_no_progress(machine)
    now, request_id = drive_to_plan_wait(machine, now + 0.05)
    deadline = next(
        event["plan_wait_deadline"]
        for event in reversed(machine.events())
        if event["event"] == "state_transition"
        and event.get("to_state") == RecoveryState.WAIT_FRESH_PLAN.value
    )
    commands = submit(machine, request_id, FRESH, float(deadline) + 0.01)
    assert RecoveryAction.ACCEPT_FRESH_PLAN not in [item.action for item in commands]
    assert machine.event_statistics()["event_counts"][
        "late_replanned_trajectory_dropped"
    ] == 1


def test_retreat_requires_fresh_rear_clearance() -> None:
    machine = machine_with_episode()
    now, _ = trigger_no_progress(machine)
    now, request_id = drive_to_plan_wait(machine, now + 0.05)
    for _ in range(machine.config.maximum_plan_rejections_per_stage):
        commands = submit(machine, request_id, OLD_A, now)
        now += 0.05
        if machine.pending_command is not None and (
            machine.pending_command.action == RecoveryAction.REQUEST_REPLAN
        ):
            pending, _ = ack(machine, now)
            request_id = pending.operation_id
            now += 0.05
    assert machine.pending_command is not None
    assert machine.pending_command.action == RecoveryAction.SAFE_RETREAT
    commands = machine.tick(
        monotonic_sec=now,
        safety_clear=True,
        sim_estop_ready=True,
        sim_estop_active=False,
        rear_clearance_m=None,
    )
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]


def test_generation_reset_drops_old_ack_without_polluting_new_episode() -> None:
    machine = machine_with_episode()
    now, old_command = trigger_no_progress(machine)
    reset_commands = machine.begin_episode("new-episode", 2, now + 0.1)
    assert [item.action for item in reset_commands] == [
        RecoveryAction.LATCH_SAFE_STOP
    ]
    commands = machine.acknowledge(
        operation_id=old_command.operation_id,
        episode_id="episode",
        reset_generation=1,
        recovery_id=old_command.recovery_id,
        success=True,
        monotonic_sec=now + 0.2,
    )
    assert commands == []
    assert machine.state == RecoveryState.MONITORING
    assert machine.event_statistics()["stale_ack_drop_count"] == 1


def test_every_initialized_generation_reset_reasserts_safe_stop() -> None:
    machine = machine_with_episode()
    commands = machine.begin_episode("episode-2", 2, 0.1)
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]
    assert machine.state == RecoveryState.MONITORING


def test_invalid_runtime_inputs_fail_closed_instead_of_raising() -> None:
    machine = machine_with_episode()
    commands = machine.observe(observation(0.0, x=float("nan")))
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]

    second = machine_with_episode()
    commands = second.tick(
        monotonic_sec=0.0,
        safety_clear=1,  # type: ignore[arg-type]
        sim_estop_ready=True,
        sim_estop_active=False,
        rear_clearance_m=0.5,
    )
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]


def test_wrong_identity_ack_and_time_regression_fail_closed() -> None:
    machine = machine_with_episode()
    now, pending = trigger_no_progress(machine)
    commands = machine.acknowledge(
        operation_id=pending.operation_id,
        episode_id="wrong",
        reset_generation=1,
        recovery_id=machine.recovery_id,
        success=True,
        monotonic_sec=now + 0.05,
    )
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]

    second = machine_with_episode()
    second.observe(observation(1.0, plan=OLD_A, plan_update=True))
    commands = second.tick(
        monotonic_sec=0.5,
        safety_clear=True,
        sim_estop_ready=True,
        sim_estop_active=False,
        rear_clearance_m=0.5,
    )
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]


def test_malformed_ack_identifier_fails_closed_without_type_error() -> None:
    machine = machine_with_episode()
    now, _ = trigger_no_progress(machine)
    commands = machine.acknowledge(
        operation_id=[],  # type: ignore[arg-type]
        episode_id="episode",
        reset_generation=1,
        recovery_id=machine.recovery_id,
        success=True,
        monotonic_sec=now + 0.05,
    )
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]
    assert machine.state == RecoveryState.TERMINAL_SAFE_STOP


def test_absolute_recovery_deadline_precedes_accept_ack_deadline() -> None:
    short = replace(config(), maximum_recovery_duration_sec=5.0)
    machine = RecoveryMachine(short)
    machine.begin_episode("episode", 1, 0.0)
    now, _ = trigger_no_progress(machine)
    recovery_started = next(
        event["monotonic_sec"]
        for event in machine.events()
        if event["event"] == "recovery_triggered"
    )
    now, request_id = drive_to_plan_wait(machine, now + 0.05)
    commands = submit(machine, request_id, FRESH, now)
    assert [item.action for item in commands] == [RecoveryAction.ACCEPT_FRESH_PLAN]
    accept = machine.pending_command
    assert accept is not None
    late = float(recovery_started) + short.maximum_recovery_duration_sec + 0.01
    assert late < accept.deadline_monotonic_sec
    commands = machine.acknowledge(
        operation_id=accept.operation_id,
        episode_id=machine.episode_id,
        reset_generation=machine.reset_generation,
        recovery_id=machine.recovery_id,
        success=True,
        monotonic_sec=late,
    )
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]
    assert machine.event_statistics()["recoveries_completed"] == 0


def test_episode_identity_cannot_reset_budgets_without_new_generation() -> None:
    machine = machine_with_episode()
    commands = machine.begin_episode("episode", 1, 0.1)
    assert [item.action for item in commands] == [RecoveryAction.LATCH_SAFE_STOP]
    assert machine.state == RecoveryState.TERMINAL_SAFE_STOP


def test_command_budget_is_enforced_independently_of_retry_parameters() -> None:
    bounded = replace(config(), maximum_commands_per_recovery=5)
    machine = RecoveryMachine(bounded)
    machine.begin_episode("episode", 1, 0.0)
    now, _ = trigger_no_progress(machine)
    # Drive through cancel, clear, scan, replan and one further request.
    now, request_id = drive_to_plan_wait(machine, now + 0.05)
    submit(machine, request_id, OLD_A, now)
    assert machine.pending_command is not None
    pending, _ = ack(machine, now + 0.05)
    assert pending.action == RecoveryAction.REQUEST_REPLAN
    assert machine.state == RecoveryState.WAIT_FRESH_PLAN
    submit(machine, pending.operation_id, OLD_A, now + 0.10)
    assert machine.state in {
        RecoveryState.SAFE_RETREAT,
        RecoveryState.TERMINAL_SAFE_STOP,
    }
    machine.tick(
        monotonic_sec=now + 0.15,
        safety_clear=True,
        sim_estop_ready=True,
        sim_estop_active=False,
        rear_clearance_m=0.5,
    )
    assert machine.state == RecoveryState.TERMINAL_SAFE_STOP
    assert machine.event_statistics()["commands_this_recovery"] == 5


def test_event_history_is_bounded() -> None:
    bounded = replace(config(), event_history_capacity=64)
    machine = RecoveryMachine(bounded)
    for generation in range(100):
        machine.begin_episode("episode", generation, float(generation))
    stats = machine.event_statistics()
    assert stats["retained_event_count"] == 64
    assert stats["dropped_event_count"] > 36
