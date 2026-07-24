from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
ROS2_PACKAGE_ROOT = ROOT / "internvla_ros2"
if str(ROS2_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS2_PACKAGE_ROOT))

import internvla_ros2.recovery_contract as recovery_contract  # noqa: E402
from internvla_ros2.recovery_contract import (  # noqa: E402
    OP_CANCEL_AND_DISABLE,
    OP_REQUEST_REPLAN,
    STATUS_STALE,
    STATUS_TIMEOUT,
    RecoveryContractError,
    goal_identity,
    operation_identity,
    recovery_deadline_ns,
    recovery_identity,
    semantic_age_sec,
    system2_replan_policy,
    system2_primitive_signature,
    t5_completion_sim_enabled,
    trajectory_signature,
    validate_request,
)


SIM_NOW_NS = 20_000_000_000


def _request(
    *,
    operation: int,
    deadline_offset_sec: float = 5.0,
    now_ns: int = SIM_NOW_NS,
) -> SimpleNamespace:
    deadline_ns = now_ns + int(deadline_offset_sec * 1_000_000_000)
    episode_id = "episode-7"
    generation = 3
    goal_id = goal_identity(episode_id, generation, 11)
    recovery_id = recovery_identity(episode_id, generation, 2)
    return SimpleNamespace(
        episode_id=episode_id,
        reset_generation=generation,
        goal_id=goal_id,
        recovery_id=recovery_id,
        operation_id=operation_identity(recovery_id, operation),
        operation=operation,
        deadline=SimpleNamespace(
            sec=deadline_ns // 1_000_000_000,
            nanosec=deadline_ns % 1_000_000_000,
        ),
        excluded_absolute_sha256=[],
        excluded_shape_sha256=[],
    )


def test_identity_scoped_request_accepts_only_the_active_goal() -> None:
    request = _request(operation=OP_CANCEL_AND_DISABLE)
    identity = validate_request(
        request,
        expected_operation=OP_CANCEL_AND_DISABLE,
        active_episode_id=request.episode_id,
        active_reset_generation=request.reset_generation,
        active_goal_id=request.goal_id,
        now_ns=SIM_NOW_NS,
    )
    assert identity.operation_id == request.operation_id

    with pytest.raises(RecoveryContractError) as caught:
        validate_request(
            request,
            expected_operation=OP_CANCEL_AND_DISABLE,
            active_episode_id=request.episode_id,
            active_reset_generation=request.reset_generation + 1,
            active_goal_id=request.goal_id,
            now_ns=SIM_NOW_NS,
        )
    assert caught.value.status_code == STATUS_STALE


def test_episode_identity_accepts_dataset_paths_but_rejects_control_characters() -> None:
    request = _request(operation=OP_CANCEL_AND_DISABLE)
    request.episode_id = "hm3d/00800-TEEsavR23oF/episode-7"
    request.goal_id = goal_identity(
        request.episode_id, request.reset_generation, 11
    )
    identity = validate_request(
        request,
        expected_operation=OP_CANCEL_AND_DISABLE,
        active_episode_id=request.episode_id,
        active_reset_generation=request.reset_generation,
        active_goal_id=request.goal_id,
        now_ns=SIM_NOW_NS,
    )
    assert identity.episode_id == request.episode_id

    request.episode_id = "episode\n7"
    with pytest.raises(RecoveryContractError):
        validate_request(
            request,
            expected_operation=OP_CANCEL_AND_DISABLE,
            active_episode_id=request.episode_id,
            active_reset_generation=request.reset_generation,
            active_goal_id=request.goal_id,
            now_ns=SIM_NOW_NS,
        )


def test_deadline_and_exclusions_are_fail_closed() -> None:
    expired = _request(
        operation=OP_CANCEL_AND_DISABLE,
        deadline_offset_sec=-1.0,
    )
    with pytest.raises(RecoveryContractError) as caught:
        validate_request(
            expired,
            expected_operation=OP_CANCEL_AND_DISABLE,
            active_episode_id=expired.episode_id,
            active_reset_generation=expired.reset_generation,
            active_goal_id=expired.goal_id,
            now_ns=SIM_NOW_NS,
        )
    assert caught.value.status_code == STATUS_TIMEOUT

    replan = _request(operation=OP_REQUEST_REPLAN)
    replan.excluded_absolute_sha256 = ["a" * 64]
    replan.excluded_shape_sha256 = ["b" * 64]
    identity = validate_request(
        replan,
        expected_operation=OP_REQUEST_REPLAN,
        active_episode_id=replan.episode_id,
        active_reset_generation=replan.reset_generation,
        active_goal_id=replan.goal_id,
        now_ns=SIM_NOW_NS,
    )
    assert identity.operation == OP_REQUEST_REPLAN


def test_recovery_deadline_stays_in_the_supplied_sim_clock_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(operation=OP_CANCEL_AND_DISABLE)
    monkeypatch.setattr(
        recovery_contract.time,
        "time_ns",
        lambda: SIM_NOW_NS + 10_000_000_000_000,
    )
    identity = validate_request(
        request,
        expected_operation=OP_CANCEL_AND_DISABLE,
        active_episode_id=request.episode_id,
        active_reset_generation=request.reset_generation,
        active_goal_id=request.goal_id,
        now_ns=SIM_NOW_NS,
    )
    assert identity.deadline_ns == SIM_NOW_NS + 5_000_000_000

    with pytest.raises(RecoveryContractError) as caught:
        validate_request(
            request,
            expected_operation=OP_CANCEL_AND_DISABLE,
            active_episode_id=request.episode_id,
            active_reset_generation=request.reset_generation,
            active_goal_id=request.goal_id,
            now_ns=identity.deadline_ns,
        )
    assert caught.value.status_code == STATUS_TIMEOUT

    with pytest.raises(RecoveryContractError) as unavailable:
        validate_request(
            request,
            expected_operation=OP_CANCEL_AND_DISABLE,
            active_episode_id=request.episode_id,
            active_reset_generation=request.reset_generation,
            active_goal_id=request.goal_id,
            now_ns=0,
        )
    assert unavailable.value.status_code == STATUS_TIMEOUT


def test_recovery_time_helpers_fail_closed_without_wall_fallback() -> None:
    assert recovery_deadline_ns(SIM_NOW_NS, 5.0) == SIM_NOW_NS + 5_000_000_000
    with pytest.raises(ValueError):
        recovery_deadline_ns(0, 5.0)
    with pytest.raises(ValueError):
        recovery_deadline_ns(SIM_NOW_NS, 120.000_000_001)

    assert semantic_age_sec(SIM_NOW_NS, SIM_NOW_NS) == 0.0
    assert semantic_age_sec(SIM_NOW_NS + 3_000_000_000, SIM_NOW_NS) == 3.0
    assert semantic_age_sec(SIM_NOW_NS - 1, SIM_NOW_NS) is None
    assert semantic_age_sec(0, SIM_NOW_NS) is None


def test_t5_recovery_sim_clock_guard_is_exact() -> None:
    exact = {
        "INTERNNAV_RUNTIME_POLICY": "completion_sim",
        "INTERNNAV_SIMULATION_TARGET": "isaac",
        "INTERNNAV_T5_LANE": "a",
    }
    assert t5_completion_sim_enabled(exact)
    assert t5_completion_sim_enabled({**exact, "INTERNNAV_T5_LANE": "b"})
    assert not t5_completion_sim_enabled({**exact, "INTERNNAV_T5_LANE": "c"})
    assert not t5_completion_sim_enabled(
        {**exact, "INTERNNAV_RUNTIME_POLICY": "strict_evidence"}
    )
    assert not t5_completion_sim_enabled(
        {**exact, "INTERNNAV_SIMULATION_TARGET": "habitat"}
    )


def test_system2_replan_policy_is_t5_only_and_raw_wire_is_lane_a_recovery() -> None:
    exact = {
        "INTERNNAV_RUNTIME_POLICY": "completion_sim",
        "INTERNNAV_SIMULATION_TARGET": "isaac",
        "INTERNNAV_T5_LANE": "a",
        "INTERNNAV_T5_CANDIDATE_PROFILE": "recovery_a",
    }
    assert system2_replan_policy({}) == "observation_bound"
    assert system2_replan_policy(
        {**exact, "INTERNVLA_T5_SYSTEM2_REPLAN_POLICY": "strict"}
    ) == "strict"
    assert system2_replan_policy(
        {**exact, "INTERNVLA_T5_SYSTEM2_REPLAN_POLICY": "raw_wire_warn"}
    ) == "raw_wire_warn"
    with pytest.raises(ValueError, match="T5 Isaac completion_sim"):
        system2_replan_policy(
            {
                "INTERNVLA_T5_SYSTEM2_REPLAN_POLICY": "strict",
                "INTERNNAV_RUNTIME_POLICY": "strict_evidence",
            }
        )
    with pytest.raises(ValueError, match="Lane A Recovery A"):
        system2_replan_policy(
            {
                **exact,
                "INTERNNAV_T5_LANE": "b",
                "INTERNVLA_T5_SYSTEM2_REPLAN_POLICY": "raw_wire_warn",
            }
        )


def test_trajectory_signature_rejects_translated_repetition_by_shape() -> None:
    original = trajectory_signature([(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)])
    translated = trajectory_signature(
        [(5.0, -2.0), (5.25, -2.0), (5.5, -2.0), (6.0, -2.0)]
    )
    assert original.absolute_sha256 != translated.absolute_sha256
    assert original.shape_sha256 == translated.shape_sha256


def test_system2_primitive_shape_excludes_same_observation_across_sequence_and_drift() -> None:
    original = system2_primitive_signature(
        action=2,
        episode_id="a::259",
        reset_generation=1,
        sequence_id=17,
        observation_digest="observation:17",
        x=1.25,
        y=-0.5,
        yaw_rad=0.1,
    )
    next_sequence = system2_primitive_signature(
        action=2,
        episode_id="a::259",
        reset_generation=1,
        sequence_id=18,
        observation_digest="observation:17",
        x=1.251,
        y=-0.499,
        yaw_rad=0.101,
    )
    different_action = system2_primitive_signature(
        action=3,
        episode_id="a::259",
        reset_generation=1,
        sequence_id=18,
        observation_digest="observation:17",
        x=1.251,
        y=-0.499,
        yaw_rad=0.101,
    )

    assert original.absolute_sha256 != next_sequence.absolute_sha256
    assert original.shape_sha256 == next_sequence.shape_sha256
    assert original.shape_sha256 != different_action.shape_sha256

    refreshed_observation = system2_primitive_signature(
        action=2,
        episode_id="a::259",
        reset_generation=1,
        sequence_id=18,
        observation_digest="observation:18",
        x=1.251,
        y=-0.499,
        yaw_rad=0.101,
    )
    assert original.shape_sha256 != refreshed_observation.shape_sha256


def test_ros_runtime_uses_typed_transactions_and_recovery_latch() -> None:
    cmake = (ROOT / "internvla_ros2_msgs/CMakeLists.txt").read_text(encoding="utf-8")
    service = (ROOT / "internvla_ros2_msgs/srv/RecoveryControl.srv").read_text(
        encoding="utf-8"
    )
    adapter = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/adapter_node.py"
    ).read_text(encoding="utf-8")
    model = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/model_node.py"
    ).read_text(encoding="utf-8")
    supervisor = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")
    client = (
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/client_node.py"
    ).read_text(encoding="utf-8")
    active = (
        ROOT / "internvla_nav2_adapter/internvla_nav2_adapter/active_node.py"
    ).read_text(encoding="utf-8")

    assert '"srv/RecoveryControl.srv"' in cmake
    for field in (
        "episode_id",
        "reset_generation",
        "goal_id",
        "recovery_id",
        "operation_id",
        "deadline",
        "excluded_absolute_sha256",
        "excluded_shape_sha256",
        "cache_epoch",
    ):
        assert field in service
    assert "recovery_latched" in adapter
    assert "recovery_latch_suppressed_trajectory" in adapter
    assert "fresh trajectory gate armed" in adapter
    assert "_on_typed_clear_history" in model
    assert "cache_epoch" in model
    assert "RecoveryControl.Request()" in supervisor
    assert "self.replan_publisher" not in supervisor
    assert "old_trajectory_rejected" in client
    assert "response.nav2_goal_sent" in client
    assert "response.nav2_plan_valid" in client
    assert 'getattr(result, "goals_canceling", [])' in active
    assert "if self.active_goal is handle" in active
    assert "if wait:\n                raise" in active


def test_pending_stand_sequence_is_delegated_before_fresh_consumption() -> None:
    client = (
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/client_node.py"
    ).read_text(encoding="utf-8")
    method_start = client.index("    def _resolve_nav2(")
    method = client[
        method_start : client.index("    def _on_t4_odometry(", method_start)
    ]
    rejected_block = method[
        method.index("                would_reject = any(") :
        method.index(
            "        response = self._resolve_nav2_with_terminal_fallback(command)"
        )
    ]

    # A stand/old sequence such as seq5 must reach the recovery adapter so its
    # identity barrier advances before a fresh seq6 is resolved.
    assert "return self._stand_resolution" not in rejected_block
    assert "raise ClientFailure" not in rejected_block
    delegate_at = method.index(
        "        response = self._resolve_nav2_with_terminal_fallback(command)"
    )
    release_guard_at = method.index(
        '            message = "adapter released an excluded recovery trajectory"'
    )
    consumed_at = method.index('                    "consumed_by_fresh_model_step"')
    assert delegate_at < release_guard_at < consumed_at
    assert "bool(response.nav2_goal_sent) or bool(response.nav2_plan_valid)" in method

    adapter = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/adapter_node.py"
    ).read_text(encoding="utf-8")
    latched_start = adapter.index("    def _resolve_recovery_latched(")
    latched = adapter[
        latched_start : adapter.index("    def _on_typed_recovery(", latched_start)
    ]
    resolved_at = latched.index("        resolved = super()._resolve(request, response)")
    accepted_at = latched.index(
        '                "event": "recovery_latch_fresh_trajectory_accepted"'
    )
    released_at = latched.index("        self.recovery_latched = False", accepted_at)
    assert resolved_at < accepted_at < released_at
    failed_resolution = latched[
        latched.index("        if not (") : latched.index("        self._append(", resolved_at)
    ]
    assert "int(resolved.status_code) == 0" in failed_resolution
    assert "bool(resolved.nav2_goal_sent)" in failed_resolution
    assert "bool(resolved.nav2_plan_valid)" in failed_resolution
    assert "self._publish_motion(False)" in failed_resolution
    assert "return resolved" in failed_resolution


def test_pending_replan_reset_sequence_zero_is_delegated_to_adapter() -> None:
    client = (
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/client_node.py"
    ).read_text(encoding="utf-8")
    method_start = client.index("    def _resolve_nav2(")
    method = client[
        method_start : client.index("    def _on_t4_odometry(", method_start)
    ]
    reset_start = method.index("            if (\n")
    reset_block = method[reset_start : method.index("            else:", reset_start)]

    # The first seq0 after an episode reset must clear the client request but
    # still reach the adapter, which owns the reset/sequence barrier.
    assert 'str(command.episode_id) != pending["episode_id"]' in reset_block
    assert "self._pending_typed_replan = None" in reset_block
    assert "identity_changed = True" in reset_block
    assert "return" not in reset_block
    assert "raise" not in reset_block
    assert method.index("identity_changed = True") < method.index(
        "response = self._resolve_nav2_with_terminal_fallback(command)"
    )
    delegated_reset = method[
        method.index(
            "        response = self._resolve_nav2_with_terminal_fallback(command)"
        ) :
    ]
    assert "        if identity_changed:\n            return response" in delegated_reset


def test_fresh_recovery_resolution_exceptions_fail_closed_before_release() -> None:
    adapter = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/adapter_node.py"
    ).read_text(encoding="utf-8")
    latched_start = adapter.index("    def _resolve_recovery_latched(")
    latched = adapter[
        latched_start : adapter.index("    def _on_typed_recovery(", latched_start)
    ]

    try_at = latched.index("        try:\n            self._apply_ablation(request)")
    resolved_at = latched.index(
        "            resolved = super()._resolve(request, response)", try_at
    )
    accepted_at = latched.index(
        '                    "event": "recovery_latch_fresh_trajectory_accepted"',
        resolved_at,
    )
    except_at = latched.index("        except BaseException:", accepted_at)
    safe_stop_at = latched.index("            self._publish_motion(False)", except_at)
    raise_at = latched.index("            raise", safe_stop_at)
    release_at = latched.index("        self.recovery_latched = False", raise_at)

    assert try_at < resolved_at < accepted_at < except_at
    assert except_at < safe_stop_at < raise_at < release_at


def test_recovery_actions_follow_the_node_namespace() -> None:
    supervisor = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")

    assert 'Spin,\n            "spin",' in supervisor
    assert 'BackUp,\n            "backup",' in supervisor
    assert 'Spin,\n            "/spin",' not in supervisor
    assert 'BackUp,\n            "/backup",' not in supervisor
    assert "recovery_semantic_deadline_ns = (" in supervisor
    assert "recovery_deadline_ns(started_ns, self.maximum_recovery_duration)" in supervisor
    assert "13.0 if self.allow_spin_timeout_warn_only else 0.5" in supervisor
    assert "- int(post_spin_reserve_sec * 1_000_000_000)" in supervisor
    assert "spin_result_future = owned_effect(handle.get_result_async)" in supervisor
    assert "self._wait_future_until_semantic_deadline(" in supervisor
    assert "spin_result_future," in supervisor
    assert "spin_semantic_deadline_ns," in supervisor
    timeout_at = supervisor.index("                    except Exception as exc:")
    disable_at = supervisor.index(
        "                                    self.motion_publisher.publish, disabled",
        timeout_at,
    )
    cancel_at = supervisor.index(
        "                        cancel_future = handle.cancel_goal_async()", timeout_at
    )
    assert timeout_at < disable_at < cancel_at
    assert "cancel_future = handle.cancel_goal_async()" in supervisor
    assert "cancel_response = _wait(cancel_future, 2.0)" in supervisor
    assert 'getattr(cancel_response, "goals_canceling", [])' in supervisor
    assert "Nav2 Spin cancellation not confirmed" in supervisor
    assert "if not isinstance(exc, _SemanticDeadlineExpired):" in supervisor
    assert "if not self.allow_spin_timeout_warn_only:" in supervisor
    assert 'raise _SemanticDeadlineExpired(\n                                "Nav2 Spin exceeded' in supervisor
    assert "terminal_result = _wait(spin_result_future, 2.0)" in supervisor
    assert "Nav2 Spin cancellation did not reach terminal state" in supervisor
    assert 'record["scan_cancel_terminal_status"]' in supervisor
    assert 'record["scan_timeout_warn_only"] = True' in supervisor
    assert 'record["scan_warning"] = (' in supervisor
    assert "self.enable_backup\n                    and not self.allow_spin_timeout_warn_only" in supervisor
    assert "_wait(handle.get_result_async(), 15.0)" not in supervisor
    recover = supervisor.split("    def _recover(", 1)[1].split("\n\ndef main(", 1)[0]
    assert "time.monotonic()" not in recover
    assert "time.time_ns()" not in recover


def test_recovery_daemon_side_effects_are_owned_by_episode_epoch() -> None:
    supervisor = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")
    command = supervisor.split("    def _on_command(", 1)[1].split(
        "    def _progress_metrics(", 1
    )[0]
    adopt = supervisor.split(
        "    def _adopt_episode_identity_locked(", 1
    )[1].split("    def _on_episode_prime(", 1)[0]
    recover = supervisor.split("    def _recover(", 1)[1].split(
        "\n\ndef main(", 1
    )[0]
    request = supervisor.split("    def _recovery_request(", 1)[1].split(
        "    @staticmethod", 1
    )[0]

    assert "self._adopt_episode_identity_locked(" in command
    assert "self._episode_epoch += 1" in adopt
    assert "self.recovery_active = False" in adopt
    assert "self.last_recovery_finished_semantic_ns = 0" in adopt
    assert "recovery_epoch = self._episode_epoch" in supervisor
    assert "recovery_episode_id = self.episode_id" in supervisor
    assert "recovery_generation = self.generation" in supervisor
    assert "recovery_epoch: int" in recover
    assert "recovery_episode_id: str" in recover
    assert "recovery_generation: int" in recover

    # Old daemons never publish or start service/action work directly.  The
    # only unguarded action operation is cancellation of their own old handle.
    assert ".publish(" not in recover
    assert "owned_effect(self.motion_publisher.publish" in recover
    assert "owned_effect(self.stop_publisher.publish" in recover
    assert "owned_effect(client.call_async, request)" in supervisor
    assert "owned_effect(\n                    self.spin_client.send_goal_async" in recover
    assert "handle.cancel_goal_async()" in recover

    assert "episode_id = self.episode_id" not in request
    assert "generation = self.generation" not in request
    assert "episode_id=recovery_episode_id" in recover
    assert "generation=recovery_generation" in recover
    finalizer = recover.rsplit("        finally:", 1)[1]
    owner_guard = finalizer.index("require_owner()")
    clear_samples = finalizer.index("self.samples.clear()")
    clear_cooldown = finalizer.index(
        "self.last_recovery_finished_semantic_ns = ended_ns or 0"
    )
    assert owner_guard < clear_samples < clear_cooldown


def test_recovery_episode_prime_invalidates_reset_boundary_state() -> None:
    supervisor = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")
    init = supervisor.split("    def __init__(", 1)[1].split(
        "    def _append(", 1
    )[0]
    adopt = supervisor.split(
        "    def _adopt_episode_identity_locked(", 1
    )[1].split("    def _on_episode_prime(", 1)[0]
    prime = supervisor.split("    def _on_episode_prime(", 1)[1].split(
        "    def _on_command(", 1
    )[0]
    command = supervisor.split("    def _on_command(", 1)[1].split(
        "    def _progress_metrics(", 1
    )[0]

    assert '"/internvla_t4/episode_prime"' in init
    assert "String," in init
    assert "generation < self.generation" in adopt
    assert "generation == self.generation and self.episode_id" in adopt
    assert "self._episode_epoch += 1" in adopt
    for reset_statement in (
        "self.samples.clear()",
        "self.path_hashes.clear()",
        "self.latest_command_signature = None",
        "self.last_sequence_id = -1",
        "self.recovery_active = False",
        "self.motion_enabled = False",
    ):
        assert reset_statement in adopt
    assert 'value = json.loads(message.data)' in prime
    assert 'value.get("schema_version", 0)' in prime
    assert "ignoring malformed episode prime" in prime
    assert "self._adopt_episode_identity_locked(episode_id, generation)" in prime
    assert "if not self._adopt_episode_identity_locked(" in command
    assert "return" in command


def test_replan_wait_starts_before_spin_and_prearm_commands_are_not_rejected() -> None:
    supervisor = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")
    recovery_start = supervisor.index("    def _recover(")
    recovery = supervisor[
        recovery_start : supervisor.index("\n\ndef main(", recovery_start)
    ]

    client_replan_at = recovery.index(
        "client_replan_request = self._recovery_request("
    )
    spin_at = recovery.index(
        "spin_result = self._wait_future_until_semantic_deadline(",
        client_replan_at,
    )
    adapter_replan_at = recovery.index(
        "adapter_replan_request = self._recovery_request(", spin_at
    )

    assert client_replan_at < spin_at < adapter_replan_at
    assert recovery.count("client_replan_request = self._recovery_request(") == 1

    adapter = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/adapter_node.py"
    ).read_text(encoding="utf-8")
    client = (
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/client_node.py"
    ).read_text(encoding="utf-8")
    marker = "recovery latch holding until replan gate is armed"
    assert marker in adapter
    pre_arm_at = client.index("        if pre_arm_hold:")
    return_at = client.index("            return response", pre_arm_at)
    rejection_at = client.index(
        '                    "old_trajectory_rejected"', return_at
    )
    quarantine_at = client.index(
        '                    "pre_release_trajectory_quarantined"', rejection_at
    )
    assert pre_arm_at < return_at < rejection_at < quarantine_at


def test_spin_timeout_warn_only_is_t5_isaac_completion_sim_only() -> None:
    supervisor = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")
    contract = (
        ROOT / "internvla_ros2/internvla_ros2/recovery_contract.py"
    ).read_text(encoding="utf-8")
    assert 'values.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"' in contract
    assert 'values.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"' in contract
    assert 'values.get("INTERNNAV_T5_LANE", "") in {"a", "b"}' in contract
    assert "self._recovery_uses_sim_time = t5_completion_sim_enabled()" in supervisor
    assert "T5 completion_sim recovery requires use_sim_time=true" in supervisor


def test_recovery_control_producer_and_consumers_inject_one_clock_domain() -> None:
    supervisor = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")
    adapter = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/adapter_node.py"
    ).read_text(encoding="utf-8")
    model = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/model_node.py"
    ).read_text(encoding="utf-8")

    request_method = supervisor.split("    def _recovery_request(", 1)[1].split(
        "    @staticmethod", 1
    )[0]
    assert "recovery_deadline_ns(self._contract_now_ns(), timeout_sec)" in request_method
    assert "time.time_ns()" not in request_method
    assert adapter.count("now_ns=self._recovery_contract_now_ns(),") == 2
    assert model.count("now_ns=self._recovery_contract_now_ns(),") == 1
    for consumer in (adapter, model):
        assert "value <= 0 or value < self._last_recovery_contract_sim_ns" in consumer
        assert "T5 completion_sim recovery requires use_sim_time=true" in consumer


def test_recovery_progress_and_cooldown_use_semantic_time_only() -> None:
    supervisor = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")
    semantic_section = supervisor.split("    def _on_path(", 1)[1].split(
        "    def _recovery_request(", 1
    )[0]
    assert "time.time_ns()" not in semantic_section
    assert "time.monotonic()" not in semantic_section
    assert "self.latest_path_semantic_ns" in semantic_section
    assert "self.latest_command_semantic_ns" in semantic_section
    assert "self.last_recovery_finished_semantic_ns" in semantic_section
    assert "semantic_age_sec(" in semantic_section

    waiter = supervisor.split(
        "    def _wait_future_until_semantic_deadline(", 1
    )[1].split("    def _begin_spin_telemetry(", 1)[0]
    assert "now_ns >= deadline_ns" in waiter
    assert "_SIM_CLOCK_STALL_WATCHDOG_SEC" in waiter
    assert "time.monotonic()" in waiter


def test_full_spin_records_bounded_performance_telemetry() -> None:
    supervisor = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")
    onboard = (ROOT / "scripts/run_t4_dgx_onboard.sh").read_text(
        encoding="utf-8"
    )

    assert 'self.declare_parameter("recovery_scan_speed_rps", 0.3)' in supervisor
    assert "if self._spin_measurement_active and math.isfinite(yaw_rate):" in supervisor
    assert "spin_telemetry = self._begin_spin_telemetry()" in supervisor
    assert "self._finish_spin_telemetry(" in supervisor
    assert "record, spin_telemetry" in supervisor
    for field in (
        "spin_wall_duration_sec",
        "spin_sim_duration_sec",
        "spin_sim_duration_valid",
        "spin_rtf",
        "commanded_yaw_rate_rps",
        "measured_yaw_rate_rps",
        "measured_yaw_rate_sample_count",
        "measured_yaw_rate_peak_rps",
        "command_age_sec_at_spin_start",
        "command_age_sec_at_spin_end",
    ):
        assert f'"{field}"' in supervisor
    assert 'json.dumps(payload, sort_keys=True, allow_nan=False)' in supervisor
    assert (
        '-p recovery_scan_speed_rps:="${INTERNVLA_T4_RECOVERY_SCAN_SPEED_RPS:-0.3}"'
        in onboard
    )


def test_replan_deadline_is_bounded_and_defaults_to_frozen_t4_value() -> None:
    supervisor = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")
    onboard = (ROOT / "scripts/run_t4_dgx_onboard.sh").read_text(
        encoding="utf-8"
    )

    assert 'self.declare_parameter("replan_deadline_sec", 30.0)' in supervisor
    assert "5.0 <= self.replan_deadline <= 120.0" in supervisor
    assert "timeout_sec=self.replan_deadline" in supervisor
    assert "timeout_sec=30.0" not in supervisor
    assert (
        '-p replan_deadline_sec:="${INTERNVLA_T4_REPLAN_DEADLINE_SEC:-30.0}"'
        in onboard
    )


def test_t4_adapter_defers_virtual_summary_until_subclass_is_ready() -> None:
    adapter = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/adapter_node.py"
    ).read_text(encoding="utf-8")

    guard = "self._t4_summary_ready = False"
    parent_init = "super().__init__()"
    ready = "self._t4_summary_ready = True"
    fallback = (
        "if not self._t4_summary_ready:\n"
        "            ActiveNav2Adapter._write_summary(self, status)\n"
        "            return"
    )
    assert adapter.index(guard) < adapter.index(parent_init)
    assert adapter.index(ready) > adapter.index("self.activation_counts =")
    assert fallback in adapter
    assert 'self._write_summary("READY")' in adapter


@pytest.mark.parametrize(
    ("profile_name", "expected_speed"),
    [("profile_a.json", "0.3"), ("profile_b.json", "0.35")],
)
def test_profile_materializes_bounded_nav2_recovery_overlay(
    tmp_path: Path, profile_name: str, expected_speed: str
) -> None:
    output = tmp_path / "nav2.yaml"
    manifest = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts/t4_recovery_runtime.py"),
            "--profile",
            str(ROOT / "configs/completion_sim/recovery" / profile_name),
            "--nav2-input",
            str(ROOT / "configs/completion_sim/map/nav2_static_lidar.yaml"),
            "--nav2-output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    text = output.read_text(encoding="utf-8")
    assert f"    max_rotational_vel: {expected_speed}\n" in text
    assert "    min_rotational_vel: 0.15\n" in text
    assert "    rotational_acc_lim: 0.7\n" in text
    assert "      max_vel_theta: 1.0\n" in text
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "RECOVERY_RUNTIME_READY"
    assert payload["short_backup_enabled"] is False
    assert payload["runtime_policy"] == "completion_sim"


def test_sensor_gate_binds_profile_and_rejects_non_sim_recovery() -> None:
    gate = (ROOT / "scripts/run_t4_sensor_gate.sh").read_text(encoding="utf-8")
    overlay = (ROOT / "scripts/build_t4_sensor_phase_overlay.py").read_text(
        encoding="utf-8"
    )
    assert 'INTERNNAV_RUNTIME_POLICY:-}" = completion_sim' in gate
    assert 'INTERNNAV_SIMULATION_TARGET:-}" = isaac' in gate
    assert "t4_recovery_runtime.py" in gate
    assert "profile_b.json" in gate
    assert "RECOVERY_RUNTIME_MANIFEST" in gate
    assert "enable_scheduled_refresh:=false" in overlay
    assert "enable_short_backup:=false" in overlay
    assert "recovery_profile_sha256" in overlay
    assert "ros2 interface show internvla_ros2_msgs/srv/RecoveryControl" in gate


def test_scheduled_recovery_refresh_is_explicitly_bounded() -> None:
    node = (ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py").read_text(
        encoding="utf-8"
    )
    assert 'self.declare_parameter("maximum_scheduled_refreshes_per_episode", 1)' in node
    assert "0 <= self.maximum_scheduled_refreshes <= 3" in node
    assert "self.scheduled_refreshes_this_episode = 0" in node
    assert "< self.maximum_scheduled_refreshes" in node
    assert "self.scheduled_refreshes_this_episode += 1" in node
    assert "self.latest_command_semantic_ns = now_ns if now_ns is not None else 0" in node
    assert "if self.motion_enabled:" in node
    assert "now_ns, self.latest_command_semantic_ns" in node
    assert "command_age <= self.trajectory_validity" in node
    assert 'self.latest_signature_source = "nav2_active_path"' in node
    assert 'self.latest_signature_source = "navigation_command"' in node
    assert "if bool(message.stop):" in node
    assert "elif points:" in node
    assert "if self.latest_command_signature is None:" in node
    assert '"active_signature_source": active_signature_source' in node


def test_navigate_to_pose_plan_is_published_for_recovery_supervision() -> None:
    adapter = (
        ROOT / "internvla_nav2_adapter/internvla_nav2_adapter/active_node.py"
    ).read_text(encoding="utf-8")
    on_plan = adapter.split("    def _on_plan", 1)[1].split("    def _on_odom", 1)[0]
    assert "self.active_path_publisher.publish(message)" in on_plan


def test_system2_recovery_replan_is_completion_sim_only() -> None:
    client = (
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/client_node.py"
    ).read_text(encoding="utf-8")
    adapter = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/adapter_node.py"
    ).read_text(encoding="utf-8")
    supervisor = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")
    for source in (client, adapter):
        assert 'os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"' in source
        assert 'os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"' in source
        assert "allow_system2_recovery_replan" in source
        assert "int(command.action_source) == 1" in source
        assert "int(command.discrete_action) in {1, 2, 3}" in source
    assert 'signature_source = "system2_generated_path"' in adapter
    assert '"system2_primitive" if fresh_system2_action else "local_path"' in client
    assert "self.latest_command_signature = system2_primitive_signature(" in supervisor
    assert 'self.latest_signature_source = "system2_primitive"' in supervisor
    assert "primitive_signature = system2_primitive_signature(" in adapter
    assert "signature = system2_primitive_signature(" in client
    suppressed = adapter.split("        if not fresh:", 1)[1].split(
        "        self._append(", 1
    )[0]
    assert "self._check_identity(command)" in suppressed
    assert "response.nav2_goal_sent = False" in suppressed
    assert "response.nav2_plan_valid = False" in suppressed


def _load_recovery_yaw_helpers() -> dict[str, object]:
    source = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)
    selected = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name
        in {
            "_wrapped_angle",
            "_yaw_from_quaternion",
            "_directed_system2_yaw_progress",
        }
    ]
    namespace: dict[str, object] = {
        "math": math,
        "Any": object,
        "ACTION_LEFT": 2,
        "ACTION_RIGHT": 3,
        "_SYSTEM2_TURN_ACTIONS": frozenset({2, 3}),
    }
    code = compile(
        ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
        str(ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"),
        "exec",
    )
    exec(code, namespace)
    return namespace


def test_system2_turn_progress_uses_quaternion_yaw_and_unwraps_boundary() -> None:
    helpers = _load_recovery_yaw_helpers()
    yaw_from_quaternion = helpers["_yaw_from_quaternion"]
    directed_progress = helpers["_directed_system2_yaw_progress"]

    def orientation(degrees: float) -> SimpleNamespace:
        radians = math.radians(degrees)
        return SimpleNamespace(
            x=0.0,
            y=0.0,
            z=math.sin(radians / 2.0),
            w=math.cos(radians / 2.0),
        )

    yaws = [
        yaw_from_quaternion(orientation(degrees))
        for degrees in (179.0, -175.0, -164.0)
    ]
    left_samples = [
        (9, 0.0, 0.0, 0.0),
        (10, 0.0, 0.0, yaws[0]),
        (11, 0.0, 0.0, yaws[1]),
        (12, 0.0, 0.0, yaws[2]),
    ]
    assert directed_progress(left_samples, 2, 10) == pytest.approx(
        math.radians(17.0)
    )

    right_samples = [
        (10, 0.0, 0.0, math.radians(-179.0)),
        (11, 0.0, 0.0, math.radians(175.0)),
        (12, 0.0, 0.0, math.radians(164.0)),
    ]
    assert directed_progress(right_samples, 3, 10) == pytest.approx(
        math.radians(17.0)
    )
    assert directed_progress(left_samples, 2, 0) == 0.0
    assert directed_progress([(10, 0.0, 0.0, math.nan)] * 2, 2, 10) == 0.0


def test_system2_yaw_progress_requires_execution_ack_and_preserves_strict_xy() -> None:
    source = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")
    motion = source.split("    def _on_motion(", 1)[1].split(
        "    def _adopt_episode_identity_locked(", 1
    )[0]
    odom = source.split("    def _on_odom(", 1)[1].split(
        "    def _read_sim_clock_ns(", 1
    )[0]
    command = source.split("    def _on_command(", 1)[1].split(
        "    def _progress_metrics(", 1
    )[0]
    tick = source.split("    def _tick(", 1)[1].split(
        "    def _recovery_request(", 1
    )[0]

    assert "elif not was_enabled:" in motion
    assert "self._motion_true_edge_serial += 1" in motion
    assert "self._system2_turn_progress_armed = True" in motion
    assert "self._system2_turn_motion_started_ns = now_ns" in motion
    assert "self._system2_turn_progress_armed = False" in command
    assert "DDS may deliver the adapter's execution edge before this" in command
    assert "self._motion_true_edge_serial" in command
    assert "yaw = math.nan" in odom
    assert "self.samples.append((now_ns, x, y, yaw))" in odom
    assert "return\n        if not all(math.isfinite(value) for value in (x, y))" not in odom
    assert "_SYSTEM2_TURN_PROGRESS_RAD = math.radians(12.0)" in source
    assert "displacement < self.minimum_progress and not yaw_progress_sufficient" in tick
    assert tick.count("and not yaw_progress_sufficient") == 3


def test_system2_progress_extension_is_completion_sim_only_and_audit_neutral() -> None:
    source = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/recovery_node.py"
    ).read_text(encoding="utf-8")
    command = source.split("    def _on_command(", 1)[1].split(
        "    def _progress_metrics(", 1
    )[0]
    metrics = source.split("    def _progress_metrics(", 1)[1].split(
        "    def _tick(", 1
    )[0]
    tick = source.split("    def _tick(", 1)[1].split(
        "    def _recovery_request(", 1
    )[0]

    assert "self._recovery_uses_sim_time\n                and not bool(message.stop)" in command
    assert "if self._recovery_uses_sim_time\n            and self._system2_turn_progress_armed" in metrics
    assert "self._recovery_uses_sim_time\n                and self._system2_turn_progress_armed" in tick
    assert "system2_yaw_progress_rad" not in source


def test_completion_reset_reconciliation_is_exact_and_bounded() -> None:
    client = (ROOT / "internvla_ros2/internvla_ros2/client_node.py").read_text(
        encoding="utf-8"
    )
    assert 'os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"' in client
    assert 'os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"' in client
    assert '== "reset barrier is behind the committed sequence"' in client
    assert "remote_barrier == local_barrier + 1" in client
    assert '"bounded completion reset reconciliation"' in client
    assert "self.reset_barrier_reconcile_count += 1" in client


def test_replan_gate_keeps_the_cancel_time_goal_identity() -> None:
    adapter = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/adapter_node.py"
    ).read_text(encoding="utf-8")
    assert 'self.recovery_latched_goal_id = identity.goal_id' in adapter
    assert 'self.recovery_latched_id == str(request.recovery_id)' in adapter
    assert 'active_goal_id = self.recovery_latched_goal_id' in adapter
    assert 'active_goal_id=active_goal_id' in adapter
    assert adapter.count('self.recovery_latched_goal_id = ""') >= 3


def test_ros_overlay_updates_are_lease_gated_and_rebuild_messages() -> None:
    isaac = (ROOT / "scripts/update_t4_isaac_ros_overlay.sh").read_text(
        encoding="utf-8"
    )
    dgx = (ROOT / "scripts/build_t4_host_ros.sh").read_text(encoding="utf-8")
    assert 'isaac|dgx+isaac)' in isaac
    assert '"$HOME/internnav-t4/isaac_ros_ws_45")' in isaac
    assert "rsync -a --delete" in isaac
    assert "internvla_ros2_msgs" in isaac
    assert "--packages-up-to internvla_t4_sensors internvla_t4_recovery" in isaac
    assert "RecoveryControl" in isaac
    assert isaac.count("  set +u\n  source /opt/ros/jazzy/setup.bash\n") == 2
    assert isaac.count("  set -u\n") >= 2
    assert "strict_evidence_modified\":False" in isaac
    assert 'dgx|dgx+isaac)' in dgx
    assert "PACKAGES=(" in dgx
    assert "internvla_t4_sensors" in dgx
    assert "internvla_t4_recovery" in dgx
    assert "go2_sensor_bridge" in dgx
    assert "nav2_bringup" in dgx
    assert "--packages-up-to" in dgx
    assert "RecoveryControl" in dgx


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def test_functional_recovery_gate_warns_on_quality_without_allowing_old_path(
    tmp_path: Path,
) -> None:
    (tmp_path / "per_episode.json").write_text(
        json.dumps(
            {
                "completed_episode_count": 1,
                "episodes": [{"termination_reason": "stuck"}],
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        tmp_path / "recovery_records.jsonl",
        [
            {
                "event": "recovery",
                "episode_id": "episode-7",
                "recovery_id": "recovery-1",
                "recovery_index": 1,
                "full_recovery": True,
                "typed_transaction": True,
                "cancel_success": True,
                "history_clear_success": True,
                "fresh_trajectory_requested": True,
                "adapter_replan_gate_armed": True,
                "cache_epoch": 1,
                "excluded_absolute_sha256": ["a" * 64],
                "excluded_shape_sha256": ["b" * 64],
            }
        ],
    )
    _write_jsonl(
        tmp_path / "replan_request_records.jsonl",
        [
            {"event": "requested", "request_index": 1},
            {"event": "consumed_by_fresh_model_step", "request_index": 1},
        ],
    )
    _write_jsonl(
        tmp_path / "active_records.jsonl",
        [
            {
                "event": "recovery_latch_fresh_trajectory_accepted",
                "recovery_id": "recovery-1",
                "absolute_sha256": "c" * 64,
                "shape_sha256": "d" * 64,
            }
        ],
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts/analyze_t4_recovery.py"),
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads((tmp_path / "recovery_metrics.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["quality_status"] == "WARN"
    assert payload["old_trajectory_execution_count"] == 0

    records = [
        {
            "event": "recovery_latch_fresh_trajectory_accepted",
            "recovery_id": "recovery-1",
            "absolute_sha256": "a" * 64,
            "shape_sha256": "d" * 64,
        }
    ]
    _write_jsonl(tmp_path / "active_records.jsonl", records)
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts/analyze_t4_recovery.py"),
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    payload = json.loads((tmp_path / "recovery_metrics.json").read_text())
    assert payload["status"] == "FAIL"
    assert payload["old_trajectory_execution_count"] == 1


def test_recovery_analyzer_accepts_split_host_evidence(tmp_path: Path) -> None:
    isaac = tmp_path / "isaac"
    dgx = tmp_path / "dgx"
    isaac.mkdir()
    dgx.mkdir()
    per_episode = isaac / "per_episode.json"
    per_episode.write_text(
        json.dumps(
            {
                "completed_episode_count": 1,
                "episodes": [{"termination_reason": "stuck"}],
            }
        ),
        encoding="utf-8",
    )
    records = dgx / "recovery_records.jsonl"
    _write_jsonl(
        records,
        [
            {
                "event": "recovery",
                "episode_id": "episode-7",
                "recovery_id": "recovery-1",
                "recovery_index": 1,
                "recovery_profile_id": "A",
                "recovery_profile_sha256": "e" * 64,
                "full_recovery": True,
                "typed_transaction": True,
                "cancel_success": True,
                "history_clear_success": True,
                "fresh_trajectory_requested": True,
                "adapter_replan_gate_armed": True,
                "cache_epoch": 1,
                "excluded_absolute_sha256": ["a" * 64],
                "excluded_shape_sha256": ["b" * 64],
            }
        ],
    )
    replan = isaac / "replan_request_records.jsonl"
    _write_jsonl(
        replan,
        [
            {"event": "requested", "request_index": 1},
            {"event": "consumed_by_fresh_model_step", "request_index": 1},
        ],
    )
    active = dgx / "active_records.jsonl"
    _write_jsonl(
        active,
        [
            {
                "event": "recovery_latch_fresh_trajectory_accepted",
                "recovery_id": "recovery-1",
                "absolute_sha256": "c" * 64,
                "shape_sha256": "d" * 64,
            }
        ],
    )
    output = tmp_path / "recovery_metrics.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts/analyze_t4_recovery.py"),
            str(tmp_path),
            "--per-episode",
            str(per_episode),
            "--records",
            str(records),
            "--replan-records",
            str(replan),
            "--adapter-records",
            str(active),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["recovery_profile_ids"] == ["A"]
    assert payload["recovery_profile_sha256"] == ["e" * 64]
    assert payload["evidence_paths"] == {
        "per_episode": per_episode.as_posix(),
        "recovery_records": records.as_posix(),
        "replan_records": replan.as_posix(),
        "adapter_records": active.as_posix(),
    }
