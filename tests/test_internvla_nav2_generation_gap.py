from types import SimpleNamespace
from threading import Condition, Lock
import json
import math
import time

import pytest


rclpy = pytest.importorskip("rclpy")
pytest.importorskip("internvla_ros2_msgs")

from internvla_nav2_adapter.active_node import (
    ActiveNav2Adapter,
    _load_onboard_restart_identity,
)
from internvla_nav2_adapter import active_node as active_node_module
from internvla_nav2_adapter.shadow_node import InternVLANav2Shadow
from internvla_go2_controller.bridge_node import (
    _active_identity_query_response,
    _load_onboard_restart_identity as _load_controller_restart_identity,
)
from internvla_ros2.fault_injection import FAULT_PROFILE, build_fault_restart_session
from internvla_ros2.identity import CHECKPOINT_REVISION, MODEL_REVISION
from action_msgs.msg import GoalStatus
from nav_msgs.msg import Path as NavPath


def command(generation: int, episode: str, sequence: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        reset_generation=generation,
        episode_id=episode,
        sequence_id=sequence,
    )


def onboard_restart_session(tmp_path) -> object:
    payload = build_fault_restart_session(
        {
            "status_code": 0,
            "status_message": "ready",
            "initialized": True,
            "lifecycle_state": 2,
            "episode_id": "a::259",
            "reset_generation": 16,
            "last_sequence_id": 0,
            "model_revision": MODEL_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
        },
        lane="a",
        event_id="fi-03-dgx-ros-node-restart",
        action="dgx_ros_node_restart",
        captured_unix=1.0,
    )
    path = tmp_path / "restart-session.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def enable_onboard_restart(monkeypatch, path) -> None:
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNNAV_T5_FAULT_INJECTION_PROFILE", FAULT_PROFILE)
    monkeypatch.setenv("INTERNVLA_T5_ONBOARD_RESTART_SESSION_FILE", str(path))


def test_active_restores_exact_onboard_restart_identity(
    tmp_path, monkeypatch
) -> None:
    path = onboard_restart_session(tmp_path)
    enable_onboard_restart(monkeypatch, path)

    episode, generation, sequence = _load_onboard_restart_identity()
    fake = SimpleNamespace(
        active_episode=episode,
        active_generation=generation,
        last_sequence=sequence,
    )
    ActiveNav2Adapter._check_identity(
        fake, command(generation, episode, sequence + 1)
    )

    assert (fake.active_episode, fake.active_generation, fake.last_sequence) == (
        "a::259",
        16,
        1,
    )


def test_controller_restores_same_safe_stopped_onboard_identity(
    tmp_path, monkeypatch
) -> None:
    path = onboard_restart_session(tmp_path)
    enable_onboard_restart(monkeypatch, path)

    identity = _load_controller_restart_identity()
    assert identity == ("a::259", 16, 0)
    response = _active_identity_query_response(identity, 0)
    assert response["status"] == "ok"
    assert response["active_episode_id"] == "a::259"
    assert response["active_reset_generation"] == 16
    assert response["active_sequence_id"] == 0
    assert response["emergency_stop"] is True


def test_onboard_restart_identity_is_opt_in_and_fail_closed(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("INTERNVLA_T5_ONBOARD_RESTART_SESSION_FILE", raising=False)
    assert _load_onboard_restart_identity() is None

    path = onboard_restart_session(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["action"] = "model_service_restart"
    path.write_text(json.dumps(payload), encoding="utf-8")
    enable_onboard_restart(monkeypatch, path)
    with pytest.raises(ValueError, match="session identity mismatch"):
        _load_onboard_restart_identity()


def navigation_command_pose(
    *,
    gps: list[float] | None = None,
    rotation_wxyz: list[float] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        global_gps=[1.25, -0.5, 0.0] if gps is None else gps,
        global_rotation_wxyz=(
            [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
            if rotation_wxyz is None
            else rotation_wxyz
        ),
    )


def continuous_anchor_adapter(
    *,
    odom: tuple[float, float, float] | None,
    odom_age_sec: float,
    allow_fallback: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        execution_mode="continuous",
        odom_lock=Lock(),
        latest_odom=odom,
        latest_odom_monotonic=time.monotonic() - odom_age_sec,
        allow_command_pose_anchor_fallback=allow_fallback,
        command_pose_anchor_fallback_count=0,
        current_command_pose_anchor_fallback=False,
        current_path_anchor_source=None,
        current_path_anchor_odom_age_sec=None,
        last_path_anchor_source=None,
        last_path_anchor_odom_age_sec=None,
    )


def test_active_accepts_unobserved_completed_episode_generation() -> None:
    fake = SimpleNamespace(
        active_generation=3,
        active_episode="episode-3",
        last_sequence=8,
        plan_condition=Condition(),
        latest_plan=[object()],
        latest_plan_monotonic=1.0,
        reset_settle_sec=0.0,
        reset_settle_until=0.0,
        execution_mode="continuous",
        _publish_motion=lambda _enabled: None,
        _cancel_active=lambda wait=True: None,
    )
    ActiveNav2Adapter._check_identity(fake, command(5, "episode-5"))
    assert (fake.active_generation, fake.active_episode, fake.last_sequence) == (
        5,
        "episode-5",
        0,
    )
    assert fake.latest_plan == []


def test_active_rejects_generation_rollback() -> None:
    fake = SimpleNamespace(
        active_generation=5,
        active_episode="episode-5",
        last_sequence=0,
    )
    with pytest.raises(ValueError, match="invalid reset generation"):
        ActiveNav2Adapter._check_identity(fake, command(4, "episode-4"))


def test_active_accepts_same_episode_at_higher_reset_generation() -> None:
    fake = SimpleNamespace(
        active_generation=5,
        active_episode="episode-5",
        last_sequence=8,
        plan_condition=Condition(),
        latest_plan=[object()],
        latest_plan_monotonic=1.0,
        reset_settle_sec=0.0,
        reset_settle_until=0.0,
        execution_mode="continuous",
        _publish_motion=lambda _enabled: None,
        _cancel_active=lambda wait=True: None,
    )

    ActiveNav2Adapter._check_identity(fake, command(6, "episode-5"))

    assert (fake.active_generation, fake.active_episode, fake.last_sequence) == (
        6,
        "episode-5",
        0,
    )
    assert fake.latest_plan == []


def test_shadow_does_not_count_forward_gap_as_pollution() -> None:
    fake = SimpleNamespace(
        current_generation=3,
        current_episode="episode-3",
        last_sequence=4,
        reset_count=0,
        cross_episode_pollution=0,
    )
    InternVLANav2Shadow._check_barrier(fake, command(5, "episode-5"))
    assert fake.reset_count == 1
    assert fake.cross_episode_pollution == 0


def test_shadow_accepts_same_episode_at_higher_reset_generation() -> None:
    fake = SimpleNamespace(
        current_generation=5,
        current_episode="episode-5",
        last_sequence=8,
        reset_count=0,
        cross_episode_pollution=0,
    )

    InternVLANav2Shadow._check_barrier(fake, command(6, "episode-5"))

    assert fake.reset_count == 1
    assert fake.cross_episode_pollution == 0


def test_shadow_accepts_first_observed_generation_after_warmup_fall() -> None:
    fake = SimpleNamespace(
        current_generation=-1,
        current_episode="",
        last_sequence=-1,
        reset_count=0,
        cross_episode_pollution=0,
    )
    InternVLANav2Shadow._check_barrier(fake, command(2, "episode-2"))
    assert fake.cross_episode_pollution == 0


def test_continuous_wait_accepts_fresh_zero_velocity() -> None:
    fake = SimpleNamespace(
        cmd_wait_timeout=0.1,
        cmd_condition=Condition(),
        latest_cmd=(0.0, 0.0),
        cmd_serial=2,
        latest_cmd_monotonic=time.monotonic(),
    )
    assert ActiveNav2Adapter._wait_for_fresh_cmd(fake, 1) == (0.0, 0.0)


def test_continuous_path_update_can_reuse_fresh_hold_command() -> None:
    fake = SimpleNamespace(
        cmd_wait_timeout=0.1,
        cmd_condition=Condition(),
        latest_cmd=(0.0, 0.0),
        cmd_serial=4,
        latest_cmd_monotonic=time.monotonic(),
    )
    assert ActiveNav2Adapter._wait_for_fresh_cmd(
        fake, 4, require_new=False
    ) == (0.0, 0.0)


def test_continuous_resolver_returns_safe_zero_for_stale_cmd() -> None:
    fake = SimpleNamespace(
        cmd_condition=Condition(),
        latest_cmd=(0.2, 0.4),
        cmd_serial=4,
        latest_cmd_monotonic=time.monotonic() - 1.0,
    )
    assert ActiveNav2Adapter._latest_continuous_cmd(fake) == (0.0, 0.0)


def test_continuous_path_anchor_prefers_fresh_live_odometry() -> None:
    fake = continuous_anchor_adapter(
        odom=(3.0, 4.0, 0.25), odom_age_sec=0.1, allow_fallback=True
    )

    anchor = ActiveNav2Adapter._path_anchor(fake, navigation_command_pose())

    assert anchor == (3.0, 4.0, 0.25, "odom")
    assert fake.command_pose_anchor_fallback_count == 0
    assert fake.current_command_pose_anchor_fallback is False
    assert fake.current_path_anchor_source == "live_navigation_odometry"
    assert fake.current_path_anchor_odom_age_sec == pytest.approx(0.1, abs=0.1)


def test_continuous_path_anchor_uses_validated_command_pose_for_stale_odom() -> None:
    fake = continuous_anchor_adapter(
        odom=(3.0, 4.0, 0.25), odom_age_sec=4.0, allow_fallback=True
    )
    command_pose = navigation_command_pose(
        rotation_wxyz=[2.0 * math.sqrt(0.5), 0.0, 0.0, 2.0 * math.sqrt(0.5)]
    )

    x, y, yaw, frame = ActiveNav2Adapter._path_anchor(fake, command_pose)

    assert (x, y, frame) == (1.25, -0.5, "odom")
    assert yaw == pytest.approx(math.pi / 2.0)
    assert fake.command_pose_anchor_fallback_count == 1
    assert fake.current_command_pose_anchor_fallback is True
    assert fake.current_path_anchor_source == "command_navigation_odometry_fallback"
    assert fake.current_path_anchor_odom_age_sec == pytest.approx(4.0, abs=0.1)


@pytest.mark.parametrize(
    ("odom", "expected_source"),
    [
        (None, "missing_navigation_odometry_rejected"),
        ((3.0, 4.0, 0.25), "stale_navigation_odometry_rejected"),
    ],
)
def test_continuous_path_anchor_remains_fail_closed_when_fallback_disabled(
    odom: tuple[float, float, float] | None, expected_source: str
) -> None:
    fake = continuous_anchor_adapter(
        odom=odom, odom_age_sec=4.0, allow_fallback=False
    )

    with pytest.raises(TimeoutError):
        ActiveNav2Adapter._path_anchor(fake, navigation_command_pose())

    assert fake.command_pose_anchor_fallback_count == 0
    assert fake.current_command_pose_anchor_fallback is False
    assert fake.current_path_anchor_source == expected_source


@pytest.mark.parametrize(
    "command_pose",
    [
        navigation_command_pose(gps=[math.nan, 0.0, 0.0]),
        navigation_command_pose(rotation_wxyz=[0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_continuous_path_anchor_rejects_invalid_command_pose_fallback(
    command_pose: SimpleNamespace,
) -> None:
    fake = continuous_anchor_adapter(
        odom=(3.0, 4.0, 0.25), odom_age_sec=4.0, allow_fallback=True
    )

    with pytest.raises(ValueError, match="invalid command navigation pose fallback"):
        ActiveNav2Adapter._path_anchor(fake, command_pose)

    assert fake.command_pose_anchor_fallback_count == 0
    assert fake.current_command_pose_anchor_fallback is False
    assert (
        fake.current_path_anchor_source
        == "command_navigation_odometry_fallback_invalid"
    )


def test_follow_path_update_does_not_cancel_previous_goal(monkeypatch) -> None:
    old_handle = object()
    new_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: SimpleNamespace(add_done_callback=lambda _callback: None),
    )
    client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: True,
        send_goal_async=lambda _goal: object(),
    )
    published = []
    fake = SimpleNamespace(
        server_timeout=0.1,
        follow_client=client,
        active_goal=old_handle,
        active_path_publisher=SimpleNamespace(publish=published.append),
        _goal_finished=lambda _handle: None,
        _cancel_active=lambda: pytest.fail("path update canceled the active goal"),
    )
    monkeypatch.setattr(active_node_module, "_future_result", lambda _future, _timeout: new_handle)
    path = NavPath()
    ActiveNav2Adapter._send_path(fake, path)
    assert fake.active_goal is new_handle
    assert published == [path]


def test_cancel_accepts_goal_that_reached_terminal_status_during_cancel(
    monkeypatch,
) -> None:
    handle = SimpleNamespace(
        status=GoalStatus.STATUS_SUCCEEDED,
        cancel_goal_async=lambda: object(),
    )
    cleared = []
    fake = SimpleNamespace(
        active_goal=handle,
        last_navigate_goal=(1.0, 2.0, 0.5),
        last_navigate_goal_monotonic=4.0,
        _clear_frozen_system1_target=lambda: cleared.append(True),
    )
    monkeypatch.setattr(
        active_node_module,
        "_future_result",
        lambda _future, _timeout: SimpleNamespace(goals_canceling=[]),
    )

    assert ActiveNav2Adapter._cancel_active(fake, wait=True) is True
    assert fake.active_goal is None
    assert fake.last_navigate_goal is None
    assert fake.last_navigate_goal_monotonic == 0.0
    assert cleared == [True]


def test_cancel_rejects_empty_ack_while_goal_is_still_active(monkeypatch) -> None:
    handle = SimpleNamespace(
        status=GoalStatus.STATUS_EXECUTING,
        cancel_goal_async=lambda: object(),
    )
    fake = SimpleNamespace(
        active_goal=handle,
        last_navigate_goal=(1.0, 2.0, 0.5),
        last_navigate_goal_monotonic=4.0,
        _clear_frozen_system1_target=lambda: pytest.fail(
            "active goal state was cleared without cancellation proof"
        ),
    )
    monkeypatch.setattr(
        active_node_module,
        "_future_result",
        lambda _future, _timeout: SimpleNamespace(goals_canceling=[]),
    )

    with pytest.raises(RuntimeError, match="did not confirm"):
        ActiveNav2Adapter._cancel_active(fake, wait=True)

    assert fake.active_goal is handle
    assert fake.last_navigate_goal == (1.0, 2.0, 0.5)
    assert fake.last_navigate_goal_monotonic == 4.0


def test_completion_sim_cancel_ack_timeout_safe_stops_and_warns() -> None:
    published_motion = []
    published_ack = []
    warnings = []
    cancel_calls = []

    def cancel_active(wait=False, *, preserve_system1_target=False):
        cancel_calls.append((wait, preserve_system1_target))
        if wait:
            raise TimeoutError
        return True

    fake = SimpleNamespace(
        _t5_sim_time_semantics=True,
        stop_ack_publisher=SimpleNamespace(publish=published_ack.append),
        operation_lock=Lock(),
        execution_mode="continuous",
        _publish_motion=published_motion.append,
        active_episode="episode-1",
        active_generation=2,
        last_sequence=3,
        frozen_system1_target=None,
        _cancel_active=cancel_active,
        get_logger=lambda: SimpleNamespace(warning=warnings.append),
    )
    request = SimpleNamespace(
        data=json.dumps(
            {
                "schema_version": 1,
                "token": "stop-1",
                "episode_id": "episode-1",
                "reset_generation": 2,
                "sequence_id": 3,
                "request_id": "request-1",
                "reason": "measured motion timed out",
            }
        )
    )

    ActiveNav2Adapter._on_stop_request(fake, request)

    assert published_motion == [False]
    assert cancel_calls == [(True, False), (False, False)]
    assert len(warnings) == 1
    ack = json.loads(published_ack[0].data)
    assert ack["status"] == "ok"
    assert ack["episode_id"] == "episode-1"
    assert ack["reset_generation"] == 2
    assert ack["sequence_id"] == 3
    assert ack["request_id"] == "request-1"
    assert ack["detail"].startswith("WARN completion_sim")


def test_nav2_plan_is_forwarded_to_recovery_supervisor() -> None:
    published = []
    fake = SimpleNamespace(
        plan_condition=Condition(),
        latest_plan=[],
        latest_plan_monotonic=0.0,
        plan_serial=0,
        active_path_publisher=SimpleNamespace(publish=published.append),
    )
    message = SimpleNamespace(
        poses=[
            SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=1.25, y=-0.5)
                )
            )
        ]
    )

    ActiveNav2Adapter._on_plan(fake, message)

    assert fake.latest_plan == [(1.25, -0.5)]
    assert fake.plan_serial == 1
    assert published == [message]
