from __future__ import annotations

from pathlib import Path

from scripts.t5_live_frontier_snapshot_node import _resolve_runtime_topics


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run_t5_live_frontier_capture.sh"
NODE = ROOT / "scripts/t5_live_frontier_snapshot_node.py"
LANE = ROOT / "scripts/run_t5_dgx_lane.sh"
FAST = ROOT / "coordination/run_t5_fast_lane_online.sh"
PROBE = ROOT / "scripts/probe_t5_live_frontier_capture_ros.py"
INTERNVLA_CLIENT = ROOT / "internvla_ros2/internvla_ros2/client_node.py"


def test_capture_wrapper_enters_only_the_leased_lane_b_ros_overlay() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert 'test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = lane-b' in text
    assert 'test "${INTERNNAV_T5_LANE:-}" = b' in text
    assert 'test "$namespace" = /t5/lane_b' in text
    assert 'test "${ROS_DOMAIN_ID:-}" = 76' in text
    assert "source /opt/ros/jazzy/setup.bash" in text
    assert 'source "$ros_ws/install/setup.bash"' in text
    assert "--dependency-check-only" in text
    assert 'exec python3 -u "$root/scripts/t5_live_frontier_snapshot_node.py"' in text
    assert '--namespace "$namespace"' in text
    assert '--output "$snapshot" --status-output "$status"' in text


def test_capture_node_drops_publishable_state_for_any_bad_identity() -> None:
    text = NODE.read_text(encoding="utf-8")
    exception = text.split(
        "except (LiveFrontierCaptureError, TypeError, ValueError) as exc:", 1
    )[1].split("self._record_status(", 1)[0]
    assert "self._metadata = None" in exception
    assert "self._capture.clear_for_missing_identity()" in exception
    assert "self._clear_sensor_inputs()" in exception
    assert "STALE_CAPTURE_IDENTITY" not in exception
    assert "create_publisher" not in text
    assert "ActionClient" not in text


def test_capture_node_opens_an_input_barrier_before_binding_a_new_epoch() -> None:
    text = NODE.read_text(encoding="utf-8")
    callback = text.split("def _on_metadata", 1)[1].split(
        "def _clear_sensor_inputs", 1
    )[0]
    assert "epoch = (identity.episode_id, identity.reset_id)" in callback
    assert "if epoch != self._identity_epoch:" in callback
    assert callback.index("self._clear_sensor_inputs()") < callback.index(
        "self._capture.observe_identity(identity)"
    )

    clear = text.split("def _clear_sensor_inputs", 1)[1].split(
        "def _on_odometry", 1
    )[0]
    assert "self._odometry = None" in clear
    assert "self._costmaps.clear()" in clear
    assert "self._tf_buffer.clear()" in clear
    assert '"snapshot_ready_count": self._snapshot_ready_count' in text
    assert "self._snapshot_ready_count += 1" in text


def test_dgx_lane_supervises_capture_as_a_zero_authority_sidecar() -> None:
    text = LANE.read_text(encoding="utf-8")
    assert (
        'live_frontier_capture="${INTERNNAV_T5_LIVE_FRONTIER_CAPTURE:-$step3_live_advisor}"'
        in text
    )
    assert 'test "$lane" = b' in text
    assert 'setsid env "${common_env[@]}" INTERNNAV_T5_RESOURCE_LEASE_ACK=lane-b' in text
    assert 'bash "$root/scripts/run_t5_live_frontier_capture.sh"' in text
    assert "record_pid_event live_frontier" in text
    assert "stop_group live_frontier" in text
    assert 'rm -f "$result_dir/live_frontier/current.json"' in text
    assert 'leader_is_alive "$live_frontier_pid"' in text
    assert 'T5_LANE_B_LIVE_FRONTIER_PATH="$result_dir/live_frontier/current.json"' in text
    assert text.count('"live_frontier_capture": {') >= 3
    assert '"snapshot_ready_count": int(' in text
    assert '"runtime_status": (live_frontier_status or {}).get("status")' in text
    for authority in (
        "motion_authority",
        "terminal_stop_authority",
        "goal_authority",
        "model_request_authority",
    ):
        assert authority in text


def test_fast_lane_propagates_explicit_capture_opt_in_without_new_protocol() -> None:
    text = FAST.read_text(encoding="utf-8")
    assert 'INTERNNAV_T5_LIVE_FRONTIER_CAPTURE: 0 | 1' in text
    assert (
        'live_frontier_capture="${INTERNNAV_T5_LIVE_FRONTIER_CAPTURE:-$step3_live_advisor}"'
        in text
    )
    assert 'test "$live_frontier_capture" != 1 || test "$lane" = b || usage' in text
    assert text.count(
        'INTERNNAV_T5_LIVE_FRONTIER_CAPTURE="$live_frontier_capture"'
    ) >= 2
    assert "live_frontier_capture=\"${16}\"" in text
    assert '"live_frontier_capture":live_frontier_capture' in text
    assert '"live_frontier_status":live_frontier_status' in text


def test_only_dedicated_capture_canary_requires_a_ready_snapshot() -> None:
    text = FAST.read_text(encoding="utf-8")
    scope = text.split("live_frontier_ready_required=0", 1)[1].split(
        "strict_extension_profile=", 1
    )[0]
    for condition in (
        'test "$lane" = b',
        'test "$profile" = canary60',
        'test "$isaac_sensor_profile" = baseline',
        'test "$step3_live_advisor" = 0',
        'test "$live_frontier_capture" = 1',
    ):
        assert condition in scope
    assert '"live_frontier_ready_gate": (not live_frontier_ready_required) or (' in text
    assert 'int(live_frontier_status.get("snapshot_ready_count",0)) >= 1' in text
    assert '"live_frontier_ready_required":live_frontier_ready_required' in text


def test_wiring_does_not_change_shared_interfaces_or_gain_control_authority() -> None:
    changed_runtime = "\n".join(
        path.read_text(encoding="utf-8") for path in (WRAPPER, NODE, LANE, FAST)
    )
    assert "internnav_t5_lane_b_msgs" not in WRAPPER.read_text(encoding="utf-8")
    assert "PrepareFrontiers" not in changed_runtime
    assert "CommitFrontier" not in changed_runtime
    assert "create_publisher" not in NODE.read_text(encoding="utf-8")
    assert "ActionClient" not in NODE.read_text(encoding="utf-8")


def test_ros_probe_publishes_only_synthetic_inputs_and_checks_v1_reset_stale() -> None:
    text = PROBE.read_text(encoding="utf-8")
    for allowed_topic in (
        '"/clock"',
        '"/t5/lane_b/tf"',
        '"/internvla/observation/metadata"',
        '"/odom"',
        '"/t5/lane_b/local_costmap/costmap"',
        '"/t5/lane_b/global_costmap/costmap"',
    ):
        assert allowed_topic in text
    for forbidden in ("cmd_vel", "NavigateToPose", "ActionClient"):
        assert forbidden not in text
    assert '"b::probe-episode::0::0"' in text
    assert '"b::probe-episode::1::0"' in text
    assert '"STALE_CAPTURE_IDENTITY"' in text
    assert 'checks["reset_metadata_cleared_pre_reset_inputs"]' in text
    assert 'barrier.get("blocker_code") == "MISSING_ODOMETRY"' in text
    assert 'checks["snapshot_ready_count_persisted"]' in text


def test_capture_subscribes_to_the_frozen_production_topic_scopes() -> None:
    text = NODE.read_text(encoding="utf-8")
    production = INTERNVLA_CLIENT.read_text(encoding="utf-8")
    assert 'ObservationMetadata, "/internvla/observation/metadata", image_qos' in production
    assert 'self.create_publisher(Odometry, "/odom", image_qos)' in production
    assert '_ROOT_SCOPED_PRODUCTION_INPUTS = frozenset({"metadata", "odometry"})' in text
    assert 'resolved[name] = f"/{relative.strip(\'/\')}"' in text
    assert "resolved[name] = lane_topic" in text
    assert _resolve_runtime_topics(
        "/t5/lane_b",
        {
            "metadata": "internvla/observation/metadata",
            "odometry": "odom",
            "local_costmap": "local_costmap/costmap",
            "global_costmap": "global_costmap/costmap",
        },
    ) == {
        "metadata": "/internvla/observation/metadata",
        "odometry": "/odom",
        "local_costmap": "/t5/lane_b/local_costmap/costmap",
        "global_costmap": "/t5/lane_b/global_costmap/costmap",
    }
