import math

from isaac_vln_benchmark.v10_geometry_utils import (
    crawl_turn_command,
    evaluate_target_approach,
    proportional_command,
    update_yaw_polarity,
)
from isaac_vln_benchmark.v10_geometry_utils import analyze_target_episode


def _row(**overrides):
    row = {
        "controller": "waypoint",
        "target_vector_valid": True,
        "target_id_consistent": True,
        "reset_acknowledged": True,
        "distance_decrease_ratio": 0.85,
        "reached_stop_threshold": True,
        "stop_oracle_triggered": True,
        "bridge_received_stop": True,
        "safe_cmd_vel_zero_time": 20.0,
        "semantic_success": True,
        "visible_to_stop_latency_sec": 1.0,
        "max_safe_linear_x_mps": 0.20,
        "fall_or_reset_count": 0,
    }
    row.update(overrides)
    return row


def test_phase1_gate_requires_reset_ack_target_identity_and_low_speed():
    stale = {"runtime_stale_discards": 0, "timebase_error": 0}
    rows = [_row(), _row(), _row(semantic_success=False)]
    assert evaluate_target_approach(rows, {"episodes": [{}, {}, {}]}, stale)["pass"] is True

    rows[0]["reset_acknowledged"] = False
    assert evaluate_target_approach(rows, {"episodes": [{}, {}, {}]}, stale)["pass"] is False
    rows[0]["reset_acknowledged"] = True
    rows[1]["max_safe_linear_x_mps"] = 0.201
    assert evaluate_target_approach(rows, {"episodes": [{}, {}, {}]}, stale)["pass"] is False


def test_phase1_gate_rejects_latency_and_target_mismatch():
    stale = {"runtime_stale_discards": 0, "timebase_error": 0}
    rows = [_row(), _row(), _row()]
    rows[0]["visible_to_stop_latency_sec"] = 2.01
    assert evaluate_target_approach(rows, {"episodes": [{}, {}, {}]}, stale)["pass"] is False
    rows[0]["visible_to_stop_latency_sec"] = 1.0
    rows[2]["target_id_consistent"] = False
    assert evaluate_target_approach(rows, {"episodes": [{}, {}, {}]}, stale)["pass"] is False


def test_proportional_controller_orients_before_translating_with_hysteresis():
    phase, orienting, linear, angular = proportional_command(math.radians(20.0), orienting=False)
    assert (phase, orienting, linear) == ("orient", True, 0.0)
    assert angular > 0.0

    phase, orienting, linear, _ = proportional_command(math.radians(10.0), orienting=True)
    assert (phase, orienting, linear) == ("orient", True, 0.0)
    phase, orienting, linear, _ = proportional_command(math.radians(7.0), orienting=True)
    assert phase == "approach"
    assert orienting is False
    assert 0.0 < linear <= 0.20


def test_crawl_turn_is_visible_motion_only_when_local_path_is_clear():
    phase, orienting, linear, angular = crawl_turn_command(math.radians(28.0), local_costmap_clear=True)
    assert (phase, orienting) == ("orient", True)
    assert linear == 0.12
    assert angular == 0.30

    phase, orienting, linear, angular = crawl_turn_command(
        math.radians(28.0), orienting=True, local_costmap_clear=False
    )
    assert (phase, orienting, linear, angular) == ("orient", True, -0.08, 0.30)


def test_crawl_turn_does_not_drive_forward_on_extreme_heading_error():
    phase, orienting, linear, angular = crawl_turn_command(math.radians(-100.0), local_costmap_clear=True)
    assert (phase, orienting, linear, angular) == ("orient", True, -0.08, -0.30)


def test_crawl_turn_reorients_in_hysteresis_band_when_path_is_blocked():
    phase, orienting, linear, angular = crawl_turn_command(
        math.radians(11.0), orienting=False, local_costmap_clear=False
    )
    assert (phase, orienting, linear, angular) == ("orient", True, -0.08, 0.30)

    phase, orienting, linear, angular = crawl_turn_command(
        math.radians(-11.0), orienting=False, local_costmap_clear=False
    )
    assert (phase, orienting, linear, angular) == ("orient", True, -0.08, -0.30)

    phase, orienting, linear, angular = crawl_turn_command(
        math.radians(8.0), orienting=False, local_costmap_clear=False
    )
    assert phase == "orient"
    assert orienting is True
    assert linear == -0.08
    assert angular == 0.0


def test_yaw_polarity_flips_only_after_repeated_opposite_response():
    polarity, mismatches, flipped = update_yaw_polarity(1.0, 0, 0.15, -0.02)
    assert (polarity, mismatches, flipped) == (1.0, 1, False)
    polarity, mismatches, flipped = update_yaw_polarity(polarity, mismatches, 0.15, -0.02)
    assert (polarity, mismatches, flipped) == (-1.0, 0, True)

    polarity, mismatches, flipped = update_yaw_polarity(polarity, mismatches, 0.15, 0.02)
    assert (polarity, mismatches, flipped) == (-1.0, 0, False)


def test_commanded_reset_ack_is_not_counted_as_runtime_reset():
    episode = {"episode_id": "ep", "mode": "waypoint", "task_id": "semantic_seed0", "success": False}
    task = {"target_object": "target"}
    scene = {"scene_type": "micro"}
    trace = [
        {
            "controller": "waypoint",
            "phase": "approach",
            "distance_m": 4.0,
            "target_id": "target",
            "success_judge_target_id": "target",
            "telemetry_event": "reset",
            "telemetry_episode_id": "ep",
            "robot_fallen_or_unstable": False,
        },
        {
            "controller": "waypoint",
            "phase": "approach",
            "distance_m": 3.9,
            "target_id": "target",
            "success_judge_target_id": "target",
            "telemetry_event": "pose",
            "telemetry_episode_id": "ep",
            "robot_fallen_or_unstable": False,
        },
    ]
    row = analyze_target_episode(episode, task, scene, [{"event": "reset_ack", "t": 0.0}], trace)
    assert row["fall_or_reset_count"] == 0

    trace[1]["telemetry_event"] = "reset"
    trace[1]["telemetry_episode_id"] = ""
    row = analyze_target_episode(episode, task, scene, [{"event": "reset_ack", "t": 0.0}], trace)
    assert row["fall_or_reset_count"] == 1
