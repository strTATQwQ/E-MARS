import json

from isaac_vln_benchmark.v10_geometry_utils import (
    branch_membership,
    evaluate_branch_controller,
    evaluate_target_approach,
    route_geometry_audit,
    target_pose_audit,
    target_vector,
    write_common_v10_artifacts,
    write_sim2real_gate_v10,
)


def test_v10_route_geometry_maps_left_right_polygons():
    audit = route_geometry_audit()
    scene = {
        "semantic_zones": [{"class": "intersection", "center": audit["intersection_center"]}],
    }

    assert audit["direction_mapping_valid"] is True
    assert audit["turn_left_angular_z_sign"] == 1
    assert audit["turn_right_angular_z_sign"] == -1
    assert branch_membership([4.5, 0.8, 0.0], scene) == {"left": True, "right": False}
    assert branch_membership([4.5, -0.8, 0.0], scene) == {"left": False, "right": True}


def test_v10_target_pose_audit_uses_metrics_visibility_signature():
    audit = target_pose_audit()

    assert audit["pass"] is True
    assert audit["target_id"] == "fire_extinguisher_1"
    assert audit["line_of_sight"] is True
    assert audit["distance_2d_m"] > 2.5


def test_v10_target_vector_reports_desired_yaw_and_distance():
    vec = target_vector([0.0, 0.0, 0.0], [3.0, 4.0, 0.0])

    assert round(vec["distance_2d_m"], 3) == 5.0
    assert round(vec["desired_yaw_rad"], 3) == 0.927
    assert round(vec["yaw_error_rad"], 3) == 0.927


def test_v10_branch_gate_requires_five_of_six_and_left_right_balance():
    rows = []
    for index in range(6):
        branch = "left" if index < 3 else "right"
        rows.append(
            {
                "mode": "v10_branch_yaw75_fwd08",
                "instruction_branch": branch,
                "decision_published": True,
                "rotate_phase_entered": True,
                "post_turn_forward_executed": True,
                "entered_correct_branch": index != 5,
            }
        )
    stale = {"runtime_stale_discards": 0, "timebase_error": 0}

    result = evaluate_branch_controller(rows, {"episodes": [{} for _ in rows]}, stale)

    assert result["pass"] is True
    rows[0]["entered_correct_branch"] = False
    assert evaluate_branch_controller(rows, {"episodes": [{} for _ in rows]}, stale)["pass"] is False


def test_v10_semantic_gate_requires_threshold_stop_and_two_successes():
    rows = []
    for index in range(3):
        rows.append(
            {
                "controller": "rotate_then_forward",
                "target_vector_valid": True,
                "target_id_consistent": True,
                "reset_acknowledged": True,
                "distance_decrease_ratio": 0.9,
                "reached_stop_threshold": True,
                "stop_oracle_triggered": True,
                "bridge_received_stop": True,
                "safe_cmd_vel_zero_time": 3.2,
                "semantic_success": index < 2,
                "visible_to_stop_latency_sec": 1.2,
                "max_safe_linear_x_mps": 0.2,
                "fall_or_reset_count": 0,
            }
        )
    stale = {"runtime_stale_discards": 0, "timebase_error": 0}

    result = evaluate_target_approach(rows, {"episodes": [{} for _ in rows]}, stale)

    assert result["pass"] is True
    rows[0]["reached_stop_threshold"] = False
    assert evaluate_target_approach(rows, {"episodes": [{} for _ in rows]}, stale)["pass"] is False


def test_v10_common_artifacts_and_gate_are_written(tmp_path):
    out = tmp_path / "v10"
    out.mkdir()
    (out / "events.jsonl").write_text("", encoding="utf-8")
    metrics = {"episodes": []}
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    stale = write_common_v10_artifacts(out, metrics, geometry_audit=route_geometry_audit())
    gate = write_sim2real_gate_v10(out, route={"pass": True}, semantic={"pass": True})

    assert stale["runtime_stale_discards"] == 0
    assert gate["ready_for_v11"] is True
    for name in [
        "summary.md",
        "geometry_audit.json",
        "branch_confusion_matrix.csv",
        "target_approach_metrics.csv",
        "controller_trace.csv",
        "trajectory.csv",
        "stale_attribution.csv",
        "failure_table.csv",
        "visual/viewport.png",
        "sim2real_gate_v10.md",
    ]:
        if name == "summary.md":
            continue
        assert (out / name).exists()
