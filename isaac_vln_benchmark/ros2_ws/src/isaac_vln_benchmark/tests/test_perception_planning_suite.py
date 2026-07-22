from isaac_vln_benchmark.perception_planning_suite import (
    audit_step_visible_request,
    build_controlled_visual_cases,
    build_mp3d_visual_cases,
    build_planning_cases,
    build_tracking_sequences,
    evaluate_planning,
    evaluate_robustness,
    evaluate_tracking_sequences,
    evaluate_visual_cases,
    planning_gate,
    planning_rows_from_micro,
    mp3d_background_transform,
    reset_payload_for_visual_case,
    step_visible_request,
    tracking_gate,
    tracking_visual_case,
    visual_gate,
    visual_rows_from_micro,
)


def test_case_manifests_have_required_counts_and_no_visible_oracle_context():
    controlled = build_controlled_visual_cases()
    mp3d = build_mp3d_visual_cases()
    tracking = build_tracking_sequences()
    planning = build_planning_cases()

    assert len(controlled) == 60
    assert sum(row["role"] == "route_choice" for row in controlled) == 24
    assert sum(row["role"] == "semantic_stop" for row in controlled) == 36
    assert len(mp3d) == 30
    assert len(tracking) == 30
    assert all(8 <= len(row["frames"]) <= 20 for row in tracking)
    assert all(row["frames"][2]["frame_seq"] == row["frames"][1]["frame_seq"] for row in tracking)
    assert all(row["frames"][2]["visible"] == row["frames"][1]["visible"] for row in tracking)
    assert all(
        sum(bool(frame["occluded"]) for frame in row["frames"]) == 3
        for row in tracking
        if row["family"] == "temporary_occlusion"
    )
    moving = next(row for row in tracking if row["family"] == "moving_camera")
    hidden_case = tracking_visual_case(moving, moving["frames"][0])
    assert hidden_case["target_present"] is True
    assert hidden_case["expected"] is True
    assert hidden_case["camera_yaw_deg"] == 0.0
    assert hidden_case["camera_pose"] == [0.0, 1.0, 0.0]
    occluded = next(row for row in tracking if row["family"] == "temporary_occlusion")
    occluded_frame = next(frame for frame in occluded["frames"] if frame["occluded"])
    occluded_case = tracking_visual_case(occluded, occluded_frame)
    assert occluded_case["target_present"] is True
    assert occluded_case["occlusion_fraction"] == 1.0
    occluded_reset = reset_payload_for_visual_case(occluded_case, episode_id="occluded_episode")
    full_occluder = next(row for row in occluded_reset["obstacles"] if row["id"] == "visual_occluder")
    assert full_occluder["pose"][2] == 3.0
    assert full_occluder["size"] == [0.5, 5.0, 6.0]
    assert full_occluder["color"] == "black"
    assert len(planning["route"]) == 30
    assert all(
        row["task_id"] == ("turn_001" if row["expected"] == "left" else "turn_002")
        for row in planning["route"]
    )
    assert len(planning["semantic"]) == 30
    assert all(audit_step_visible_request(step_visible_request(row))["pass"] for row in controlled + mp3d)
    semantic = [row for row in controlled if row["role"] == "semantic_stop"]
    for target in {row["target"] for row in semantic}:
        positives = [row for row in semantic if row["target"] == target and row["expected"]]
        negatives = [row for row in semantic if row["target"] == target and not row["expected"]]
        assert len(positives) == 6
        assert len(negatives) == 6
        assert {row["visual_variant"] for row in positives} == {
            "clear", "similar_color", "partial_occlusion", "dim", "camera_yaw", "far_scale"
        }

    reset = reset_payload_for_visual_case(controlled[0], episode_id="episode_a")
    assert reset["episode_id"] == "episode_a"
    assert reset["scene_id"] == "intersection_001"
    assert reset["visual_overlay_only"] is True
    assert len(reset["objects"]) == 2

    moving_reset = reset_payload_for_visual_case(hidden_case, episode_id="moving_episode")
    assert moving_reset["pose"] == [0.0, 1.0, 0.0]

    semantic_cases = planning["semantic"]
    positive_reset = reset_payload_for_visual_case(semantic_cases[0], episode_id="semantic_positive")
    negative_reset = reset_payload_for_visual_case(semantic_cases[-1], episode_id="semantic_negative")
    assert next(row for row in positive_reset["objects"] if row["id"] == "fire_extinguisher_1")["pose"][0] == 3.5
    assert next(row for row in negative_reset["objects"] if row["id"] == "fire_extinguisher_1")["pose"][:2] == [100.0, 100.0]
    assert reset["objects"][0]["pose"][1] > 0.0

    mp3d_case = build_mp3d_visual_cases(["17DRP5sb8fy/scene.usd"])[0]
    mp3d_case["scene_usd_path"] = "/data/mp3d_pe/17DRP5sb8fy/scene.usd"
    mp3d_reset = reset_payload_for_visual_case(mp3d_case, episode_id="episode_mp3d")
    assert mp3d_reset["background_usd"] == "/data/mp3d_pe/17DRP5sb8fy/scene.usd"
    assert mp3d_reset["background_visual_only"] is True
    assert "background_usd" not in step_visible_request(mp3d_case)

    static_route = next(row for row in planning["route"] if row["scene_family"] == "static_obstacle")
    static_reset = reset_payload_for_visual_case(static_route, episode_id="planning_static")
    assert any(row["id"] == "planning_static_obstacle" for row in static_reset["obstacles"])
    distractor_route = next(row for row in planning["route"] if row["scene_family"] == "visual_distractor")
    distractor_reset = reset_payload_for_visual_case(distractor_route, episode_id="planning_distractor")
    assert any(row["id"] == "planning_visual_distractor" for row in distractor_reset["objects"])


def test_request_audit_rejects_nested_oracle_fields():
    result = audit_step_visible_request({"event": {"oracle_visibility": True}, "expected_route": "left"})
    assert result["pass"] is False
    assert result["oracle_context_leakage"] == 2


def test_mp3d_background_transform_places_nearest_x_behind_injected_props():
    transform = mp3d_background_transform(
        {"min": [-8.0, -4.0, -3.0], "center": [1.0, 2.0, 0.5]}, scale=0.45
    )
    assert transform["background_pose"][0] + -8.0 * 0.45 == 5.0
    assert transform["background_pose"][1] + 2.0 * 0.45 == 0.0
    assert transform["background_pose"][2] + 0.5 * 0.45 == 1.5


def test_visual_gate_passes_threshold_metrics():
    rows = []
    for index in range(12):
        expected = "left" if index % 2 == 0 else "right"
        rows.append({"role": "route_choice", "expected": expected, "predicted": expected, "fresh_image": True, "target": "landmark", "step_latency_sec": 1.0})
    for target in ("fire extinguisher", "yellow hydrant"):
        for _ in range(5):
            rows.append({"role": "semantic_stop", "expected": True, "predicted": True, "fresh_image": True, "target": target, "step_latency_sec": 1.2})
            rows.append({"role": "semantic_stop", "expected": False, "predicted": False, "fresh_image": True, "target": target, "step_latency_sec": 1.1})
    assert visual_gate(evaluate_visual_cases(rows), domain="controlled")["pass"] is True


def test_step_request_latency_gate_allows_up_to_five_seconds():
    def passing_rows(latency: float):
        rows = []
        for index in range(12):
            expected = "left" if index % 2 == 0 else "right"
            rows.append(
                {
                    "role": "route_choice",
                    "expected": expected,
                    "predicted": expected,
                    "fresh_image": True,
                    "target": "landmark",
                    "step_latency_sec": latency,
                }
            )
        for _ in range(5):
            rows.append(
                {
                    "role": "semantic_stop",
                    "expected": True,
                    "predicted": True,
                    "fresh_image": True,
                    "target": "traffic cone",
                    "step_latency_sec": latency,
                }
            )
            rows.append(
                {
                    "role": "semantic_stop",
                    "expected": False,
                    "predicted": False,
                    "fresh_image": True,
                    "target": "traffic cone",
                    "step_latency_sec": latency,
                }
            )
        return rows

    assert visual_gate(evaluate_visual_cases(passing_rows(4.9)), domain="controlled")["pass"] is True
    failed_gate = visual_gate(evaluate_visual_cases(passing_rows(5.1)), domain="controlled")
    assert failed_gate["pass"] is False
    assert "Step request p95 latency > 5 sec" in failed_gate["failures"]


def test_mp3d_visual_gate_requires_a_detected_background_for_every_row():
    rows = []
    for index in range(12):
        expected = "left" if index % 2 == 0 else "right"
        rows.append(
            {
                "role": "route_choice",
                "expected": expected,
                "predicted": expected,
                "fresh_image": True,
                "target": "landmark",
                "step_latency_sec": 1.0,
                "background_usd": f"/scene/{index}.usd",
                "background_visual_detected": True,
            }
        )
    for index in range(18):
        expected = index % 2 == 0
        rows.append(
            {
                "role": "semantic_stop",
                "expected": expected,
                "predicted": expected,
                "fresh_image": True,
                "target": "traffic cone",
                "step_latency_sec": 1.0,
                "background_usd": f"/scene/{index}.usd",
                "background_visual_detected": True,
            }
        )
    assert visual_gate(evaluate_visual_cases(rows), domain="mp3d_pe")["pass"] is True
    rows[0]["background_visual_detected"] = False
    assert visual_gate(evaluate_visual_cases(rows), domain="mp3d_pe")["pass"] is False


def test_robustness_uses_five_second_step_request_latency_gate():
    profiles = (
        "delay", "drop", "duplicate", "out_of_order", "reset",
        "episode_mismatch", "timestamp_mismatch", "horizontal_flip",
    )
    rows = [
        {
            "profile": profile,
            "pass": True,
            "step_latency_sec": 4.9,
            "old_track_cleared": True if profile == "reset" else None,
        }
        for profile in profiles
    ]

    assert evaluate_robustness(rows)["pass"] is True
    rows[-1]["step_latency_sec"] = 5.2
    rows[-2]["step_latency_sec"] = 5.2
    assert evaluate_robustness(rows)["pass"] is False


def test_visual_micro_conversion_requires_accepted_fresh_multimodal_call():
    rows = visual_rows_from_micro(
        [
            {
                "case_id": "c1",
                "domain": "controlled",
                "role": "route_choice",
                "target": "red cone",
                "expected": "left",
                "response": {"route_choice": "left", "step_latency_sec": 1.0},
                "reset_ack_received": True,
                "post_reset_images_ready": True,
                "observation_responses": [{"multimodal": True, "image_snapshot": {"age_sec": 0.1}}],
                "step_http_events": [{"result": "accepted", "latency_s": 1.0}],
                "request_audits": [{"oracle_context_leakage": 0}],
            }
        ]
    )

    assert rows[0]["predicted"] == "left"
    assert rows[0]["fresh_image"] is True
    assert rows[0]["parse_error"] is False


def test_tracking_gate_detects_duplicate_hit_and_stale_action():
    rows = [
        {"sequence_id": "s1", "episode_id": "e1", "target": "box", "track_episode_id": "e1", "track_target": "box", "frame_seq": 1, "hits": 1, "visible_truth": True},
        {"sequence_id": "s1", "episode_id": "e1", "target": "box", "track_episode_id": "e1", "track_target": "box", "frame_seq": 1, "hits": 2, "visible_truth": True, "action_triggered": True, "track_stale": True},
    ]
    result = tracking_gate(evaluate_tracking_sequences(rows))
    assert result["pass"] is False
    assert result["metrics"]["duplicate_frame_miscounts"] == 1
    assert result["metrics"]["stale_track_action_count"] == 1


def test_tracking_false_confirmation_requires_a_visible_false_observation():
    rows = [
        {
            "sequence_id": "s1",
            "episode_id": "e1",
            "target": "box",
            "track_episode_id": "e1",
            "track_target": "box",
            "frame_seq": 1,
            "hits": 2,
            "confirmed": True,
            "track_visible": True,
            "visible_truth": True,
        },
        {
            "sequence_id": "s1",
            "episode_id": "e1",
            "target": "box",
            "track_episode_id": "e1",
            "track_target": "box",
            "frame_seq": 2,
            "hits": 0,
            "confirmed": True,
            "track_visible": False,
            "visible_truth": False,
        },
    ]
    metrics = evaluate_tracking_sequences(rows)
    assert metrics["confirmed_false_target_rate"] == 0.0
    assert metrics["occlusion_reacquired_sequences"] == 0


def test_planning_gate_accepts_complete_normal_chain_results():
    route = []
    for index in range(30):
        side = "left" if index % 2 == 0 else "right"
        route.append({"expected": side, "entered_correct_branch": True, "decision_confirmed": True, "normal_chain_complete": True})
    semantic = []
    for index in range(30):
        positive = index < 15
        semantic.append(
            {
                "expected": positive,
                "entered_coverage": positive,
                "success": positive,
                "stop": positive,
                "normal_chain_complete": True,
                "threshold_to_safe_stop_sec": 1.5 if positive else None,
                "approach_steps": [{"distance_before": 3.0, "distance_after": 2.9}] if positive else [],
            }
        )
    assert planning_gate(evaluate_planning(route, semantic))["pass"] is True


def test_planning_micro_conversion_keeps_chain_latency_and_safety_evidence():
    converted = planning_rows_from_micro(
        [
            {
                "role": "semantic_stop",
                "episode_id": "semantic_a",
                "expected": True,
                "response": {"target_visible": True},
                "coverage_status": {"distance_to_target": 2.4},
                "semantic_success": True,
                "distance_m_used": 1.9,
                "threshold_to_safe_stop_sec": 1.2,
                "coverage_to_stop_sec": 4.0,
                "approach_steps": [{"distance_before": 2.4, "distance_after": 2.3}],
                "case_primitives": [{"primitive": "follow_waypoint"}, {"primitive": "stop"}],
                "primitives_after_response": [{"primitive": "stop"}],
                "case_safe_cmds": [{"linear_x": 0.2, "angular_z": 0.0}, {"linear_x": 0.0, "angular_z": 0.0}],
                "case_metrics": [],
            }
        ]
    )["semantic"][0]

    assert converted["entered_coverage"] is True
    assert converted["normal_chain_complete"] is True
    assert converted["threshold_to_safe_stop_sec"] == 1.2
    assert converted["collision"] == 0


def test_unconfirmed_route_with_controller_primitive_is_not_a_chain_bypass():
    converted = planning_rows_from_micro(
        [
            {
                "role": "route_choice",
                "episode_id": "route_scan",
                "expected": "right",
                "response": {"route_choice": "front"},
                "case_primitives": [{"primitive": "enter_branch", "phase": "advance"}],
                "primitives_after_response": [],
                "case_safe_cmds": [{"linear_x": 0.2, "angular_z": 0.0}],
                "entered_correct_branch": False,
            }
        ]
    )["route"][0]

    assert converted["decision_confirmed"] is False
    assert converted["normal_chain_complete"] is True
