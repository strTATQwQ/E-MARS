import math

import pytest

from omninav_step_scheduler.sensor_only_planning import (
    CameraIntrinsics,
    SensorRouteController,
    SpatialTargetTrackState,
    instruction_visual_target,
    payload_oracle_fields,
    project_depth_pixel,
    semantic_track_primitive,
    validate_target_observation,
)


def observation(seq: int, *, visible: bool = True, distance: float = 2.4, bearing: float = 0.1):
    return {
        "episode_id": "episode_a",
        "target_id": "fire extinguisher",
        "frame_seq": seq,
        "source_stamp_sec": float(seq),
        "visible": visible,
        "confidence": 0.92 if visible else 0.0,
        "distance_m": distance if visible else None,
        "bearing_rad": bearing if visible else None,
    }


def test_camera_projection_has_ros_left_positive_bearing():
    intrinsics = CameraIntrinsics.from_camera_info(
        {"width": 256, "height": 144, "k": [300.0, 0.0, 127.5, 0.0, 300.0, 71.5, 0.0, 0.0, 1.0]}
    )
    center = project_depth_pixel(127.5, 71.5, 2.0, intrinsics)
    left = project_depth_pixel(80.0, 71.5, 2.0, intrinsics)
    right = project_depth_pixel(180.0, 71.5, 2.0, intrinsics)
    assert center["bearing_rad"] == pytest.approx(0.0)
    assert left["bearing_rad"] > 0.0
    assert right["bearing_rad"] < 0.0


def test_spatial_tracker_requires_two_distinct_frames_and_reacquires():
    tracker = SpatialTargetTrackState(required_hits=2, max_misses=1)
    first = tracker.update(observation(1), timestamp=1.0)
    duplicate = tracker.update(observation(1), timestamp=1.1)
    second = tracker.update(observation(2), timestamp=2.0)
    assert first["confirmed"] is False
    assert duplicate["hits"] == 1
    assert duplicate["duplicate_frames"] == 1
    assert second["confirmed"] is True

    tracker.update(observation(3, visible=False), timestamp=3.0)
    lost = tracker.update(observation(4, visible=False), timestamp=4.0)
    assert lost["confirmed"] is False
    tracker.update(observation(5), timestamp=5.0)
    reacquired = tracker.update(observation(6), timestamp=6.0)
    assert reacquired["confirmed"] is True
    assert reacquired["reacquired_count"] == 1


def test_confirmed_track_uses_lower_retention_threshold_without_single_frame_confirmation():
    tracker = SpatialTargetTrackState(required_hits=2, min_confidence=0.60, retain_confidence=0.25)
    assert tracker.update(observation(1), timestamp=1.0)["confirmed"] is False
    assert tracker.update(observation(2), timestamp=2.0)["confirmed"] is True
    lower = observation(3)
    lower["confidence"] = 0.32
    summary = tracker.update(lower, timestamp=3.0)
    assert summary["confirmed"] is True
    assert summary["hits"] == 3


def test_spatial_tracker_rejects_target_switch_distance_jump():
    tracker = SpatialTargetTrackState(required_hits=2, max_distance_jump_m=1.0)
    tracker.update(observation(1, distance=5.5), timestamp=1.0)
    confirmed = tracker.update(observation(2, distance=5.4), timestamp=2.0)
    switched = tracker.update(observation(3, distance=1.4), timestamp=3.0)
    assert confirmed["confirmed"] is True
    assert switched["update_result"] == "miss"
    assert switched["distance_m"] == 5.4
    assert switched["discontinuity_rejections"] == 1


def test_unconfirmed_false_detection_does_not_lock_out_later_real_target():
    tracker = SpatialTargetTrackState(required_hits=2, max_distance_jump_m=1.0)
    false_first = tracker.update(observation(1, distance=1.2, bearing=-0.4), timestamp=1.0)
    real_first = tracker.update(observation(2, distance=5.0, bearing=0.3), timestamp=2.0)
    real_second = tracker.update(observation(3, distance=4.9, bearing=0.29), timestamp=3.0)
    assert false_first["confirmed"] is False
    assert real_first["confirmed"] is False
    assert real_first["hits"] == 1
    assert real_first["provisional_switches"] == 1
    assert real_second["confirmed"] is True
    assert real_second["distance_m"] == 4.9


def test_spatial_tracker_compensates_camera_bearing_with_odom_yaw():
    tracker = SpatialTargetTrackState(required_hits=2, max_bearing_jump_deg=10.0)
    first = observation(1, distance=4.0, bearing=math.radians(40.0))
    first["observer_pose"] = [0.0, 0.0, 0.0]
    target_x = 4.0 * math.cos(math.radians(40.0))
    target_y = 4.0 * math.sin(math.radians(40.0))
    second_distance = math.hypot(target_x - 1.0, target_y)
    second_bearing = math.atan2(target_y, target_x - 1.0) - math.radians(40.0)
    second = observation(2, distance=second_distance, bearing=second_bearing)
    second["observer_pose"] = [1.0, 0.0, math.radians(40.0)]
    tracker.update(first, timestamp=1.0)
    confirmed = tracker.update(second, timestamp=2.0)
    assert confirmed["confirmed"] is True
    assert confirmed["discontinuity_rejections"] == 0
    assert confirmed["world_x_m"] == pytest.approx(target_x, abs=0.03)
    assert confirmed["world_y_m"] == pytest.approx(target_y, abs=0.03)


def test_spatial_tracker_selects_continuous_candidate_over_higher_score_distractor():
    tracker = SpatialTargetTrackState(required_hits=2)
    first = observation(1, distance=5.5, bearing=0.3)
    first["observer_pose"] = [0.0, 0.0, 0.0]
    tracker.update(first, timestamp=1.0)
    confirming = observation(2, distance=5.4, bearing=0.28)
    confirming["observer_pose"] = [0.1, 0.0, 0.02]
    assert tracker.update(confirming, timestamp=2.0)["confirmed"] is True
    second = observation(3, distance=1.4, bearing=-0.4)
    second["observer_pose"] = [0.2, 0.0, 0.05]
    second["candidates"] = [
        {
            "candidate_index": 0,
            "visible": True,
            "confidence": 0.95,
            "distance_m": 1.4,
            "bearing_rad": -0.4,
            "mask_sha256": "distractor",
        },
        {
            "candidate_index": 1,
            "visible": True,
            "confidence": 0.70,
            "distance_m": 5.3,
            "bearing_rad": 0.25,
            "mask_sha256": "target",
            "candidate_source": "sam2_video_propagation",
        },
    ]
    selected = tracker.update(second, timestamp=3.0)
    assert selected["confirmed"] is True
    assert selected["selected_candidate_index"] == 1
    assert selected["mask_sha256"] == "target"
    assert selected["candidate_source"] == "sam2_video_propagation"
    assert selected["distance_m"] == 5.3


def test_episode_or_target_change_resets_track_without_cross_talk():
    tracker = SpatialTargetTrackState()
    tracker.update(observation(1), timestamp=1.0)
    other = observation(2)
    other["episode_id"] = "episode_b"
    summary = tracker.update(other, timestamp=2.0)
    assert summary["episode_id"] == "episode_b"
    assert summary["hits"] == 1
    assert summary["confirmed"] is False


def test_semantic_planner_stops_for_stale_track_and_holds_at_threshold():
    stale = semantic_track_primitive({"confirmed": False, "fresh": False, "visible": False})
    assert stale["primitive"] == "stop"
    near = semantic_track_primitive(
        {"confirmed": True, "fresh": True, "visible": True, "distance_m": 1.95, "bearing_rad": 0.0}
    )
    assert near["primitive"] == "stop"
    assert near["phase"] == "await_step_stop"
    far = semantic_track_primitive(
        {"confirmed": True, "fresh": True, "visible": True, "distance_m": 3.0, "bearing_rad": 0.05}
    )
    assert far["primitive"] == "target_relative_approach"
    assert 0.0 < far["linear_x_mps"] <= 0.20


def route_track(distance: float, bearing_deg: float) -> dict:
    return {
        "confirmed": True,
        "fresh": True,
        "visible": True,
        "distance_m": distance,
        "bearing_rad": math.radians(bearing_deg),
    }


def test_route_controller_uses_pose_depth_and_visual_track_not_branch_truth():
    controller = SensorRouteController(turn_angle_deg=45.0, post_turn_forward_m=3.0)
    controller.start({"route_choice": "left", "episode_id": "episode_a"}, [0.0, 0.0, 0.0])
    approach = controller.command(
        [0.0, 0.0, 0.0], {"left": 3.0, "front": 2.0, "right": 2.0}, route_track(5.0, 20.0)
    )
    assert approach["phase"] == "approach_intersection"
    rotate = controller.command(
        [3.3, 0.0, 0.0], {"left": 3.0, "front": 2.0, "right": 2.0}, route_track(3.0, 38.0)
    )
    assert rotate["phase"] == "rotate_to_branch"
    assert rotate["angular_z_radps"] > 0.0
    enter = controller.command([3.3, 0.0, math.radians(36.0)], {"left": 3.0}, route_track(2.8, 2.0))
    assert enter["phase"] == "advance_into_branch"
    assert enter["linear_x_mps"] > 0.0
    verify = controller.command([4.5, 1.2, math.radians(36.0)], {"left": 3.0}, route_track(1.5, 0.0))
    assert verify["primitive"] == "stop"
    assert verify["phase"] == "verify_branch"


def test_route_controller_executes_staged_depth_gap_bypass_without_truth_map():
    controller = SensorRouteController()
    controller.start({"route_choice": "right", "episode_id": "episode_a"}, [0.0, 0.0, 0.0])
    rotate = controller.command(
        [1.0, 0.0, 0.0], {"left": 0.9, "front": 1.6, "right": 2.0}, route_track(5.0, -20.0)
    )
    assert rotate["phase"] == "avoid_obstacle_rotate"
    assert rotate["linear_x_mps"] == 0.0
    assert rotate["angular_z_radps"] < 0.0
    advance = controller.command(
        [1.0, 0.0, math.radians(-38.0)],
        {"left": 1.5, "front": 2.5, "right": 3.0},
        route_track(4.8, -15.0),
    )
    assert advance["phase"] == "avoid_obstacle_advance"
    assert advance["linear_x_mps"] > 0.0
    rejoin = controller.command(
        [2.1, -0.9, math.radians(-38.0)],
        {"left": 2.0, "front": 3.0, "right": 3.0},
        route_track(4.0, -10.0),
    )
    assert rejoin["phase"] == "avoid_obstacle_rejoin"
    assert rejoin["linear_x_mps"] == 0.0
    assert rejoin["angular_z_radps"] > 0.0


def test_route_controller_scans_toward_selected_branch_to_reacquire_landmark():
    controller = SensorRouteController()
    controller.start({"route_choice": "left", "episode_id": "episode_a"}, [0.0, 0.0, 0.0])
    controller.phase = "advance_into_branch"
    controller.phase_start_pose = [3.0, 0.0, 0.5]
    controller.desired_yaw = 0.8
    reacquire = controller.command([3.0, 0.0, 0.5], {"front": 2.0}, None)
    assert reacquire["primitive"] == "enter_branch"
    assert reacquire["phase"] == "reacquire_landmark"
    assert reacquire["angular_z_radps"] > 0.0
    hold = controller.command([3.0, 0.0, 0.78], {"front": 2.0}, None)
    assert hold["primitive"] == "stop"
    assert hold["phase"] == "await_landmark_reacquire"


def test_route_choice_bearing_check_applies_before_turn_not_after_heading_crosses_target():
    controller = SensorRouteController()
    controller.start({"route_choice": "left", "episode_id": "episode_a"}, [0.0, 0.0, 0.0])
    mismatch = controller.command([0.0, 0.0, 0.0], {"front": 3.0}, route_track(4.0, -12.0))
    assert mismatch["phase"] == "landmark_choice_mismatch"

    controller.phase = "advance_into_branch"
    controller.phase_start_pose = [3.0, 0.0, math.radians(45.0)]
    controller.desired_yaw = math.radians(45.0)
    recenter = controller.command(
        [3.2, 0.2, math.radians(45.0)],
        {"front": 3.0},
        route_track(2.4, -12.0),
    )
    assert recenter["phase"] == "advance_into_branch"
    assert recenter["angular_z_radps"] < 0.0

    verified = controller.command(
        [3.5, 0.5, math.radians(40.0)],
        {"front": 3.0},
        route_track(1.5, -8.0),
    )
    assert verified["phase"] == "verify_branch"


def test_instruction_visual_target_uses_only_user_text():
    assert instruction_visual_target("At the intersection, turn left toward the red sign.") == "red sign"
    assert instruction_visual_target("Approach the red fire extinguisher and stop.") == "red fire extinguisher"


def test_oracle_fields_are_rejected_at_any_depth():
    assert payload_oracle_fields({"nested": {"expected_branch": "left"}}) == ["nested.expected_branch"]
    payload = observation(1)
    payload["debug"] = {"target_pose": [1.0, 2.0, 0.0]}
    with pytest.raises(ValueError, match="oracle fields"):
        validate_target_observation(payload)
