from isaac_vln_benchmark.semantic_marker_judge_node import SemanticMarkerJudgeCore, _semantic_target_matches


def test_semantic_target_matching_ignores_marker_scaffolding_but_requires_identity():
    assert _semantic_target_matches(
        "red fire extinguisher marker", "nearest red fire extinguisher marker"
    )
    assert not _semantic_target_matches("blue box marker", "nearest red fire extinguisher marker")


def task(*, category="referential", recovery="", event=""):
    return {
        "task_id": "case",
        "task_type": "semantic_navigation",
        "category": category,
        "oracle_plan": [
            {"subgoal_type": "find"},
            {"subgoal_type": "approach"},
            {"subgoal_type": "verify"},
        ],
        "success": {"distance_to_target_m": 2.0},
        "judge": {"required_recovery": recovery, "injected_event": event},
        "semantic_runtime": {
            "subgoal_object_ids": {"0": "marker", "1": "marker", "2": "marker"}
        },
    }


def scene():
    return {
        "robot_start_pose": [0.0, 0.0, 0.0],
        "objects": [{"id": "marker", "pose": [2.3, 0.0, 0.5]}],
    }


def subgoal(index, kind):
    return {
        "episode_id": "ep",
        "subgoal_index": index,
        "subgoal_type": kind,
        "target": "red marker",
    }


def test_marker_judge_requires_two_visible_find_frames_then_safe_verify():
    core = SemanticMarkerJudgeCore(task(), scene(), "ep")
    assert core.accept_subgoal(subgoal(0, "find")) is None
    assert core.update([0.0, 0.0, 0.0]) is None
    event = core.update([0.0, 0.0, 0.0])
    assert event["type"] == "target_track_confirmed"

    assert core.accept_subgoal(subgoal(1, "approach")) is None
    event = core.update([0.4, 0.0, 0.0])
    assert event["type"] == "target_within_stop_distance"

    assert core.accept_subgoal(subgoal(2, "verify")) is None
    assert core.update([0.4, 0.0, 0.0], linear_x=0.2) is None
    event = core.update([0.4, 0.0, 0.0], linear_x=0.0)
    assert event["type"] == "completion_verified"
    assert core.status()["success"] is True
    assert core.status()["qualification_evidence"] is False


def test_recovery_must_match_and_complete_before_next_subgoal():
    core = SemanticMarkerJudgeCore(
        task(category="recovery", recovery="backtrack", event="blocked_path_detected"),
        scene(),
        "ep",
    )
    injected = core.accept_subgoal(subgoal(0, "find"))
    assert injected["type"] == "blocked_path_detected"
    assert core.status()["awaiting_recovery"] is True

    core.observe_recovery({"recovery": "backtrack"})
    core.observe_event({"type": "semantic_recovery_completed", "recovery": "backtrack"})

    status = core.status()
    assert status["recovery_matched"] is True
    assert status["completed_subgoal_indices"] == []
    assert status["recovered_subgoal_indices"] == [0]
    assert core.update([0.0, 0.0, 0.0]) is None
    assert core.update([0.0, 0.0, 0.0])["type"] == "target_track_confirmed"
    assert core.accept_subgoal(subgoal(1, "approach")) is None


def test_wrong_recovery_is_a_semantic_error():
    core = SemanticMarkerJudgeCore(
        task(category="recovery", recovery="scan", event="target_track_lost"),
        scene(),
        "ep",
    )
    core.accept_subgoal(subgoal(0, "find"))
    core.observe_recovery({"recovery": "backtrack"})
    core.observe_event({"type": "semantic_recovery_completed", "recovery": "backtrack"})
    assert core.status()["semantic_errors"] >= 1
    assert core.status()["recovery_matched"] is False


def test_episode_and_subgoal_order_are_enforced():
    core = SemanticMarkerJudgeCore(task(), scene(), "ep")
    assert core.accept_subgoal(subgoal(1, "approach")) is None
    assert core.accept_subgoal(subgoal(0, "find") | {"episode_id": "old"}) is None
    assert core.status()["semantic_errors"] == 2


def test_motion_subgoal_emits_one_bounded_no_progress_recovery_event():
    core = SemanticMarkerJudgeCore(task(), scene(), "ep", no_progress_timeout_sec=0.0)
    core.completed_indices.append(0)
    assert core.accept_subgoal(subgoal(1, "approach") | {"recovery": "scan"}) is None

    event = core.update([0.0, 0.0, 0.0])

    assert event["type"] == "no_progress_timeout"
    assert event["recovery"] == "scan"
    assert core.status()["awaiting_recovery"] is True


def test_v2_completion_takes_priority_over_no_progress_timeout():
    value = task()
    value["oracle_plan"] = [{"subgoal_type": "pass"}]
    value["semantic_runtime"] = {
        "subgoal_object_ids": {"0": "marker"},
        "judge_completion_policy": "route_progress_v2",
        "pass_reach_distance_m": 1.65,
    }
    core = SemanticMarkerJudgeCore(value, scene(), "ep", no_progress_timeout_sec=0.0)
    assert core.accept_subgoal(subgoal(0, "pass") | {"recovery": "scan"}) is None

    event = core.update([0.7, 0.0, 0.0])

    assert event["type"] == "landmark_passed"
    assert event["completion_method"] == "pass_marker_reached"
    assert core.status()["awaiting_recovery"] is False


def test_v2_route_plane_can_complete_pass_without_controller_oracle_input():
    value = task()
    value["oracle_plan"] = [{"subgoal_type": "pass"}]
    value["semantic_runtime"] = {
        "subgoal_object_ids": {"0": "marker"},
        "judge_completion_policy": "route_progress_v2",
        "pass_reach_distance_m": 1.20,
        "pass_cross_track_max_m": 1.75,
    }
    core = SemanticMarkerJudgeCore(value, scene(), "ep")
    core.accept_subgoal(subgoal(0, "pass"))

    event = core.update([2.4, 1.70, 0.0])

    assert event["type"] == "landmark_passed"
    assert event["completion_method"] == "pass_completion_plane"
    assert event["cross_track_m"] == 1.7


def test_judge_normalizes_nonzero_scene_start_into_telemetry_frame():
    value = task()
    value["oracle_plan"] = [{"subgoal_type": "approach"}]
    value["semantic_runtime"] = {
        "subgoal_object_ids": {"0": "marker"},
        "judge_completion_policy": "route_progress_v2",
    }
    shifted_scene = {
        "robot_start_pose": [5.9, -0.75, 0.548549],
        "objects": [{"id": "marker", "pose": [7.7, 0.35, 0.5]}],
    }
    core = SemanticMarkerJudgeCore(value, shifted_scene, "ep")
    core.accept_subgoal(subgoal(0, "approach"))

    event = core.update([0.2, 0.0, 0.0])

    assert event["type"] == "target_within_stop_distance"
    assert event["distance_m"] < 2.0
