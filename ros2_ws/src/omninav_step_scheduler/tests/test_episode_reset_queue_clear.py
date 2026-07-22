import json
from types import SimpleNamespace

from omninav_step_scheduler.omninav_scheduler_node import OmniNavSchedulerNode


def test_omninav_scheduler_mode_reset_clears_pending_queue():
    node = object.__new__(OmniNavSchedulerNode)
    node.base_config = {}
    node.config = {}
    node.active_episode_id = "old_ep"
    node.pending_request_ids = {"old_req"}
    node.inflight = True
    node.low_confidence_count = 2
    node.last_request = 123.0
    node.last_valid_action = 0.0
    node.event_pub = None
    node.publish_json = lambda *args, **kwargs: None
    node.publish_metric = lambda *args, **kwargs: None

    node.on_benchmark_mode(SimpleNamespace(data=json.dumps({"episode_id": "new_ep", "mode_config": {}})))

    assert node.active_episode_id == "new_ep"
    assert node.pending_request_ids == set()
    assert node.inflight is False
    assert node.low_confidence_count == 0


def test_omninav_scheduler_tracks_external_requests_for_active_episode():
    node = object.__new__(OmniNavSchedulerNode)
    node.active_episode_id = "ep1"
    node.pending_request_ids = set()

    node.on_omninav_request_seen(SimpleNamespace(data=json.dumps({"episode_id": "ep1", "request_id": "fresh_req"})))
    node.on_omninav_request_seen(SimpleNamespace(data=json.dumps({"episode_id": "old_ep", "request_id": "old_req"})))

    assert node.pending_request_ids == {"fresh_req"}


def test_oracle_override_is_episode_scoped_and_reset_clears_it():
    node = object.__new__(OmniNavSchedulerNode)
    node.active_episode_id = "ep1"
    node.oracle_override_active = False
    node.publish_metric = lambda *args, **kwargs: None

    node.on_oracle_override(SimpleNamespace(data=json.dumps({"episode_id": "old_ep", "active": True})))
    assert node.oracle_override_active is False
    node.on_oracle_override(SimpleNamespace(data=json.dumps({"episode_id": "ep1", "active": True})))
    assert node.oracle_override_active is True

    node.base_config = {}
    node.config = {}
    node.pending_request_ids = set()
    node.inflight = False
    node.low_confidence_count = 0
    node.last_request = 0.0
    node.last_valid_action = 0.0
    node.event_pub = None
    node.publish_json = lambda *args, **kwargs: None
    node.on_benchmark_mode(SimpleNamespace(data=json.dumps({"episode_id": "ep2", "mode_config": {}})))
    assert node.oracle_override_active is False


def test_new_semantic_goal_supersedes_old_omninav_request_without_motion():
    node = object.__new__(OmniNavSchedulerNode)
    node.active_episode_id = "ep1"
    node.pending_request_ids = {"old_req"}
    node.inflight = True
    node.last_request = 12.0
    node.semantic_goal_active = False
    metrics = []
    node.publish_metric = lambda event, **fields: metrics.append((event, fields))

    node.on_semantic_goal(
        SimpleNamespace(
            data=json.dumps(
                {
                    "episode_id": "ep1",
                    "subgoal_type": "find",
                    "target": "red fire extinguisher",
                    "relation": "beside the exit sign",
                    "constraints": [],
                    "completion_evidence": "confirmed fresh target track",
                }
            )
        )
    )

    assert node.semantic_goal_active is True
    assert node.inflight is False
    assert node.pending_request_ids == set()
    assert node.active_subgoal["semantic_goal"]["target"] == "red fire extinguisher"
    assert metrics[-1][1]["superseded_requests"] == 1


def test_semantic_goal_with_waypoint_is_rejected():
    node = object.__new__(OmniNavSchedulerNode)
    node.active_episode_id = "ep1"
    node.pending_request_ids = set()
    node.inflight = False
    node.last_request = 0.0
    node.semantic_goal_active = False
    node.active_subgoal = {}
    metrics = []
    node.publish_metric = lambda event, **fields: metrics.append((event, fields))

    node.on_semantic_goal(
        SimpleNamespace(
            data=json.dumps(
                {
                    "episode_id": "ep1",
                    "subgoal_type": "find",
                    "target": "printer",
                    "relation": "",
                    "constraints": [],
                    "completion_evidence": "printer confirmed",
                    "waypoint": [1.0, 2.0],
                }
            )
        )
    )

    assert node.semantic_goal_active is False
    assert metrics[-1][0] == "semantic_goal_invalid"


def test_semantic_clear_with_hold_blocks_base_goal_and_publishes_normal_chain_stop():
    node = object.__new__(OmniNavSchedulerNode)
    node.active_episode_id = "ep1"
    node.pending_request_ids = {"old_req"}
    node.inflight = True
    node.semantic_goal_active = True
    node.semantic_hold_active = False
    node.active_subgoal = {"semantic_goal": {"subgoal_type": "enter"}}
    node.primitive_pub = object()
    published = []
    metrics = []
    node.publish_json = lambda publisher, payload: published.append((publisher, payload))
    node.publish_metric = lambda event, **fields: metrics.append((event, fields))

    node.on_semantic_goal(
        SimpleNamespace(
            data=json.dumps(
                {
                    "clear": True,
                    "hold": True,
                    "episode_id": "ep1",
                    "reason": "subgoal_completed",
                }
            )
        )
    )

    assert node.semantic_goal_active is True
    assert node.semantic_hold_active is True
    assert node.pending_request_ids == set()
    assert published[-1][1]["primitive"] == "stop"
    assert published[-1][1]["source"] == "omninav_semantic_hold"
    assert metrics[-1][1]["result"] == "held"
