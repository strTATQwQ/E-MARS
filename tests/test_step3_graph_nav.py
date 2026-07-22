from __future__ import annotations

import json
import math

import pytest

from slow_benchmark.oracle_graph import MatterportGraph
from step3_graph_nav.evaluation import GraphEpisodeState, candidate_specs
from step3_graph_nav.arrival_verifier import (
    ArrivalCurrentView,
    ArrivalHistoryFrame,
    ArrivalRequest,
    parse_arrival,
)
from step3_graph_nav.model import _clean_reasoning
from step3_graph_nav.navigation_policy import (
    NAVIGATION_SYSTEM_PROMPT,
    build_navigation_prompt,
    parse_navigation_action,
)
from step3_graph_nav.protocol import (
    CandidateView,
    GraphNavProtocolError,
    GraphNavRequest,
    InvalidCandidateId,
)


def _row(image_id, xyz, links):
    pose = [
        1.0,
        0.0,
        0.0,
        xyz[0],
        0.0,
        1.0,
        0.0,
        xyz[1],
        0.0,
        0.0,
        1.0,
        xyz[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    return {"image_id": image_id, "included": True, "pose": pose, "unobstructed": links}


def _graph(tmp_path):
    rows = [
        _row("a", (0.0, 0.0, 1.4), [False, True, True]),
        _row("b", (1.0, 0.0, 1.4), [True, False, True]),
        _row("c", (1.0, 1.0, 1.4), [True, True, False]),
    ]
    path = tmp_path / "tiny_connectivity.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return MatterportGraph.load(path)


def _candidate(candidate_id: int, target_id: int) -> CandidateView:
    return CandidateView(
        candidate_id=candidate_id,
        target_viewpoint_id=target_id,
        relative_heading_deg=float(candidate_id * 10),
        graph_distance_m=1.0,
        jpeg=b"jpeg",
        width=640,
        height=360,
    )


def _request() -> GraphNavRequest:
    return GraphNavRequest(
        episode_id="ep",
        snapshot_id="snap",
        instruction="Go to the kitchen.",
        current_viewpoint_id=4,
        step_index=0,
        candidates=(_candidate(0, 7), _candidate(1, 9)),
    )


def test_navigation_policy_accepts_only_exact_move_shape():
    request = _request()
    assert parse_navigation_action(
        '{"action":"move","candidate_id":1}', request.candidate_ids
    ).to_mapping() == {"action": "move", "candidate_id": 1}
    with pytest.raises(GraphNavProtocolError, match="exactly"):
        parse_navigation_action('{"action":"stop"}', request.candidate_ids)
    with pytest.raises(GraphNavProtocolError, match="exactly"):
        parse_navigation_action('{"action":"move","candidate_id":1,"arrived":false}', request.candidate_ids)


def test_invalid_candidate_id_is_a_distinct_hard_gate_error():
    with pytest.raises(InvalidCandidateId):
        parse_navigation_action('{"action":"move","candidate_id":2}', frozenset({0, 1}))


def test_request_requires_fixed_ordinals_and_sorted_targets():
    with pytest.raises(GraphNavProtocolError, match="zero-based ordinals"):
        GraphNavRequest(
            episode_id="ep",
            snapshot_id="snap",
            instruction="go",
            current_viewpoint_id=0,
            step_index=0,
            candidates=(_candidate(1, 7),),
        )
    with pytest.raises(GraphNavProtocolError, match="sorted"):
        GraphNavRequest(
            episode_id="ep",
            snapshot_id="snap",
            instruction="go",
            current_viewpoint_id=0,
            step_index=0,
            candidates=(_candidate(0, 9), _candidate(1, 7)),
        )


def test_navigation_prompt_never_exposes_target_viewpoint_ids_or_stop():
    prompt = build_navigation_prompt(_request())
    assert "candidate_id=0" in prompt
    assert "image_index=1" in prompt
    assert "forbidden to STOP" in NAVIGATION_SYSTEM_PROMPT
    assert "target_viewpoint" not in prompt
    assert "candidate_id=7" not in prompt


def test_reasoning_is_bounded_and_final_json_is_removed_from_private_text():
    assert _clean_reasoning('inspect hallway </think> {"arrived":true}') == "inspect hallway"
    assert _clean_reasoning('turn left ```json') == "turn left"


def test_arrival_protocol_has_no_candidate_ids_and_requires_history():
    current = (
        ArrivalCurrentView(0, -45.0, b"jpeg0", 640, 360),
        ArrivalCurrentView(1, 45.0, b"jpeg1", 640, 360),
    )
    history = (
        ArrivalHistoryFrame(0, b"old", 640, 360),
        ArrivalHistoryFrame(1, b"new", 640, 360),
    )
    request = ArrivalRequest(
        episode_id="ep",
        snapshot_id="snap",
        instruction="Wait at the doorway.",
        step_index=2,
        current_views=current,
        history_frames=history,
        action_history=("turned left and moved", "continued forward and moved"),
    )
    metadata = json.dumps(request.metadata())
    assert "candidate_id" not in metadata
    assert parse_arrival('{"arrived":true}') is True
    assert parse_arrival('{"arrived":false}') is False
    with pytest.raises(GraphNavProtocolError, match="exactly arrived"):
        parse_arrival('{"arrived":true,"candidate_id":0}')
    with pytest.raises(GraphNavProtocolError, match="2 to 4"):
        ArrivalRequest(
            episode_id="ep",
            snapshot_id="snap",
            instruction="stop",
            step_index=1,
            current_views=current,
            history_frames=history[:1],
            action_history=("moved",),
        )


def test_candidate_order_and_graph_metrics(tmp_path):
    graph = _graph(tmp_path)
    specs = candidate_specs(graph, 0, 0.0)
    assert [item.candidate_id for item in specs] == [0, 1]
    assert [item.target_viewpoint_id for item in specs] == [1, 2]
    assert specs[0].relative_heading_deg == pytest.approx(0.0)
    assert specs[1].relative_heading_deg == pytest.approx(45.0)

    episode = {
        "benchmark_episode_id": "tiny:1",
        "episode_id": 1,
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, -math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0)],
        "goals": [{"position": [1.0, 0.0, -1.0], "radius": 0.1}],
        "instruction": {"instruction_text": "Go to c."},
    }
    state = GraphEpisodeState.from_episode(graph, episode)
    assert state.current_node == 0
    assert state.goal_node == 2
    selected = next(item for item in candidate_specs(graph, 0, state.yaw_rad) if item.target_viewpoint_id == 2)
    state.move(selected)
    assert state.goal_positive
    state.stop()
    result = state.result(calls=1, failure_reason="", wall_seconds=1.0)
    assert result["success"] is True
    assert result["oracle_success"] is True
    assert result["ne_m"] == pytest.approx(0.0)
    assert result["spl"] == pytest.approx(1.0)
