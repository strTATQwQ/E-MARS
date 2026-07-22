from __future__ import annotations

from dataclasses import replace

import pytest

from slow_planner.base import (
    CandidateFrontier,
    OrderedImage,
    PlannerMetrics,
    PlannerRunner,
    SlowPlanner,
    SlowPlannerProtocolError,
    SlowPlannerRequest,
    parse_decision,
)


def request() -> SlowPlannerRequest:
    return SlowPlannerRequest(
        episode_id="episode-1",
        snapshot_id="snap-7",
        instruction="Walk through the doorway and wait by the table.",
        ordered_images=(OrderedImage("front", (0.0, 0.0, 0.0), b"jpeg", 640, 360),),
        candidate_frontiers=(
            CandidateFrontier(3, (1.0, 0.5), 1.118, 26.5),
            CandidateFrontier(8, (0.2, -1.2), 1.217, -80.5),
        ),
        agent_pose=(0.0, 0.0, 0.0),
        visited_frontiers=(8,),
    )


def test_parse_valid_frontier_decision_and_ignore_leading_text() -> None:
    decision = parse_decision(
        request(),
        'brief preamble {"decision":"select_frontier","frontier_id":3,"target_relative_xz":null,"confidence":0.75}',
        attempts=1,
    )
    assert decision.frontier_id == 3
    assert decision.snapshot_id == "snap-7"


def test_parse_rejects_invalid_frontier_id() -> None:
    with pytest.raises(SlowPlannerProtocolError, match="not a current candidate"):
        parse_decision(
            request(),
            '{"decision":"select_frontier","frontier_id":9,"target_relative_xz":null,"confidence":0.75}',
            attempts=1,
        )


def test_parse_rejects_unknown_schema_key() -> None:
    with pytest.raises(SlowPlannerProtocolError, match="unknown keys"):
        parse_decision(
            request(),
            '{"decision":"abstain","frontier_id":null,"target_relative_xz":null,"confidence":0.2,"reasoning":"x"}',
            attempts=1,
        )


class FakePlanner(SlowPlanner):
    model_variant = "fake"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def generate_raw(self, request, *, correction=""):
        return self.responses.pop(0), PlannerMetrics(model_variant=self.model_variant)


def test_runner_retries_once_then_accepts() -> None:
    planner = FakePlanner(
        [
            "not json",
            '{"decision":"select_frontier","frontier_id":3,"target_relative_xz":null,"confidence":0.8}',
        ]
    )
    decision, metrics = PlannerRunner(planner, max_retries=1).decide(request())
    assert decision.frontier_id == 3
    assert decision.parse_attempts == 2
    assert metrics.retry_count == 1


def test_runner_deterministic_fallback_avoids_visited_frontier() -> None:
    planner = FakePlanner(["bad", "still bad"])
    decision, _ = PlannerRunner(planner, max_retries=1).decide(request())
    assert decision.fallback_used is True
    assert decision.frontier_id == 3


def test_runner_deterministic_fallback_backtracks_at_dead_end() -> None:
    planner = FakePlanner(["bad", "still bad"])
    all_visited = replace(request(), visited_frontiers=(3, 8))
    decision, _ = PlannerRunner(planner, max_retries=1).decide(all_visited)
    assert decision.fallback_used is True
    assert decision.frontier_id == 3


def test_wire_requires_image_count_match() -> None:
    metadata = request().metadata()
    with pytest.raises(SlowPlannerProtocolError, match="count mismatch"):
        SlowPlannerRequest.from_wire(metadata, [])


def test_snapshot_is_immutable_and_echoed() -> None:
    first = request()
    second = replace(first, snapshot_id="snap-8")
    decision = parse_decision(
        second,
        '{"decision":"abstain","frontier_id":null,"target_relative_xz":null,"confidence":0.1}',
        attempts=1,
    )
    assert first.snapshot_id == "snap-7"
    assert decision.snapshot_id == "snap-8"
