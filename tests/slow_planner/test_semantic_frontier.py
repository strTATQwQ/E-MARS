from __future__ import annotations

import json

import pytest

from slow_planner.base import CandidateFrontier, OrderedImage, SlowPlannerRequest
from slow_planner.semantic_frontier import (
    FrontierFeatures,
    SemanticFrontierError,
    SemanticFrontierScorer,
)


def request(*, frontiers: tuple[CandidateFrontier, ...] | None = None, visited=()):
    return SlowPlannerRequest(
        episode_id="episode-7",
        snapshot_id="b::episode-7::2::9",
        instruction="find the kitchen",
        ordered_images=(OrderedImage("front", (0.0,) * 7, b"jpeg", 1, 1),),
        candidate_frontiers=frontiers
        if frontiers is not None
        else (
            CandidateFrontier(3, (-0.4, 1.2), 1.2649, -18.435),
            CandidateFrontier(8, (0.7, 1.0), 1.2207, 34.992),
        ),
        agent_pose=(1.0, 2.0, 0.25),
        visited_frontiers=visited,
    )


def features(*, semantic=True):
    return (
        FrontierFeatures(3, 0.1 if semantic else None, 0.9, 0.7, 0, 0.0),
        FrontierFeatures(8, 0.9 if semantic else None, 0.2, 1.2, 0, 0.0),
    )


def test_semantic_evidence_ranks_only_current_frontier_ids_without_motion_authority():
    outcome = SemanticFrontierScorer().rank(request(), features())
    assert outcome.source == "semantic_model"
    assert outcome.decision.frontier_id == 8
    public = outcome.to_mapping()
    assert public["motion_authority"] == "none"
    assert public["terminal_stop_authority"] == "none"
    encoded = json.dumps(public)
    assert "cmd_vel" not in encoded
    assert '"target_relative_xz": null' in encoded


def test_missing_or_abstained_semantics_reproduces_frozen_fallback_exactly():
    scorer = SemanticFrontierScorer()
    missing = scorer.rank(request(), features(semantic=False))
    abstained = scorer.rank(request(), features(), semantic_abstain=True)
    assert missing.source == abstained.source == "frozen_fallback"
    assert missing.decision.frontier_id == abstained.decision.frontier_id == 3
    assert missing.scores == abstained.scores == ()
    assert missing.decision.fallback_reason == "semantic_missing_or_abstained"

    visited = scorer.rank(request(visited=(3,)), features(semantic=False))
    assert visited.decision.frontier_id == 8


def test_empty_frontiers_abstain_and_mismatched_feature_ids_fail_closed():
    scorer = SemanticFrontierScorer()
    empty = scorer.rank(request(frontiers=()), ())
    assert empty.decision.decision == "abstain"
    assert empty.decision.frontier_id is None

    with pytest.raises(SemanticFrontierError, match="exactly"):
        scorer.rank(request(), features()[:1])
    with pytest.raises(SemanticFrontierError, match="repeat"):
        scorer.rank(request(), (features()[0], features()[0]))
