"""Deterministic, bounded semantic-frontier ranking for Lane B.

The scorer is deliberately independent of oracle/reference-path labels.  Those
labels belong only in the offline benchmark.  At runtime it may consume a
bounded semantic relevance value from InternVLA or Step3, but it can only rank
the current protocol-v1 frontier IDs or fall back to geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from .base import PlannerDecision, SlowPlannerRequest, deterministic_fallback


class SemanticFrontierError(ValueError):
    """Raised when features do not match the frozen frontier request."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SemanticFrontierError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SemanticFrontierError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise SemanticFrontierError(f"{name} must be finite")
    return number


def _unit(value: Any, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise SemanticFrontierError(f"{name} must be in [0, 1]")
    return number


@dataclass(frozen=True)
class FrontierFeatures:
    frontier_id: int
    semantic_relevance: float | None
    information_gain: float
    path_cost_m: float
    revisit_count: int = 0
    stuck_penalty: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.frontier_id, bool) or self.frontier_id < 0:
            raise SemanticFrontierError("frontier_id must be non-negative")
        if self.semantic_relevance is not None:
            _unit(self.semantic_relevance, "semantic_relevance")
        _unit(self.information_gain, "information_gain")
        if _finite(self.path_cost_m, "path_cost_m") < 0.0:
            raise SemanticFrontierError("path_cost_m must be non-negative")
        if isinstance(self.revisit_count, bool) or self.revisit_count < 0:
            raise SemanticFrontierError("revisit_count must be non-negative")
        _unit(self.stuck_penalty, "stuck_penalty")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrontierFeatures":
        frontier_id = value.get("frontier_id")
        if isinstance(frontier_id, bool):
            raise SemanticFrontierError("frontier_id must be an integer")
        try:
            frontier_id = int(frontier_id)
            revisit_count = int(value.get("revisit_count", 0))
        except (TypeError, ValueError) as exc:
            raise SemanticFrontierError("frontier_id/revisit_count must be integers") from exc
        semantic = value.get("semantic_relevance")
        return cls(
            frontier_id=frontier_id,
            semantic_relevance=None if semantic is None else _unit(semantic, "semantic_relevance"),
            information_gain=_unit(value.get("information_gain", 0.0), "information_gain"),
            path_cost_m=_finite(value.get("path_cost_m"), "path_cost_m"),
            revisit_count=revisit_count,
            stuck_penalty=_unit(value.get("stuck_penalty", 0.0), "stuck_penalty"),
        )


@dataclass(frozen=True)
class SemanticFrontierWeights:
    semantic_relevance: float = 0.50
    information_gain: float = 0.25
    path_cost: float = 0.15
    revisit: float = 0.07
    stuck: float = 0.03

    def __post_init__(self) -> None:
        values = tuple(_finite(value, name) for name, value in asdict(self).items())
        if any(value < 0.0 for value in values):
            raise SemanticFrontierError("weights must be non-negative")
        if sum(values) <= 0.0:
            raise SemanticFrontierError("at least one weight must be positive")


@dataclass(frozen=True)
class FrontierScore:
    frontier_id: int
    total: float
    semantic: float
    information_gain: float
    path_cost_penalty: float
    revisit_penalty: float
    stuck_penalty: float

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticFrontierOutcome:
    decision: PlannerDecision
    source: str
    scores: tuple[FrontierScore, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "t5_semantic_frontier_outcome",
            "source": self.source,
            "decision": self.decision.to_mapping(),
            "scores": [score.to_mapping() for score in self.scores],
            "motion_authority": "none",
            "terminal_stop_authority": "none",
        }


class SemanticFrontierScorer:
    """Rank only current legal frontiers; never emit coordinates or STOP."""

    def __init__(
        self,
        weights: SemanticFrontierWeights | None = None,
        *,
        semantic_min_confidence: float = 0.05,
    ) -> None:
        self.weights = weights or SemanticFrontierWeights()
        self.semantic_min_confidence = _unit(
            semantic_min_confidence, "semantic_min_confidence"
        )

    def rank(
        self,
        request: SlowPlannerRequest,
        features: Sequence[FrontierFeatures],
        *,
        semantic_abstain: bool = False,
        semantic_source: str = "semantic_model",
    ) -> SemanticFrontierOutcome:
        request_ids = {frontier.frontier_id for frontier in request.candidate_frontiers}
        feature_ids = [feature.frontier_id for feature in features]
        if len(feature_ids) != len(set(feature_ids)):
            raise SemanticFrontierError("features repeat a frontier_id")
        if set(feature_ids) != request_ids:
            raise SemanticFrontierError(
                "features must cover exactly the current legal frontier IDs"
            )
        if not request_ids:
            decision = PlannerDecision(
                episode_id=request.episode_id,
                snapshot_id=request.snapshot_id,
                decision="abstain",
                frontier_id=None,
                target_relative_xz=None,
                confidence=0.0,
                raw_text="",
                fallback_used=True,
                fallback_reason="no_current_legal_frontiers",
            )
            return SemanticFrontierOutcome(decision, "abstain", ())

        semantic_values = [
            feature.semantic_relevance
            for feature in features
            if feature.semantic_relevance is not None
        ]
        use_semantic = (
            not semantic_abstain
            and bool(semantic_values)
            and max(semantic_values) >= self.semantic_min_confidence
        )
        if not use_semantic:
            decision = deterministic_fallback(
                request,
                raw_text="",
                reason="semantic_missing_or_abstained",
                attempts=1,
            )
            return SemanticFrontierOutcome(decision, "frozen_fallback", ())

        source = semantic_source
        maximum_cost = max(feature.path_cost_m for feature in features)
        scores = []
        for feature in features:
            semantic = float(feature.semantic_relevance or 0.0)
            cost = feature.path_cost_m / maximum_cost if maximum_cost > 0.0 else 0.0
            revisit = min(float(feature.revisit_count), 3.0) / 3.0
            total = (
                self.weights.semantic_relevance * semantic
                + self.weights.information_gain * feature.information_gain
                - self.weights.path_cost * cost
                - self.weights.revisit * revisit
                - self.weights.stuck * feature.stuck_penalty
            )
            scores.append(
                FrontierScore(
                    frontier_id=feature.frontier_id,
                    total=total,
                    semantic=semantic,
                    information_gain=feature.information_gain,
                    path_cost_penalty=cost,
                    revisit_penalty=revisit,
                    stuck_penalty=feature.stuck_penalty,
                )
            )
        ranked = tuple(sorted(scores, key=lambda item: (-item.total, item.frontier_id)))
        selected = ranked[0]
        confidence = max(
            feature.semantic_relevance or 0.0
            for feature in features
            if feature.frontier_id == selected.frontier_id
        )
        decision = PlannerDecision(
            episode_id=request.episode_id,
            snapshot_id=request.snapshot_id,
            decision="select_frontier",
            frontier_id=selected.frontier_id,
            target_relative_xz=None,
            confidence=confidence,
            raw_text="",
            fallback_used=False,
            fallback_reason="",
        )
        return SemanticFrontierOutcome(decision, source, ranked)
