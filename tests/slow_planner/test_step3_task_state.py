from __future__ import annotations

import json
from dataclasses import replace

import pytest

from slow_planner.base import (
    CandidateFrontier,
    OrderedImage,
    SlowPlannerProtocolError,
    SlowPlannerRequest,
)
from slow_planner.client import SlowPlannerClient
from slow_planner.step3_task_state import (
    Step3TaskStateSlowPlanner,
    TaskStateContext,
    TaskStatePlannerDecision,
    fallback_instruction_clauses,
    task_state_context,
)


INSTRUCTION = (
    "Walk straight into the kitchen. Turn left and exit the kitchen."
)


def request(
    *,
    clauses: tuple[str, ...] = (),
    active_clause_id: int = 0,
    revision: int = 0,
) -> SlowPlannerRequest:
    state = TaskStateContext(
        instruction=INSTRUCTION,
        clauses=clauses,
        active_clause_id=active_clause_id,
        revision=revision,
    )
    return SlowPlannerRequest(
        episode_id="a::121",
        snapshot_id="a::a::121::0::7",
        instruction="Action 1 timed out; choose only legal IDs 2 or 3.",
        ordered_images=(
            OrderedImage(
                view_id="front",
                pose=(0.2, 0.0, 0.2, 0.0, -0.17),
                jpeg=b"jpeg",
                width=640,
                height=480,
            ),
        ),
        candidate_frontiers=(
            CandidateFrontier(2, (-0.18, 0.18), 0.255, -45.0),
            CandidateFrontier(3, (0.18, 0.18), 0.255, 45.0),
        ),
        agent_pose=(0.0, 0.0, 0.0, 0.0),
        compact_history=(state.to_history(),),
    )


def raw(**changes: object) -> str:
    value = {
        "task_clauses": [
            "Walk straight into the kitchen",
            "Turn left and exit the kitchen",
        ],
        "active_clause_id": 1,
        "clause_transition": "advance",
        "recommended_frontier": 2,
        "confidence": 0.82,
        "evidence": ["kitchen threshold is behind the robot"],
        "target_found": False,
        "abstain": False,
    }
    value.update(changes)
    return json.dumps(value, separators=(",", ":"))


def planner() -> Step3TaskStateSlowPlanner:
    return object.__new__(Step3TaskStateSlowPlanner)


def test_initial_request_decomposes_exact_spans_and_updates_state() -> None:
    decision = planner().parse_decision(request(), raw(), attempts=1)

    assert isinstance(decision, TaskStatePlannerDecision)
    assert decision.task_clauses == (
        "Walk straight into the kitchen",
        "Turn left and exit the kitchen",
    )
    assert decision.active_clause_id == 1
    assert decision.clause_transition == "advance"
    assert decision.frontier_id == 2
    assert decision.evidence == ("kitchen threshold is behind the robot",)


def test_initialized_request_forbids_plan_replacement_and_echoes_active_clause() -> None:
    current = request(
        clauses=(
            "Walk straight into the kitchen",
            "Turn left and exit the kitchen",
        ),
        active_clause_id=2,
        revision=3,
    )
    decision = planner().parse_decision(
        current,
        raw(
            task_clauses=[],
            active_clause_id=2,
            clause_transition="hold",
            recommended_frontier=3,
            evidence=["left doorway remains ahead"],
        ),
        attempts=1,
    )
    assert decision.task_clauses == ()
    assert decision.active_clause_id == 2
    with pytest.raises(
        SlowPlannerProtocolError, match="forbids plan replacement"
    ):
        planner().parse_decision(current, raw(active_clause_id=2), attempts=1)


@pytest.mark.parametrize(
    "changes",
    [
        {"task_clauses": ["Go upstairs"]},
        {"task_clauses": ["Turn left", "Walk straight into the kitchen"]},
        {"active_clause_id": 2},
        {"clause_transition": "advance", "evidence": []},
        {"recommended_frontier": 99},
        {"target_found": True, "abstain": True, "recommended_frontier": None},
    ],
)
def test_task_state_parser_rejects_hallucinated_or_inconsistent_updates(
    changes: dict[str, object],
) -> None:
    with pytest.raises(SlowPlannerProtocolError):
        planner().parse_decision(request(), raw(**changes), attempts=1)


def test_final_clause_target_found_is_only_a_structured_candidate() -> None:
    current = request(
        clauses=(
            "Walk straight into the kitchen",
            "Turn left and exit the kitchen",
        ),
        active_clause_id=2,
        revision=4,
    )
    decision = planner().parse_decision(
        current,
        raw(
            task_clauses=[],
            active_clause_id=2,
            clause_transition="advance",
            recommended_frontier=None,
            evidence=["exit threshold is beside the robot"],
            target_found=True,
            abstain=True,
        ),
        attempts=1,
    )
    assert decision.decision == "abstain"
    assert decision.target_found is True
    assert decision.frontier_id is None


def test_task_prompt_exposes_state_without_hidden_memory_or_stop_authority() -> None:
    prompt = planner().format_prompt(request())
    assert "high-level task decomposer and state updater" in prompt
    assert INSTRUCTION in prompt
    assert "ordered exact text spans" in prompt
    assert "does not grant terminal STOP" in prompt
    assert "chain-of-thought" not in prompt


def arrival_request() -> SlowPlannerRequest:
    current = request(
        clauses=(
            "Walk straight into the kitchen",
            "Turn left and exit the kitchen",
        ),
        active_clause_id=2,
        revision=3,
    )
    return replace(
        current,
        instruction="observation-only arrival shadow",
        candidate_frontiers=(),
        compact_history=current.compact_history
        + ("arrival_evidence_contract=goal_region|strict_semantic_arrival",),
    )


def test_arrival_shadow_requires_two_explicit_labeled_verdicts() -> None:
    current = arrival_request()
    prompt = planner().format_prompt(current)
    assert "ARRIVAL_SHADOW_ONLY" in prompt
    assert '"task_clauses":[]' in prompt
    assert '"active_clause_id":2' in prompt
    assert '"goal_region: no - not visible"' in prompt
    assert '"semantic_arrival: no - not visible"' in prompt
    assert "maximum three words each" in prompt
    assert "both goal_region and semantic_arrival are yes" in prompt
    assert "never grants terminal STOP" in prompt
    assert "high-level task decomposer and state updater" not in prompt
    assert "For hold/recover, prefer evidence=[]" not in prompt

    negative = raw(
        task_clauses=[],
        active_clause_id=2,
        clause_transition="hold",
        recommended_frontier=None,
        evidence=(
            "goal_region: yes - white rug nearby",
            "semantic_arrival: no - rug remains ahead",
        ),
        target_found=False,
        abstain=True,
    )
    observed = planner().parse_decision(current, negative, attempts=1)
    assert observed.target_found is False
    assert len(observed.evidence) == 2

    with pytest.raises(
        SlowPlannerProtocolError, match="two labeled yes/no"
    ):
        planner().parse_decision(
            current,
            raw(
                task_clauses=[],
                active_clause_id=2,
                clause_transition="hold",
                recommended_frontier=None,
                evidence=[],
                abstain=True,
            ),
            attempts=1,
        )

    with pytest.raises(
        SlowPlannerProtocolError, match="both arrival verdicts yes"
    ):
        planner().parse_decision(
            current,
            raw(
                task_clauses=[],
                active_clause_id=2,
                clause_transition="advance",
                recommended_frontier=None,
                evidence=(
                    "goal_region: yes - white rug nearby",
                    "semantic_arrival: no - rug remains ahead",
                ),
                target_found=True,
                abstain=True,
            ),
            attempts=1,
        )

    with pytest.raises(
        SlowPlannerProtocolError, match="both arrival verdicts yes"
    ):
        planner().parse_decision(
            current,
            raw(
                task_clauses=[],
                active_clause_id=2,
                clause_transition="advance",
                recommended_frontier=None,
                evidence=(
                    "goal_region: no - outside room",
                    "semantic_arrival: yes - rug beside robot",
                ),
                target_found=True,
                abstain=True,
            ),
            attempts=1,
        )


def test_context_and_deterministic_fallback_are_bounded() -> None:
    parsed = task_state_context(request())
    assert parsed is not None
    assert parsed.instruction == INSTRUCTION
    assert parsed.active_clause_id == 0
    assert fallback_instruction_clauses(INSTRUCTION) == (
        "Walk straight into the kitchen",
        "Turn left and exit the kitchen.",
    )
    assert fallback_instruction_clauses("Walk to the doorway.") == (
        "Walk to the doorway.",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instruction", None),
        ("clauses", [1]),
        ("active_clause_id", True),
        ("revision", False),
    ],
)
def test_task_state_marker_rejects_bool_or_non_string_identity(
    field: str, value: object
) -> None:
    state = {
        "schema_version": 1,
        "instruction": INSTRUCTION,
        "clauses": [],
        "active_clause_id": 0,
        "revision": 0,
    }
    state[field] = value
    invalid = SlowPlannerRequest(
        episode_id="a::121",
        snapshot_id="a::a::121::0::7",
        instruction="choose legal motion",
        ordered_images=request().ordered_images,
        candidate_frontiers=request().candidate_frontiers,
        agent_pose=(0.0, 0.0, 0.0, 0.0),
        compact_history=(
            "task_state_v1="
            + json.dumps(state, separators=(",", ":")),
        ),
    )
    with pytest.raises(SlowPlannerProtocolError):
        task_state_context(invalid)


def test_client_preserves_private_task_state_decision() -> None:
    decision = planner().parse_decision(request(), raw(), attempts=1)
    response = {
        "ok": True,
        "decision": decision.to_mapping(),
        "metrics": {},
        "server_total_ms": 0.0,
    }

    class Socket:
        def send_multipart(self, _parts: list[bytes]) -> None:
            return None

        def recv(self) -> bytes:
            return json.dumps(response).encode("utf-8")

    client = object.__new__(SlowPlannerClient)
    client.socket = Socket()
    observed, _metrics = client.decide(request())
    assert isinstance(observed, TaskStatePlannerDecision)
    assert observed.task_clauses == decision.task_clauses
    assert observed.clause_transition == "advance"


def test_task_state_health_is_private_protocol_v1_extension() -> None:
    class Parent:
        def health(self) -> dict[str, object]:
            return {"protocol_version": 1}

    instance = planner()
    original = Step3TaskStateSlowPlanner.__mro__[1].health
    try:
        Step3TaskStateSlowPlanner.__mro__[1].health = Parent.health
        health = instance.health()
    finally:
        Step3TaskStateSlowPlanner.__mro__[1].health = original
    assert health["planner_mode"] == "task_state_v1"
    assert health["private_task_state_contract"] == 1
