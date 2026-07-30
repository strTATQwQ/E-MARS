from __future__ import annotations

import threading
from pathlib import Path

from scripts.t5_step3_timeout_advisor_node import (
    TimeoutAdvisorNode,
    _apply_task_state_decision,
    _fallback_decomposed_task_state,
    _new_task_state,
)
from slow_planner.step3_task_state import (
    TaskStatePlannerDecision,
    fallback_instruction_clauses,
)


def context(*, sequence_id: int = 7) -> dict:
    return {
        "schema_version": 1,
        "kind": "motion_timeout_after_confirmed_safe_stop",
        "episode_id": "a::121",
        "reset_generation": 0,
        "trigger_sequence_id": sequence_id,
        "trigger_request_id": f"a::121:0:{sequence_id}",
        "expected_sequence_id": sequence_id + 1,
        "excluded_action": 1,
        "instruction": (
            "Walk straight into the kitchen. Turn left and exit the kitchen."
        ),
        "camera_order": ["front_left", "front", "front_right", "rear"],
        "advisor_round": 0,
    }


def decision(**changes: object) -> TaskStatePlannerDecision:
    value = {
        "episode_id": "a::121",
        "snapshot_id": "a::a::121::0::7",
        "decision": "select_frontier",
        "frontier_id": 2,
        "target_relative_xz": None,
        "confidence": 0.82,
        "task_clauses": (
            "Walk straight into the kitchen",
            "Turn left and exit the kitchen",
        ),
        "active_clause_id": 1,
        "clause_transition": "advance",
        "recommended_frontier": 2,
        "evidence": ("kitchen threshold is behind the robot",),
        "target_found": False,
        "abstain": False,
    }
    value.update(changes)
    return TaskStatePlannerDecision(**value)


def test_step3_decomposition_and_transition_advance_monotonically() -> None:
    state, event = _apply_task_state_decision(
        _new_task_state(context()),
        context(),
        decision(),
        snapshot_sim_stamp_ns=9_000_000_000,
        transition_minimum_confidence=0.70,
    )

    assert state["decomposition_source"] == "step3_exact_instruction_spans"
    assert state["completed_clause_ids"] == [1]
    assert state["active_clause_id"] == 2
    assert state["last_transition"] == "advance"
    assert event["active_clause"] == "Turn left and exit the kitchen"


def test_low_confidence_advance_becomes_hold_without_state_regression() -> None:
    state, _ = _apply_task_state_decision(
        _new_task_state(context()),
        context(),
        decision(confidence=0.60),
        snapshot_sim_stamp_ns=9_000_000_000,
        transition_minimum_confidence=0.70,
    )
    assert state["completed_clause_ids"] == []
    assert state["active_clause_id"] == 1
    assert state["last_transition"] == "hold"


def test_two_near_landmark_confirmations_advance_intermediate_clause() -> None:
    state, _ = _fallback_decomposed_task_state(
        _new_task_state(context()), context()
    )
    first = decision(
        task_clauses=(),
        clause_transition="recover",
        confidence=0.80,
        evidence=("front view shows kitchen cabinets",),
    )
    state, first_event = _apply_task_state_decision(
        state,
        context(),
        first,
        snapshot_sim_stamp_ns=9_000_000_000,
        transition_minimum_confidence=0.70,
    )
    assert state["active_clause_id"] == 1
    assert state["active_clause_semantic_confirmations"] == 1
    assert first_event["accepted_transition"] == "recover"

    second = decision(
        snapshot_id="a::a::121::0::8",
        task_clauses=(),
        clause_transition="recover",
        confidence=0.80,
        evidence=("front view shows kitchen floor and cabinets",),
    )
    state, second_event = _apply_task_state_decision(
        state,
        context(sequence_id=8),
        second,
        snapshot_sim_stamp_ns=10_000_000_000,
        transition_minimum_confidence=0.70,
    )
    assert state["active_clause_id"] == 2
    assert state["completed_clause_ids"] == [1]
    assert state["active_clause_semantic_confirmations"] == 0
    assert second_event["accepted_transition"] == "advance"
    assert (
        second_event["transition_reason"]
        == "intermediate_clause_semantically_confirmed_twice"
    )


def test_distant_landmark_evidence_does_not_advance_clause() -> None:
    state, _ = _fallback_decomposed_task_state(
        _new_task_state(context()), context()
    )
    for sequence_id in (7, 8):
        state, _ = _apply_task_state_decision(
            state,
            context(sequence_id=sequence_id),
            decision(
                snapshot_id=f"a::a::121::0::{sequence_id}",
                task_clauses=(),
                clause_transition="recover",
                confidence=0.80,
                evidence=("front view shows kitchen area ahead",),
            ),
            snapshot_sim_stamp_ns=(sequence_id + 2) * 1_000_000_000,
            transition_minimum_confidence=0.70,
        )
    assert state["active_clause_id"] == 1
    assert state["active_clause_semantic_confirmations"] == 0


def test_negated_landmark_evidence_does_not_advance_clause() -> None:
    state, _ = _fallback_decomposed_task_state(
        _new_task_state(context()), context()
    )
    for sequence_id, evidence in (
        (7, "couch not visible in any view"),
        (8, "there is no couch visible from the front"),
    ):
        state, _ = _apply_task_state_decision(
            state,
            context(sequence_id=sequence_id),
            decision(
                snapshot_id=f"a::a::121::0::{sequence_id}",
                task_clauses=(),
                clause_transition="recover",
                confidence=0.80,
                evidence=(evidence,),
            ),
            snapshot_sim_stamp_ns=(sequence_id + 2) * 1_000_000_000,
            transition_minimum_confidence=0.70,
        )
    assert state["active_clause_id"] == 1
    assert state["active_clause_semantic_confirmations"] == 0


def test_final_clause_completion_is_candidate_not_terminal_stop() -> None:
    state, _ = _apply_task_state_decision(
        _new_task_state(context()),
        context(),
        decision(),
        snapshot_sim_stamp_ns=9_000_000_000,
        transition_minimum_confidence=0.70,
    )
    final_decision = decision(
        snapshot_id="a::a::121::0::8",
        decision="abstain",
        frontier_id=None,
        confidence=0.90,
        task_clauses=(),
        active_clause_id=2,
        clause_transition="advance",
        recommended_frontier=None,
        evidence=("exit threshold is beside the robot",),
        target_found=True,
        abstain=True,
    )
    state, event = _apply_task_state_decision(
        state,
        context(sequence_id=8),
        final_decision,
        snapshot_sim_stamp_ns=10_000_000_000,
        transition_minimum_confidence=0.70,
    )
    assert state["active_clause_id"] == 2
    assert state["completed_clause_ids"] == [1]
    assert state["final_arrival_candidate"] is True
    assert state["last_transition"] == "hold"
    assert (
        event["transition_reason"]
        == "final_clause_arrival_candidate_requires_existing_gate"
    )


def test_deterministic_decomposition_is_only_a_fail_closed_fallback() -> None:
    state, event = _fallback_decomposed_task_state(
        _new_task_state(context()), context()
    )
    assert state["decomposition_source"] == "deterministic_exact_span_fallback"
    assert state["active_clause_id"] == 1
    assert event["source"] == "deterministic_exact_span_fallback"


def test_long_and_navigation_chain_has_bounded_exact_span_fallback() -> None:
    instruction = (
        "facing the couch with the counter at your back take a left down the "
        "hallway and go straight past the sitting area and take the right right "
        "into the hallway and stop past the letter picture in front of the "
        "sliding door to the pantry."
    )

    clauses = fallback_instruction_clauses(instruction)

    assert clauses == (
        "facing the couch with the counter at your back take a left down the hallway",
        "go straight past the sitting area",
        "take the right right into the hallway",
        "stop past the letter picture in front of the sliding door to the pantry.",
    )
    assert all(len(clause) <= 160 for clause in clauses)


def test_terminal_only_stop_is_joined_to_observable_destination_clause() -> None:
    instruction = (
        "Go straight. Pass the bar with the stools. Walk straight until you get "
        "to a table with chairs then stop."
    )

    assert fallback_instruction_clauses(instruction) == (
        "Go straight",
        "Pass the bar with the stools",
        "Walk straight until you get to a table with chairs then stop.",
    )


def test_next_to_is_a_spatial_relation_not_a_sequence_boundary() -> None:
    instruction = "Wait next to the chair. Then walk into the hallway."

    assert fallback_instruction_clauses(instruction) == (
        "Wait next to the chair",
        "walk into the hallway.",
    )


def test_extends_forward_does_not_auto_advance_a_hallway_clause() -> None:
    state, _ = _fallback_decomposed_task_state(
        _new_task_state(context()), context()
    )
    for sequence_id in (7, 8):
        state, _ = _apply_task_state_decision(
            state,
            context(sequence_id=sequence_id),
            decision(
                snapshot_id=f"a::a::121::0::{sequence_id}",
                task_clauses=(),
                clause_transition="recover",
                confidence=0.80,
                evidence=("hallway extends forward",),
            ),
            snapshot_sim_stamp_ns=(sequence_id + 2) * 1_000_000_000,
            transition_minimum_confidence=0.70,
        )
    assert state["active_clause_id"] == 1
    assert state["active_clause_semantic_confirmations"] == 0


def test_first_online_state_is_initialized_before_step3_request(
    tmp_path: Path,
) -> None:
    node = object.__new__(TimeoutAdvisorNode)
    node._lock = threading.Lock()
    node._task_states = {}
    node._task_state_root = tmp_path / "task_state"
    node._task_state_root.mkdir()

    state = node._get_task_state(context())

    assert state["clauses"] == [
        "Walk straight into the kitchen",
        "Turn left and exit the kitchen.",
    ]
    assert state["active_clause_id"] == 1
    assert state["decomposition_source"] == "deterministic_exact_span_fallback"
    assert (node._task_state_root / "events.jsonl").is_file()
