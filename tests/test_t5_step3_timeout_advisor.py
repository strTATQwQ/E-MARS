from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.t5_step3_timeout_advisor_node as timeout_advisor
from slow_planner.base import StructuredPlannerDecision
from internvla_t4_sensors.internvla_t4_sensors.step3_arrival_gate import (
    APPLY_BOUNDED_ESCAPE,
    CONTINUE_NAVIGATION,
    REQUEST_CONFIRMATION,
    TERMINATE_ASSISTED,
    arrival_transition,
    completed_motion_action,
    is_unconfirmed_model_stop,
    model_stop_escape_transition,
    pending_model_action,
)
from scripts.t5_step3_timeout_advisor_node import (
    CAMERA_ORDER,
    PRIMITIVES,
    TimeoutAdvisorError,
    _arrival_outcome,
    _bounded_visible_motion_advice,
    _context,
    _decision_audit_fields,
    _instruction_turn_fallback,
    _planner_request,
    _task_clause_motion_fallback,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stateful_oracle_candidate_is_completion_sim_only() -> None:
    candidate = json.loads(
        (
            ROOT
            / "configs/internnav_t5/step3_stateful_oracle_candidate.json"
        ).read_text(encoding="utf-8")
    )
    assert candidate["runtime_policy"] == "completion_sim"
    assert candidate["promotion_status"] == "REJECTED_SCREEN_433_121"
    assert candidate["runtime"]["termination_mode"] == "oracle_termination"
    assert candidate["components"]["task_state_advice"] == {
        "source": "SV09",
        "maximum_nonterminal_model_hold_deferrals": 1,
    }
    refresh = candidate["components"]["post_override_queue_refresh"]
    assert refresh["invalidate_action_queue"] is True
    assert refresh["trigger"] == "motion_timeout_after_confirmed_safe_stop_only"
    assert refresh["task_state_checkpoint_preserves_queue"] is True
    assert refresh["preserve_system2_latent"] is True
    assert refresh["preserve_policy_history"] is True
    assert candidate["claims"]["proves_navigation_reach"] is False
    assert candidate["claims"]["mechanism_only"] is True
    assert candidate["claims"]["credits_internvla_stop"] is False
    assert candidate["claims"]["credits_step3_arrival"] is False
    assert candidate["real_go2_allowed"] is False


def valid_context() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "motion_timeout_after_confirmed_safe_stop",
        "episode_id": "a::episode-1",
        "reset_generation": 0,
        "trigger_sequence_id": 7,
        "trigger_request_id": "a::episode-1:0:7",
        "expected_sequence_id": 8,
        "excluded_action": 1,
        "instruction": "walk to the doorway",
        "camera_order": list(CAMERA_ORDER),
        "advisor_round": 0,
    }


def valid_arrival_context(
    *, advisor_round: int = 1, model_stop_candidate: bool = False
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "arrival_check_after_completed_motion",
        "episode_id": "a::episode-1",
        "reset_generation": 0,
        "trigger_sequence_id": 7,
        "trigger_request_id": "a::episode-1:0:7",
        "expected_sequence_id": 8,
        "completed_action": 1,
        "instruction": "walk to the doorway",
        "camera_order": list(CAMERA_ORDER),
        "camera_sensor_stamp_ns": 7_000_000_000,
        "minimum_snapshot_sim_stamp_ns": 0,
        "advisor_round": advisor_round,
        "required_confirmations": 2,
        "model_stop_candidate": model_stop_candidate,
    }


def valid_task_checkpoint_context() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "task_state_checkpoint_after_completed_motion",
        "episode_id": "a::episode-1",
        "reset_generation": 0,
        "trigger_sequence_id": 7,
        "trigger_request_id": "a::episode-1:0:7",
        "expected_sequence_id": 8,
        "completed_action": 1,
        "instruction": "walk to the doorway",
        "camera_order": list(CAMERA_ORDER),
        "camera_sensor_stamp_ns": 7_000_000_000,
        "advisor_round": 0,
    }


def test_timeout_context_is_identity_bound_and_excludes_timed_out_action() -> None:
    value = _context(valid_context())
    assert value["expected_sequence_id"] == 8
    assert set(PRIMITIVES) - {value["excluded_action"]} == {2, 3}


def test_model_stop_request_filters_the_recent_timed_out_escape(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "snapshot.json"
    sidecar.write_text(
        json.dumps({"sim_stamp_before_ns": 7_000_000_000}),
        encoding="utf-8",
    )
    images = []
    for identity in CAMERA_ORDER:
        image = tmp_path / f"{identity}.jpg"
        image.write_bytes(b"jpeg")
        images.append({"identity": identity, "path": image.name})
    context = valid_arrival_context(model_stop_candidate=True)
    context["excluded_action"] = 1
    context["recent_timeout_sequence_id"] = 6

    value = _context(context)
    request = _planner_request(
        value,
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
    )

    assert [item.frontier_id for item in request.candidate_frontiers] == [2, 3]
    assert "1=short_forward timed out and is illegal" in request.instruction
    assert "recent_timed_out_escape=6:1=short_forward" in request.compact_history


@pytest.mark.parametrize(
    ("excluded_action", "recent_timeout_sequence_id"),
    [(True, 6), (4, 6), (1, True), (1, 7)],
)
def test_model_stop_timeout_exclusion_rejects_invalid_identity(
    excluded_action: object, recent_timeout_sequence_id: object
) -> None:
    context = valid_arrival_context(model_stop_candidate=True)
    context["excluded_action"] = excluded_action
    context["recent_timeout_sequence_id"] = recent_timeout_sequence_id

    with pytest.raises(TimeoutAdvisorError):
        _context(context)


def test_non_model_arrival_cannot_carry_escape_exclusion() -> None:
    context = valid_arrival_context(model_stop_candidate=False)
    context["excluded_action"] = 1
    context["recent_timeout_sequence_id"] = 6

    with pytest.raises(TimeoutAdvisorError):
        _context(context)


def test_lane_b_timeout_context_and_snapshot_identity_are_lane_scoped(
    tmp_path: Path,
) -> None:
    context = valid_context()
    context["episode_id"] = "b::episode-1"
    context["trigger_request_id"] = "b::episode-1:0:7"
    value = _context(context)

    sidecar = tmp_path / "snapshot.json"
    sidecar.write_text(
        json.dumps({"sim_stamp_before_ns": 7_000_000_000}), encoding="utf-8"
    )
    images = []
    for identity in CAMERA_ORDER:
        image = tmp_path / f"{identity}.jpg"
        image.write_bytes(b"jpeg")
        images.append({"identity": identity, "path": image.name})

    request = _planner_request(
        value,
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
    )

    assert request.episode_id == "b::episode-1"
    assert request.snapshot_id == "b::b::episode-1::0::7"


def test_task_checkpoint_is_identity_bound_and_keeps_all_primitives() -> None:
    value = _context(valid_task_checkpoint_context())
    assert value["expected_sequence_id"] == 8
    assert value["completed_action"] == 1


def test_task_checkpoint_prompt_updates_active_clause_with_all_views(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "snapshot.json"
    sidecar.write_text(
        json.dumps({"sim_stamp_before_ns": 7_000_000_000}),
        encoding="utf-8",
    )
    images = []
    for identity in CAMERA_ORDER:
        image = tmp_path / f"{identity}.jpg"
        image.write_bytes(b"jpeg")
        images.append({"identity": identity, "path": image.name})
    task_state = {
        "instruction": "walk to the doorway",
        "clauses": ["walk to the doorway"],
        "active_clause_id": 1,
        "revision": 1,
    }

    request = _planner_request(
        valid_task_checkpoint_context(),
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
        task_state=task_state,
    )

    assert [item.frontier_id for item in request.candidate_frontiers] == [1, 2, 3]
    assert "update the frozen active task clause" in request.instruction
    assert "do not silently follow a later clause" in request.instruction
    assert request.compact_history[0] == "completed_action=1=short_forward"


@pytest.mark.parametrize(
    ("clause", "expected"),
    [
        ("Walk straight into the kitchen area", 1),
        ("Turn left and exit the kitchen", 2),
        ("Turn right into the hallway", 3),
        ("Travel to the end of the hallway where there is a vase", 1),
        ("Take a left and go forward until you reach the doorway", 2),
        ("Make a right turn at the first right corner", 3),
        ("Pass the clock on your right", None),
        ("Find the kitchen", None),
    ],
)
def test_task_clause_fallback_is_bounded_and_non_oracle(
    tmp_path: Path, clause: str, expected: int | None
) -> None:
    sidecar = tmp_path / "snapshot.json"
    sidecar.write_text(
        json.dumps({"sim_stamp_before_ns": 7_000_000_000}),
        encoding="utf-8",
    )
    images = []
    for identity in CAMERA_ORDER:
        image = tmp_path / f"{identity}.jpg"
        image.write_bytes(b"jpeg")
        images.append({"identity": identity, "path": image.name})
    context = valid_task_checkpoint_context()
    context["instruction"] = clause
    request = _planner_request(
        context,
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
        task_state={
            "instruction": str(context["instruction"]),
            "clauses": [clause],
            "active_clause_id": 1,
            "revision": 1,
        },
    )
    state = {"clauses": [clause], "active_clause_id": 1}

    assert _task_clause_motion_fallback(context, request, (), state) == expected
    if expected is not None:
        assert (
            _task_clause_motion_fallback(
                context, request, (expected, expected), state
            )
            is None
        )


def test_task_clause_fallback_applies_after_motion_timeout(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "snapshot.json"
    sidecar.write_text(
        json.dumps({"sim_stamp_before_ns": 7_000_000_000}),
        encoding="utf-8",
    )
    images = []
    for identity in CAMERA_ORDER:
        image = tmp_path / f"{identity}.jpg"
        image.write_bytes(b"jpeg")
        images.append({"identity": identity, "path": image.name})
    clause = "Walk straight into the kitchen area"
    state = {"clauses": [clause], "active_clause_id": 1}

    context = valid_context()
    context["instruction"] = clause
    context["excluded_action"] = 3
    request = _planner_request(
        context,
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
        task_state={
            "instruction": clause,
            "clauses": [clause],
            "active_clause_id": 1,
            "revision": 1,
        },
    )
    assert _task_clause_motion_fallback(context, request, (), state) == 1

    context["excluded_action"] = 1
    request = _planner_request(
        context,
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
        task_state={
            "instruction": clause,
            "clauses": [clause],
            "active_clause_id": 1,
            "revision": 1,
        },
    )
    assert _task_clause_motion_fallback(context, request, (), state) is None


def test_unconfirmed_model_stop_abstain_uses_active_clause_bounded_escape(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "snapshot.json"
    sidecar.write_text(
        json.dumps({"sim_stamp_before_ns": 7_000_000_000}),
        encoding="utf-8",
    )
    images = []
    for identity in CAMERA_ORDER:
        image = tmp_path / f"{identity}.jpg"
        image.write_bytes(b"jpeg")
        images.append({"identity": identity, "path": image.name})
    clause = "Turn left and exit the kitchen and stop there"
    state = {"clauses": [clause], "active_clause_id": 1}
    context = valid_arrival_context(model_stop_candidate=True)
    context["instruction"] = "Walk into the kitchen. " + clause
    request = _planner_request(
        context,
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
        task_state={
            "instruction": str(context["instruction"]),
            "clauses": [clause],
            "active_clause_id": 1,
            "revision": 2,
        },
    )

    assert _task_clause_motion_fallback(context, request, (), state) == 2
    context["model_stop_candidate"] = False
    assert _task_clause_motion_fallback(context, request, (), state) is None


def test_model_stop_forward_timeout_uses_explicit_side_cue_as_legal_turn(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "snapshot.json"
    sidecar.write_text(
        json.dumps({"sim_stamp_before_ns": 7_000_000_000}),
        encoding="utf-8",
    )
    images = []
    for identity in CAMERA_ORDER:
        image = tmp_path / f"{identity}.jpg"
        image.write_bytes(b"jpeg")
        images.append({"identity": identity, "path": image.name})
    clause = "walk into the sitting area on the left and wait there"
    state = {"clauses": [clause], "active_clause_id": 1}
    context = valid_arrival_context(model_stop_candidate=True)
    context["instruction"] = "turn right after the couch. " + clause
    context["excluded_action"] = 1
    context["recent_timeout_sequence_id"] = 6
    request = _planner_request(
        context,
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
        task_state={
            "instruction": str(context["instruction"]),
            "clauses": [clause],
            "active_clause_id": 1,
            "revision": 2,
        },
    )

    assert [item.frontier_id for item in request.candidate_frontiers] == [2, 3]
    assert _task_clause_motion_fallback(context, request, (), state) == 2

    context.pop("excluded_action")
    context.pop("recent_timeout_sequence_id")
    unrestricted = _planner_request(
        context,
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
        task_state={
            "instruction": str(context["instruction"]),
            "clauses": [clause],
            "active_clause_id": 1,
            "revision": 2,
        },
    )
    assert _task_clause_motion_fallback(context, unrestricted, (), state) == 1


def test_non_model_arrival_shadow_has_no_frontier_or_stop_authority(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "snapshot.json"
    sidecar.write_text(
        json.dumps({"sim_stamp_before_ns": 7_000_000_000}),
        encoding="utf-8",
    )
    images = []
    for identity in CAMERA_ORDER:
        image = tmp_path / f"{identity}.jpg"
        image.write_bytes(b"jpeg")
        images.append({"identity": identity, "path": image.name})
    context = valid_arrival_context()
    context["oracle_terminal_shadow"] = True
    context["instruction"] = "Walk down the hall. Stop beside the white rug."

    request = _planner_request(
        context,
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
        task_state={
            "instruction": str(context["instruction"]),
            "clauses": ["Walk down the hall", "Stop beside the white rug."],
            "active_clause_id": 2,
            "revision": 3,
        },
    )

    assert request.candidate_frontiers == ()
    assert "goal_region: yes|no - <visible fact>" in request.instruction
    assert "semantic_arrival: yes|no - <visible fact>" in request.instruction
    assert "evidence must contain exactly two entries" in request.instruction
    assert "target_found=true only" in request.instruction
    assert "no movement or terminal STOP authority" in request.instruction
    assert "The existing fresh two-frame gate remains authoritative" in (
        request.instruction
    )
    assert "termination_candidate=oracle_terminal_shadow" in request.compact_history
    assert "task_state_active_clause=2/2@r3:Stop beside the white rug." in (
        request.compact_history
    )
    assert "task_state_terminal_clause=Stop beside the white rug." in (
        request.compact_history
    )
    assert any(
        item.startswith("task_state_v1=") for item in request.compact_history
    )

    from slow_planner.step3_task_state import Step3TaskStateSlowPlanner

    prompt = object.__new__(Step3TaskStateSlowPlanner).format_prompt(request)
    assert "ARRIVAL_SHADOW_ONLY" in prompt
    assert '"active_clause_id":2' in prompt


def test_decision_audit_fields_are_bounded_inert_and_do_not_log_raw_text() -> None:
    fields = _decision_audit_fields(
        SimpleNamespace(
            scene_summary="hallway <private>\nvisible",
            target_evidence=(
                "goal_region: rug beside robot",
                "semantic_arrival: white rug matches terminal clause",
            ),
            fallback_used=True,
            fallback_reason="schema `fallback`",
            raw_text="hidden chain of thought",
        )
    )

    assert fields == {
        "scene_summary": "hallway private visible",
        "target_evidence": [
            "goal_region: rug beside robot",
            "semantic_arrival: white rug matches terminal clause",
        ],
        "fallback_used": True,
        "fallback_reason": "schema fallback",
    }
    assert "raw_text" not in fields


def test_task_state_arrival_evidence_is_preserved_for_shadow_audit() -> None:
    fields = _decision_audit_fields(
        SimpleNamespace(
            evidence=(
                "goal_region: yes - rug nearby",
                "semantic_arrival: no - rug ahead",
            ),
            fallback_used=False,
            fallback_reason="",
        )
    )

    assert fields["target_evidence"] == [
        "goal_region: yes - rug nearby",
        "semantic_arrival: no - rug ahead",
    ]


def test_visible_step3_checkpoint_action_is_not_replaced_by_clause_lexicon() -> None:
    context = valid_task_checkpoint_context()
    context["instruction"] = (
        "Turn right after the couch, then walk into the sitting area on the "
        "left and wait there."
    )

    assert _bounded_visible_motion_advice(context, 2) == (
        2,
        "bounded_visible_task_primitive",
    )


def test_visible_step3_timeout_action_keeps_escape_reason() -> None:
    assert _bounded_visible_motion_advice(valid_context(), 3) == (
        3,
        "bounded_visible_escape_primitive",
    )


def test_timeout_prompt_names_only_current_legal_action_ids(tmp_path: Path) -> None:
    sidecar = tmp_path / "snapshot.json"
    sidecar.write_text(
        json.dumps({"sim_stamp_before_ns": 7_000_000_000}),
        encoding="utf-8",
    )
    images = []
    for index, identity in enumerate(CAMERA_ORDER):
        image = tmp_path / f"{identity}.jpg"
        image.write_bytes(b"jpeg")
        images.append({"identity": identity, "path": image.name})
    request = _planner_request(
        valid_context(),
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
    )

    assert [item.frontier_id for item in request.candidate_frontiers] == [2, 3]
    assert "1=short_forward" in request.instruction
    assert "illegal for this response" in request.instruction
    assert "2=short_turn_left, 3=short_turn_right" in request.instruction
    assert "do not repeat the timed-out ID" in request.instruction
    assert request.compact_history[0] == "timed_out_action=1=short_forward"
    assert request.compact_history[1] == (
        "legal_actions=2=short_turn_left, 3=short_turn_right"
    )
    assert request.compact_history[2] == "prior_timeout_advice=none"


def test_timeout_turn_prefers_forward_over_immediate_counter_turn(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(
        json.dumps({"sim_stamp_before_ns": 7_000_000_000}),
        encoding="utf-8",
    )
    images = []
    for identity in CAMERA_ORDER:
        image = tmp_path / f"{identity}.jpg"
        image.write_bytes(b"jpeg")
        images.append({"identity": identity, "path": image.name})
    context = valid_context()
    context["excluded_action"] = 3

    request = _planner_request(
        context,
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
    )

    assert [item.frontier_id for item in request.candidate_frontiers] == [1, 2]
    assert "Prefer 1=short_forward" in request.instruction
    assert "Do not choose 2=short_turn_left merely to undo" in request.instruction
    assert "only when short_forward is visibly blocked" in request.instruction


def test_timeout_prompt_tracks_bounded_prior_advice_without_shared_schema(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(
        json.dumps({"sim_stamp_before_ns": 7_000_000_000}),
        encoding="utf-8",
    )
    images = []
    for identity in CAMERA_ORDER:
        image = tmp_path / f"{identity}.jpg"
        image.write_bytes(b"jpeg")
        images.append({"identity": identity, "path": image.name})

    request = _planner_request(
        valid_context(),
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
        prior_advised_actions=(2, 3, 2),
    )

    assert request.compact_history[2] == "prior_timeout_advice=2,3,2"
    assert "best advances the navigation instruction" in request.instruction
    assert "clearance is a safety constraint, not the objective" in request.instruction
    assert "Avoid repeating the recent timeout-advice cycle" in request.instruction


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        (
            "Walk straight into the kitchen. Turn left and exit the kitchen.",
            2,
        ),
        ("Walk to the doorway, then turn right.", 3),
        ("Walk to the doorway.", None),
    ],
)
def test_instruction_turn_fallback_is_narrow_and_non_oracle(
    tmp_path: Path, instruction: str, expected: int | None
) -> None:
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(
        json.dumps({"sim_stamp_before_ns": 7_000_000_000}),
        encoding="utf-8",
    )
    images = []
    for identity in CAMERA_ORDER:
        image = tmp_path / f"{identity}.jpg"
        image.write_bytes(b"jpeg")
        images.append({"identity": identity, "path": image.name})
    context = valid_context()
    context["instruction"] = instruction
    request = _planner_request(
        context,
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
    )

    assert _instruction_turn_fallback(context, request, ()) == expected


def test_instruction_turn_fallback_refuses_turn_timeout_and_repeat(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(
        json.dumps({"sim_stamp_before_ns": 7_000_000_000}),
        encoding="utf-8",
    )
    images = []
    for identity in CAMERA_ORDER:
        image = tmp_path / f"{identity}.jpg"
        image.write_bytes(b"jpeg")
        images.append({"identity": identity, "path": image.name})
    context = valid_context()
    context["instruction"] = "Walk straight, then turn left."
    request = _planner_request(
        context,
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
    )

    assert _instruction_turn_fallback(context, request, (2,)) is None
    context["excluded_action"] = 3
    request = _planner_request(
        context,
        {"sidecar": sidecar.name, "images": images},
        tmp_path,
    )
    assert _instruction_turn_fallback(context, request, ()) is None


def test_snapshot_capture_uses_trigger_not_next_action_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context = valid_context()
    request_path = tmp_path / "request.json"
    observed: dict[str, object] = {}

    def matching_ack(_path: Path, _request_id: str, _remaining: float) -> dict:
        observed.update(json.loads(request_path.read_text(encoding="utf-8")))
        return {"status": "CAPTURED"}

    monkeypatch.setattr(timeout_advisor, "wait_for_matching_ack", matching_ack)
    monkeypatch.setattr(
        timeout_advisor,
        "validate_capture",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    timeout_advisor._request_snapshot(
        context=context,
        request_path=request_path,
        ack_path=tmp_path / "ack.json",
        result_root=tmp_path,
        contract_path=tmp_path / "contract.json",
        deadline=timeout_advisor.time.monotonic() + 1.0,
    )
    assert observed["sequence_id"] == context["trigger_sequence_id"] == 7
    assert observed["sequence_id"] != context["expected_sequence_id"]


def test_snapshot_capture_retries_one_known_render_realign(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context = valid_context()
    request_path = tmp_path / "request.json"
    observed: list[dict[str, object]] = []

    def matching_ack(_path: Path, request_id: str, _remaining: float) -> dict:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        observed.append(request)
        request_path.unlink()
        if len(observed) == 1:
            return {
                "status": "WARN_CAPTURE_FAILED",
                "request_id": request_id,
                "episode_id": context["episode_id"],
                "reset_generation": context["reset_generation"],
                "sequence_id": context["trigger_sequence_id"],
                "error": (
                    "Rev-C sensor t5_revc_front ReferenceTime changed at "
                    "paused render barrier"
                ),
            }
        return {"status": "CAPTURED", "request_id": request_id}

    monkeypatch.setattr(timeout_advisor, "wait_for_matching_ack", matching_ack)
    monkeypatch.setattr(
        timeout_advisor,
        "validate_capture",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    result = timeout_advisor._request_snapshot(
        context=context,
        request_path=request_path,
        ack_path=tmp_path / "ack.json",
        result_root=tmp_path,
        contract_path=tmp_path / "contract.json",
        deadline=timeout_advisor.time.monotonic() + 1.0,
    )

    assert result["status"] == "PASS"
    assert len(observed) == 2
    assert observed[0]["request_id"] != observed[1]["request_id"]
    assert {item["sequence_id"] for item in observed} == {7}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_id", "c::episode-1"),
        ("expected_sequence_id", 9),
        ("excluded_action", True),
        ("excluded_action", 4),
        ("camera_order", ["front", "rear"]),
    ],
)
def test_timeout_context_rejects_cross_lane_or_unbounded_values(
    field: str, value: object
) -> None:
    context = valid_context()
    context[field] = value
    with pytest.raises(TimeoutAdvisorError):
        _context(context)


def test_arrival_context_requires_two_identity_bound_rounds() -> None:
    assert _context(valid_arrival_context())["advisor_round"] == 1
    assert _context(valid_arrival_context(advisor_round=2))["advisor_round"] == 2
    invalid = valid_arrival_context(advisor_round=2)
    invalid["minimum_snapshot_sim_stamp_ns"] = -1
    with pytest.raises(TimeoutAdvisorError):
        _context(invalid)
    invalid = valid_arrival_context()
    invalid["model_stop_candidate"] = 1
    with pytest.raises(TimeoutAdvisorError):
        _context(invalid)


def test_safe_hold_attributes_timeout_and_arrival_contexts_without_key_error() -> None:
    assert pending_model_action(valid_context()) == 1
    assert pending_model_action(valid_arrival_context()) == 1
    assert pending_model_action(
        {"kind": "arrival_check_after_completed_motion", "completed_action": True}
    ) == 0
    assert pending_model_action(
        {"kind": "motion_timeout_after_confirmed_safe_stop"}
    ) == 0
    assert pending_model_action(
        {
            "kind": "arrival_check_after_completed_motion",
            "completed_action": 3,
            "model_stop_candidate": True,
        }
    ) == 0


def test_model_stop_requires_independent_arrival_confirmation() -> None:
    assert is_unconfirmed_model_stop(
        {"stop": True, "model_discrete_action": 0}
    )
    assert not is_unconfirmed_model_stop(
        {
            "stop": True,
            "model_discrete_action": 0,
            "step3_arrival_confirmed": True,
        }
    )
    assert not is_unconfirmed_model_stop(
        {"stop": True, "model_discrete_action": 1}
    )
    assert not is_unconfirmed_model_stop(
        {"stop": False, "model_discrete_action": 0}
    )
    assert not is_unconfirmed_model_stop(
        {"stop": True, "model_discrete_action": True}
    )
    oracle_rejected = {
        "stop": False,
        "model_stop": True,
        "model_discrete_action": 0,
        "geometric_success": False,
    }
    assert not is_unconfirmed_model_stop(oracle_rejected)
    assert is_unconfirmed_model_stop(oracle_rejected, oracle_rejected=True)
    assert not is_unconfirmed_model_stop(
        {**oracle_rejected, "geometric_success": True}, oracle_rejected=True
    )
    assert not is_unconfirmed_model_stop(
        {**oracle_rejected, "model_stop": False}, oracle_rejected=True
    )
    assert not is_unconfirmed_model_stop(
        {**oracle_rejected, "model_discrete_action": 1}, oracle_rejected=True
    )


@pytest.mark.parametrize(
    ("decision", "confidence", "fallback", "expected"),
    [
        ("target_found", 0.80, False, "ARRIVED"),
        ("target_found", 0.79, False, "NOT_ARRIVED"),
        ("target_found", 0.99, True, "NOT_ARRIVED"),
        ("abstain", 0.99, False, "NOT_ARRIVED"),
    ],
)
def test_arrival_outcome_is_confidence_bounded_and_fail_closed(
    decision: str, confidence: float, fallback: bool, expected: str
) -> None:
    status, _ = _arrival_outcome(
        SimpleNamespace(
            decision=decision,
            confidence=confidence,
            fallback_used=fallback,
        ),
        0.80,
    )
    assert status == expected


def test_arrival_outcome_accepts_real_step3_structured_target_found() -> None:
    decision = StructuredPlannerDecision(
        episode_id="a::episode-1",
        snapshot_id="a::a::episode-1::0::7",
        decision="abstain",
        frontier_id=None,
        target_relative_xz=None,
        confidence=0.80,
        scene_summary="white rug visible in hallway",
        target_evidence=("white rug is beside the robot",),
        blocked_directions=(),
        recommended_frontier=None,
        target_found=True,
        abstain=True,
    )

    status, reason = _arrival_outcome(decision, 0.80)

    assert status == "ARRIVED"
    assert reason == "step3_visible_destination_reached"


def test_arrival_gate_requires_two_rounds_before_assisted_termination() -> None:
    first = arrival_transition(
        valid_arrival_context(advisor_round=1),
        {"status": "ARRIVED", "snapshot_sim_stamp_ns": 200},
        required_confirmations=2,
    )
    second = arrival_transition(
        valid_arrival_context(advisor_round=2),
        {"status": "ARRIVED", "snapshot_sim_stamp_ns": 230},
        required_confirmations=2,
    )
    negative = arrival_transition(
        valid_arrival_context(advisor_round=1),
        {"status": "NOT_ARRIVED"},
        required_confirmations=2,
    )

    assert first.action == REQUEST_CONFIRMATION
    assert first.snapshot_sim_stamp_ns == 200
    assert second.action == TERMINATE_ASSISTED
    assert second.snapshot_sim_stamp_ns == 230
    assert negative.action == CONTINUE_NAVIGATION


def test_model_stop_escape_requires_two_direction_consistent_rounds() -> None:
    first = model_stop_escape_transition(
        valid_arrival_context(
            advisor_round=1, model_stop_candidate=True
        ),
        {
            "status": "NOT_ARRIVED",
            "advised_action": 2,
            "snapshot_sim_stamp_ns": 200,
        },
        escape_count=0,
        escape_limit=2,
    )
    matching = model_stop_escape_transition(
        {
            **valid_arrival_context(
                advisor_round=2, model_stop_candidate=True
            ),
            "first_advised_action": 2,
        },
        {
            "status": "NOT_ARRIVED",
            "advised_action": 2,
            "snapshot_sim_stamp_ns": 230,
        },
        escape_count=0,
        escape_limit=2,
    )
    conflicting = model_stop_escape_transition(
        {
            **valid_arrival_context(
                advisor_round=2, model_stop_candidate=True
            ),
            "first_advised_action": 2,
        },
        {
            "status": "NOT_ARRIVED",
            "advised_action": 3,
            "snapshot_sim_stamp_ns": 230,
        },
        escape_count=0,
        escape_limit=2,
    )
    excluded = model_stop_escape_transition(
        {
            **valid_arrival_context(
                advisor_round=2, model_stop_candidate=True
            ),
            "first_advised_action": 1,
            "excluded_action": 1,
        },
        {
            "status": "NOT_ARRIVED",
            "advised_action": 1,
            "snapshot_sim_stamp_ns": 230,
        },
        escape_count=0,
        escape_limit=2,
    )
    burst_exhausted = model_stop_escape_transition(
        {
            **valid_arrival_context(
                advisor_round=2, model_stop_candidate=True
            ),
            "first_advised_action": 2,
        },
        {
            "status": "NOT_ARRIVED",
            "advised_action": 2,
            "snapshot_sim_stamp_ns": 230,
        },
        escape_count=1,
        escape_limit=1,
    )

    assert first.action == REQUEST_CONFIRMATION
    assert first.advised_action == 2
    assert matching.action == APPLY_BOUNDED_ESCAPE
    assert matching.advised_action == 2
    assert conflicting.action == CONTINUE_NAVIGATION
    assert excluded.action == CONTINUE_NAVIGATION
    assert burst_exhausted.action == CONTINUE_NAVIGATION


def test_arrival_followup_retains_the_completed_bounded_action() -> None:
    assert completed_motion_action({"action": 1}) == 1
    assert completed_motion_action({"completed_action": 3}) == 3
    assert completed_motion_action({"completed_action": True}) is None
    assert completed_motion_action({"completed_action": 4}) is None


def test_revc_geometry_and_step3_runtime_are_frozen() -> None:
    contract = json.loads(
        (ROOT / "configs/internnav_t5/revc_four_camera_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["camera_order"] == list(CAMERA_ORDER)
    assert [row["position_F_M_mm"] for row in contract["cameras"]] == [
        [30.0, 51.962, 20.0],
        [60.0, 0.0, 20.0],
        [30.0, -51.962, 20.0],
        [-60.0, 0.0, 20.0],
    ]
    config = (
        ROOT / "configs/slow_models/step3_vl_10b_timeout_advisor.yaml"
    ).read_text(encoding="utf-8")
    assert "revision: 5026053b0c2f5dfaa08fc2d149384162c3c8bca1" in config
    assert "precision_mode: bf16" in config
    assert "redact_raw_text: true" in config
    assert "planner_mode: task_state_v1" in config
    assert "bind: ${STEP3_TIMEOUT_BIND}" in config


def test_advisor_has_no_direct_motion_or_terminal_stop_authority() -> None:
    advisor = (ROOT / "scripts/t5_step3_timeout_advisor_node.py").read_text(
        encoding="utf-8"
    )
    client = (
        ROOT
        / "internvla_t4_sensors/internvla_t4_sensors/client_node.py"
    ).read_text(encoding="utf-8")
    assert 'create_publisher(\n            String, "/internvla/t5_step3_timeout_advice"' in advisor
    assert "cmd_vel" not in advisor
    assert '"status": "ADVISE"' not in advisor  # emitted through bounded fields
    assert 'command.action_source = 1' in client
    assert 'command.trajectory_valid = False' in client
    assert '"step3_assisted_stop": True' in client
    assert '"step3_assisted_arrival"' in client
    assert '"internvla_model_stop_step3_arrival_confirmed"' in client
    assert '"internvla_model_stop_step3_unconfirmed"' not in client
    assert '"internvla_model_stop_retained"' not in client
    assert "arrival_terminal = (dict(pending), dict(advice))" in client
    assert "arrival_fallback = (dict(pending), dict(advice))" in client
    assert '"step3_arrival_gate_only": True' in client
    assert "_gate_internvla_model_stop(result)" in client
    assert 'result["internvla_stop_candidate"] = True' in client
    assert "step3_model_stop_rejected_with_bounded_escape" in advisor
    assert "active_clause_bounded_escape_after_" in advisor
    assert "InternVLA proposed STOP" in advisor
    assert "STEP3_ARRIVAL_REQUIRED_CONFIRMATIONS = 2" in client
    assert "_queued_arrival_confirmation" in advisor
    assert "identity[:4] == self._active_identity[:4]" in advisor
    assert 'self.system2_replan_policy == "observation_bound"' in client
    assert 'INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"' in client
    assert 'result["geometric_success"] = geometric_success' in client
    assert 'result["model_stop_success"] = model_stop_success' in client
    assert 'result["semantic_arrival_confirmed"] = False' in client
    assert '"oracle_termination_geometric_success"' in client
    assert 'if self._termination_mode != "oracle_termination":' in client
    assert "self._gate_internvla_model_stop(result, oracle_rejected=True)" in client
    assert '== "model_stop_escape"' in client
    assert "and not current_model_stop_escape" in client
    assert "_begin_oracle_terminal_shadow" in client
    assert "_finish_oracle_terminal_shadow" in client
    assert '"oracle_terminal_shadow": True' in client
    assert 'result["step3_oracle_terminal_shadow_confirmed"] = confirmed' in client
    assert '"oracle_termination_preserved": True' in client
    assert '"stop_shadow_internvla_model_stop"' in client
    assert '"stop_shadow_step3_arrival_single"' in client
    assert '"stop_shadow_step3_arrival_confirmed"' in client
    assert '"stop_shadow_step3_task_state_target_found"' in client


def test_timeout_advisor_has_one_escape_per_measured_motion_burst() -> None:
    client = (
        ROOT
        / "internvla_t4_sensors/internvla_t4_sensors/client_node.py"
    ).read_text(encoding="utf-8")
    assert 'INTERNVLA_T5_STEP3_TIMEOUT_MAX_INTERVENTIONS", "12"' in client
    assert "1 <= self._step3_timeout_max_interventions <= 12" in client
    assert "STEP3_MODEL_STOP_ESCAPE_BURST_MAX = 1" in client
    assert "escape_count=self._step3_model_stop_escape_burst_count" in client
    assert "self._step3_model_stop_escape_burst_count += 1" in client
    assert (
        '"step3_model_stop_escape_rearmed_after_measured_motion"' in client
    )
    assert "STEP3_TASK_STATE_ACTION_INTERVAL = 1" in client
    assert '"step3_task_state_checkpoint_semantic_only"' in client
    assert '"step3_model_queue_invalidated"' in client
    assert "OP_CLEAR_MODEL_CACHE" in client
    recovery_model = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/model_node.py"
    ).read_text(encoding="utf-8")
    assert 'identity.recovery_id.startswith("step3-refresh:")' in recovery_model
    assert '"step3_action_queue_invalidated"' in recovery_model
    assert '"policy_history_preserved": True' in recovery_model
    locked_step = client.split("def _step_arrays_locked", maxsplit=1)[1]
    assert locked_step.index("self._refresh_model_after_step3_override()") < (
        locked_step.index("result = super().step_arrays(")
    )
    assert '"task_state_checkpoint_after_completed_motion"' in client
    post_reset = client.split(
        "def _install_post_reset_sensor_barrier", maxsplit=1
    )[1].split("def _reset_motion_gate", maxsplit=1)[0]
    assert "self._step3_timeout_interventions = 0" in post_reset
    assert "self._step3_arrival_completed_actions = 0" in post_reset
    assert "self._step3_arrival_checks = 0" in post_reset
    assert "self._step3_last_completed_action = None" in post_reset
    assert "self._step3_model_stop_escapes = 0" in post_reset
    assert "self._step3_model_stop_escape_burst_count = 0" in post_reset
    assert "self._step3_model_stop_escape_motion = None" in post_reset
    assert "self._step3_model_refresh_pending = None" in post_reset
    assert "self._step3_model_refresh_count = 0" in post_reset


def test_model_stop_escape_confirmation_preserves_request_identity() -> None:
    client = (
        ROOT
        / "internvla_t4_sensors/internvla_t4_sensors/client_node.py"
    ).read_text(encoding="utf-8")
    arrival_publisher = client.split(
        "def _publish_step3_arrival_context", maxsplit=1
    )[1].split("def _maybe_publish_step3_arrival_context", maxsplit=1)[0]
    assert (
        'pending.get("trigger_request_id", pending.get("request_id", ""))'
        in arrival_publisher
    )


def test_x86_advisor_runs_inside_the_existing_ros_container() -> None:
    runner = (ROOT / "scripts/run_t5_distributed_isaac.sh").read_text(
        encoding="utf-8"
    )
    assert "setsid docker exec --user admin --workdir \"$root\"" in runner
    assert "pyzmq-27.1.0.dist-info" in runner
    assert '-e ROS_LOCALHOST_ONLY=0 -e "PYTHONPATH=$root:$advisor_python_deps"' in runner
    assert "source /opt/ros/jazzy/setup.bash" in runner
    assert "source /workspaces/isaac/install/setup.bash" in runner
    assert runner.index('python3 -c "import rclpy, zmq, slow_planner') < runner.index(
        'printf "%s\\n" "$$" >"$STEP3_TIMEOUT_ADVISOR_PID_FILE"'
    )
    assert "step3_timeout_advisor.container.pid" in runner
    assert "stop_step3_timeout_advisor" in runner
    assert 'tcp://$edge_ip:8200' in runner


def test_timeout_service_is_owned_by_selected_lane_run_root() -> None:
    service = (ROOT / "scripts/run_t5_step3_timeout_service.sh").read_text(
        encoding="utf-8"
    )
    lane = (ROOT / "scripts/run_t5_dgx_lane.sh").read_text(encoding="utf-8")
    coordinator = (
        ROOT / "coordination/run_t5_fast_lane_online.sh"
    ).read_text(encoding="utf-8")

    assert 'a) expected_user=railgun; lane_ip=10.100.100.128; expected_lease=lane-a' in service
    assert 'b) expected_user=rail; lane_ip=10.100.120.122; expected_lease=lane-b' in service
    assert 'test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = "$expected_lease"' in service
    assert 'STEP3_TIMEOUT_BIND="tcp://$lane_ip:8200"' in service
    assert '"task_state": (' in service
    assert 'value.get("planner_mode") == "task_state_v1"' in service
    assert '[[ "$result_root" = "$HOME"/* ]]' in service
    assert "for _ in $(seq 1 100)" in service
    assert "for _ in $(seq 1 50)" in service
    assert "run_t5_step3_timeout_service.sh" in lane
    assert 'INTERNNAV_T5_RESOURCE_LEASE_ACK="$expected_lease"' in lane
    assert 'step3_log_path="$result_dir/logs/step3_timeout_service.log"' in lane
    assert 'test -f "$result_dir/step3_timeout_service/health.json"' in lane
    assert 'elif test "$component" = step3; then' in lane
    assert 'kill -TERM -- "-$pid"' in lane
    assert 'test "$step3_timeout_advisor" != 1 || step3_dgx_ports="8200"' in coordinator
