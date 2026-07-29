#!/usr/bin/env python3
"""Advise one bounded primitive or classify arrival from Rev-C snapshots.

The node has no velocity or terminal-stop publisher.  It only binds a verified
same-render-tick Rev-C snapshot to SlowPlanner protocol v1.  The client-side
termination arbiter, not this node, owns any eventual Step3-assisted STOP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from slow_planner.base import CandidateFrontier, OrderedImage, SlowPlannerRequest
from slow_planner.client import SlowPlannerClient
from slow_planner.step3_task_state import (
    TaskStateContext,
    TaskStatePlannerDecision,
    fallback_instruction_clauses,
)
from scripts.probe_t5_revc_snapshot_smoke import (
    CAMERA_ORDER,
    SmokeContractError,
    _atomic_json,
    _load_regular_json,
    validate_capture,
    wait_for_matching_ack,
)


CAMERA_POSES = {
    "front_left": (0.170, 0.051962, 0.200, math.radians(60.0), math.radians(-10.0)),
    "front": (0.200, 0.0, 0.200, 0.0, math.radians(-10.0)),
    "front_right": (0.170, -0.051962, 0.200, math.radians(-60.0), math.radians(-10.0)),
    "rear": (0.080, 0.0, 0.200, math.pi, math.radians(-10.0)),
}
PRIMITIVES = {
    1: CandidateFrontier(1, (0.0, 0.25), 0.25, 0.0),
    2: CandidateFrontier(2, (-0.18, 0.18), 0.255, -45.0),
    3: CandidateFrontier(3, (0.18, 0.18), 0.255, 45.0),
}
PRIMITIVE_LABELS = {
    1: "short_forward",
    2: "short_turn_left",
    3: "short_turn_right",
}
RENDER_REALIGN_ERROR = re.compile(
    r"Rev-C sensor t5_revc_(?:front_left|front|front_right|rear) "
    r"ReferenceTime changed at paused render barrier"
)


class TimeoutAdvisorError(RuntimeError):
    pass


_LOG_SECRET = re.compile(r"(?i)\b(?:hf|sk)[-_][a-z0-9_-]{8,}\b")


def _safe_log_text(value: Any, *, maximum_length: int = 160) -> str:
    """Return bounded inert audit text without model reasoning or credentials."""

    if not isinstance(value, str):
        return ""
    text = _LOG_SECRET.sub("[redacted]", value)
    text = re.sub(r"[\x00-\x1f\x7f<>`]", " ", text)
    return " ".join(text.split())[:maximum_length]


def _decision_audit_fields(decision: Any) -> dict[str, Any]:
    evidence = getattr(
        decision,
        "target_evidence",
        getattr(decision, "evidence", ()),
    )
    if not isinstance(evidence, (list, tuple)):
        evidence = ()
    return {
        "scene_summary": _safe_log_text(
            getattr(decision, "scene_summary", ""), maximum_length=160
        ),
        "target_evidence": [
            text
            for item in evidence[:2]
            if (text := _safe_log_text(item, maximum_length=96))
        ],
        "fallback_used": bool(getattr(decision, "fallback_used", False)),
        "fallback_reason": _safe_log_text(
            getattr(decision, "fallback_reason", ""), maximum_length=128
        ),
    }


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(descriptor, line.encode("utf-8")) != len(line.encode("utf-8")):
            raise OSError("short JSONL append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _context(value: dict[str, Any]) -> dict[str, Any]:
    kind = value.get("kind")
    episode_id = str(value.get("episode_id", ""))
    if (
        value.get("schema_version") != 1
        or kind
        not in {
            "motion_timeout_after_confirmed_safe_stop",
            "task_state_checkpoint_after_completed_motion",
            "arrival_check_after_completed_motion",
        }
        or re.fullmatch(r"[ab]::.+", episode_id) is None
        or isinstance(value.get("reset_generation"), bool)
        or not isinstance(value.get("reset_generation"), int)
        or isinstance(value.get("trigger_sequence_id"), bool)
        or not isinstance(value.get("trigger_sequence_id"), int)
        or isinstance(value.get("expected_sequence_id"), bool)
        or not isinstance(value.get("expected_sequence_id"), int)
        or value.get("expected_sequence_id") != value.get("trigger_sequence_id") + 1
        or value.get("camera_order") != list(CAMERA_ORDER)
        or not str(value.get("trigger_request_id", ""))
        or not str(value.get("instruction", "")).strip()
        or isinstance(value.get("advisor_round"), bool)
        or not isinstance(value.get("advisor_round"), int)
    ):
        raise TimeoutAdvisorError("invalid lane-scoped Step3 context")
    if kind == "motion_timeout_after_confirmed_safe_stop":
        if (
            value.get("advisor_round") != 0
            or isinstance(value.get("excluded_action"), bool)
            or not isinstance(value.get("excluded_action"), int)
            or value.get("excluded_action") not in PRIMITIVES
        ):
            raise TimeoutAdvisorError("invalid lane-scoped timeout context")
    elif kind == "task_state_checkpoint_after_completed_motion":
        if (
            value.get("advisor_round") != 0
            or isinstance(value.get("completed_action"), bool)
            or not isinstance(value.get("completed_action"), int)
            or value.get("completed_action") not in PRIMITIVES
            or isinstance(value.get("camera_sensor_stamp_ns"), bool)
            or not isinstance(value.get("camera_sensor_stamp_ns"), int)
            or value.get("camera_sensor_stamp_ns") <= 0
        ):
            raise TimeoutAdvisorError("invalid lane-scoped task-state checkpoint")
    else:
        if (
            value.get("advisor_round") not in {1, 2}
            or value.get("required_confirmations") != 2
            or isinstance(value.get("completed_action"), bool)
            or not isinstance(value.get("completed_action"), int)
            or value.get("completed_action") not in PRIMITIVES
            or isinstance(value.get("camera_sensor_stamp_ns"), bool)
            or not isinstance(value.get("camera_sensor_stamp_ns"), int)
            or value.get("camera_sensor_stamp_ns") <= 0
            or isinstance(value.get("minimum_snapshot_sim_stamp_ns"), bool)
            or not isinstance(value.get("minimum_snapshot_sim_stamp_ns"), int)
            or value.get("minimum_snapshot_sim_stamp_ns") < 0
            or not isinstance(value.get("model_stop_candidate", False), bool)
        ):
            raise TimeoutAdvisorError("invalid lane-scoped arrival context")
        first_advised_action = value.get("first_advised_action")
        if (
            first_advised_action is not None
            and (
                value.get("advisor_round") != 2
                or value.get("model_stop_candidate") is not True
                or isinstance(first_advised_action, bool)
                or not isinstance(first_advised_action, int)
                or first_advised_action not in PRIMITIVES
            )
        ):
            raise TimeoutAdvisorError("invalid model-STOP confirmation context")
        excluded_action = value.get("excluded_action")
        recent_timeout_sequence_id = value.get("recent_timeout_sequence_id")
        if (excluded_action is None) != (recent_timeout_sequence_id is None) or (
            excluded_action is not None
            and (
                value.get("model_stop_candidate") is not True
                or isinstance(excluded_action, bool)
                or not isinstance(excluded_action, int)
                or excluded_action not in PRIMITIVES
                or isinstance(recent_timeout_sequence_id, bool)
                or not isinstance(recent_timeout_sequence_id, int)
                or recent_timeout_sequence_id < 0
                or recent_timeout_sequence_id
                >= value.get("trigger_sequence_id", -1)
            )
        ):
            raise TimeoutAdvisorError("invalid recent model-STOP escape timeout")
    return value


def _request_snapshot(
    *,
    context: dict[str, Any],
    request_path: Path,
    ack_path: Path,
    result_root: Path,
    contract_path: Path,
    deadline: float,
) -> dict[str, Any]:
    # Capture the observation that produced the timed-out action.  The next
    # sequence is reserved for the advised action identity; Isaac's active
    # render identity remains the trigger sequence until that action begins.
    sequence_id = int(context["trigger_sequence_id"])
    lane_id = str(context["episode_id"]).split("::", 1)[0]
    for attempt in range(2):
        request_id = (
            f"{lane_id}::step3-timeout::{context['reset_generation']}::"
            f"{context['trigger_sequence_id']}::{attempt}::{time.time_ns()}"
        )
        request = {
            "schema_version": 1,
            "request_id": request_id,
            "episode_id": context["episode_id"],
            "reset_generation": context["reset_generation"],
            "sequence_id": sequence_id,
        }
        if request_path.exists() or request_path.is_symlink():
            raise TimeoutAdvisorError("Rev-C snapshot request path is occupied")
        _atomic_json(request_path, request)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutAdvisorError("snapshot deadline expired")
        ack = wait_for_matching_ack(ack_path, request_id, remaining)
        if ack.get("status") == "CAPTURED":
            sidecar_count = len(
                list((result_root / "revc_snapshots").glob("*/snapshot.json"))
            )
            return validate_capture(
                result_root,
                contract_path,
                request,
                ack,
                expected_sidecar_count=sidecar_count,
                profile=f"lane_{lane_id}_step3_timeout_advisor",
                expected_lane=lane_id,
            )
        if (
            attempt == 0
            and ack.get("status") == "WARN_CAPTURE_FAILED"
            and ack.get("episode_id") == context["episode_id"]
            and ack.get("reset_generation") == context["reset_generation"]
            and ack.get("sequence_id") == sequence_id
            and isinstance(ack.get("error"), str)
            and RENDER_REALIGN_ERROR.fullmatch(ack["error"]) is not None
        ):
            # A failed paused barrier aligns a lagging Replicator
            # ReferenceTime. Re-request once on the next producer sample; the
            # producer still proves all four images share one render identity.
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            continue
        if (
            attempt == 0
            and ack.get("status") == "IDENTITY_MISMATCH"
            and ack.get("mismatched_field") == "sequence_id"
            and ack.get("episode_id") == context["episode_id"]
            and ack.get("reset_generation") == context["reset_generation"]
            and isinstance(ack.get("sequence_id"), int)
            and not isinstance(ack.get("sequence_id"), bool)
            and ack["sequence_id"] >= sequence_id
        ):
            sequence_id = int(ack["sequence_id"])
            continue
        raise TimeoutAdvisorError(f"snapshot capture failed: {ack.get('status')}")
    raise TimeoutAdvisorError("snapshot identity did not converge")


def _planner_request(
    context: dict[str, Any],
    capture: dict[str, Any],
    result_root: Path,
    *,
    prior_advised_actions: tuple[int, ...] = (),
    task_state: dict[str, Any] | None = None,
) -> SlowPlannerRequest:
    sidecar = _load_regular_json(result_root / capture["sidecar"], "snapshot sidecar")
    images_by_id = {
        str(row["identity"]): result_root / str(row["path"])
        for row in capture["images"]
    }
    ordered = tuple(
        OrderedImage(
            view_id=view_id,
            pose=CAMERA_POSES[view_id],
            jpeg=images_by_id[view_id].read_bytes(),
            width=640,
            height=480,
        )
        for view_id in CAMERA_ORDER
    )
    if context["kind"] in {
        "motion_timeout_after_confirmed_safe_stop",
        "task_state_checkpoint_after_completed_motion",
    }:
        if task_state is None:
            task_state = _new_task_state(context)
        if (
            len(prior_advised_actions) > 4
            or any(action not in PRIMITIVES for action in prior_advised_actions)
        ):
            raise TimeoutAdvisorError("invalid bounded timeout-advice history")
        timeout_context = (
            context["kind"] == "motion_timeout_after_confirmed_safe_stop"
        )
        excluded_action = (
            int(context["excluded_action"]) if timeout_context else None
        )
        legal_actions = tuple(
            action
            for action in sorted(PRIMITIVES)
            if action != excluded_action
        )
        legal_action_text = ", ".join(
            f"{action}={PRIMITIVE_LABELS[action]}" for action in legal_actions
        )
        candidates = tuple(
            primitive
            for action, primitive in PRIMITIVES.items()
            if action != excluded_action
        )
        if not timeout_context:
            candidates = tuple(PRIMITIVES.values())
            instruction = (
                f"{context['instruction']} The robot completed bounded action "
                f"{context['completed_action']}="
                f"{PRIMITIVE_LABELS[int(context['completed_action'])]} and is now "
                "physically safe-stopped. Inspect all four current views and update "
                "the frozen active task clause. If the active clause is incomplete "
                "or recent motion is semantically unproductive, use recover and "
                "choose one listed bounded primitive that changes the route toward "
                "the active clause; do not silently follow a later clause. Choose "
                "abstain only when no listed primitive is justified by the views. "
                "Do not stop and do not invent a goal."
            )
            history = (
                (
                    f"completed_action={context['completed_action']}="
                    f"{PRIMITIVE_LABELS[int(context['completed_action'])]}"
                ),
                f"legal_actions={legal_action_text}",
                (
                    "prior_task_advice="
                    + (
                        ",".join(str(action) for action in prior_advised_actions)
                        if prior_advised_actions
                        else "none"
                    )
                ),
                f"snapshot_sim_stamp_ns={sidecar['sim_stamp_before_ns']}",
                TaskStateContext(
                    instruction=str(task_state["instruction"]),
                    clauses=tuple(str(item) for item in task_state["clauses"]),
                    active_clause_id=int(task_state["active_clause_id"]),
                    revision=int(task_state["revision"]),
                ).to_history(),
            )
        elif excluded_action in {2, 3}:
            opposite_action = 3 if excluded_action == 2 else 2
            escape_policy = (
                "The timed-out action was a turn. Prefer 1=short_forward when "
                "the front camera shows enough traversable floor and clearance. "
                f"Do not choose {opposite_action}="
                f"{PRIMITIVE_LABELS[opposite_action]} merely to undo the previous "
                "turn; choose it only when short_forward is visibly blocked."
            )
        else:
            escape_policy = (
                "The timed-out action was short_forward, so choose the clearer "
                "of the two legal short turns using the left/front/right views."
            )
        if timeout_context:
            instruction = (
                f"{context['instruction']} Previous bounded action "
                f"{excluded_action}={PRIMITIVE_LABELS[excluded_action]} timed out "
                "after confirmed safe-stop and is illegal for this response. The "
                f"only legal response IDs are exactly {legal_action_text}. Choose "
                "one of those listed short primitives that changes the local motion "
                "and best advances the navigation instruction. Inspect all four "
                "views for the named destination or the doorway/corridor most "
                "consistent with it; visible clearance is a safety constraint, not "
                "the objective. Avoid repeating the recent timeout-advice cycle "
                "unless the current view clearly changed. "
                f"{escape_policy} The timed-out ID remains illegal: do not repeat "
                "the timed-out ID, do not stop, and do not invent a goal."
            )
            history = (
                (
                    f"timed_out_action={excluded_action}="
                    f"{PRIMITIVE_LABELS[excluded_action]}"
                ),
                f"legal_actions={legal_action_text}",
                (
                    "prior_timeout_advice="
                    + (
                        ",".join(str(action) for action in prior_advised_actions)
                        if prior_advised_actions
                        else "none"
                    )
                ),
                f"snapshot_sim_stamp_ns={sidecar['sim_stamp_before_ns']}",
                TaskStateContext(
                    instruction=str(task_state["instruction"]),
                    clauses=tuple(str(item) for item in task_state["clauses"]),
                    active_clause_id=int(task_state["active_clause_id"]),
                    revision=int(task_state["revision"]),
                ).to_history(),
            )
    else:
        if context.get("model_stop_candidate") is True:
            excluded_action = context.get("excluded_action")
            candidates = tuple(
                primitive
                for action, primitive in PRIMITIVES.items()
                if action != excluded_action
            )
            recent_timeout = (
                ""
                if excluded_action is None
                else (
                    f" The previous bounded escape {excluded_action}="
                    f"{PRIMITIVE_LABELS[int(excluded_action)]} timed out and is "
                    "illegal for this response; choose only a listed alternative."
                )
            )
            instruction = (
                f"{context['instruction']} InternVLA proposed STOP while the robot "
                "is physically safe-stopped. Independently decide from the current "
                "four-camera evidence whether the instruction's destination is "
                "visibly reached. Return target_found only when arrival is clear; "
                "otherwise select one listed short bounded primitive that best "
                f"continues navigation.{recent_timeout} Do not invent a goal."
            )
            history = (
                f"arrival_confirmation_round={context['advisor_round']}",
                f"required_confirmations={context['required_confirmations']}",
                (
                    f"first_bounded_escape_action={context['first_advised_action']}"
                    if "first_advised_action" in context
                    else "first_bounded_escape_action=none"
                ),
                "termination_candidate=internvla_model_stop",
                (
                    f"recent_timed_out_escape={context['recent_timeout_sequence_id']}:"
                    f"{excluded_action}={PRIMITIVE_LABELS[int(excluded_action)]}"
                    if excluded_action is not None
                    else "recent_timed_out_escape=none"
                ),
                f"snapshot_sim_stamp_ns={sidecar['sim_stamp_before_ns']}",
            )
        else:
            if task_state is None or not task_state.get("clauses"):
                raise TimeoutAdvisorError(
                    "arrival shadow requires initialized task state"
                )
            state_context = TaskStateContext(
                instruction=str(task_state["instruction"]),
                clauses=tuple(str(item) for item in task_state["clauses"]),
                active_clause_id=int(task_state["active_clause_id"]),
                revision=int(task_state["revision"]),
            )
            active_clause = state_context.clauses[
                state_context.active_clause_id - 1
            ]
            terminal_clause = state_context.clauses[-1]
            candidates = ()
            instruction = (
                f"{context['instruction']} The robot is physically safe-stopped "
                f"after bounded action {context['completed_action']}. This is an "
                "observation-only completion_sim arrival shadow with no movement "
                "or terminal STOP authority; candidate frontiers are deliberately "
                "empty. Inspect all four current views and report two independent "
                "verdicts in evidence. evidence must contain exactly two entries: "
                "'goal_region: yes|no - <visible fact>' followed by "
                "'semantic_arrival: yes|no - <visible fact>'. The active task "
                f"clause is '{active_clause}' and the strict terminal clause is "
                f"'{terminal_clause}'. Set target_found=true only when the strict "
                "terminal clause is visibly satisfied; goal-region evidence alone "
                "is not semantic arrival. Otherwise abstain. The existing fresh "
                "two-frame gate remains authoritative. Do not select a movement "
                "primitive, issue STOP, or invent a goal."
            )
            history = (
                f"arrival_confirmation_round={context['advisor_round']}",
                f"required_confirmations={context['required_confirmations']}",
                (
                    "termination_candidate=oracle_terminal_shadow"
                    if context.get("oracle_terminal_shadow") is True
                    else "termination_candidate=periodic_arrival_probe"
                ),
                (
                    "completion_sim_goal_region_trigger="
                    + (
                        "oracle_terminal_shadow"
                        if context.get("oracle_terminal_shadow") is True
                        else "periodic_shadow"
                    )
                ),
                (
                    "task_state_active_clause="
                    f"{state_context.active_clause_id}/{len(state_context.clauses)}"
                    f"@r{state_context.revision}:{active_clause}"
                ),
                f"task_state_terminal_clause={terminal_clause}",
                "arrival_evidence_contract=goal_region|strict_semantic_arrival",
                f"snapshot_sim_stamp_ns={sidecar['sim_stamp_before_ns']}",
                state_context.to_history(),
            )
    lane_id = str(context["episode_id"]).split("::", 1)[0]
    snapshot_id = (
        f"{lane_id}::{context['episode_id']}::{context['reset_generation']}::"
        f"{context['trigger_sequence_id']}"
    )
    return SlowPlannerRequest(
        episode_id=context["episode_id"],
        snapshot_id=snapshot_id,
        instruction=instruction,
        ordered_images=ordered,
        candidate_frontiers=candidates,
        agent_pose=(0.0, 0.0, 0.0, 0.0),
        compact_history=history,
        timestamp=time.time(),
    )


def _arrival_outcome(
    decision: Any, minimum_confidence: float
) -> tuple[str, str]:
    target_found = decision.decision == "target_found" or (
        decision.decision == "abstain"
        and getattr(decision, "target_found", False) is True
    )
    if (
        target_found
        and not decision.fallback_used
        and float(decision.confidence) >= minimum_confidence
    ):
        return "ARRIVED", "step3_visible_destination_reached"
    return "NOT_ARRIVED", "step3_arrival_not_confirmed"


def _instruction_turn_fallback(
    context: dict[str, Any],
    request: SlowPlannerRequest,
    prior_advised_actions: tuple[int, ...],
) -> int | None:
    """Recover one explicit route turn after Step3 abstains.

    This is deliberately narrower than a general language planner: it only
    handles a forward timeout, takes the first explicit turn cue from the
    original navigation instruction, and refuses an immediate repeated
    timeout-advice turn.  It never consumes evaluator pose or target data.
    """

    if (
        context.get("kind") != "motion_timeout_after_confirmed_safe_stop"
        or context.get("excluded_action") != 1
    ):
        return None
    match = re.search(
        r"\b(?:turn|bear|veer|go|head)\s+(?:to\s+the\s+)?(left|right)\b",
        str(context.get("instruction", "")),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    action = 2 if match.group(1).lower() == "left" else 3
    if action not in {item.frontier_id for item in request.candidate_frontiers}:
        return None
    if prior_advised_actions and prior_advised_actions[-1] == action:
        return None
    return action


def _task_clause_motion_fallback(
    context: dict[str, Any],
    request: SlowPlannerRequest,
    prior_advised_actions: tuple[int, ...],
    task_state: dict[str, Any] | None,
) -> int | None:
    """Map one frozen active-clause cue to an existing bounded primitive.

    This completion_sim-only fallback is deliberately lexical and local.  It
    never consumes evaluator pose or goal data, cannot publish control or STOP,
    and is used only after the four-view Step3 updater abstains or misses its
    response deadline.
    """

    kind = context.get("kind")
    model_stop_arrival = (
        kind == "arrival_check_after_completed_motion"
        and context.get("model_stop_candidate") is True
    )
    if (
        kind
        not in {
            "motion_timeout_after_confirmed_safe_stop",
            "task_state_checkpoint_after_completed_motion",
        }
        and not model_stop_arrival
    ) or task_state is None:
        return None
    clauses = task_state.get("clauses")
    active_clause_id = task_state.get("active_clause_id")
    if (
        not isinstance(clauses, list)
        or isinstance(active_clause_id, bool)
        or not isinstance(active_clause_id, int)
        or not 1 <= active_clause_id <= len(clauses)
    ):
        return None
    clause = str(clauses[active_clause_id - 1])
    legal_actions = {item.frontier_id for item in request.candidate_frontiers}
    turn = re.search(
        r"\b(?:(?:turn|bear|veer|go|head)\s+(?:to\s+the\s+)?"
        r"|(?:take|make)\s+an?\s+)(left|right)\b",
        clause,
        flags=re.IGNORECASE,
    )
    # A model-STOP retry may deliberately exclude the just-timed-out forward
    # primitive.  In that narrow case, preserve an explicit object/area side
    # cue (for example, "the sitting area on the left") as a bounded turn.
    # Ordinary checkpoints keep the stricter turn-verb-only interpretation.
    side_cue = (
        re.search(
            r"\b(?:on|to)\s+(?:the\s+)?(left|right)\b",
            clause,
            flags=re.IGNORECASE,
        )
        if model_stop_arrival and context.get("excluded_action") == 1
        else None
    )
    if turn is not None:
        action = 2 if turn.group(1).lower() == "left" else 3
    elif side_cue is not None:
        action = 2 if side_cue.group(1).lower() == "left" else 3
    elif re.search(
        r"\b(?:straight|forward|enter|travel|walk\s+(?:straight\s+)?into)\b",
        clause,
        flags=re.IGNORECASE,
    ):
        action = 1
    else:
        return None
    if action not in legal_actions:
        return None
    if len(prior_advised_actions) >= 2 and prior_advised_actions[-2:] == (
        action,
        action,
    ):
        return None
    return action


def _bounded_visible_motion_advice(
    context: dict[str, Any],
    step3_action: int,
) -> tuple[int, str]:
    """Preserve a valid visual Step3 primitive at normal checkpoints.

    The active-clause lexical mapping is intentionally only a fallback for an
    abstain or unusable response.  It must not replace a legal Step3 decision:
    phrases such as ``walk into the sitting area on the left`` are not a
    reliable instruction to move straight in the robot's current frame.
    """

    reason = (
        "bounded_visible_task_primitive"
        if context.get("kind")
        == "task_state_checkpoint_after_completed_motion"
        else "bounded_visible_escape_primitive"
    )
    return step3_action, reason


def _new_task_state(context: dict[str, Any]) -> dict[str, Any]:
    instruction = " ".join(str(context["instruction"]).split())
    return {
        "schema_version": 1,
        "episode_id": str(context["episode_id"]),
        "reset_generation": int(context["reset_generation"]),
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(
            instruction.encode("utf-8")
        ).hexdigest(),
        "clauses": [],
        "completed_clause_ids": [],
        "active_clause_id": 0,
        "revision": 0,
        "last_sequence_id": -1,
        "last_snapshot_sim_stamp_ns": 0,
        "last_transition": "uninitialized",
        "last_evidence": [],
        "active_clause_semantic_confirmations": 0,
        "final_arrival_candidate": False,
        "decomposition_source": "pending_step3",
    }


def _validate_task_state(
    state: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    state = dict(state)
    state.setdefault("active_clause_semantic_confirmations", 0)
    expected = _new_task_state(context)
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != 1
        or state.get("episode_id") != expected["episode_id"]
        or state.get("reset_generation") != expected["reset_generation"]
        or state.get("instruction") != expected["instruction"]
        or state.get("instruction_sha256") != expected["instruction_sha256"]
        or not isinstance(state.get("clauses"), list)
        or any(not isinstance(item, str) for item in state["clauses"])
        or len(state["clauses"]) > 4
        or not isinstance(state.get("completed_clause_ids"), list)
        or isinstance(state.get("active_clause_id"), bool)
        or not isinstance(state.get("active_clause_id"), int)
        or isinstance(state.get("revision"), bool)
        or not isinstance(state.get("revision"), int)
        or state["revision"] < 0
        or isinstance(state.get("last_sequence_id"), bool)
        or not isinstance(state.get("last_sequence_id"), int)
        or isinstance(state.get("last_snapshot_sim_stamp_ns"), bool)
        or not isinstance(state.get("last_snapshot_sim_stamp_ns"), int)
        or not isinstance(state.get("last_evidence"), list)
        or len(state["last_evidence"]) > 2
        or any(not isinstance(item, str) for item in state["last_evidence"])
        or isinstance(state.get("active_clause_semantic_confirmations"), bool)
        or not isinstance(state.get("active_clause_semantic_confirmations"), int)
        or not 0 <= state["active_clause_semantic_confirmations"] <= 2
        or not isinstance(state.get("final_arrival_candidate"), bool)
    ):
        raise TimeoutAdvisorError("invalid persisted task state")
    clauses = tuple(state["clauses"])
    if clauses:
        TaskStateContext(
            instruction=expected["instruction"],
            clauses=clauses,
            active_clause_id=state["active_clause_id"],
            revision=state["revision"],
        )
    elif state["active_clause_id"] != 0:
        raise TimeoutAdvisorError("uninitialized task state has active clause")
    completed = state["completed_clause_ids"]
    if (
        any(isinstance(item, bool) or not isinstance(item, int) for item in completed)
        or completed != list(range(1, state["active_clause_id"]))
    ):
        raise TimeoutAdvisorError("task-state completed clauses are not monotonic")
    return state


_CLAUSE_LANDMARK_STOPWORDS = {
    "after",
    "ahead",
    "area",
    "around",
    "before",
    "continue",
    "enter",
    "exit",
    "forward",
    "from",
    "head",
    "into",
    "left",
    "move",
    "near",
    "right",
    "room",
    "straight",
    "stop",
    "there",
    "through",
    "toward",
    "turn",
    "until",
    "walk",
}
_DISTANT_EVIDENCE_CUES = (
    " ahead",
    "extends forward",
    "far ",
    "farther",
    "in front",
    "toward",
    "through the doorway",
)


def _active_clause_semantically_present(
    clause: str, evidence: tuple[str, ...]
) -> bool:
    """Confirm a clause landmark without evaluator pose or goal information."""

    clause_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", clause.lower())
        if len(token) >= 4 and token not in _CLAUSE_LANDMARK_STOPWORDS
    }
    if not clause_tokens:
        return False
    for item in evidence:
        normalized = " ".join(item.lower().split())
        if any(cue in f" {normalized}" for cue in _DISTANT_EVIDENCE_CUES):
            continue
        evidence_tokens = set(re.findall(r"[a-z0-9]+", normalized))
        if clause_tokens & evidence_tokens:
            return True
    return False


def _fallback_decomposed_task_state(
    state: dict[str, Any], context: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = dict(_validate_task_state(state, context))
    if state["clauses"]:
        return state, {
            "event": "decomposition_fallback_skipped_initialized",
            "revision": state["revision"],
        }
    clauses = list(fallback_instruction_clauses(state["instruction"]))
    state.update(
        clauses=clauses,
        active_clause_id=1,
        revision=state["revision"] + 1,
        last_transition="decomposed",
        decomposition_source="deterministic_exact_span_fallback",
    )
    return state, {
        "event": "task_decomposed",
        "source": state["decomposition_source"],
        "clauses": clauses,
        "active_clause_id": 1,
        "revision": state["revision"],
    }


def _apply_task_state_decision(
    state: dict[str, Any],
    context: dict[str, Any],
    decision: TaskStatePlannerDecision,
    *,
    snapshot_sim_stamp_ns: int,
    transition_minimum_confidence: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = dict(_validate_task_state(state, context))
    sequence_id = int(context["trigger_sequence_id"])
    if (
        sequence_id <= int(state["last_sequence_id"])
        or snapshot_sim_stamp_ns <= int(state["last_snapshot_sim_stamp_ns"])
    ):
        raise TimeoutAdvisorError("task-state update is stale or replayed")
    clauses = list(state["clauses"])
    if not clauses:
        if not decision.task_clauses:
            raise TimeoutAdvisorError("Step3 did not decompose the instruction")
        clauses = list(decision.task_clauses)
        state.update(
            clauses=clauses,
            active_clause_id=1,
            decomposition_source="step3_exact_instruction_spans",
        )
    elif decision.task_clauses:
        raise TimeoutAdvisorError("Step3 attempted to replace a frozen task plan")
    active_clause_id = int(state["active_clause_id"])
    if decision.active_clause_id != active_clause_id:
        raise TimeoutAdvisorError("Step3 active clause identity mismatch")

    requested_transition = decision.clause_transition
    accepted_transition = requested_transition
    transition_reason = "step3_state_update"
    final_arrival_candidate = False
    completed = list(state["completed_clause_ids"])
    semantic_confirmations = int(
        state["active_clause_semantic_confirmations"]
    )
    if (
        active_clause_id < len(clauses)
        and requested_transition in {"hold", "recover"}
        and decision.confidence >= transition_minimum_confidence
        and _active_clause_semantically_present(
            clauses[active_clause_id - 1], decision.evidence
        )
    ):
        semantic_confirmations = min(2, semantic_confirmations + 1)
        if semantic_confirmations >= 2:
            accepted_transition = "advance"
            transition_reason = "intermediate_clause_semantically_confirmed_twice"
    if requested_transition == "advance":
        if (
            decision.confidence < transition_minimum_confidence
            or not decision.evidence
        ):
            accepted_transition = "hold"
            transition_reason = "advance_below_confidence_or_without_evidence"
        elif active_clause_id < len(clauses):
            completed.append(active_clause_id)
            active_clause_id += 1
            transition_reason = "intermediate_clause_visibly_complete"
        else:
            accepted_transition = "hold"
            final_arrival_candidate = bool(decision.target_found)
            transition_reason = (
                "final_clause_arrival_candidate_requires_existing_gate"
                if final_arrival_candidate
                else "final_clause_advance_without_arrival_candidate"
            )
    if accepted_transition == "advance" and requested_transition != "advance":
        completed.append(active_clause_id)
        active_clause_id += 1
    if active_clause_id != int(state["active_clause_id"]):
        semantic_confirmations = 0
    state.update(
        completed_clause_ids=completed,
        active_clause_id=active_clause_id,
        revision=int(state["revision"]) + 1,
        last_sequence_id=sequence_id,
        last_snapshot_sim_stamp_ns=snapshot_sim_stamp_ns,
        last_transition=accepted_transition,
        last_evidence=list(decision.evidence),
        active_clause_semantic_confirmations=semantic_confirmations,
        final_arrival_candidate=final_arrival_candidate,
    )
    _validate_task_state(state, context)
    return state, {
        "event": "task_state_updated",
        "episode_id": state["episode_id"],
        "reset_generation": state["reset_generation"],
        "trigger_sequence_id": sequence_id,
        "snapshot_sim_stamp_ns": snapshot_sim_stamp_ns,
        "revision": state["revision"],
        "clauses": clauses if state["revision"] == 1 else [],
        "active_clause_id": active_clause_id,
        "active_clause": clauses[active_clause_id - 1],
        "completed_clause_ids": completed,
        "requested_transition": requested_transition,
        "accepted_transition": accepted_transition,
        "transition_reason": transition_reason,
        "confidence": float(decision.confidence),
        "evidence": list(decision.evidence),
        "active_clause_semantic_confirmations": semantic_confirmations,
        "final_arrival_candidate": final_arrival_candidate,
    }


class TimeoutAdvisorNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("t5_step3_timeout_advisor")
        self.args = args
        self._lock = threading.Lock()
        self._active = False
        self._active_identity: tuple[str, int, int, str, int] | None = None
        self._queued_arrival_confirmation: dict[str, Any] | None = None
        self._motion_advice_history: dict[tuple[str, int], list[int]] = {}
        self._task_states: dict[tuple[str, int], dict[str, Any]] = {}
        self._task_state_root = self.args.result_root / "task_state"
        self._task_state_root.mkdir(parents=True, exist_ok=True)
        self.publisher = self.create_publisher(
            String, "/internvla/t5_step3_timeout_advice", 10
        )
        self.create_subscription(
            String, "/internvla/t5_step3_timeout_context", self._on_context, 10
        )

    def _get_task_state(self, context: dict[str, Any]) -> dict[str, Any]:
        key = (
            str(context["episode_id"]),
            int(context["reset_generation"]),
        )
        initialization_event: dict[str, Any] | None = None
        with self._lock:
            state = self._task_states.get(key)
            if state is None:
                current_path = self._task_state_root / "current.json"
                if current_path.is_file() and not current_path.is_symlink():
                    try:
                        candidate = _load_regular_json(
                            current_path, "task state"
                        )
                        state = _validate_task_state(candidate, context)
                    except (OSError, ValueError, TimeoutAdvisorError):
                        state = None
                if state is None:
                    state = _new_task_state(context)
                if not state["clauses"]:
                    state, initialization_event = (
                        _fallback_decomposed_task_state(state, context)
                    )
                self._task_states[key] = state
                while len(self._task_states) > 8:
                    self._task_states.pop(next(iter(self._task_states)))
            state_copy = json.loads(json.dumps(state))
        if initialization_event is not None:
            self._store_task_state(context, state_copy, initialization_event)
        return state_copy

    def _store_task_state(
        self,
        context: dict[str, Any],
        state: dict[str, Any],
        event: dict[str, Any],
    ) -> None:
        state = _validate_task_state(state, context)
        key = (
            str(context["episode_id"]),
            int(context["reset_generation"]),
        )
        with self._lock:
            self._task_states[key] = json.loads(json.dumps(state))
        identity = hashlib.sha256(
            f"{key[0]}\0{key[1]}".encode("utf-8")
        ).hexdigest()[:16]
        _atomic_json(self._task_state_root / "current.json", state)
        _atomic_json(
            self._task_state_root / f"state-{identity}.json", state
        )
        _append_jsonl(
            self._task_state_root / "events.jsonl",
            {"wall_time_unix": time.time(), **event},
        )

    def _publish(self, context: dict[str, Any], **fields: Any) -> None:
        value = {
            "schema_version": 1,
            "episode_id": context["episode_id"],
            "reset_generation": context["reset_generation"],
            "trigger_sequence_id": context["trigger_sequence_id"],
            "trigger_request_id": context["trigger_request_id"],
            "advisor_round": context["advisor_round"],
            **fields,
        }
        message = String()
        message.data = json.dumps(value, sort_keys=True, separators=(",", ":"))
        self.publisher.publish(message)
        _append_jsonl(self.args.output, {"wall_time_unix": time.time(), **value})

    def _on_context(self, message: String) -> None:
        try:
            context = _context(json.loads(str(message.data)))
        except (TypeError, ValueError, json.JSONDecodeError, TimeoutAdvisorError) as exc:
            self.get_logger().warning(f"discarding timeout context: {exc}")
            return
        identity = (
            context["episode_id"],
            context["reset_generation"],
            context["trigger_sequence_id"],
            context["trigger_request_id"],
            context["advisor_round"],
        )
        with self._lock:
            if self._active:
                if identity == self._active_identity:
                    return
                if (
                    self._active_identity is not None
                    and identity[:4] == self._active_identity[:4]
                    and self._active_identity[4] == 1
                    and identity[4] == 2
                    and self._queued_arrival_confirmation is None
                ):
                    self._queued_arrival_confirmation = context
                    return
                self._publish(
                    context,
                    status="FALLBACK",
                    confidence=0.0,
                    reason="advisor_request_already_active",
                )
                return
            self._active = True
            self._active_identity = identity
        threading.Thread(target=self._work, args=(context,), daemon=True).start()

    def _work(self, context: dict[str, Any]) -> None:
        started = time.monotonic()
        deadline = started + self.args.deadline_sec
        task_state: dict[str, Any] | None = None
        try:
            capture = _request_snapshot(
                context=context,
                request_path=self.args.request,
                ack_path=self.args.ack,
                result_root=self.args.result_root,
                contract_path=self.args.contract,
                deadline=deadline,
            )
            history_key = (
                str(context["episode_id"]),
                int(context["reset_generation"]),
            )
            with self._lock:
                prior_advised_actions = tuple(
                    self._motion_advice_history.get(history_key, [])[-4:]
                )
            if context["kind"] in {
                "motion_timeout_after_confirmed_safe_stop",
                "task_state_checkpoint_after_completed_motion",
                "arrival_check_after_completed_motion",
            }:
                task_state = self._get_task_state(context)
            request = _planner_request(
                context,
                capture,
                self.args.result_root,
                prior_advised_actions=prior_advised_actions,
                task_state=task_state,
            )
            sidecar = _load_regular_json(
                self.args.result_root / capture["sidecar"], "snapshot sidecar"
            )
            snapshot_sim_stamp_ns = int(sidecar["sim_stamp_before_ns"])
            if snapshot_sim_stamp_ns <= int(
                context.get("minimum_snapshot_sim_stamp_ns", 0)
            ):
                raise TimeoutAdvisorError(
                    "arrival snapshot did not advance in simulation time"
                )
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000.0))
            if remaining_ms <= 1:
                raise TimeoutAdvisorError("no wall budget remains for Step3")
            with SlowPlannerClient(self.args.endpoint, timeout_ms=remaining_ms) as client:
                decision, metrics = client.decide(request)
            task_event: dict[str, Any] | None = None
            if context["kind"] in {
                "motion_timeout_after_confirmed_safe_stop",
                "task_state_checkpoint_after_completed_motion",
            }:
                if not isinstance(decision, TaskStatePlannerDecision):
                    fallback_action = _instruction_turn_fallback(
                        context, request, prior_advised_actions
                    )
                    if fallback_action is None:
                        fallback_action = _task_clause_motion_fallback(
                            context,
                            request,
                            prior_advised_actions,
                            task_state,
                        )
                    if fallback_action is not None:
                        with self._lock:
                            history = self._motion_advice_history.setdefault(
                                history_key, []
                            )
                            history.append(fallback_action)
                            del history[:-4]
                        self._publish(
                            context,
                            status="ADVISE",
                            advised_action=fallback_action,
                            confidence=self.args.minimum_confidence,
                            reason=(
                                "bounded_clause_fallback_after_schema_failure"
                            ),
                            snapshot_id=request.snapshot_id,
                            snapshot_sim_stamp_ns=snapshot_sim_stamp_ns,
                            camera_count=len(request.ordered_images),
                            service_wall_latency_sec=time.monotonic() - started,
                            metrics=metrics.to_mapping(),
                        )
                        return
                    raise TimeoutAdvisorError(
                        "Step3 service did not return task-state decision"
                    )
                assert task_state is not None
                task_state, task_event = _apply_task_state_decision(
                    task_state,
                    context,
                    decision,
                    snapshot_sim_stamp_ns=snapshot_sim_stamp_ns,
                    transition_minimum_confidence=(
                        self.args.task_transition_minimum_confidence
                    ),
                )
                self._store_task_state(
                    context, task_state, task_event
                )
            if context["kind"] == "arrival_check_after_completed_motion":
                status, reason = _arrival_outcome(
                    decision, self.args.arrival_minimum_confidence
                )
                escape_action: int | None = None
                advice_confidence = float(decision.confidence)
                if (
                    status == "NOT_ARRIVED"
                    and context.get("model_stop_candidate") is True
                    and decision.decision == "select_frontier"
                    and decision.frontier_id
                    in {item.frontier_id for item in request.candidate_frontiers}
                    and float(decision.confidence) >= self.args.minimum_confidence
                    and not decision.fallback_used
                ):
                    escape_action = int(decision.frontier_id)
                    reason = "step3_model_stop_rejected_with_bounded_escape"
                elif (
                    status == "NOT_ARRIVED"
                    and context.get("model_stop_candidate") is True
                ):
                    # Single-variable completion_sim escape: Step3 has already
                    # rejected arrival, but an abstain previously caused the
                    # unconfirmed InternVLA STOP to terminate the episode.  Use
                    # only the frozen active clause to select one existing
                    # bounded primitive.  The client still requires a second
                    # fresh, direction-consistent four-camera decision and
                    # enforces its per-episode escape budget.
                    escape_action = _task_clause_motion_fallback(
                        context,
                        request,
                        (),
                        task_state,
                    )
                    if escape_action is not None:
                        advice_confidence = self.args.minimum_confidence
                        reason = (
                            "active_clause_bounded_escape_after_"
                            "unconfirmed_model_stop"
                        )
                self._publish(
                    context,
                    status=status,
                    confidence=advice_confidence,
                    reason=reason,
                    snapshot_id=request.snapshot_id,
                    snapshot_sim_stamp_ns=snapshot_sim_stamp_ns,
                    camera_count=len(request.ordered_images),
                    service_wall_latency_sec=time.monotonic() - started,
                    metrics=metrics.to_mapping(),
                    **_decision_audit_fields(decision),
                    **(
                        {"advised_action": escape_action}
                        if escape_action is not None
                        else {}
                    ),
                )
            elif (
                isinstance(decision, TaskStatePlannerDecision)
                and decision.target_found
            ):
                self._publish(
                    context,
                    status="FALLBACK",
                    confidence=float(decision.confidence),
                    reason=(
                        "step3_task_final_arrival_candidate_requires_existing_gate"
                    ),
                    snapshot_id=request.snapshot_id,
                    snapshot_sim_stamp_ns=snapshot_sim_stamp_ns,
                    camera_count=len(request.ordered_images),
                    service_wall_latency_sec=time.monotonic() - started,
                    metrics=metrics.to_mapping(),
                )
            elif (
                decision.decision != "select_frontier"
                or decision.frontier_id not in {
                    item.frontier_id for item in request.candidate_frontiers
                }
                or decision.confidence < self.args.minimum_confidence
                or decision.fallback_used
            ):
                fallback_action = _instruction_turn_fallback(
                    context, request, prior_advised_actions
                )
                if fallback_action is None:
                    fallback_action = _task_clause_motion_fallback(
                        context,
                        request,
                        prior_advised_actions,
                        task_state,
                    )
                if fallback_action is None:
                    self._publish(
                        context,
                        status="FALLBACK",
                        confidence=float(decision.confidence),
                        reason="step3_abstain_or_low_confidence",
                        snapshot_id=request.snapshot_id,
                        snapshot_sim_stamp_ns=snapshot_sim_stamp_ns,
                        camera_count=len(request.ordered_images),
                        service_wall_latency_sec=time.monotonic() - started,
                    )
                else:
                    with self._lock:
                        history = self._motion_advice_history.setdefault(
                            history_key, []
                        )
                        history.append(fallback_action)
                        del history[:-4]
                    self._publish(
                        context,
                        status="ADVISE",
                        advised_action=fallback_action,
                        confidence=self.args.minimum_confidence,
                        reason=(
                            "task_clause_motion_fallback_after_step3_abstain"
                            if context["kind"]
                            == "task_state_checkpoint_after_completed_motion"
                            else "instruction_direction_fallback_after_step3_abstain"
                        ),
                        snapshot_id=request.snapshot_id,
                        snapshot_sim_stamp_ns=snapshot_sim_stamp_ns,
                        camera_count=len(request.ordered_images),
                        service_wall_latency_sec=time.monotonic() - started,
                        metrics=metrics.to_mapping(),
                    )
            else:
                advised_action, advice_reason = (
                    _bounded_visible_motion_advice(
                        context, int(decision.frontier_id)
                    )
                )
                with self._lock:
                    history = self._motion_advice_history.setdefault(history_key, [])
                    history.append(advised_action)
                    del history[:-4]
                self._publish(
                    context,
                    status="ADVISE",
                    advised_action=advised_action,
                    confidence=float(decision.confidence),
                    reason=advice_reason,
                    snapshot_id=request.snapshot_id,
                    snapshot_sim_stamp_ns=snapshot_sim_stamp_ns,
                    camera_count=len(request.ordered_images),
                    service_wall_latency_sec=time.monotonic() - started,
                    metrics=metrics.to_mapping(),
                )
        except BaseException as exc:
            if (
                context.get("kind")
                in {
                    "motion_timeout_after_confirmed_safe_stop",
                    "task_state_checkpoint_after_completed_motion",
                }
                and task_state is not None
                and not task_state.get("clauses")
            ):
                try:
                    task_state, event = _fallback_decomposed_task_state(
                        task_state, context
                    )
                    self._store_task_state(context, task_state, event)
                except BaseException as state_exc:
                    self.get_logger().warning(
                        f"task-state fallback failed: {state_exc}"
                    )
            self._publish(
                context,
                status="FALLBACK",
                confidence=0.0,
                reason=f"advisor_failure:{type(exc).__name__}"[:128],
                failure_message=(
                    _safe_log_text(str(exc), maximum_length=160)
                    or type(exc).__name__
                ),
                service_wall_latency_sec=time.monotonic() - started,
            )
            self.get_logger().warning(f"Step3 timeout advisor fallback: {exc}")
        finally:
            queued: dict[str, Any] | None
            with self._lock:
                self._active = False
                self._active_identity = None
                queued = self._queued_arrival_confirmation
                self._queued_arrival_confirmation = None
                if queued is not None:
                    self._active = True
                    self._active_identity = (
                        queued["episode_id"],
                        queued["reset_generation"],
                        queued["trigger_sequence_id"],
                        queued["trigger_request_id"],
                        queued["advisor_round"],
                    )
            if queued is not None:
                threading.Thread(
                    target=self._work, args=(queued,), daemon=True
                ).start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:8202")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--ack", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-sec", type=float, default=12.0)
    parser.add_argument("--minimum-confidence", type=float, default=0.35)
    parser.add_argument("--arrival-minimum-confidence", type=float, default=0.80)
    parser.add_argument(
        "--task-transition-minimum-confidence", type=float, default=0.70
    )
    args = parser.parse_args()
    if not 1.0 <= args.deadline_sec <= 12.0:
        parser.error("deadline must be in [1, 12] seconds")
    if not 0.35 <= args.minimum_confidence <= 1.0:
        parser.error("minimum confidence must be in [0.35, 1]")
    if not 0.80 <= args.arrival_minimum_confidence <= 1.0:
        parser.error("arrival minimum confidence must be in [0.80, 1]")
    if not 0.70 <= args.task_transition_minimum_confidence <= 1.0:
        parser.error("task transition minimum confidence must be in [0.70, 1]")
    args.result_root = args.result_root.resolve()
    for path in (args.request, args.ack, args.output):
        if not path.resolve().is_relative_to(args.result_root):
            parser.error("runtime paths must stay under result root")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = TimeoutAdvisorNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
