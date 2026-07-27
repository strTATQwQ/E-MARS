"""Pure transition rules for separately attributed Step3 arrival advice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


CONTINUE_NAVIGATION = "continue_navigation"
REQUEST_CONFIRMATION = "request_confirmation"
TERMINATE_ASSISTED = "terminate_assisted"
APPLY_BOUNDED_ESCAPE = "apply_bounded_escape"
BOUNDED_MOTION_ACTIONS = frozenset({1, 2, 3})


@dataclass(frozen=True)
class ArrivalTransition:
    action: str
    advisor_round: int
    snapshot_sim_stamp_ns: int | None
    advised_action: int | None = None


def completed_motion_action(pending: Mapping[str, Any]) -> int | None:
    """Read either a motion-gate record or its arrival follow-up context."""

    value = pending.get("completed_action", pending.get("action"))
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value in BOUNDED_MOTION_ACTIONS else None


def is_unconfirmed_model_stop(
    result: Mapping[str, Any], *, oracle_rejected: bool = False
) -> bool:
    """Identify an InternVLA STOP that still lacks independent arrival evidence.

    ``oracle_rejected`` is an explicit completion-simulation bridge for the
    case where the adapter has already cleared ``stop`` after proving that the
    robot is outside the oracle radius.  It does not grant Step3 termination
    authority: it only lets the existing bounded model-STOP escape run.
    """

    action = result.get("model_discrete_action")
    stop_candidate = result.get("stop") is True or (
        oracle_rejected
        and result.get("model_stop") is True
        and result.get("geometric_success") is False
    )
    return (
        stop_candidate
        and not isinstance(action, bool)
        and action == 0
        and result.get("step3_arrival_confirmed") is not True
    )


def pending_model_action(pending: Mapping[str, Any]) -> int:
    """Attribute a safe hold without assuming timeout-only context fields."""

    if pending.get("model_stop_candidate") is True:
        value = 0
    elif pending.get("kind") == "arrival_check_after_completed_motion":
        value = completed_motion_action(pending)
    else:
        value = pending.get("excluded_action")
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in BOUNDED_MOTION_ACTIONS
        ):
            value = None
    return int(value) if value is not None else 0


def arrival_transition(
    pending: Mapping[str, Any],
    advice: Mapping[str, Any],
    *,
    required_confirmations: int,
) -> ArrivalTransition:
    """Return the next action without granting Step3 direct control authority."""

    if required_confirmations != 2:
        raise ValueError("Step3 arrival requires exactly two confirmations")
    advisor_round = int(pending.get("advisor_round", -1))
    if advisor_round not in {1, 2}:
        raise ValueError("invalid Step3 arrival advisor round")
    if advice.get("status") != "ARRIVED":
        return ArrivalTransition(CONTINUE_NAVIGATION, advisor_round, None)
    snapshot_sim_stamp_ns = advice.get("snapshot_sim_stamp_ns")
    if isinstance(snapshot_sim_stamp_ns, bool) or not isinstance(
        snapshot_sim_stamp_ns, int
    ):
        raise ValueError("Step3 arrival advice lacks a simulation stamp")
    if advisor_round < required_confirmations:
        return ArrivalTransition(
            REQUEST_CONFIRMATION, advisor_round, snapshot_sim_stamp_ns
        )
    return ArrivalTransition(
        TERMINATE_ASSISTED, advisor_round, snapshot_sim_stamp_ns
    )


def model_stop_escape_transition(
    pending: Mapping[str, Any],
    advice: Mapping[str, Any],
    *,
    escape_count: int,
    escape_limit: int,
) -> ArrivalTransition:
    """Require two fresh, direction-consistent NOT_ARRIVED decisions."""

    advisor_round = int(pending.get("advisor_round", -1))
    if advisor_round not in {1, 2}:
        raise ValueError("invalid Step3 model-STOP advisor round")
    action = advice.get("advised_action")
    excluded_action = pending.get("excluded_action")
    excluded_action_invalid = excluded_action is not None and (
        isinstance(excluded_action, bool)
        or not isinstance(excluded_action, int)
        or excluded_action not in BOUNDED_MOTION_ACTIONS
    )
    if (
        pending.get("model_stop_candidate") is not True
        or escape_count >= escape_limit
        or advice.get("status") != "NOT_ARRIVED"
        or isinstance(action, bool)
        or not isinstance(action, int)
        or action not in BOUNDED_MOTION_ACTIONS
        or excluded_action_invalid
        or action == excluded_action
    ):
        return ArrivalTransition(CONTINUE_NAVIGATION, advisor_round, None)
    snapshot_sim_stamp_ns = advice.get("snapshot_sim_stamp_ns")
    if isinstance(snapshot_sim_stamp_ns, bool) or not isinstance(
        snapshot_sim_stamp_ns, int
    ):
        raise ValueError("Step3 model-STOP advice lacks a simulation stamp")
    if advisor_round == 1:
        return ArrivalTransition(
            REQUEST_CONFIRMATION,
            advisor_round,
            snapshot_sim_stamp_ns,
            action,
        )
    if pending.get("first_advised_action") != action:
        return ArrivalTransition(CONTINUE_NAVIGATION, advisor_round, None)
    return ArrivalTransition(
        APPLY_BOUNDED_ESCAPE,
        advisor_round,
        snapshot_sim_stamp_ns,
        action,
    )
