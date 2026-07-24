"""Pure transition rules for separately attributed Step3 arrival advice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


CONTINUE_NAVIGATION = "continue_navigation"
REQUEST_CONFIRMATION = "request_confirmation"
TERMINATE_ASSISTED = "terminate_assisted"
BOUNDED_MOTION_ACTIONS = frozenset({1, 2, 3})


@dataclass(frozen=True)
class ArrivalTransition:
    action: str
    advisor_round: int
    snapshot_sim_stamp_ns: int | None


def completed_motion_action(pending: Mapping[str, Any]) -> int | None:
    """Read either a motion-gate record or its arrival follow-up context."""

    value = pending.get("completed_action", pending.get("action"))
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value in BOUNDED_MOTION_ACTIONS else None


def pending_model_action(pending: Mapping[str, Any]) -> int:
    """Attribute a safe hold without assuming timeout-only context fields."""

    if pending.get("kind") == "arrival_check_after_completed_motion":
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
