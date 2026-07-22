"""Private typed Lane-B viewpoint selection with proposal-only authority.

This module is deliberately separate from the frozen SlowPlanner v1 frontier
wire contract.  A reachable free-space viewpoint remains a viewpoint; it is
never relabelled as a Nav2 unknown-boundary frontier.  The only positive output
is an inert, bounded Nav2 goal *proposal* artifact.  Runtime goal publication,
``cmd_vel`` and terminal STOP authority are outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .lane_b import FROZEN_INTERNVLA_FALLBACK_CANDIDATE, LaneBSnapshotIdentity
from .reachable_viewpoint import (
    REACHABLE_VIEWPOINT_CANDIDATE_TYPE,
    REACHABLE_VIEWPOINT_KIND,
    validate_reachable_viewpoint_mapping,
)


PRIVATE_PROTOCOL_VERSION = 2
PRIVATE_DECISION_KIND = "t5_lane_b_candidate_set_v2_decision"
PRIVATE_RESOLUTION_KIND = "t5_lane_b_candidate_set_v2_resolution"
BOUNDED_PROPOSAL_KIND = "bounded_nav2_goal_proposal"
SELECT_VIEWPOINT = "select_viewpoint"
ABSTAIN = "abstain"
ABSOLUTE_MAXIMUM_GOAL_DISTANCE_M = 5.0
RELATIVE_GOAL_FRAME_ID = "base_link"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@+-]{0,255}$")
_REASON_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_DECISION_KEYS = {
    "protocol_version",
    "kind",
    "lane_id",
    "episode_id",
    "reset_id",
    "sequence_id",
    "snapshot_id",
    "source_artifact",
    "source_candidate_set_sha256",
    "grid_frame_id",
    "decision",
    "candidate_id",
    "confidence",
}
_RESOLUTION_KEYS = _DECISION_KEYS | {
    "proposal",
    "fallback",
    "authority",
    "resolution_sha256",
}
_AUTHORITY = {
    "mode": "proposal_only",
    "publish_navigation_goal": False,
    "publish_cmd_vel": False,
    "publish_terminal_stop": False,
}


class CandidateSetV2Error(ValueError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def _fail(message: str, code: str) -> CandidateSetV2Error:
    return CandidateSetV2Error(message, code=code)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise _fail(f"{name} must be numeric", "NONFINITE_OR_INVALID_NUMBER")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _fail(f"{name} must be numeric", "NONFINITE_OR_INVALID_NUMBER") from exc
    if not math.isfinite(result):
        raise _fail(f"{name} must be finite", "NONFINITE_OR_INVALID_NUMBER")
    return result


def _digest(value: Any, name: str) -> str:
    raw = str(value or "")
    if not _SHA256_RE.fullmatch(raw):
        raise _fail(f"{name} must be a lowercase SHA-256", "DIGEST_MISMATCH")
    return raw


def _artifact(value: Any) -> str:
    raw = str(value or "")
    if (
        not _ARTIFACT_RE.fullmatch(raw)
        or raw.startswith("/")
        or "\\" in raw
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        raise _fail(
            "source_artifact must be a normalized repository-relative identifier",
            "SOURCE_ARTIFACT_MISMATCH",
        )
    return raw


def _frame(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw or len(raw) > 128 or any(character.isspace() for character in raw):
        raise _fail("grid_frame_id is invalid", "FRAME_MISMATCH")
    return raw


def _reason(value: Any, *, default: str = "PLANNER_FAILURE") -> str:
    raw = str(value or "").strip()
    return raw if _REASON_RE.fullmatch(raw) else default


def _identity_fields(value: Mapping[str, Any]) -> LaneBSnapshotIdentity:
    if value.get("lane_id") != "b":
        raise _fail("candidate-set v2 is private to Lane B", "LANE_MISMATCH")
    try:
        identity = LaneBSnapshotIdentity(
            str(value.get("episode_id") or ""),
            value.get("reset_id"),
            value.get("sequence_id"),
        )
    except ValueError as exc:
        raise _fail("invalid Lane-B identity", "IDENTITY_MISMATCH") from exc
    if value.get("snapshot_id") != identity.snapshot_id:
        raise _fail("snapshot identity fields disagree", "IDENTITY_MISMATCH")
    return identity


@dataclass(frozen=True)
class CandidateSetV2Decision:
    identity: LaneBSnapshotIdentity
    source_artifact: str
    source_candidate_set_sha256: str
    grid_frame_id: str
    decision: str
    candidate_id: int | None
    confidence: float

    def __post_init__(self) -> None:
        _artifact(self.source_artifact)
        _digest(self.source_candidate_set_sha256, "source_candidate_set_sha256")
        _frame(self.grid_frame_id)
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise _fail("confidence must be in [0,1]", "INVALID_DECISION")
        if self.decision == SELECT_VIEWPOINT:
            if (
                isinstance(self.candidate_id, bool)
                or not isinstance(self.candidate_id, int)
                or self.candidate_id < 0
            ):
                raise _fail(
                    "select_viewpoint requires a non-negative integer candidate_id",
                    "INVALID_DECISION",
                )
        elif self.decision == ABSTAIN:
            if self.candidate_id is not None:
                raise _fail("abstain requires null candidate_id", "INVALID_DECISION")
        else:
            raise _fail("decision must be select_viewpoint or abstain", "INVALID_DECISION")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateSetV2Decision":
        if not isinstance(value, Mapping) or set(value) != _DECISION_KEYS:
            raise _fail("private decision keys differ", "INVALID_DECISION")
        if (
            value.get("protocol_version") != PRIVATE_PROTOCOL_VERSION
            or value.get("kind") != PRIVATE_DECISION_KIND
        ):
            raise _fail("unsupported private decision protocol", "INVALID_DECISION")
        return cls(
            identity=_identity_fields(value),
            source_artifact=_artifact(value.get("source_artifact")),
            source_candidate_set_sha256=_digest(
                value.get("source_candidate_set_sha256"),
                "source_candidate_set_sha256",
            ),
            grid_frame_id=_frame(value.get("grid_frame_id")),
            decision=str(value.get("decision") or ""),
            candidate_id=value.get("candidate_id"),
            confidence=_finite(value.get("confidence"), "confidence"),
        )


def parse_candidate_set_v2_decision(raw_text: str) -> CandidateSetV2Decision:
    """Parse exactly one private JSON object; prose/CoT/wrappers are rejected."""

    text = str(raw_text).strip()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise _fail(f"duplicate key: {key}", "INVALID_DECISION")
            result[key] = item
        return result

    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    try:
        value, end = decoder.raw_decode(text)
    except (json.JSONDecodeError, CandidateSetV2Error) as exc:
        if isinstance(exc, CandidateSetV2Error):
            raise
        raise _fail("decision is not complete JSON", "INVALID_DECISION") from exc
    if text[end:].strip() or not isinstance(value, Mapping):
        raise _fail("decision must be exactly one JSON object", "INVALID_DECISION")
    return CandidateSetV2Decision.from_mapping(value)


def _canonical_sha256(value: Mapping[str, Any], *, omitted: str) -> str:
    content = {key: value[key] for key in value if key != omitted}
    encoded = json.dumps(
        content, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fallback_resolution(
    *,
    identity: LaneBSnapshotIdentity,
    source_artifact: str,
    source_digest: str,
    grid_frame_id: str,
    reason: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "protocol_version": PRIVATE_PROTOCOL_VERSION,
        "kind": PRIVATE_RESOLUTION_KIND,
        "lane_id": "b",
        "episode_id": identity.episode_id,
        "reset_id": identity.reset_id,
        "sequence_id": identity.sequence_id,
        "snapshot_id": identity.snapshot_id,
        "source_artifact": source_artifact,
        "source_candidate_set_sha256": source_digest,
        "grid_frame_id": grid_frame_id,
        "decision": ABSTAIN,
        "candidate_id": None,
        "confidence": 0.0,
        "proposal": None,
        "fallback": {
            "used": True,
            "mode": "shadow_only",
            "candidate": FROZEN_INTERNVLA_FALLBACK_CANDIDATE,
            "reason": _reason(reason),
        },
        "authority": dict(_AUTHORITY),
        "resolution_sha256": "",
    }
    value["resolution_sha256"] = _canonical_sha256(
        value, omitted="resolution_sha256"
    )
    validate_candidate_set_v2_resolution(value)
    return value


def materialize_candidate_set_v2_shadow(
    *,
    candidate_set: Mapping[str, Any],
    decision_text: str | None,
    expected_identity: LaneBSnapshotIdentity,
    expected_source_artifact: str,
    expected_candidate_set_sha256: str,
    expected_grid_frame_id: str,
    now_sim_time_s: float,
    maximum_goal_distance_m: float = 5.0,
    failure_reason: str = "timeout",
) -> dict[str, Any]:
    """Resolve one private decision into an inert proposal or safe fallback.

    Every failure is deterministic ``abstain``.  In particular, this function
    never emits a fallback viewpoint, goal, velocity, or STOP command.
    """

    artifact = _artifact(expected_source_artifact)
    expected_digest = _digest(
        expected_candidate_set_sha256, "expected_candidate_set_sha256"
    )
    expected_frame = _frame(expected_grid_frame_id)
    now = _finite(now_sim_time_s, "now_sim_time_s")
    maximum_distance = _finite(maximum_goal_distance_m, "maximum_goal_distance_m")
    if not 0.0 < maximum_distance <= ABSOLUTE_MAXIMUM_GOAL_DISTANCE_M:
        raise _fail("maximum goal distance is outside the private bound", "INVALID_BOUND")

    def fallback(reason: str) -> dict[str, Any]:
        return _fallback_resolution(
            identity=expected_identity,
            source_artifact=artifact,
            source_digest=expected_digest,
            grid_frame_id=expected_frame,
            reason=reason,
        )

    try:
        validate_reachable_viewpoint_mapping(candidate_set)
    except Exception:
        return fallback("INVALID_CANDIDATE_SET")
    if candidate_set.get("kind") != REACHABLE_VIEWPOINT_KIND:
        return fallback("INVALID_CANDIDATE_SET")
    if candidate_set.get("candidate_type") != REACHABLE_VIEWPOINT_CANDIDATE_TYPE:
        return fallback("CANDIDATE_TYPE_MISMATCH")
    source_identity = LaneBSnapshotIdentity(
        str(candidate_set.get("episode_id") or ""),
        candidate_set.get("reset_id"),
        candidate_set.get("sequence_id"),
    )
    if source_identity != expected_identity:
        return fallback("STALE_OR_CROSS_RESET_SOURCE")
    if candidate_set.get("candidate_set_sha256") != expected_digest:
        return fallback("SOURCE_DIGEST_MISMATCH")
    if candidate_set.get("grid_frame_id") != expected_frame:
        return fallback("SOURCE_FRAME_MISMATCH")
    captured = _finite(candidate_set.get("captured_sim_time_s"), "captured_sim_time_s")
    valid_until = _finite(
        candidate_set.get("valid_until_sim_time_s"), "valid_until_sim_time_s"
    )
    if now < captured or now > valid_until:
        return fallback("STALE_OR_FUTURE_SOURCE")
    if decision_text is None:
        return fallback(_reason(failure_reason, default="timeout"))
    try:
        decision = parse_candidate_set_v2_decision(decision_text)
    except CandidateSetV2Error:
        return fallback("INVALID_DECISION")
    if decision.identity != expected_identity:
        return fallback("STALE_OR_CROSS_RESET_DECISION")
    if decision.source_artifact != artifact:
        return fallback("SOURCE_ARTIFACT_MISMATCH")
    if decision.source_candidate_set_sha256 != expected_digest:
        return fallback("DECISION_DIGEST_MISMATCH")
    if decision.grid_frame_id != expected_frame:
        return fallback("DECISION_FRAME_MISMATCH")
    if decision.decision == ABSTAIN:
        return fallback("MODEL_ABSTAIN")

    candidates = {
        item["candidate_id"]: item for item in candidate_set.get("candidates", ())
    }
    selected = candidates.get(decision.candidate_id)
    if selected is None:
        return fallback("UNKNOWN_CANDIDATE")
    if selected.get("candidate_xy_frame") != expected_frame:
        return fallback("CANDIDATE_FRAME_MISMATCH")
    try:
        distance = _finite(selected.get("distance_m"), "distance_m")
        relative_xz = selected.get("relative_xz")
        if (
            not isinstance(relative_xz, Sequence)
            or isinstance(relative_xz, (str, bytes))
            or len(relative_xz) != 2
        ):
            raise _fail("relative_xz is invalid", "INVALID_CANDIDATE_COORDINATE")
        # Source convention is (right, forward); ROS base_link XY is
        # (forward, left).  The proposal intentionally uses this bounded
        # relative coordinate rather than grid_xy: the v1 source does not carry
        # a robot pose in every possible grid frame, so grid_xy alone cannot
        # prove a bounded displacement.
        relative_right = _finite(relative_xz[0], "relative_right")
        relative_forward = _finite(relative_xz[1], "relative_forward")
        relative_distance = math.hypot(relative_right, relative_forward)
        goal_xy = [relative_forward, -relative_right]
    except CandidateSetV2Error:
        return fallback("INVALID_CANDIDATE_COORDINATE")
    if (
        distance <= 0.0
        or distance > maximum_distance
        or relative_distance > maximum_distance
        or not math.isclose(distance, relative_distance, rel_tol=0.0, abs_tol=1e-6)
    ):
        return fallback("CANDIDATE_OUT_OF_BOUNDS")

    value: dict[str, Any] = {
        "protocol_version": PRIVATE_PROTOCOL_VERSION,
        "kind": PRIVATE_RESOLUTION_KIND,
        "lane_id": "b",
        "episode_id": expected_identity.episode_id,
        "reset_id": expected_identity.reset_id,
        "sequence_id": expected_identity.sequence_id,
        "snapshot_id": expected_identity.snapshot_id,
        "source_artifact": artifact,
        "source_candidate_set_sha256": expected_digest,
        "grid_frame_id": expected_frame,
        "decision": SELECT_VIEWPOINT,
        "candidate_id": decision.candidate_id,
        "confidence": decision.confidence,
        "proposal": {
            "kind": BOUNDED_PROPOSAL_KIND,
            "frame_id": RELATIVE_GOAL_FRAME_ID,
            "position_xy": goal_xy,
            "distance_m": distance,
            "valid_until_sim_time_s": valid_until,
            "proposal_only": True,
        },
        "fallback": {
            "used": False,
            "mode": "shadow_only",
            "candidate": None,
            "reason": "",
        },
        "authority": dict(_AUTHORITY),
        "resolution_sha256": "",
    }
    value["resolution_sha256"] = _canonical_sha256(
        value, omitted="resolution_sha256"
    )
    validate_candidate_set_v2_resolution(value)
    return value


def validate_candidate_set_v2_resolution(value: Mapping[str, Any]) -> None:
    """Validate an archived private resolution without granting execution authority."""

    if not isinstance(value, Mapping) or set(value) != _RESOLUTION_KEYS:
        raise _fail("resolution keys differ", "INVALID_RESOLUTION")
    if (
        value.get("protocol_version") != PRIVATE_PROTOCOL_VERSION
        or value.get("kind") != PRIVATE_RESOLUTION_KIND
    ):
        raise _fail("unsupported resolution protocol", "INVALID_RESOLUTION")
    _identity_fields(value)
    _artifact(value.get("source_artifact"))
    _digest(value.get("source_candidate_set_sha256"), "source_candidate_set_sha256")
    _frame(value.get("grid_frame_id"))
    confidence = _finite(value.get("confidence"), "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise _fail("resolution confidence is invalid", "INVALID_RESOLUTION")
    if value.get("authority") != _AUTHORITY:
        raise _fail("resolution acquired forbidden authority", "FORBIDDEN_AUTHORITY")
    if value.get("decision") == SELECT_VIEWPOINT:
        if (
            isinstance(value.get("candidate_id"), bool)
            or not isinstance(value.get("candidate_id"), int)
            or value.get("candidate_id") < 0
        ):
            raise _fail("selected candidate_id is invalid", "INVALID_RESOLUTION")
        proposal = value.get("proposal")
        if not isinstance(proposal, Mapping) or set(proposal) != {
            "kind",
            "frame_id",
            "position_xy",
            "distance_m",
            "valid_until_sim_time_s",
            "proposal_only",
        }:
            raise _fail("select_viewpoint lacks a bounded proposal", "INVALID_RESOLUTION")
        if (
            proposal.get("kind") != BOUNDED_PROPOSAL_KIND
            or proposal.get("frame_id") != RELATIVE_GOAL_FRAME_ID
            or proposal.get("proposal_only") is not True
        ):
            raise _fail("proposal type/frame/authority mismatch", "INVALID_RESOLUTION")
        position = proposal.get("position_xy")
        if not isinstance(position, list) or len(position) != 2:
            raise _fail("proposal coordinate encoding is invalid", "INVALID_RESOLUTION")
        proposal_x = _finite(position[0], "proposal.x")
        proposal_y = _finite(position[1], "proposal.y")
        position_distance = math.hypot(proposal_x, proposal_y)
        proposal_distance = _finite(proposal.get("distance_m"), "proposal.distance_m")
        if (
            not 0.0 < proposal_distance <= ABSOLUTE_MAXIMUM_GOAL_DISTANCE_M
            or position_distance > ABSOLUTE_MAXIMUM_GOAL_DISTANCE_M
            or not math.isclose(
                proposal_distance, position_distance, rel_tol=0.0, abs_tol=1e-6
            )
        ):
            raise _fail("proposal distance is invalid", "INVALID_RESOLUTION")
        _finite(proposal.get("valid_until_sim_time_s"), "valid_until_sim_time_s")
        if value.get("fallback") != {
            "used": False,
            "mode": "shadow_only",
            "candidate": None,
            "reason": "",
        }:
            raise _fail("positive proposal fallback fields differ", "INVALID_RESOLUTION")
    elif value.get("decision") == ABSTAIN:
        if (
            value.get("candidate_id") is not None
            or value.get("proposal") is not None
            or confidence != 0.0
        ):
            raise _fail("abstain must not contain a proposal", "INVALID_RESOLUTION")
        fallback = value.get("fallback")
        if (
            not isinstance(fallback, Mapping)
            or set(fallback) != {"used", "mode", "candidate", "reason"}
            or fallback.get("used") is not True
            or fallback.get("mode") != "shadow_only"
            or fallback.get("candidate") != FROZEN_INTERNVLA_FALLBACK_CANDIDATE
            or not str(fallback.get("reason") or "")
        ):
            raise _fail("abstain fallback fields differ", "INVALID_RESOLUTION")
        if _reason(fallback.get("reason")) != fallback.get("reason"):
            raise _fail("fallback reason is not a public code", "INVALID_RESOLUTION")
    else:
        raise _fail("resolution decision is invalid", "INVALID_RESOLUTION")
    supplied = _digest(value.get("resolution_sha256"), "resolution_sha256")
    if supplied != _canonical_sha256(value, omitted="resolution_sha256"):
        raise _fail("resolution digest mismatch", "DIGEST_MISMATCH")


def candidate_set_v2_compatibility_receipt() -> dict[str, Any]:
    return {
        "private_protocol_version": PRIVATE_PROTOCOL_VERSION,
        "status": "OFFLINE_SHADOW_READY",
        "input_kind": REACHABLE_VIEWPOINT_KIND,
        "decision_kind": PRIVATE_DECISION_KIND,
        "decisions": [SELECT_VIEWPOINT, ABSTAIN],
        "proposal_frame_id": RELATIVE_GOAL_FRAME_ID,
        "grid_xy_has_goal_authority": False,
        "existing_slow_planner_v1_changed": False,
        "reachable_viewpoint_relabelled_as_frontier": False,
        "shared_ros_schema_changed": False,
        "online_runtime_wired": False,
        "authority": dict(_AUTHORITY),
        "invalid_timeout_or_abstain_fallback": FROZEN_INTERNVLA_FALLBACK_CANDIDATE,
    }
