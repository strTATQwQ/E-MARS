"""Step3-first natural-language mission normalization contracts.

The raw operator instruction terminates at Step3.  Downstream InternVLA and
navigation code may consume only :class:`CanonicalMission`, never the raw
instruction carried by :class:`MissionNormalizationRequest`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .base import PlannerMetrics, SlowPlannerProtocolError


MISSION_PROTOCOL_VERSION = 1
MISSION_NORMALIZATION_TYPE = "normalize_instruction"
MISSION_ROUTE = "step3_instruction_normalization_v1"
MISSION_NORMALIZATION_KEYS = frozenset(
    {
        "source_language",
        "canonical_instruction",
        "target_description",
        "constraints",
        "confidence",
        "abstain",
    }
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LANGUAGE_RE = re.compile(r"^[a-z]{2,8}(?:-[A-Z]{2})?$|^other$")
_SAFE_SOURCE_RE = re.compile(r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]{1,480}$")
_SAFE_ENGLISH_RE = re.compile(r"^[\x20-\x7e]{1,240}$")


def instruction_sha256(instruction: str) -> str:
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()


def _identifier(value: Any, name: str) -> str:
    result = str(value or "")
    if not _IDENTIFIER_RE.fullmatch(result):
        raise SlowPlannerProtocolError(f"{name} is not a bounded identifier")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise SlowPlannerProtocolError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SlowPlannerProtocolError(
            f"{name} must be a non-negative integer"
        ) from exc
    if result < 0:
        raise SlowPlannerProtocolError(f"{name} must be a non-negative integer")
    return result


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SlowPlannerProtocolError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SlowPlannerProtocolError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise SlowPlannerProtocolError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class MissionNormalizationRequest:
    mission_id: str
    episode_id: str
    reset_generation: int
    sequence_id: int
    instruction: str
    config_sha256: str
    timestamp: float = 0.0
    protocol_version: int = MISSION_PROTOCOL_VERSION
    route: str = MISSION_ROUTE

    def __post_init__(self) -> None:
        if self.protocol_version != MISSION_PROTOCOL_VERSION:
            raise SlowPlannerProtocolError("mission protocol version mismatch")
        if self.route != MISSION_ROUTE:
            raise SlowPlannerProtocolError("raw instruction did not select Step3-first route")
        _identifier(self.mission_id, "mission_id")
        _identifier(self.episode_id, "episode_id")
        _nonnegative_integer(self.reset_generation, "reset_generation")
        _nonnegative_integer(self.sequence_id, "sequence_id")
        normalized = " ".join(str(self.instruction).split())
        if not _SAFE_SOURCE_RE.fullmatch(normalized):
            raise SlowPlannerProtocolError("instruction is empty, too long, or unsafe")
        if not _SHA256_RE.fullmatch(str(self.config_sha256)):
            raise SlowPlannerProtocolError("config_sha256 must be a lowercase digest")
        timestamp = _finite(self.timestamp, "timestamp")
        if timestamp <= 0:
            raise SlowPlannerProtocolError("timestamp must be positive")

    @property
    def identity(self) -> str:
        return (
            f"real::{self.mission_id}::{self.reset_generation}::{self.sequence_id}"
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "route": self.route,
            "mission_id": self.mission_id,
            "episode_id": self.episode_id,
            "reset_generation": self.reset_generation,
            "sequence_id": self.sequence_id,
            "identity": self.identity,
            "instruction": " ".join(self.instruction.split()),
            "source_instruction_sha256": instruction_sha256(
                " ".join(self.instruction.split())
            ),
            "config_sha256": self.config_sha256,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "MissionNormalizationRequest":
        return cls(
            mission_id=str(value.get("mission_id") or ""),
            episode_id=str(value.get("episode_id") or ""),
            reset_generation=_nonnegative_integer(
                value.get("reset_generation"), "reset_generation"
            ),
            sequence_id=_nonnegative_integer(value.get("sequence_id"), "sequence_id"),
            instruction=str(value.get("instruction") or ""),
            config_sha256=str(value.get("config_sha256") or ""),
            timestamp=_finite(value.get("timestamp"), "timestamp"),
            protocol_version=int(
                value.get("protocol_version", MISSION_PROTOCOL_VERSION)
            ),
            route=str(value.get("route") or ""),
        )


@dataclass(frozen=True)
class CanonicalMission:
    mission_id: str
    episode_id: str
    reset_generation: int
    sequence_id: int
    source_language: str
    canonical_instruction: str
    target_description: str
    constraints: tuple[str, ...]
    confidence: float
    abstain: bool
    source_instruction_sha256: str
    config_sha256: str
    protocol_version: int = MISSION_PROTOCOL_VERSION
    route: str = MISSION_ROUTE

    def __post_init__(self) -> None:
        if self.protocol_version != MISSION_PROTOCOL_VERSION or self.route != MISSION_ROUTE:
            raise SlowPlannerProtocolError("canonical mission route/version mismatch")
        _identifier(self.mission_id, "mission_id")
        _identifier(self.episode_id, "episode_id")
        _nonnegative_integer(self.reset_generation, "reset_generation")
        _nonnegative_integer(self.sequence_id, "sequence_id")
        if not _LANGUAGE_RE.fullmatch(self.source_language):
            raise SlowPlannerProtocolError("source_language is invalid")
        if not isinstance(self.abstain, bool):
            raise SlowPlannerProtocolError("abstain must be boolean")
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise SlowPlannerProtocolError("confidence must be in [0,1]")
        if self.abstain:
            if self.canonical_instruction or self.target_description or self.constraints:
                raise SlowPlannerProtocolError(
                    "abstain must not release a downstream instruction"
                )
        else:
            if not _SAFE_ENGLISH_RE.fullmatch(self.canonical_instruction):
                raise SlowPlannerProtocolError(
                    "canonical_instruction must be bounded printable English/ASCII"
                )
            if not _SAFE_ENGLISH_RE.fullmatch(self.target_description):
                raise SlowPlannerProtocolError(
                    "target_description must be bounded printable English/ASCII"
                )
            if len(self.constraints) > 4 or any(
                not _SAFE_ENGLISH_RE.fullmatch(item) for item in self.constraints
            ):
                raise SlowPlannerProtocolError("constraints must be bounded English/ASCII")
        if not _SHA256_RE.fullmatch(self.source_instruction_sha256):
            raise SlowPlannerProtocolError("source instruction digest is invalid")
        if not _SHA256_RE.fullmatch(self.config_sha256):
            raise SlowPlannerProtocolError("config digest is invalid")

    @property
    def identity(self) -> str:
        return (
            f"real::{self.mission_id}::{self.reset_generation}::{self.sequence_id}"
        )

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["constraints"] = list(self.constraints)
        value["identity"] = self.identity
        value["internvla_raw_instruction_allowed"] = False
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CanonicalMission":
        return cls(
            mission_id=str(value.get("mission_id") or ""),
            episode_id=str(value.get("episode_id") or ""),
            reset_generation=_nonnegative_integer(
                value.get("reset_generation"), "reset_generation"
            ),
            sequence_id=_nonnegative_integer(value.get("sequence_id"), "sequence_id"),
            source_language=str(value.get("source_language") or ""),
            canonical_instruction=str(value.get("canonical_instruction") or ""),
            target_description=str(value.get("target_description") or ""),
            constraints=tuple(str(item) for item in (value.get("constraints") or [])),
            confidence=_finite(value.get("confidence"), "confidence"),
            abstain=value.get("abstain"),
            source_instruction_sha256=str(
                value.get("source_instruction_sha256") or ""
            ),
            config_sha256=str(value.get("config_sha256") or ""),
            protocol_version=int(
                value.get("protocol_version", MISSION_PROTOCOL_VERSION)
            ),
            route=str(value.get("route") or ""),
        )


def normalization_prompt(request: MissionNormalizationRequest) -> str:
    payload = {
        "instruction": " ".join(request.instruction.split()),
        "mission_id": request.mission_id,
    }
    return (
        "INPUT="
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\nNormalize this robot navigation request. Detect the source language and "
        "translate it into one concise English navigation instruction while preserving "
        "the destination, spatial relations, exclusions, and stop condition. Do not add "
        "objects or goals. If it is not a navigation request, is unsafe, or is too "
        "ambiguous to preserve, set abstain=true and return empty strings and an empty "
        "constraints array. No reasoning or prose. JSON only, exactly six keys: "
        '{"source_language":"zh","canonical_instruction":"Navigate to the target and stop.",'
        '"target_description":"the target","constraints":[],"confidence":0.0,'
        '"abstain":false}'
    )


def _normalization_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("</think>"):
        text = text[len("</think>") :].lstrip()
    fenced = False
    if text.startswith("```json"):
        text = text[len("```json") :].lstrip()
        fenced = True
    elif text.startswith("```"):
        text = text[len("```") :].lstrip()
        fenced = True
    if not text.startswith("{"):
        raise SlowPlannerProtocolError("mission normalization contains prose or reasoning")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise SlowPlannerProtocolError(
                    f"mission normalization repeats key: {key}"
                )
            value[key] = item
        return value

    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    try:
        value, end = decoder.raw_decode(text)
    except json.JSONDecodeError as exc:
        raise SlowPlannerProtocolError(
            "mission normalization is not complete JSON"
        ) from exc
    suffix = text[end:].strip()
    if suffix != ("```" if fenced else ""):
        raise SlowPlannerProtocolError("mission normalization has an unsafe suffix")
    if not isinstance(value, dict) or set(value) != MISSION_NORMALIZATION_KEYS:
        raise SlowPlannerProtocolError(
            "mission normalization must contain exactly six keys"
        )
    return value


def contains_complete_normalization_json(raw_text: str) -> bool:
    try:
        _normalization_json_object(raw_text)
    except SlowPlannerProtocolError:
        return False
    return True


def parse_canonical_mission(
    request: MissionNormalizationRequest, raw_text: str
) -> CanonicalMission:
    value = _normalization_json_object(raw_text)
    abstain = value["abstain"]
    if not isinstance(abstain, bool):
        raise SlowPlannerProtocolError("abstain must be boolean")
    constraints_raw = value["constraints"]
    if not isinstance(constraints_raw, list):
        raise SlowPlannerProtocolError("constraints must be an array")
    confidence = _finite(value["confidence"], "confidence")
    return CanonicalMission(
        mission_id=request.mission_id,
        episode_id=request.episode_id,
        reset_generation=request.reset_generation,
        sequence_id=request.sequence_id,
        source_language=str(value["source_language"]),
        canonical_instruction=" ".join(str(value["canonical_instruction"]).split()),
        target_description=" ".join(str(value["target_description"]).split()),
        constraints=tuple(" ".join(str(item).split()) for item in constraints_raw),
        confidence=confidence,
        abstain=abstain,
        source_instruction_sha256=instruction_sha256(
            " ".join(request.instruction.split())
        ),
        config_sha256=request.config_sha256,
    )


def normalization_response(
    mission: CanonicalMission,
    metrics: PlannerMetrics,
    *,
    server_total_ms: float,
) -> dict[str, Any]:
    return {
        "ok": True,
        "normalization": mission.to_mapping(),
        "metrics": metrics.to_mapping(),
        "server_total_ms": float(server_total_ms),
    }


def new_request(**kwargs: Any) -> MissionNormalizationRequest:
    """Small convenience for callers that do not supply an explicit timestamp."""

    kwargs.setdefault("timestamp", time.time())
    return MissionNormalizationRequest(**kwargs)
