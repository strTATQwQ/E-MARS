from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .base import (
    PlannerDecision,
    SlowPlannerProtocolError,
    SlowPlannerRequest,
)
from .step3_vl_10b import (
    Step3VLSlowPlanner,
    _semantic_list,
    _semantic_text,
    _step3_json_object,
)


TASK_STATE_PREFIX = "task_state_v1="
TASK_STATE_TRANSITIONS = frozenset({"hold", "advance", "recover"})
TASK_STATE_KEYS = frozenset(
    {
        "task_clauses",
        "active_clause_id",
        "clause_transition",
        "recommended_frontier",
        "confidence",
        "evidence",
        "target_found",
        "abstain",
    }
)
_SEQUENCE_CUE = re.compile(
    r"(?:[.;。；！？!?](?=\s*\S)\s*|"
    r"\b(?:then|and then|after that|next)\b|"
    r"(?:然后|接着|随后|之后|再))",
    flags=re.IGNORECASE,
)
_NAVIGATION_CONJUNCTION = re.compile(
    r"\band\s+(?=(?:go|walk|head|take|make|turn|bear|veer|continue|"
    r"proceed|pass|enter|exit|stop)\b)",
    flags=re.IGNORECASE,
)
_TERMINAL_ONLY_STOP = re.compile(
    r"^stop(?:\s+(?:there|here))?[.!?]*$", flags=re.IGNORECASE
)
_TERMINAL_STOP_SUFFIX = re.compile(
    r"\b(?:then|and)\s+stop(?:\s+(?:there|here))?[.!?]*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class TaskStateContext:
    instruction: str
    clauses: tuple[str, ...]
    active_clause_id: int
    revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.instruction, str):
            raise SlowPlannerProtocolError(
                "task-state instruction must be a string"
            )
        instruction = " ".join(self.instruction.split())
        if (
            not instruction
            or len(instruction) > 512
            or instruction != self.instruction
        ):
            raise SlowPlannerProtocolError(
                "task-state instruction must be normalized and bounded"
            )
        if (
            not isinstance(self.clauses, tuple)
            or any(not isinstance(item, str) for item in self.clauses)
            or len(self.clauses) > 4
        ):
            raise SlowPlannerProtocolError("task state supports at most four clauses")
        if (
            isinstance(self.active_clause_id, bool)
            or not isinstance(self.active_clause_id, int)
        ):
            raise SlowPlannerProtocolError("active clause ID must be an integer")
        if self.clauses:
            validate_instruction_clauses(self.instruction, self.clauses)
            if not 1 <= self.active_clause_id <= len(self.clauses):
                raise SlowPlannerProtocolError("active clause ID is out of range")
        elif self.active_clause_id != 0:
            raise SlowPlannerProtocolError(
                "uninitialized task state must use active_clause_id=0"
            )
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise SlowPlannerProtocolError("task-state revision must be non-negative")

    def to_history(self) -> str:
        value = {
            "schema_version": 1,
            "instruction": self.instruction,
            "clauses": list(self.clauses),
            "active_clause_id": self.active_clause_id,
            "revision": self.revision,
        }
        return TASK_STATE_PREFIX + json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


@dataclass(frozen=True)
class TaskStatePlannerDecision(PlannerDecision):
    """Private Step3 task-state extension beside the frozen protocol-v1 request."""

    task_clauses: tuple[str, ...] = ()
    active_clause_id: int = 0
    clause_transition: str = "hold"
    recommended_frontier: int | None = None
    evidence: tuple[str, ...] = ()
    target_found: bool = False
    abstain: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            isinstance(self.active_clause_id, bool)
            or not isinstance(self.active_clause_id, int)
            or not 1 <= self.active_clause_id <= 4
        ):
            raise SlowPlannerProtocolError("active_clause_id must be in [1, 4]")
        if self.clause_transition not in TASK_STATE_TRANSITIONS:
            raise SlowPlannerProtocolError("invalid task-state transition")
        if (
            not isinstance(self.task_clauses, tuple)
            or len(self.task_clauses) > 4
            or any(
                not isinstance(item, str)
                or not item
                or item != " ".join(item.split())
                or len(item) > 160
                for item in self.task_clauses
            )
        ):
            raise SlowPlannerProtocolError(
                "task_clauses must contain at most four normalized strings"
            )
        if self.task_clauses and self.active_clause_id > len(self.task_clauses):
            raise SlowPlannerProtocolError(
                "active_clause_id exceeds task_clauses"
            )
        if (
            not isinstance(self.evidence, tuple)
            or len(self.evidence) > 2
            or any(
                not isinstance(item, str)
                or not item
                or item != " ".join(item.split())
                or len(item) > 96
                for item in self.evidence
            )
        ):
            raise SlowPlannerProtocolError(
                "evidence must contain at most two normalized strings"
            )
        if not isinstance(self.target_found, bool) or not isinstance(
            self.abstain, bool
        ):
            raise SlowPlannerProtocolError(
                "target_found and abstain must be booleans"
            )
        if self.clause_transition == "advance" and not self.evidence:
            raise SlowPlannerProtocolError(
                "advance requires bounded visible evidence"
            )
        if self.decision == "select_frontier":
            if self.abstain or self.target_found:
                raise SlowPlannerProtocolError(
                    "task-state frontier selection cannot abstain or target_found"
                )
            if self.recommended_frontier != self.frontier_id:
                raise SlowPlannerProtocolError(
                    "recommended frontier must match selected frontier"
                )
        elif self.decision == "abstain":
            if not self.abstain or self.recommended_frontier is not None:
                raise SlowPlannerProtocolError(
                    "task-state abstain requires null recommended frontier"
                )
        else:
            raise SlowPlannerProtocolError(
                "task-state decisions may only select a frontier or abstain"
            )
        if self.target_found and not self.evidence:
            raise SlowPlannerProtocolError(
                "target_found requires bounded visible evidence"
            )


def validate_instruction_clauses(
    instruction: str, clauses: tuple[str, ...]
) -> tuple[str, ...]:
    if not 1 <= len(clauses) <= 4:
        raise SlowPlannerProtocolError(
            "task decomposition must contain one to four clauses"
        )
    normalized_instruction = " ".join(instruction.split())
    lowered = normalized_instruction.casefold()
    cursor = 0
    normalized: list[str] = []
    for index, value in enumerate(clauses):
        clause = _semantic_text(
            value, f"task_clauses[{index}]", max_length=160
        )
        offset = lowered.find(clause.casefold(), cursor)
        if offset < 0:
            raise SlowPlannerProtocolError(
                "task clauses must be ordered exact spans of the instruction"
            )
        cursor = offset + len(clause)
        normalized.append(clause)
    merged_terminal_single = (
        len(normalized) == 1
        and normalized[0].casefold() == normalized_instruction.casefold()
        and _TERMINAL_STOP_SUFFIX.search(normalized_instruction) is not None
    )
    if (
        _SEQUENCE_CUE.search(normalized_instruction)
        and len(normalized) < 2
        and not merged_terminal_single
    ):
        raise SlowPlannerProtocolError(
            "instruction with sequence cues requires at least two clauses"
        )
    return tuple(normalized)


def fallback_instruction_clauses(instruction: str) -> tuple[str, ...]:
    """Return a deterministic bounded fallback when Step3 decomposition fails."""

    normalized = " ".join(instruction.split())
    if not normalized:
        raise SlowPlannerProtocolError("instruction is required")
    spans: list[str] = []
    cursor = 0
    for match in _SEQUENCE_CUE.finditer(normalized):
        value = normalized[cursor : match.start()].strip(" ,")
        if value:
            spans.append(value)
        cursor = match.end()
    tail = normalized[cursor:].strip(" ,")
    if tail:
        spans.append(tail)
    if not spans:
        spans = [normalized]

    # Some R2R instructions are one long chain joined only by "and".  Keep
    # ordinary short phrases such as "turn left and exit" intact, but split an
    # over-limit span at a following navigation verb so the deterministic
    # fallback remains both semantic and bounded.
    if any(len(span) > 160 for span in spans):
        refined: list[str] = []
        search_cursor = 0
        for span in spans:
            start = normalized.casefold().find(span.casefold(), search_cursor)
            if start < 0:
                raise SlowPlannerProtocolError(
                    "fallback clause is not an exact instruction span"
                )
            end = start + len(span)
            local_cursor = start
            for match in _NAVIGATION_CONJUNCTION.finditer(
                normalized, start, end
            ):
                value = normalized[local_cursor : match.start()].strip(" ,")
                if value:
                    refined.append(value)
                local_cursor = match.end()
            value = normalized[local_cursor:end].strip(" ,")
            if value:
                refined.append(value)
            search_cursor = end
        spans = refined

    # A standalone final "stop" contains no observable destination.  Attach it
    # to the preceding exact-span destination clause so arrival classification
    # sees the landmark and the terminal instruction together.
    if len(spans) >= 2 and _TERMINAL_ONLY_STOP.fullmatch(spans[-1]):
        lowered = normalized.casefold()
        terminal_start = lowered.rfind(spans[-1].casefold())
        previous_start = lowered.rfind(
            spans[-2].casefold(), 0, terminal_start
        )
        if previous_start < 0 or terminal_start < 0:
            raise SlowPlannerProtocolError(
                "terminal stop cannot be joined to its destination clause"
            )
        terminal_end = terminal_start + len(spans[-1])
        spans[-2:] = [normalized[previous_start:terminal_end].strip(" ,")]

    if len(spans) > 4:
        spans = spans[:3] + [
            normalized[normalized.casefold().find(spans[3].casefold()) :]
        ]
    return validate_instruction_clauses(normalized, tuple(spans))


def task_state_context(request: SlowPlannerRequest) -> TaskStateContext | None:
    rows = [
        item
        for item in request.compact_history
        if item.startswith(TASK_STATE_PREFIX)
    ]
    if not rows:
        return None
    if len(rows) != 1:
        raise SlowPlannerProtocolError(
            "request must contain exactly one task-state marker"
        )
    try:
        value = json.loads(rows[0][len(TASK_STATE_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise SlowPlannerProtocolError("task-state marker is not JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "instruction",
        "clauses",
        "active_clause_id",
        "revision",
    }:
        raise SlowPlannerProtocolError("invalid task-state marker schema")
    if value["schema_version"] != 1:
        raise SlowPlannerProtocolError("invalid task-state schema version")
    if not isinstance(value["instruction"], str):
        raise SlowPlannerProtocolError(
            "task-state instruction must be a string"
        )
    clauses = value["clauses"]
    if not isinstance(clauses, list) or any(
        not isinstance(item, str) for item in clauses
    ):
        raise SlowPlannerProtocolError("task-state clauses must be an array")
    if (
        isinstance(value["active_clause_id"], bool)
        or not isinstance(value["active_clause_id"], int)
        or isinstance(value["revision"], bool)
        or not isinstance(value["revision"], int)
    ):
        raise SlowPlannerProtocolError(
            "task-state identity and revision must be integers"
        )
    return TaskStateContext(
        instruction=" ".join(value["instruction"].split()),
        clauses=tuple(clauses),
        active_clause_id=value["active_clause_id"],
        revision=value["revision"],
    )


def _task_state_json_object(raw_text: str) -> dict[str, Any]:
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
        raise SlowPlannerProtocolError(
            "task-state response has prose or unsupported reasoning"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise SlowPlannerProtocolError(
                    f"task-state response repeats key: {key}"
                )
            value[key] = item
        return value

    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    try:
        value, end = decoder.raw_decode(text)
    except json.JSONDecodeError as exc:
        raise SlowPlannerProtocolError(
            "task-state response is not complete JSON"
        ) from exc
    suffix = text[end:].strip()
    if suffix != ("```" if fenced else ""):
        raise SlowPlannerProtocolError(
            "task-state response has prose or unsupported suffix"
        )
    if not isinstance(value, dict) or set(value) != TASK_STATE_KEYS:
        raise SlowPlannerProtocolError(
            "task-state response must contain exactly eight keys"
        )
    return value


class Step3TaskStateSlowPlanner(Step3VLSlowPlanner):
    """Step3 prompt/parser for bounded instruction clauses and state updates."""

    planner_mode = "task_state_v1"

    def format_prompt(
        self, request: SlowPlannerRequest, *, correction: str = ""
    ) -> str:
        state = task_state_context(request)
        if state is None:
            return super().format_prompt(request, correction=correction)
        payload = {
            "instruction": state.instruction,
            "task_state": {
                "clauses": list(state.clauses),
                "active_clause_id": state.active_clause_id,
                "revision": state.revision,
            },
            "motion_constraint": request.instruction,
            "views": [
                {
                    "i": index,
                    "id": image.view_id,
                    "pose": [round(value, 4) for value in image.pose],
                }
                for index, image in enumerate(request.ordered_images)
            ],
            "frontiers": [
                {
                    "id": item.frontier_id,
                    "xz": [round(value, 3) for value in item.relative_xz],
                    "d": round(item.distance_m, 3),
                    "b": round(item.bearing_deg, 2),
                }
                for item in request.candidate_frontiers
            ],
            "history": [
                item
                for item in request.compact_history[-8:]
                if not item.startswith(TASK_STATE_PREFIX)
            ],
        }
        correction_line = f"\nINVALID={correction}" if correction else ""
        initialized = bool(state.clauses)
        arrival_shadow = (
            not request.candidate_frontiers
            and "arrival_evidence_contract=goal_region|strict_semantic_arrival"
            in request.compact_history
        )
        clause_rule = (
            "task_clauses must be [] and active_clause_id must echo "
            f"{state.active_clause_id}."
            if initialized
            else "Decompose the instruction into 1..4 ordered exact text spans; "
            "do not paraphrase or invent text. active_clause_id must be 1."
        )
        evidence_rule = (
            "For this observation-only arrival shadow, evidence must contain "
            "exactly two entries in this order: 'goal_region: yes|no - <visible "
            "fact>' and 'semantic_arrival: yes|no - <visible fact>'. "
            if arrival_shadow
            else "For hold/recover, prefer evidence=[]; otherwise evidence contains "
            "at most one visible fact of six words. "
        )
        if arrival_shadow:
            # The general eight-field task-state instructions encouraged Step3
            # to spend the entire realtime decode budget restating the task.  An
            # arrival probe has no movement authority and all fields except the
            # two visual verdicts are identity-bound constants, so give it one
            # short copy-and-edit template.  The wire response and parser remain
            # unchanged; this only bounds the model's private generated text.
            template = {
                "task_clauses": [],
                "active_clause_id": state.active_clause_id,
                "clause_transition": "hold",
                "recommended_frontier": None,
                "confidence": 0.0,
                "evidence": [
                    "goal_region: no - not visible",
                    "semantic_arrival: no - not visible",
                ],
                "target_found": False,
                "abstain": True,
            }
            return (
                "INPUT="
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + correction_line
                + "\nARRIVAL_SHADOW_ONLY. Return one compact JSON object and no "
                "reasoning or prose. Copy this exact eight-key object and change "
                "only yes/no, the two facts (maximum three words each), confidence, "
                "clause_transition, and target_found: "
                + json.dumps(template, ensure_ascii=False, separators=(",", ":"))
                + " Use clause_transition=advance and target_found=true only when "
                "both goal_region and semantic_arrival are yes; otherwise keep "
                "hold and false. recommended_frontier must remain null, abstain "
                "must remain true, task_clauses must remain empty, and "
                f"active_clause_id must remain {state.active_clause_id}. This is "
                "shadow evidence only and never grants terminal STOP."
            )
        return (
            "INPUT="
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + correction_line
            + "\nYou are the high-level task decomposer and state updater. "
            + clause_rule
            + " Inspect the four current views. clause_transition is hold, advance, "
            "or recover. Use advance only when visible evidence shows the active "
            "clause is satisfied; if it is not the final clause, recommend a legal "
            "frontier for the next clause when possible. Use recover when the active "
            "clause is visibly blocked or the recent cycle is unproductive; when any "
            "listed recovery frontier is usable, select it instead of abstaining. "
            + evidence_rule
            + "recommended_frontier is one "
            "listed integer or null. confidence is 0..1. target_found is true only "
            "for visible completion of the final clause; it implies abstain=true and "
            "does not grant terminal STOP. Otherwise, if no legal action is justified, "
            "set abstain=true. Fit the response in the realtime budget: use compact "
            "JSON, a one-decimal confidence, and at most six words per evidence item. "
            "Return JSON only with exactly eight keys: "
            '{"task_clauses":[],"active_clause_id":1,'
            '"clause_transition":"hold|advance|recover",'
            '"recommended_frontier":null,"confidence":0.0,"evidence":[],'
            '"target_found":false,"abstain":true}'
        )

    def parse_decision(
        self, request: SlowPlannerRequest, raw_text: str, *, attempts: int
    ) -> PlannerDecision:
        state = task_state_context(request)
        if state is None:
            return super().parse_decision(request, raw_text, attempts=attempts)
        value = _task_state_json_object(raw_text)
        clauses = _semantic_list(
            value["task_clauses"],
            "task_clauses",
            maximum_items=4,
            item_max_length=160,
        )
        if state.clauses:
            if clauses:
                raise SlowPlannerProtocolError(
                    "initialized task state forbids plan replacement"
                )
            expected_active = state.active_clause_id
            effective_clauses = state.clauses
        else:
            clauses = validate_instruction_clauses(state.instruction, clauses)
            expected_active = 1
            effective_clauses = clauses
        active_clause_id = value["active_clause_id"]
        if (
            isinstance(active_clause_id, bool)
            or not isinstance(active_clause_id, int)
            or active_clause_id != expected_active
        ):
            raise SlowPlannerProtocolError(
                "Step3 must echo the identity-current active clause"
            )
        transition = value["clause_transition"]
        if transition not in TASK_STATE_TRANSITIONS:
            raise SlowPlannerProtocolError("invalid clause_transition")
        evidence = _semantic_list(
            value["evidence"],
            "evidence",
            maximum_items=2,
            item_max_length=96,
        )
        arrival_shadow = (
            not request.candidate_frontiers
            and "arrival_evidence_contract=goal_region|strict_semantic_arrival"
            in request.compact_history
        )
        if arrival_shadow:
            patterns = (
                re.compile(r"^goal_region:\s*(?:yes|no)\b", re.IGNORECASE),
                re.compile(
                    r"^semantic_arrival:\s*(?:yes|no)\b", re.IGNORECASE
                ),
            )
            if len(evidence) != 2 or any(
                pattern.search(item) is None
                for pattern, item in zip(patterns, evidence)
            ):
                raise SlowPlannerProtocolError(
                    "arrival shadow requires two labeled yes/no evidence verdicts"
                )
        confidence = value["confidence"]
        if isinstance(confidence, bool):
            raise SlowPlannerProtocolError("confidence must be numeric")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise SlowPlannerProtocolError("confidence must be numeric") from exc
        if not 0.0 <= confidence <= 1.0:
            raise SlowPlannerProtocolError("confidence must be in [0, 1]")
        target_found = value["target_found"]
        abstain = value["abstain"]
        if not isinstance(target_found, bool) or not isinstance(abstain, bool):
            raise SlowPlannerProtocolError(
                "target_found and abstain must be booleans"
            )
        final_clause = active_clause_id == len(effective_clauses)
        if target_found and (not final_clause or transition != "advance"):
            raise SlowPlannerProtocolError(
                "target_found requires final-clause advance"
            )
        if arrival_shadow and target_found and any(
            re.match(pattern, item, re.IGNORECASE) is None
            for pattern, item in zip(
                (r"^goal_region:\s*yes\b", r"^semantic_arrival:\s*yes\b"),
                evidence,
            )
        ):
            raise SlowPlannerProtocolError(
                "target_found requires both arrival verdicts yes"
            )
        if transition == "advance" and not evidence:
            raise SlowPlannerProtocolError(
                "advance requires visible evidence"
            )
        recommended = value["recommended_frontier"]
        if recommended is not None and (
            isinstance(recommended, bool) or not isinstance(recommended, int)
        ):
            raise SlowPlannerProtocolError(
                "recommended_frontier must be integer or null"
            )
        legal_ids = {item.frontier_id for item in request.candidate_frontiers}
        if abstain:
            if recommended is not None:
                raise SlowPlannerProtocolError(
                    "abstain requires null recommended_frontier"
                )
            decision = "abstain"
            frontier_id = None
        else:
            if target_found:
                raise SlowPlannerProtocolError(
                    "target_found requires abstain=true"
                )
            if recommended not in legal_ids:
                raise SlowPlannerProtocolError(
                    "recommended_frontier is not currently legal"
                )
            decision = "select_frontier"
            frontier_id = recommended
        return TaskStatePlannerDecision(
            episode_id=request.episode_id,
            snapshot_id=request.snapshot_id,
            decision=decision,
            frontier_id=frontier_id,
            target_relative_xz=None,
            confidence=confidence,
            raw_text=raw_text,
            parse_attempts=attempts,
            task_clauses=clauses,
            active_clause_id=active_clause_id,
            clause_transition=transition,
            recommended_frontier=recommended,
            evidence=evidence,
            target_found=target_found,
            abstain=abstain,
        )

    def contains_complete_json(self, raw_text: str) -> bool:
        try:
            _task_state_json_object(raw_text)
        except SlowPlannerProtocolError:
            try:
                _step3_json_object(raw_text)
            except SlowPlannerProtocolError:
                return False
        return True

    def health(self) -> dict[str, Any]:
        return {
            **super().health(),
            "planner_mode": self.planner_mode,
            "private_task_state_contract": 1,
        }
