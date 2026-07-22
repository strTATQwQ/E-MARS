from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from .model import Step3SplitModel
from .protocol import GraphNavProtocolError, GraphNavRequest, InvalidCandidateId


NAVIGATION_SYSTEM_PROMPT = (
    "You are NavigationPolicy. Choose the best adjacent graph direction for following the instruction. "
    "You are not an arrival detector and you are forbidden to STOP. Every RGB is photographed from "
    "the current viewpoint looking toward one adjacent edge. Think briefly, then select exactly one "
    "listed integer candidate_id. Do not output coordinates, waypoints, speed, turn commands, or prose "
    "after the final answer."
)


@dataclass(frozen=True)
class NavigationAction:
    candidate_id: int

    def to_mapping(self) -> dict[str, Any]:
        return {"action": "move", "candidate_id": self.candidate_id}


def parse_navigation_action(
    raw_text: str, valid_candidate_ids: frozenset[int]
) -> NavigationAction:
    try:
        value = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        raise GraphNavProtocolError(f"navigation output is not one JSON object: {exc.msg}") from exc
    if not isinstance(value, dict) or set(value) != {"action", "candidate_id"}:
        raise GraphNavProtocolError("navigation JSON must contain exactly action and candidate_id")
    if value.get("action") != "move":
        raise GraphNavProtocolError("NavigationPolicy may only output move")
    candidate_id = value.get("candidate_id")
    if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
        raise GraphNavProtocolError("candidate_id must be an integer")
    if candidate_id not in valid_candidate_ids:
        raise InvalidCandidateId(f"candidate_id={candidate_id} is not a current candidate")
    return NavigationAction(candidate_id=candidate_id)


def build_navigation_prompt(request: GraphNavRequest) -> str:
    candidates = "\n".join(
        (
            f"- candidate_id={item.candidate_id}, image_index={index}, "
            f"relative_heading_deg={item.relative_heading_deg:.1f}, "
            f"edge_distance_m={item.graph_distance_m:.2f}"
        )
        for index, item in enumerate(request.candidates)
    )
    history = "\n".join(f"- {item}" for item in request.history[-4:]) or "- none"
    return f"""INSTRUCTION:
{request.instruction}

CURRENT STEP: {request.step_index}

ADJACENT RGB DIRECTIONS (images are attached in exactly this order):
{candidates}

RECENT EXECUTED ACTION SUMMARY:
{history}

Briefly reason about instruction progress and visual direction. The final response is strictly:
{{"action":"move","candidate_id":N}}

N must be one listed integer candidate_id. STOP and arrival judgments are forbidden."""


class NavigationPolicy:
    def __init__(self, model: Step3SplitModel) -> None:
        self.model = model

    def decide(self, request: GraphNavRequest) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        images, image_decode_ms = self.model.prepare_images(item.jpeg for item in request.candidates)
        branches = {str(candidate_id): str(candidate_id) for candidate_id in sorted(request.candidate_ids)}
        scores, metrics = self.model.score_branches(
            system_prompt=NAVIGATION_SYSTEM_PROMPT,
            user_prompt=build_navigation_prompt(request),
            images=images,
            assistant_prefix='{"action":"move","candidate_id":',
            branch_texts=branches,
        )
        ranked = sorted((int(key) for key in scores), key=lambda key: (-scores[str(key)], key))
        selected = ranked[0]
        raw_output = json.dumps(
            {"action": "move", "candidate_id": selected}, separators=(",", ":")
        )
        return raw_output, {
            **metrics,
            "image_decode_ms": image_decode_ms,
            "image_count": len(images),
            "raw_image_resolutions": [[item.width, item.height] for item in request.candidates],
            "candidate_scores": {key: scores[key] for key in sorted(scores, key=int)},
            "ranked_candidate_ids": ranked,
            "policy_total_ms": (time.perf_counter() - started) * 1000.0,
            "output_schema": "move_only_candidate_id",
        }

