from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from .base import (
    PlannerDecision,
    PlannerMetrics,
    SlowPlannerProtocolError,
    SlowPlannerRequest,
    StructuredPlannerDecision,
)
from .lane_b import REV_C_VIEW_ORDER
from .hf_base import StepVisionLanguagePlanner
from .mission import (
    CanonicalMission,
    MissionNormalizationRequest,
    contains_complete_normalization_json,
    normalization_prompt,
    parse_canonical_mission,
)


CHECKPOINT_KEY_MAPPING_NAME = "step3_flat_checkpoint_to_nested_v1"
CHECKPOINT_KEY_MAPPING = {
    "^vision_model": "model.vision_model",
    r"^model(?!\.(language_model|vision_model))": "model.language_model",
    "^vit_large_projector": "model.vit_large_projector",
}
STEP3_CHAT_GENERATION_SUFFIX = "<|im_start|>assistant\n<think>\n"
STEP3_NO_THINKING_PREFILL = "</think>\n"
STEP3_STRUCTURED_KEYS = frozenset(
    {
        "scene_summary",
        "target_evidence",
        "blocked_directions",
        "recommended_frontier",
        "confidence",
        "target_found",
        "abstain",
    }
)
_SAFE_SEMANTIC_TEXT = re.compile(r"^[^\x00-\x1f\x7f<>`]{1,160}$")
STEP3_SYSTEM_PROMPT = (
    "Navigation adviser. Use only attached images and supplied data. "
    "Describe visible facts, never private reasoning. Never invent a frontier ID. "
    "Return one compact JSON object. You have no cmd_vel, coordinate, or terminal "
    "STOP authority."
)


def _interleaved_labeled_images(
    images: list[Any], prompt: str
) -> list[dict[str, Any]]:
    """Bind image labels only for terminal/arrival classification requests."""

    view_ids = [f"image_{index}" for index in range(len(images))]
    arrival_request = False
    first_line = prompt.splitlines()[0] if prompt else ""
    if first_line.startswith("INPUT="):
        try:
            payload = json.loads(first_line[len("INPUT=") :])
            views = payload.get("views", [])
            arrival_request = any(
                str(item).startswith("termination_candidate=")
                for item in payload.get("history", [])
            )
            parsed = [
                str(item["id"])
                for index, item in enumerate(views)
                if isinstance(item, dict) and item.get("i") == index
            ]
            if len(parsed) == len(images) and len(set(parsed)) == len(parsed):
                view_ids = parsed
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    if not arrival_request:
        return [{"type": "image", "image": image} for image in images]
    content: list[dict[str, Any]] = []
    for index, (image, view_id) in enumerate(zip(images, view_ids, strict=True)):
        content.append(
            {
                "type": "text",
                "text": f"IMAGE_INDEX={index}; VIEW_ID={view_id}\n",
            }
        )
        content.append({"type": "image", "image": image})
    return content


def _step3_json_object(raw_text: str) -> dict[str, Any]:
    """Decode one exact seven-field object with only inert wrapper tokens allowed."""

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
            "Step3 response has prose, reasoning, or an unsupported prefix"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise SlowPlannerProtocolError(f"Step3 response repeats key: {key}")
            value[key] = item
        return value

    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    try:
        value, end = decoder.raw_decode(text)
    except json.JSONDecodeError as exc:
        raise SlowPlannerProtocolError("Step3 response is not complete JSON") from exc
    suffix = text[end:].strip()
    if suffix != ("```" if fenced else ""):
        raise SlowPlannerProtocolError(
            "Step3 response has prose, reasoning, or an unsupported suffix"
        )
    if not isinstance(value, dict) or set(value) != STEP3_STRUCTURED_KEYS:
        raise SlowPlannerProtocolError(
            "Step3 response must contain exactly seven structured decision keys"
        )
    return value


def _semantic_text(value: Any, name: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise SlowPlannerProtocolError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if len(normalized) > max_length or not _SAFE_SEMANTIC_TEXT.fullmatch(normalized):
        raise SlowPlannerProtocolError(f"{name} is not a bounded semantic string")
    return normalized


def _semantic_list(
    value: Any,
    name: str,
    *,
    maximum_items: int,
    item_max_length: int,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise SlowPlannerProtocolError(f"{name} must be a bounded array")
    result = tuple(
        _semantic_text(item, f"{name}[{index}]", max_length=item_max_length)
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise SlowPlannerProtocolError(f"{name} must not contain duplicates")
    return result


class Step3VLSlowPlanner(StepVisionLanguagePlanner):
    model_variant = "step3_vl_10b_bf16"
    precision_mode = "bf16"

    def format_prompt(
        self, request: SlowPlannerRequest, *, correction: str = ""
    ) -> str:
        payload = {
            "instruction": request.instruction,
            "agent_pose": [round(value, 4) for value in request.agent_pose],
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
            "visited": list(request.visited_frontiers),
            "history": list(request.compact_history[-8:]),
        }
        correction_line = f"\nINVALID={correction}" if correction else ""
        return (
            "INPUT="
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + correction_line
            + "\nRules: scene_summary <=12 words; target_evidence is 0..2 short "
            "visible facts; blocked_directions is a subset of "
            '["front_left","front","front_right","rear"]. '
            "recommended_frontier is one listed integer or null. confidence is 0..1. "
            "If uncertain, blocked, or no listed choice is justified: abstain=true and "
            "recommended_frontier=null. If the destination is visibly reached: "
            "target_found=true, abstain=true, recommended_frontier=null. Otherwise set "
            "both booleans false and recommend one listed frontier. JSON only, exactly "
            "seven keys: "
            '{"scene_summary":"visible scene","target_evidence":[],'
            '"blocked_directions":[],"recommended_frontier":null,"confidence":0.0,'
            '"target_found":false,"abstain":true}'
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str = "cuda",
        max_new_tokens: int = 96,
        fix_mistral_regex: bool = True,
        processor_use_fast: bool = False,
        expected_transformers_version: str = "4.57.6",
        checkpoint_key_mapping: str = CHECKPOINT_KEY_MAPPING_NAME,
        require_clean_checkpoint_load: bool = True,
        require_all_parameters_bf16: bool = True,
        skip_private_reasoning: bool = True,
        stop_on_complete_json: bool = True,
        generation_wall_budget_s: float = 10.5,
        generation_join_grace_s: float = 0.5,
    ) -> "Step3VLSlowPlanner":
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor

        if transformers.__version__ != expected_transformers_version:
            raise RuntimeError(
                "Step3 runtime transformers version mismatch: "
                f"expected={expected_transformers_version!r} actual={transformers.__version__!r}"
            )
        if checkpoint_key_mapping != CHECKPOINT_KEY_MAPPING_NAME:
            raise RuntimeError(
                "Step3 checkpoint key mapping mismatch: "
                f"expected={CHECKPOINT_KEY_MAPPING_NAME!r} actual={checkpoint_key_mapping!r}"
            )

        processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            fix_mistral_regex=fix_mistral_regex,
            use_fast=processor_use_fast,
        )
        loaded = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
            key_mapping=dict(CHECKPOINT_KEY_MAPPING),
            output_loading_info=True,
        )
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise RuntimeError("Step3 loader did not return model plus loading_info")
        model, loading_info = loaded
        loading_counts = {
            name: len(loading_info.get(name) or [])
            for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        }
        if require_clean_checkpoint_load and any(loading_counts.values()):
            raise RuntimeError(f"Step3 checkpoint load was not clean: {loading_counts}")

        parameter_dtype_counts: dict[str, int] = {}
        parameter_count = 0
        parameter_bytes = 0
        for parameter in model.parameters():
            count = int(parameter.numel())
            dtype_name = str(parameter.dtype)
            parameter_dtype_counts[dtype_name] = parameter_dtype_counts.get(dtype_name, 0) + count
            parameter_count += count
            parameter_bytes += count * int(parameter.element_size())
        if require_all_parameters_bf16 and set(parameter_dtype_counts) != {str(torch.bfloat16)}:
            raise RuntimeError(f"Step3 parameter dtypes are not all BF16: {parameter_dtype_counts}")

        model.eval()
        planner = cls(
            model=model,
            processor=processor,
            device=device,
            max_new_tokens=max_new_tokens,
            processor_use_fast=processor_use_fast,
            stop_on_complete_json=stop_on_complete_json,
            generation_wall_budget_s=generation_wall_budget_s,
            generation_join_grace_s=generation_join_grace_s,
        )
        if not skip_private_reasoning:
            raise RuntimeError("T5 Step3 runtime requires skip_private_reasoning=true")
        planner.skip_private_reasoning = True
        planner.fix_mistral_regex = fix_mistral_regex
        planner.runtime_transformers_version = transformers.__version__
        planner.checkpoint_key_mapping = checkpoint_key_mapping
        planner.checkpoint_loading_counts = loading_counts
        planner.checkpoint_load_clean = not any(loading_counts.values())
        planner.parameter_count = parameter_count
        planner.parameter_bytes = parameter_bytes
        planner.parameter_dtype_counts = parameter_dtype_counts
        return planner

    def parse_decision(
        self, request: SlowPlannerRequest, raw_text: str, *, attempts: int
    ) -> PlannerDecision:
        value = _step3_json_object(raw_text)
        scene_summary = _semantic_text(
            value["scene_summary"], "scene_summary", max_length=160
        )
        target_evidence = _semantic_list(
            value["target_evidence"],
            "target_evidence",
            maximum_items=2,
            item_max_length=96,
        )
        blocked_directions = _semantic_list(
            value["blocked_directions"],
            "blocked_directions",
            maximum_items=len(REV_C_VIEW_ORDER),
            item_max_length=16,
        )
        if any(item not in REV_C_VIEW_ORDER for item in blocked_directions):
            raise SlowPlannerProtocolError(
                "blocked_directions must use only the four Rev-C view IDs"
            )
        confidence = value["confidence"]
        if isinstance(confidence, bool):
            raise SlowPlannerProtocolError("confidence must be numeric")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise SlowPlannerProtocolError("confidence must be numeric") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise SlowPlannerProtocolError("confidence must be in [0, 1]")
        target_found = value["target_found"]
        abstain = value["abstain"]
        if not isinstance(target_found, bool) or not isinstance(abstain, bool):
            raise SlowPlannerProtocolError("target_found and abstain must be booleans")
        recommended = value["recommended_frontier"]
        if recommended is not None and (
            isinstance(recommended, bool) or not isinstance(recommended, int)
        ):
            raise SlowPlannerProtocolError(
                "recommended_frontier must be an integer or null"
            )
        legal_ids = {item.frontier_id for item in request.candidate_frontiers}
        if abstain:
            if recommended is not None:
                raise SlowPlannerProtocolError(
                    "abstain requires null recommended_frontier"
                )
            if target_found and not target_evidence:
                raise SlowPlannerProtocolError(
                    "target_found requires visible target_evidence"
                )
            decision = "abstain"
            frontier_id = None
        else:
            if target_found:
                raise SlowPlannerProtocolError("target_found requires abstain=true")
            if recommended not in legal_ids:
                raise SlowPlannerProtocolError(
                    "recommended_frontier is not a current legal frontier"
                )
            decision = "select_frontier"
            frontier_id = recommended
        return StructuredPlannerDecision(
            episode_id=request.episode_id,
            snapshot_id=request.snapshot_id,
            decision=decision,
            frontier_id=frontier_id,
            target_relative_xz=None,
            confidence=confidence,
            raw_text=raw_text,
            parse_attempts=attempts,
            scene_summary=scene_summary,
            target_evidence=target_evidence,
            blocked_directions=blocked_directions,
            recommended_frontier=recommended,
            target_found=target_found,
            abstain=abstain,
        )

    def contains_complete_json(self, raw_text: str) -> bool:
        try:
            _step3_json_object(raw_text)
        except SlowPlannerProtocolError:
            return False
        return True

    def normalize_instruction(
        self, request: MissionNormalizationRequest
    ) -> tuple[CanonicalMission, PlannerMetrics]:
        """Normalize multilingual operator text before InternVLA can observe it."""

        raw_text, metrics = self.generate_prompt_raw(
            normalization_prompt(request),
            complete_json_predicate=contains_complete_normalization_json,
        )
        return parse_canonical_mission(request, raw_text), metrics

    def prepare_inputs(self, *, images: list[Any], prompt: str) -> tuple[Any, float, float]:
        """Close Step3's forced thinking block before deterministic JSON generation."""

        content = _interleaved_labeled_images(images, prompt)
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": STEP3_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        started = time.perf_counter()
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not text.endswith(STEP3_CHAT_GENERATION_SUFFIX):
            raise RuntimeError("unexpected Step3 chat-template generation suffix")
        text += STEP3_NO_THINKING_PREFILL
        template_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        processor_ms = (time.perf_counter() - started) * 1000.0
        return inputs, template_ms, processor_ms

    def health(self) -> dict[str, Any]:
        return {
            **super().health(),
            "pacore": False,
            "multi_crop": True,
            "skip_private_reasoning": bool(
                getattr(self, "skip_private_reasoning", False)
            ),
            "fix_mistral_regex": bool(getattr(self, "fix_mistral_regex", True)),
            "runtime_transformers_version": str(getattr(self, "runtime_transformers_version", "")),
            "checkpoint_key_mapping": str(getattr(self, "checkpoint_key_mapping", "")),
            "checkpoint_load_clean": bool(getattr(self, "checkpoint_load_clean", False)),
            "checkpoint_loading_counts": dict(getattr(self, "checkpoint_loading_counts", {})),
            "parameter_count": int(getattr(self, "parameter_count", 0)),
            "parameter_bytes": int(getattr(self, "parameter_bytes", 0)),
            "parameter_dtype_counts": dict(getattr(self, "parameter_dtype_counts", {})),
        }
