from __future__ import annotations

import threading
import time
from abc import abstractmethod
from dataclasses import replace
from io import BytesIO
from typing import Any, Callable, Sequence

from .base import (
    PlannerMetrics,
    SlowPlanner,
    SlowPlannerRequest,
    SlowPlannerProtocolError,
    strict_decision_json_object,
)
from .prompt import SYSTEM_PROMPT, build_prompt


def _sync_cuda(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _input_token_count(inputs: Any) -> int:
    value = inputs.get("input_ids")
    return int(value.shape[-1]) if value is not None else 0


def _visual_token_count(inputs: Any, image_token_id: int | None) -> int:
    if image_token_id is not None and inputs.get("input_ids") is not None:
        return int((inputs["input_ids"] == int(image_token_id)).sum().item())
    grid = inputs.get("image_grid_thw")
    if grid is not None:
        return int(grid.prod(dim=-1).sum().item())
    return 0


def _contains_complete_decision_json(text: str) -> bool:
    """Return true only for one complete, prefix-free exact decision object."""

    try:
        strict_decision_json_object(text)
    except SlowPlannerProtocolError:
        return False
    return True


class GenerationThreadTimeout(RuntimeError):
    """Fatal: model generation outlived its hard wall budget and join grace."""


def instrumented_generate(
    *,
    model: Any,
    tokenizer: Any,
    inputs: Any,
    device: str,
    max_new_tokens: int,
    stop_on_complete_json: bool = False,
    generation_wall_budget_s: float | None = None,
    generation_join_grace_s: float = 0.5,
    complete_json_predicate: Any = _contains_complete_decision_json,
) -> tuple[str, dict[str, float | int]]:
    """Run deterministic generation while timing first streamed text and decode."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

    if max_new_tokens <= 0 or max_new_tokens > 96:
        raise ValueError("max_new_tokens must be in [1, 96]")
    if generation_wall_budget_s is not None and generation_wall_budget_s <= 0:
        raise ValueError("generation_wall_budget_s must be positive")
    if generation_join_grace_s < 0 or generation_join_grace_s > 2.0:
        raise ValueError("generation_join_grace_s must be in [0, 2]")
    moved = inputs.to(device)
    input_tokens = _input_token_count(moved)
    state: dict[str, Any] = {}

    class TokenTimingStreamer(TextIteratorStreamer):
        def put(self, value: Any) -> None:
            was_prompt = bool(getattr(self, "next_tokens_are_prompt", False))
            super().put(value)
            if not (was_prompt and self.skip_prompt) and "first_token_at" not in state:
                _sync_cuda(torch)
                state["first_token_at"] = time.perf_counter()

    class CompleteDecisionJson(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            del scores, kwargs
            complete = []
            for row in input_ids:
                generated = tokenizer.decode(
                    row[input_tokens:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                complete.append(bool(complete_json_predicate(generated)))
            return torch.tensor(complete, dtype=torch.bool, device=input_ids.device)

    streamer = TokenTimingStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=1.0,
    )
    kwargs = dict(moved)
    kwargs.update(
        streamer=streamer,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        use_cache=True,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync_cuda(torch)
    started = time.perf_counter()

    class GenerationWallBudget(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            del scores, kwargs
            expired = time.perf_counter() - started >= float(generation_wall_budget_s)
            return torch.full(
                (input_ids.shape[0],),
                expired,
                dtype=torch.bool,
                device=input_ids.device,
            )

    criteria = []
    if stop_on_complete_json:
        criteria.append(CompleteDecisionJson())
    if generation_wall_budget_s is not None:
        criteria.append(GenerationWallBudget())
    if criteria:
        kwargs["stopping_criteria"] = StoppingCriteriaList(criteria)

    def worker() -> None:
        try:
            state["output"] = model.generate(**kwargs)
        except BaseException as exc:  # propagated after streamer/worker completion
            state["error"] = exc
            streamer.end()

    thread = threading.Thread(target=worker, name="slow-planner-generate", daemon=True)
    thread.start()
    join_timeout = (
        None
        if generation_wall_budget_s is None
        else generation_wall_budget_s + generation_join_grace_s
    )
    thread.join(timeout=join_timeout)
    if thread.is_alive():
        raise GenerationThreadTimeout(
            "generation thread exceeded wall budget plus join grace; service restart required"
        )
    _sync_cuda(torch)
    finished = time.perf_counter()
    if "error" in state:
        raise state["error"]
    output = state["output"]
    sequences = output.sequences if hasattr(output, "sequences") else output
    generated_ids = sequences[0, input_tokens:]
    output_tokens = int(generated_ids.numel())
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    first_token_at = float(state.get("first_token_at", finished))
    decode_s = max(0.0, finished - first_token_at)
    decode_token_count = max(0, output_tokens - 1)
    return raw_text, {
        "input_token_count": input_tokens,
        "output_token_count": output_tokens,
        "prefill_ttft_ms": (first_token_at - started) * 1000.0,
        "decode_ms": decode_s * 1000.0,
        "decode_tokens_per_s": decode_token_count / decode_s if decode_s > 0 and decode_token_count else 0.0,
        "model_generate_ms": (finished - started) * 1000.0,
        "peak_memory_mib": (
            float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            if torch.cuda.is_available()
            else 0.0
        ),
    }


class HuggingFaceSlowPlanner(SlowPlanner):
    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        device: str = "cuda",
        max_new_tokens: int = 48,
        processor_use_fast: bool = False,
        stop_on_complete_json: bool = False,
        generation_wall_budget_s: float | None = None,
        generation_join_grace_s: float = 0.5,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.stop_on_complete_json = bool(stop_on_complete_json)
        self.generation_wall_budget_s = generation_wall_budget_s
        self.generation_join_grace_s = float(generation_join_grace_s)
        image_processor = getattr(processor, "image_processor", None)
        self.processor_use_fast = bool(
            getattr(image_processor, "is_fast", False)
            or type(image_processor).__name__.lower().endswith("fast")
        )
        if self.processor_use_fast != bool(processor_use_fast):
            raise RuntimeError(
                f"processor backend mismatch: expected use_fast={processor_use_fast}, "
                f"actual image processor={type(image_processor).__name__}"
            )

    @property
    def tokenizer(self) -> Any:
        return self.processor.tokenizer

    @property
    def image_token_id(self) -> int | None:
        value = getattr(getattr(self.model, "config", None), "image_token_id", None)
        return int(value) if value is not None else None

    def _decode_images(self, request: SlowPlannerRequest) -> tuple[list[Any], float]:
        from PIL import Image

        started = time.perf_counter()
        images = []
        for item in request.ordered_images:
            image = Image.open(BytesIO(item.jpeg))
            image.load()
            images.append(image.convert("RGB"))
        return images, (time.perf_counter() - started) * 1000.0

    def format_prompt(self, request: SlowPlannerRequest, *, correction: str = "") -> str:
        return build_prompt(request, correction=correction)

    def contains_complete_json(self, raw_text: str) -> bool:
        return _contains_complete_decision_json(raw_text)

    @abstractmethod
    def prepare_inputs(self, *, images: list[Any], prompt: str) -> tuple[Any, float, float]:
        """Return model inputs, template milliseconds, and processor milliseconds."""
        raise NotImplementedError

    def generate_raw(
        self,
        request: SlowPlannerRequest,
        *,
        correction: str = "",
    ) -> tuple[str, PlannerMetrics]:
        images, decode_ms = self._decode_images(request)
        prompt = self.format_prompt(request, correction=correction)
        return self.generate_prompt_raw(
            prompt,
            images=images,
            raw_image_resolutions=tuple(
                (item.width, item.height) for item in request.ordered_images
            ),
            image_decode_ms=decode_ms,
            complete_json_predicate=self.contains_complete_json,
        )

    def generate_prompt_raw(
        self,
        prompt: str,
        *,
        images: Sequence[Any] = (),
        raw_image_resolutions: Sequence[tuple[int, int]] = (),
        image_decode_ms: float = 0.0,
        complete_json_predicate: Callable[[str], bool] | None = None,
    ) -> tuple[str, PlannerMetrics]:
        """Generate one bounded JSON response from a caller-owned prompt.

        This is used by the one-time Step3 mission normalizer.  It deliberately
        accepts no raw navigation authority and can run text-only; the normal
        frontier planner continues to call :meth:`generate_raw` with images.
        """

        prepared_images = list(images)
        resolutions = tuple((int(width), int(height)) for width, height in raw_image_resolutions)
        if len(resolutions) != len(prepared_images):
            raise ValueError("image resolutions must match prepared images")
        predicate = complete_json_predicate or self.contains_complete_json
        inputs, template_ms, processor_ms = self.prepare_inputs(
            images=prepared_images, prompt=prompt
        )
        raw_text, generated = instrumented_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            inputs=inputs,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            stop_on_complete_json=self.stop_on_complete_json,
            generation_wall_budget_s=self.generation_wall_budget_s,
            generation_join_grace_s=self.generation_join_grace_s,
            complete_json_predicate=predicate,
        )
        preprocessing_ms = float(image_decode_ms) + template_ms + processor_ms
        metrics = PlannerMetrics(
            image_count=len(prepared_images),
            raw_image_resolutions=resolutions,
            visual_token_count=_visual_token_count(inputs, self.image_token_id),
            input_token_count=int(generated["input_token_count"]),
            output_token_count=int(generated["output_token_count"]),
            image_decode_ms=float(image_decode_ms),
            prompt_template_ms=template_ms,
            processor_ms=processor_ms,
            preprocessing_ms=preprocessing_ms,
            prefill_ttft_ms=float(generated["prefill_ttft_ms"]),
            decode_ms=float(generated["decode_ms"]),
            decode_tokens_per_s=float(generated["decode_tokens_per_s"]),
            model_generate_ms=float(generated["model_generate_ms"]),
            end_to_end_ms=preprocessing_ms + float(generated["model_generate_ms"]),
            peak_memory_mib=float(generated["peak_memory_mib"]),
            model_variant=self.model_variant,
            precision_mode=self.precision_mode,
        )
        return raw_text, metrics

    def health(self) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        return {
            **super().health(),
            "model_type": getattr(config, "model_type", ""),
            "max_new_tokens": self.max_new_tokens,
            "deterministic_decoding": True,
            "stop_on_complete_json": self.stop_on_complete_json,
            "generation_wall_budget_s": self.generation_wall_budget_s,
            "generation_join_grace_s": self.generation_join_grace_s,
            "processor_use_fast": self.processor_use_fast,
            "processor_class": type(self.processor).__name__,
            "image_processor_class": type(getattr(self.processor, "image_processor", None)).__name__,
        }


class QwenVisionLanguagePlanner(HuggingFaceSlowPlanner):
    def prepare_inputs(self, *, images: list[Any], prompt: str) -> tuple[Any, float, float]:
        from qwen_vl_utils import process_vision_info

        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        started = time.perf_counter()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        template_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        processor_ms = (time.perf_counter() - started) * 1000.0
        return inputs, template_ms, processor_ms


class StepVisionLanguagePlanner(HuggingFaceSlowPlanner):
    def prepare_inputs(self, *, images: list[Any], prompt: str) -> tuple[Any, float, float]:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        started = time.perf_counter()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        template_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        processor_ms = (time.perf_counter() - started) * 1000.0
        return inputs, template_ms, processor_ms
