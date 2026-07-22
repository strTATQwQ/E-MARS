from __future__ import annotations

import threading
import time
from typing import Any, Iterable

from slow_planner.step3_vl_10b import Step3VLSlowPlanner


CHAT_SUFFIX = "<|im_start|>assistant\n<think>\n"


class Step3SplitModel:
    """One resident Step3 checkpoint shared by navigation and arrival branches.

    The model is never given a STOP logit bias. Both branches first generate a
    bounded private reasoning prefix, then score an explicit finite output
    schema. Navigation scores only valid candidate ordinals. Arrival exposes an
    unmodified true-vs-false logit margin for calibration.
    """

    def __init__(self, planner: Step3VLSlowPlanner, *, reasoning_tokens: int) -> None:
        self.planner = planner
        self.reasoning_tokens = int(reasoning_tokens)
        if not 16 <= self.reasoning_tokens <= 32:
            raise ValueError("reasoning_tokens must be between 16 and 32")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Step3SplitModel":
        import os

        model_cfg = config["model"]
        model_path = os.path.expanduser(os.path.expandvars(str(model_cfg["model_path"])))
        planner = Step3VLSlowPlanner.from_pretrained(
            model_path,
            device=str(model_cfg.get("device", "cuda")),
            max_new_tokens=int(model_cfg["max_new_tokens"]),
            fix_mistral_regex=bool(model_cfg.get("fix_mistral_regex", True)),
            processor_use_fast=bool(model_cfg.get("processor_use_fast", False)),
            expected_transformers_version=str(model_cfg["transformers_version"]),
            checkpoint_key_mapping=str(model_cfg["checkpoint_key_mapping"]),
            require_clean_checkpoint_load=bool(model_cfg.get("require_clean_checkpoint_load", True)),
            require_all_parameters_bf16=bool(model_cfg.get("require_all_parameters_bf16", True)),
        )
        return cls(planner, reasoning_tokens=int(model_cfg["reasoning_tokens"]))

    @property
    def processor(self) -> Any:
        return self.planner.processor

    @property
    def tokenizer(self) -> Any:
        return self.planner.tokenizer

    def prepare_images(self, jpeg_payloads: Iterable[bytes]) -> tuple[list[Any], float]:
        from io import BytesIO
        from PIL import Image

        started = time.perf_counter()
        images = []
        for payload in jpeg_payloads:
            image = Image.open(BytesIO(payload))
            image.load()
            images.append(image.convert("RGB"))
        return images, (time.perf_counter() - started) * 1000.0

    def score_branches(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[Any],
        assistant_prefix: str,
        branch_texts: dict[str, str],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        import torch

        total_started = time.perf_counter()
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": user_prompt})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        template_started = time.perf_counter()
        base_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not base_text.endswith(CHAT_SUFFIX):
            raise RuntimeError("unexpected Step3 chat-template generation suffix")
        template_ms = (time.perf_counter() - template_started) * 1000.0

        reasoning_text, reasoning_metrics = _bounded_reasoning_generate(
            model=self.planner.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            text=base_text,
            images=images,
            device=self.planner.device,
            max_new_tokens=self.reasoning_tokens,
        )
        reasoning_text = _clean_reasoning(reasoning_text)
        final_text = base_text + reasoning_text + "\n</think>\n" + assistant_prefix
        classify_preprocess_started = time.perf_counter()
        inputs = self.processor(text=[final_text], images=images, padding=True, return_tensors="pt")
        classify_preprocess_ms = (time.perf_counter() - classify_preprocess_started) * 1000.0

        token_ids: dict[str, int] = {}
        for name, branch in branch_texts.items():
            encoded = self.tokenizer.encode(branch, add_special_tokens=False)
            if len(encoded) != 1:
                raise RuntimeError(f"branch {branch!r} must encode as one Step3 token")
            token_ids[name] = int(encoded[0])

        moved = inputs.to(self.planner.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        classify_started = time.perf_counter()
        with torch.inference_mode():
            output = self.planner.model(**dict(moved), use_cache=False, return_dict=True)
            logits = output.logits[0, -1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        classify_ms = (time.perf_counter() - classify_started) * 1000.0
        scores = {name: float(logits[token_id].item()) for name, token_id in token_ids.items()}
        return scores, {
            **reasoning_metrics,
            "reasoning_text": reasoning_text,
            "reasoning_token_limit": self.reasoning_tokens,
            "prompt_template_ms": template_ms,
            "classification_preprocess_ms": classify_preprocess_ms,
            "classification_forward_ms": classify_ms,
            "branch_token_ids": token_ids,
            "model_total_ms": (time.perf_counter() - total_started) * 1000.0,
            "model_variant": self.planner.model_variant,
            "precision_mode": self.planner.precision_mode,
            "assistant_prefix": assistant_prefix,
        }

    def health(self) -> dict[str, Any]:
        return {
            **self.planner.health(),
            "reasoning_tokens": self.reasoning_tokens,
            "split_policy": True,
            "stop_logit_bias": False,
            "pacore": False,
        }

    def close(self) -> None:
        self.planner.close()


def _clean_reasoning(text: str) -> str:
    value = text
    for marker in ("</think>", "<|im_end|>", "```", "{\"action\"", "{\"arrived\""):
        value = value.split(marker, 1)[0]
    return value.strip()


def _bounded_reasoning_generate(
    *,
    model: Any,
    tokenizer: Any,
    processor: Any,
    text: str,
    images: list[Any],
    device: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, float | int]]:
    import torch
    from transformers import TextIteratorStreamer

    preprocess_started = time.perf_counter()
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
    moved = inputs.to(device)
    state: dict[str, Any] = {}

    class TokenTimingStreamer(TextIteratorStreamer):
        def put(self, value: Any) -> None:
            was_prompt = bool(getattr(self, "next_tokens_are_prompt", False))
            super().put(value)
            if not (was_prompt and self.skip_prompt) and "first_token_at" not in state:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                state["first_token_at"] = time.perf_counter()

    streamer = TokenTimingStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=3600.0
    )
    kwargs = dict(moved)
    kwargs.update(
        streamer=streamer,
        do_sample=False,
        num_beams=1,
        max_new_tokens=int(max_new_tokens),
        return_dict_in_generate=True,
        use_cache=True,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()

    def worker() -> None:
        try:
            state["output"] = model.generate(**kwargs)
        except BaseException as exc:
            state["error"] = exc
            streamer.end()

    thread = threading.Thread(target=worker, name="step3-bounded-reasoning", daemon=True)
    thread.start()
    try:
        for _ in streamer:
            pass
    finally:
        thread.join()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    finished = time.perf_counter()
    if "error" in state:
        raise state["error"]
    output = state["output"]
    sequences = output.sequences if hasattr(output, "sequences") else output
    input_tokens = int(moved["input_ids"].shape[-1])
    generated_ids = sequences[0, input_tokens:]
    output_tokens = int(generated_ids.numel())
    generated_text = tokenizer.decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    first_token_at = float(state.get("first_token_at", finished))
    decode_s = max(0.0, finished - first_token_at)
    decode_count = max(0, output_tokens - 1)
    return generated_text, {
        "input_token_count": input_tokens,
        "reasoning_output_tokens": output_tokens,
        "reasoning_ttft_ms": (first_token_at - started) * 1000.0,
        "reasoning_decode_ms": decode_s * 1000.0,
        "reasoning_tokens_per_s": decode_count / decode_s if decode_s > 0 and decode_count else 0.0,
        "reasoning_generate_ms": (finished - started) * 1000.0,
        "reasoning_preprocess_ms": preprocess_ms,
        "peak_memory_mib": (
            float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            if torch.cuda.is_available()
            else 0.0
        ),
    }

