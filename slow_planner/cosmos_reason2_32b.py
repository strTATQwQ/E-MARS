from __future__ import annotations

from pathlib import Path
from typing import Any

from .hf_base import QwenVisionLanguagePlanner


class CosmosReason2SlowPlanner(QwenVisionLanguagePlanner):
    model_variant = "cosmos_reason2_32b_bf16"
    precision_mode = "bf16"

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        attn_implementation: str = "sdpa",
        device: str = "cuda",
        max_new_tokens: int = 48,
        processor_use_fast: bool = False,
    ) -> "CosmosReason2SlowPlanner":
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(str(model_path), use_fast=processor_use_fast)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_implementation,
        )
        model.eval()
        return cls(
            model=model,
            processor=processor,
            device=device,
            max_new_tokens=max_new_tokens,
            processor_use_fast=processor_use_fast,
        )

    def health(self) -> dict[str, Any]:
        return {**super().health(), "reasoning_budget": "disabled_by_compact_json_prompt"}
