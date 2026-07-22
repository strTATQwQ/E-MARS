from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .hf_base import QwenVisionLanguagePlanner


class Qwen25BaselineSlowPlanner(QwenVisionLanguagePlanner):
    """Independent slow `generate()` instance using the released OmniNav checkpoint."""

    model_variant = "qwen25_baseline_bf16"
    precision_mode = "bf16"

    @classmethod
    def from_pretrained(
        cls,
        *,
        repo: str | Path,
        model_path: str | Path,
        helper_path: str | Path,
        attn_implementation: str = "flash_attention_2",
        device: str = "cuda",
        max_new_tokens: int = 48,
        processor_use_fast: bool = False,
    ) -> "Qwen25BaselineSlowPlanner":
        helper_path = Path(helper_path)
        spec = importlib.util.spec_from_file_location("slow_qwen25_helper", helper_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import OmniNav helper: {helper_path}")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        helper.add_repo_paths(Path(repo))
        _, processor, model = helper.load_model(str(model_path), attn_implementation)
        model.eval()
        return cls(
            model=model,
            processor=processor,
            device=device,
            max_new_tokens=max_new_tokens,
            processor_use_fast=processor_use_fast,
        )

    def health(self) -> dict[str, Any]:
        return {
            **super().health(),
            "independent_slow_instance": True,
            "contains_omninav_action_head_but_never_calls_action_former": True,
        }
