from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from ..action_head import ActionHeadConfig, NavigationActionHead
from ..contracts import NavigationOutput, NavigationRequest
from ..vision_cache import EpisodeVisionCache, VisionCacheKey
from .base import BackboneAdapter


@dataclass
class CachedQwen3VisionFeatures:
    image_embedding: torch.Tensor
    deepstack_embeddings: tuple[torch.Tensor, ...]


@dataclass
class VisualInput:
    key: VisionCacheKey
    image: Image.Image
    label: str


class CosmosQwen3VLAdapter(BackboneAdapter):
    """Cosmos-Reason2/Qwen3-VL direct-action BF16 adapter.

    Qwen3-VL DeepStack features are cached together with the final vision
    embedding. No tokenizer IDs are hard-coded; the processor owns all image
    placeholders and multimodal RoPE layout.
    """

    model_variant = "nvidia-cosmos-reason2-8b-qwen3-vl"
    precision_mode = "bf16"

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        action_head: NavigationActionHead,
        model_fingerprint: str,
        model_variant: str = "nvidia-cosmos-reason2-8b-qwen3-vl",
        device: str = "cuda",
        history_frames: int = 4,
        cache_enabled: bool = True,
        arrive_threshold: float = 0.5,
        waypoint_scale: float = 1.0,
        action_head_trained: bool = False,
        allow_untrained_action_head: bool = False,
    ) -> None:
        if history_frames < 0:
            raise ValueError("history_frames must be non-negative")
        if not action_head_trained and not allow_untrained_action_head:
            raise RuntimeError("a trained Cosmos navigation action head is required")
        self.model = model
        self.processor = processor
        self.action_head = action_head
        self.model_variant = model_variant
        self.model_fingerprint = model_fingerprint
        self.device = device
        self.history_limit = history_frames
        self.arrive_threshold = float(arrive_threshold)
        self.waypoint_scale = float(waypoint_scale)
        self.action_head_trained = bool(action_head_trained)
        self.allow_untrained_action_head = bool(allow_untrained_action_head)
        self.cache = EpisodeVisionCache(capacity=max(3, history_frames + 3), enabled=cache_enabled)
        self._history: list[tuple[int, Image.Image]] = []
        self._episode_id = ""

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        action_head_path: str | Path | None = None,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
        history_frames: int = 4,
        cache_enabled: bool = True,
        allow_untrained_action_head: bool = False,
        seed: int = 20260713,
        model_variant: str = "nvidia-cosmos-reason2-8b-qwen3-vl",
    ) -> "CosmosQwen3VLAdapter":
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_path = Path(model_path)
        processor = AutoProcessor.from_pretrained(str(model_path), use_fast=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        )
        model.eval()
        hidden_size = int(model.config.text_config.hidden_size)
        torch.manual_seed(seed)
        action_head = NavigationActionHead(
            ActionHeadConfig(
                hidden_size=hidden_size,
                waypoint_count=5,
                attention_heads=4,
                action_former_layers=1,
                arrive_count=5,
                predict_heading=True,
                predict_confidence=True,
                normalize_heading=True,
            )
        ).to(device=device, dtype=torch.bfloat16)
        action_head_trained = False
        if action_head_path is not None:
            state = cls._load_action_head_state(Path(action_head_path), device="cpu")
            action_head.load_state_dict(state, strict=True)
            action_head_trained = True
        action_head.eval()
        fingerprint = cls._model_fingerprint(model_path)
        return cls(
            model=model,
            processor=processor,
            action_head=action_head,
            model_fingerprint=fingerprint,
            model_variant=model_variant,
            device=device,
            history_frames=history_frames,
            cache_enabled=cache_enabled,
            action_head_trained=action_head_trained,
            allow_untrained_action_head=allow_untrained_action_head,
        )

    @staticmethod
    def _load_action_head_state(path: Path, *, device: str) -> dict[str, torch.Tensor]:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(path), device=device)
        loaded = torch.load(path, map_location=device, weights_only=True)
        if isinstance(loaded, dict) and "state_dict" in loaded:
            loaded = loaded["state_dict"]
        if not isinstance(loaded, dict):
            raise TypeError("action head checkpoint must contain a state dict")
        return loaded

    @staticmethod
    def _model_fingerprint(model_path: Path) -> str:
        digest = hashlib.sha256()
        for filename in ("config.json", "preprocessor_config.json", "model.safetensors.index.json"):
            path = model_path / filename
            if path.exists():
                digest.update(filename.encode("utf-8"))
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def health(self) -> dict[str, Any]:
        return {
            **super().health(),
            "action_head_trained": self.action_head_trained,
            "cache_enabled": self.cache.enabled,
            "model_fingerprint": self.model_fingerprint,
        }

    def reset_episode(self, episode_id: str) -> None:
        self._episode_id = str(episode_id)
        self._history.clear()
        self.cache.begin_episode(self._episode_id)
        base = self.model.model
        if hasattr(base, "rope_deltas"):
            base.rope_deltas = None

    def infer(self, request: NavigationRequest) -> NavigationOutput:
        if request.episode_id != self._episode_id:
            raise RuntimeError("reset_episode must be called before inference for a new episode")
        images = self._visual_inputs(request)
        inputs = self._prepare_inputs(request.instruction, images)
        inputs = inputs.to(self.device)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        model_started = time.perf_counter()
        with torch.inference_mode():
            hidden_states, cache_hit, vision_latency_ms = self._backbone_hidden_states(inputs, images)
            head_output = self.action_head(hidden_states, inputs.get("attention_mask"))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        model_latency_ms = (time.perf_counter() - model_started) * 1000.0

        waypoints_tensor = head_output.waypoints[0].float().cpu() * self.waypoint_scale
        heading_tensor = head_output.heading_sin_cos[0].float().cpu()
        arrive_logits = head_output.arrive_logits[0].float().cpu()
        arrive_prob = torch.sigmoid(arrive_logits)
        stop = bool(torch.all(arrive_prob >= self.arrive_threshold).item())
        confidence = float(head_output.confidence[0].float().cpu().item())
        output = NavigationOutput(
            episode_id=request.episode_id,
            frame_id=request.frame_id,
            waypoints=tuple((float(x), float(y)) for x, y in waypoints_tensor.tolist()),
            heading_sin_cos=tuple((float(s), float(c)) for s, c in heading_tensor.tolist()),
            arrive_or_stop=stop,
            confidence=confidence,
            model_latency_ms=model_latency_ms,
            vision_latency_ms=vision_latency_ms,
            request_timestamp=request.timestamp,
            model_variant=self.model_variant,
            precision_mode=self.precision_mode,
            arrive_logits=tuple(float(value) for value in arrive_logits.tolist()),
            cache_hit=cache_hit,
            action_head_trained=self.action_head_trained,
            peak_memory_mib=(
                float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                if torch.cuda.is_available()
                else 0.0
            ),
        )
        if self.history_limit:
            front = next(item.image for item in images if item.label == "current front")
            self._history.append((request.frame_id, front.copy()))
            self._history = self._history[-self.history_limit :]
        return output

    def _visual_inputs(self, request: NavigationRequest) -> list[VisualInput]:
        current = self._decode_images(request)
        result: list[VisualInput] = []
        for frame_id, image in self._history:
            result.append(self._visual_input(request.episode_id, frame_id, "front", image, "history front"))
        result.extend(
            self._visual_input(request.episode_id, request.frame_id, view, current[view], f"current {view}")
            for view in ("left", "front", "right")
            if view in current
        )
        return result

    def _visual_input(
        self,
        episode_id: str,
        frame_id: int,
        view: str,
        image: Image.Image,
        label: str,
    ) -> VisualInput:
        width, height = image.size
        return VisualInput(
            key=VisionCacheKey(episode_id, frame_id, view, width, height, self.model_fingerprint),
            image=image,
            label=label,
        )

    def _prepare_inputs(self, instruction: str, images: Sequence[VisualInput]) -> Any:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are the visual reasoning backbone of a navigation robot. "
                    "Use the observations in chronological order and ground the instruction in visible free space, "
                    "obstacles, rooms, doors, and named objects.\n"
                ),
            }
        ]
        for item in images:
            content.append({"type": "text", "text": f"\n{item.label}:\n"})
            content.append({"type": "image", "image": item.image})
        content.append(
            {
                "type": "text",
                "text": (
                    f"\nNavigation instruction: {instruction}\n"
                    "Prepare a compact hidden representation for five safe planar waypoints, heading, stop, and confidence."
                ),
            }
        )
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )

    def _backbone_hidden_states(
        self, inputs: Any, images: Sequence[VisualInput]
    ) -> tuple[torch.Tensor, bool, float]:
        if not self.cache.enabled:
            visual_latency_ms = 0.0

            def before_visual(_module, _args):
                nonlocal visual_started
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                visual_started = time.perf_counter()

            def after_visual(_module, _args, _output):
                nonlocal visual_latency_ms
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                visual_latency_ms += (time.perf_counter() - visual_started) * 1000.0

            visual_started = time.perf_counter()
            pre_hook = self.model.model.visual.register_forward_pre_hook(before_visual)
            post_hook = self.model.model.visual.register_forward_hook(after_visual)
            try:
                outputs = self.model.model(**inputs, use_cache=False, return_dict=True)
            finally:
                pre_hook.remove()
                post_hook.remove()
            return outputs.last_hidden_state, False, visual_latency_ms
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        image_grid_thw = inputs["image_grid_thw"]
        pixel_values = inputs["pixel_values"]
        if len(images) != image_grid_thw.shape[0]:
            raise RuntimeError("processor image order does not match visual input manifest")
        patch_counts = image_grid_thw.prod(dim=-1).tolist()
        pixel_slices = torch.split(pixel_values, [int(value) for value in patch_counts], dim=0)
        encoded: list[CachedQwen3VisionFeatures | None] = []
        cache_hits = 0
        base = self.model.model
        visual_latency_ms = 0.0
        missing_indexes: list[int] = []
        for index, item in enumerate(images):
            feature = self.cache.get(item.key)
            if feature is None:
                missing_indexes.append(index)
            else:
                cache_hits += 1
            encoded.append(feature)
        if missing_indexes:
            missing_pixels = torch.cat([pixel_slices[index] for index in missing_indexes], dim=0)
            missing_grids = torch.stack([image_grid_thw[index] for index in missing_indexes], dim=0)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            visual_started = time.perf_counter()
            new_image_embeddings, new_deepstack_embeddings = base.get_image_features(
                missing_pixels,
                missing_grids,
            )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            visual_latency_ms += (time.perf_counter() - visual_started) * 1000.0
            split_sizes = (
                missing_grids.prod(dim=-1) // int(base.visual.spatial_merge_size) ** 2
            ).tolist()
            deepstack_splits = [
                torch.split(level_embeddings, [int(value) for value in split_sizes], dim=0)
                for level_embeddings in new_deepstack_embeddings
            ]
            for new_index, image_index in enumerate(missing_indexes):
                feature = CachedQwen3VisionFeatures(
                    image_embedding=new_image_embeddings[new_index].detach(),
                    deepstack_embeddings=tuple(level[new_index].detach() for level in deepstack_splits),
                )
                encoded[image_index] = feature
                self.cache.put(images[image_index].key, feature)
        if any(item is None for item in encoded):
            raise RuntimeError("failed to assemble Qwen3-VL visual features")
        complete = [item for item in encoded if item is not None]
        image_embeddings = torch.cat([item.image_embedding for item in complete], dim=0)
        deepstack_count = len(complete[0].deepstack_embeddings)
        if any(len(item.deepstack_embeddings) != deepstack_count for item in complete):
            raise RuntimeError("inconsistent Qwen3-VL DeepStack cache entry")
        deepstack_visual = [
            torch.cat([item.deepstack_embeddings[level] for item in complete], dim=0)
            for level in range(deepstack_count)
        ]
        inputs_embeds = base.get_input_embeddings()(input_ids)
        image_embeddings = image_embeddings.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = base.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeddings,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeddings)
        visual_pos_masks = image_mask[..., 0]
        position_ids, rope_deltas = base.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            attention_mask=attention_mask,
        )
        base.rope_deltas = rope_deltas
        language_output = base.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual,
            use_cache=False,
            return_dict=True,
        )
        return language_output.last_hidden_state, cache_hits > 0, visual_latency_ms

    @staticmethod
    def _decode_images(request: NavigationRequest) -> dict[str, Image.Image]:
        def decode(value: bytes) -> Image.Image:
            image = Image.open(BytesIO(value))
            image.load()
            return image.convert("RGB")

        front = decode(request.rgb_front)
        images = {"front": front}
        if request.rgb_left is not None:
            images["left"] = decode(request.rgb_left)
        if request.rgb_right is not None:
            images["right"] = decode(request.rgb_right)
        return images
