from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path
from typing import Any

from .base import BackboneAdapter
from ..contracts import NavigationOutput, NavigationRequest


class Qwen25LegacyAdapter(BackboneAdapter):
    """Compatibility wrapper around the released custom Qwen2.5-VL forward."""

    model_variant = "qwen2.5-vl-omninav-legacy"
    precision_mode = "bf16"

    def health(self) -> dict[str, Any]:
        return {
            **super().health(),
            "action_head_trained": True,
        }

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        prepare_inputs: Any,
        helper_config: Any,
        helper_module: Any | None = None,
        device: str = "cuda",
        arrive_threshold: float = 0.5,
        predict_scale: float = 0.3,
        legacy_waypoint_forward_axis: str = "raw",
        input_contract: str = "helper_config",
    ) -> None:
        self.model = model
        self.processor = processor
        self.prepare_inputs = prepare_inputs
        self.helper_config = helper_config
        self.helper_module = helper_module
        self.device = device
        self.arrive_threshold = float(arrive_threshold)
        self.predict_scale = float(predict_scale)
        if legacy_waypoint_forward_axis not in {"raw", "x", "y"}:
            raise ValueError("legacy_waypoint_forward_axis must be raw, x, or y")
        self.legacy_waypoint_forward_axis = legacy_waypoint_forward_axis
        if input_contract not in {"helper_config", "slowfast_original_7view"}:
            raise ValueError("unsupported Qwen2.5 input contract")
        self.input_contract = input_contract
        self._history_frames: list[Any] = []
        self._history_poses: list[tuple[float, ...]] = []
        self._episode_id = ""
        self._step = 0

    @classmethod
    def from_helper(
        cls,
        *,
        repo: str | Path,
        model_path: str | Path,
        helper_path: str | Path,
        config_name: str = "A_baseline_3view_hist5",
        attn_implementation: str = "flash_attention_2",
        device: str = "cuda",
        predict_scale: float = 0.3,
        legacy_waypoint_forward_axis: str = "raw",
        input_contract: str = "helper_config",
    ) -> "Qwen25LegacyAdapter":
        helper_path = Path(helper_path)
        spec = importlib.util.spec_from_file_location("omninav_qwen25_helper", helper_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import OmniNav helper: {helper_path}")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        helper.add_repo_paths(Path(repo))
        _, processor, model = helper.load_model(str(model_path), attn_implementation)
        configs = {config.name: config for config in helper.CONFIGS}
        if config_name not in configs:
            raise KeyError(f"unknown helper config {config_name!r}")
        return cls(
            model=model,
            processor=processor,
            prepare_inputs=helper.prepare_inputs,
            helper_config=configs[config_name],
            helper_module=helper,
            device=device,
            predict_scale=predict_scale,
            legacy_waypoint_forward_axis=legacy_waypoint_forward_axis,
            input_contract=input_contract,
        )

    def reset_episode(self, episode_id: str) -> None:
        self._history_frames.clear()
        self._history_poses.clear()
        self._episode_id = str(episode_id)
        self._step = 0
        if hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = None

    def infer_prepared(self, request: NavigationRequest, inputs: Any, *, vision_latency_ms: float = 0.0) -> NavigationOutput:
        import torch

        sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
        inputs = inputs.to(self.device)
        sync()
        started = time.perf_counter()
        with torch.inference_mode():
            waypoints, arrive_logits, sin_angle, cos_angle = self.model.forward(
                **inputs,
                action_former=True,
                gt_waypoints=0,
                train=False,
                train_branch=["continue"],
            )
        sync()
        latency_ms = (time.perf_counter() - started) * 1000.0
        waypoints = waypoints.detach().float().cpu() * self.predict_scale
        arrive_logits = arrive_logits.detach().float().cpu()
        sin_angle = sin_angle.detach().float().cpu()
        cos_angle = cos_angle.detach().float().cpu()
        probabilities = torch.sigmoid(arrive_logits).reshape(-1)
        stop = bool(torch.all(probabilities >= self.arrive_threshold).item())
        confidence = float(torch.maximum(probabilities, 1.0 - probabilities).mean().item())
        raw_points = [(float(x), float(y)) for x, y in waypoints.reshape(-1, 2).tolist()]
        if self.legacy_waypoint_forward_axis == "y":
            points = tuple((raw_y, raw_x) for raw_x, raw_y in raw_points)
        else:
            points = tuple(raw_points)
        headings = tuple(
            (float(sin_value), float(cos_value))
            for sin_value, cos_value in zip(sin_angle.reshape(-1).tolist(), cos_angle.reshape(-1).tolist())
        )
        return NavigationOutput(
            episode_id=request.episode_id,
            frame_id=request.frame_id,
            waypoints=points,
            heading_sin_cos=headings,
            arrive_or_stop=stop,
            confidence=confidence,
            model_latency_ms=latency_ms,
            vision_latency_ms=vision_latency_ms,
            request_timestamp=request.timestamp,
            model_variant=self.model_variant,
            precision_mode=self.precision_mode,
            arrive_logits=tuple(float(value) for value in arrive_logits.reshape(-1).tolist()),
            action_head_trained=True,
            peak_memory_mib=(
                float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                if torch.cuda.is_available()
                else 0.0
            ),
        )

    def infer(self, request: NavigationRequest) -> NavigationOutput:
        if request.episode_id != self._episode_id:
            raise RuntimeError("reset_episode must be called before inference for a new episode")
        decode_started = time.perf_counter()
        frames = self._decode_images(request)
        decode_latency_ms = (time.perf_counter() - decode_started) * 1000.0
        if self.input_contract == "slowfast_original_7view":
            inputs, prep_timing = self._prepare_slowfast_original(request, frames)
        else:
            original_make_rgb = None
            if self.helper_module is not None:
                original_make_rgb = self.helper_module.make_rgb
                self.helper_module.make_rgb = lambda width, height, step, view: frames.get(view, frames["front"])
            try:
                inputs, prep_timing, _, _ = self.prepare_inputs(
                    self.processor,
                    self.helper_config,
                    self._step,
                    list(self._history_frames),
                    request.instruction,
                )
            finally:
                if self.helper_module is not None and original_make_rgb is not None:
                    self.helper_module.make_rgb = original_make_rgb
        vision_latency_ms = decode_latency_ms + 1000.0 * sum(float(value) for value in prep_timing.values())
        history_limit = 12 if self.input_contract == "slowfast_original_7view" else int(
            getattr(self.helper_config, "history_images", 0)
        )
        if history_limit > 0:
            self._history_frames.append(frames["front"])
            self._history_frames = self._history_frames[-history_limit:]
            self._history_poses.append(tuple(request.agent_pose))
            self._history_poses = self._history_poses[-history_limit:]
        else:
            self._history_frames.clear()
            self._history_poses.clear()
        self._step += 1
        return self.infer_prepared(request, inputs, vision_latency_ms=vision_latency_ms)

    def _prepare_slowfast_original(self, request: NavigationRequest, frames: dict[str, Any]):
        """Reproduce upstream slow-fast prompt, seven images and input_waypoints."""
        import numpy as np
        import torch

        if self.helper_module is None:
            raise RuntimeError("slowfast input contract requires the frozen OmniNav helper")
        target_world = request.last_action.get("selected_target_world")
        origin_pose = request.last_action.get("subgoal_origin_pose")
        if not isinstance(target_world, (list, tuple)) or len(target_world) < 3:
            raise ValueError("slowfast input contract requires selected_target_world")
        if not isinstance(origin_pose, (list, tuple)) or len(origin_pose) < 4:
            raise ValueError("slowfast input contract requires subgoal_origin_pose")
        current_pose = tuple(request.agent_pose)
        if len(current_pose) < 4:
            raise ValueError("slowfast input contract requires xyz+yaw agent_pose")

        history = list(zip(self._history_frames, self._history_poses))
        if not history:
            history = [(frames["front"], current_pose)]
        history = history[-12:]
        indices = np.linspace(0, len(history) - 1, 4, dtype=int).tolist()
        selected = [history[index] for index in indices]
        images = [row[0] for row in selected] + [frames["left"], frames["front"], frames["right"]]
        # Upstream process_vision() first normalizes every prompt image to
        # 486x420 before its history/current-view preprocess routine.
        images = [image.resize((486, 420)) for image in images]
        positions = [row[1] for row in selected] + [current_pose, tuple(target_world)]

        origin_x, origin_y, origin_z, origin_yaw = (float(value) for value in origin_pose[:4])
        local_positions = []
        for pose in positions:
            dx = float(pose[0]) - origin_x
            dy = float(pose[1]) - origin_y
            forward = np.cos(origin_yaw) * dx + np.sin(origin_yaw) * dy
            left = -np.sin(origin_yaw) * dx + np.cos(origin_yaw) * dy
            local_positions.append([-left, forward])  # upstream [local x/right, local z/forward]

        content_text = (
            "The following are observation images from the past 4 frames:<image>,<image>,<image>,<image>\n"
            "The current tri-view is shown below: leftside:<image>,frontside:<image>,rightside:<image>\n"
            "Position coordinates for the past 4 frames:<input_pos1><input_pos2><input_pos3><input_pos4>\n"
            "The current observation represents the coordinate: <input_pos5>\n"
            "Target position coordinate: <input_target>\n"
            "Please predict the position coordinates for the next 5 frames based on the above information.<|NAV|>\n"
            "Output the waypoint"
        )
        messages = [{"role": "user", "content": [{"type": "image", "image": images}, {"type": "text", "text": content_text}]}]
        started = time.perf_counter()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
        text = text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        template_s = time.perf_counter() - started
        started = time.perf_counter()
        image_inputs = self._slowfast_preprocess(images)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs["input_waypoints"] = (torch.tensor(local_positions, dtype=torch.float32) / self.predict_scale)[None]
        processor_s = time.perf_counter() - started
        return inputs, {"template_s": template_s, "processor_s": processor_s}

    @staticmethod
    def _slowfast_preprocess(images: list[Any]) -> list[Any]:
        """Exact deterministic image geometry from upstream qwen_utils.preprocess."""
        import math

        result = []
        final_count = len(images)
        for index, image in enumerate(images):
            width, height = image.size
            if width >= height:
                resized = image.resize((640, int(height * (640 / width))))
            else:
                resized = image.resize((int(width * (640 / height)), 640))
            if index not in {final_count - 1, final_count - 3}:
                resized = resized.resize((max(1, int(resized.width / 4)), max(1, int(resized.height / 4))))
            factor = 28
            rounded_h = max(factor, round(resized.height / factor) * factor)
            rounded_w = max(factor, round(resized.width / factor) * factor)
            pixels = rounded_h * rounded_w
            if pixels > 12_845_056:
                beta = math.sqrt((resized.height * resized.width) / 12_845_056)
                rounded_h = max(factor, math.floor(resized.height / beta / factor) * factor)
                rounded_w = max(factor, math.floor(resized.width / beta / factor) * factor)
            elif pixels < 3_136:
                beta = math.sqrt(3_136 / (resized.height * resized.width))
                rounded_h = max(factor, math.ceil(resized.height * beta / factor) * factor)
                rounded_w = max(factor, math.ceil(resized.width * beta / factor) * factor)
            result.append(resized.resize((rounded_w, rounded_h)))
        return result

    def health(self) -> dict[str, Any]:
        return {**super().health(), "action_head_trained": True, "input_contract": self.input_contract}

    @staticmethod
    def _decode_images(request: NavigationRequest) -> dict[str, Any]:
        from io import BytesIO
        from PIL import Image

        def decode(value: bytes | None, fallback: Any = None) -> Any:
            if value is None:
                return fallback
            image = Image.open(BytesIO(value))
            image.load()
            return image.convert("RGB")

        front = decode(request.rgb_front)
        return {
            "front": front,
            "left": decode(request.rgb_left, front),
            "right": decode(request.rgb_right, front),
        }
