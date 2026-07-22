#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omninav_cosmos.backbones.cosmos_qwen3vl import CosmosQwen3VLAdapter
from omninav_cosmos.contracts import NavigationRequest


def image_bytes(step: int, view: str) -> bytes:
    import numpy as np
    from PIL import Image

    width, height = 320, 240
    view_index = {"left": 0, "front": 1, "right": 2}[view]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[..., 0] = np.mod(x + step * 0.03 + view_index * 0.1, 1.0)
    image[..., 1] = np.mod(y + step * 0.02, 1.0)
    image[..., 2] = 0.25 + view_index * 0.15
    buffer = BytesIO()
    Image.fromarray((image.clip(0.0, 1.0) * 255).astype("uint8")).save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Qwen3-VL DeepStack cache-on and official cache-off paths.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("at least two steps are required to observe a history cache hit")

    import torch
    import transformers

    torch.manual_seed(20260713)
    cached = CosmosQwen3VLAdapter.from_pretrained(
        args.model_path,
        cache_enabled=True,
        history_frames=4,
        allow_untrained_action_head=True,
        seed=20260713,
    )
    uncached = CosmosQwen3VLAdapter(
        model=cached.model,
        processor=cached.processor,
        action_head=copy.deepcopy(cached.action_head),
        model_fingerprint=cached.model_fingerprint,
        device=cached.device,
        history_frames=4,
        cache_enabled=False,
        action_head_trained=False,
        allow_untrained_action_head=True,
    )
    cached.reset_episode("cache-regression")
    uncached.reset_episode("cache-regression")
    records = []
    tolerances = {"waypoint": 0.005, "heading": 0.01, "arrive": 0.125, "confidence": 0.01}
    for step in range(args.steps):
        request = NavigationRequest(
            episode_id="cache-regression",
            frame_id=step,
            timestamp=time.time(),
            instruction="Go around the box, pass through the door, and stop near the red chair.",
            rgb_front=image_bytes(step, "front"),
            rgb_left=image_bytes(step, "left"),
            rgb_right=image_bytes(step, "right"),
            reset_episode=step == 0,
        )
        cache_output = cached.infer(request)
        no_cache_output = uncached.infer(request)
        cache_waypoints = torch.tensor(cache_output.waypoints)
        no_cache_waypoints = torch.tensor(no_cache_output.waypoints)
        cache_heading = torch.tensor(cache_output.heading_sin_cos)
        no_cache_heading = torch.tensor(no_cache_output.heading_sin_cos)
        cache_arrive = torch.tensor(cache_output.arrive_logits)
        no_cache_arrive = torch.tensor(no_cache_output.arrive_logits)
        errors = {
            "waypoint": float(torch.max(torch.abs(cache_waypoints - no_cache_waypoints))),
            "heading": float(torch.max(torch.abs(cache_heading - no_cache_heading))),
            "arrive": float(torch.max(torch.abs(cache_arrive - no_cache_arrive))),
            "confidence": abs(cache_output.confidence - no_cache_output.confidence),
        }
        failed = {key: value for key, value in errors.items() if value > tolerances[key]}
        if failed:
            raise RuntimeError(f"cache regression exceeded BF16 tolerances at step {step}: {failed}")
        records.append(
            {
                "step": step,
                "cache": cache_output.to_mapping(),
                "cache_off": no_cache_output.to_mapping(),
                "max_abs_error": errors,
            }
        )
    output = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "QWEN3_VL_DEV_CACHE_REGRESSION_NOT_COSMOS_NOT_ISAAC",
        "seed": 20260713,
        "model_path": str(args.model_path),
        "model_fingerprint": cached.model_fingerprint,
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "tolerances": tolerances,
        "steps": records,
        "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "peak_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024 / 1024,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT_PATH {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
