#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omninav_cosmos.backbones.qwen25_legacy import Qwen25LegacyAdapter
from omninav_cosmos.contracts import NavigationRequest


def deterministic_image(step: int, view: str, width: int = 640, height: int = 569) -> bytes:
    import numpy as np
    from PIL import Image

    view_index = {"left": 0, "front": 1, "right": 2}[view]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[..., 0] = np.mod(x + 0.035 * step + 0.11 * view_index, 1.0)
    image[..., 1] = np.mod(y + 0.025 * step, 1.0)
    image[..., 2] = 0.20 + 0.10 * view_index
    if view == "front":
        image[..., 1] = np.where(np.abs(x - 0.50) < 0.055, 0.92, image[..., 1])
    buffer = BytesIO()
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype("uint8")).save(buffer, format="PNG")
    return buffer.getvalue()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNCOMMITTED_OR_NO_COMMIT"


def main() -> int:
    parser = argparse.ArgumentParser(description="Record deterministic OmniNav action-head regression tensors.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--helper-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-name", default="F_front_semantic_history_384tok")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--instruction",
        default="Go around the box, pass through the doorway, and stop near the red chair.",
    )
    args = parser.parse_args()

    if args.steps <= 0 or args.warmup < 0:
        parser.error("steps must be positive and warmup must be non-negative")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import numpy as np

    torch.manual_seed(20260713)
    np.random.seed(20260713)
    adapter = Qwen25LegacyAdapter.from_helper(
        repo=args.repo,
        model_path=args.model_path,
        helper_path=args.helper_path,
        config_name=args.config_name,
        attn_implementation=args.attn_implementation,
    )
    import transformers
    from omninav_cosmos.action_head import ActionHeadConfig, NavigationActionHead

    model_config = adapter.model.config
    standalone_head = NavigationActionHead(
        ActionHeadConfig(
            hidden_size=int(model_config.hidden_size),
            waypoint_count=int(model_config.waypoint_number),
            attention_heads=4,
            action_former_layers=int(getattr(model_config, "query_action_layer", 1)),
            arrive_count=(
                int(model_config.waypoint_number) if bool(getattr(model_config, "use_arrive_list", False)) else 1
            ),
            predict_heading=bool(getattr(model_config, "predict_angle", False)),
            predict_confidence=False,
            normalize_heading=False,
        )
    ).to(device=adapter.model.query_action.device, dtype=adapter.model.query_action.dtype)
    standalone_head.load_legacy_state_dict(adapter.model.state_dict(), strict=True)
    standalone_head.eval()
    captured_hidden: list[Any] = []

    def capture_hidden(_module, _inputs, output):
        captured_hidden[:] = [output[0].detach()]

    hook = adapter.model.model.register_forward_hook(capture_hidden)

    adapter.reset_episode("fixed-regression")
    records: list[dict[str, Any]] = []
    input_manifest: list[dict[str, Any]] = []
    equivalence_tolerances = {
        "waypoint_max_abs_error": 1e-3,
        "heading_max_abs_error": 5e-3,
        "arrive_max_abs_error": 0.125,
    }
    total = args.steps + args.warmup
    try:
        for step in range(total):
            frames = {view: deterministic_image(step, view) for view in ("front", "left", "right")}
            input_manifest.append(
                {"step": step, **{f"{view}_sha256": sha256_bytes(value) for view, value in frames.items()}}
            )
            request = NavigationRequest(
                episode_id="fixed-regression",
                frame_id=step,
                timestamp=time.time(),
                instruction=args.instruction,
                rgb_front=frames["front"],
                rgb_left=frames["left"],
                rgb_right=frames["right"],
                reset_episode=step == 0,
            )
            captured_hidden.clear()
            result = adapter.infer(request)
            if not captured_hidden:
                raise RuntimeError("failed to capture backbone hidden states")
            with torch.inference_mode():
                standalone = standalone_head(captured_hidden[0])
            legacy_waypoints = torch.tensor(result.waypoints, dtype=torch.float32)
            legacy_headings = torch.tensor(result.heading_sin_cos, dtype=torch.float32)
            legacy_arrive = torch.tensor(result.arrive_logits, dtype=torch.float32)
            equivalence = {
                "waypoint_max_abs_error": float(
                    torch.max(torch.abs(standalone.waypoints[0].float().cpu() * adapter.predict_scale - legacy_waypoints))
                ),
                "heading_max_abs_error": float(
                    torch.max(torch.abs(standalone.heading_sin_cos[0].float().cpu() - legacy_headings))
                ),
                "arrive_max_abs_error": float(
                    torch.max(torch.abs(standalone.arrive_logits[0].float().cpu() - legacy_arrive))
                ),
            }
            failed = {
                key: {"observed": equivalence[key], "tolerance": tolerance}
                for key, tolerance in equivalence_tolerances.items()
                if equivalence[key] > tolerance
            }
            if failed:
                raise RuntimeError(f"standalone action-head regression failed: {equivalence}")
            if step >= args.warmup:
                record = result.to_mapping()
                record["standalone_action_head_equivalence"] = equivalence
                records.append(record)
    finally:
        hook.remove()

    model_path = Path(args.model_path)
    source_files = [
        Path(args.repo) / "infer_ovon" / "agent" / "waypoint_agent_ovon.py",
        Path(args.repo)
        / "train_code"
        / "transformers-main"
        / "src"
        / "transformers"
        / "models"
        / "qwen2_5_vl"
        / "modeling_qwen2_5_vl.py",
        Path(args.helper_path),
    ]
    artifact = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "FIXED_INPUT_REGRESSION_NOT_ISAAC_BACKTEST",
        "seed": 20260713,
        "instruction": args.instruction,
        "model_variant": adapter.model_variant,
        "precision_mode": adapter.precision_mode,
        "config_name": args.config_name,
        "attn_implementation": args.attn_implementation,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "workspace_git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "model_config_sha256": file_sha256(model_path / "config.json"),
        "source_sha256": {str(path): file_sha256(path) for path in source_files},
        "input_manifest": input_manifest,
        "warmup": args.warmup,
        "standalone_action_head_equivalence_tolerances": equivalence_tolerances,
        "steps": records,
        "peak_memory_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        ),
        "peak_memory_reserved_mib": (
            torch.cuda.max_memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        ),
    }
    destination = output_dir / "fixed_regression.json"
    destination.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT_PATH {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
