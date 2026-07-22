from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .backbones.qwen25_legacy import Qwen25LegacyAdapter
from .service import NavigationInferenceService
from .transports.zmq_server import ZmqNavigationServer


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for YAML service configs") from exc
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("service config must be an object")
    return value


def expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def build_adapter(config: dict[str, Any]):
    variant = str(config["model_variant"])
    if variant == "qwen25_legacy":
        return Qwen25LegacyAdapter.from_helper(
            repo=expand(config["repo"]),
            model_path=expand(config["model_path"]),
            helper_path=expand(config["helper_path"]),
            config_name=str(config.get("input_config", "F_front_semantic_history_384tok")),
            attn_implementation=str(config.get("attn_implementation", "flash_attention_2")),
            predict_scale=float(config.get("waypoint_scale", 0.3)),
            legacy_waypoint_forward_axis=str(config.get("legacy_waypoint_forward_axis", "y")),
            input_contract=str(config.get("input_contract", "helper_config")),
        )
    if variant in {"cosmos_qwen3vl", "qwen3vl_dev"}:
        from .backbones.cosmos_qwen3vl import CosmosQwen3VLAdapter

        action_head = config.get("action_head_path")
        return CosmosQwen3VLAdapter.from_pretrained(
            expand(config["model_path"]),
            action_head_path=expand(action_head) if action_head else None,
            attn_implementation=str(config.get("attn_implementation", "sdpa")),
            history_frames=int(config.get("history_frames", 4)),
            cache_enabled=bool(config.get("cache_enabled", True)),
            allow_untrained_action_head=bool(config.get("allow_untrained_action_head", False)),
            seed=int(config.get("seed", 20260713)),
            model_variant=(
                "qwen3-vl-8b-dev-untrained-action-head"
                if variant == "qwen3vl_dev"
                else "nvidia-cosmos-reason2-8b-qwen3-vl"
            ),
        )
    raise ValueError(f"unsupported model_variant={variant!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve OmniNav backbone-neutral direct actions over ZeroMQ.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(Path(args.config))
    adapter = build_adapter(config)
    service = NavigationInferenceService(adapter, max_request_age_s=float(config.get("max_request_age_s", 1.0)))
    server = ZmqNavigationServer(
        service,
        bind=str(config.get("bind", "tcp://0.0.0.0:8100")),
        log_path=expand(str(config.get("log_path", "results/model_service.jsonl"))),
    )
    try:
        server.run()
    finally:
        server.close()
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
