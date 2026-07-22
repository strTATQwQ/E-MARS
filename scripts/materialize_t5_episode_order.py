#!/usr/bin/env python3
"""Materialize the exact fresh upstream evaluator order for a T5 model run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value: Any, label: str) -> str:
    result = str(value).strip()
    if not result or any(ord(character) < 32 or ord(character) == 127 for character in result):
        raise RuntimeError(f"invalid {label}")
    return result


def _episode_key(item: Any) -> str:
    if not isinstance(item, dict):
        raise RuntimeError("dataset episode is not an object")
    return "%s_%s" % (
        _text(item.get("trajectory_id", ""), "trajectory_id"),
        _text(item.get("episode_id", ""), "episode_id"),
    )


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _payload_from_path_key_data(
    dataset_file: Path,
    expected_count: int,
    path_key_data: Any,
    source_contract: dict[str, Any],
) -> dict[str, Any]:
    if dataset_file.is_symlink() or not dataset_file.is_file():
        raise RuntimeError("T5 dataset must be a regular non-symlink file")
    with gzip.open(dataset_file, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    episodes = value.get("episodes") if isinstance(value, dict) else None
    if not isinstance(episodes, list) or len(episodes) != expected_count:
        raise RuntimeError("T5 dataset episode count does not match the run contract")
    raw_keys = [_episode_key(item) for item in episodes]
    raw_episode_ids = [_text(item.get("episode_id", ""), "episode_id") for item in episodes]
    if len(set(raw_keys)) != len(raw_keys) or len(set(raw_episode_ids)) != len(raw_episode_ids):
        raise RuntimeError("T5 model dataset identities must be unique")
    if not isinstance(path_key_data, dict):
        raise RuntimeError("BasePathKeyEpisodeloader path_key_data is not a dictionary")
    materialized_keys = [str(key) for key in path_key_data.keys()]
    if (
        len(materialized_keys) != expected_count
        or len(set(materialized_keys)) != len(materialized_keys)
        or set(materialized_keys) != set(raw_keys)
    ):
        raise RuntimeError(
            "frozen T5 dataset changed under exact upstream path-key materialization"
        )
    episode_id_by_key = dict(zip(raw_keys, raw_episode_ids))
    for key in materialized_keys:
        item = path_key_data[key]
        observed_episode_id = _text(item.get("episode_id", ""), "episode_id")
        if observed_episode_id != episode_id_by_key[key]:
            raise RuntimeError("upstream path_key_data episode identity drifted")
    ordered_keys = list(reversed(materialized_keys))
    return {
        "schema_version": 1,
        "status": "PASS",
        "dataset_file": str(dataset_file),
        "dataset_sha256": _sha256(dataset_file),
        "dataset_episode_count": expected_count,
        "raw_episode_keys": raw_keys,
        "materialized_pre_reverse_episode_keys": materialized_keys,
        "ordered_episode_keys": ordered_keys,
        "ordered_episode_ids": [episode_id_by_key[key] for key in ordered_keys],
        "loader_contract": source_contract,
    }


def materialize(
    runtime_overlay: Path,
    config: Path,
    dataset_file: Path,
    expected_count: int,
    output: Path,
) -> dict[str, Any]:
    for path, label in (
        (runtime_overlay, "runtime overlay"),
        (config, "config"),
        (dataset_file, "dataset"),
    ):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"T5 {label} must be a regular non-symlink file")
    runtime_overlay = runtime_overlay.resolve()
    config = config.resolve()
    dataset_file = dataset_file.resolve()
    runtime = _load_module(runtime_overlay, "internnav_go2_runtime")
    runtime.install_go2_runtime()
    config_module = _load_module(config, "internnav_t5_episode_order_config")
    from internnav.configs.evaluator.vln_default_config import get_config
    from internnav.env.utils.episode_loader.base import BasePathKeyEpisodeloader

    cfg = get_config(config_module.eval_cfg)
    if not cfg.eval_settings["use_agent_server"]:
        raise RuntimeError("T5 evaluator config must use the agent server")
    if cfg.task.task_settings["use_distributed"]:
        raise RuntimeError("distributed evaluator ordering needs an explicit rank contract")
    settings = cfg.dataset.dataset_settings
    loader = BasePathKeyEpisodeloader(
        dataset_type=cfg.dataset.dataset_type,
        base_data_dir=settings["base_data_dir"],
        split_data_types=settings["split_data_types"],
        robot_offset=settings["robot_offset"],
        filter_same_trajectory=settings["filter_same_trajectory"],
        revise_data=True,
        filter_stairs=settings["filter_stairs"],
        rank=0,
        world_size=1,
    )
    base_module = sys.modules[BasePathKeyEpisodeloader.__module__]
    base_source = Path(base_module.__file__).resolve()
    dataset_utils_source = base_source.with_name("dataset_utils.py")
    source_contract = {
        "materializer": "BasePathKeyEpisodeloader.path_key_data",
        "fresh_resumable_order": "reverse_materialized_path_keys",
        "rank": 0,
        "world_size": 1,
        "dataset_type": str(cfg.dataset.dataset_type),
        "dataset_settings": {
            "base_data_dir": str(settings["base_data_dir"]),
            "split_data_types": list(settings["split_data_types"]),
            "robot_offset": list(settings["robot_offset"]),
            "filter_same_trajectory": bool(settings["filter_same_trajectory"]),
            "filter_stairs": bool(settings["filter_stairs"]),
            "revise_data": True,
        },
        "runtime_overlay_sha256": _sha256(runtime_overlay),
        "config_sha256": _sha256(config),
        "base_loader_sha256": _sha256(base_source),
        "dataset_utils_sha256": _sha256(dataset_utils_source),
    }
    payload = _payload_from_path_key_data(
        dataset_file, expected_count, loader.path_key_data, source_contract
    )
    _write_atomic(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-overlay", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(
        args.runtime_overlay,
        args.config,
        args.dataset_file,
        args.expected_count,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
