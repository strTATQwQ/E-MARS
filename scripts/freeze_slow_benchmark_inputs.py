#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FROZEN_FILES = (
    "OMNINAV_SLOW_MODEL_ISAAC_BENCHMARK_TASK_FINAL.md",
    "configs/isaac_mp3d/benchmark.yaml",
    "configs/isaac/models/qwen25_bf16.yaml",
    "configs/slow_models/qwen25_baseline_bf16.yaml",
    "configs/slow_models/cosmos_reason2_32b_bf16.yaml",
    "configs/slow_models/step3_vl_10b_bf16.yaml",
    "slow_planner/base.py",
    "slow_planner/prompt.py",
    "slow_planner/hf_base.py",
    "slow_planner/qwen25_baseline.py",
    "slow_planner/cosmos_reason2_32b.py",
    "slow_planner/step3_vl_10b.py",
    "slow_planner/client.py",
    "slow_planner/serve.py",
    "slow_benchmark/oracle_graph.py",
    "slow_benchmark/control.py",
    "slow_benchmark/video.py",
    "omninav_cosmos/contracts.py",
    "omninav_cosmos/service.py",
    "omninav_cosmos/backbones/qwen25_legacy.py",
    "omninav_cosmos/transports/wire.py",
    "omninav_cosmos/transports/zmq_client.py",
    "scripts/convert_mp3d_to_usd.py",
    "scripts/probe_isaac_mp3d_scene.py",
    "scripts/run_isaac_mp3d_slow_benchmark.py",
    "scripts/run_isaac_mp3d_batch.py",
    "scripts/run_slow_benchmark.sh",
    "scripts/collect_benchmark.py",
    "scripts/validate_slow_benchmark_run.py",
    "scripts/summarize_slow_benchmark_run.py",
    "scripts/capture_slow_model_hardware.py",
    "scripts/run_slow_model_service.py",
    "scripts/probe_slow_model_runtime.py",
    "scripts/paired_slow_benchmark_statistics.py",
    "results/slow_model_benchmark/manifest_lock.json",
    "results/slow_model_benchmark/formal_episodes.jsonl",
    "results/slow_model_benchmark/pilot_episodes.jsonl",
    "results/slow_model_benchmark/smoke_episodes.jsonl",
    "results/slow_model_benchmark/nvfp4_calibration_episodes.jsonl",
    "results/slow_model_benchmark/mp3d_collision_v4/conversion_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze all local inputs to the five-way paired benchmark.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", choices=("candidate", "frozen"), default="candidate")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []
    missing = []
    for relative in FROZEN_FILES:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        rows.append({"path": relative.replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if missing:
        raise FileNotFoundError(f"freeze inputs missing: {missing}")
    payload = {
        "schema_version": 1,
        "status": args.status,
        "benchmark": "omninav_slow_models_mp3d_r2r_oracle_map",
        "formal_pairing_key": "benchmark_episode_id",
        "formal_episode_count": 110,
        "formal_scene_count": 11,
        "selection_seed": 20260713,
        "model_revisions": {
            "cosmos_reason2_32b": "4ed9828334c4397ace8b0c62134961adbe5aed0e",
            "step3_vl_10b": "5026053b0c2f5dfaa08fc2d149384162c3c8bca1",
            "qwen25_fast_and_baseline": "local_checkpoint_sha256_manifest",
        },
        "files": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["lock_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "files": len(rows), "lock": payload["lock_content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
