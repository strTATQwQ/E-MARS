#!/usr/bin/env python3
"""Sequentially launch one isolated Isaac Sim process per frozen MP3D scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_state(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--slow-config", required=True)
    parser.add_argument("--collision-root", required=True)
    parser.add_argument("--connectivity-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runner", default="/home/song/dgx-unitree/scripts/run_isaac_mp3d_slow_benchmark.py")
    parser.add_argument("--launcher", default="/home/song/IsaacLab/isaaclab.sh")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-episodes-per-scene", type=int, default=0)
    parser.add_argument("--scenes", nargs="*")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "scene_logs"
    logs.mkdir(exist_ok=True)
    manifest_rows = list(read_jsonl(manifest))
    scenes = sorted({str(row["scene_key"]) for row in manifest_rows})
    if args.scenes:
        requested = set(args.scenes)
        missing = sorted(requested - set(scenes))
        if missing:
            raise ValueError(f"requested scenes absent from manifest: {missing}")
        scenes = [scene for scene in scenes if scene in requested]

    config_path = Path(args.config).resolve()
    slow_config_path = Path(args.slow_config).resolve()
    slow_config = yaml.safe_load(slow_config_path.read_text(encoding="utf-8"))
    if not isinstance(slow_config, dict):
        raise ValueError("slow model config must be an object")
    expected_slow_variant = str(slow_config["model_variant"])
    expected_slow_precision = str(slow_config["precision_mode"])
    slow_config_sha256 = sha256_file(slow_config_path)
    runner_path = Path(args.runner).resolve()
    collision_manifest = Path(args.collision_root).resolve() / "conversion_manifest.json"
    input_lock = {
        "schema_version": 2,
        "run_id": args.run_id,
        "device": args.device,
        "max_episodes_per_scene": args.max_episodes_per_scene,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "slow_config": {
            "path": str(slow_config_path),
            "sha256": slow_config_sha256,
            "model_variant": expected_slow_variant,
            "precision_mode": expected_slow_precision,
            "revision": str(slow_config.get("revision", "")),
        },
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest), "episode_count": len(manifest_rows)},
        "runner": {"path": str(runner_path), "sha256": sha256_file(runner_path)},
        "collision_manifest": {
            "path": str(collision_manifest),
            "sha256": sha256_file(collision_manifest),
        },
        "scenes": scenes,
    }
    lock_path = output / "inputs.lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != input_lock:
        raise RuntimeError("frozen batch inputs differ from existing inputs.lock.json")
    lock_path.write_text(json.dumps(input_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for source, name in (
        (config_path, "benchmark_config.yaml"),
        (slow_config_path, "slow_model_config.yaml"),
        (manifest, "episode_manifest.jsonl"),
    ):
        target = output / name
        if target.exists() and sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"frozen input copy differs: {target}")
        if not target.exists():
            shutil.copy2(source, target)

    state_path = output / "batch_state.json"
    state = {
        "schema_version": 1,
        "run_id": args.run_id,
        "manifest": str(manifest),
        "scenes": scenes,
        "completed_scenes": [],
        "failed_scenes": [],
        "started_at": time.time(),
    }
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("run_id") != args.run_id or previous.get("manifest") != str(manifest):
            raise RuntimeError("existing batch_state.json belongs to a different run")
        state.update(previous)

    env = dict(os.environ)
    env["PYTHONPATH"] = "/home/song/dgx-unitree" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    for scene in scenes:
        if scene in state["completed_scenes"]:
            continue
        command = [
            args.launcher,
            "-p",
            args.runner,
            "--config",
            str(config_path),
            "--manifest",
            str(manifest),
            "--scene",
            scene,
            "--scene-usd",
            str(Path(args.collision_root).resolve() / scene / "scene_collision.usd"),
            "--connectivity",
            str(Path(args.connectivity_root).resolve() / f"{scene}_connectivity.json"),
            "--output",
            str(output),
            "--run-id",
            args.run_id,
            "--expected-slow-model-variant",
            expected_slow_variant,
            "--expected-slow-precision-mode",
            expected_slow_precision,
            "--expected-slow-config-sha256",
            slow_config_sha256,
            "--enable_cameras",
            "--device",
            args.device,
            "--kit_args=--/renderer/multiGpu/enabled=0 --/renderer/multiGpu/autoEnable=0",
        ]
        if args.max_episodes_per_scene > 0:
            command.extend(["--max-episodes", str(args.max_episodes_per_scene)])
        with (output / "commands.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp": time.time(),
                        "scene_id": scene,
                        "run_id": args.run_id,
                        "command": command,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
        log_path = logs / f"{scene}.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        phases_path = output / "phases.jsonl"
        phases = [row for row in read_jsonl(phases_path)] if phases_path.exists() else []
        scene_phases = [row for row in phases if row.get("scene_id") == scene]
        finished = bool(scene_phases and scene_phases[-1].get("phase") == "run_finished")
        if process.returncode != 0 or not finished:
            state["failed_scenes"] = sorted(set(state["failed_scenes"]) | {scene})
            state["last_error"] = {
                "scene": scene,
                "returncode": process.returncode,
                "last_phase": scene_phases[-1] if scene_phases else None,
                "log": str(log_path),
            }
            write_state(state_path, state)
            return 2
        state["completed_scenes"] = sorted(set(state["completed_scenes"]) | {scene})
        state["failed_scenes"] = sorted(set(state["failed_scenes"]) - {scene})
        state["updated_at"] = time.time()
        write_state(state_path, state)
    state["finished_at"] = time.time()
    write_state(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
