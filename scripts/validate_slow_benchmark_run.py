#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_evidenced_slow_only_target_terminal(episode: dict, slow_rows: list[dict]) -> bool:
    """Allow a terminal target_found decision before FastPolicy is ever invoked.

    This is a model outcome, not missing FastPolicy evidence: the benchmark contract
    explicitly permits target_found to terminate an episode and scores false positives.
    Keep the exception narrow so an arbitrary zero-fast episode still fails validation.
    """
    if int(episode.get("fast_steps", -1)) != 0 or float(episode.get("executed_path_m", -1.0)) != 0.0:
        return False
    if int(episode.get("slow_decisions", -1)) != len(slow_rows) or not slow_rows:
        return False
    if not any(row.get("decision", {}).get("decision") == "target_found" for row in slow_rows):
        return False
    true_positive = bool(episode.get("success")) and int(episode.get("target_found_tp", 0)) >= 1
    false_positive = (
        not bool(episode.get("success"))
        and episode.get("failure_reason") == "false_target_found"
        and int(episode.get("target_found_fp", 0)) >= 1
    )
    return true_positive or false_positive


def scene_key(row: dict) -> str:
    explicit = row.get("scene_key")
    if explicit:
        return str(explicit)
    return Path(str(row["scene_id"])).stem


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one frozen slow-model benchmark run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    manifest = read_jsonl(Path(args.manifest))
    expected = [str(row["benchmark_episode_id"]) for row in manifest]
    expected_scenes = sorted({scene_key(row) for row in manifest})
    episodes = [row for row in read_jsonl(root / "episodes.jsonl") if row.get("run_id") == args.run_id]
    latency = [row for row in read_jsonl(root / "latency.jsonl") if row.get("run_id") == args.run_id]
    phases = read_jsonl(root / "phases.jsonl")
    errors: list[str] = []
    counts = Counter(str(row.get("episode_id")) for row in episodes)
    duplicates = sorted(key for key, value in counts.items() if value != 1)
    if duplicates:
        errors.append(f"duplicate episode results: {duplicates[:10]}")
    unexpected = sorted(set(counts) - set(expected))
    missing = sorted(set(expected) - set(counts))
    if unexpected:
        errors.append(f"unexpected episodes: {unexpected[:10]}")
    if args.require_complete and missing:
        errors.append(f"missing episodes: {len(missing)}")
    fatal_phases = [row for row in phases if row.get("phase") == "fatal_error"]
    if fatal_phases:
        errors.append("phases.jsonl contains fatal_error")
    finished_counts = Counter(str(row.get("scene_id")) for row in phases if row.get("phase") == "run_finished")
    if args.require_complete:
        missing_finished = sorted(set(expected_scenes) - set(finished_counts))
        duplicate_finished = sorted(scene for scene, count in finished_counts.items() if count != 1)
        unexpected_finished = sorted(set(finished_counts) - set(expected_scenes))
        if missing_finished:
            errors.append(f"scenes missing run_finished: {missing_finished}")
        if duplicate_finished:
            errors.append(f"duplicate run_finished scenes: {duplicate_finished}")
        if unexpected_finished:
            errors.append(f"unexpected run_finished scenes: {unexpected_finished}")
    fast_rows = [row for row in latency if row.get("kind") == "fast" and "action_head_trained" in row]
    slow_rows = [row for row in latency if row.get("kind") == "slow"]
    unexpected_latency_ids = sorted(
        {str(row.get("episode_id")) for row in latency} - set(expected)
    )
    if unexpected_latency_ids:
        errors.append(f"latency rows contain unexpected episodes: {unexpected_latency_ids[:10]}")
    if episodes and not slow_rows:
        errors.append("no slow latency rows")
    if episodes and not fast_rows:
        errors.append("no full FastPolicy latency rows")
    if any(not row.get("action_head_trained") for row in fast_rows):
        errors.append("a FastPolicy row lacks trained action head")
    if any(not isinstance(row.get("evaluation"), dict) for row in slow_rows):
        errors.append("a SlowPlanner row lacks evaluator-only frontier metrics")
    if episodes:
        fast_episode_ids = {str(row.get("episode_id")) for row in fast_rows}
        slow_episode_ids = {str(row.get("episode_id")) for row in slow_rows}
        result_episode_ids = set(counts)
        episodes_by_id = {str(row.get("episode_id")): row for row in episodes}
        slow_rows_by_id = {
            episode_id: [row for row in slow_rows if str(row.get("episode_id")) == episode_id]
            for episode_id in result_episode_ids
        }
        slow_only_target_terminals = sorted(
            episode_id
            for episode_id in result_episode_ids - fast_episode_ids
            if is_evidenced_slow_only_target_terminal(episodes_by_id[episode_id], slow_rows_by_id[episode_id])
        )
        without_fast = sorted((result_episode_ids - fast_episode_ids) - set(slow_only_target_terminals))
        without_slow = sorted(result_episode_ids - slow_episode_ids)
        if without_fast:
            errors.append(f"episodes without FastPolicy rows: {without_fast[:10]}")
        if without_slow:
            errors.append(f"episodes without SlowPlanner rows: {without_slow[:10]}")
    precision_path = root / "precision_manifest.json"
    hardware_path = root / "hardware.json"
    if not precision_path.exists():
        errors.append("precision_manifest.json missing")
    if not hardware_path.exists():
        errors.append("hardware.json missing")
    precision = read_json(precision_path)
    if precision:
        if precision.get("run_id") != args.run_id:
            errors.append("precision manifest run_id mismatch")
        if precision.get("map_setting") != "oracle_map":
            errors.append("precision manifest is not oracle_map")
        if precision.get("fast_action_head_required") is not True:
            errors.append("precision manifest does not require the trained FastPolicy action head")
        state_scope = precision.get("fast_episode_state_scope")
        if state_scope not in (None, "run_id_plus_canonical_episode_id"):
            errors.append(f"unexpected FastPolicy episode state scope: {state_scope}")
        if precision.get("video_evidence_required") and state_scope != "run_id_plus_canonical_episode_id":
            errors.append("frozen run lacks run-scoped FastPolicy episode state isolation")
        if precision.get("collision_definition") != "selected_view_depth_guard_breach":
            errors.append("unexpected or overstated collision definition")
        if precision.get("synchronous_safe_hold") is not True:
            errors.append("benchmark is not synchronous safe-hold")
        if precision.get("video_evidence_required"):
            video_paths: set[str] = set()
            for episode in episodes:
                video = episode.get("video")
                if not isinstance(video, dict):
                    errors.append(f"episode lacks video evidence: {episode.get('episode_id')}")
                    continue
                relative = str(video.get("path", ""))
                candidate = (root / relative).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    errors.append(f"video path escapes run directory: {relative}")
                    continue
                if relative in video_paths:
                    errors.append(f"duplicate video path: {relative}")
                video_paths.add(relative)
                if not candidate.is_file():
                    errors.append(f"video file missing: {relative}")
                    continue
                if video.get("sha256") != sha256_file(candidate):
                    errors.append(f"video hash mismatch: {relative}")
                if int(video.get("bytes", -1)) != candidate.stat().st_size:
                    errors.append(f"video byte count mismatch: {relative}")
                if int(video.get("frame_count", 0)) <= 0:
                    errors.append(f"video has no frames: {relative}")
                if video.get("source") != "isaac_front_render_product":
                    errors.append(f"unexpected video source: {relative}")
    lock = read_json(root / "inputs.lock.json")
    if not lock:
        errors.append("inputs.lock.json missing")
    else:
        if int(lock.get("schema_version", 0)) < 2:
            errors.append("input lock predates slow-model/device binding")
        if lock.get("run_id") != args.run_id:
            errors.append("input lock run_id mismatch")
        if int(lock.get("manifest", {}).get("episode_count", -1)) != len(expected):
            errors.append("input lock episode count mismatch")
        copied_inputs = (
            ("manifest", root / "episode_manifest.jsonl"),
            ("config", root / "benchmark_config.yaml"),
            ("slow_config", root / "slow_model_config.yaml"),
        )
        for key, copied_path in copied_inputs:
            if not copied_path.exists():
                errors.append(f"frozen {key} copy missing")
                continue
            if lock.get(key, {}).get("sha256") != sha256_file(copied_path):
                errors.append(f"frozen {key} copy hash mismatch")
        if precision:
            if precision.get("benchmark_config_sha256") != lock.get("config", {}).get("sha256"):
                errors.append("precision manifest benchmark config hash mismatch")
            if precision.get("slow_service_config_sha256") != lock.get("slow_config", {}).get("sha256"):
                errors.append("precision manifest slow service config hash mismatch")
            if precision.get("slow_model_variant") != lock.get("slow_config", {}).get("model_variant"):
                errors.append("precision manifest slow model variant mismatch")
            if precision.get("slow_precision_mode") != lock.get("slow_config", {}).get("precision_mode"):
                errors.append("precision manifest slow precision mismatch")
            if precision.get("device") != lock.get("device"):
                errors.append("precision manifest simulation device mismatch")
    required_episode_fields = (
        "simulation_seconds",
        "fast_active_wall_seconds",
        "fast_effective_control_hz_excluding_slow_hold",
        "fast_control_hz_during_slow_hold",
        "slow_calls_per_minute",
        "frontier_decisions",
        "frontier_valid_decisions",
        "frontier_geodesic_improved",
        "frontier_geodesic_delta_sum_m",
        "repeated_frontier_decisions",
    )
    for episode in episodes:
        absent = [key for key in required_episode_fields if key not in episode]
        if absent:
            errors.append(f"episode {episode.get('episode_id')} lacks metrics: {absent}")
        if episode.get("map_setting") != "oracle_map":
            errors.append(f"episode {episode.get('episode_id')} is not labeled oracle_map")
    commands = read_jsonl(root / "commands.jsonl")
    commanded_scenes = {str(row.get("scene_id")) for row in commands if row.get("run_id") == args.run_id}
    if args.require_complete and commanded_scenes != set(expected_scenes):
        errors.append("reproducible command scene set mismatch")
    hardware_index = read_json(hardware_path)
    if args.require_complete and hardware_index:
        expected_hardware = {f"hardware_{scene}.json" for scene in expected_scenes}
        actual_hardware = set(map(str, hardware_index.get("scene_files", [])))
        if actual_hardware != expected_hardware:
            errors.append("hardware evidence scene set mismatch")
        for name in expected_hardware:
            if not (root / name).is_file():
                errors.append(f"hardware evidence missing: {name}")
    state = read_json(root / "batch_state.json")
    if args.require_complete:
        if not state:
            errors.append("batch_state.json missing")
        else:
            if state.get("run_id") != args.run_id:
                errors.append("batch state run_id mismatch")
            if sorted(map(str, state.get("completed_scenes", []))) != expected_scenes:
                errors.append("batch state completed scene set mismatch")
            if state.get("failed_scenes"):
                errors.append(f"batch state has failed scenes: {state['failed_scenes']}")
            if not state.get("finished_at"):
                errors.append("batch state lacks finished_at")
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "valid": not errors,
        "require_complete": args.require_complete,
        "expected_episodes": len(expected),
        "completed_episodes": len(episodes),
        "missing_episodes": len(missing),
        "expected_scenes": len(expected_scenes),
        "finished_scenes": len(finished_counts),
        "fatal_phases": len(fatal_phases),
        "video_evidence_required": bool(precision.get("video_evidence_required")),
        "commanded_scenes": len(commanded_scenes),
        "slow_rows": len(slow_rows),
        "fast_rows": len(fast_rows),
        "slow_only_target_terminal_episodes": len(slow_only_target_terminals) if episodes else 0,
        "successes": sum(bool(row.get("success")) for row in episodes),
        "collision_episodes": sum(int(row.get("collisions", 0)) > 0 for row in episodes),
        "errors": errors,
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
