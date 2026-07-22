from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


OBSTACLE_CATEGORIES = (
    "static_clutter",
    "narrow_door",
    "narrow_corridor",
    "corner_dead_end",
    "dynamic_crossing",
)

REQUIRED_EPISODE_FIELDS = (
    "success",
    "spl",
    "path_length",
    "shortest_path_length",
    "collision",
    "collision_count",
    "minimum_obstacle_distance",
    "emergency_stop_count",
    "timeout",
    "stuck",
    "arrive_false_positive",
    "decision_latency_ms",
    "vision_latency_ms",
    "model_latency_ms",
    "network_latency_ms",
    "control_frequency_hz",
    "peak_memory",
    "subgoal_completion_rate",
    "random_seed",
    "model_variant",
    "precision_mode",
)


class BacktestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ExpectedModel:
    model_variant: str
    precision_mode: str
    require_trained_action_head: bool = True


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BacktestValidationError(f"invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise BacktestValidationError(f"JSONL line {line_number} is not an object")
        records.append(value)
    return records


def manifest_episodes(manifest: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    phases = manifest.get("phases")
    if not isinstance(phases, Mapping) or not isinstance(phases.get(phase), list):
        raise BacktestValidationError(f"seed manifest has no phase {phase!r}")
    return [dict(value) for value in phases[phase] if isinstance(value, Mapping)]


def validate_seed_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    smoke = manifest_episodes(manifest, "smoke")
    formal = manifest_episodes(manifest, "formal")
    errors: list[str] = []
    if len(smoke) != 5:
        errors.append(f"smoke must contain exactly 5 episodes, got {len(smoke)}")
    ids = [str(row.get("episode_key") or "") for row in formal]
    if any(not value for value in ids):
        errors.append("every formal episode needs episode_key")
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate formal episode keys: {duplicates[:5]}")
    seeds = [row.get("random_seed") for row in formal]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in seeds):
        errors.append("every formal episode needs an integer random_seed")
    obstacle_counts = Counter(
        str(row.get("obstacle_category") or "")
        for row in formal
        if str(row.get("benchmark_group") or "") == "obstacle_avoidance"
    )
    for category in OBSTACLE_CATEGORIES:
        if obstacle_counts[category] < 20:
            errors.append(f"obstacle category {category!r} has {obstacle_counts[category]} episodes; need >=20")
    semantic_count = sum(
        1
        for row in formal
        if str(row.get("benchmark_group") or "") in {"natural_language", "complex_semantic"}
    )
    if semantic_count < 30:
        errors.append(f"natural-language/complex episodes={semantic_count}; need >=30")
    complex_count = sum(1 for row in formal if str(row.get("benchmark_group") or "") == "complex_semantic")
    if errors:
        raise BacktestValidationError("; ".join(errors))
    return {
        "smoke_episodes": len(smoke),
        "formal_episodes": len(formal),
        "obstacle_counts": dict(obstacle_counts),
        "semantic_episodes": semantic_count,
        "complex_semantic_episodes": complex_count,
    }


def validate_service_health(health: Mapping[str, Any], expected: ExpectedModel) -> dict[str, Any]:
    errors: list[str] = []
    if not bool(health.get("ready")):
        errors.append("model service is not ready")
    actual_variant = str(health.get("model_variant") or "")
    actual_precision = str(health.get("precision_mode") or "")
    if actual_variant != expected.model_variant:
        errors.append(f"model_variant={actual_variant!r}, expected {expected.model_variant!r}")
    if actual_precision != expected.precision_mode:
        errors.append(f"precision_mode={actual_precision!r}, expected {expected.precision_mode!r}")
    if expected.require_trained_action_head and health.get("action_head_trained") is not True:
        errors.append("action_head_trained is not true")
    if "untrained" in actual_variant.lower():
        errors.append("model variant is explicitly marked untrained")
    if errors:
        raise BacktestValidationError("; ".join(errors))
    return dict(health)


def validate_episode_records(
    records: Iterable[Mapping[str, Any]],
    expected_rows: Iterable[Mapping[str, Any]],
    expected_model: ExpectedModel,
) -> dict[str, Any]:
    rows = [dict(value) for value in records]
    expected = [dict(value) for value in expected_rows]
    by_key = {str(row.get("episode_key") or row.get("task_id") or ""): row for row in rows}
    errors: list[str] = []
    if len(rows) != len(expected):
        errors.append(f"result episode count={len(rows)}, expected {len(expected)}")
    for wanted in expected:
        key = str(wanted.get("episode_key") or "")
        record = by_key.get(key)
        if record is None:
            errors.append(f"missing result for {key}")
            continue
        missing = [field for field in REQUIRED_EPISODE_FIELDS if field not in record]
        if missing:
            errors.append(f"{key}: missing fields {missing}")
        if record.get("random_seed") != wanted.get("random_seed"):
            errors.append(f"{key}: random_seed mismatch")
        if record.get("mock") is not False or record.get("use_isaac") is not True or record.get("live_closed_loop") is not True:
            errors.append(f"{key}: not a real live Isaac record")
        if record.get("isaac_headless") is not True:
            errors.append(f"{key}: isaac_headless is not true")
        if str(record.get("camera_source") or "") != "real_isaac_render_product_udp_15012":
            errors.append(f"{key}: camera source is not the real Isaac render product")
        if record.get("action_head_trained") is not True:
            errors.append(f"{key}: action head was not proven trained")
        if int(record.get("fallback_count", 0) or 0) != 0:
            errors.append(f"{key}: fallback_count is nonzero")
        if int(record.get("num_omninav_real_image_calls", 0) or 0) <= 0:
            errors.append(f"{key}: no real-image OmniNav call")
        if str(record.get("model_variant") or "") != expected_model.model_variant:
            errors.append(f"{key}: model variant mismatch")
        if str(record.get("precision_mode") or "") != expected_model.precision_mode:
            errors.append(f"{key}: precision mode mismatch")
    if errors:
        preview = errors[:30]
        suffix = f"; ... and {len(errors) - len(preview)} more" if len(errors) > len(preview) else ""
        raise BacktestValidationError("; ".join(preview) + suffix)
    return {
        "episodes": len(rows),
        "successes": sum(bool(row.get("success")) for row in rows),
        "collisions": sum(bool(row.get("collision")) for row in rows),
        "model_variant": expected_model.model_variant,
        "precision_mode": expected_model.precision_mode,
    }


def validate_artifact_tree(run_dir: str | Path, *, require_video: bool = True) -> dict[str, str]:
    root = Path(run_dir)
    required = {
        "episodes_jsonl": root / "episodes.jsonl",
        "config": root / "config.yaml",
        "seeds": root / "seeds.json",
        "runner_log": root / "runner.log",
        "isaac_log": root / "isaac.log",
        "model_health": root / "model_health.json",
        "versions": root / "versions.json",
        "commands": root / "commands.txt",
    }
    if require_video:
        required["video"] = root / "videos" / "representative.mp4"
        required["video_metadata"] = root / "videos" / "representative.json"
    missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise BacktestValidationError(f"missing or empty artifacts: {missing}")
    return {name: str(path) for name, path in required.items()}
