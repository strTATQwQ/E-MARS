#!/usr/bin/env python3
"""Convert one migrated T4.6 online arm into strict episode records."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import statistics
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from t4_completion.ablation.contract import (  # noqa: E402
    VARIANT_IDS,
    canonical_sha256,
    load_matrix,
    resolve_variant_configs,
)
from t4_completion.ablation.records import validate_episode_record  # noqa: E402
from scripts.summarize_internnav_progress import summarize  # noqa: E402


MATRIX_PATH = ROOT / "configs/completion_sim/ablation/frozen_matrix_v1.json"
DATASET_MEMBER = "smoke_dataset/val_unseen/val_unseen.json.gz"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{number}")
        result.append(value)
    return result


def _archive_member(archive: Path, name: str, *, maximum_bytes: int) -> bytes:
    with tarfile.open(archive, mode="r:gz") as stream:
        matches = [member for member in stream.getmembers() if member.name == name]
        if len(matches) != 1:
            raise ValueError(f"archive must contain exactly one {name}")
        member = matches[0]
        if not member.isfile() or member.size > maximum_bytes:
            raise ValueError(f"unsafe archive member: {name}")
        handle = stream.extractfile(member)
        if handle is None:
            raise ValueError(f"cannot read archive member: {name}")
        payload = handle.read(maximum_bytes + 1)
        if len(payload) != member.size:
            raise ValueError(f"archive member size changed: {name}")
        return payload


def latency_summary(values: Iterable[float]) -> dict[str, int | float | None]:
    samples = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("latency values must be finite and nonnegative")
    if not samples:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}

    def percentile(fraction: float) -> float:
        index = fraction * (len(samples) - 1)
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return samples[lower]
        return samples[lower] + (samples[upper] - samples[lower]) * (index - lower)

    return {
        "count": len(samples),
        "mean": statistics.fmean(samples),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": samples[-1],
    }


def _seed(episode_key: str) -> int:
    digest = hashlib.sha256(
        f"internnav-t4.6-ablation-v1\0{episode_key}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _dataset_identity(episode: dict[str, Any]) -> tuple[str, str]:
    episode_id = str(episode.get("episode_id", episode.get("id", ""))).strip()
    trajectory_id = str(episode.get("trajectory_id", "")).strip()
    if not episode_id:
        raise ValueError("dataset episode lacks episode_id")
    return episode_id, f"{trajectory_id}_{episode_id}" if trajectory_id else episode_id


def _index_by_episode(
    records: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        episode_id = str(record.get("episode_id", ""))
        if episode_id:
            result[episode_id].append(record)
    return dict(result)


def _recover_authoritative_metrics(
    result_dir: Path, completed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if all(isinstance(item.get("official_metrics"), dict) for item in completed):
        return completed

    parsed_candidates: list[list[dict[str, Any]]] = []
    for relative in ("isaac-run/logs_sanitized/eval.log", "isaac-run/logs/eval.log"):
        path = result_dir / relative
        if not path.is_file():
            continue
        parsed = summarize(path).get("episodes")
        if not isinstance(parsed, list) or len(parsed) != len(completed):
            continue
        if not all(isinstance(item.get("official_metrics"), dict) for item in parsed):
            continue
        parsed_candidates.append(parsed)
    if not parsed_candidates:
        raise ValueError("authoritative per-episode metrics are absent from evidence logs")

    recovered_by_key: dict[str, dict[str, Any]] = {}
    for candidate in parsed_candidates:
        candidate_by_key = {str(item.get("trajectory_id", "")): item for item in candidate}
        if len(candidate_by_key) != len(completed):
            raise ValueError("authoritative evaluator log contains duplicate episode identities")
        for item in completed:
            key = str(item.get("trajectory_id", ""))
            recovered = candidate_by_key.get(key)
            if recovered is None:
                raise ValueError(f"authoritative evaluator log lacks {key}")
            for field in ("step_count", "termination_reason"):
                if field in item and item[field] != recovered.get(field):
                    raise ValueError(f"authoritative evaluator log disagrees on {key} {field}")
            previous = recovered_by_key.get(key)
            if previous is not None and previous["official_metrics"] != recovered["official_metrics"]:
                raise ValueError(f"authoritative evaluator logs disagree on metrics for {key}")
            recovered_by_key[key] = recovered

    result = []
    for item in completed:
        key = str(item.get("trajectory_id", ""))
        recovered = recovered_by_key[key]
        merged = dict(item)
        existing = merged.get("official_metrics")
        if isinstance(existing, dict) and existing != recovered["official_metrics"]:
            raise ValueError(f"per-episode metrics disagree with evaluator log for {key}")
        merged["official_metrics"] = dict(recovered["official_metrics"])
        result.append(merged)
    return result


def _termination(
    *,
    factors: dict[str, str],
    result_reason: str,
    official: dict[str, Any],
    client_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_map = {
        "success": "success",
        "stuck": "stuck",
        "timeout": "timeout",
        "exceed_total_max_step": "timeout",
        "collision": "collision",
        "fall": "fall",
        "error": "error",
    }
    reason = reason_map.get(result_reason, "failure")
    last_model_stop = bool(client_rows and client_rows[-1].get("model_stop"))
    if reason == "stuck":
        effective = "stuck"
    elif reason == "timeout":
        effective = "timeout"
    elif reason in {"collision", "fall"}:
        effective = "safety"
    elif reason == "error":
        effective = "error"
    elif last_model_stop:
        effective = "model_stop"
    else:
        effective = "environment"

    oracle = factors["termination_mode"] == "oracle_termination"
    terminated_by_oracle = bool(oracle and reason == "success" and not last_model_stop)
    if terminated_by_oracle:
        effective = "oracle_distance"
        stability = "missing"
        oracle_reason = "stop_unstable_missing"
    elif last_model_stop:
        stability = "stable" if reason == "success" else "premature"
        oracle_reason = None
    else:
        stability = "not_observed"
        oracle_reason = None
    return {
        "configured_policy": factors["termination_mode"],
        "effective_source": effective,
        "reason": reason,
        "stop_stability": stability,
        "model_stop_observed": last_model_stop,
        "model_stop_suppressed_count": 0,
        "oracle_mode_enabled": oracle,
        "terminated_by_oracle": terminated_by_oracle,
        "oracle_reason": oracle_reason,
        "oracle_threshold_m": 2.5 if oracle else None,
        "oracle_threshold_source": "frozen_evaluator_success_radius" if oracle else None,
        "oracle_distance_source": "evaluator_ground_truth" if oracle else None,
        "distance_at_termination_m": float(official["ne_m"]) if oracle else None,
    }


def build_records(
    result_dir: Path,
    variant_id: str,
    runner_commit_sha: str,
    *,
    repository_root: Path = ROOT,
    matrix_path: Path = MATRIX_PATH,
    write_sources: bool = True,
) -> list[dict[str, Any]]:
    result_dir = result_dir.resolve()
    repository_root = repository_root.resolve()
    if variant_id not in VARIANT_IDS:
        raise ValueError(f"unknown variant: {variant_id}")
    if len(runner_commit_sha) != 40 or any(c not in "0123456789abcdef" for c in runner_commit_sha):
        raise ValueError("runner commit must be a full lowercase Git SHA")

    matrix = load_matrix(matrix_path)
    config = resolve_variant_configs(matrix)[variant_id]
    factors = config["factors"]
    archive = result_dir / "migration_inputs.tgz"
    dataset_bytes = _archive_member(archive, DATASET_MEMBER, maximum_bytes=64 * 1024 * 1024)
    archived_config = json.loads(
        _archive_member(
            archive,
            f"ablation_configs/variants/{variant_id}.json",
            maximum_bytes=1024 * 1024,
        ).decode("utf-8")
    )
    if canonical_sha256(archived_config) != canonical_sha256(config):
        raise ValueError("archived variant config does not match the frozen matrix")
    with gzip.GzipFile(fileobj=io.BytesIO(dataset_bytes), mode="rb") as stream:
        dataset = json.loads(stream.read().decode("utf-8"))
    episodes = dataset.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 20:
        raise ValueError("ablation dataset must contain exactly 20 episodes")

    per_episode = load_json(result_dir / "isaac-run/per_episode.json")
    completed = per_episode.get("episodes")
    if not isinstance(completed, list) or len(completed) != 20:
        raise ValueError("per_episode must contain exactly 20 completed episodes")
    if not all(isinstance(item, dict) for item in completed):
        raise ValueError("per_episode entries must be objects")
    completed = _recover_authoritative_metrics(result_dir, completed)
    completed_by_key = {str(item.get("trajectory_id", "")): item for item in completed}
    completed_by_episode = {
        str(item.get("trajectory_id", "")).rsplit("_", 1)[-1]: item
        for item in completed
    }

    client = _index_by_episode(load_jsonl(result_dir / "isaac-run/client_records.jsonl"))
    client_summary = load_json(result_dir / "isaac-run/client_summary.json")
    active = _index_by_episode(load_jsonl(result_dir / "dgx-run/active_records.jsonl"))
    history = _index_by_episode(load_jsonl(result_dir / "model-run/model_recovery_audit.jsonl"))
    recovery = _index_by_episode(load_jsonl(result_dir / "dgx-run/recovery_records.jsonl"))
    controller_rows = _index_by_episode(load_jsonl(result_dir / "dgx-run/controller_records.jsonl"))
    controller = load_json(result_dir / "dgx-run/controller_summary.json")
    relay = load_json(result_dir / "dgx-run/warn_only_relay_summary.json")

    identities = []
    for ordinal, raw in enumerate(episodes):
        if not isinstance(raw, dict):
            raise ValueError("dataset episode must be an object")
        episode_id, episode_key = _dataset_identity(raw)
        seed = int(raw["seed"]) if isinstance(raw.get("seed"), int) else _seed(episode_key)
        identities.append(
            {"episode_id": episode_id, "episode_key": episode_key, "ordinal": ordinal, "seed": seed}
        )
    episode_manifest_sha = canonical_sha256(identities)
    dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()
    run_id = result_dir.name
    try:
        result_rel = result_dir.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError("result directory must be inside the repository") from exc

    records = []
    snapshots = []
    for identity in identities:
        episode_id = identity["episode_id"]
        episode_key = identity["episode_key"]
        completed_row = completed_by_key.get(episode_key) or completed_by_episode.get(episode_id)
        if completed_row is None or not isinstance(completed_row.get("official_metrics"), dict):
            raise ValueError(f"missing authoritative metrics for {episode_key}")
        official = dict(completed_row["official_metrics"])
        clients = client.get(episode_id, [])
        active_rows = active.get(episode_id, [])
        transforms = [row for row in active_rows if row.get("event") == "ablation_transform"]
        audit = history.get(episode_id, [])
        transform_sequences = {
            int(row["sequence_id"])
            for row in transforms
            if isinstance(row.get("sequence_id"), int)
        }
        active_sequences = {
            int(row["sequence_id"])
            for row in active_rows
            if isinstance(row.get("sequence_id"), int) and isinstance(row.get("action_source"), int)
        }
        consumed_sequences = {
            int(row["sequence_id"])
            for row in audit
            if row.get("event") == "ablation_history_frame_consumed"
            and isinstance(row.get("sequence_id"), int)
        }
        correlated_client_gap = not clients and bool(
            transform_sequences & active_sequences & consumed_sequences
        )
        source_timeout_gap = (
            not clients
            and not active_rows
            and bool(consumed_sequences)
            and any(row.get("event") == "ablation_history_generation_reset" for row in audit)
            and str(completed_row.get("termination_reason", ""))
            in {"exceed_total_max_step", "not_reach_goal", "timeout"}
            and int(official.get("sr", -1)) == 0
            and client_summary.get("status") == "FINISHED"
            and int(client_summary.get("step_count", 0)) > 0
        )
        if not clients and not correlated_client_gap and not source_timeout_gap:
            raise ValueError(f"missing uncorroborated client records for {episode_key}")
        recovery_rows = [row for row in recovery.get(episode_id, []) if row.get("event") == "recovery"]
        controls = [row for row in controller_rows.get(episode_id, []) if row.get("identity_valid") is True]

        system1_raw = [row for row in clients if int(row.get("action_source", 0)) in {2, 3}]
        system2_raw = [row for row in clients if int(row.get("action_source", 0)) == 1]
        if factors["system_mode"] == "oracle_high_level_system1":
            system1_count, system2_count = len(system1_raw), 0
            oracle_high_count = sum(int(row.get("original_action_source", 0)) == 1 for row in transforms)
            oracle_local_count = 0
            system1_latency = [float(row["inference_latency_sec"]) * 1000.0 for row in system1_raw]
            system2_latency: list[float] = []
        elif factors["system_mode"] == "system2_oracle_local_path":
            system1_count, system2_count = 0, len(system2_raw)
            oracle_high_count = 0
            oracle_local_count = len(transforms)
            system1_latency = []
            system2_latency = [float(row["inference_latency_sec"]) * 1000.0 for row in system2_raw]
        else:
            system1_count, system2_count = len(system1_raw), len(system2_raw)
            oracle_high_count = oracle_local_count = 0
            system1_latency = [float(row["inference_latency_sec"]) * 1000.0 for row in system1_raw]
            system2_latency = [float(row["inference_latency_sec"]) * 1000.0 for row in system2_raw]
        end_to_end = [
            (float(row["inference_latency_sec"]) + float(row.get("nav2_resolution_latency_sec", 0.0))) * 1000.0
            for row in clients
        ]
        history_frames = sum(row.get("event") == "ablation_history_frame_consumed" for row in audit)
        if factors["history_mode"] == "off":
            history_frames = 0
        history_resets = sum(
            row.get("event") in {"ablation_history_episode_initialized", "ablation_history_generation_reset"}
            for row in audit
        )
        bounded_violations = sum(
            abs(float(row.get("desired_linear_x", 0.0))) > 0.25 + 1e-9
            or abs(float(row.get("desired_angular_z", 0.0))) > 1.0 + 1e-9
            for row in controls
        )
        stale_motion = sum(
            row.get("command_fresh") is False and row.get("motion_enabled") is True
            for row in controls
        )
        reset_contamination = sum(
            row.get("identity_valid") is False and row.get("motion_enabled") is True
            for row in controller_rows.get(episode_id, [])
        )
        termination = _termination(
            factors=factors,
            result_reason=str(completed_row.get("termination_reason", "")),
            official=official,
            client_rows=clients,
        )
        record = {
            "schema_version": 1,
            "matrix_id": matrix["matrix_id"],
            "matrix_sha256": canonical_sha256(matrix),
            "variant_id": variant_id,
            "variant_config_sha256": canonical_sha256(config),
            "dataset_sha256": dataset_sha,
            "episode_id": episode_key,
            "episode_ordinal": identity["ordinal"],
            "seed": identity["seed"],
            "evidence_kind": "completion_sim",
            "runtime_policy": "completion_sim",
            "execution_target": "isaac_simulation",
            "official_metrics": {
                "sr": int(official["sr"]),
                "os": int(official["os"]),
                "spl": float(official["spl"]),
                "ne_m": float(official["ne_m"]),
            },
            "diagnostics": {
                "stuck": termination["reason"] == "stuck",
                "latency_ms": {
                    "system1": latency_summary(system1_latency),
                    "system2": latency_summary(system2_latency),
                    "end_to_end": latency_summary(end_to_end),
                },
            },
            "termination": termination,
            "activation": {
                "system1_count": system1_count,
                "system2_count": system2_count,
                "oracle_high_level_count": oracle_high_count,
                "oracle_local_path_count": oracle_local_count,
                "trajectory_transform_count": sum(
                    factors["trajectory_mode"] != "full_trajectory"
                    and int(row.get("original_point_count", 0)) > 0
                    for row in transforms
                ),
                "observed_trajectory_mode": factors["trajectory_mode"],
                "history_frame_count": history_frames,
                "history_reset_count": history_resets,
                "recovery_trigger_count": len(recovery_rows),
                "visual_frame_count": len(clients) if clients else len(consumed_sequences),
                "observed_view_mode": factors["view_mode"],
                "visual_input_only": True,
            },
            "safety": {
                "collision_count": int(any(row.get("physical_collision") is True for row in controls)),
                "fall_count": int(any(row.get("fallen") is True for row in controls)),
                "stale_command_count": stale_motion,
                "reset_contamination_count": reset_contamination,
                "bounded_velocity_violation_count": bounded_violations,
                "simulation_estop_available": relay.get("status") == "PASS"
                and "simulation_estop" in (relay.get("reason_counts") or {}),
            },
            "warnings": (
                ["recorder_partial_frame", "source_timeout_extended"]
                if source_timeout_gap
                else ["recorder_partial_frame"]
                if correlated_client_gap
                else []
            ),
            "provenance": {
                "run_id": run_id,
                "runner_commit_sha": runner_commit_sha,
                "episode_manifest_sha256": episode_manifest_sha,
                "source_result_path": f"{result_rel}/episode-sources/{identity['ordinal']:03d}.json",
            },
        }
        # Fail closed on global invariants before accepting any per-episode row.
        if int(controller.get("stale_motion_execution_count", -1)) != 0:
            raise ValueError("controller reports stale motion execution")
        if float(relay.get("maximum_abs_linear_mps", 999.0)) > 0.25 + 1e-9:
            raise ValueError("relay exceeded the frozen linear bound")
        if float(relay.get("maximum_abs_angular_rps", 999.0)) > 1.0 + 1e-9:
            raise ValueError("relay exceeded the frozen angular bound")
        validate_episode_record(record, matrix)
        snapshot = {
            "schema_version": 1,
            "variant_id": variant_id,
            "episode_id": episode_key,
            "per_episode": completed_row,
            "evidence_counts": {
                "client": len(clients),
                "ablation_transform": len(transforms),
                "history_audit": len(audit),
                "recovery": len(recovery_rows),
                "controller": len(controls),
            },
        }
        snapshots.append((identity["ordinal"], snapshot))
        records.append(record)
    if write_sources:
        source_dir = result_dir / "episode-sources"
        source_dir.mkdir(exist_ok=False)
        for ordinal, snapshot in snapshots:
            (source_dir / f"{ordinal:03d}.json").write_text(
                json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--variant-id", choices=VARIANT_IDS, required=True)
    parser.add_argument("--runner-commit-sha", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.result_dir / "episode_records.jsonl"
    if output.exists():
        raise FileExistsError(output)
    records = build_records(args.result_dir, args.variant_id, args.runner_commit_sha)
    output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "variant_id": args.variant_id, "episode_count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
