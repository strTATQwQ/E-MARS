#!/usr/bin/env python3
"""Compare shadow-only STOP candidates against oracle termination over A10+B10."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


ORACLE_RADIUS_M = 2.5
OPERATIONAL_METHODS = (
    "internvla_model_stop",
    "step3_arrived_single",
    "step3_arrived_confirmed",
    "step3_task_state_target_found",
    "internvla_and_step3_single",
    "internvla_and_step3_confirmed",
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        values.append(value)
    return values


def _candidate_key(row: dict[str, Any], sequence_field: str) -> tuple[str, int, int] | None:
    try:
        return (
            str(row["episode_id"]),
            int(row.get("reset_generation", 0)),
            int(row[sequence_field]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _probe_key(row: dict[str, Any]) -> tuple[str, int, int] | None:
    """Read an oracle-terminal probe identity from its event payload or token."""

    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    sequence_id = payload.get("trigger_sequence_id", payload.get("sequence_id"))
    if isinstance(sequence_id, bool) or not isinstance(sequence_id, int):
        token = str(row.get("token", ""))
        try:
            sequence_id = int(token.rsplit(":", 1)[-1])
        except ValueError:
            return None
    try:
        return (
            str(row["episode_id"]),
            int(row.get("reset_generation", 0)),
            int(sequence_id),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _temporal_candidate_statistics(
    candidates: Iterable[tuple[str, int, int]],
    active: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    """Compress per-sequence candidates into episode-local contiguous bursts."""

    unique = sorted(set(candidates))
    by_generation: dict[tuple[str, int], list[tuple[str, int, int]]] = defaultdict(list)
    for key in unique:
        by_generation[key[:2]].append(key)

    bursts: list[list[tuple[str, int, int]]] = []
    for keys in by_generation.values():
        current: list[tuple[str, int, int]] = []
        for key in keys:
            if current and key[2] != current[-1][2] + 1:
                bursts.append(current)
                current = []
            current.append(key)
        if current:
            bursts.append(current)

    def classification(key: tuple[str, int, int]) -> str:
        row = active.get(key)
        if not isinstance(row, dict) or not isinstance(
            row.get("oracle_distance_m"), (int, float)
        ):
            return "unresolved"
        return (
            "true_positive"
            if float(row["oracle_distance_m"]) <= ORACLE_RADIUS_M
            else "false_positive"
        )

    burst_rows: list[dict[str, Any]] = []
    per_generation: dict[tuple[str, int], dict[str, Any]] = {}
    for generation, keys in by_generation.items():
        per_generation[generation] = {
            "episode_id": generation[0],
            "reset_generation": generation[1],
            "candidate_frame_count": len(keys),
            "burst_count": 0,
            "onset_sequence_ids": [],
        }
    for burst in bursts:
        labels = [classification(key) for key in burst]
        onset = burst[0]
        row = {
            "episode_id": onset[0],
            "reset_generation": onset[1],
            "onset_sequence_id": onset[2],
            "end_sequence_id": burst[-1][2],
            "candidate_frame_count": len(burst),
            "onset_classification": labels[0],
            "true_positive_frame_count": labels.count("true_positive"),
            "false_positive_frame_count": labels.count("false_positive"),
            "unresolved_frame_count": labels.count("unresolved"),
        }
        burst_rows.append(row)
        generation_row = per_generation[onset[:2]]
        generation_row["burst_count"] += 1
        generation_row["onset_sequence_ids"].append(onset[2])

    onset_labels = [row["onset_classification"] for row in burst_rows]
    return {
        "candidate_count_unit": "unique_sequence_frames_not_independent_judgments",
        "candidate_frame_count": len(unique),
        "candidate_episode_count": len({key[0] for key in unique}),
        "candidate_episode_generation_count": len(by_generation),
        "candidate_burst_count": len(burst_rows),
        "candidate_onset_count": len(burst_rows),
        "true_positive_onset_count": onset_labels.count("true_positive"),
        "false_positive_onset_count": onset_labels.count("false_positive"),
        "unresolved_onset_count": onset_labels.count("unresolved"),
        "max_burst_length_frames": max(
            (row["candidate_frame_count"] for row in burst_rows), default=0
        ),
        "candidate_bursts": burst_rows,
        "episode_generation_statistics": [
            per_generation[key] for key in sorted(per_generation)
        ],
    }


def _summarize_oracle_terminal_probes(
    probes: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failure_reason_counts: Counter[str] = Counter()
    for key in sorted(probes):
        state = probes[key]
        confirmed = bool(state.get("confirmed"))
        reasons = set(str(value) for value in state.get("failure_reasons", set()))
        if not confirmed:
            if state.get("unavailable"):
                reasons.add("oracle_terminal_shadow_unavailable")
            elif state.get("started") and not state.get("completed"):
                reasons.add("probe_incomplete_after_start")
            elif not state.get("started") and not state.get("completed"):
                reasons.add("probe_not_started")
            elif not reasons:
                reasons.add("not_confirmed")
            failure_reason_counts.update(reasons)
        records.append(
            {
                "episode_id": key[0],
                "reset_generation": key[1],
                "sequence_id": key[2],
                "started": bool(state.get("started")),
                "completed": bool(state.get("completed")),
                "confirmed": confirmed,
                "failed": not confirmed,
                "failure_reasons": sorted(reasons) if not confirmed else [],
            }
        )
    return {
        "attempted_count": len(records),
        "started_count": sum(row["started"] for row in records),
        "completed_count": sum(row["completed"] for row in records),
        "confirmed_count": sum(row["confirmed"] for row in records),
        "failed_count": sum(row["failed"] for row in records),
        "attempted_episode_count": len({row["episode_id"] for row in records}),
        "completed_episode_count": len(
            {row["episode_id"] for row in records if row["completed"]}
        ),
        "confirmed_episode_count": len(
            {row["episode_id"] for row in records if row["confirmed"]}
        ),
        "failed_episode_count": len(
            {row["episode_id"] for row in records if row["failed"]}
        ),
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "probe_records": records,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _per_episode(root: Path) -> dict[str, dict[str, Any]]:
    paths = list((root / "remote" / "x86" / "evaluator").glob("*/per_episode.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one per_episode.json below {root}, found {len(paths)}")
    payload = _object(paths[0])
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise ValueError(f"episodes missing from {paths[0]}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("trajectory_id"), str):
            raise ValueError(f"invalid episode row in {paths[0]}")
        result[row["trajectory_id"]] = row
    return result


def _collect_lane(root: Path, expected_lane: str) -> dict[str, Any]:
    root = root.resolve()
    final = _object(root / "fast_lane_final_summary.json")
    binding = _object(root / "input_binding.json")
    manifest_path = root / "remote" / "x86" / "ordered_episode_manifest.json"
    manifest = _object(manifest_path)
    keys = manifest.get("ordered_episode_keys")
    if not isinstance(keys, list) or len(keys) != 10 or len(set(keys)) != 10:
        raise ValueError(f"Lane {expected_lane} does not contain exactly 10 unique episodes")
    if binding.get("lane") != expected_lane:
        raise ValueError(f"Lane binding mismatch under {root}")
    if binding.get("candidate_profile") != "recovery_a":
        raise ValueError(f"Lane {expected_lane} is not the frozen WP-03 recovery_a profile")
    if binding.get("isaac_sensor_profile") != "dual_lane_wp03_stop_shadow":
        raise ValueError(f"Lane {expected_lane} is not a STOP-shadow run")
    if final.get("status") != "PASS":
        raise ValueError(f"Lane {expected_lane} final summary is not PASS")

    active_rows = _jsonl(root / "remote" / "dgx" / "onboard" / "active_records.jsonl")
    active: dict[tuple[str, int, int], dict[str, Any]] = {}
    candidates: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    oracle_terminal: dict[str, tuple[str, int, int]] = {}
    for row in active_rows:
        if row.get("event") != "ablation_transform":
            continue
        key = _candidate_key(row, "sequence_id")
        if key is None:
            continue
        active[key] = row
        distance = row.get("oracle_distance_m")
        if isinstance(distance, (int, float)) and float(distance) <= ORACLE_RADIUS_M:
            oracle_terminal.setdefault(key[0], key)
        if row.get("original_model_stop") is True:
            candidates["internvla_model_stop"].add(key)

    advice_rows = _jsonl(
        root / "remote" / "x86" / "evaluator" / "step3_timeout_advice.jsonl"
    )
    for row in advice_rows:
        if row.get("status") != "ARRIVED":
            continue
        key = _candidate_key(row, "trigger_sequence_id")
        if key is not None:
            candidates["step3_arrived_single"].add(key)

    task_rows = _jsonl(
        root / "remote" / "x86" / "evaluator" / "task_state" / "events.jsonl"
    )
    for row in task_rows:
        if row.get("event") != "task_state_updated" or row.get("final_arrival_candidate") is not True:
            continue
        key = _candidate_key(row, "trigger_sequence_id")
        if key is not None:
            candidates["step3_task_state_target_found"].add(key)

    gate_rows = _jsonl(
        root / "remote" / "dgx" / "client" / "motion_observation_gate_records.jsonl"
    )
    oracle_probe_confirmed: set[tuple[str, int, int]] = set()
    oracle_terminal_probes: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in gate_rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        if row.get("event") == "stop_shadow_step3_arrival_confirmed":
            candidate = {**row, "trigger_sequence_id": payload.get("trigger_sequence_id")}
            key = _candidate_key(candidate, "trigger_sequence_id")
            if key is not None:
                candidates["step3_arrived_confirmed"].add(key)
        elif (
            row.get("event") == "step3_oracle_terminal_shadow_completed"
            and payload.get("confirmed") is True
        ):
            token = str(row.get("token", ""))
            try:
                sequence_id = int(token.rsplit(":", 1)[-1])
            except ValueError:
                continue
            key = _candidate_key({**row, "sequence_id": sequence_id}, "sequence_id")
            if key is not None:
                oracle_probe_confirmed.add(key)
        event = str(row.get("event", ""))
        token = str(row.get("token", ""))
        if event == "stop_shadow_oracle_reference" or event.startswith(
            "step3_oracle_terminal_shadow_"
        ) or (
            event == "step3_arrival_not_confirmed"
            and ":oracle-terminal-shadow:" in token
        ):
            key = _probe_key(row)
            if key is None:
                continue
            state = oracle_terminal_probes.setdefault(
                key,
                {
                    "started": False,
                    "completed": False,
                    "confirmed": False,
                    "unavailable": False,
                    "failure_reasons": set(),
                },
            )
            if event == "step3_oracle_terminal_shadow_started":
                state["started"] = True
            elif event == "step3_oracle_terminal_shadow_completed":
                state["completed"] = True
                state["confirmed"] = payload.get("confirmed") is True
            elif event == "step3_oracle_terminal_shadow_unavailable":
                state["unavailable"] = True
            elif event == "step3_arrival_not_confirmed":
                reason = payload.get("reason")
                if isinstance(reason, str) and reason:
                    state["failure_reasons"].add(reason)

    candidates["internvla_and_step3_single"] = (
        candidates["internvla_model_stop"] & candidates["step3_arrived_single"]
    )
    candidates["internvla_and_step3_confirmed"] = (
        candidates["internvla_model_stop"] & candidates["step3_arrived_confirmed"]
    )
    episode_rows = _per_episode(root)
    if set(episode_rows) != set(keys):
        raise ValueError(f"Lane {expected_lane} per-episode rows do not match its frozen split")
    return {
        "lane": expected_lane,
        "root": str(root),
        "code_ref_sha": binding.get("code_ref_sha"),
        "binding": binding,
        "final_status": final.get("status"),
        "manifest_sha256": _sha256(manifest_path),
        "episode_keys": keys,
        "episode_rows": episode_rows,
        "active": active,
        "oracle_terminal": oracle_terminal,
        "candidates": candidates,
        "oracle_probe_confirmed": oracle_probe_confirmed,
        "oracle_terminal_probes": oracle_terminal_probes,
    }


def _score_method(
    name: str,
    candidates: Iterable[tuple[str, int, int]],
    active: dict[tuple[str, int, int], dict[str, Any]],
    oracle_terminal: dict[str, tuple[str, int, int]],
) -> dict[str, Any]:
    unique = sorted(set(candidates))
    resolved = [key for key in unique if key in active]
    unresolved = [key for key in unique if key not in active]
    true_candidates = [
        key for key in resolved if float(active[key]["oracle_distance_m"]) <= ORACLE_RADIUS_M
    ]
    false_candidates = [
        key for key in resolved if float(active[key]["oracle_distance_m"]) > ORACLE_RADIUS_M
    ]
    true_episodes = sorted({key[0] for key in true_candidates})
    false_episodes = sorted({key[0] for key in false_candidates})
    oracle_episodes = sorted(oracle_terminal)
    missed_episodes = sorted(set(oracle_episodes) - set(true_episodes))
    lead_sequences = [
        key[2] - oracle_terminal[key[0]][2]
        for key in resolved
        if key[0] in oracle_terminal
    ]
    precision = len(true_candidates) / len(resolved) if resolved else 0.0
    recall = len(true_episodes) / len(oracle_episodes) if oracle_episodes else 0.0
    return {
        "method": name,
        "candidate_count": len(unique),
        **_temporal_candidate_statistics(unique, active),
        "resolved_candidate_count": len(resolved),
        "unresolved_candidate_count": len(unresolved),
        "true_positive_candidate_count": len(true_candidates),
        "false_positive_candidate_count": len(false_candidates),
        "true_positive_episode_count": len(true_episodes),
        "false_positive_episode_count": len(false_episodes),
        "oracle_success_episode_count": len(oracle_episodes),
        "precision": precision,
        "oracle_success_recall": recall,
        "mean_signed_sequence_delta_to_oracle": (
            statistics.fmean(lead_sequences) if lead_sequences else None
        ),
        "true_positive_episodes": true_episodes,
        "false_positive_episodes": false_episodes,
        "missed_oracle_success_episodes": missed_episodes,
        "unresolved_candidates": [list(key) for key in unresolved],
    }


def analyze(lane_a_root: Path, lane_b_root: Path) -> dict[str, Any]:
    lanes = [_collect_lane(lane_a_root, "a"), _collect_lane(lane_b_root, "b")]
    keys_a, keys_b = set(lanes[0]["episode_keys"]), set(lanes[1]["episode_keys"])
    if keys_a & keys_b or len(keys_a | keys_b) != 20:
        raise ValueError("A10/B10 episode splits are not disjoint or do not total 20")
    code_shas = {lane["code_ref_sha"] for lane in lanes}
    if len(code_shas) != 1:
        raise ValueError("Lane code SHAs differ")

    active: dict[tuple[str, int, int], dict[str, Any]] = {}
    oracle_terminal: dict[str, tuple[str, int, int]] = {}
    candidates: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    oracle_probe: set[tuple[str, int, int]] = set()
    oracle_terminal_probes: dict[tuple[str, int, int], dict[str, Any]] = {}
    episode_rows: dict[str, dict[str, Any]] = {}
    for lane in lanes:
        active.update(lane["active"])
        oracle_terminal.update(lane["oracle_terminal"])
        episode_rows.update(lane["episode_rows"])
        oracle_probe.update(lane["oracle_probe_confirmed"])
        oracle_terminal_probes.update(lane["oracle_terminal_probes"])
        for name, values in lane["candidates"].items():
            candidates[name].update(values)

    scores = [
        _score_method(name, candidates[name], active, oracle_terminal)
        for name in OPERATIONAL_METHODS
    ]
    ranking = sorted(
        scores,
        key=lambda item: (
            -item["true_positive_episode_count"],
            item["false_positive_candidate_count"],
            item["unresolved_candidate_count"],
            -(item["precision"]),
            item["method"],
        ),
    )
    best = ranking[0]
    ready = best["true_positive_episode_count"] > 0 and best["false_positive_candidate_count"] == 0
    oracle_probe_score = _score_method(
        "step3_oracle_terminal_probe", oracle_probe, active, oracle_terminal
    )
    oracle_probe_observability = _summarize_oracle_terminal_probes(
        oracle_terminal_probes
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "experiment": "wp03_oracle_stop_shadow_pilot20",
        "decision_authority": "oracle_termination_only",
        "oracle_radius_m": ORACLE_RADIUS_M,
        "code_ref_sha": next(iter(code_shas)),
        "episode_count": 20,
        "oracle_navigation_success_count": len(oracle_terminal),
        "oracle_navigation_success_rate": len(oracle_terminal) / 20.0,
        "evaluator_success_count": sum(row.get("success") is True for row in episode_rows.values()),
        "lanes": {
            lane["lane"]: {
                "result_root": lane["root"],
                "final_status": lane["final_status"],
                "episode_keys": lane["episode_keys"],
                "manifest_sha256": lane["manifest_sha256"],
            }
            for lane in lanes
        },
        "operational_method_scores": scores,
        "ranking": [item["method"] for item in ranking],
        "best_operational_method": best["method"] if ready else None,
        "promotion_recommendation": "PROMISING_SHADOW_METHOD" if ready else "NO_SHADOW_METHOD_READY",
        "oracle_triggered_probe": {
            **oracle_probe_score,
            **oracle_probe_observability,
            "ranking_eligible": False,
            "reason": "queried only after oracle success, so it is diagnostic rather than deployable",
            "candidate_count_semantics": "confirmed probe outputs only",
        },
        "notes": [
            "All candidate STOP outputs were shadow-only and never controlled termination.",
            "Oracle success is the reference label and is not ranked as a candidate.",
            "Navigation success and STOP-classifier quality are reported separately.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-a-result", required=True, type=Path)
    parser.add_argument("--lane-b-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = analyze(args.lane_a_result, args.lane_b_result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"STOP shadow analysis failed: {error}")
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
