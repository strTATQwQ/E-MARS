#!/usr/bin/env python3
"""Summarize the frozen retrospective T5 paired-10 experiment.

The comparison unit is ``(pair_set, episode_key)``.  An episode is never
paired across physical lanes: each Lane runs its own frozen ten episodes once
with InternVLA alone and once with the bounded Step3 timeout advisor.

Online failures are ordinary evidence and produce an ``INCOMPLETE`` report.
Contract violations (duplicate arms/episodes, a changed episode set, or a
cross-Lane pairing) fail closed.  Wall-latency percentiles are recomputed from
raw unified-timeline samples; this tool deliberately never combines already
aggregated p95 values.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
ARMS = ("internvla_only", "internvla_step3")
OUTCOMES = ("both_success", "only_internvla", "only_step3", "neither")
METRICS = ("SR", "SPL", "nDTW", "NE", "stuck", "RTF", "command_age_sec")


class PairedAnalysisError(ValueError):
    """The authored pairing contract or materialized evidence is malformed."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_bad_json)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PairedAnalysisError(f"invalid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PairedAnalysisError(f"{path} must contain a JSON object")
    return value


def _bad_json(value: str) -> None:
    raise PairedAnalysisError(f"non-finite JSON constant is forbidden: {value}")


def _rows(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PairedAnalysisError(f"cannot read {path}: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw, parse_constant=_bad_json)
        except json.JSONDecodeError as exc:
            raise PairedAnalysisError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise PairedAnalysisError(f"{path}:{line_number} must contain an object")
        output.append(value)
    return output


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairedAnalysisError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise PairedAnalysisError(f"{label} must be finite")
    return converted


def _optional_number(values: Iterable[tuple[str, Any]], label: str) -> float | None:
    for _name, value in values:
        if value is not None:
            return _finite(value, label)
    return None


def _optional_boolean(values: Iterable[tuple[str, Any]], label: str) -> bool | None:
    for _name, value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            converted = _finite(value, label)
            if converted in {0.0, 1.0}:
                return converted == 1.0
        raise PairedAnalysisError(f"{label} must be boolean or zero/one")
    return None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: Sequence[float], expected: int | None = None) -> dict[str, Any]:
    selected = list(values)
    return {
        "count": len(selected),
        "missing_count": max(0, expected - len(selected)) if expected is not None else None,
        "mean": statistics.fmean(selected) if selected else None,
        "p50": _percentile(selected, 0.50),
        "p95": _percentile(selected, 0.95),
        "max": max(selected) if selected else None,
        "percentile_method": "linear_interpolation",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PairedAnalysisError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise PairedAnalysisError(f"{label} escapes the repository: {value}")
    return path


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_inputs(
    paired_input: Path, repo_root: Path | None
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    raw_input = paired_input.resolve(strict=True)
    summary_path = (
        raw_input / "paired10_execution_summary.json"
        if raw_input.is_dir()
        else raw_input
    )
    if summary_path.name != "paired10_execution_summary.json":
        raise PairedAnalysisError(
            "input must be a paired root or paired10_execution_summary.json"
        )
    summary = _load_object(summary_path)
    paired_root = summary_path.parent.resolve()
    manifest_relative = _safe_relative(summary.get("manifest"), "summary manifest")
    if repo_root is None:
        candidates = [paired_root, *paired_root.parents]
        roots = [candidate for candidate in candidates if (candidate / manifest_relative).is_file()]
        if len(roots) != 1:
            raise PairedAnalysisError(
                "could not identify exactly one repository root from the manifest"
            )
        repository = roots[0].resolve()
    else:
        repository = repo_root.resolve(strict=True)
    manifest_path = (repository / manifest_relative).resolve(strict=True)
    if not _within(manifest_path, repository) or manifest_path.is_symlink():
        raise PairedAnalysisError("manifest is outside the repository or is a symlink")
    manifest = _load_object(manifest_path)
    return repository, paired_root, manifest_path, summary, manifest


def _authored_slots(manifest: Mapping[str, Any]) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, list[str]]]:
    classification = manifest.get("evidence_classification")
    if (
        manifest.get("status") != "FROZEN_FOR_EXECUTION"
        or not isinstance(classification, dict)
        or classification.get("held_out") is not False
    ):
        raise PairedAnalysisError("manifest is not frozen retrospective/non-held-out evidence")
    sets = manifest.get("episode_sets")
    if not isinstance(sets, dict) or set(sets) != {"paired10_a", "paired10_b"}:
        raise PairedAnalysisError("manifest must define exactly paired10_a and paired10_b")
    expected_sets: dict[str, list[str]] = {}
    for pair_set, expected_lane in (("paired10_a", "a"), ("paired10_b", "b")):
        value = sets.get(pair_set)
        keys = value.get("episode_keys") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or value.get("lane") != expected_lane
            or not isinstance(keys, list)
            or len(keys) != 10
            or len(set(keys)) != 10
            or not all(isinstance(key, str) and key for key in keys)
        ):
            raise PairedAnalysisError(f"invalid frozen episode set {pair_set}")
        expected_sets[pair_set] = list(keys)
    if set(expected_sets["paired10_a"]) & set(expected_sets["paired10_b"]):
        raise PairedAnalysisError("the two paired-10 episode sets overlap")

    rounds = manifest.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 2:
        raise PairedAnalysisError("manifest must define exactly two rounds")
    slots: dict[tuple[str, str], dict[str, str]] = {}
    observed_arms: set[tuple[str, str]] = set()
    for round_value in rounds:
        if not isinstance(round_value, dict):
            raise PairedAnalysisError("round entry must be an object")
        round_id = round_value.get("round_id")
        if round_id not in {"round1", "round2"}:
            raise PairedAnalysisError("unexpected round id")
        for lane in ("a", "b"):
            key = (str(round_id), f"lane_{lane}")
            value = round_value.get(f"lane_{lane}")
            if not isinstance(value, dict):
                raise PairedAnalysisError(f"missing authored slot {key}")
            pair_set = value.get("pair_set")
            arm = value.get("evaluation_arm")
            if pair_set not in expected_sets or arm not in ARMS:
                raise PairedAnalysisError(f"invalid authored slot {key}")
            if sets[pair_set].get("lane") != lane:
                raise PairedAnalysisError(f"authored slot {key} crosses physical lanes")
            if (pair_set, str(arm)) in observed_arms:
                raise PairedAnalysisError(f"duplicate authored arm for {pair_set}: {arm}")
            observed_arms.add((pair_set, str(arm)))
            slots[key] = {"lane": lane, "pair_set": str(pair_set), "arm": str(arm)}
    required = {(pair_set, arm) for pair_set in expected_sets for arm in ARMS}
    if observed_arms != required:
        raise PairedAnalysisError("authored rounds do not cover each pair-set arm exactly once")
    return slots, expected_sets


def _normalize_episode_key(raw: Any, expected: Sequence[str], label: str) -> str:
    if not isinstance(raw, (str, int)) or not str(raw):
        raise PairedAnalysisError(f"{label} has no episode identity")
    candidate = str(raw).rsplit("::", 1)[-1]
    if candidate in expected:
        return candidate
    suffix_matches = [key for key in expected if key.rsplit("_", 1)[-1] == candidate.rsplit("_", 1)[-1]]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise PairedAnalysisError(f"{label} episode {raw!r} is outside its frozen pair set")


def _episode_metrics(
    row: Mapping[str, Any], key: str, physics_hz: float | None
) -> dict[str, Any]:
    official = row.get("official_metrics")
    official = official if isinstance(official, dict) else {}
    success = _optional_boolean(
        (("official.sr", official.get("sr")), ("success", row.get("success"))),
        f"{key} Success",
    )
    spl = _optional_number(
        (("official.spl", official.get("spl")), ("SPL", row.get("SPL")), ("spl", row.get("spl"))),
        f"{key} SPL",
    )
    ndtw = _optional_number(
        (("official.ndtw", official.get("ndtw")), ("nDTW", row.get("nDTW")), ("ndtw", row.get("ndtw")), ("NDTW", row.get("NDTW"))),
        f"{key} nDTW",
    )
    ne = _optional_number(
        (("official.ne_m", official.get("ne_m")), ("NE", row.get("NE")), ("ne", row.get("ne"))),
        f"{key} NE",
    )
    declared_stuck = _optional_boolean((("stuck", row.get("stuck")),), f"{key} stuck")
    reason = row.get("termination_reason", row.get("reason"))
    stuck = declared_stuck if declared_stuck is not None else (
        str(reason).lower() == "stuck" if reason is not None else None
    )
    wall = _optional_number((("duration_sec", row.get("duration_sec")),), f"{key} wall duration")
    step_count = row.get("step_count")
    rtf: float | None = None
    sim_duration: float | None = None
    if wall is not None and step_count is not None and physics_hz is not None:
        if wall <= 0.0 or isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 0:
            raise PairedAnalysisError(f"{key} has invalid duration/step_count")
        sim_duration = step_count / physics_hz
        rtf = sim_duration / wall
    return {
        "episode_key": key,
        "SR": float(success) if success is not None else None,
        "SPL": spl,
        "nDTW": ndtw,
        "NE": ne,
        "stuck": float(stuck) if stuck is not None else None,
        "RTF": rtf,
        "command_age_sec": None,
        "command_age_sample_count": 0,
        "wall_duration_sec": wall,
        "sim_duration_sec_estimated": sim_duration,
        "rtf_uses_estimated_sim_duration": rtf is not None,
        "termination_reason": reason,
    }


def _physics_hz(run_root: Path) -> float | None:
    evaluator = run_root / "remote" / "x86" / "evaluator"
    matches = sorted(path for path in evaluator.rglob("go2_runtime_audit.jsonl") if path.is_file())
    if not matches:
        return None
    if len(matches) != 1:
        raise PairedAnalysisError(f"expected one go2_runtime_audit.jsonl under {run_root}")
    observed: list[float] = []
    for row in _rows(matches[0]):
        if "physics_hz" in row:
            value = _finite(row["physics_hz"], "physics_hz")
            if value <= 0.0:
                raise PairedAnalysisError("physics_hz must be positive")
            observed.append(value)
    if not observed:
        return None
    if any(not math.isclose(value, observed[0], rel_tol=0.0, abs_tol=1e-9) for value in observed):
        raise PairedAnalysisError("go2 runtime contains inconsistent physics_hz")
    return observed[0]


def _timeline_evidence(
    run_root: Path, expected_keys: Sequence[str]
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, Any]]:
    index_path = run_root / "replay" / "timeline_index.json"
    timeline_path = run_root / "replay" / "timeline.jsonl"
    if not index_path.is_file() and not timeline_path.is_file():
        return {}, {}, {"status": "MISSING", "reason": "timeline_and_index_absent"}
    if not index_path.is_file() or not timeline_path.is_file():
        return {}, {}, {"status": "INCOMPLETE", "reason": "timeline_or_index_absent"}
    index = _load_object(index_path)
    expected_sha = index.get("timeline_sha256")
    actual_sha = _sha256(timeline_path)
    if expected_sha != actual_sha:
        raise PairedAnalysisError(f"timeline SHA does not match its index under {run_root}")
    command_age: dict[str, list[float]] = defaultdict(list)
    latency: dict[str, list[float]] = defaultdict(list)
    event_count = 0
    for event in _rows(timeline_path):
        event_count += 1
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        age = payload.get("command_age_sec")
        if age is not None:
            key = _normalize_episode_key(
                event.get("episode_id"), expected_keys, "timeline command-age event"
            )
            converted = _finite(age, "command_age_sec")
            if converted < 0.0:
                raise PairedAnalysisError("command_age_sec must be nonnegative")
            command_age[key].append(converted)
        raw_latencies = payload.get("wall_latencies_ms")
        if raw_latencies is None:
            continue
        if not isinstance(raw_latencies, dict):
            raise PairedAnalysisError("wall_latencies_ms must be an object")
        for component, raw in raw_latencies.items():
            if not isinstance(component, str) or not component:
                raise PairedAnalysisError("wall latency component must be a string")
            converted = _finite(raw, f"wall latency {component}")
            if converted < 0.0:
                raise PairedAnalysisError("wall latency must be nonnegative")
            latency[component].append(converted)
    if index.get("event_count") != event_count:
        raise PairedAnalysisError(f"timeline event count differs from its index under {run_root}")
    return dict(command_age), dict(latency), {
        "status": "PASS",
        "index": index_path.relative_to(run_root).as_posix(),
        "timeline": timeline_path.relative_to(run_root).as_posix(),
        "timeline_sha256": actual_sha,
        "event_count": event_count,
    }


def _extract_run(
    run_root: Path,
    expected: Mapping[str, str],
    expected_keys: Sequence[str],
    code_sha: str | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[float]]]:
    fast_path = run_root / "fast_lane_summary.json"
    final_path = run_root / "fast_lane_final_summary.json"
    if not fast_path.is_file():
        return ({
            **expected,
            "status": "INCOMPLETE",
            "result_root": str(run_root),
            "missing": ["fast_lane_summary.json"],
        }, {}, {})
    fast = _load_object(fast_path)
    final = _load_object(final_path) if final_path.is_file() else None
    if fast.get("lane") != expected["lane"]:
        raise PairedAnalysisError(f"{run_root} is assigned to the wrong physical Lane")
    if fast.get("pair_set") != expected["pair_set"]:
        raise PairedAnalysisError(f"{run_root} is assigned to the wrong pair_set")
    if fast.get("evaluation_arm") != expected["arm"]:
        raise PairedAnalysisError(f"{run_root} is assigned to the wrong evaluation arm")
    if code_sha is not None and fast.get("code_ref_sha") != code_sha:
        raise PairedAnalysisError(f"{run_root} code SHA differs from paired execution")
    binding = fast.get("input_binding")
    if not isinstance(binding, dict):
        binding_path = run_root / "input_binding.json"
        binding = _load_object(binding_path) if binding_path.is_file() else None
    if isinstance(binding, dict):
        if binding.get("evaluation_arm") != expected["arm"]:
            raise PairedAnalysisError(f"{run_root} input binding arm drifted")
        if binding.get("pair_set") != expected["pair_set"]:
            raise PairedAnalysisError(f"{run_root} input binding pair_set drifted")
        bound_keys = binding.get("execution_episode_keys")
        if bound_keys != list(expected_keys):
            raise PairedAnalysisError(f"{run_root} frozen execution episode set drifted")

    evaluator = run_root / "remote" / "x86" / "evaluator"
    matches = sorted(path for path in evaluator.rglob("per_episode.json") if path.is_file())
    if len(matches) > 1:
        raise PairedAnalysisError(f"expected at most one per_episode.json under {run_root}")
    physics_hz = _physics_hz(run_root)
    episodes: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    if not matches:
        missing.append("remote/x86/evaluator/**/per_episode.json")
    else:
        payload = _load_object(matches[0])
        rows = payload.get("episodes")
        if not isinstance(rows, list):
            raise PairedAnalysisError(f"{matches[0]} has no episode list")
        for ordinal, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise PairedAnalysisError(f"{matches[0]} episode {ordinal} is not an object")
            raw_key = row.get("episode_key", row.get("trajectory_id", row.get("episode_id")))
            key = _normalize_episode_key(raw_key, expected_keys, f"{matches[0]}:{ordinal}")
            if key in episodes:
                raise PairedAnalysisError(f"duplicate episode {key} in {run_root}")
            episodes[key] = _episode_metrics(row, key, physics_hz)
        if not set(episodes).issubset(set(expected_keys)):
            raise PairedAnalysisError(f"{run_root} contains episodes outside the frozen set")

    command_age, latency, timeline = _timeline_evidence(run_root, expected_keys)
    if timeline.get("status") != "PASS":
        missing.append("replay/timeline.jsonl+timeline_index.json")
    for key, values in command_age.items():
        if key not in episodes:
            # A timeline may include bootstrap activity for an episode that did
            # not reach an evaluator finish record.  It remains unpaired and is
            # not silently promoted into an episode result.
            continue
        episodes[key]["command_age_sec"] = statistics.fmean(values) if values else None
        episodes[key]["command_age_sample_count"] = len(values)

    absent_keys = [key for key in expected_keys if key not in episodes]
    if absent_keys:
        missing.append("episode_results:" + ",".join(absent_keys))
    if any(row["RTF"] is None for row in episodes.values()):
        missing.append("episode_rtf")
    if any(row["command_age_sec"] is None for row in episodes.values()):
        missing.append("episode_command_age")
    child_pass = fast.get("status") == "PASS" and (
        final is None or final.get("status") == "PASS"
    )
    complete = child_pass and not missing and len(episodes) == len(expected_keys)
    run_record = {
        **expected,
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "result_root": str(run_root),
        "fast_lane_status": fast.get("status"),
        "fast_lane_final_status": final.get("status") if isinstance(final, dict) else None,
        "code_ref_sha": fast.get("code_ref_sha"),
        "expected_episode_count": len(expected_keys),
        "completed_episode_count": len(episodes),
        "missing_episode_keys": absent_keys,
        "physics_hz": physics_hz,
        "rtf_method": "step_count / physics_hz / evaluator_wall_duration",
        "rtf_sim_duration_is_estimated": True,
        "timeline": timeline,
        "latency_raw_sample_counts": {
            component: len(values) for component, values in sorted(latency.items())
        },
        "missing": missing,
    }
    return run_record, episodes, latency


def _arm_aggregate(rows: Sequence[Mapping[str, Any]], expected: int) -> dict[str, Any]:
    metrics = {
        metric: _stats(
            [float(row[metric]) for row in rows if row.get(metric) is not None], expected
        )
        for metric in METRICS
    }
    successes = metrics["SR"]
    stuck = metrics["stuck"]
    metrics["SR"].update({
        "success_count": int(sum(float(row["SR"]) for row in rows if row.get("SR") is not None)),
        "rate": successes["mean"],
    })
    metrics["stuck"].update({
        "stuck_count": int(sum(float(row["stuck"]) for row in rows if row.get("stuck") is not None)),
        "rate": stuck["mean"],
    })
    ages = sum(int(row.get("command_age_sample_count", 0)) for row in rows)
    metrics["command_age_sec"]["raw_sample_count"] = ages
    wall = [float(row["wall_duration_sec"]) for row in rows if row.get("RTF") is not None]
    sim = [float(row["sim_duration_sec_estimated"]) for row in rows if row.get("RTF") is not None]
    metrics["RTF"]["weighted_total_rtf"] = sum(sim) / sum(wall) if wall and sum(wall) > 0 else None
    return {"expected_episode_count": expected, "observed_episode_count": len(rows), "metrics": metrics}


def _delta_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in METRICS:
        values = [
            float(pair["delta_step3_minus_internvla"][metric])
            for pair in pairs
            if pair["delta_step3_minus_internvla"].get(metric) is not None
        ]
        output[metric] = _stats(values, len(pairs))
    return output


def _outcome_counts(pairs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        **{name: sum(pair.get("outcome") == name for pair in pairs) for name in OUTCOMES},
        "incomplete": sum(pair.get("outcome") is None for pair in pairs),
        "pair_count": len(pairs),
    }


def _latency_report(samples: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, list[float]]] = {"internvla": {}, "step3": {}}
    for component, values in sorted(samples.items()):
        model = component.split(".", 1)[0]
        if model in by_model:
            by_model[model][component] = list(values)
    preferences = {
        "internvla": ("internvla.inference",),
        "step3": (
            "step3.private_trace_response",
            "step3.server_total",
            "step3.end_to_end",
            "step3.advisor_service",
        ),
    }
    output: dict[str, Any] = {}
    for model, components in by_model.items():
        primary = next((name for name in preferences[model] if components.get(name)), None)
        output[model] = {
            "timebase": "wall_monotonic_or_measured_wall_duration",
            "unit": "milliseconds",
            "aggregation_source": "raw replay/timeline.jsonl event samples",
            "aggregated_quantiles_reused": False,
            "primary_response_component": primary,
            "primary_response": _stats(components.get(primary, [])) if primary else _stats([]),
            "components": {
                component: _stats(values) for component, values in sorted(components.items())
            },
        }
    return output


def analyze(paired_input: Path, repo_root: Path | None = None) -> dict[str, Any]:
    repository, paired_root, manifest_path, execution, manifest = _resolve_inputs(
        paired_input, repo_root
    )
    slots, expected_sets = _authored_slots(manifest)
    if execution.get("manifest") != manifest_path.relative_to(repository).as_posix():
        raise PairedAnalysisError("execution summary points at a different manifest")
    code_sha = execution.get("code_ref_sha")
    if code_sha is not None and (
        not isinstance(code_sha, str) or len(code_sha) != 40
    ):
        raise PairedAnalysisError("execution code_ref_sha is invalid")
    lane_results = execution.get("lane_results")
    if not isinstance(lane_results, dict):
        raise PairedAnalysisError("execution summary has no lane_results")

    run_records: list[dict[str, Any]] = []
    episode_table: dict[tuple[str, str, str], dict[str, Any]] = {}
    latency_samples: dict[str, list[float]] = defaultdict(list)
    seen_roots: set[Path] = set()
    for round_id in ("round1", "round2"):
        round_result = lane_results.get(round_id)
        if not isinstance(round_result, dict):
            raise PairedAnalysisError(f"execution summary has no {round_id}")
        for lane_name in ("lane_a", "lane_b"):
            expected = slots[(round_id, lane_name)]
            raw_root = round_result.get(lane_name)
            if raw_root is None:
                if round_result.get("status") == "PASS":
                    raise PairedAnalysisError(f"PASS {round_id} is missing {lane_name}")
                run_records.append({
                    **expected,
                    "round_id": round_id,
                    "status": "NOT_RUN",
                    "result_root": None,
                    "missing": ["child_result_root"],
                })
                continue
            relative = _safe_relative(raw_root, f"{round_id}.{lane_name}")
            run_root = (repository / relative).resolve()
            if not _within(run_root, repository):
                raise PairedAnalysisError("child result root escapes repository")
            if run_root in seen_roots:
                raise PairedAnalysisError("the same child result root is reused by two slots")
            seen_roots.add(run_root)
            if not run_root.is_dir():
                if round_result.get("status") == "PASS":
                    raise PairedAnalysisError(f"PASS child root is absent: {run_root}")
                run_records.append({
                    **expected,
                    "round_id": round_id,
                    "status": "INCOMPLETE",
                    "result_root": str(run_root),
                    "missing": ["child_result_root"],
                })
                continue
            run_record, episodes, run_latency = _extract_run(
                run_root, expected, expected_sets[expected["pair_set"]], code_sha
            )
            run_record["round_id"] = round_id
            run_records.append(run_record)
            for key, row in episodes.items():
                identity = (expected["pair_set"], expected["arm"], key)
                if identity in episode_table:
                    raise PairedAnalysisError(f"duplicate paired episode arm: {identity}")
                episode_table[identity] = row
            for component, values in run_latency.items():
                latency_samples[component].extend(values)

    pair_set_reports: dict[str, Any] = {}
    all_pairs: list[dict[str, Any]] = []
    for pair_set, keys in expected_sets.items():
        lane = "a" if pair_set == "paired10_a" else "b"
        pairs: list[dict[str, Any]] = []
        arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
        for key in keys:
            baseline = episode_table.get((pair_set, "internvla_only", key))
            step3 = episode_table.get((pair_set, "internvla_step3", key))
            if baseline is not None:
                arm_rows["internvla_only"].append(baseline)
            if step3 is not None:
                arm_rows["internvla_step3"].append(step3)
            outcome = None
            if baseline is not None and step3 is not None:
                base_success, step_success = baseline.get("SR"), step3.get("SR")
                if base_success is not None and step_success is not None:
                    outcome = (
                        "both_success" if base_success == 1.0 and step_success == 1.0
                        else "only_internvla" if base_success == 1.0
                        else "only_step3" if step_success == 1.0
                        else "neither"
                    )
            delta = {
                metric: (
                    float(step3[metric]) - float(baseline[metric])
                    if baseline is not None and step3 is not None
                    and baseline.get(metric) is not None and step3.get(metric) is not None
                    else None
                )
                for metric in METRICS
            }
            pair = {
                "pair_set": pair_set,
                "lane": lane,
                "episode_key": key,
                "internvla_only": baseline,
                "internvla_step3": step3,
                "outcome": outcome,
                "delta_step3_minus_internvla": delta,
            }
            pairs.append(pair)
            all_pairs.append(pair)
        pair_set_reports[pair_set] = {
            "lane": lane,
            "episode_keys": keys,
            "outcomes": _outcome_counts(pairs),
            "arm_aggregates": {
                arm: _arm_aggregate(arm_rows[arm], len(keys)) for arm in ARMS
            },
            "paired_delta_step3_minus_internvla": _delta_summary(pairs),
            "pairs": pairs,
        }

    complete = (
        execution.get("status") == "PASS"
        and len(run_records) == 4
        and all(record.get("status") == "COMPLETE" for record in run_records)
        and all(pair.get("internvla_only") is not None and pair.get("internvla_step3") is not None for pair in all_pairs)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "stage": "t5_paired10_retrospective_analysis",
        "run_id": execution.get("run_id"),
        "code_ref_sha": code_sha,
        "paired_root": str(paired_root),
        "manifest": manifest_path.relative_to(repository).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "evidence_classification": {
            "classification": "RETROSPECTIVE_DEVELOPMENT_NOT_HELD_OUT",
            "held_out": False,
            "promotion_evidence": False,
            "disclosure": (
                "All 20 episode keys were used in earlier T3/T5 development or "
                "evaluation; this is paired engineering evidence, not a held-out benchmark."
            ),
            "same_episode_dual_arm_runs_are_not_additional_samples": True,
        },
        "pairing_method": {
            "unit": "pair_set plus episode_key",
            "same_physical_lane_across_arms": True,
            "cross_lane_episode_pairing": False,
            "overall_pooling": "pair-set-stratified then combined",
            "delta_direction": "InternVLA+Step3 minus InternVLA-only",
        },
        "execution_summary_status": execution.get("status"),
        "runs": run_records,
        "pair_sets": pair_set_reports,
        "overall": {
            "unique_episode_count": len(all_pairs),
            "expected_execution_count": 2 * len(all_pairs),
            "observed_execution_count": sum(
                pair.get("internvla_only") is not None
                for pair in all_pairs
            ) + sum(pair.get("internvla_step3") is not None for pair in all_pairs),
            "outcomes": _outcome_counts(all_pairs),
            "paired_delta_step3_minus_internvla": _delta_summary(all_pairs),
        },
        "wall_response_latency": _latency_report(latency_samples),
        "limitations": [
            "RTF uses evaluator step_count / frozen physics_hz as estimated sim duration.",
            "Command-age pairing uses each episode's mean raw controller sample age.",
            "Latency quantiles are computed from raw timeline samples, never from per-run quantiles.",
            "Missing online evidence remains null and counted; it is not imputed.",
        ],
    }


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise PairedAnalysisError(f"output symlink is forbidden: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paired_input", type=Path,
        help="paired result root or paired10_execution_summary.json",
    )
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    default_root = (
        arguments.paired_input
        if arguments.paired_input.is_dir()
        else arguments.paired_input.parent
    )
    output = arguments.output or default_root / "paired10_analysis.json"
    try:
        payload = analyze(arguments.paired_input, arguments.repo_root)
        exit_code = 0 if payload["status"] == "COMPLETE" else 1
    except (OSError, PairedAnalysisError) as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "stage": "t5_paired10_retrospective_analysis",
            "error": str(exc),
        }
        exit_code = 2
    try:
        _write_atomic(output, payload)
    except (OSError, PairedAnalysisError) as exc:
        print(f"ERROR: cannot write analysis: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
