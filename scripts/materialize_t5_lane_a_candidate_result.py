#!/usr/bin/env python3
"""Materialize one exact Lane-A fast-lane run for candidate analysis.

The online runner intentionally keeps raw evidence separate.  This tool is the
small, offline bridge into ``analyze_t5_lane_a_candidates.py``.  It copies only
explicit per-episode metrics and fails closed when a metric or provenance seal
is absent; aggregate run metrics are never distributed across episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from analyze_t5_lane_a_candidates import (
    canonical_json_sha256,
    code_bundle_sha256,
)
from resolve_t5_lane_a_candidate import MANIFEST_RELATIVE, resolve as resolve_candidate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "configs" / "internnav_t5" / "lane_a_candidates" / "manifest.json"
)
PROFILE_COUNTS = {"screen1": 1, "screen3": 3, "fixed5": 5}
BLOCKED_FAMILY = "trajectory_horizon_refresh"
BLOCKING_CANDIDATE = "a0_action_gate_only"


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _git_sha(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a full lowercase Git SHA")
    return text


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _binary(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    numeric = _finite(value, label)
    if numeric not in {0.0, 1.0}:
        raise ValueError(f"{label} must be boolean or 0/1")
    return bool(numeric)


def _require_pass(value: dict[str, Any], label: str) -> None:
    checks = value.get("checks")
    if value.get("status") != "PASS":
        raise ValueError(f"{label} did not PASS")
    if not isinstance(checks, dict) or not checks or not all(
        item is True for item in checks.values()
    ):
        raise ValueError(f"{label} does not contain an all-true check set")


def _inside(root: Path, path: Path, label: str) -> Path:
    root = root.resolve()
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the fast-lane result root") from error
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _manifest_context(manifest_path: Path) -> tuple[Path, dict[str, Any], str, str]:
    manifest_path = manifest_path.resolve()
    manifest = _load_object(manifest_path, "candidate manifest")
    if manifest.get("schema_version") != 1 or manifest.get("lane") != "a":
        raise ValueError("candidate manifest identity is invalid")
    try:
        project_root = manifest_path.parents[3]
    except IndexError as error:
        raise ValueError("candidate manifest path has no project root") from error
    if manifest_path != (project_root / MANIFEST_RELATIVE).resolve():
        raise ValueError("candidate manifest must use the registered project path")
    manifest_sha256 = canonical_json_sha256(manifest)
    provenance = manifest.get("provenance_contract")
    if not isinstance(provenance, dict):
        raise ValueError("candidate manifest is missing provenance_contract")
    paths = provenance.get("code_paths")
    if not isinstance(paths, list) or not paths or not all(
        isinstance(item, str) and item for item in paths
    ):
        raise ValueError("candidate code bundle paths are invalid")
    actual_code_sha256 = code_bundle_sha256(project_root, paths)
    expected_code_sha256 = _sha256(
        provenance.get("code_bundle_sha256"), "manifest code_bundle_sha256"
    )
    if actual_code_sha256 != expected_code_sha256:
        raise ValueError("candidate code bundle changed after manifest registration")
    return project_root, manifest, manifest_sha256, actual_code_sha256


def _metric_candidates(record: dict[str, Any], names: Iterable[str]) -> list[Any]:
    values = [record[name] for name in names if name in record]
    official = record.get("official_metrics")
    if isinstance(official, dict):
        values.extend(official[name] for name in names if name in official)
    return values


def _one_number(record: dict[str, Any], label: str, names: Iterable[str]) -> float:
    values = _metric_candidates(record, names)
    if not values:
        raise ValueError(f"episode record is missing explicit {label}")
    converted = [_finite(value, label) for value in values]
    if any(value != converted[0] for value in converted[1:]):
        raise ValueError(f"episode record has conflicting {label} values")
    return converted[0]


def _one_boolean(record: dict[str, Any], label: str, names: Iterable[str]) -> bool:
    values = _metric_candidates(record, names)
    if not values:
        raise ValueError(f"episode record is missing explicit {label}")
    converted = [_binary(value, label) for value in values]
    if any(value != converted[0] for value in converted[1:]):
        raise ValueError(f"episode record has conflicting {label} values")
    return converted[0]


def _episode_key(record: dict[str, Any]) -> str:
    observed = []
    for name in ("episode_key", "trajectory_id"):
        if name in record:
            value = str(record[name]).strip()
            if value:
                observed.append(value)
    if not observed:
        raise ValueError("episode record lacks an exact episode_key/trajectory_id")
    if any(value != observed[0] for value in observed[1:]):
        raise ValueError("episode record identity fields disagree")
    return observed[0]


def _stuck(record: dict[str, Any]) -> tuple[bool, str]:
    declared = _metric_candidates(record, ("stuck",))
    reason_present = "termination_reason" in record
    reason = str(record.get("termination_reason", "")).strip().lower()
    if not declared and not reason_present:
        raise ValueError("episode record is missing explicit stuck/termination_reason")
    values = [_binary(value, "stuck") for value in declared]
    if reason_present:
        values.append(reason == "stuck")
    if any(value != values[0] for value in values[1:]):
        raise ValueError("episode record has conflicting stuck values")
    return values[0], reason


def _materialize_episode(record: dict[str, Any]) -> dict[str, Any]:
    key = _episode_key(record)
    success = _one_boolean(record, "success", ("success", "sr", "SR"))
    stuck, termination_reason = _stuck(record)
    spl = _one_number(record, "SPL", ("SPL", "spl"))
    ndtw = _one_number(record, "nDTW", ("nDTW", "NDTW", "ndtw"))
    ne = _one_number(record, "NE", ("NE", "ne_m"))
    command_age = _one_number(
        record, "command_age_sec", ("command_age_sec", "mean_command_age_sec")
    )
    if not 0.0 <= spl <= 1.0:
        raise ValueError("episode SPL must be inside [0,1]")
    if not 0.0 <= ndtw <= 1.0:
        raise ValueError("episode nDTW must be inside [0,1]")
    if ne < 0.0 or command_age < 0.0:
        raise ValueError("episode NE and command age must be nonnegative")
    result: dict[str, Any] = {
        "episode_key": key,
        "success": success,
        "stuck": stuck,
        "SPL": spl,
        "nDTW": ndtw,
        "NE": ne,
        "command_age_sec": command_age,
        "source_record_sha256": canonical_json_sha256(record),
    }
    if termination_reason:
        result["termination_reason"] = termination_reason
    oracle = _metric_candidates(record, ("oracle_success", "os", "OS"))
    if oracle:
        converted = [_binary(value, "oracle_success") for value in oracle]
        if any(value != converted[0] for value in converted[1:]):
            raise ValueError("episode record has conflicting oracle-success values")
        result["oracle_success"] = converted[0]
    return result


def _find_episode_metrics(result_root: Path) -> Path:
    evaluator = result_root / "remote" / "x86" / "evaluator"
    authoritative = sorted(
        path
        for path in evaluator.rglob("candidate_episode_metrics.json")
        if path.is_file() and not path.is_symlink()
    )
    if len(authoritative) > 1:
        raise ValueError(
            "fast-lane result must contain exactly one explicit per-episode metrics "
            f"artifact; observed={len(authoritative)} candidate artifacts"
        )
    if authoritative:
        return authoritative[0]
    candidates = sorted(
        path
        for path in evaluator.rglob("per_episode.json")
        if path.is_file() and not path.is_symlink()
    )
    if len(candidates) != 1:
        raise ValueError(
            "fast-lane result must contain exactly one explicit per-episode metrics "
            f"artifact; observed={len(candidates)} evaluator artifacts"
        )
    return candidates[0]


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {line_number} is not a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _attach_exact_command_age(
    result_root: Path,
    rows: list[dict[str, Any]],
    expected_keys: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    missing = [
        _episode_key(row)
        for row in rows
        if not _metric_candidates(
            row, ("command_age_sec", "mean_command_age_sec")
        )
    ]
    if not missing:
        return rows, None

    manifest_path = result_root / "remote" / "x86" / "ordered_episode_manifest.json"
    controller_path = (
        result_root / "remote" / "dgx" / "onboard" / "controller_records.jsonl"
    )
    if not manifest_path.is_file() or not controller_path.is_file():
        raise ValueError(
            "episode record is missing explicit command_age_sec and exact "
            "per-update controller evidence is unavailable"
        )
    manifest = _load_object(manifest_path, "ordered episode manifest")
    ordered_keys = manifest.get("ordered_episode_keys")
    ordered_ids = manifest.get("ordered_episode_ids")
    if (
        not isinstance(ordered_keys, list)
        or len(ordered_keys) != len(expected_keys)
        or len(set(ordered_keys)) != len(ordered_keys)
        or set(ordered_keys) != set(expected_keys)
    ):
        raise ValueError("ordered episode manifest differs from the execution prefix")
    if (
        not isinstance(ordered_ids, list)
        or len(ordered_ids) != len(expected_keys)
        or not all(isinstance(value, (str, int)) for value in ordered_ids)
    ):
        raise ValueError("ordered episode IDs are invalid")
    identity_by_key = {
        key: (f"a::{episode_id}", generation)
        for generation, (key, episode_id) in enumerate(
            zip(ordered_keys, ordered_ids)
        )
    }
    controller_rows = _load_jsonl(controller_path, "controller records")
    values_by_key: dict[str, list[float]] = {key: [] for key in missing}
    for record in controller_rows:
        for key, (episode_id, generation) in identity_by_key.items():
            if key not in values_by_key:
                continue
            if (
                record.get("episode_id") != episode_id
                or record.get("reset_generation") != generation
                or record.get("state_only") is not False
                or record.get("identity_valid") is not True
            ):
                continue
            if "command_age_sec" not in record:
                raise ValueError(
                    "controller record is missing explicit sim-time command_age_sec"
                )
            value = record["command_age_sec"]
            if value is None:
                continue
            age = _finite(value, "controller command_age_sec")
            if age < 0.0:
                raise ValueError("controller command_age_sec must be nonnegative")
            values_by_key[key].append(age)

    attached: list[dict[str, Any]] = []
    sample_counts: dict[str, int] = {}
    for source in rows:
        row = dict(source)
        key = _episode_key(row)
        if key in values_by_key:
            values = values_by_key[key]
            if not values:
                raise ValueError(f"episode {key} has no finite command-age samples")
            row["command_age_sec"] = math.fsum(values) / len(values)
            row["command_age_sample_count"] = len(values)
            row["command_age_aggregation"] = (
                "mean_sim_time_current_identity_non_state_updates"
            )
            sample_counts[key] = len(values)
        attached.append(row)
    return attached, {
        "policy": "mean_sim_time_current_identity_non_state_updates",
        "controller_records_relative_path": controller_path.relative_to(
            result_root
        ).as_posix(),
        "ordered_episode_manifest_relative_path": manifest_path.relative_to(
            result_root
        ).as_posix(),
        "sample_count_by_episode": sample_counts,
        "source_sha256": {
            "controller_records": _file_sha256(controller_path),
            "ordered_episode_manifest": _file_sha256(manifest_path),
        },
    }


def materialize_candidate_result(
    result_root: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    episode_metrics_path: Path | None = None,
) -> dict[str, Any]:
    result_root = result_root.resolve()
    if not result_root.is_dir() or result_root.is_symlink():
        raise ValueError("fast-lane result root must be a regular directory")
    project_root, manifest, manifest_sha, code_sha = _manifest_context(manifest_path)

    paths = {
        "candidate_resolution": result_root / "candidate_resolution.json",
        "input_binding": result_root / "input_binding.json",
        "fast_lane_summary": result_root / "fast_lane_summary.json",
        "fast_lane_final_summary": result_root / "fast_lane_final_summary.json",
        "audit_candidate_resolution": result_root
        / "audits"
        / "dgx_candidate_resolution.json",
        "remote_candidate_resolution": result_root
        / "remote"
        / "dgx"
        / "candidate_resolution.json",
    }
    loaded = {name: _load_object(path, name) for name, path in paths.items()}
    resolution = loaded["candidate_resolution"]
    for name in ("audit_candidate_resolution", "remote_candidate_resolution"):
        if loaded[name] != resolution:
            raise ValueError(f"{name} differs from the coordinator resolution")

    selector = str(resolution.get("candidate_selector", ""))
    if resolution != resolve_candidate(project_root, selector):
        raise ValueError("fast-lane candidate resolution differs from local registry")
    if resolution.get("selector_kind") != "preregistered_composite":
        raise ValueError("fast-lane result is not a preregistered candidate")
    selected_ids = resolution["selected_candidate_ids"]
    selected_family = resolution["selected_families"][-1]
    candidate_id = selected_ids[-1]
    family = next(
        item
        for item in manifest["candidate_families"]
        if item["family_id"] == selected_family
    )
    candidate_config_sha = _sha256(
        family["config_sha256"], "candidate family config SHA"
    )
    resolution_provenance = resolution["provenance"]
    fixed_keys = resolution_provenance["fixed_episode_keys"]
    binding = loaded["input_binding"]
    summary = loaded["fast_lane_summary"]
    final = loaded["fast_lane_final_summary"]
    _require_pass(binding, "input binding")
    _require_pass(summary, "fast-lane summary")
    _require_pass(final, "fast-lane final summary")
    if final.get("runtime_summary") != summary:
        raise ValueError("final summary does not seal the exact runtime summary")
    if summary.get("input_binding") != binding:
        raise ValueError("runtime summary does not seal the exact input binding")

    profile = str(summary.get("profile", ""))
    expected_count = PROFILE_COUNTS.get(profile)
    if expected_count is None:
        raise ValueError("only screen1/screen3/fixed5 results can be materialized")
    resolution_sha = _sha256(
        resolution.get("resolution_sha256"), "candidate resolution SHA"
    )
    resolution_file_sha = _file_sha256(paths["candidate_resolution"])
    code_ref_sha = _git_sha(summary.get("code_ref_sha"), "fast-lane code_ref_sha")
    run_id = str(summary.get("run_id", ""))
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]{7,95}", run_id) is None:
        raise ValueError("fast-lane run_id is invalid")
    if result_root.name != f"fast-lane-a-{profile}-{run_id}":
        raise ValueError("fast-lane result directory does not match run identity")
    expected_keys = fixed_keys[:expected_count]
    if not all(
        (
            binding.get("lane") == "a",
            summary.get("lane") == "a",
            summary.get("candidate_profile") == selector,
            binding.get("candidate_profile") == selector,
            binding.get("candidate_resolution_sha256") == resolution_sha,
            binding.get("candidate_resolution_file_sha256") == resolution_file_sha,
            binding.get("candidate_binding") == resolution.get("canonical_binding"),
            summary.get("candidate_resolution_sha256") == resolution_sha,
            summary.get("candidate_binding") == resolution.get("canonical_binding"),
            binding.get("execution_profile") == profile,
            binding.get("execution_episode_count") == expected_count,
            binding.get("execution_episode_keys") == expected_keys,
            binding.get("episode_keys") == fixed_keys,
            binding.get("code_ref_sha") == code_ref_sha,
        )
    ):
        raise ValueError("fast-lane candidate/input/profile binding mismatch")
    source_receipts = binding.get("source_receipts")
    expected_receipts = {
        "prepare_summary_sha256",
        "prepare_final_sha256",
        "fast_prepare_input_sha256",
        "dataset_audit_sha256",
    }
    if not isinstance(source_receipts, dict) or set(source_receipts) != expected_receipts:
        raise ValueError("input binding source receipt set is incomplete")
    for name, value in source_receipts.items():
        _sha256(value, f"input binding {name}")

    metrics_path = (
        _inside(result_root, episode_metrics_path, "episode metrics")
        if episode_metrics_path is not None
        else _find_episode_metrics(result_root)
    )
    metrics = _load_object(metrics_path, "episode metrics")
    rows = metrics.get("episodes")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("episode metrics must contain an episodes object array")
    if len(rows) != expected_count:
        raise ValueError("episode metrics count differs from the execution profile")
    for count_name in ("expected_episode_count", "completed_episode_count"):
        if count_name in metrics and metrics[count_name] != expected_count:
            raise ValueError(f"episode metrics {count_name} mismatch")
    # Validate every higher-priority metric before consulting the independent
    # controller stream for the final command-age tie break.
    for row in rows:
        _episode_key(row)
        _one_boolean(row, "success", ("success", "sr", "SR"))
        _stuck(row)
        _one_number(row, "SPL", ("SPL", "spl"))
        _one_number(row, "nDTW", ("nDTW", "NDTW", "ndtw"))
        _one_number(row, "NE", ("NE", "ne_m"))
    rows, command_age_provenance = _attach_exact_command_age(
        result_root, rows, expected_keys
    )
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        materialized = _materialize_episode(row)
        key = materialized["episode_key"]
        if key in by_key:
            raise ValueError(f"duplicate per-episode metrics for {key}")
        by_key[key] = materialized
    if set(by_key) != set(expected_keys):
        raise ValueError(
            "per-episode metric identities differ from the frozen execution prefix"
        )

    relative_metrics = metrics_path.relative_to(result_root).as_posix()
    source_hashes = {name: _file_sha256(path) for name, path in paths.items()}
    source_hashes["episode_metrics"] = _file_sha256(metrics_path)
    if command_age_provenance is not None:
        source_hashes.update(command_age_provenance["source_sha256"])
    return {
        "schema_version": 2,
        "candidate_id": candidate_id,
        "provenance": {
            "predecessor_candidate_ids": list(
                resolution_provenance["predecessor_candidate_ids"]
            ),
            "code_bundle_sha256": code_sha,
            "candidate_config_sha256": candidate_config_sha,
            "candidate_manifest_sha256": manifest_sha,
        },
        "episodes": [by_key[key] for key in expected_keys],
        "source_provenance": {
            "kind": "t5_fast_lane_result",
            "lane": "a",
            "profile": profile,
            "run_id": run_id,
            "code_ref_sha": code_ref_sha,
            "candidate_selector": selector,
            "candidate_resolution_sha256": resolution_sha,
            "execution_episode_keys": expected_keys,
            "episode_metrics_relative_path": relative_metrics,
            "command_age_provenance": command_age_provenance,
            "source_file_sha256": source_hashes,
            "input_source_receipts": source_receipts,
        },
    }


def materialize_blocked_family(
    analysis_path: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    family_id: str = BLOCKED_FAMILY,
) -> dict[str, Any]:
    _project_root, manifest, manifest_sha, code_sha = _manifest_context(manifest_path)
    if family_id != BLOCKED_FAMILY:
        raise ValueError("only the recovery-dependent trajectory family may be blocked")
    analysis = _load_object(analysis_path.resolve(), "candidate analysis")
    validated = analysis.get("validated_provenance")
    expected_configs = {
        str(family["family_id"]): str(family["config_sha256"])
        for family in manifest["candidate_families"]
    }
    if validated != {
        "candidate_manifest_sha256": manifest_sha,
        "code_bundle_sha256": code_sha,
        "config_sha256_by_family": expected_configs,
    }:
        raise ValueError("candidate analysis provenance does not match the manifest")
    reports = analysis.get("family_reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError("candidate analysis has no family reports")
    first = reports[0]
    if (
        not isinstance(first, dict)
        or first.get("family_id") != "action_observation_recovery"
        or first.get("status") != "COMPLETE"
        or first.get("winner") != BLOCKING_CANDIDATE
    ):
        raise ValueError("C-family BLOCKED requires an exact a0 fixed-5 winner")
    return {
        "schema_version": 1,
        "record_type": "candidate_family_disposition",
        "status": "BLOCKED",
        "family_id": BLOCKED_FAMILY,
        "reason_code": "RECOVERY_A_NOT_SELECTED",
        "reason": (
            "C-family candidates require recovery-enabled a1, but the frozen "
            "action/recovery family winner is a0."
        ),
        "blocked_by_candidate_id": BLOCKING_CANDIDATE,
        "candidate_manifest_sha256": manifest_sha,
        "code_bundle_sha256": code_sha,
        "analysis_file_sha256": _file_sha256(analysis_path.resolve()),
    }


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--result-root", type=Path)
    mode.add_argument("--blocked-family", choices=(BLOCKED_FAMILY,))
    parser.add_argument("--episode-metrics", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.result_root is not None:
            if arguments.analysis is not None:
                raise ValueError("--analysis is only valid with --blocked-family")
            output = materialize_candidate_result(
                arguments.result_root,
                manifest_path=arguments.manifest,
                episode_metrics_path=arguments.episode_metrics,
            )
        else:
            if arguments.analysis is None:
                raise ValueError("--blocked-family requires --analysis")
            if arguments.episode_metrics is not None:
                raise ValueError("--episode-metrics is invalid with --blocked-family")
            output = materialize_blocked_family(
                arguments.analysis,
                manifest_path=arguments.manifest,
                family_id=arguments.blocked_family,
            )
        _write_new(arguments.output, output)
    except (FileExistsError, OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": output.get("status", "PASS"),
                "candidate_id": output.get("candidate_id"),
                "family_id": output.get("family_id"),
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
