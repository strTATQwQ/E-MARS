#!/usr/bin/env python3
"""Analyze preregistered Lane-A candidates without consulting held-out runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _boolean_metric(item: dict[str, Any], *names: str) -> bool:
    for name in names:
        if name not in item:
            continue
        value = item[name]
        if isinstance(value, bool):
            return value
        numeric = _finite_number(value, name)
        if numeric not in {0.0, 1.0}:
            raise ValueError(f"{name} must be boolean or 0/1")
        return bool(numeric)
    raise ValueError(f"missing metric; expected one of {names}")


def _number_metric(item: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in item:
            return _finite_number(item[name], name)
    raise ValueError(f"missing metric; expected one of {names}")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def code_bundle_sha256(root: Path, relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"code bundle path is not a regular file: {relative}")
        # Git archives use repository LF bytes, while a Windows worktree may
        # materialize the same tracked file with CRLF.  Provenance must bind
        # source content, not the checkout platform's newline policy.
        content = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _episode_key(item: dict[str, Any]) -> str:
    for name in ("episode_key", "episode_id", "key"):
        value = str(item.get(name, ""))
        if value:
            return value
    raise ValueError("episode record is missing episode_key")


def _episodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("episodes", payload.get("per_episode"))
    if not isinstance(value, list):
        raise ValueError("candidate result needs an episodes array")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("candidate episode records must be objects")
    return value


def aggregate_prefix(
    payload: dict[str, Any], episode_keys: list[str], count: int
) -> dict[str, Any]:
    by_key: dict[str, dict[str, Any]] = {}
    for item in _episodes(payload):
        key = _episode_key(item)
        if key in by_key:
            raise ValueError(f"duplicate episode record {key}")
        by_key[key] = item
    required = episode_keys[:count]
    missing = [key for key in required if key not in by_key]
    if missing:
        return {"complete": False, "missing_episode_keys": missing}
    records = [by_key[key] for key in required]
    successes = [_boolean_metric(item, "success", "SR") for item in records]
    spl = [_finite_number(item.get("SPL"), "SPL") for item in records]
    ndtw = [_number_metric(item, "nDTW", "NDTW") for item in records]
    ne = [_finite_number(item.get("NE"), "NE") for item in records]
    command_age = [
        _number_metric(item, "command_age_sec", "mean_command_age_sec")
        for item in records
    ]
    for label, values in (("SPL", spl), ("nDTW", ndtw)):
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"{label} must be inside [0,1]")
    if any(value < 0.0 for value in ne):
        raise ValueError("NE must be nonnegative")
    if any(value < 0.0 for value in command_age):
        raise ValueError("command age must be nonnegative")
    stuck = []
    for item in records:
        declared = _boolean_metric(item, "stuck") if "stuck" in item else False
        termination = str(item.get("termination_reason", "")).strip().lower()
        stuck.append(declared or termination == "stuck")
    return {
        "complete": True,
        "episode_count": count,
        "episode_keys": required,
        "success_count": sum(successes),
        "SR": sum(successes) / count,
        "mean_SPL": sum(spl) / count,
        "mean_nDTW": sum(ndtw) / count,
        "mean_NE": sum(ne) / count,
        "mean_command_age_sec": sum(command_age) / count,
        "stuck_count": sum(stuck),
    }


def _rank_key(item: tuple[str, dict[str, Any]]) -> tuple[Any, ...]:
    candidate_id, metrics = item
    return (
        -int(metrics["success_count"]),
        int(metrics["stuck_count"]),
        -float(metrics["mean_SPL"]),
        -float(metrics["mean_nDTW"]),
        float(metrics["mean_NE"]),
        float(metrics["mean_command_age_sec"]),
        candidate_id,
    )


def _candidate_index(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    families = manifest.get("candidate_families")
    if not isinstance(families, list) or len(families) != 3:
        raise ValueError("manifest must preregister exactly three candidate families")
    seen: set[str] = set()
    for family in families:
        if not isinstance(family, dict):
            raise ValueError("candidate family must be an object")
        _sha256(family.get("config_sha256"), "candidate family config_sha256")
        ids = family.get("candidate_ids")
        if not isinstance(ids, list) or len(ids) < 2:
            raise ValueError("each candidate family needs at least two candidates")
        for candidate_id in ids:
            candidate_id = str(candidate_id)
            if not candidate_id or candidate_id in seen:
                raise ValueError(f"duplicate/empty candidate_id {candidate_id!r}")
            seen.add(candidate_id)
    return families, seen


def _validate_result_provenance(
    payload: dict[str, Any],
    *,
    candidate_id: str,
    expected_predecessors: tuple[str, ...],
    expected_code_sha256: str,
    expected_config_sha256: str,
    expected_manifest_sha256: str,
) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{candidate_id} is missing provenance")
    observed_predecessors = provenance.get("predecessor_candidate_ids")
    if not isinstance(observed_predecessors, list) or tuple(
        map(str, observed_predecessors)
    ) != expected_predecessors:
        raise ValueError(
            f"{candidate_id} predecessor tuple mismatch: "
            f"expected={expected_predecessors!r} observed={observed_predecessors!r}"
        )
    expected = {
        "code_bundle_sha256": expected_code_sha256,
        "candidate_config_sha256": expected_config_sha256,
        "candidate_manifest_sha256": expected_manifest_sha256,
    }
    for name, wanted in expected.items():
        observed = _sha256(provenance.get(name), f"{candidate_id} {name}")
        if observed != wanted:
            raise ValueError(
                f"{candidate_id} {name} mismatch: expected={wanted} observed={observed}"
            )


def analyze(
    manifest: dict[str, Any],
    results: dict[str, dict[str, Any]],
    *,
    manifest_sha256: str | None = None,
    config_sha256_by_family: dict[str, str] | None = None,
    actual_code_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    families, known_candidates = _candidate_index(manifest)
    provenance_contract = manifest.get("provenance_contract")
    if not isinstance(provenance_contract, dict):
        raise ValueError("manifest is missing provenance_contract")
    expected_code_sha256 = _sha256(
        provenance_contract.get("code_bundle_sha256"), "code_bundle_sha256"
    )
    actual_code_sha256 = _sha256(
        actual_code_bundle_sha256 or "", "actual_code_bundle_sha256"
    )
    if actual_code_sha256 != expected_code_sha256:
        raise ValueError(
            "code bundle changed after manifest registration: "
            f"expected={expected_code_sha256} observed={actual_code_sha256}"
        )
    if config_sha256_by_family is None:
        raise ValueError("actual candidate config SHA map is required")
    expected_manifest_sha256 = _sha256(
        manifest_sha256 or canonical_json_sha256(manifest),
        "candidate_manifest_sha256",
    )
    unknown = sorted(set(results) - known_candidates)
    if unknown:
        raise ValueError(f"results contain unregistered candidates: {unknown}")
    fixed = manifest["fixed_evaluation"]
    episode_keys = [str(value) for value in fixed["episode_keys"]]
    if len(episode_keys) != int(fixed["episode_count"]):
        raise ValueError("fixed episode count differs from episode_keys")
    rounds = manifest["successive_halving"]["rounds"]

    family_reports: list[dict[str, Any]] = []
    fixed5_winners: list[tuple[str, dict[str, Any]]] = []
    pending_runs: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        expected_predecessors = tuple(
            candidate_id for candidate_id, _ in fixed5_winners
        )
        if len(expected_predecessors) != family_index:
            family_reports.append(
                {
                    "family_id": family["family_id"],
                    "status": "BLOCKED_BY_PREDECESSOR",
                    "winner": None,
                    "winner_fixed_5_metrics": None,
                    "expected_predecessor_count": family_index,
                    "observed_predecessor_winner_count": len(expected_predecessors),
                    "rounds": [],
                }
            )
            continue
        active = [str(value) for value in family["candidate_ids"]]
        expected_config_sha256 = _sha256(
            family["config_sha256"], "candidate family config_sha256"
        )
        actual_config_sha256 = _sha256(
            config_sha256_by_family.get(str(family["family_id"]), ""),
            f"{family['family_id']} actual config_sha256",
        )
        if actual_config_sha256 != expected_config_sha256:
            raise ValueError(
                f"{family['family_id']} config changed after registration"
            )
        round_reports: list[dict[str, Any]] = []
        family_complete = True
        winner: str | None = None
        winner_metrics: dict[str, Any] | None = None
        for round_config in rounds:
            count = int(round_config["cumulative_episode_count"])
            metrics_by_candidate: dict[str, dict[str, Any]] = {}
            missing: list[dict[str, Any]] = []
            for candidate_id in active:
                payload = results.get(candidate_id)
                if payload is None:
                    aggregate = {
                        "complete": False,
                        "missing_episode_keys": episode_keys[:count],
                    }
                else:
                    declared = str(payload.get("candidate_id", candidate_id))
                    if declared != candidate_id:
                        raise ValueError(
                            f"result identity mismatch: key={candidate_id} payload={declared}"
                        )
                    _validate_result_provenance(
                        payload,
                        candidate_id=candidate_id,
                        expected_predecessors=expected_predecessors,
                        expected_code_sha256=expected_code_sha256,
                        expected_config_sha256=expected_config_sha256,
                        expected_manifest_sha256=expected_manifest_sha256,
                    )
                    aggregate = aggregate_prefix(payload, episode_keys, count)
                if aggregate["complete"]:
                    metrics_by_candidate[candidate_id] = aggregate
                else:
                    item = {
                        "family_id": family["family_id"],
                        "round_id": round_config["round_id"],
                        "candidate_id": candidate_id,
                        "required_cumulative_episode_count": count,
                        "missing_episode_keys": aggregate["missing_episode_keys"],
                    }
                    missing.append(item)
                    pending_runs.append(item)
            if missing:
                family_complete = False
                round_reports.append(
                    {
                        "round_id": round_config["round_id"],
                        "status": "WAITING_FOR_RESULTS",
                        "active_candidates": active,
                        "metrics": metrics_by_candidate,
                        "missing": missing,
                    }
                )
                break
            ranking = sorted(metrics_by_candidate.items(), key=_rank_key)
            survivor_count = min(int(round_config["survivor_count"]), len(ranking))
            survivors = [candidate_id for candidate_id, _ in ranking[:survivor_count]]
            round_reports.append(
                {
                    "round_id": round_config["round_id"],
                    "status": "COMPLETE",
                    "active_candidates": active,
                    "ranking": [candidate_id for candidate_id, _ in ranking],
                    "survivors": survivors,
                    "metrics": metrics_by_candidate,
                }
            )
            active = survivors
            if count == int(fixed["episode_count"]):
                winner = survivors[0]
                winner_metrics = metrics_by_candidate[winner]
        if family_complete and winner is not None and winner_metrics is not None:
            fixed5_winners.append((winner, winner_metrics))
        family_reports.append(
            {
                "family_id": family["family_id"],
                "status": "COMPLETE" if family_complete and winner else "INCOMPLETE",
                "winner": winner,
                "winner_fixed_5_metrics": winner_metrics,
                "rounds": round_reports,
            }
        )

    plateau = _plateau(manifest, fixed5_winners)
    complete = len(fixed5_winners) == len(families)
    return {
        "schema_version": 1,
        "manifest_id": manifest["manifest_id"],
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "validated_provenance": {
            "candidate_manifest_sha256": expected_manifest_sha256,
            "code_bundle_sha256": expected_code_sha256,
            "config_sha256_by_family": {
                str(family["family_id"]): str(family["config_sha256"])
                for family in families
            },
        },
        "family_reports": family_reports,
        "family_winners": [candidate_id for candidate_id, _ in fixed5_winners],
        "plateau": plateau,
        "pending_runs": pending_runs,
    }


def _plateau(
    manifest: dict[str, Any], winners: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    required = len(manifest["candidate_families"])
    if len(winners) != required:
        return {
            "status": "NOT_EVALUATED",
            "reason": "all three family winners need fixed-5 evidence",
            "fixed_5_family_winner_count": len(winners),
            "required_count": required,
        }
    rule = manifest["plateau_rule"]
    threshold = rule["improvement_if_any"]
    comparisons: list[dict[str, Any]] = []
    for (before_id, before), (after_id, after) in zip(winners, winners[1:]):
        success_gain = int(after["success_count"]) - int(before["success_count"])
        ne_reduction = float(before["mean_NE"]) - float(after["mean_NE"])
        stuck_reduction = int(before["stuck_count"]) - int(after["stuck_count"])
        improving = bool(
            success_gain >= int(threshold["additional_success_count"])
            or ne_reduction >= float(threshold["mean_NE_reduction_m"])
            or stuck_reduction >= int(threshold["stuck_count_reduction"])
        )
        comparisons.append(
            {
                "before": before_id,
                "after": after_id,
                "success_count_gain": success_gain,
                "mean_NE_reduction_m": ne_reduction,
                "stuck_count_reduction": stuck_reduction,
                "material_improvement": improving,
            }
        )
    required_non_improving = int(rule["successive_non_improving_family_winners"])
    trailing = comparisons[-required_non_improving:]
    reached = len(trailing) == required_non_improving and not any(
        item["material_improvement"] for item in trailing
    )
    return {
        "status": "PLATEAU" if reached else "IMPROVING",
        "candidate_set_exhausted": True,
        "comparisons": comparisons,
        "rule": rule,
    }


def _parse_result_spec(spec: str) -> tuple[str, Path]:
    candidate_id, separator, path = spec.partition("=")
    if not separator or not candidate_id or not path:
        raise ValueError("--result must be candidate_id=/path/to/result.json")
    return candidate_id, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--result", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    results: dict[str, dict[str, Any]] = {}
    input_hashes: dict[str, str] = {
        "manifest": hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    }
    for spec in args.result:
        candidate_id, path = _parse_result_spec(spec)
        if candidate_id in results:
            raise ValueError(f"duplicate --result for {candidate_id}")
        results[candidate_id] = _load_json(path)
        input_hashes[candidate_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_digest = canonical_json_sha256(manifest)
    project_root = args.manifest.resolve().parents[3]
    config_hashes: dict[str, str] = {}
    for family in manifest.get("candidate_families", []):
        config_path = project_root / str(family["config"])
        config_hashes[str(family["family_id"])] = canonical_json_sha256(
            _load_json(config_path)
        )
    provenance_contract = manifest.get("provenance_contract", {})
    code_path_values = provenance_contract.get("code_paths", [])
    if not isinstance(code_path_values, list) or not all(
        isinstance(value, str) and value for value in code_path_values
    ):
        raise ValueError("manifest provenance code_paths are invalid")
    output = analyze(
        manifest,
        results,
        manifest_sha256=manifest_digest,
        config_sha256_by_family=config_hashes,
        actual_code_bundle_sha256=code_bundle_sha256(
            project_root, list(code_path_values)
        ),
    )
    output["input_sha256"] = input_hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
