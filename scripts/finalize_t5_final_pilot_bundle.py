#!/usr/bin/env python3
"""Finalize A10+B10 with exact profile and relocatable split bindings.

``finalize_t5_final_pilot.py`` owns metric semantics.  This thin wrapper adds
the online-runner contracts that are intentionally outside that metric tool:
both lanes used the identical frozen final profile, both bindings seal the
same split-audit bytes, and local evidence paths remain usable after moving the
whole result tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINALIZER = ROOT / "scripts" / "finalize_t5_final_pilot.py"
EXPECTED_PROFILE = {
    "candidate_profile": "a1+b1+c1",
    "rtf_ablation_profile": "navigation_fast",
    "isaac_sensor_profile": "baseline",
    "strict_extension_profile": "off",
    "nvblox_mode": "off",
    "run_mode": "model",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _load_finalizer(path: Path) -> ModuleType:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"metric finalizer is missing: {path}")
    spec = importlib.util.spec_from_file_location("t5_final_pilot_metric_finalizer", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load metric finalizer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "finalize", None)):
        raise ValueError("metric finalizer has no callable finalize()")
    return module


def _portable_path(base: Path, path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _receipt_path(prepare_root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} is not a portable relative path")
    path = (prepare_root / relative).resolve()
    try:
        path.relative_to(prepare_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the preparation result") from error
    return path


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError("final bundle output already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def finalize_bundle(
    *,
    lane_a_root: Path,
    lane_b_root: Path,
    prepare_root: Path,
    output: Path,
    metric_finalizer_path: Path = DEFAULT_FINALIZER,
) -> dict[str, Any]:
    lane_roots = {"a": lane_a_root.resolve(), "b": lane_b_root.resolve()}
    prepare_root = prepare_root.resolve()
    output = output.resolve(strict=False)
    receipt_path = prepare_root / "final_pilot_prepare_receipt.json"
    receipt = _object(receipt_path, "final-pilot preparation receipt")
    split = receipt.get("split") if isinstance(receipt.get("split"), dict) else {}
    lane_receipts = split.get("lanes") if isinstance(split.get("lanes"), dict) else {}
    split_path = _receipt_path(
        prepare_root, split.get("audit_relative_path"), "split audit"
    )
    split_audit = _object(split_path, "split audit")
    actual_split_sha = _sha256(split_path)
    relocated = json.loads(json.dumps(split_audit))
    source_path = _receipt_path(
        prepare_root, split.get("source_relative_path"), "source dataset"
    )
    relocated.setdefault("source", {})["dataset"] = str(source_path)
    relocated_lanes = relocated.setdefault("lanes", {})
    for lane in ("a", "b"):
        lane_receipt = (
            lane_receipts.get(lane)
            if isinstance(lane_receipts.get(lane), dict)
            else {}
        )
        lane_dataset = _receipt_path(
            prepare_root,
            lane_receipt.get("dataset_relative_path"),
            f"Lane {lane} dataset",
        )
        relocated_lanes.setdefault(lane, {})["output_dataset"] = str(lane_dataset)

    temporary_audit = output.parent / f".{output.name}.relocated-split.{os.getpid()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if temporary_audit.exists() or temporary_audit.is_symlink():
        raise ValueError("temporary relocated split path already exists")
    temporary_audit.write_text(
        json.dumps(relocated, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        metric_finalizer = _load_finalizer(metric_finalizer_path.resolve())
        report = metric_finalizer.finalize(
            lane_roots["a"], lane_roots["b"], temporary_audit
        )
    finally:
        temporary_audit.unlink(missing_ok=True)
    if not isinstance(report, dict):
        raise ValueError("metric finalizer did not return a JSON object")

    bindings = {
        lane: _object(lane_roots[lane] / "input_binding.json", f"Lane {lane} binding")
        for lane in ("a", "b")
    }
    fields = tuple(EXPECTED_PROFILE)
    expected_receipt_sha = _sha256(receipt_path)
    config_by_lane = {
        lane: {name: bindings[lane].get(name) for name in fields}
        for lane in ("a", "b")
    }
    source_receipts = {
        lane: (
            bindings[lane].get("source_receipts")
            if isinstance(bindings[lane].get("source_receipts"), dict)
            else {}
        )
        for lane in ("a", "b")
    }
    runner_checks = {
        "prepare_receipt_pass": receipt.get("status") == "PASS"
        and isinstance(receipt.get("checks"), dict)
        and bool(receipt["checks"])
        and all(value is True for value in receipt["checks"].values()),
        "original_split_audit_sha_exact": actual_split_sha == split.get("audit_sha256"),
        "both_lanes_final10": all(
            bindings[lane].get("execution_profile") == "final10"
            and bindings[lane].get("execution_episode_count") == 10
            for lane in ("a", "b")
        ),
        "lane_profile_config_equal": config_by_lane["a"] == config_by_lane["b"],
        "lane_profile_config_frozen": config_by_lane["a"] == EXPECTED_PROFILE,
        "lane_split_audit_sha_bound": all(
            bindings[lane].get("split_audit_sha256") == actual_split_sha
            and source_receipts[lane].get("split_audit_sha256") == actual_split_sha
            for lane in ("a", "b")
        ),
        "lane_prepare_receipt_sha_bound": all(
            source_receipts[lane].get("final_pilot_prepare_receipt_sha256")
            == expected_receipt_sha
            for lane in ("a", "b")
        ),
        "lane_dataset_sha_bound": all(
            bindings[lane].get("dataset_sha256")
            == (lane_receipts.get(lane) or {}).get("dataset_sha256")
            for lane in ("a", "b")
        ),
        "same_code_candidate_and_map": bindings["a"].get("code_ref_sha")
        == bindings["b"].get("code_ref_sha")
        == receipt.get("code_ref_sha")
        and bindings["a"].get("candidate_resolution_sha256")
        == bindings["b"].get("candidate_resolution_sha256")
        and bindings["a"].get("candidate_binding")
        == bindings["b"].get("candidate_binding")
        and bindings["a"].get("static_map_manifest_sha256")
        == bindings["b"].get("static_map_manifest_sha256")
        == (receipt.get("static_maps") or {}).get("manifest_sha256"),
    }
    runner_pass = all(runner_checks.values())
    integrity = report.get("integrity")
    if not isinstance(integrity, dict) or not isinstance(integrity.get("checks"), dict):
        raise ValueError("metric finalizer report lacks integrity checks")
    integrity["checks"]["final_runner_profile_and_split_binding"] = runner_pass
    integrity["runner_contract"] = {
        "status": "PASS" if runner_pass else "FAIL",
        "checks": runner_checks,
        "profile": EXPECTED_PROFILE,
        "split_audit_sha256": actual_split_sha,
        "prepare_receipt_sha256": expected_receipt_sha,
    }
    original_integrity_pass = integrity.get("status") == "PASS"
    overall_pass = original_integrity_pass and runner_pass
    integrity["status"] = "PASS" if overall_pass else "FAIL"
    report["status"] = "PASS" if overall_pass else "FAIL"
    if not overall_pass:
        report["aggregate"] = None
        promotion = report.get("promotion")
        if isinstance(promotion, dict):
            promotion["status"] = "NOT_EVALUABLE"
            if isinstance(promotion.get("checks"), dict):
                promotion["checks"]["integrity_pass"] = False
    for lane in ("a", "b"):
        evidence = (report.get("lanes") or {}).get(lane)
        if isinstance(evidence, dict):
            evidence["result_root"] = _portable_path(output.parent, lane_roots[lane])
    report["path_contract"] = {
        "base": "report_parent",
        "lane_a_result_root": _portable_path(output.parent, lane_roots["a"]),
        "lane_b_result_root": _portable_path(output.parent, lane_roots["b"]),
        "prepare_result_root": _portable_path(output.parent, prepare_root),
        "all_local_paths_relative": True,
        "relocation_rule": "move the common results tree without changing relative layout",
    }
    _write_atomic(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-a-root", required=True, type=Path)
    parser.add_argument("--lane-b-root", required=True, type=Path)
    parser.add_argument("--prepare-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metric-finalizer", type=Path, default=DEFAULT_FINALIZER)
    arguments = parser.parse_args()
    try:
        report = finalize_bundle(
            lane_a_root=arguments.lane_a_root,
            lane_b_root=arguments.lane_b_root,
            prepare_root=arguments.prepare_root,
            output=arguments.output,
            metric_finalizer_path=arguments.metric_finalizer,
        )
    except (OSError, ValueError) as error:
        print(f"final-pilot bundle finalization failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
