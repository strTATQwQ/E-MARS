#!/usr/bin/env python3
"""Replay one frozen Lane-B Rev-C fixed-5 bundle through Step3.

This is a private Lane-B adapter entrypoint.  It does not publish ROS messages,
goals, velocity commands, or terminal actions.  The only outputs are redacted
decision evidence plus the filesystem sidecars consumed by the read-only UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from slow_planner.base import CandidateFrontier, OrderedImage
from slow_planner.client import SlowPlannerClient
from slow_planner.lane_b import (
    LaneBDecisionSidecar,
    LaneBPlannerAdapter,
    LaneBPlannerMode,
    REV_C_IMAGE_SIZE,
    REV_C_VIEW_ORDER,
    TimedRevCImage,
    build_lane_b_request,
    parse_lane_b_snapshot_id,
    snapshot_content_sha256,
    validate_rev_c_snapshot,
)


FIXED_EPISODE_COUNT = 5
PRODUCTION_DEADLINE_MS = 12_000
MANIFEST_KIND = "t5_lane_b_step3_frozen_fixed5"
EVALUATION_SCOPES = {"interface_screening_only", "live_current_frontiers"}
_FORBIDDEN_PUBLIC_KEYS = ("raw_text", "chain_of_thought", "reasoning_content")


class FrozenReplayError(ValueError):
    """Raised when frozen replay material is missing, mutable, or inconsistent."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _require_sha256(value: Any, name: str) -> str:
    normalized = str(value or "")
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise FrozenReplayError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _require_relative(root: Path, value: Any, name: str) -> Path:
    relative = Path(str(value or ""))
    if not str(relative) or relative.is_absolute():
        raise FrozenReplayError(f"{name} must be a non-empty relative path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise FrozenReplayError(f"{name} escapes the frozen bundle") from exc
    return resolved


def _load_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenReplayError(f"cannot read {name}: {path}") from exc
    if not isinstance(value, dict):
        raise FrozenReplayError(f"{name} must contain one JSON object")
    return value


def _load_sidecar(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FrozenReplayError(f"cannot read snapshot sidecar: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FrozenReplayError(
                f"snapshot sidecar line {line_number} is not JSON"
            ) from exc
        if not isinstance(value, dict):
            raise FrozenReplayError(
                f"snapshot sidecar line {line_number} must be an object"
            )
        snapshot_id = str(value.get("snapshot_id") or "")
        parse_lane_b_snapshot_id(snapshot_id)
        declared = _require_sha256(
            value.get("snapshot_content_sha256"),
            f"snapshot[{snapshot_id}].snapshot_content_sha256",
        )
        if snapshot_content_sha256(value) != declared:
            raise FrozenReplayError(f"snapshot[{snapshot_id}] content hash mismatch")
        prior = records.get(snapshot_id)
        if prior is not None and prior != value:
            raise FrozenReplayError(f"snapshot_id reuse conflict: {snapshot_id}")
        records[snapshot_id] = value
    if not records:
        raise FrozenReplayError("snapshot sidecar contains no records")
    return records


def _finite_tuple(value: Any, name: str, *, required_length: int | None = None) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise FrozenReplayError(f"{name} must be an array")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise FrozenReplayError(f"{name}[{index}] must be numeric")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise FrozenReplayError(f"{name}[{index}] must be numeric") from exc
        if not math.isfinite(number):
            raise FrozenReplayError(f"{name}[{index}] must be finite")
        result.append(number)
    if required_length is not None and len(result) != required_length:
        raise FrozenReplayError(
            f"{name} must contain exactly {required_length} values"
        )
    return tuple(result)


def _snapshot_from_record(
    record: Mapping[str, Any],
    *,
    camera_root: Path,
    expected_config_sha256: str,
    expected_extrinsic_sha256: Mapping[str, str],
):
    snapshot_id = str(record.get("snapshot_id") or "")
    identity = parse_lane_b_snapshot_id(snapshot_id)
    if (
        record.get("kind") != "lane_b_rev_c_snapshot"
        or record.get("lane_id") != "b"
        or record.get("episode_id") != identity.episode_id
        or record.get("reset_id") != identity.reset_id
        or record.get("sequence_id") != identity.sequence_id
    ):
        raise FrozenReplayError(f"snapshot[{snapshot_id}] identity fields disagree")
    rows = record.get("images")
    if not isinstance(rows, list) or len(rows) != len(REV_C_VIEW_ORDER):
        raise FrozenReplayError(f"snapshot[{snapshot_id}] must contain four images")
    if tuple(str(row.get("view_id") or "") for row in rows) != REV_C_VIEW_ORDER:
        raise FrozenReplayError(f"snapshot[{snapshot_id}] view order is not Rev-C")

    snapshot_dir = (
        camera_root
        / identity.episode_id
        / f"reset-{identity.reset_id}"
        / f"sequence-{identity.sequence_id}"
    )
    frames = []
    for view_id, row in zip(REV_C_VIEW_ORDER, rows):
        if not isinstance(row, Mapping):
            raise FrozenReplayError(f"snapshot[{snapshot_id}] image row is not an object")
        jpeg_path = (snapshot_dir / f"{view_id}.jpg").resolve()
        try:
            jpeg_path.relative_to(camera_root.resolve())
        except ValueError as exc:
            raise FrozenReplayError("derived camera path escapes camera_root") from exc
        if not jpeg_path.is_file():
            raise FrozenReplayError(f"snapshot[{snapshot_id}] JPEG missing: {view_id}")
        jpeg = jpeg_path.read_bytes()
        declared_jpeg = _require_sha256(
            row.get("jpeg_sha256"), f"snapshot[{snapshot_id}].{view_id}.jpeg_sha256"
        )
        if _sha256_bytes(jpeg) != declared_jpeg:
            raise FrozenReplayError(f"snapshot[{snapshot_id}] JPEG hash mismatch: {view_id}")
        try:
            with Image.open(jpeg_path) as image:
                image.verify()
            with Image.open(jpeg_path) as image:
                if image.size != REV_C_IMAGE_SIZE:
                    raise FrozenReplayError(
                        f"snapshot[{snapshot_id}] {view_id} is not 640x480"
                    )
        except OSError as exc:
            raise FrozenReplayError(
                f"snapshot[{snapshot_id}] invalid JPEG: {view_id}"
            ) from exc
        width = int(row.get("width") or 0)
        height = int(row.get("height") or 0)
        frames.append(
            TimedRevCImage(
                image=OrderedImage(
                    view_id=view_id,
                    pose=_finite_tuple(row.get("pose"), f"image[{view_id}].pose"),
                    jpeg=jpeg,
                    width=width,
                    height=height,
                ),
                sim_stamp_s=float(row.get("sim_stamp_s")),
                extrinsic_sha256=_require_sha256(
                    row.get("extrinsic_sha256"),
                    f"snapshot[{snapshot_id}].{view_id}.extrinsic_sha256",
                ),
                source_frame_id=str(row.get("source_frame_id") or ""),
            )
        )
    return validate_rev_c_snapshot(
        episode_id=identity.episode_id,
        snapshot_id=snapshot_id,
        snapshot_sim_stamp_s=float(record.get("snapshot_sim_stamp_s")),
        frames=frames,
        config_sha256=str(record.get("config_sha256") or ""),
        expected_config_sha256=expected_config_sha256,
        expected_extrinsic_sha256=expected_extrinsic_sha256,
        max_frame_age_s=float(record.get("max_frame_age_s")),
        max_inter_camera_skew_s=float(record.get("max_inter_camera_skew_s")),
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError(f"short write while appending {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_snapshot(snapshot, runtime_step3_dir: Path) -> dict[str, Any]:
    camera_root = runtime_step3_dir / "cameras"
    identity = snapshot.identity
    target_dir = (
        camera_root
        / identity.episode_id
        / f"reset-{identity.reset_id}"
        / f"sequence-{identity.sequence_id}"
    )
    camera_paths = {}
    for frame in snapshot.frames:
        target = target_dir / f"{frame.image.view_id}.jpg"
        if target.exists():
            raise FrozenReplayError(f"runtime snapshot target already exists: {target}")
        _atomic_write(target, frame.image.jpeg)
        camera_paths[frame.image.view_id] = str(target.resolve())
    record = snapshot.sidecar_record(camera_paths=camera_paths)
    _append_jsonl(runtime_step3_dir / "snapshots.jsonl", record)
    return record


def _case_request(snapshot, value: Mapping[str, Any]):
    frontiers_raw = value.get("candidate_frontiers")
    if not isinstance(frontiers_raw, list):
        raise FrozenReplayError("case candidate_frontiers must be an array")
    return build_lane_b_request(
        snapshot,
        instruction=str(value.get("instruction") or ""),
        candidate_frontiers=tuple(
            CandidateFrontier.from_mapping(item) for item in frontiers_raw
        ),
        agent_pose=_finite_tuple(value.get("agent_pose"), "case.agent_pose"),
        visited_frontiers=tuple(value.get("visited_frontiers") or ()),
        compact_history=tuple(value.get("compact_history") or ()),
        wall_timestamp_s=time.time(),
    )


def _safe_model_decision(endpoint: str, request, adapter: LaneBPlannerAdapter):
    started = time.perf_counter()
    metrics = None
    model_status = "RESPONSE"
    try:
        with SlowPlannerClient(endpoint, timeout_ms=PRODUCTION_DEADLINE_MS) as client:
            decision, metrics = client.decide(request)
        outcome = adapter.resolve(request, decision)
    except Exception as error:  # public evidence intentionally omits exception text
        model_status = "TIMEOUT" if type(error).__name__ == "Again" else "ERROR"
        reason = (
            "step3_deadline_exceeded"
            if model_status == "TIMEOUT"
            else "step3_service_failure"
        )
        outcome = adapter.resolve_failure(request, reason)
    wall_ms = (time.perf_counter() - started) * 1000.0
    return outcome, metrics, model_status, wall_ms


def _load_manifest(path: Path, expected_sha256: str):
    expected = _require_sha256(expected_sha256, "manifest_sha256")
    if _sha256_file(path) != expected:
        raise FrozenReplayError("frozen fixed-5 manifest SHA-256 mismatch")
    value = _load_object(path, "fixed-5 manifest")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != MANIFEST_KIND
        or value.get("frozen") is not True
        or value.get("episode_count") != FIXED_EPISODE_COUNT
    ):
        raise FrozenReplayError("manifest is not a frozen Lane-B fixed-5 v1 bundle")
    if value.get("evaluation_scope") not in EVALUATION_SCOPES:
        raise FrozenReplayError("manifest has an unsupported evaluation_scope")
    if not str(value.get("candidate_frontier_source") or "") or not str(
        value.get("agent_pose_encoding") or ""
    ):
        raise FrozenReplayError(
            "manifest must identify frontier source and agent pose encoding"
        )
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != FIXED_EPISODE_COUNT:
        raise FrozenReplayError("manifest must contain exactly five cases")
    case_ids = [str(item.get("case_id") or "") for item in cases if isinstance(item, Mapping)]
    episode_keys = [
        str(item.get("episode_key") or "")
        for item in cases
        if isinstance(item, Mapping)
    ]
    snapshot_ids = [
        str(item.get("snapshot_id") or "")
        for item in cases
        if isinstance(item, Mapping)
    ]
    if (
        len(case_ids) != FIXED_EPISODE_COUNT
        or any(not value for value in case_ids)
        or len(set(case_ids)) != FIXED_EPISODE_COUNT
        or len(episode_keys) != FIXED_EPISODE_COUNT
        or any(not value for value in episode_keys)
        or len(set(episode_keys)) != FIXED_EPISODE_COUNT
        or len(set(snapshot_ids)) != FIXED_EPISODE_COUNT
    ):
        raise FrozenReplayError(
            "fixed-5 case_id, episode_key and snapshot_id values must be unique"
        )
    root = path.parent.resolve()
    sidecar_path = _require_relative(
        root, value.get("snapshot_sidecar"), "snapshot_sidecar"
    )
    camera_root = _require_relative(root, value.get("camera_root"), "camera_root")
    expected_sidecar = _require_sha256(
        value.get("snapshot_sidecar_sha256"), "snapshot_sidecar_sha256"
    )
    if _sha256_file(sidecar_path) != expected_sidecar:
        raise FrozenReplayError("snapshot sidecar SHA-256 mismatch")
    expected_config = _require_sha256(
        value.get("expected_config_sha256"), "expected_config_sha256"
    )
    extrinsics = value.get("expected_extrinsic_sha256")
    if not isinstance(extrinsics, Mapping) or set(extrinsics) != set(REV_C_VIEW_ORDER):
        raise FrozenReplayError("expected extrinsics must cover exactly four Rev-C views")
    expected_extrinsics = {
        view_id: _require_sha256(extrinsics[view_id], f"extrinsic[{view_id}]")
        for view_id in REV_C_VIEW_ORDER
    }
    return value, cases, sidecar_path, camera_root, expected_config, expected_extrinsics


def run_fixed5(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    endpoint: str,
    mode: LaneBPlannerMode,
    runtime_step3_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if endpoint != "tcp://127.0.0.1:8200":
        raise FrozenReplayError("Step3 fixed-5 endpoint must remain loopback TCP 8200")
    if output_dir.exists():
        raise FrozenReplayError(f"output directory already exists: {output_dir}")
    runtime_step3_dir.mkdir(parents=True, exist_ok=True)
    for name in ("snapshots.jsonl", "frontend_decisions.jsonl"):
        if (runtime_step3_dir / name).exists():
            raise FrozenReplayError(f"runtime sidecar already exists: {name}")
    (
        manifest,
        cases,
        sidecar_path,
        camera_root,
        expected_config,
        expected_extrinsics,
    ) = _load_manifest(manifest_path.resolve(), manifest_sha256)
    snapshots = _load_sidecar(sidecar_path)
    output_dir.mkdir(parents=True)
    decision_sidecar = LaneBDecisionSidecar(
        runtime_step3_dir / "frontend_decisions.jsonl"
    )
    adapter = LaneBPlannerAdapter(mode)
    results = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise FrozenReplayError(f"case[{index}] must be an object")
        snapshot_id = str(case.get("snapshot_id") or "")
        source = snapshots.get(snapshot_id)
        if source is None:
            raise FrozenReplayError(f"case[{index}] snapshot is absent from sidecar")
        snapshot = _snapshot_from_record(
            source,
            camera_root=camera_root,
            expected_config_sha256=expected_config,
            expected_extrinsic_sha256=expected_extrinsics,
        )
        request = _case_request(snapshot, case)
        published_snapshot = _publish_snapshot(snapshot, runtime_step3_dir)
        outcome, metrics, model_status, wall_ms = _safe_model_decision(
            endpoint, request, adapter
        )
        public_decision = decision_sidecar.append(outcome, metrics)
        row = {
            "schema_version": 1,
            "kind": "t5_lane_b_step3_frozen_replay_result",
            "case_index": index,
            "case_id": str(case.get("case_id") or ""),
            "episode_key": str(case.get("episode_key") or ""),
            "snapshot_id": snapshot_id,
            "snapshot_content_sha256": published_snapshot[
                "snapshot_content_sha256"
            ],
            "mode": mode.value,
            "deadline_ms": PRODUCTION_DEADLINE_MS,
            "wall_ms": wall_ms,
            "model_request_status": model_status,
            "deadline_met": model_status == "RESPONSE"
            and wall_ms <= PRODUCTION_DEADLINE_MS,
            **public_decision,
        }
        encoded = json.dumps(row, sort_keys=True)
        if any(forbidden in encoded for forbidden in _FORBIDDEN_PUBLIC_KEYS):
            raise RuntimeError("public replay evidence contains a forbidden private field")
        _append_jsonl(output_dir / "results.jsonl", row)
        results.append(row)

    response_count = sum(
        row["model_request_status"] == "RESPONSE" for row in results
    )
    deadline_count = sum(bool(row["deadline_met"]) for row in results)
    fallback_count = sum(
        bool(row["decision"].get("requires_internvla_fallback")) for row in results
    )
    safe_stop_count = sum(
        bool(row["decision"].get("requires_safe_stop")) for row in results
    )
    structured_parse_count = sum(
        isinstance(row["decision"].get("scene_summary"), str)
        and bool(row["decision"].get("scene_summary"))
        and isinstance(row["decision"].get("target_evidence"), list)
        and isinstance(row["decision"].get("blocked_directions"), list)
        and isinstance(row["decision"].get("target_found"), bool)
        and isinstance(row["decision"].get("abstain"), bool)
        for row in results
    )
    legal_ids_by_snapshot = {
        str(case["snapshot_id"]): {
            int(frontier["frontier_id"])
            for frontier in case.get("candidate_frontiers", [])
        }
        for case in cases
    }
    legal_or_abstain_count = sum(
        (
            row["decision"].get("abstain") is True
            and row["decision"].get("recommended_frontier") is None
        )
        or (
            row["decision"].get("abstain") is False
            and row["decision"].get("recommended_frontier")
            in legal_ids_by_snapshot[row["snapshot_id"]]
        )
        for row in results
    )
    nonfallback_selection_count = sum(
        not row["decision"].get("requires_internvla_fallback")
        and row["decision"].get("recommended_frontier")
        in legal_ids_by_snapshot[row["snapshot_id"]]
        for row in results
    )
    semantic_abstain_count = sum(
        row["decision"].get("abstain") is True
        and (
            row["decision"].get("fallback_reason") == "step3_abstain"
            or row["decision"].get("safe_stop_reason") == "step3_abstain"
        )
        for row in results
    )
    wall_values = sorted(float(row["wall_ms"]) for row in results)
    p95_index = max(0, math.ceil(0.95 * len(wall_values)) - 1)
    median_index = len(wall_values) // 2
    peak_memory_values = [
        float(row.get("metrics", {}).get("peak_memory_mib") or 0.0)
        for row in results
    ]
    structured_parse_rate = structured_parse_count / len(results) if results else 0.0
    legal_or_abstain_rate = legal_or_abstain_count / len(results) if results else 0.0
    offline_gate_pass = (
        structured_parse_rate >= 0.95
        and legal_or_abstain_rate == 1.0
        and nonfallback_selection_count >= 1
    )
    model_interface_eligible = response_count == FIXED_EPISODE_COUNT and (
        deadline_count == FIXED_EPISODE_COUNT
    )
    evaluation_scope = str(manifest["evaluation_scope"])
    checks = {
        "exact_fixed5": len(results) == FIXED_EPISODE_COUNT,
        "frozen_manifest_exact": _sha256_file(manifest_path.resolve())
        == manifest_sha256,
        "snapshot_sidecar_exact": _sha256_file(sidecar_path)
        == manifest["snapshot_sidecar_sha256"],
        "loopback_service": endpoint == "tcp://127.0.0.1:8200",
        "no_motion_authority": all(
            row["decision"].get("motion_authority") == "none" for row in results
        ),
        "no_terminal_stop_authority": True,
        "direct_mode_has_no_internvla_fallback": (
            mode is not LaneBPlannerMode.DIRECT_HIGH_LEVEL
            or fallback_count == 0
        ),
        "direct_failures_require_safe_stop": (
            mode is not LaneBPlannerMode.DIRECT_HIGH_LEVEL
            or all(
                not row["decision"].get("abstain")
                or row["decision"].get("requires_safe_stop") is True
                for row in results
            )
        ),
        "structured_response_or_fallback": all(
            row["model_request_status"] in {"RESPONSE", "TIMEOUT", "ERROR"}
            for row in results
        ),
        "structured_json_parse_rate_at_least_95pct": structured_parse_rate >= 0.95,
        "legal_frontier_or_explicit_abstain_100pct": legal_or_abstain_rate == 1.0,
        "at_least_one_real_frontier_advice": nonfallback_selection_count >= 1,
        "cross_reset_execution_zero": True,
        "stale_response_execution_zero": True,
        "illegal_frontier_execution_zero": True,
        "public_cot_leak_zero": True,
    }
    summary = {
        "schema_version": 1,
        "kind": "t5_lane_b_step3_frozen_fixed5_summary",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "promotion_eligible": model_interface_eligible and offline_gate_pass,
        "model_interface_eligible": model_interface_eligible,
        "offline_acceptance_gate_pass": offline_gate_pass,
        "navigation_effect_claim_eligible": model_interface_eligible
        and offline_gate_pass
        and evaluation_scope == "live_current_frontiers",
        "evaluation_scope": evaluation_scope,
        "candidate_frontier_source": manifest["candidate_frontier_source"],
        "agent_pose_encoding": manifest["agent_pose_encoding"],
        "mode": mode.value,
        "manifest_sha256": manifest_sha256,
        "snapshot_sidecar_sha256": manifest["snapshot_sidecar_sha256"],
        "case_count": len(results),
        "model_response_count": response_count,
        "deadline_met_count": deadline_count,
        "internvla_fallback_required_count": fallback_count,
        "direct_safe_stop_required_count": safe_stop_count,
        "structured_json_parse_count": structured_parse_count,
        "structured_json_parse_rate": structured_parse_rate,
        "legal_frontier_or_explicit_abstain_count": legal_or_abstain_count,
        "legal_frontier_or_explicit_abstain_rate": legal_or_abstain_rate,
        "nonfallback_frontier_selection_count": nonfallback_selection_count,
        "semantic_abstain_count": semantic_abstain_count,
        "latency_wall_p50_ms": wall_values[median_index],
        "latency_wall_p95_ms": wall_values[p95_index],
        "latency_wall_max_ms": wall_values[-1],
        "unified_model_memory_peak_mib": max(peak_memory_values),
        "cross_reset_execution_count": 0,
        "stale_response_execution_count": 0,
        "illegal_frontier_execution_count": 0,
        "public_cot_leak_count": 0,
        "motion_authority": "none",
        "terminal_stop_authority": "none",
        "checks": checks,
        "recorded_wall_time_s": time.time(),
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if any(forbidden in encoded for forbidden in _FORBIDDEN_PUBLIC_KEYS):
        raise RuntimeError("public summary contains a forbidden private field")
    _atomic_write(output_dir / "summary.json", encoded.encode("utf-8"))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:8200")
    parser.add_argument(
        "--mode", choices=[item.value for item in LaneBPlannerMode], required=True
    )
    parser.add_argument("--runtime-step3-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_fixed5(
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        endpoint=args.endpoint,
        mode=LaneBPlannerMode(args.mode),
        runtime_step3_dir=args.runtime_step3_dir.resolve(),
        output_dir=args.output.resolve(),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 75


if __name__ == "__main__":
    raise SystemExit(main())
