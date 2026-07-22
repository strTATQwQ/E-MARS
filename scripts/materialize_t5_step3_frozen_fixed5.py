#!/usr/bin/env python3
"""Materialize five real Isaac Rev-C captures for the frozen Step3 runner.

The input context is an operator-frozen JSON file containing exactly five
request contexts.  Every case points at one real Isaac ``snapshot.json`` and
binds its exact file SHA-256.  This tool validates the original PNG hashes,
same-render-tick identity, completion-sim camera contract, and
episode/reset/sequence identity before writing the private consumer bundle.

Example::

    python scripts/materialize_t5_step3_frozen_fixed5.py \
      --contexts configs/internnav_t5/step3_fixed5_capture_contexts.json \
      --revc-contract configs/internnav_t5/revc_four_camera_snapshot.json \
      --output results/internnav_t5/step3/frozen-fixed5

The output directory must not already exist.  On success it contains
``manifest.json``, ``snapshots.jsonl``, ``materialization.json`` and four
640x480 JPEGs per case under ``cameras/``.  The command prints the manifest
SHA-256 required by ``run_t5_step3_frozen_fixed5.py``.  No source image or
request context is synthesized, and one capture cannot be duplicated to fill
the fixed five.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from PIL import Image

# Keep the documented ``python scripts/...`` invocation independent of an
# operator-specific PYTHONPATH while importing only repository-local contracts.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slow_planner.base import CandidateFrontier
from slow_planner.lane_b import (
    REV_C_IMAGE_SIZE,
    REV_C_VIEW_ORDER,
    parse_lane_b_snapshot_id,
    snapshot_content_sha256,
)


CONTEXT_KIND = "t5_lane_b_step3_fixed5_capture_contexts"
MANIFEST_KIND = "t5_lane_b_step3_frozen_fixed5"
SNAPSHOT_KIND = "lane_b_rev_c_snapshot"
FIXED_EPISODE_COUNT = 5
MAX_FRAME_AGE_S = 0.5
MAX_INTER_CAMERA_SKEW_S = 0.2
EVALUATION_SCOPES = {"interface_screening_only", "live_current_frontiers"}


class MaterializationError(ValueError):
    """Raised when real capture evidence is absent, mutable, or inconsistent."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise MaterializationError(f"cannot read file: {path}") from exc


def _require_sha256(value: Any, name: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise MaterializationError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _load_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read {name}: {path}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"{name} must contain one JSON object")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise MaterializationError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MaterializationError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise MaterializationError(f"{name} must be finite")
    return number


def _finite_array(value: Any, name: str) -> list[float]:
    if not isinstance(value, list):
        raise MaterializationError(f"{name} must be an array")
    return [_finite(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MaterializationError(f"{name} must be a non-negative integer")
    return value


def _resolve_context_path(context_path: Path, value: Any, name: str) -> Path:
    raw = str(value or "")
    if not raw:
        raise MaterializationError(f"{name} is required")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = context_path.parent / candidate
    resolved = candidate.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise MaterializationError(f"{name} must name one regular, non-symlink file")
    return resolved


def _load_contract(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, str], dict[str, tuple[float, ...]]]:
    observed_sha = _sha256_file(path)
    if observed_sha != expected_sha256:
        raise MaterializationError("Rev-C contract SHA-256 mismatch")
    contract = _load_object(path, "Rev-C camera contract")
    if (
        contract.get("schema_version") != 1
        or contract.get("contract_id") != "internnav-t5-revc-four-camera-v1"
        or contract.get("scope") != "completion_sim_only"
        or tuple(contract.get("camera_order") or ()) != REV_C_VIEW_ORDER
    ):
        raise MaterializationError("unsupported or non-completion-sim Rev-C contract")
    optics = contract.get("optics")
    frames = contract.get("frames")
    cameras = contract.get("cameras")
    if (
        not isinstance(optics, Mapping)
        or tuple(optics.get("resolution") or ()) != REV_C_IMAGE_SIZE
        or _finite(optics.get("hfov_deg"), "optics.hfov_deg") != 73.0
        or _finite(optics.get("pitch_down_deg"), "optics.pitch_down_deg") != 10.0
        or not isinstance(frames, Mapping)
        or not isinstance(cameras, list)
        or len(cameras) != len(REV_C_VIEW_ORDER)
    ):
        raise MaterializationError("Rev-C optics, frames, or camera count drifted")
    mast = frames.get("temporary_T_base_link_F_M")
    if not isinstance(mast, Mapping):
        raise MaterializationError("temporary T_base_link_F_M is required")
    translation = _finite_array(mast.get("translation_m"), "mast.translation_m")
    rotation = _finite_array(mast.get("rotation_wxyz"), "mast.rotation_wxyz")
    if translation != [0.14, 0.0, 0.18] or rotation != [1.0, 0.0, 0.0, 0.0]:
        raise MaterializationError("temporary T_base_link_F_M drifted")

    extrinsics: dict[str, str] = {}
    poses: dict[str, tuple[float, ...]] = {}
    for index, (expected_view, camera) in enumerate(zip(REV_C_VIEW_ORDER, cameras)):
        if not isinstance(camera, Mapping) or camera.get("identity") != expected_view:
            raise MaterializationError(f"camera[{index}] identity/order drifted")
        position_m = _finite_array(camera.get("position_F_M_m"), f"camera[{expected_view}].position_F_M_m")
        if len(position_m) != 3:
            raise MaterializationError(f"camera[{expected_view}] position must have three values")
        yaw_deg = _finite(camera.get("yaw_deg"), f"camera[{expected_view}].yaw_deg")
        pose = (
            translation[0] + position_m[0],
            translation[1] + position_m[1],
            translation[2] + position_m[2],
            math.radians(yaw_deg),
            math.radians(-10.0),
        )
        poses[expected_view] = pose
        extrinsics[expected_view] = _canonical_sha256(
            {
                "schema_version": 1,
                "base_frame": frames.get("base_frame"),
                "mast_frame": frames.get("mast_frame"),
                "temporary_T_base_link_F_M": mast,
                "camera": camera,
                "pitch_down_deg": optics.get("pitch_down_deg"),
            }
        )
    return contract, extrinsics, poses


def _validate_context_case(case: Mapping[str, Any], index: int) -> dict[str, Any]:
    required = (
        "case_id",
        "episode_key",
        "snapshot_id",
        "source_snapshot_sidecar",
        "source_snapshot_sidecar_sha256",
        "instruction",
        "candidate_frontiers",
        "agent_pose",
        "visited_frontiers",
        "compact_history",
    )
    missing = [name for name in required if name not in case]
    if missing:
        raise MaterializationError(f"case[{index}] lacks required fields: {missing}")
    case_id = str(case.get("case_id") or "")
    episode_key = str(case.get("episode_key") or "")
    instruction = str(case.get("instruction") or "")
    snapshot_id = str(case.get("snapshot_id") or "")
    if not case_id or not episode_key or not instruction.strip():
        raise MaterializationError(f"case[{index}] identifiers and instruction are required")
    try:
        identity = parse_lane_b_snapshot_id(snapshot_id)
    except Exception as exc:
        raise MaterializationError(f"case[{index}] has invalid snapshot_id") from exc

    frontiers_raw = case.get("candidate_frontiers")
    if not isinstance(frontiers_raw, list):
        raise MaterializationError(f"case[{index}].candidate_frontiers must be an array")
    try:
        frontiers = [CandidateFrontier.from_mapping(item) for item in frontiers_raw]
    except Exception as exc:
        raise MaterializationError(f"case[{index}] has invalid candidate_frontiers") from exc
    if len({item.frontier_id for item in frontiers}) != len(frontiers):
        raise MaterializationError(f"case[{index}] repeats a frontier_id")
    agent_pose = _finite_array(case.get("agent_pose"), f"case[{index}].agent_pose")
    visited = case.get("visited_frontiers")
    history = case.get("compact_history")
    if not isinstance(visited, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in visited
    ):
        raise MaterializationError(f"case[{index}].visited_frontiers is invalid")
    if not isinstance(history, list) or any(not isinstance(item, str) for item in history):
        raise MaterializationError(f"case[{index}].compact_history is invalid")

    return {
        "case_id": case_id,
        "episode_key": episode_key,
        "snapshot_id": snapshot_id,
        "instruction": instruction,
        "candidate_frontiers": [item.to_mapping() for item in frontiers],
        "agent_pose": agent_pose,
        "visited_frontiers": list(visited),
        "compact_history": list(history),
    }


def _source_image_path(sidecar_path: Path, raw_path: Any, view_id: str) -> Path:
    relative = PurePosixPath(str(raw_path or ""))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise MaterializationError(f"source image path is unsafe for {view_id}")
    if sidecar_path.parent.parent.name != "revc_snapshots":
        raise MaterializationError("source snapshot.json must remain under revc_snapshots/<id>")
    evaluator_root = sidecar_path.parent.parent.parent
    candidate = (evaluator_root / Path(*relative.parts)).resolve()
    expected_parent = sidecar_path.parent.resolve()
    if candidate.parent != expected_parent or not candidate.is_file() or candidate.is_symlink():
        raise MaterializationError(f"source image is not scoped to its snapshot: {view_id}")
    return candidate


def _render_time(raw: Mapping[str, Any]) -> float:
    identity = raw.get("render_identity_value")
    if not isinstance(identity, Mapping):
        raise MaterializationError("source snapshot lacks render_identity_value")
    numerator = _nonnegative_int(identity.get("referenceTimeNumerator"), "referenceTimeNumerator")
    denominator = _nonnegative_int(identity.get("referenceTimeDenominator"), "referenceTimeDenominator")
    rendering_time = _finite(identity.get("rendering_time"), "rendering_time")
    if denominator == 0 or rendering_time < 0 or not math.isclose(
        numerator / denominator, rendering_time, rel_tol=0.0, abs_tol=1e-9
    ):
        raise MaterializationError("source ReferenceTime fields disagree")
    return rendering_time


def _validate_raw_snapshot(
    *,
    sidecar_path: Path,
    sidecar_sha256: str,
    expected_snapshot_id: str,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> tuple[dict[str, Any], list[tuple[str, Path]], float]:
    if _sha256_file(sidecar_path) != sidecar_sha256:
        raise MaterializationError(f"source sidecar SHA-256 mismatch: {sidecar_path}")
    raw = _load_object(sidecar_path, "Isaac Rev-C snapshot sidecar")
    identity = parse_lane_b_snapshot_id(expected_snapshot_id)
    if (
        raw.get("schema_version") != 1
        or raw.get("scope") != "completion_sim_only"
        or raw.get("contract_id") != contract.get("contract_id")
        or raw.get("contract_sha256") != contract_sha256
        or raw.get("same_render_tick") is not True
        or raw.get("cuvslam_stereo_is_separate") is not True
        or raw.get("episode_id") != f"b::{identity.episode_id}"
        or raw.get("reset_generation") != identity.reset_id
        or raw.get("sequence_id") != identity.sequence_id
        or tuple(raw.get("camera_order") or ()) != REV_C_VIEW_ORDER
        or raw.get("sim_stamp_before_ns") != raw.get("sim_stamp_after_ns")
    ):
        raise MaterializationError(f"source capture identity or Rev-C contract mismatch: {sidecar_path}")
    request_id = str(raw.get("request_id") or "")
    render_barrier_id = str(raw.get("render_barrier_id") or "")
    if not request_id or not render_barrier_id:
        raise MaterializationError("source request_id and render_barrier_id are required")
    _render_time(raw)
    sim_stamp_ns = raw.get("sim_stamp_before_ns")
    if (
        isinstance(sim_stamp_ns, bool)
        or not isinstance(sim_stamp_ns, int)
        or sim_stamp_ns <= 0
    ):
        raise MaterializationError("source snapshot lacks a valid simulation stamp")
    sim_stamp_s = sim_stamp_ns / 1_000_000_000.0

    cameras = raw.get("cameras")
    contract_cameras = contract.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != len(REV_C_VIEW_ORDER):
        raise MaterializationError("source snapshot must contain exactly four cameras")
    sources: list[tuple[str, Path]] = []
    for index, (view_id, camera, expected) in enumerate(
        zip(REV_C_VIEW_ORDER, cameras, contract_cameras)
    ):
        if not isinstance(camera, Mapping) or not isinstance(expected, Mapping):
            raise MaterializationError(f"source camera[{index}] must be an object")
        source_identity = camera.get("render_identity")
        if (
            camera.get("identity") != view_id
            or camera.get("order_index") != index
            or tuple(camera.get("resolution") or ()) != REV_C_IMAGE_SIZE
            or camera.get("encoding") != "png_rgb8"
            or camera.get("frame_id") != expected.get("frame_id")
            or camera.get("sensor_name") != expected.get("sensor_name")
            or camera.get("prim_path") != expected.get("prim_path")
            or camera.get("position_F_M_mm") != expected.get("position_F_M_mm")
            or _finite(camera.get("yaw_deg"), f"camera[{view_id}].yaw_deg")
            != _finite(expected.get("yaw_deg"), f"contract.camera[{view_id}].yaw_deg")
            or _finite(camera.get("pitch_down_deg"), f"camera[{view_id}].pitch_down_deg") != 10.0
            or _finite(camera.get("hfov_deg"), f"camera[{view_id}].hfov_deg") != 73.0
            or source_identity != raw.get("render_identity_value")
        ):
            raise MaterializationError(f"source camera contract drift: {view_id}")
        source_path = _source_image_path(sidecar_path, camera.get("path"), view_id)
        declared = _require_sha256(camera.get("sha256"), f"camera[{view_id}].sha256")
        if _sha256_file(source_path) != declared:
            raise MaterializationError(f"source PNG SHA-256 mismatch: {view_id}")
        if source_path.stat().st_size != camera.get("bytes"):
            raise MaterializationError(f"source PNG byte count mismatch: {view_id}")
        try:
            with Image.open(source_path) as image:
                image.verify()
            with Image.open(source_path) as image:
                if image.size != REV_C_IMAGE_SIZE or image.mode != "RGB" or image.format != "PNG":
                    raise MaterializationError(f"source image is not 640x480 PNG RGB8: {view_id}")
        except OSError as exc:
            raise MaterializationError(f"source PNG cannot be decoded: {view_id}") from exc
        sources.append((view_id, source_path))
    return raw, sources, sim_stamp_s


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jpeg(source: Path, target: Path) -> bytes:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.save(
            target,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
    payload = target.read_bytes()
    try:
        with Image.open(target) as image:
            image.verify()
        with Image.open(target) as image:
            if image.size != REV_C_IMAGE_SIZE or image.format != "JPEG":
                raise MaterializationError(f"materialized JPEG is invalid: {target}")
    except OSError as exc:
        raise MaterializationError(f"materialized JPEG cannot be decoded: {target}") from exc
    return payload


def materialize_bundle(
    *, contexts_path: Path, revc_contract_path: Path, output_dir: Path
) -> dict[str, Any]:
    contexts_path = contexts_path.resolve()
    revc_contract_path = revc_contract_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise MaterializationError(f"output directory already exists: {output_dir}")
    contexts = _load_object(contexts_path, "fixed-5 capture contexts")
    cases_raw = contexts.get("cases")
    if (
        contexts.get("schema_version") != 1
        or contexts.get("kind") != CONTEXT_KIND
        or contexts.get("frozen") is not True
        or not isinstance(cases_raw, list)
        or len(cases_raw) != FIXED_EPISODE_COUNT
    ):
        raise MaterializationError("capture context must contain exactly five frozen cases")
    evaluation_scope = str(contexts.get("evaluation_scope") or "")
    candidate_frontier_source = str(contexts.get("candidate_frontier_source") or "")
    agent_pose_encoding = str(contexts.get("agent_pose_encoding") or "")
    if evaluation_scope not in EVALUATION_SCOPES:
        raise MaterializationError("capture context has an unsupported evaluation_scope")
    if not candidate_frontier_source or not agent_pose_encoding:
        raise MaterializationError(
            "capture context must identify frontier source and agent pose encoding"
        )
    expected_contract_sha = _require_sha256(
        contexts.get("revc_contract_sha256"), "revc_contract_sha256"
    )
    contract, extrinsics, camera_poses = _load_contract(
        revc_contract_path, expected_contract_sha
    )

    validated_cases = []
    source_paths = []
    source_hashes = []
    for index, raw_case in enumerate(cases_raw):
        if not isinstance(raw_case, Mapping):
            raise MaterializationError(f"case[{index}] must be an object")
        validated = _validate_context_case(raw_case, index)
        source = _resolve_context_path(
            contexts_path,
            raw_case.get("source_snapshot_sidecar"),
            f"case[{index}].source_snapshot_sidecar",
        )
        source_paths.append(source)
        source_hashes.append(
            _require_sha256(
                raw_case.get("source_snapshot_sidecar_sha256"),
                f"case[{index}].source_snapshot_sidecar_sha256",
            )
        )
        validated_cases.append((validated, source, source_hashes[-1]))

    public_cases = [item[0] for item in validated_cases]
    for label, values in (
        ("case_id", [item["case_id"] for item in public_cases]),
        ("episode_key", [item["episode_key"] for item in public_cases]),
        ("snapshot_id", [item["snapshot_id"] for item in public_cases]),
    ):
        if len(set(values)) != FIXED_EPISODE_COUNT:
            raise MaterializationError(f"fixed-5 {label} values must be unique")
    if len(set(source_paths)) != FIXED_EPISODE_COUNT or len(set(source_hashes)) != FIXED_EPISODE_COUNT:
        raise MaterializationError("five distinct real snapshot sidecars are required; duplication is forbidden")

    temporary = output_dir.with_name(f".{output_dir.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise MaterializationError(f"temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    source_receipts = []
    sidecar_records = []
    request_ids: set[str] = set()
    try:
        for case, source, source_sha in validated_cases:
            raw, sources, sim_stamp_s = _validate_raw_snapshot(
                sidecar_path=source,
                sidecar_sha256=source_sha,
                expected_snapshot_id=case["snapshot_id"],
                contract=contract,
                contract_sha256=expected_contract_sha,
            )
            request_id = str(raw["request_id"])
            if request_id in request_ids:
                raise MaterializationError("five distinct real capture request_id values are required")
            request_ids.add(request_id)
            identity = parse_lane_b_snapshot_id(case["snapshot_id"])
            destination = (
                temporary
                / "cameras"
                / identity.episode_id
                / f"reset-{identity.reset_id}"
                / f"sequence-{identity.sequence_id}"
            )
            image_rows = []
            for view_id, source_image in sources:
                target = destination / f"{view_id}.jpg"
                jpeg = _write_jpeg(source_image, target)
                image_rows.append(
                    {
                        "view_id": view_id,
                        "source_frame_id": next(
                            str(row["frame_id"])
                            for row in raw["cameras"]
                            if row["identity"] == view_id
                        ),
                        "sim_stamp_s": sim_stamp_s,
                        "age_s": 0.0,
                        "width": REV_C_IMAGE_SIZE[0],
                        "height": REV_C_IMAGE_SIZE[1],
                        "pose": list(camera_poses[view_id]),
                        "extrinsic_sha256": extrinsics[view_id],
                        "jpeg_sha256": _sha256_bytes(jpeg),
                        "jpeg_path": str(
                            Path("cameras")
                            / identity.episode_id
                            / f"reset-{identity.reset_id}"
                            / f"sequence-{identity.sequence_id}"
                            / f"{view_id}.jpg"
                        ).replace("\\", "/"),
                    }
                )
            record = {
                "schema_version": 1,
                "kind": SNAPSHOT_KIND,
                "lane_id": "b",
                "episode_id": identity.episode_id,
                "reset_id": identity.reset_id,
                "sequence_id": identity.sequence_id,
                "snapshot_id": identity.snapshot_id,
                "snapshot_sim_stamp_s": sim_stamp_s,
                "written_wall_time_s": _finite(raw.get("wall_time_unix"), "wall_time_unix"),
                "config_sha256": expected_contract_sha,
                "view_order": list(REV_C_VIEW_ORDER),
                "inter_camera_skew_s": 0.0,
                "max_frame_age_s": MAX_FRAME_AGE_S,
                "max_inter_camera_skew_s": MAX_INTER_CAMERA_SKEW_S,
                "images": image_rows,
            }
            record["snapshot_content_sha256"] = snapshot_content_sha256(record)
            sidecar_records.append(record)
            source_receipts.append(
                {
                    "case_id": case["case_id"],
                    "snapshot_id": case["snapshot_id"],
                    "source_snapshot_sidecar": str(source),
                    "source_snapshot_sidecar_sha256": source_sha,
                    "source_request_id": request_id,
                    "source_render_barrier_id": raw["render_barrier_id"],
                    "snapshot_content_sha256": record["snapshot_content_sha256"],
                }
            )

        sidecar_path = temporary / "snapshots.jsonl"
        sidecar_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in sidecar_records),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "kind": MANIFEST_KIND,
            "frozen": True,
            "evaluation_scope": evaluation_scope,
            "candidate_frontier_source": candidate_frontier_source,
            "agent_pose_encoding": agent_pose_encoding,
            "episode_count": FIXED_EPISODE_COUNT,
            "snapshot_sidecar": "snapshots.jsonl",
            "snapshot_sidecar_sha256": _sha256_file(sidecar_path),
            "camera_root": "cameras",
            "expected_config_sha256": expected_contract_sha,
            "expected_extrinsic_sha256": extrinsics,
            "cases": public_cases,
        }
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        manifest_sha = _sha256_file(manifest_path)
        receipt = {
            "schema_version": 1,
            "kind": "t5_lane_b_step3_fixed5_materialization",
            "status": "PASS",
            "evaluation_scope": evaluation_scope,
            "candidate_frontier_source": candidate_frontier_source,
            "agent_pose_encoding": agent_pose_encoding,
            "context_sha256": _sha256_file(contexts_path),
            "revc_contract_sha256": expected_contract_sha,
            "manifest_sha256": manifest_sha,
            "snapshot_sidecar_sha256": manifest["snapshot_sidecar_sha256"],
            "case_count": FIXED_EPISODE_COUNT,
            "jpeg_count": FIXED_EPISODE_COUNT * len(REV_C_VIEW_ORDER),
            "source_captures": source_receipts,
        }
        _write_json(temporary / "materialization.json", receipt)
        os.replace(temporary, output_dir)
        receipt["output_dir"] = str(output_dir)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        required=True,
        help="Frozen JSON with exactly five real capture paths and request contexts.",
    )
    parser.add_argument(
        "--revc-contract",
        type=Path,
        required=True,
        help="Exact completion-sim Rev-C camera contract bound by the contexts SHA.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh output bundle directory; existing paths are rejected.",
    )
    args = parser.parse_args()
    receipt = materialize_bundle(
        contexts_path=args.contexts,
        revc_contract_path=args.revc_contract,
        output_dir=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
