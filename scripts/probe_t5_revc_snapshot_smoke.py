#!/usr/bin/env python3
"""Issue and validate one Lane-B Rev-C snapshot during a bounded T5 canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any


CAMERA_ORDER = ("front_left", "front", "front_right", "rear")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class SmokeContractError(RuntimeError):
    """Raised when the bounded Rev-C smoke contract is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_regular_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SmokeContractError(f"{label} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SmokeContractError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise SmokeContractError(f"{label} must contain one JSON object")
    return value


def _scoped_regular(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise SmokeContractError(f"{label} must be a non-empty POSIX relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise SmokeContractError(f"{label} escapes the run root")
    unresolved = root / candidate
    if unresolved.is_symlink():
        raise SmokeContractError(f"{label} must not be a symlink")
    path = unresolved.resolve()
    if not path.is_relative_to(root):
        raise SmokeContractError(f"{label} escapes the run root")
    if path.is_symlink() or not path.is_file():
        raise SmokeContractError(f"{label} must be a regular non-symlink file")
    return path


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def wait_for_order_manifest(path: Path, timeout_sec: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_error = "not created"
    while time.monotonic() < deadline:
        try:
            value = _load_regular_json(path, "episode-order manifest")
            ordered_ids = value.get("ordered_episode_ids")
            if (
                value.get("schema_version") == 1
                and value.get("status") == "PASS"
                and value.get("dataset_episode_count") == 5
                and isinstance(ordered_ids, list)
                and len(ordered_ids) == 5
                and len(set(ordered_ids)) == 5
                and all(isinstance(item, str) and item for item in ordered_ids)
            ):
                return value
            last_error = "manifest does not bind the frozen five"
        except SmokeContractError as error:
            last_error = str(error)
        time.sleep(0.05)
    raise SmokeContractError(f"episode-order manifest timeout: {last_error}")


def make_request(
    order: dict[str, Any], run_token: str, *, sequence_id: int = 0, attempt: int = 0
) -> dict[str, Any]:
    ordered_ids = order["ordered_episode_ids"]
    request_id = f"b::revc-smoke::{run_token}::{attempt}::{time.time_ns()}"
    if len(request_id) > 160:
        raise SmokeContractError("generated request_id exceeds the protocol bound")
    return {
        "schema_version": 1,
        "request_id": request_id,
        "episode_id": f"b::{ordered_ids[0]}",
        "reset_generation": 0,
        "sequence_id": sequence_id,
    }


def publish_request(path: Path, request: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SmokeContractError("snapshot request path was not fresh")
    _atomic_json(path, request)


def wait_for_matching_ack(
    path: Path, request_id: str, timeout_sec: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_status = "not created"
    while time.monotonic() < deadline:
        try:
            value = _load_regular_json(path, "snapshot ack")
            last_status = str(value.get("status", "missing status"))
            if value.get("request_id") == request_id:
                return value
        except SmokeContractError as error:
            last_status = str(error)
        time.sleep(0.05)
    raise SmokeContractError(f"matching snapshot ack timeout: {last_status}")


def _next_sequence_after_identity_mismatch(
    request: dict[str, Any], ack: dict[str, Any]
) -> int:
    if ack.get("status") != "IDENTITY_MISMATCH":
        raise SmokeContractError(f"snapshot ack is not CAPTURED: {ack.get('status')}")
    if (
        ack.get("mismatched_field") != "sequence_id"
        or ack.get("episode_id") != request.get("episode_id")
        or ack.get("reset_generation") != 0
        or isinstance(ack.get("sequence_id"), bool)
        or not isinstance(ack.get("sequence_id"), int)
        or ack["sequence_id"] <= request.get("sequence_id", -1)
    ):
        raise SmokeContractError(
            "identity mismatch crossed episode/reset or did not advance sequence"
        )
    return int(ack["sequence_id"])


def _png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    if not payload.startswith(PNG_SIGNATURE):
        raise SmokeContractError(f"invalid PNG signature: {path.name}")
    offset = len(PNG_SIGNATURE)
    dimensions: tuple[int, int] | None = None
    color_type: int | None = None
    idat = bytearray()
    saw_iend = False
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise SmokeContractError(f"truncated PNG chunk: {path.name}")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(payload):
            raise SmokeContractError(f"truncated PNG data: {path.name}")
        chunk = payload[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length : chunk_end])[0]
        actual_crc = zlib.crc32(chunk_type + chunk) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise SmokeContractError(f"invalid PNG CRC: {path.name}")
        if chunk_type == b"IHDR":
            if dimensions is not None or length != 13:
                raise SmokeContractError(f"invalid PNG IHDR: {path.name}")
            dimensions = struct.unpack(">II", chunk[:8])
            color_type = chunk[9]
            if chunk[8] != 8 or color_type not in {2, 6} or chunk[10:] != b"\0\0\0":
                raise SmokeContractError(f"PNG is not 8-bit RGB/RGBA: {path.name}")
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        elif chunk_type == b"IEND":
            if length != 0 or chunk_end != len(payload):
                raise SmokeContractError(f"invalid PNG IEND: {path.name}")
            saw_iend = True
        offset = chunk_end
    if dimensions is None or color_type is None or not idat or not saw_iend:
        raise SmokeContractError(f"incomplete PNG evidence: {path.name}")
    channels = 3 if color_type == 2 else 4
    expected_row_bytes = dimensions[0] * channels
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(bytes(idat), dimensions[1] * (expected_row_bytes + 1) + 1)
    except zlib.error as error:
        raise SmokeContractError(f"invalid PNG deflate stream: {path.name}") from error
    if (
        len(raw) != dimensions[1] * (expected_row_bytes + 1)
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
        or any(raw[row * (expected_row_bytes + 1)] > 4 for row in range(dimensions[1]))
    ):
        raise SmokeContractError(f"invalid PNG scanline payload: {path.name}")
    return dimensions


def validate_capture(
    result_root: Path,
    contract_path: Path,
    request: dict[str, Any],
    ack: dict[str, Any],
    *,
    expected_sidecar_count: int = 1,
    profile: str = "lane_b_revc_smoke",
    expected_lane: str = "b",
) -> dict[str, Any]:
    if expected_lane not in {"a", "b"}:
        raise SmokeContractError("expected lane must be a or b")
    result_root = result_root.resolve()
    contract = _load_regular_json(contract_path.resolve(), "Rev-C contract")
    expected_contract_sha = _sha256(contract_path.resolve())
    expected_identity = {
        "request_id": request["request_id"],
        "episode_id": request["episode_id"],
        "reset_generation": request["reset_generation"],
        "sequence_id": request["sequence_id"],
    }
    if ack.get("schema_version") != 1 or ack.get("status") != "CAPTURED":
        raise SmokeContractError(f"snapshot ack is not CAPTURED: {ack.get('status')}")
    for key, value in expected_identity.items():
        if type(ack.get(key)) is not type(value) or ack.get(key) != value:
            raise SmokeContractError(f"snapshot ack {key} does not match request")
    sidecar_path = _scoped_regular(result_root, ack.get("sidecar"), "sidecar")
    sidecar = _load_regular_json(sidecar_path, "snapshot sidecar")
    for key, value in expected_identity.items():
        if type(sidecar.get(key)) is not type(value) or sidecar.get(key) != value:
            raise SmokeContractError(f"snapshot sidecar {key} does not match request")
    if (
        sidecar.get("schema_version") != 1
        or sidecar.get("contract_id") != contract.get("contract_id")
        or sidecar.get("contract_sha256") != expected_contract_sha
        or sidecar.get("scope") != "completion_sim_only"
        or sidecar.get("same_render_tick") is not True
        or sidecar.get("sim_stamp_before_ns") != sidecar.get("sim_stamp_after_ns")
        or not isinstance(sidecar.get("sim_stamp_before_ns"), int)
        or isinstance(sidecar.get("sim_stamp_before_ns"), bool)
        or sidecar.get("sim_stamp_before_ns", 0) <= 0
        or sidecar.get("camera_order") != list(CAMERA_ORDER)
        or sidecar.get("cuvslam_stereo_is_separate") is not True
    ):
        raise SmokeContractError("snapshot sidecar violates the frozen Rev-C scope")
    preview_hz = sidecar.get("external_preview_max_hz")
    if (
        isinstance(preview_hz, bool)
        or not isinstance(preview_hz, (int, float))
        or not 0.0 < float(preview_hz) <= 1.0
    ):
        raise SmokeContractError("external preview exceeds 1 Hz")
    render_identity = sidecar.get("render_identity_value")
    if isinstance(render_identity, dict):
        numerator = render_identity.get("referenceTimeNumerator")
        denominator = render_identity.get("referenceTimeDenominator")
        rendering_time = render_identity.get("rendering_time")
    else:
        numerator = denominator = rendering_time = None
    if (
        sidecar.get("render_identity_source")
        != "replicator_annotator:ReferenceTime+SimulationManager"
        or not isinstance(render_identity, dict)
        or ack.get("render_identity_value") != render_identity
        or ack.get("render_barrier_id") != sidecar.get("render_barrier_id")
        or ack.get("render_identity_source")
        != sidecar.get("render_identity_source")
        or isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
        or isinstance(rendering_time, bool)
        or not isinstance(rendering_time, (int, float))
        or not math.isfinite(float(rendering_time))
        or not math.isclose(
            float(rendering_time),
            numerator / denominator if isinstance(numerator, int) else math.nan,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise SmokeContractError("ack and sidecar render identities differ")

    cameras = sidecar.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != len(CAMERA_ORDER):
        raise SmokeContractError("snapshot must contain exactly four cameras")
    image_evidence: list[dict[str, Any]] = []
    contract_cameras = contract.get("cameras")
    if not isinstance(contract_cameras, list) or len(contract_cameras) != 4:
        raise SmokeContractError("Rev-C contract camera list is invalid")
    optics = contract.get("optics")
    if not isinstance(optics, dict):
        raise SmokeContractError("Rev-C contract optics are invalid")
    for index, (identity, camera, expected_camera) in enumerate(
        zip(CAMERA_ORDER, cameras, contract_cameras, strict=True)
    ):
        if (
            not isinstance(camera, dict)
            or not isinstance(expected_camera, dict)
            or camera.get("identity") != identity
            or camera.get("order_index") != index
            or camera.get("sensor_name") != expected_camera.get("sensor_name")
            or camera.get("frame_id") != expected_camera.get("frame_id")
            or camera.get("prim_path") != expected_camera.get("prim_path")
            or camera.get("position_F_M_mm")
            != expected_camera.get("position_F_M_mm")
            or camera.get("yaw_deg") != expected_camera.get("yaw_deg")
            or camera.get("pitch_down_deg") != optics.get("pitch_down_deg")
            or camera.get("hfov_deg") != optics.get("hfov_deg")
            or camera.get("resolution") != optics.get("resolution")
            or camera.get("encoding") != "png_rgb8"
            or camera.get("render_identity") != render_identity
        ):
            raise SmokeContractError(f"camera {identity} metadata is invalid")
        image_path = _scoped_regular(
            result_root, camera.get("path"), f"camera {identity} image"
        )
        byte_count = image_path.stat().st_size
        if (
            camera.get("bytes") != byte_count
            or camera.get("sha256") != _sha256(image_path)
            or _png_dimensions(image_path) != (640, 480)
        ):
            raise SmokeContractError(f"camera {identity} PNG evidence is invalid")
        image_evidence.append(
            {
                "identity": identity,
                "path": image_path.relative_to(result_root).as_posix(),
                "sha256": camera["sha256"],
                "bytes": byte_count,
            }
        )
    if expected_sidecar_count < 1:
        raise SmokeContractError("expected snapshot sidecar count must be positive")
    sidecars = sorted((result_root / "revc_snapshots").glob("*/snapshot.json"))
    if len(sidecars) != expected_sidecar_count or sidecar_path not in sidecars:
        raise SmokeContractError(
            f"{profile} must produce exactly {expected_sidecar_count} snapshot sidecars"
        )
    return {
        "schema_version": 1,
        "status": "PASS",
        "profile": profile,
        "lane": expected_lane,
        "request": request,
        "ack_canonical_sha256": hashlib.sha256(
            json.dumps(ack, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "sidecar": sidecar_path.relative_to(result_root).as_posix(),
        "sidecar_sha256": _sha256(sidecar_path),
        "contract_sha256": expected_contract_sha,
        "render_barrier_id": sidecar["render_barrier_id"],
        "render_identity": render_identity,
        "same_render_tick": True,
        "external_preview_max_hz": float(preview_hz),
        "camera_order": list(CAMERA_ORDER),
        "images": image_evidence,
        "recorded_unix": time.time(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.result_root.is_symlink() or not args.result_root.is_dir():
        raise SmokeContractError("result root must be a regular directory")
    result_root = args.result_root.resolve()
    for path in (args.request, args.ack, args.output):
        if not path.resolve().is_relative_to(result_root):
            raise SmokeContractError("request, ack, and output must stay in result root")
    if args.request.name != "revc_snapshot.request.json":
        raise SmokeContractError("request filename violates the Rev-C contract")
    if args.ack.name != "revc_snapshot.ack.json":
        raise SmokeContractError("ack filename violates the Rev-C contract")
    if args.ack.exists() or args.ack.is_symlink():
        raise SmokeContractError("snapshot ack path was not fresh")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_token):
        raise SmokeContractError("run token is invalid")
    deadline = time.monotonic() + args.timeout_sec
    order = wait_for_order_manifest(
        args.order_manifest, max(0.01, deadline - time.monotonic())
    )
    sequence_id = 0
    attempts: list[dict[str, Any]] = []
    for attempt in range(3):
        request = make_request(
            order, args.run_token, sequence_id=sequence_id, attempt=attempt
        )
        publish_request(args.request, request)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise SmokeContractError("snapshot smoke deadline expired")
        ack = wait_for_matching_ack(args.ack, request["request_id"], remaining)
        attempts.append(
            {
                "attempt": attempt,
                "request_id": request["request_id"],
                "sequence_id": request["sequence_id"],
                "ack_status": ack.get("status"),
                "active_sequence_id": ack.get("sequence_id"),
            }
        )
        if ack.get("status") == "CAPTURED":
            payload = validate_capture(result_root, args.contract, request, ack)
            payload["identity_sync_attempt_count"] = attempt
            payload["request_attempts"] = attempts
            return payload
        sequence_id = _next_sequence_after_identity_mismatch(request, ack)
        if args.ack.is_symlink() or not args.ack.is_file():
            raise SmokeContractError("identity mismatch ack is not a regular file")
        args.ack.unlink()
    raise SmokeContractError("snapshot identity did not stabilize within three attempts")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--order-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--ack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 30.0 <= args.timeout_sec <= 900.0:
        raise SystemExit("--timeout-sec must be between 30 and 900")
    try:
        payload = run(args)
        exit_code = 0
    except Exception as error:
        payload = {
            "schema_version": 1,
            "status": "FAIL",
            "profile": "lane_b_revc_smoke",
            "lane": "b",
            "error_type": type(error).__name__,
            "error": str(error)[:500],
            "recorded_unix": time.time(),
        }
        exit_code = 75
    try:
        _atomic_json(args.output, payload)
    except OSError as error:
        print(f"failed to write Rev-C smoke result: {error}", file=sys.stderr)
        return 74
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
