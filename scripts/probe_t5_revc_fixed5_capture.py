#!/usr/bin/env python3
"""Capture one real same-render-tick Rev-C snapshot for each frozen episode."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import probe_t5_revc_snapshot_smoke as smoke  # noqa: E402


PROFILE = "lane_b_revc_fixed5_capture"
CAPTURE_COUNT = 5
RENDER_REALIGN_RETRY_LIMIT = 4
RENDER_REALIGN_ERROR = re.compile(
    r"Rev-C sensor t5_revc_(?:front_left|front|front_right|rear) "
    r"ReferenceTime changed at paused render barrier"
)


class Fixed5CaptureError(smoke.SmokeContractError):
    """Raised when five unique natural episode captures cannot be proven."""


def make_request(
    episode_id: str,
    run_token: str,
    capture_index: int,
    attempt: int,
    *,
    reset_generation: int = 0,
    sequence_id: int = 0,
) -> dict[str, Any]:
    request_id = (
        f"b::revc-fixed5::{run_token}::{capture_index}::{attempt}::{time.time_ns()}"
    )
    if len(request_id) > 160:
        raise Fixed5CaptureError("generated request_id exceeds the protocol bound")
    return {
        "schema_version": 1,
        "request_id": request_id,
        "episode_id": episode_id,
        "reset_generation": reset_generation,
        "sequence_id": sequence_id,
    }


def _active_identity(ack: dict[str, Any]) -> tuple[str, int, int]:
    episode_id = ack.get("episode_id")
    reset_generation = ack.get("reset_generation")
    sequence_id = ack.get("sequence_id")
    if (
        not isinstance(episode_id, str)
        or not episode_id.startswith("b::")
        or isinstance(reset_generation, bool)
        or not isinstance(reset_generation, int)
        or reset_generation < 0
        or isinstance(sequence_id, bool)
        or not isinstance(sequence_id, int)
        or sequence_id < 0
    ):
        raise Fixed5CaptureError("snapshot ack has invalid active execution identity")
    return episode_id, reset_generation, sequence_id


def classify_retry(
    request: dict[str, Any],
    ack: dict[str, Any],
    ordered_episode_ids: list[str],
    capture_index: int,
) -> tuple[str, int, int]:
    """Return retry disposition and the next reset/sequence identity."""

    status = ack.get("status")
    active_episode, active_reset, active_sequence = _active_identity(ack)
    target_episode = request["episode_id"]
    if status == "WARN_CAPTURE_FAILED":
        if (
            active_episode != target_episode
            or active_reset != request["reset_generation"]
            or active_sequence != request["sequence_id"]
        ):
            raise Fixed5CaptureError(
                "capture-failed ack changed execution identity"
            )
        error = ack.get("error")
        if not isinstance(error, str) or RENDER_REALIGN_ERROR.fullmatch(error) is None:
            raise Fixed5CaptureError(f"snapshot ack is not CAPTURED: {status}")
        return "render_realign", active_reset, active_sequence
    if status == "RATE_LIMITED":
        if (
            active_episode != target_episode
            or active_reset != request["reset_generation"]
            or active_sequence != request["sequence_id"]
        ):
            raise Fixed5CaptureError("rate-limited ack changed execution identity")
        return "rate_limited", active_reset, active_sequence
    if status != "IDENTITY_MISMATCH":
        raise Fixed5CaptureError(f"snapshot ack is not CAPTURED: {status}")

    active_plain = active_episode[3:]
    if active_plain not in ordered_episode_ids:
        raise Fixed5CaptureError("active episode is outside the frozen fixed-five")
    active_index = ordered_episode_ids.index(active_plain)
    if active_index > capture_index:
        raise Fixed5CaptureError("natural evaluation advanced past an uncaptured episode")
    if active_index < capture_index:
        return "wait_for_episode", 0, 0
    if active_episode != target_episode:
        raise Fixed5CaptureError("target episode identity is inconsistent with order")
    return "sync_identity", active_reset, active_sequence


def _remove_regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise Fixed5CaptureError(f"{label} is not a regular file")
    path.unlink()


def _validate_paths(args: argparse.Namespace) -> Path:
    if args.result_root.is_symlink() or not args.result_root.is_dir():
        raise Fixed5CaptureError("result root must be a regular directory")
    result_root = args.result_root.resolve()
    for path in (args.request, args.ack, args.output):
        if not path.resolve().is_relative_to(result_root):
            raise Fixed5CaptureError("request, ack, and output must stay in result root")
    if args.request.name != "revc_snapshot.request.json":
        raise Fixed5CaptureError("request filename violates the Rev-C contract")
    if args.ack.name != "revc_snapshot.ack.json":
        raise Fixed5CaptureError("ack filename violates the Rev-C contract")
    if any(path.exists() or path.is_symlink() for path in (args.request, args.ack)):
        raise Fixed5CaptureError("snapshot request and ack paths must be fresh")
    snapshot_root = result_root / "revc_snapshots"
    if snapshot_root.exists() or snapshot_root.is_symlink():
        raise Fixed5CaptureError("fixed-five snapshot root was not fresh")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_token):
        raise Fixed5CaptureError("run token is invalid")
    return result_root


def run(args: argparse.Namespace) -> dict[str, Any]:
    result_root = _validate_paths(args)
    deadline = time.monotonic() + args.timeout_sec
    order = smoke.wait_for_order_manifest(
        args.order_manifest, max(0.01, deadline - time.monotonic())
    )
    ordered_ids = list(order["ordered_episode_ids"])
    captures: list[dict[str, Any]] = []
    execution_identities: set[tuple[str, int, int]] = set()
    attempts: list[dict[str, Any]] = []

    for capture_index, plain_episode_id in enumerate(ordered_ids):
        target_episode = f"b::{plain_episode_id}"
        reset_generation = 0
        sequence_id = 0
        attempt = 0
        render_realign_retries = 0
        while time.monotonic() < deadline:
            request = make_request(
                target_episode,
                args.run_token,
                capture_index,
                attempt,
                reset_generation=reset_generation,
                sequence_id=sequence_id,
            )
            smoke.publish_request(args.request, request)
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise Fixed5CaptureError("fixed-five capture deadline expired")
            ack = smoke.wait_for_matching_ack(args.ack, request["request_id"], remaining)
            attempt_row = {
                "capture_index": capture_index,
                "attempt": attempt,
                "request_id": request["request_id"],
                "requested_episode_id": target_episode,
                "requested_reset_generation": reset_generation,
                "requested_sequence_id": sequence_id,
                "ack_status": ack.get("status"),
                "active_episode_id": ack.get("episode_id"),
                "active_reset_generation": ack.get("reset_generation"),
                "active_sequence_id": ack.get("sequence_id"),
                "ack_error": ack.get("error"),
            }
            attempts.append(attempt_row)
            if ack.get("status") == "CAPTURED":
                identity = _active_identity(ack)
                if identity[0] != target_episode or identity in execution_identities:
                    raise Fixed5CaptureError("capture execution identity was reused or stale")
                capture = smoke.validate_capture(
                    result_root,
                    args.contract,
                    request,
                    ack,
                    expected_sidecar_count=capture_index + 1,
                    profile=PROFILE,
                )
                capture["capture_index"] = capture_index
                capture["ordered_episode_id"] = plain_episode_id
                capture["attempt_count"] = attempt + 1
                captures.append(capture)
                execution_identities.add(identity)
                _remove_regular(args.ack, "captured snapshot ack")
                break

            disposition, reset_generation, sequence_id = classify_retry(
                request, ack, ordered_ids, capture_index
            )
            attempt_row["retry_disposition"] = disposition
            if disposition == "render_realign":
                render_realign_retries += 1
                if render_realign_retries > RENDER_REALIGN_RETRY_LIMIT:
                    raise Fixed5CaptureError(
                        "paused render identity did not stabilize within the "
                        "finite realignment limit"
                    )
            _remove_regular(args.ack, "snapshot retry ack")
            attempt += 1
            if disposition == "rate_limited":
                time.sleep(min(args.retry_interval_sec, max(0.0, deadline - time.monotonic())))
            elif disposition == "wait_for_episode":
                time.sleep(min(args.episode_poll_sec, max(0.0, deadline - time.monotonic())))
            elif disposition == "render_realign":
                # The failed paused barrier has aligned a lagging Replicator
                # ReferenceTime to the current simulation state.  Re-request
                # on the next producer sample; the producer still requires all
                # four post-barrier identities to be equal and same-tick.
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            else:
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        else:
            raise Fixed5CaptureError("fixed-five capture deadline expired")

    sidecars = sorted((result_root / "revc_snapshots").glob("*/snapshot.json"))
    snapshot_dirs = sorted(path for path in (result_root / "revc_snapshots").iterdir())
    if (
        len(captures) != CAPTURE_COUNT
        or len(execution_identities) != CAPTURE_COUNT
        or len(sidecars) != CAPTURE_COUNT
        or len(snapshot_dirs) != CAPTURE_COUNT
        or any(path.is_symlink() or not path.is_dir() for path in snapshot_dirs)
        or {path.parent for path in sidecars} != set(snapshot_dirs)
    ):
        raise Fixed5CaptureError("fixed-five capture did not produce five unique snapshots")
    if [capture["ordered_episode_id"] for capture in captures] != ordered_ids:
        raise Fixed5CaptureError("captured episode order differs from the frozen manifest")
    return {
        "schema_version": 1,
        "status": "PASS",
        "profile": PROFILE,
        "lane": "b",
        "capture_count": CAPTURE_COUNT,
        "ordered_episode_ids": ordered_ids,
        "unique_execution_identities": True,
        "same_render_tick_all": all(
            capture.get("same_render_tick") is True for capture in captures
        ),
        "camera_order": list(smoke.CAMERA_ORDER),
        "contract_sha256": captures[0]["contract_sha256"],
        "snapshots": captures,
        "request_attempt_count": len(attempts),
        "request_attempts": attempts,
        "recorded_unix": time.time(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--order-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--ack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--timeout-sec", type=float, default=24000.0)
    parser.add_argument("--episode-poll-sec", type=float, default=2.0)
    parser.add_argument("--retry-interval-sec", type=float, default=1.1)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 300.0 <= args.timeout_sec <= 24000.0:
        raise SystemExit("--timeout-sec must be between 300 and 24000")
    if not 0.25 <= args.episode_poll_sec <= 10.0:
        raise SystemExit("--episode-poll-sec must be between 0.25 and 10")
    if not 1.0 <= args.retry_interval_sec <= 10.0:
        raise SystemExit("--retry-interval-sec must be between 1 and 10")
    try:
        payload = run(args)
        exit_code = 0
    except Exception as error:
        payload = {
            "schema_version": 1,
            "status": "FAIL",
            "profile": PROFILE,
            "lane": "b",
            "error_type": type(error).__name__,
            "error": str(error)[:500],
            "recorded_unix": time.time(),
        }
        exit_code = 75
    try:
        smoke._atomic_json(args.output, payload)
    except OSError as error:
        print(f"failed to write Rev-C fixed-five result: {error}", file=sys.stderr)
        return 74
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
