#!/usr/bin/env python3
"""Run one fixed Rev-C four-view request through the bounded Lane-B adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Sequence

from PIL import Image

from slow_planner.base import CandidateFrontier, OrderedImage, SlowPlannerRequest
from slow_planner.client import SlowPlannerClient
from slow_planner.lane_b import (
    LaneBPlannerAdapter,
    LaneBPlannerMode,
    REV_C_VIEW_ORDER,
    frontend_decision_record,
)


DEADLINE_MS = 12_000
WARMUP_TIMEOUT_MS = 180_000
IMAGE_SIZE = (640, 480)
CAMERA_POSES = {
    "front_left": (0.170, 0.051962, 0.200, math.radians(60.0), math.radians(-10.0)),
    "front": (0.200, 0.0, 0.200, 0.0, math.radians(-10.0)),
    "front_right": (0.170, -0.051962, 0.200, math.radians(-60.0), math.radians(-10.0)),
    "rear": (0.080, 0.0, 0.200, math.pi, math.radians(-10.0)),
}


def _load_images(paths: Sequence[Path]) -> tuple[OrderedImage, ...]:
    if len(paths) != len(REV_C_VIEW_ORDER):
        raise ValueError("replay requires exactly four ordered Rev-C JPEGs")
    rows: list[OrderedImage] = []
    for view_id, path in zip(REV_C_VIEW_ORDER, paths, strict=True):
        payload = path.read_bytes()
        if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise ValueError(f"{view_id} input is not a complete JPEG")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != IMAGE_SIZE:
                raise ValueError(
                    f"{view_id} must be {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
                )
        rows.append(
            OrderedImage(
                view_id=view_id,
                pose=CAMERA_POSES[view_id],
                jpeg=payload,
                width=IMAGE_SIZE[0],
                height=IMAGE_SIZE[1],
            )
        )
    return tuple(rows)


def _request(
    images: tuple[OrderedImage, ...], *, episode_id: str, sequence_id: int
) -> SlowPlannerRequest:
    return SlowPlannerRequest(
        episode_id=episode_id,
        snapshot_id=f"b::{episode_id}::0::{sequence_id}",
        instruction="Walk to the nearest visible doorway without entering an obstacle.",
        ordered_images=images,
        candidate_frontiers=(
            CandidateFrontier(0, (-0.8, 1.2), 1.44, -33.7),
            CandidateFrontier(1, (0.0, 1.8), 1.8, 0.0),
            CandidateFrontier(2, (0.8, 1.2), 1.44, 33.7),
        ),
        agent_pose=(0.0, 0.0, 0.0, 0.0),
        timestamp=time.time(),
    )


def run_replay(
    *,
    endpoint: str,
    paths: Sequence[Path],
    mode: LaneBPlannerMode,
    episode_id: str,
    sequence_id: int,
    deadline_ms: int = DEADLINE_MS,
) -> dict[str, object]:
    if deadline_ms not in {DEADLINE_MS, WARMUP_TIMEOUT_MS}:
        raise ValueError("replay deadline must be the production deadline or fixed warmup timeout")
    images = _load_images(paths)
    request = _request(images, episode_id=episode_id, sequence_id=sequence_id)
    adapter = LaneBPlannerAdapter(mode)
    started = time.perf_counter()
    model_status = "RESPONSE"
    metrics = None
    try:
        with SlowPlannerClient(endpoint, timeout_ms=deadline_ms) as client:
            decision, metrics = client.decide(request)
        outcome = adapter.resolve(request, decision)
    except Exception as error:  # deterministic safe fallback is part of the contract
        model_status = "TIMEOUT" if type(error).__name__ == "Again" else "ERROR"
        reason = (
            "step3_deadline_exceeded"
            if model_status == "TIMEOUT"
            else "step3_service_failure"
        )
        outcome = adapter.resolve_failure(request, reason)
    wall_ms = (time.perf_counter() - started) * 1000.0
    record = frontend_decision_record(outcome, metrics)
    public = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_version": request.protocol_version,
        "deadline_ms": deadline_ms,
        "wall_ms": wall_ms,
        "model_request_status": model_status,
        "deadline_met": model_status == "RESPONSE" and wall_ms <= deadline_ms,
        "view_order": list(REV_C_VIEW_ORDER),
        "images": [
            {
                "view_id": image.view_id,
                "width": image.width,
                "height": image.height,
                "jpeg_sha256": hashlib.sha256(image.jpeg).hexdigest(),
            }
            for image in images
        ],
        **record,
    }
    encoded = json.dumps(public, sort_keys=True)
    for forbidden in ("raw_text", "chain_of_thought", "reasoning_content"):
        if forbidden in encoded:
            raise RuntimeError(f"public replay result leaked forbidden field {forbidden}")
    return public


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:8200")
    parser.add_argument("--mode", choices=[item.value for item in LaneBPlannerMode], required=True)
    parser.add_argument("--episode-id", default="step3-replay")
    parser.add_argument("--sequence-id", type=int, default=0)
    parser.add_argument(
        "--request-class", choices=("production", "warmup"), default="production"
    )
    parser.add_argument("--image", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.episode_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in args.episode_id
    ):
        raise ValueError("episode-id must be a Lane-B-safe identifier")
    if args.sequence_id < 0:
        raise ValueError("sequence-id must be non-negative")
    payload = run_replay(
        endpoint=args.endpoint,
        paths=args.image,
        mode=LaneBPlannerMode(args.mode),
        episode_id=args.episode_id,
        sequence_id=args.sequence_id,
        deadline_ms=(
            DEADLINE_MS if args.request_class == "production" else WARMUP_TIMEOUT_MS
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
