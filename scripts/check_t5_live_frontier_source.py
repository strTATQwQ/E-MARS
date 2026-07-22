#!/usr/bin/env python3
"""Emit a machine-readable gate for the coordinator-owned live frontier source."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slow_planner.live_frontier import (  # noqa: E402
    LiveFrontierContractError,
    LiveNav2FrontierFile,
)
from slow_planner.lane_b import parse_lane_b_snapshot_id  # noqa: E402


def _write_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sim-now-s", type=float, required=True)
    parser.add_argument("--max-age-s", type=float, default=5.0)
    parser.add_argument("--expected-snapshot-id", default="")
    args = parser.parse_args()

    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "t5_live_nav2_frontier_source_gate",
        "status": "BLOCKED",
        "blocker_code": "UNKNOWN_LIVE_NAV2_FRONTIER_FAILURE",
        "source_path": str(args.snapshot),
        "motion_authority": "none",
        "terminal_stop_authority": "none",
        "bounded_advisor_request_admission": False,
        "live_navigation_promotion_eligible": False,
        "recorded_wall_time_s": time.time(),
    }
    exit_code = 75
    try:
        snapshot = LiveNav2FrontierFile(args.snapshot).read()
        expected = (
            parse_lane_b_snapshot_id(args.expected_snapshot_id)
            if args.expected_snapshot_id
            else snapshot.identity
        )
        snapshot.require_current(
            sim_now_s=args.sim_now_s,
            max_age_s=args.max_age_s,
            expected_identity=expected,
        )
        if not snapshot.candidate_frontiers:
            raise LiveFrontierContractError(
                "live source supplied no legal frontier", code="NO_CURRENT_LEGAL_FRONTIERS"
            )
        payload.update(
            {
                "status": "SOURCE_READY",
                "blocker_code": None,
                "snapshot_id": snapshot.identity.snapshot_id,
                "frontier_set_sha256": snapshot.frontier_set_sha256,
                "candidate_frontier_ids": [
                    item.frontier_id for item in snapshot.candidate_frontiers
                ],
                "candidate_frontier_count": len(snapshot.candidate_frontiers),
                "bounded_advisor_request_admission": True,
                "live_navigation_promotion_eligible": False,
            }
        )
        exit_code = 0
    except LiveFrontierContractError as exc:
        payload.update(
            {
                "status": "BLOCKED",
                "blocker_code": exc.code,
                "detail": str(exc),
            }
        )
    except (TypeError, ValueError) as exc:
        payload.update(
            {
                "status": "BLOCKED",
                "blocker_code": "INVALID_LIVE_NAV2_FRONTIER_SOURCE",
                "detail": str(exc),
            }
        )
    _write_atomic(args.output, payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
