#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_ROOT = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler"
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
for path in (SCHEDULER_ROOT, PKG_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaac_vln_benchmark.perception_planning_suite import evaluate_tracking_sequences, tracking_gate
from omninav_step_scheduler.step_roles import TargetTrackState


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay genuine per-frame Step observations through TargetTrackState.")
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sequences = load_json(Path(args.sequences))
    observations = load_json(Path(args.observations))
    by_key = {
        (str(row["sequence_id"]), int(row["frame_index"])): row
        for row in observations
        if isinstance(row, dict)
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for sequence in sequences:
        tracker = TargetTrackState(
            required_hits=int(sequence.get("required_hits", 2)),
            high_confidence_single_hit=1.01,
            min_confidence=0.60,
            max_misses=int(sequence.get("max_misses", 2)),
            max_age_sec=3.0,
        )
        was_lost = False
        was_confirmed = False
        for frame in sequence["frames"]:
            key = (str(sequence["sequence_id"]), int(frame["frame_index"]))
            observation = by_key.get(key)
            if observation is None:
                missing.append({"sequence_id": key[0], "frame_index": key[1]})
                continue
            track = tracker.update(
                timestamp=float(observation["timestamp"]),
                episode_id=str(sequence["episode_id"]),
                target=str(sequence["target"]),
                visible=bool(observation["target_visible"]),
                confidence=float(observation["confidence"]),
                visible_in_view=str(observation.get("visible_in_view", "front")),
                frame_seq=int(observation["frame_seq"]),
            )
            lost = bool(was_confirmed and not track["confirmed"])
            if lost:
                was_lost = True
            reacquired = bool(was_lost and track["confirmed"])
            confirmed_event = bool(not was_confirmed and track["confirmed"] and not reacquired)
            first_seen = bool(
                track.get("first_seen_time") is not None
                and abs(float(track["first_seen_time"]) - float(observation["timestamp"])) <= 1.0e-6
            )
            rows.append(
                {
                    "sequence_id": sequence["sequence_id"],
                    "family": sequence["family"],
                    "episode_id": sequence["episode_id"],
                    "target": sequence["target"],
                    "frame_index": frame["frame_index"],
                    "frame_seq": observation["frame_seq"],
                    "visible_truth": frame["visible"],
                    "occluded": frame["occluded"],
                    "track_episode_id": track["episode_id"],
                    "track_target": track["target"],
                    "hits": track["hits"],
                    "misses": track["misses"],
                    "confirmed": track["confirmed"],
                    "track_stale": not bool(track["confirmed"]),
                    "first_seen": first_seen,
                    "confirmed_event": confirmed_event,
                    "lost": lost,
                    "reacquired": reacquired,
                    "action_triggered": bool(observation.get("action_triggered", False)),
                    "confidence": track["confidence"],
                }
            )
            was_confirmed = bool(track["confirmed"])
            if reacquired:
                was_lost = False
    metrics = evaluate_tracking_sequences(rows)
    gate = tracking_gate(metrics)
    if missing:
        gate["pass"] = False
        gate["failures"].append(f"missing observations: {len(missing)}")
    (output / "tracking_results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (output / "tracking_missing_observations.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
    (output / "tracking_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
