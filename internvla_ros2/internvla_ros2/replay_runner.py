"""Replay the frozen legacy observations through the typed ROS 2 path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy

from .client_node import ClientRuntime
from .protocol import STATUS_OK


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def run_replay(root: Path, output: Path, expected_steps: int) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if manifest.get("status") != "PASS" or int(manifest.get("step_count", -1)) != expected_steps:
        raise RuntimeError("frozen replay manifest is not the expected passing corpus")
    if len(records) != expected_steps:
        raise RuntimeError(f"expected {expected_steps} records, observed {len(records)}")

    record_digest = hashlib.sha256()
    for index, record in enumerate(records):
        if int(record["global_index"]) != index:
            raise RuntimeError(f"non-contiguous global index at {index}")
        record_digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    if record_digest.hexdigest() != manifest["record_manifest_sha256"]:
        raise RuntimeError("records.jsonl does not match the frozen manifest")

    runtime = ClientRuntime()
    started_wall = time.time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAIL",
        "source_mode": "frozen_legacy_observation_replay",
        "transport_mode": "typed_ros2_dds",
        "expected_steps": expected_steps,
        "completed_steps": 0,
        "action_matches": 0,
        "action_mismatches": [],
        "protocol_failures": [],
        "trajectory_valid_count": 0,
        "trajectory_invalid_count": 0,
        "trajectory_finite_count": 0,
        "trajectory_nonempty_count": 0,
        "trajectory_frame_counts": {},
        "valid_trajectory_frame_counts": {},
        "trajectory_point_counts": [],
        "action_source_counts": {},
        "selected_action_counts": {},
        "nav2_goal_sent_count": 0,
        "nav2_plan_valid_count": 0,
        "reset_count": 0,
        "reset_pollution_count": 0,
        "started_unix": started_wall,
    }
    end_to_end: list[float] = []
    inference: list[float] = []
    current_generation: int | None = None
    try:
        runtime.start()
        health_before = runtime.node.health()
        report["health_before"] = health_before
        first_generation = int(records[0]["reset_generation"])
        if first_generation != 0:
            raise RuntimeError("frozen replay must begin at reset generation zero")
        runtime.node.initialize("frozen-replay-generation-0")
        current_generation = 0

        for index, record in enumerate(records):
            generation = int(record["reset_generation"])
            sequence = int(record["sequence_id"])
            if generation != current_generation:
                if generation != int(current_generation) + 1 or sequence != 0:
                    raise RuntimeError(
                        f"invalid generation transition {current_generation}->{generation} at {index}"
                    )
                reset = runtime.node.reset(f"frozen-replay-generation-{generation}")
                report["reset_count"] += 1
                if int(reset["reset_generation"]) != generation:
                    report["reset_pollution_count"] += 1
                    raise RuntimeError("reset generation response mismatch")
                current_generation = generation

            payload_path = root / str(record["file"])
            if _sha256(payload_path) != str(record["file_sha256"]):
                raise RuntimeError(f"payload hash mismatch at replay step {index}")
            with np.load(payload_path, allow_pickle=False) as payload:
                rgb = payload["rgb"]
                depth = payload["depth"]
                instruction = str(payload["instruction"].item())
                tokens = [int(value) for value in payload["instruction_tokens"].tolist()]
                gps = [float(value) for value in payload["globalgps"].tolist()]
                rotation = [float(value) for value in payload["globalrotation"].tolist()]

            step_started = time.perf_counter()
            try:
                result = runtime.node.step_arrays(
                    rgb=rgb,
                    depth=depth,
                    instruction=instruction,
                    instruction_tokens=tokens,
                    global_gps=gps,
                    global_rotation=rotation,
                    sequence_id=sequence,
                    request_id=f"frozen:{generation}:{sequence}",
                )
            except BaseException as exc:
                report["protocol_failures"].append(
                    {"global_index": index, "error": repr(exc)[:1024]}
                )
                raise
            end_to_end.append(float(time.perf_counter() - step_started))
            inference.append(float(result["inference_latency_sec"]))
            report["completed_steps"] += 1

            if (
                int(result["reset_generation"]) != generation
                or int(result["sequence_id"]) != sequence
            ):
                report["reset_pollution_count"] += 1
            selected_action = int(result["discrete_action"])
            observed_action = int(result.get("model_discrete_action", selected_action))
            expected_action = int(record["legacy_action"])
            selected_key = str(selected_action)
            selected_counts = report["selected_action_counts"]
            selected_counts[selected_key] = int(selected_counts.get(selected_key, 0)) + 1
            report["nav2_goal_sent_count"] += int(bool(result.get("nav2_goal_sent", False)))
            report["nav2_plan_valid_count"] += int(bool(result.get("nav2_plan_valid", False)))
            report["control_mode"] = str(result.get("control_mode", "model"))
            if observed_action == expected_action:
                report["action_matches"] += 1
            else:
                report["action_mismatches"].append(
                    {
                        "global_index": index,
                        "reset_generation": generation,
                        "sequence_id": sequence,
                        "expected": expected_action,
                        "observed_model": observed_action,
                        "selected": selected_action,
                    }
                )
            path = np.asarray(result["local_path"], dtype=np.float64)
            frame = str(result["local_path_frame"])
            frames = report["trajectory_frame_counts"]
            frames[frame] = int(frames.get(frame, 0)) + 1
            source_key = str(int(result["action_source"]))
            sources = report["action_source_counts"]
            sources[source_key] = int(sources.get(source_key, 0)) + 1
            if bool(result["trajectory_valid"]):
                report["trajectory_valid_count"] += 1
                valid_frames = report["valid_trajectory_frame_counts"]
                valid_frames[frame] = int(valid_frames.get(frame, 0)) + 1
                report["trajectory_point_counts"].append(int(len(path)))
                if int(result["trajectory_source"]) != 1:
                    report["protocol_failures"].append(
                        {"global_index": index, "error": "valid trajectory has wrong source"}
                    )
            else:
                report["trajectory_invalid_count"] += 1
                if int(result["trajectory_source"]) != 0 or path.size:
                    report["protocol_failures"].append(
                        {"global_index": index, "error": "absent trajectory carries payload/source"}
                    )
            if path.size > 0:
                report["trajectory_nonempty_count"] += 1
            if path.ndim == 2 and path.shape[1:] == (2,) and np.isfinite(path).all():
                report["trajectory_finite_count"] += 1

        report["health_after"] = runtime.node.health()
    except BaseException as exc:
        report["fatal_error"] = repr(exc)[:2048]
    finally:
        try:
            runtime.node._safe_stop(0, "replay runner complete")
        except BaseException:
            pass
        runtime.stop()

    report["ended_unix"] = time.time()
    report["duration_sec"] = float(report["ended_unix"] - started_wall)
    report["action_match_rate"] = float(report["action_matches"] / expected_steps)
    report["end_to_end_latency_sec"] = {
        "count": len(end_to_end),
        "mean": float(np.mean(end_to_end)) if end_to_end else 0.0,
        "p50": _percentile(end_to_end, 50),
        "p95": _percentile(end_to_end, 95),
        "max": max(end_to_end, default=0.0),
    }
    report["model_inference_latency_sec"] = {
        "count": len(inference),
        "mean": float(np.mean(inference)) if inference else 0.0,
        "p50": _percentile(inference, 50),
        "p95": _percentile(inference, 95),
        "max": max(inference, default=0.0),
    }
    passing = (
        report["completed_steps"] == expected_steps
        and report["action_matches"] == expected_steps
        and not report["action_mismatches"]
        and not report["protocol_failures"]
        and report["trajectory_valid_count"] > 0
        and report["trajectory_invalid_count"] + report["trajectory_valid_count"] == expected_steps
        and report["trajectory_nonempty_count"] == report["trajectory_valid_count"]
        and report["trajectory_finite_count"] == report["trajectory_valid_count"]
        and report["trajectory_frame_counts"] == {"base_link": expected_steps}
        and report["valid_trajectory_frame_counts"]
        == {"base_link": report["trajectory_valid_count"]}
        and report["reset_pollution_count"] == 0
        and report["reset_count"] == int(manifest["generation_count"]) - 1
    )
    report["status"] = "PASS" if passing else "FAIL"
    _atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=120)
    arguments, ros_arguments = parser.parse_known_args()
    rclpy.init(args=ros_arguments)
    report = run_replay(
        arguments.replay_root.resolve(),
        arguments.output.resolve(),
        arguments.expected_steps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
