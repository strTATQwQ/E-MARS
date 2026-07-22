#!/usr/bin/env python3
"""Legacy-primary / typed-ROS-shadow AgentClient and result finalizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from internvla_ipc_agent_client import ROS2IPCAgentClient


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _extract(action: Any) -> int:
    return int(action[0]["action"][0])


def _observation_hash(obs: list[dict[str, Any]]) -> str:
    item = obs[0]
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(item["rgb"], dtype=np.uint8).tobytes())
    digest.update(np.ascontiguousarray(item["depth"], dtype="<f4").tobytes())
    digest.update(str(item["instruction"]).encode("utf-8"))
    return digest.hexdigest()


def make_shadow_client(legacy_client_class: type) -> type:
    class ShadowAgentClient:
        def __init__(self, config: Any):
            self.legacy = legacy_client_class(config)
            self.shadow = ROS2IPCAgentClient(config)
            self.agent_name = self.legacy.agent_name
            self.root = Path(os.environ["INTERNVLA_SHADOW_RESULT_DIR"]).resolve()
            self.root.mkdir(parents=True, exist_ok=False)
            self.records_path = self.root / "records.jsonl"
            self.summary_path = self.root / "summary.json"
            self.episode_ordinal = 0
            self.sequence = 0
            self.count = 0
            self.matches = 0
            self.trajectory_valid = 0
            self.trajectory_finite = 0
            self.reset_count = 0
            self.started = time.time()
            self._write_summary("IN_PROGRESS")

        def step(self, obs: list[dict[str, Any]]) -> Any:
            legacy_started = time.perf_counter()
            legacy_action = self.legacy.step(obs)
            legacy_latency = time.perf_counter() - legacy_started
            shadow_started = time.perf_counter()
            shadow_action = self.shadow.step(obs)
            shadow_latency = time.perf_counter() - shadow_started
            if self.shadow.last_error:
                raise RuntimeError(f"typed ROS shadow failed: {self.shadow.last_error}")
            result = self.shadow.last_result or {}
            legacy_value = _extract(legacy_action)
            shadow_value = _extract(shadow_action)
            matched = legacy_value == shadow_value
            path = np.asarray(result.get("local_path", []), dtype=np.float32)
            valid = bool(result.get("trajectory_valid", False))
            finite = bool(
                valid and path.ndim == 2 and path.shape[1:] == (2,) and path.size and np.isfinite(path).all()
            )
            record = {
                "schema_version": 1,
                "global_index": self.count,
                "episode_ordinal": self.episode_ordinal,
                "sequence_id": self.sequence,
                "observation_sha256": _observation_hash(obs),
                "legacy_action": legacy_value,
                "shadow_action": shadow_value,
                "matched": matched,
                "legacy_latency_sec": legacy_latency,
                "shadow_end_to_end_latency_sec": shadow_latency,
                "shadow_inference_latency_sec": float(result.get("inference_latency_sec", 0.0)),
                "action_source": int(result.get("action_source", 0)),
                "trajectory_source": int(result.get("trajectory_source", 0)),
                "trajectory_valid": valid,
                "trajectory_finite": finite,
                "trajectory_frame": str(result.get("local_path_frame", "")),
                "trajectory_points": int(len(path)) if path.ndim == 2 else 0,
                "trajectory_sha256": hashlib.sha256(
                    np.ascontiguousarray(path, dtype="<f4").tobytes()
                ).hexdigest()
                if valid
                else "",
            }
            with self.records_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.count += 1
            self.sequence += 1
            self.matches += int(matched)
            self.trajectory_valid += int(valid)
            self.trajectory_finite += int(finite)
            self._write_summary("PASS_SO_FAR" if matched else "MISMATCH")
            return legacy_action

        def reset(self, reset_index: Any = None) -> None:
            self.legacy.reset(reset_index)
            self.shadow.reset(reset_index)
            self.reset_count += 1
            self.episode_ordinal += 1
            self.sequence = 0
            self._write_summary("PASS_SO_FAR")

        def _write_summary(self, status: str) -> None:
            _atomic_json(
                self.summary_path,
                {
                    "schema_version": 1,
                    "status": status,
                    "transport": "legacy_primary_typed_ros2_shadow",
                    "step_count": self.count,
                    "action_matches": self.matches,
                    "action_match_rate": self.matches / self.count if self.count else 0.0,
                    "trajectory_valid_count": self.trajectory_valid,
                    "trajectory_finite_count": self.trajectory_finite,
                    "reset_count": self.reset_count,
                    "completed_episode_ordinals": self.episode_ordinal,
                    "started_unix": self.started,
                    "updated_unix": time.time(),
                },
            )

    return ShadowAgentClient


def finalize(root: Path, evaluator_result: Path, expected_episodes: int) -> dict[str, Any]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    metrics = json.loads(evaluator_result.read_text(encoding="utf-8"))
    split = metrics.get("val_unseen", {})
    valid_records = [record for record in records if record["trajectory_valid"]]
    passing = (
        len(records) > 0
        and int(summary["step_count"]) == len(records)
        and all(record["matched"] for record in records)
        and valid_records
        and all(
            record["trajectory_finite"]
            and record["trajectory_frame"] == "base_link"
            and record["trajectory_points"] == 33
            and record["trajectory_source"] == 1
            for record in valid_records
        )
        and int(summary["reset_count"]) == expected_episodes - 1
        and int(split.get("Count", 0)) == expected_episodes
    )
    summary.update(
        {
            "status": "PASS" if passing else "FAIL",
            "action_matches": sum(int(record["matched"]) for record in records),
            "action_match_rate": sum(int(record["matched"]) for record in records) / len(records),
            "trajectory_valid_count": len(valid_records),
            "trajectory_finite_count": sum(int(record["trajectory_finite"]) for record in records),
            "evaluator_metrics": split,
            "expected_episodes": expected_episodes,
            "finalized_unix": time.time(),
        }
    )
    _atomic_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", type=Path, required=True)
    parser.add_argument("--evaluator-result", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=5)
    arguments = parser.parse_args()
    result = finalize(
        arguments.finalize.resolve(),
        arguments.evaluator_result.resolve(),
        arguments.expected_episodes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
