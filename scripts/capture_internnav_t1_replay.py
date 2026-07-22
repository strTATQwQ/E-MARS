#!/usr/bin/env python3
"""Capture and validate a fixed legacy InternNav observation replay.

The recorder wraps the unmodified HTTP AgentClient.  It stores the exact input
arrays before the client serializes them and the returned discrete action after
the server responds.  Raw replay payloads stay in the explicitly selected
remote root; the generated manifest contains only hashes and non-payload
metadata and is safe to copy into the task results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _extract_action(action: Any) -> int:
    try:
        return int(action[0]["action"][0])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"unexpected legacy action structure: {action!r}") from exc


def make_recording_client(legacy_client_class: type) -> type:
    """Return an AgentClient-compatible class that records fixed replay inputs."""

    class RecordingAgentClient:
        def __init__(self, config: Any):
            self._inner = legacy_client_class(config)
            self.agent_name = self._inner.agent_name
            self._root = Path(os.environ["INTERNNAV_T1_REPLAY_ROOT"]).resolve()
            self._maximum = int(os.environ.get("INTERNNAV_T1_REPLAY_STEPS", "120"))
            if self._maximum < 100:
                raise RuntimeError("T1 replay must contain at least 100 steps")
            self._root.mkdir(parents=True, exist_ok=False)
            (self._root / "steps").mkdir()
            self._records_path = self._root / "records.jsonl"
            self._events_path = self._root / "events.jsonl"
            self._global_index = 0
            self._episode_ordinal = 0
            self._generation = 0
            self._sequence = 0
            _atomic_json(
                self._root / "capture_config.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "maximum_steps": self._maximum,
                    "source_mode": "official_discrete_flash_legacy_http",
                    "agent_name": self.agent_name,
                    "checkpoint_revision": os.environ.get(
                        "INTERNNAV_T1_CHECKPOINT_REVISION", ""
                    ),
                    "internnav_commit": os.environ.get("INTERNNAV_T1_COMMIT", ""),
                },
            )

        def step(self, obs: list[dict[str, Any]]) -> Any:
            action = self._inner.step(obs)
            if self._global_index < self._maximum:
                self._record(obs, action)
            self._sequence += 1
            return action

        def reset(self, reset_index: Any = None) -> None:
            self._inner.reset(reset_index)
            event = {
                "schema_version": SCHEMA_VERSION,
                "event": "reset",
                "completed_generation": self._generation,
                "completed_episode_ordinal": self._episode_ordinal,
                "completed_sequence_count": self._sequence,
                "reset_index": _jsonable(reset_index),
            }
            with self._events_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._generation += 1
            self._episode_ordinal += 1
            self._sequence = 0

        def _record(self, obs: list[dict[str, Any]], action: Any) -> None:
            if len(obs) != 1 or not isinstance(obs[0], dict):
                raise RuntimeError(f"expected one observation dictionary, got {type(obs)!r}")
            item = obs[0]
            rgb = np.asarray(item["rgb"])
            depth = np.asarray(item["depth"])
            instruction = str(item["instruction"])
            tokens = np.asarray(item.get("instruction_tokens", []), dtype=np.int64)
            gps = np.asarray(item.get("globalgps", []), dtype=np.float64)
            rotation = np.asarray(item.get("globalrotation", []), dtype=np.float64)
            filename = f"step_{self._global_index:06d}.npz"
            target = self._root / "steps" / filename
            temporary = target.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    rgb=rgb,
                    depth=depth,
                    instruction=np.asarray(instruction),
                    instruction_tokens=tokens,
                    globalgps=gps,
                    globalrotation=rotation,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            record = {
                "schema_version": SCHEMA_VERSION,
                "global_index": self._global_index,
                "episode_ordinal": self._episode_ordinal,
                "reset_generation": self._generation,
                "sequence_id": self._sequence,
                "file": f"steps/{filename}",
                "file_sha256": _sha256(target),
                "observation_pickle_sha256": hashlib.sha256(
                    pickle.dumps(obs, protocol=pickle.HIGHEST_PROTOCOL)
                ).hexdigest(),
                "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
                "rgb_shape": list(rgb.shape),
                "rgb_dtype": str(rgb.dtype),
                "depth_shape": list(depth.shape),
                "depth_dtype": str(depth.dtype),
                "legacy_action": _extract_action(action),
            }
            with self._records_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._global_index += 1
            _atomic_json(
                self._root / "capture_state.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "captured_steps": self._global_index,
                    "current_episode_ordinal": self._episode_ordinal,
                    "current_reset_generation": self._generation,
                    "next_sequence_id": self._sequence + 1,
                },
            )

    return RecordingAgentClient


def finalize(root: Path, minimum: int, expected: int | None) -> dict[str, Any]:
    records_path = root / "records.jsonl"
    if not records_path.is_file():
        raise RuntimeError(f"missing replay records: {records_path}")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    if len(records) < minimum:
        raise RuntimeError(f"captured {len(records)} replay steps, need at least {minimum}")
    if expected is not None and len(records) != expected:
        raise RuntimeError(f"captured {len(records)} replay steps, expected exactly {expected}")

    identifiers: set[tuple[int, int]] = set()
    generation_next: dict[int, int] = {}
    action_counts: dict[str, int] = {}
    digest = hashlib.sha256()
    for index, record in enumerate(records):
        if record["global_index"] != index:
            raise RuntimeError(f"non-contiguous global index at {index}: {record}")
        generation = int(record["reset_generation"])
        sequence = int(record["sequence_id"])
        if sequence != generation_next.get(generation, 0):
            raise RuntimeError(f"non-contiguous sequence in generation {generation}: {sequence}")
        generation_next[generation] = sequence + 1
        identifier = (generation, sequence)
        if identifier in identifiers:
            raise RuntimeError(f"duplicate replay identifier: {identifier}")
        identifiers.add(identifier)
        path = root / record["file"]
        if _sha256(path) != record["file_sha256"]:
            raise RuntimeError(f"replay file hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            rgb = payload["rgb"]
            depth = payload["depth"]
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise RuntimeError(f"invalid RGB contract at step {index}: {rgb.shape}/{rgb.dtype}")
            if depth.shape != (480, 640, 1) or depth.dtype != np.float32:
                raise RuntimeError(f"invalid depth contract at step {index}: {depth.shape}/{depth.dtype}")
            if not np.isfinite(depth).all():
                raise RuntimeError(f"depth NaN/Inf at step {index}")
            if float(depth.min()) < 0.0 or float(depth.max()) > 1.0:
                raise RuntimeError(f"depth outside [0,1] at step {index}")
        action = str(int(record["legacy_action"]))
        action_counts[action] = action_counts.get(action, 0) + 1
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "source_mode": "official_discrete_flash_legacy_http",
        "step_count": len(records),
        "generation_count": len(generation_next),
        "steps_per_generation": {str(key): value for key, value in sorted(generation_next.items())},
        "action_distribution": action_counts,
        "record_manifest_sha256": digest.hexdigest(),
        "payload_location": "external_isaac_host_only",
        "payload_files_verified": len(records),
        "rgb_contract": {"shape": [480, 640, 3], "dtype": "uint8"},
        "depth_contract": {"shape": [480, 640, 1], "dtype": "float32", "range": [0.0, 1.0]},
    }
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--minimum", type=int, default=100)
    parser.add_argument("--expected", type=int)
    args = parser.parse_args()
    print(json.dumps(finalize(args.root.resolve(), args.minimum, args.expected), indent=2))


if __name__ == "__main__":
    main()
