#!/usr/bin/env python3
"""Build a deterministic, post-run T5 replay timeline.

The input is one *materialized* local coordinator run root.  This tool only
reads an explicit allow-list of replay evidence below ``remote/x86`` and
``remote/dgx``.  It never scans shell logs, environment captures, command
lines, or credential files.

Simulation stamps authored by x86 are authoritative.  A record without an
explicit stamp may inherit one only through a unique request_id or snapshot_id
anchor.  Everything else remains visibly unaligned in the index instead of
being assigned an inferred time.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
CANONICAL_FIELDS = (
    "schema_version",
    "event_id",
    "event_type",
    "run_id",
    "lane",
    "episode_id",
    "reset_generation",
    "sequence_id",
    "request_id",
    "snapshot_id",
    "sim_stamp_ns",
    "wall_time_unix_ns",
    "wall_monotonic_ns",
    "source_host",
    "source_path",
    "payload",
    "artifacts",
)

_IDENTITY_FIELDS = {
    "episode_id",
    "reset_generation",
    "sequence_id",
    "trigger_sequence_id",
    "source_sequence",
    "request_id",
    "trigger_request_id",
    "snapshot_id",
    "sim_stamp_ns",
    "snapshot_sim_stamp_ns",
    "source_sim_stamp_ns",
    "sim_stamp_before_ns",
    "sim_stamp_after_ns",
    "wall_time_unix_ns",
    "wall_time_unix",
    "wall_monotonic_ns",
    "client_wall_monotonic_ns",
    "timestamp",
}
_SENSITIVE_KEYS = {
    "api_key",
    "argv",
    "auth",
    "authorization",
    "command",
    "cookie",
    "credential",
    "credentials",
    "env",
    "environ",
    "environment",
    "openai_api_key",
    "passwd",
    "password",
    "secret",
    "token",
}
_SENSITIVE_SUFFIXES = ("_api_key", "_credential", "_password", "_secret", "_token")
_SAFE_VIEW_IDS = {"front_left", "front", "front_right", "rear"}
_CONTROLLER_PAYLOAD_FIELDS = (
    "update_index",
    "identity_valid",
    "state_only",
    "desired_linear_x",
    "desired_angular_z",
    "actual_linear_velocity_base",
    "actual_linear_velocity_world",
    "actual_angular_velocity",
    "command_age_sec",
    "command_fresh",
    "command_timeout",
    "motion_enabled",
    "emergency_stop",
    "collision_monitor_stopping",
    "physical_collision",
    "physical_collision_warn_only",
    "fallen",
    "nan_detected",
    "pose_xyz_wxyz",
    "heading_error_rad",
    "lateral_error_m",
    "maximum_collision_force",
    "point_count",
)


class ReplayBuildError(ValueError):
    """The materialized run is malformed or violates replay path safety."""


@dataclass(frozen=True)
class SourceSpec:
    name: str
    ordinal: int
    event_type: str
    paths: tuple[Path, ...]
    private: bool = False


@dataclass
class EventCandidate:
    source_name: str
    source_ordinal: int
    source_path: str
    source_host: str
    source_line: int
    event_type: str
    run_id: str
    lane: str | None
    episode_id: str | None
    reset_generation: int | None
    sequence_id: int | None
    request_id: str | None
    snapshot_id: str | None
    sim_stamp_ns: int | None
    wall_time_unix_ns: int | None
    wall_monotonic_ns: int | None
    payload: dict[str, Any]
    artifacts: list[dict[str, Any]]
    wall_latencies_ms: dict[str, float]
    explicit_sim_stamp: bool
    alignment_method: str | None = None


def _reject_constant(value: str) -> None:
    raise ReplayBuildError(f"non-finite JSON constant is forbidden: {value}")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplayBuildError(f"invalid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayBuildError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any], str]]:
    try:
        stream = path.open("r", encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReplayBuildError(f"cannot read {path}: {exc}") from exc
    with stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ReplayBuildError) as exc:
                raise ReplayBuildError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ReplayBuildError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, value, raw.rstrip("\r\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line_sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_source(path: Path, root: Path) -> tuple[Path, str]:
    if path.is_symlink():
        raise ReplayBuildError(f"source symlink is forbidden: {path}")
    resolved = path.resolve(strict=True)
    if not _is_within(resolved, root):
        raise ReplayBuildError(f"source path escapes run root: {path}")
    return resolved, resolved.relative_to(root).as_posix()


def _safe_artifact(
    value: Any,
    *,
    root: Path,
    base: Path,
) -> tuple[str, bool] | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ReplayBuildError("artifact path must be a non-empty relative string")
    text = value.strip()
    if "\\" in text or re.match(r"^[A-Za-z]:", text):
        raise ReplayBuildError(f"artifact path is not portable and relative: {text!r}")
    relative = Path(text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ReplayBuildError(f"artifact path escapes its materialized root: {text!r}")
    candidate = (base / relative).resolve(strict=False)
    if not _is_within(candidate, root):
        raise ReplayBuildError(f"artifact path escapes run root: {text!r}")
    if candidate.is_symlink():
        raise ReplayBuildError(f"artifact symlink is forbidden: {text!r}")
    return candidate.relative_to(root).as_posix(), candidate.is_file()


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _text(value: Any) -> str | None:
    if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value):
        return str(value)
    return None


def _nested_objects(row: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    yield row
    for key in ("identity", "request", "decision", "payload", "metadata"):
        value = row.get(key)
        if isinstance(value, Mapping):
            yield value


def _first_text(row: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for obj in _nested_objects(row):
        for name in names:
            result = _text(obj.get(name))
            if result is not None:
                return result
    return None


def _first_integer(row: Mapping[str, Any], names: Sequence[str]) -> int | None:
    for obj in _nested_objects(row):
        for name in names:
            result = _integer(obj.get(name))
            if result is not None:
                return result
    return None


def _snapshot_identity(snapshot_id: str | None) -> tuple[str | None, int | None, int | None]:
    if snapshot_id is None:
        return None, None, None
    parts = snapshot_id.split("::")
    if len(parts) < 4:
        return None, None, None
    try:
        generation = int(parts[-2])
        sequence = int(parts[-1])
    except ValueError:
        return None, None, None
    episode = "::".join(parts[1:-2])
    return (episode or None), generation, sequence


def _sim_stamp(row: Mapping[str, Any]) -> int | None:
    for name in (
        "sim_stamp_ns",
        "snapshot_sim_stamp_ns",
        "source_sim_stamp_ns",
        "sim_stamp_before_ns",
        "sim_stamp_after_ns",
    ):
        value = _first_integer(row, (name,))
        if value is not None:
            if value < 0:
                raise ReplayBuildError(f"{name} must be nonnegative")
            return value
    return None


def _unix_ns(row: Mapping[str, Any]) -> int | None:
    value = _first_integer(row, ("wall_time_unix_ns",))
    if value is not None:
        return value
    for obj in _nested_objects(row):
        for name in ("wall_time_unix", "timestamp"):
            candidate = obj.get(name)
            if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
                continue
            number = float(candidate)
            if not math.isfinite(number):
                raise ReplayBuildError(f"{name} must be finite")
            return int(round(number * 1_000_000_000))
    return None


def _monotonic_ns(row: Mapping[str, Any]) -> int | None:
    return _first_integer(row, ("wall_monotonic_ns", "client_wall_monotonic_ns"))


def _is_sensitive_key(key: str) -> bool:
    lowered = key.casefold()
    return lowered in _SENSITIVE_KEYS or lowered.endswith(_SENSITIVE_SUFFIXES)


def _sanitize(value: Any, *, top_level: bool = False) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if _is_sensitive_key(name) or (top_level and name in _IDENTITY_FIELDS):
                continue
            result[name] = _sanitize(item)
        return result
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReplayBuildError("payload contains a non-finite number")
        return value
    return str(value)


def _latency_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayBuildError(f"{label} wall latency must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ReplayBuildError(f"{label} wall latency must be finite and nonnegative")
    return result


def _first_latency_value(row: Mapping[str, Any], field: str) -> float | None:
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping) and field in metrics:
        return _latency_number(metrics[field], f"metrics.{field}")
    return _latency_number(row.get(field), field)


def _private_wall_timing(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return only safe wall-clock scalars from a private trace row."""

    candidates: list[Mapping[str, Any]] = [row]
    for name in ("timing", "metrics"):
        value = row.get(name)
        if isinstance(value, Mapping):
            candidates.append(value)
    pairs = (
        ("received_wall_monotonic_ns", "completed_wall_monotonic_ns"),
        ("request_received_wall_monotonic_ns", "response_completed_wall_monotonic_ns"),
        ("received_wall_time_unix_ns", "completed_wall_time_unix_ns"),
        ("request_received_wall_time_unix_ns", "response_completed_wall_time_unix_ns"),
        ("received_wall_ns", "completed_wall_ns"),
        ("request_received_ns", "response_completed_ns"),
        ("received_ns", "completed_ns"),
    )
    for candidate in candidates:
        for received_name, completed_name in pairs:
            received = _integer(candidate.get(received_name))
            completed = _integer(candidate.get(completed_name))
            if received is None and completed is None:
                continue
            if received is None or completed is None:
                raise ReplayBuildError(
                    f"private Step3 timing requires both {received_name} and {completed_name}"
                )
            if received < 0 or completed < received:
                raise ReplayBuildError("private Step3 received/completed wall ns are invalid")
            return {
                "received_field": received_name,
                "received_ns": received,
                "completed_field": completed_name,
                "completed_ns": completed,
                "duration_ns": completed - received,
            }
    return {}


def _wall_latencies(
    source_name: str, row: Mapping[str, Any]
) -> tuple[dict[str, float], dict[str, Any]]:
    """Normalize only measured wall-duration fields; never substitute sim time."""

    values: dict[str, float] = {}
    private_timing: dict[str, Any] = {}
    if source_name == "step3_timeout_advice":
        seconds = _latency_number(
            row.get("service_wall_latency_sec"), "service_wall_latency_sec"
        )
        if seconds is not None:
            values["step3.advisor_service"] = seconds * 1000.0
    elif source_name == "step3_public_trace":
        for field, component in (
            ("server_total_ms", "step3.server_total"),
            ("end_to_end_ms", "step3.end_to_end"),
            ("model_generate_ms", "step3.model_generate"),
            ("prefill_ttft_ms", "step3.prefill_ttft"),
            ("decode_ms", "step3.decode"),
        ):
            milliseconds = _first_latency_value(row, field)
            if milliseconds is not None:
                values[component] = milliseconds
    elif source_name == "step3_private_trace":
        private_timing = _private_wall_timing(row)
        if private_timing:
            values["step3.private_trace_response"] = private_timing["duration_ns"] / 1e6
    elif source_name == "client_records":
        for field, component in (
            ("inference_latency_sec", "internvla.inference"),
            ("action_round_trip_latency_sec", "internvla.action_round_trip"),
            ("network_ros_residual_latency_sec", "internvla.network_ros_residual"),
            ("observation_encode_latency_sec", "internvla.observation_encode"),
        ):
            seconds = _latency_number(row.get(field), field)
            if seconds is not None:
                values[component] = seconds * 1000.0
        for field, value in row.items():
            lowered = str(field).casefold()
            match = re.fullmatch(r"nav2_(.+_)?latency_(sec|ms)", lowered)
            if match is None:
                continue
            amount = _latency_number(value, str(field))
            if amount is None:
                continue
            milliseconds = amount * 1000.0 if match.group(2) == "sec" else amount
            component_name = (
                lowered[: -len("_latency_sec")]
                if lowered.endswith("_latency_sec")
                else lowered[: -len("_latency_ms")]
            )
            values[f"internvla.{component_name}"] = milliseconds
    return values, private_timing


def _public_payload(source_name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    # Controller records can be hundreds of megabytes because each update may
    # repeat collision geometry and point arrays.  Those remain authenticated
    # by the source SHA in the index; the replay projects only time-series
    # state required for synchronized review.
    if source_name == "controller_records":
        selected = {field: row[field] for field in _CONTROLLER_PAYLOAD_FIELDS if field in row}
        return _sanitize(selected)
    return _sanitize(row, top_level=True)


def _metadata(root: Path) -> tuple[str, str | None]:
    for name in ("input_binding.json", "fast_lane_final_summary.json", "fast_lane_summary.json"):
        path = root / name
        if not path.is_file():
            continue
        value = _load_object(path)
        run_id = _text(value.get("run_id"))
        lane = _text(value.get("lane"))
        if run_id is None and isinstance(value.get("request"), Mapping):
            run_id = _text(value["request"].get("run_id"))
            lane = lane or _text(value["request"].get("lane"))
        if run_id is not None:
            return run_id, lane.casefold() if lane else None
    return root.name, None


def _discover(root: Path) -> list[SourceSpec]:
    found: dict[Path, tuple[str, int, str, bool]] = {}

    def add(path: Path, name: str, ordinal: int, event_type: str, private: bool = False) -> None:
        if path.is_file() or path.is_symlink():
            found.setdefault(path, (name, ordinal, event_type, private))

    for host in ("x86", "dgx"):
        remote = root / "remote" / host
        if not remote.is_dir():
            continue
        for path in remote.rglob("frames.jsonl"):
            lowered = "/".join(part.casefold() for part in path.parts)
            if any(token in lowered for token in ("d435_rgb_5hz", "/d435/")):
                add(path, "d435_full_frames", 0, "d435_rgb_frame")
            elif any(
                token in lowered
                for token in (
                    "full_rgb_5hz",
                    "model_observation",
                    "model-observation",
                    "model_obs",
                )
            ):
                add(path, "model_observation_frames", 1, "model_observation_frame")

    evaluator = root / "remote" / "x86" / "evaluator"
    snapshots = evaluator / "revc_snapshots"
    if snapshots.is_dir():
        for path in snapshots.rglob("*.json"):
            if path.name in {"snapshot.json", "sidecar.json", "snapshot.sidecar.json"}:
                add(path, "revc_snapshot_sidecar", 2, "revc_snapshot")
    if evaluator.is_dir():
        for path in evaluator.rglob("step3_timeout_advice.jsonl"):
            add(path, "step3_timeout_advice", 3, "step3_timeout_advice")
        for path in evaluator.rglob("events.jsonl"):
            if path.parent.name == "task_state":
                add(path, "task_state_events", 4, "task_state_event")

    dgx = root / "remote" / "dgx"
    if dgx.is_dir():
        for path in dgx.rglob("*.jsonl"):
            lowered_parts = [part.casefold() for part in path.parts]
            lowered_name = path.name.casefold()
            if "step3" in "/".join(lowered_parts) and (
                lowered_name.startswith("step3")
                or "trace" in lowered_name
                or any(
                    "trace" in part or "private" in part or "public" in part
                    for part in lowered_parts
                )
            ):
                private = any("private" in part for part in lowered_parts)
                add(
                    path,
                    "step3_private_trace" if private else "step3_public_trace",
                    6 if private else 5,
                    "step3_private_trace" if private else "step3_public_trace",
                    private,
                )
        for path in dgx.rglob("client_records.jsonl"):
            add(path, "client_records", 7, "client_record")
        for path in dgx.rglob("*.jsonl"):
            if "motion_gate" in path.name.casefold():
                add(path, "motion_gate_records", 8, "motion_gate_record")
        for path in dgx.rglob("controller_records.jsonl"):
            add(path, "controller_records", 9, "controller_record")

    grouped: dict[tuple[str, int, str, bool], list[Path]] = defaultdict(list)
    for path, definition in found.items():
        grouped[definition].append(path)
    specs = [
        SourceSpec(name, ordinal, event_type, tuple(sorted(paths)), private)
        for (name, ordinal, event_type, private), paths in grouped.items()
    ]
    return sorted(specs, key=lambda item: (item.ordinal, item.name))


def _artifact_base(source_name: str, path: Path, root: Path) -> Path:
    if source_name == "revc_snapshot_sidecar":
        return root / "remote" / "x86" / "evaluator"
    return path.parent


def _artifacts(
    source_name: str,
    row: Mapping[str, Any],
    *,
    path: Path,
    root: Path,
    private: bool,
) -> list[dict[str, Any]]:
    if private:
        return [
            {
                "kind": "private_trace",
                "path": path.relative_to(root).as_posix(),
                "present": True,
            }
        ]
    base = _artifact_base(source_name, path, root)
    result: list[dict[str, Any]] = []
    if source_name in {"d435_full_frames", "model_observation_frames"}:
        normalized = _safe_artifact(row.get("path"), root=root, base=base)
        if normalized is not None:
            artifact_path, present = normalized
            result.append({"kind": "frame", "path": artifact_path, "present": present})
    elif source_name == "revc_snapshot_sidecar":
        cameras = row.get("cameras")
        if cameras is not None and not isinstance(cameras, list):
            raise ReplayBuildError(f"snapshot cameras must be a list: {path}")
        for camera in cameras or []:
            if not isinstance(camera, Mapping):
                raise ReplayBuildError(f"snapshot camera entry must be an object: {path}")
            normalized = _safe_artifact(camera.get("path"), root=root, base=base)
            if normalized is None:
                continue
            artifact_path, present = normalized
            view_id = _text(camera.get("identity")) or _text(camera.get("view_id"))
            kind = "revc_camera" if view_id in _SAFE_VIEW_IDS else "observer_camera"
            artifact = {
                "kind": kind,
                "view_id": view_id,
                "path": artifact_path,
                "present": present,
            }
            for field in ("encoding", "sha256", "bytes", "resolution"):
                if field in camera:
                    artifact[field] = _sanitize(camera[field])
            result.append(artifact)
        for field in ("observer", "observer_camera", "third_person", "third_person_camera"):
            observer = row.get(field)
            if observer is None:
                continue
            if not isinstance(observer, Mapping):
                raise ReplayBuildError(f"snapshot {field} must be an object: {path}")
            normalized = _safe_artifact(observer.get("path"), root=root, base=base)
            if normalized is None:
                continue
            artifact_path, present = normalized
            artifact = {
                "kind": "observer_camera",
                "view_id": _text(observer.get("identity")) or "observer",
                "path": artifact_path,
                "present": present,
            }
            for key in ("encoding", "sha256", "bytes", "resolution"):
                if key in observer:
                    artifact[key] = _sanitize(observer[key])
            result.append(artifact)
    return result


def _candidate(
    *,
    spec: SourceSpec,
    path: Path,
    source_path: str,
    line_number: int,
    row: dict[str, Any],
    raw: str,
    root: Path,
    run_id: str,
    lane: str | None,
) -> EventCandidate:
    snapshot_id = _first_text(row, ("snapshot_id",))
    episode_id = _first_text(row, ("episode_id",))
    reset_generation = _first_integer(row, ("reset_generation", "generation"))
    sequence_id = _first_integer(row, ("sequence_id", "trigger_sequence_id", "source_sequence"))
    parsed_episode, parsed_reset, parsed_sequence = _snapshot_identity(snapshot_id)
    episode_id = episode_id or parsed_episode
    reset_generation = reset_generation if reset_generation is not None else parsed_reset
    sequence_id = sequence_id if sequence_id is not None else parsed_sequence
    if (
        snapshot_id is None
        and spec.name == "revc_snapshot_sidecar"
        and episode_id is not None
        and reset_generation is not None
        and sequence_id is not None
    ):
        snapshot_lane = lane
        if snapshot_lane is None:
            possible_lane = episode_id.split("::", 1)[0].casefold()
            snapshot_lane = possible_lane if possible_lane in {"a", "b"} else None
        if snapshot_lane is not None:
            # Runtime protocol v1 uses lane::episode::reset::sequence.  Older
            # sidecars omitted the redundant string while DGX traces kept it.
            snapshot_id = (
                f"{snapshot_lane}::{episode_id}::{reset_generation}::{sequence_id}"
            )
    request_id = _first_text(row, ("request_id", "trigger_request_id"))
    sim_stamp_ns = _sim_stamp(row)
    source_host = "x86" if source_path.startswith("remote/x86/") else "dgx"
    event_type = spec.event_type
    if spec.name == "task_state_events" and isinstance(row.get("event"), str):
        event_type = f"task_state.{row['event']}"
    wall_latencies_ms, private_timing = _wall_latencies(spec.name, row)
    if spec.private:
        payload: dict[str, Any] = {
            "private_trace_reference_only": True,
            "source_line": line_number,
            "record_sha256": _line_sha256(raw),
        }
        if private_timing:
            payload["private_wall_timing"] = private_timing
    else:
        payload = _public_payload(spec.name, row)
    if wall_latencies_ms:
        payload["wall_latencies_ms"] = dict(sorted(wall_latencies_ms.items()))
    return EventCandidate(
        source_name=spec.name,
        source_ordinal=spec.ordinal,
        source_path=source_path,
        source_host=source_host,
        source_line=line_number,
        event_type=event_type,
        run_id=run_id,
        lane=lane,
        episode_id=episode_id,
        reset_generation=reset_generation,
        sequence_id=sequence_id,
        request_id=request_id,
        snapshot_id=snapshot_id,
        sim_stamp_ns=sim_stamp_ns,
        wall_time_unix_ns=_unix_ns(row),
        wall_monotonic_ns=_monotonic_ns(row),
        payload=payload,
        artifacts=_artifacts(spec.name, row, path=path, root=root, private=spec.private),
        wall_latencies_ms=wall_latencies_ms,
        explicit_sim_stamp=sim_stamp_ns is not None,
    )


def _unique_anchor(values: Iterable[EventCandidate], field: str) -> dict[str, int]:
    observed: dict[str, set[int]] = defaultdict(set)
    for event in values:
        identity = getattr(event, field)
        if identity is not None and event.sim_stamp_ns is not None:
            observed[identity].add(event.sim_stamp_ns)
    return {
        identity: next(iter(stamps))
        for identity, stamps in observed.items()
        if len(stamps) == 1
    }


def _align(events: list[EventCandidate]) -> None:
    snapshots = _unique_anchor(events, "snapshot_id")
    requests = _unique_anchor(events, "request_id")
    for event in events:
        if event.sim_stamp_ns is not None:
            continue
        if event.snapshot_id in snapshots:
            event.sim_stamp_ns = snapshots[event.snapshot_id]
            event.alignment_method = "unique_snapshot_id"
        elif event.request_id in requests:
            event.sim_stamp_ns = requests[event.request_id]
            event.alignment_method = "unique_request_id"
        if event.alignment_method is not None:
            event.payload["timeline_alignment"] = {
                "method": event.alignment_method,
                "stamp_authority": "x86_sim_stamp",
            }


def _sort_key(event: EventCandidate) -> tuple[Any, ...]:
    return (
        event.sim_stamp_ns is None,
        event.sim_stamp_ns if event.sim_stamp_ns is not None else 0,
        event.source_ordinal,
        event.source_path,
        event.source_line,
    )


def _event_dict(event: EventCandidate, ordinal: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": f"t5-replay-{ordinal:09d}",
        "event_type": event.event_type,
        "run_id": event.run_id,
        "lane": event.lane,
        "episode_id": event.episode_id,
        "reset_generation": event.reset_generation,
        "sequence_id": event.sequence_id,
        "request_id": event.request_id,
        "snapshot_id": event.snapshot_id,
        "sim_stamp_ns": event.sim_stamp_ns,
        "wall_time_unix_ns": event.wall_time_unix_ns,
        "wall_monotonic_ns": event.wall_monotonic_ns,
        "source_host": event.source_host,
        "source_path": event.source_path,
        "payload": event.payload,
        "artifacts": event.artifacts,
    }


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
    else:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ReplayBuildError("cannot calculate a percentile of no values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _wall_latency_summary(events: Sequence[EventCandidate]) -> dict[str, Any]:
    samples: dict[str, list[float]] = defaultdict(list)
    for event in events:
        for component, value in event.wall_latencies_ms.items():
            samples[component].append(value)
    return {
        "timebase": "measured_wall_duration_only",
        "unit": "milliseconds",
        "sim_time_substitution": False,
        "percentile_method": "linear_interpolation",
        "components": {
            component: {
                "count": len(values),
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
                "max_ms": max(values),
            }
            for component, values in sorted(samples.items())
        },
    }


def build_replay(run_root: Path) -> dict[str, Any]:
    """Build ``replay/timeline.jsonl`` and its deterministic index."""

    if run_root.is_symlink():
        raise ReplayBuildError(f"run root symlink is forbidden: {run_root}")
    root = run_root.resolve(strict=True)
    if not root.is_dir():
        raise ReplayBuildError(f"run root is not a directory: {root}")
    run_id, lane = _metadata(root)
    specs = _discover(root)
    events: list[EventCandidate] = []
    inventories: list[dict[str, Any]] = []

    for spec in specs:
        for source in spec.paths:
            path, source_path = _safe_source(source, root)
            source_events: list[EventCandidate] = []
            if path.suffix == ".jsonl":
                rows = _load_jsonl(path)
            else:
                row = _load_object(path)
                raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                rows = iter(((1, row, raw),))
            for line_number, row, raw in rows:
                source_events.append(
                    _candidate(
                        spec=spec,
                        path=path,
                        source_path=source_path,
                        line_number=line_number,
                        row=row,
                        raw=raw,
                        root=root,
                        run_id=run_id,
                        lane=lane,
                    )
                )
            events.extend(source_events)
            inventories.append(
                {
                    "source_name": spec.name,
                    "source_ordinal": spec.ordinal,
                    "source_host": (
                        "x86" if source_path.startswith("remote/x86/") else "dgx"
                    ),
                    "source_path": source_path,
                    "private_reference_only": spec.private,
                    "sha256": _sha256(path),
                    "record_count": len(source_events),
                    "wall_latency_sample_count": sum(
                        len(event.wall_latencies_ms) for event in source_events
                    ),
                }
            )

    _align(events)
    events.sort(key=_sort_key)
    rows = [_event_dict(event, ordinal) for ordinal, event in enumerate(events)]
    timeline_data = b"".join(_json_bytes(row) for row in rows)

    replay = root / "replay"
    if replay.is_symlink():
        raise ReplayBuildError(f"replay output symlink is forbidden: {replay}")
    replay.mkdir(parents=True, exist_ok=True)
    if not _is_within(replay.resolve(), root):
        raise ReplayBuildError("replay output escapes run root")
    timeline_path = replay / "timeline.jsonl"
    index_path = replay / "timeline_index.json"
    for output in (timeline_path, index_path):
        if output.is_symlink():
            raise ReplayBuildError(f"output symlink is forbidden: {output}")
    _write_atomic(timeline_path, timeline_data)

    per_source: dict[str, list[EventCandidate]] = defaultdict(list)
    for event in events:
        per_source[event.source_path].append(event)
    for inventory in inventories:
        source_events = per_source[inventory["source_path"]]
        inventory.update(
            {
                "explicit_sim_stamp_count": sum(
                    event.explicit_sim_stamp for event in source_events
                ),
                "matched_sim_stamp_count": sum(
                    event.alignment_method is not None for event in source_events
                ),
                "no_sim_stamp_count": sum(event.sim_stamp_ns is None for event in source_events),
                "unmatched_identity_count": sum(
                    event.episode_id is None for event in source_events
                ),
                "unmatched_artifact_count": sum(
                    artifact.get("present") is not True
                    for event in source_events
                    for artifact in event.artifacts
                ),
            }
        )

    episode_indices: dict[tuple[str, int | None], list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        if event.episode_id is not None:
            episode_indices[(event.episode_id, event.reset_generation)].append(index)
    per_episode_ranges: list[dict[str, Any]] = []
    for identity, indices in sorted(
        episode_indices.items(),
        key=lambda item: (item[0][0], item[0][1] is None, item[0][1] or 0),
    ):
        stamps = [
            events[index].sim_stamp_ns
            for index in indices
            if events[index].sim_stamp_ns is not None
        ]
        per_episode_ranges.append(
            {
                "episode_id": identity[0],
                "reset_generation": identity[1],
                "event_count": len(indices),
                "timeline_start_index": min(indices),
                "timeline_end_index": max(indices),
                "first_event_id": rows[min(indices)]["event_id"],
                "last_event_id": rows[max(indices)]["event_id"],
                "first_sim_stamp_ns": min(stamps) if stamps else None,
                "last_sim_stamp_ns": max(stamps) if stamps else None,
            }
        )

    alignment = {
        "explicit_sim_stamp_count": sum(event.explicit_sim_stamp for event in events),
        "matched_sim_stamp_count": sum(event.alignment_method is not None for event in events),
        "no_sim_stamp_count": sum(event.sim_stamp_ns is None for event in events),
        "unmatched_identity_count": sum(event.episode_id is None for event in events),
        "unmatched_artifact_count": sum(
            artifact.get("present") is not True
            for event in events
            for artifact in event.artifacts
        ),
    }
    index = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "lane": lane,
        "time_authority": "x86_sim_stamp_ns",
        "sort_contract": "sim_stamp_ns_then_source_ordinal_path_line;no_sim_last",
        "canonical_fields": list(CANONICAL_FIELDS),
        "event_count": len(events),
        "event_type_counts": dict(sorted(Counter(event.event_type for event in events).items())),
        "source_inventory": sorted(
            inventories, key=lambda item: (item["source_ordinal"], item["source_path"])
        ),
        "alignment": alignment,
        "wall_latency_summary": _wall_latency_summary(events),
        "unmatched_count": alignment["unmatched_identity_count"],
        "no_sim_stamp_count": alignment["no_sim_stamp_count"],
        "per_episode_ranges": per_episode_ranges,
        "timeline_path": "replay/timeline.jsonl",
        "timeline_bytes": len(timeline_data),
        "timeline_sha256": hashlib.sha256(timeline_data).hexdigest(),
    }
    _write_atomic(index_path, _json_bytes(index, pretty=True))
    return index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="materialized local coordinator run root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        index = build_replay(args.run_root)
    except (OSError, ReplayBuildError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(index, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
