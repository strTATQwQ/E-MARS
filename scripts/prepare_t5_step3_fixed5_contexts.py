#!/usr/bin/env python3
"""Freeze real Lane-B Rev-C captures into Step3 fixed-five request contexts.

This is an offline bridge between a completed
``lane_b_revc_fixed5_capture`` run and
``materialize_t5_step3_frozen_fixed5.py``.  It binds the five scoped capture
sidecars to the exact frozen dataset instructions and authored start poses in
the evaluator's natural episode order.  The candidate frontiers are the
pre-registered deterministic replay triplet, so the resulting bundle is
explicitly interface screening evidence, not a navigation-effect claim.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTEXT_KIND = "t5_lane_b_step3_fixed5_capture_contexts"
EVALUATION_SCOPE = "interface_screening_only"
FRONTIER_SOURCE = "deterministic_frozen_replay_triplet_not_live_nav2"
AGENT_POSE_ENCODING = "dataset_start_position_xyz_plus_rotation_xyzw"
FIXED_EPISODE_COUNT = 5
INTERFACE_FRONTIERS = (
    {
        "frontier_id": 0,
        "relative_xz": [-0.8, 1.2],
        "distance_m": 1.44,
        "bearing_deg": -33.7,
    },
    {
        "frontier_id": 1,
        "relative_xz": [0.0, 1.8],
        "distance_m": 1.8,
        "bearing_deg": 0.0,
    },
    {
        "frontier_id": 2,
        "relative_xz": [0.8, 1.2],
        "distance_m": 1.44,
        "bearing_deg": 33.7,
    },
)


class ContextPreparationError(ValueError):
    """Raised when capture, dataset, or natural-order binding is ambiguous."""


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ContextPreparationError(f"cannot read file: {path}") from exc


def _load_object(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContextPreparationError(f"{name} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContextPreparationError(f"cannot read {name}: {path}") from exc
    if not isinstance(value, dict):
        raise ContextPreparationError(f"{name} must contain one JSON object")
    return value


def _finite_array(value: Any, name: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ContextPreparationError(f"{name} must contain exactly {length} values")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ContextPreparationError(f"{name}[{index}] must be numeric")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ContextPreparationError(f"{name}[{index}] must be numeric") from exc
        if not math.isfinite(number):
            raise ContextPreparationError(f"{name}[{index}] must be finite")
        result.append(number)
    return result


def _episode_key(episode: Mapping[str, Any]) -> tuple[str, str]:
    trajectory_id = str(episode.get("trajectory_id") or "").strip()
    episode_id = str(episode.get("episode_id") or "").strip()
    if not trajectory_id or not episode_id:
        raise ContextPreparationError("dataset episode lacks trajectory_id or episode_id")
    return f"{trajectory_id}_{episode_id}", episode_id


def _instruction(episode: Mapping[str, Any]) -> str:
    value = episode.get("instruction")
    if not isinstance(value, Mapping):
        raise ContextPreparationError("dataset episode instruction must be an object")
    text = str(value.get("instruction_text") or "").strip()
    if not text:
        raise ContextPreparationError("dataset episode instruction_text is empty")
    return text


def _load_dataset(path: Path) -> tuple[dict[str, Mapping[str, Any]], str]:
    if path.is_symlink() or not path.is_file():
        raise ContextPreparationError("dataset must be a regular non-symlink file")
    digest = _sha256(path)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContextPreparationError("cannot read frozen dataset gzip") from exc
    episodes = value.get("episodes") if isinstance(value, Mapping) else None
    if not isinstance(episodes, list) or len(episodes) != FIXED_EPISODE_COUNT:
        raise ContextPreparationError("frozen dataset must contain exactly five episodes")
    by_key: dict[str, Mapping[str, Any]] = {}
    for item in episodes:
        if not isinstance(item, Mapping):
            raise ContextPreparationError("dataset episode must be an object")
        key, _ = _episode_key(item)
        if key in by_key:
            raise ContextPreparationError(f"dataset repeats episode key: {key}")
        by_key[key] = item
    return by_key, digest


def _require_pass(value: Mapping[str, Any], name: str) -> None:
    if value.get("status") != "PASS":
        raise ContextPreparationError(f"{name} is not PASS")


def _identity_from_capture(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    request = row.get("request")
    if not isinstance(request, Mapping):
        raise ContextPreparationError("capture row lacks request identity")
    episode = str(request.get("episode_id") or "")
    reset_id = request.get("reset_generation")
    sequence_id = request.get("sequence_id")
    if (
        not episode.startswith("b::")
        or isinstance(reset_id, bool)
        or not isinstance(reset_id, int)
        or reset_id < 0
        or isinstance(sequence_id, bool)
        or not isinstance(sequence_id, int)
        or sequence_id < 0
    ):
        raise ContextPreparationError("capture row has invalid Lane-B identity")
    return episode, reset_id, sequence_id, f"{episode}::{reset_id}::{sequence_id}"


def prepare_contexts(
    *, result_root: Path, dataset_file: Path, output_path: Path
) -> dict[str, Any]:
    result_root = result_root.resolve()
    dataset_file = dataset_file.resolve()
    output_path = output_path.resolve()
    if result_root.is_symlink() or not result_root.is_dir():
        raise ContextPreparationError("result root must be a regular directory")
    if output_path.exists() or output_path.is_symlink():
        raise ContextPreparationError("output context path must be fresh")

    binding_path = result_root / "input_binding.json"
    summary_path = result_root / "fast_lane_summary.json"
    order_path = result_root / "remote/x86/ordered_episode_manifest.json"
    capture_path = result_root / "remote/x86/evaluator/revc_fixed5_capture.json"
    scoped_path = result_root / "audits/revc_fixed5_scoped_pull.json"
    binding = _load_object(binding_path, "Lane-B input binding")
    summary = _load_object(summary_path, "Lane-B final summary")
    order = _load_object(order_path, "natural episode order manifest")
    capture = _load_object(capture_path, "Rev-C fixed-five capture summary")
    scoped = _load_object(scoped_path, "Rev-C scoped-pull receipt")
    for value, name in (
        (binding, "Lane-B input binding"),
        (summary, "Lane-B final summary"),
        (order, "natural episode order manifest"),
        (capture, "Rev-C fixed-five capture summary"),
        (scoped, "Rev-C scoped-pull receipt"),
    ):
        _require_pass(value, name)
    if (
        binding.get("lane") != "b"
        or binding.get("execution_profile") != "fixed5"
        or binding.get("isaac_sensor_profile") != "lane_b_revc_fixed5_capture"
        or binding.get("execution_episode_count") != FIXED_EPISODE_COUNT
        or capture.get("profile") != "lane_b_revc_fixed5_capture"
        or capture.get("lane") != "b"
        or capture.get("capture_count") != FIXED_EPISODE_COUNT
        or capture.get("same_render_tick_all") is not True
        or scoped.get("snapshot_count") != FIXED_EPISODE_COUNT
        or scoped.get("png_count") != FIXED_EPISODE_COUNT * 4
    ):
        raise ContextPreparationError("Lane-B fixed-five capture contract is not complete")

    episodes, dataset_sha = _load_dataset(dataset_file)
    declared_dataset_shas = {
        str(binding.get("dataset_sha256") or ""),
        str(order.get("dataset_sha256") or ""),
    }
    if declared_dataset_shas != {dataset_sha}:
        raise ContextPreparationError("dataset SHA differs from Lane-B capture inputs")

    ordered_keys = order.get("ordered_episode_keys")
    ordered_ids = order.get("ordered_episode_ids")
    capture_rows = capture.get("snapshots")
    if (
        not isinstance(ordered_keys, list)
        or not isinstance(ordered_ids, list)
        or not isinstance(capture_rows, list)
        or len(ordered_keys) != FIXED_EPISODE_COUNT
        or len(ordered_ids) != FIXED_EPISODE_COUNT
        or len(capture_rows) != FIXED_EPISODE_COUNT
        or len(set(map(str, ordered_keys))) != FIXED_EPISODE_COUNT
        or len(set(map(str, ordered_ids))) != FIXED_EPISODE_COUNT
        or set(map(str, ordered_keys)) != set(episodes)
        or capture.get("ordered_episode_ids") != ordered_ids
    ):
        raise ContextPreparationError("dataset, natural order, and capture count disagree")

    evaluator_root = capture_path.parent.resolve()
    cases: list[dict[str, Any]] = []
    seen_sidecars: set[Path] = set()
    seen_snapshots: set[str] = set()
    for index, (key_raw, episode_id_raw, capture_row) in enumerate(
        zip(ordered_keys, ordered_ids, capture_rows)
    ):
        key = str(key_raw)
        episode_id = str(episode_id_raw)
        if not isinstance(capture_row, Mapping):
            raise ContextPreparationError(f"capture[{index}] must be an object")
        episode = episodes[key]
        _, dataset_episode_id = _episode_key(episode)
        lane_episode, reset_id, sequence_id, snapshot_id = _identity_from_capture(
            capture_row
        )
        if (
            dataset_episode_id != episode_id
            or capture_row.get("capture_index") != index
            or str(capture_row.get("ordered_episode_id") or "") != episode_id
            or lane_episode != f"b::{episode_id}"
        ):
            raise ContextPreparationError(f"capture[{index}] episode identity drifted")
        sidecar_relative = Path(str(capture_row.get("sidecar") or ""))
        sidecar = (evaluator_root / sidecar_relative).resolve()
        scoped_root = (evaluator_root / "revc_snapshots").resolve()
        try:
            sidecar.relative_to(scoped_root)
        except ValueError as exc:
            raise ContextPreparationError("capture sidecar escapes scoped Rev-C root") from exc
        if (
            sidecar.name != "snapshot.json"
            or sidecar.parent.parent != scoped_root
            or sidecar.is_symlink()
            or not sidecar.is_file()
            or sidecar in seen_sidecars
            or _sha256(sidecar) != str(capture_row.get("sidecar_sha256") or "")
        ):
            raise ContextPreparationError("capture sidecar is missing, reused, or changed")
        raw_sidecar = _load_object(sidecar, f"capture[{index}] sidecar")
        if (
            raw_sidecar.get("episode_id") != lane_episode
            or raw_sidecar.get("reset_generation") != reset_id
            or raw_sidecar.get("sequence_id") != sequence_id
        ):
            raise ContextPreparationError("capture sidecar identity differs from request")
        if snapshot_id in seen_snapshots:
            raise ContextPreparationError("capture snapshot identity was reused")
        seen_sidecars.add(sidecar)
        seen_snapshots.add(snapshot_id)

        start_position = _finite_array(
            episode.get("start_position"), f"episode[{key}].start_position", 3
        )
        start_rotation = _finite_array(
            episode.get("start_rotation"), f"episode[{key}].start_rotation", 4
        )
        relative_sidecar = Path(os.path.relpath(sidecar, output_path.parent)).as_posix()
        cases.append(
            {
                "case_id": f"interface-{index}-{key}",
                "episode_key": key,
                "snapshot_id": snapshot_id,
                "source_snapshot_sidecar": relative_sidecar,
                "source_snapshot_sidecar_sha256": _sha256(sidecar),
                "instruction": _instruction(episode),
                "candidate_frontiers": [dict(item) for item in INTERFACE_FRONTIERS],
                "agent_pose": start_position + start_rotation,
                "visited_frontiers": [],
                "compact_history": [],
            }
        )

    payload = {
        "schema_version": 1,
        "kind": CONTEXT_KIND,
        "frozen": True,
        "evaluation_scope": EVALUATION_SCOPE,
        "candidate_frontier_source": FRONTIER_SOURCE,
        "agent_pose_encoding": AGENT_POSE_ENCODING,
        "revc_contract_sha256": str(capture.get("contract_sha256") or ""),
        "dataset_sha256": dataset_sha,
        "input_binding_sha256": _sha256(binding_path),
        "ordered_episode_manifest_sha256": _sha256(order_path),
        "capture_summary_sha256": _sha256(capture_path),
        "scoped_pull_receipt_sha256": _sha256(scoped_path),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_path)
    return {
        "schema_version": 1,
        "status": "PASS",
        "evaluation_scope": EVALUATION_SCOPE,
        "case_count": len(cases),
        "dataset_sha256": dataset_sha,
        "contexts_sha256": _sha256(output_path),
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare_contexts(
        result_root=args.result_root,
        dataset_file=args.dataset_file,
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
