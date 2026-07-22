#!/usr/bin/env python3
"""Materialize the archived Rev-C fixed-five into an oracle-map offline gate.

The output intentionally keeps goal/reference-path information in ``labels``.
Scorer features use only current candidate geometry, start-to-candidate path
cost, a static free-space information-gain proxy, and bounded model advice.
The pre-registered replay triplet is not represented as live Nav2 evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path
import tarfile
from typing import Any, Iterable, Mapping, Sequence


KIND = "t5_semantic_frontier_offline_cases"
SCOPE = "ORACLE_MAP_OFFLINE_BENCHMARK"
SEMANTIC_SOURCES = frozenset({"step3", "internvla", "step3_then_internvla"})
ACTION_TO_BEARING = {1: "front", 2: "left", 3: "right"}
NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


class MaterializationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"JSON payload must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise MaterializationError(f"JSONL row {line_number} is not an object")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read JSONL: {path}") from exc
    return rows


def _load_replay_manifest(archive: Path) -> dict[str, Any]:
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            member = next(
                (item for item in bundle.getmembers() if item.name.lstrip("./") == "manifest.json"),
                None,
            )
            if member is None or not member.isfile():
                raise MaterializationError("replay archive has no regular manifest.json")
            stream = bundle.extractfile(member)
            if stream is None:
                raise MaterializationError("cannot read replay manifest member")
            value = json.loads(stream.read().decode("utf-8"))
    except (OSError, tarfile.TarError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read replay archive: {archive}") from exc
    if not isinstance(value, dict):
        raise MaterializationError("replay manifest must be an object")
    return value


def _episodes(path: Path) -> dict[str, dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read frozen dataset: {path}") from exc
    rows = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise MaterializationError("dataset episodes are missing")
    result = {}
    for row in rows:
        key = f"{row['trajectory_id']}_{row['episode_id']}"
        result[key] = row
    return result


def _world_to_cell(
    point: tuple[float, float], map_entry: Mapping[str, Any]
) -> tuple[int, int]:
    resolution = float(map_entry["resolution_m"])
    origin_x, origin_y = (float(value) for value in map_entry["origin_xy"])
    column = math.floor((point[0] - origin_x) / resolution)
    row = math.floor((point[1] - origin_y) / resolution)
    return row, column


def _inside(cell: tuple[int, int], height: int, width: int) -> bool:
    return 0 <= cell[0] < height and 0 <= cell[1] < width


def _nearest_free(
    cell: tuple[int, int], grid: bytes, height: int, width: int, *, radius: int = 6
) -> tuple[int, int] | None:
    if _inside(cell, height, width) and grid[cell[0] * width + cell[1]] == 0:
        return cell
    candidates = []
    for delta_row in range(-radius, radius + 1):
        for delta_column in range(-radius, radius + 1):
            candidate = (cell[0] + delta_row, cell[1] + delta_column)
            if _inside(candidate, height, width) and grid[candidate[0] * width + candidate[1]] == 0:
                candidates.append((delta_row * delta_row + delta_column * delta_column, candidate))
    return min(candidates)[1] if candidates else None


def _astar(
    grid: bytes,
    height: int,
    width: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    resolution_m: float,
) -> float | None:
    if start == goal:
        return 0.0
    queue: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(queue, (0.0, 0.0, start))
    costs = {start: 0.0}
    while queue:
        _estimate, cost, current = heapq.heappop(queue)
        if cost != costs.get(current):
            continue
        if current == goal:
            return cost * resolution_m
        for delta_row, delta_column, step in NEIGHBORS:
            neighbor = (current[0] + delta_row, current[1] + delta_column)
            if not _inside(neighbor, height, width):
                continue
            if grid[neighbor[0] * width + neighbor[1]] != 0:
                continue
            if delta_row and delta_column:
                side_a = (current[0] + delta_row, current[1])
                side_b = (current[0], current[1] + delta_column)
                if (
                    grid[side_a[0] * width + side_a[1]] != 0
                    or grid[side_b[0] * width + side_b[1]] != 0
                ):
                    continue
            new_cost = cost + step
            if new_cost >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = new_cost
            heuristic = math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
            heapq.heappush(queue, (new_cost + heuristic, new_cost, neighbor))
    return None


def _free_space_proxy(
    grid: bytes, height: int, width: int, cell: tuple[int, int], radius_cells: int
) -> float:
    free = 0
    total = 0
    radius_squared = radius_cells * radius_cells
    for delta_row in range(-radius_cells, radius_cells + 1):
        for delta_column in range(-radius_cells, radius_cells + 1):
            if delta_row * delta_row + delta_column * delta_column > radius_squared:
                continue
            candidate = (cell[0] + delta_row, cell[1] + delta_column)
            if not _inside(candidate, height, width):
                continue
            total += 1
            free += grid[candidate[0] * width + candidate[1]] == 0
    return free / total if total else 0.0


def _heading_angle(rotation: Iterable[Any]) -> float:
    values = [float(value) for value in rotation]
    if len(values) != 4:
        raise MaterializationError("start_rotation must be xyzw")
    theta = 2.0 * math.atan2(values[1], values[3])
    return math.atan2(math.cos(theta), -math.sin(theta))


def _lane_b_snapshot_identity(snapshot_id: str) -> tuple[str, int, int]:
    parts = str(snapshot_id).split("::")
    if len(parts) != 4 or parts[0] != "b" or not parts[1]:
        raise MaterializationError(
            "snapshot_id must match b::<episode>::<reset>::<sequence>"
        )
    try:
        reset_generation = int(parts[2])
        sequence_id = int(parts[3])
    except ValueError as exc:
        raise MaterializationError(
            "snapshot reset and sequence must be integers"
        ) from exc
    if reset_generation < 0 or sequence_id < 0:
        raise MaterializationError(
            "snapshot reset and sequence must be non-negative"
        )
    return parts[1], reset_generation, sequence_id


def _legal_frontier_geometry(
    candidate_frontiers: Sequence[Mapping[str, Any]],
    *,
    start_xy: tuple[float, float],
    heading: float,
    map_entry: Mapping[str, Any],
    grid: bytes,
    height: int,
    width: int,
    start_cell: tuple[int, int],
    resolution: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind legal candidates without accepting any oracle goal/reference input."""

    legal_geometry: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for frontier in candidate_frontiers:
        frontier_id = int(frontier["frontier_id"])
        bearing = math.radians(float(frontier["bearing_deg"]))
        distance = float(frontier["distance_m"])
        world_angle = heading - bearing
        endpoint = (
            start_xy[0] + distance * math.cos(world_angle),
            start_xy[1] + distance * math.sin(world_angle),
        )
        endpoint_cell = _world_to_cell(endpoint, map_entry)
        if not _inside(endpoint_cell, height, width):
            rejections.append(
                {"frontier_id": frontier_id, "reason": "ENDPOINT_OUTSIDE_STATIC_MAP"}
            )
            continue
        if grid[endpoint_cell[0] * width + endpoint_cell[1]] != 0:
            rejections.append(
                {"frontier_id": frontier_id, "reason": "ENDPOINT_NOT_FREE"}
            )
            continue
        path_cost = _astar(grid, height, width, start_cell, endpoint_cell, resolution)
        if path_cost is None:
            rejections.append(
                {
                    "frontier_id": frontier_id,
                    "reason": "NO_START_TO_CANDIDATE_PATH",
                }
            )
            continue
        legal_geometry.append(
            {
                "frontier": dict(frontier),
                "endpoint_cell": endpoint_cell,
                "path_cost_m": path_cost,
                "information_gain_proxy": _free_space_proxy(
                    grid,
                    height,
                    width,
                    endpoint_cell,
                    max(1, round(1.0 / resolution)),
                ),
            }
        )
    return legal_geometry, rejections


def _semantic_row(
    case: Mapping[str, Any],
    legal_ids: set[int],
    step3_by_episode: Mapping[str, Mapping[str, Any]],
    internvla_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
    source: str,
) -> tuple[str, str, dict[int, float | None], dict[str, Any]]:
    episode_key = str(case["episode_key"])
    snapshot_id = str(case["snapshot_id"])

    def step3() -> tuple[str, str, int | None, float, dict[str, Any]]:
        row = step3_by_episode.get(episode_key)
        if row is None:
            return "MISSING", "step3", None, 0.0, {}
        decision = row.get("decision") if isinstance(row.get("decision"), Mapping) else {}
        provenance = {
            "source_snapshot_id": row.get("snapshot_id"),
            "model_request_status": row.get("model_request_status"),
        }
        if row.get("snapshot_id") != snapshot_id:
            return "STALE", "step3", None, 0.0, provenance
        if decision.get("abstain") is True:
            return "ABSTAIN", "step3", None, 0.0, provenance
        selected = decision.get("recommended_frontier")
        if isinstance(selected, bool) or not isinstance(selected, int) or selected not in legal_ids:
            return "ILLEGAL", "step3", None, 0.0, provenance
        return "SELECTION", "step3", selected, float(decision.get("confidence") or 0.0), provenance

    def internvla() -> tuple[str, str, int | None, float, dict[str, Any]]:
        rows = tuple(internvla_by_episode.get(episode_key, ()))
        if not rows:
            return "MISSING", "internvla", None, 0.0, {}
        snapshot_episode, snapshot_reset, snapshot_sequence = _lane_b_snapshot_identity(
            snapshot_id
        )
        available_identities = []
        row = None
        for candidate in rows:
            candidate_episode = str(candidate.get("episode_id") or "").split("::")[-1]
            try:
                candidate_reset = int(candidate.get("reset_generation", -1))
                candidate_sequence = int(candidate.get("sequence_id", -1))
            except (TypeError, ValueError):
                continue
            available_identities.append(
                {
                    "episode_id": candidate_episode,
                    "reset_generation": candidate_reset,
                    "sequence_id": candidate_sequence,
                }
            )
            if (
                candidate_episode == snapshot_episode
                and candidate_reset == snapshot_reset
                and candidate_sequence == snapshot_sequence
            ):
                row = candidate
                break
        if row is None:
            return (
                "STALE",
                "internvla",
                None,
                0.0,
                {
                    "required_identity": {
                        "episode_id": snapshot_episode,
                        "reset_generation": snapshot_reset,
                        "sequence_id": snapshot_sequence,
                    },
                    "available_identities": available_identities,
                },
            )
        provenance = {
            "source_episode_id": row.get("episode_id"),
            "source_reset_generation": row.get("reset_generation"),
            "source_sequence_id": row.get("sequence_id"),
            "model_discrete_action": row.get("model_discrete_action"),
        }
        action = int(row.get("model_discrete_action", -1))
        candidates = [item for item in case["candidate_frontiers"] if int(item["frontier_id"]) in legal_ids]
        if action == 1:
            selected = min(candidates, key=lambda item: (abs(float(item["bearing_deg"])), int(item["frontier_id"])))
        elif action == 2:
            selected = min(candidates, key=lambda item: (float(item["bearing_deg"]), int(item["frontier_id"])))
        elif action == 3:
            selected = max(candidates, key=lambda item: (float(item["bearing_deg"]), -int(item["frontier_id"])))
        else:
            return "ABSTAIN", "internvla", None, 0.0, provenance
        frontier_id = int(selected["frontier_id"])
        return "SELECTION", "internvla", frontier_id, 1.0, provenance

    if source == "step3":
        selected = step3()
    elif source == "internvla":
        selected = internvla()
    else:
        selected = step3()
        if selected[0] != "SELECTION":
            selected = internvla()
    status, selected_source, frontier_id, confidence, provenance = selected
    relevance = {frontier_id_: None for frontier_id_ in legal_ids}
    if status == "SELECTION" and frontier_id is not None:
        relevance = {frontier_id_: 0.0 for frontier_id_ in legal_ids}
        relevance[frontier_id] = min(max(confidence, 0.0), 1.0)
    return status, selected_source, relevance, provenance


def materialize(
    *,
    dataset: Path,
    replay_archive: Path,
    step3_results: Path,
    internvla_records: Path,
    map_manifest_path: Path,
    map_root: Path,
    semantic_source: str,
    output: Path,
) -> dict[str, Any]:
    if semantic_source not in SEMANTIC_SOURCES:
        raise MaterializationError(f"unsupported semantic source: {semantic_source}")
    if output.exists():
        raise MaterializationError(f"output already exists: {output}")
    replay = _load_replay_manifest(replay_archive)
    if replay.get("evaluation_scope") != "interface_screening_only" or replay.get(
        "candidate_frontier_source"
    ) != "deterministic_frozen_replay_triplet_not_live_nav2":
        raise MaterializationError("expected the frozen non-live replay triplet")
    if replay.get("agent_pose_encoding") != (
        "dataset_start_position_xyz_plus_rotation_xyzw"
    ):
        raise MaterializationError("unexpected frozen replay agent_pose encoding")
    episode_by_key = _episodes(dataset)
    map_manifest = _load_object(map_manifest_path)
    if map_manifest.get("dataset_sha256") != _sha256(dataset):
        raise MaterializationError("static map manifest is not bound to the frozen dataset")
    generations = {
        f"{row['trajectory_id']}_{row['episode_id']}": row
        for row in map_manifest.get("generations", [])
    }
    map_entries = map_manifest.get("maps")
    if not isinstance(map_entries, Mapping):
        raise MaterializationError("static map entries are missing")
    step3_by_episode = {
        str(row.get("episode_key")): row for row in _load_jsonl(step3_results)
    }
    internvla_by_episode: dict[str, list[Mapping[str, Any]]] = {}
    episode_id_to_key = {
        str(row["episode_id"]): key for key, row in episode_by_key.items()
    }
    for row in _load_jsonl(internvla_records):
        if row.get("motion_observation_gate_only"):
            continue
        action = row.get("model_discrete_action")
        if action not in (1, 2, 3):
            continue
        episode_id = str(row.get("episode_id") or "").split("::")[-1]
        episode_key = episode_id_to_key.get(episode_id)
        if episode_key:
            internvla_by_episode.setdefault(episode_key, []).append(row)

    cases = []
    invalid_candidates = 0
    case_audits = []
    case_exclusions = []
    for replay_case in replay.get("cases", []):
        episode_key = str(replay_case["episode_key"])
        episode = episode_by_key.get(episode_key)
        generation = generations.get(episode_key)
        if episode is None or generation is None:
            raise MaterializationError(f"missing episode/map generation: {episode_key}")
        map_entry = map_entries[generation["map_key"]]
        grid_path = map_root / str(map_entry["file"])
        if _sha256(grid_path) != map_entry.get("sha256"):
            raise MaterializationError(f"static map SHA-256 mismatch: {grid_path}")
        grid = grid_path.read_bytes()
        height, width = int(map_entry["height"]), int(map_entry["width"])
        if len(grid) != height * width:
            raise MaterializationError(f"static map byte count mismatch: {grid_path}")
        resolution = float(map_entry["resolution_m"])
        agent_pose = replay_case.get("agent_pose")
        if not isinstance(agent_pose, list) or len(agent_pose) != 7:
            raise MaterializationError(
                f"frozen replay agent_pose must contain xyz+xyzw: {episode_key}"
            )
        start_xy = (float(agent_pose[0]), -float(agent_pose[2]))
        start_cell = _nearest_free(_world_to_cell(start_xy, map_entry), grid, height, width)
        if start_cell is None:
            raise MaterializationError(f"cannot bind start to static map: {episode_key}")
        heading = _heading_angle(agent_pose[3:7])
        legal_geometry, candidate_rejections = _legal_frontier_geometry(
            replay_case["candidate_frontiers"],
            start_xy=start_xy,
            heading=heading,
            map_entry=map_entry,
            grid=grid,
            height=height,
            width=width,
            start_cell=start_cell,
            resolution=resolution,
        )
        invalid_candidates += len(candidate_rejections)
        legal_frontiers = [item["frontier"] for item in legal_geometry]
        legal_ids = {int(frontier["frontier_id"]) for frontier in legal_frontiers}
        audit = {
            "episode_key": episode_key,
            "legal_frontier_ids": sorted(legal_ids),
            "candidate_rejections": candidate_rejections,
        }
        if not legal_geometry:
            exclusion = {
                **audit,
                "reason": "NO_LEGAL_REPLAY_CANDIDATES_ENDPOINT_OR_START_PATH",
            }
            case_audits.append({**exclusion, "included": False})
            case_exclusions.append(exclusion)
            continue

        # Oracle/reference information begins here and is used only to produce
        # labels. It cannot alter legal_frontiers or any scorer feature.
        goal_position = episode["reference_path"][-1]
        goal_xy = (float(goal_position[0]), -float(goal_position[2]))
        goal_cell = _nearest_free(_world_to_cell(goal_xy, map_entry), grid, height, width)
        if goal_cell is None:
            exclusion = {**audit, "reason": "ORACLE_GOAL_NOT_BINDABLE_TO_STATIC_MAP"}
            case_audits.append({**exclusion, "included": False})
            case_exclusions.append(exclusion)
            continue
        start_remaining = _astar(grid, height, width, start_cell, goal_cell, resolution)
        if start_remaining is None:
            exclusion = {**audit, "reason": "ORACLE_START_TO_GOAL_LABEL_UNAVAILABLE"}
            case_audits.append({**exclusion, "included": False})
            case_exclusions.append(exclusion)
            continue

        raw_features = []
        raw_labels = []
        proxy_values = []
        label_failures = []
        for geometry in legal_geometry:
            frontier = geometry["frontier"]
            endpoint_cell = geometry["endpoint_cell"]
            remaining = _astar(grid, height, width, endpoint_cell, goal_cell, resolution)
            frontier_id = int(frontier["frontier_id"])
            if remaining is None:
                label_failures.append(frontier_id)
                continue
            proxy = float(geometry["information_gain_proxy"])
            proxy_values.append(proxy)
            raw_features.append(
                {
                    "frontier_id": frontier_id,
                    "semantic_relevance": None,
                    "information_gain": proxy,
                    "path_cost_m": float(geometry["path_cost_m"]),
                    "revisit_count": 0,
                    "stuck_penalty": 0.0,
                }
            )
            raw_labels.append(
                {
                    "frontier_id": frontier_id,
                    "progress_m": start_remaining - remaining,
                    "remaining_geodesic_m": remaining,
                }
            )
        if label_failures:
            exclusion = {
                **audit,
                "reason": "ORACLE_CANDIDATE_TO_GOAL_LABEL_UNAVAILABLE",
                "label_failure_frontier_ids": sorted(label_failures),
            }
            case_audits.append({**exclusion, "included": False})
            case_exclusions.append(exclusion)
            continue
        minimum_proxy, maximum_proxy = min(proxy_values), max(proxy_values)
        for feature in raw_features:
            feature["information_gain"] = (
                (feature["information_gain"] - minimum_proxy) / (maximum_proxy - minimum_proxy)
                if maximum_proxy > minimum_proxy
                else 0.5
            )
        status, selected_source, relevance, provenance = _semantic_row(
            replay_case,
            legal_ids,
            step3_by_episode,
            internvla_by_episode,
            semantic_source,
        )
        for feature in raw_features:
            feature["semantic_relevance"] = relevance[int(feature["frontier_id"])]
        cases.append(
            {
                "episode_id": str(episode["episode_id"]),
                "episode_key": episode_key,
                "snapshot_id": str(replay_case["snapshot_id"]),
                "instruction": str(episode["instruction"]["instruction_text"]),
                "candidate_frontiers": legal_frontiers,
                "visited_frontiers": [],
                "features": raw_features,
                "feature_sources": {
                    "information_gain": "static_free_space_fraction_proxy_no_oracle_goal",
                    "path_cost_m": "static_map_start_to_candidate_astar_no_oracle_goal",
                    "revisit_count": "empty_start_history",
                    "stuck_penalty": "zero_start_state",
                },
                "labels": {
                    "source": "oracle_goal_static_map_astar",
                    "usage": "evaluation_only_not_scorer_input",
                    "start_remaining_geodesic_m": start_remaining,
                    "frontiers": raw_labels,
                },
                "semantic_status": status,
                "semantic_source": selected_source,
                "semantic_provenance": provenance,
            }
        )
        case_audits.append({**audit, "included": True, "label_status": "COMPLETE"})

    payload = {
        "schema_version": 1,
        "kind": KIND,
        "evaluation_scope": SCOPE,
        "candidate_frontier_source": "offline_static_map_legal_subset_of_frozen_replay_triplet_not_live_nav2",
        "source_is_live_nav2": False,
        "semantic_source_policy": semantic_source,
        "oracle_labels_used_by_scorer": False,
        "replay_frontier_navigation_effect_claim_eligible": False,
        "case_count": len(cases),
        "excluded_case_count": len(case_exclusions),
        "invalid_or_unreachable_candidate_count": invalid_candidates,
        "case_audits": case_audits,
        "case_exclusions": case_exclusions,
        "inputs": {
            "dataset_sha256": _sha256(dataset),
            "replay_archive_sha256": _sha256(replay_archive),
            "step3_results_sha256": _sha256(step3_results),
            "internvla_records_sha256": _sha256(internvla_records),
            "map_manifest_sha256": _sha256(map_manifest_path),
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--replay-archive", type=Path, required=True)
    parser.add_argument("--step3-results", type=Path, required=True)
    parser.add_argument("--internvla-records", type=Path, required=True)
    parser.add_argument("--map-manifest", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--semantic-source", choices=sorted(SEMANTIC_SOURCES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = materialize(
        dataset=arguments.dataset.resolve(),
        replay_archive=arguments.replay_archive.resolve(),
        step3_results=arguments.step3_results.resolve(),
        internvla_records=arguments.internvla_records.resolve(),
        map_manifest_path=arguments.map_manifest.resolve(),
        map_root=arguments.map_root.resolve(),
        semantic_source=arguments.semantic_source,
        output=arguments.output.resolve(),
    )
    print(json.dumps({"case_count": payload["case_count"], "output": str(arguments.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
