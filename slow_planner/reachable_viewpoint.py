"""Private, no-authority reachable-viewpoint candidates for Lane B.

These candidates are deliberately *not* Nav2 exploration frontiers and cannot
be sent through the frozen SlowPlanner v1 ``candidate_frontiers`` field.  The
module provides a deterministic geometry-only source and an explicit protocol
compatibility receipt so an empty-unknown-boundary map has an honest next
candidate source without silently changing the existing frontier contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from typing import Any, Mapping, Sequence

from .lane_b import LaneBSnapshotIdentity
from .live_frontier_capture import OccupancyGrid2D, Pose2D


REACHABLE_VIEWPOINT_SCHEMA_VERSION = 1
REACHABLE_VIEWPOINT_KIND = "t5_reachable_viewpoint_candidate_set"
REACHABLE_VIEWPOINT_CANDIDATE_TYPE = "reachable_free_space_viewpoint"
REACHABLE_VIEWPOINT_SOURCE = "deterministic_current_costmap_free_space"
SLOW_PLANNER_V1_BLOCKER = "SLOW_PLANNER_V1_FRONTIER_SEMANTICS_BLOCKED"

Cell = tuple[int, int]
_CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = _CARDINAL + ((-1, -1), (-1, 1), (1, -1), (1, 1))
_AUTHORITY = {
    "model_request": False,
    "navigation_goal": False,
    "cmd_vel": False,
    "terminal_stop": False,
}
_ORACLE_INPUTS = {
    "instruction": False,
    "goal": False,
    "reference_path": False,
    "success_state": False,
}
_CANDIDATE_KEYS = {
    "candidate_id",
    "candidate_type",
    "source",
    "candidate_xy_frame",
    "relative_xz",
    "grid_xy",
    "distance_m",
    "geodesic_distance_m",
    "bearing_deg",
}
_MAPPING_KEYS = {
    "schema_version",
    "kind",
    "source_node",
    "source",
    "candidate_type",
    "ros_episode_id",
    "episode_id",
    "reset_id",
    "sequence_id",
    "snapshot_id",
    "captured_sim_time_s",
    "valid_until_sim_time_s",
    "geometry_frame",
    "grid_frame_id",
    "agent_pose_encoding",
    "agent_pose",
    "candidates",
    "slow_planner_v1_compatibility",
    "authority",
    "oracle_inputs",
    "candidate_set_sha256",
}


class ReachableViewpointError(ValueError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def _fail(message: str, code: str) -> ReachableViewpointError:
    return ReachableViewpointError(message, code=code)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise _fail(f"{name} must be numeric", "INVALID_VIEWPOINT_INPUT")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _fail(f"{name} must be numeric", "INVALID_VIEWPOINT_INPUT") from exc
    if not math.isfinite(result):
        raise _fail(f"{name} must be finite", "INVALID_VIEWPOINT_INPUT")
    return result


def _clean(value: float) -> float:
    result = round(float(value), 9)
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class ReachableViewpointConfig:
    free_max_value: int = 0
    nearest_free_radius_cells: int = 6
    minimum_distance_m: float = 0.75
    preferred_distance_m: float = 3.0
    maximum_distance_m: float = 5.0
    angular_sectors: int = 8
    maximum_candidates: int = 8

    def __post_init__(self) -> None:
        if self.free_max_value < 0:
            raise _fail("free_max_value must be non-negative", "INVALID_VIEWPOINT_CONFIG")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (
                self.nearest_free_radius_cells,
                self.angular_sectors,
                self.maximum_candidates,
            )
        ):
            raise _fail("viewpoint integer limits must be positive", "INVALID_VIEWPOINT_CONFIG")
        minimum = _finite(self.minimum_distance_m, "minimum_distance_m")
        preferred = _finite(self.preferred_distance_m, "preferred_distance_m")
        maximum = _finite(self.maximum_distance_m, "maximum_distance_m")
        if not 0.0 < minimum <= preferred <= maximum:
            raise _fail("viewpoint distance limits are invalid", "INVALID_VIEWPOINT_CONFIG")


@dataclass(frozen=True)
class ReachableViewpoint:
    candidate_id: int
    relative_xz: tuple[float, float]
    grid_xy: tuple[float, float]
    distance_m: float
    geodesic_distance_m: float
    bearing_deg: float
    candidate_xy_frame: str
    candidate_type: str = REACHABLE_VIEWPOINT_CANDIDATE_TYPE
    source: str = REACHABLE_VIEWPOINT_SOURCE

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_id, bool)
            or not isinstance(self.candidate_id, int)
            or self.candidate_id < 0
        ):
            raise _fail("candidate_id must be non-negative", "INVALID_VIEWPOINT_CANDIDATE")
        if self.candidate_type != REACHABLE_VIEWPOINT_CANDIDATE_TYPE:
            raise _fail("candidate_type mismatch", "INVALID_VIEWPOINT_CANDIDATE")
        if self.source != REACHABLE_VIEWPOINT_SOURCE:
            raise _fail("candidate source mismatch", "INVALID_VIEWPOINT_CANDIDATE")
        if not isinstance(self.candidate_xy_frame, str) or not self.candidate_xy_frame.strip():
            raise _fail("candidate_xy_frame is required", "INVALID_VIEWPOINT_CANDIDATE")
        for name, values in (("relative_xz", self.relative_xz), ("grid_xy", self.grid_xy)):
            if len(values) != 2:
                raise _fail(f"{name} must contain two values", "INVALID_VIEWPOINT_CANDIDATE")
            for index, value in enumerate(values):
                _finite(value, f"{name}[{index}]")
        if _finite(self.distance_m, "distance_m") <= 0.0:
            raise _fail("distance_m must be positive", "INVALID_VIEWPOINT_CANDIDATE")
        if _finite(self.geodesic_distance_m, "geodesic_distance_m") <= 0.0:
            raise _fail("geodesic_distance_m must be positive", "INVALID_VIEWPOINT_CANDIDATE")
        _finite(self.bearing_deg, "bearing_deg")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "source": self.source,
            "candidate_xy_frame": self.candidate_xy_frame,
            "relative_xz": list(self.relative_xz),
            "grid_xy": list(self.grid_xy),
            "distance_m": self.distance_m,
            "geodesic_distance_m": self.geodesic_distance_m,
            "bearing_deg": self.bearing_deg,
        }


def _is_free(grid: OccupancyGrid2D, cell: Cell, config: ReachableViewpointConfig) -> bool:
    value = grid.value(cell)
    return 0 <= value <= config.free_max_value


def _nearest_free(
    grid: OccupancyGrid2D, cell: Cell, config: ReachableViewpointConfig
) -> Cell | None:
    candidates: list[tuple[int, Cell]] = []
    radius = config.nearest_free_radius_cells
    for row_offset in range(-radius, radius + 1):
        for column_offset in range(-radius, radius + 1):
            candidate = (cell[0] + row_offset, cell[1] + column_offset)
            if grid.inside(candidate) and _is_free(grid, candidate, config):
                candidates.append((row_offset * row_offset + column_offset * column_offset, candidate))
    return min(candidates)[1] if candidates else None


def _reachable_costs(
    grid: OccupancyGrid2D, start: Cell, config: ReachableViewpointConfig
) -> dict[Cell, float]:
    costs = {start: 0.0}
    queue: list[tuple[float, Cell]] = [(0.0, start)]
    while queue:
        current_cost, current = heapq.heappop(queue)
        if current_cost != costs.get(current):
            continue
        for delta_row, delta_column in _EIGHT:
            candidate = (current[0] + delta_row, current[1] + delta_column)
            if not grid.inside(candidate) or not _is_free(grid, candidate, config):
                continue
            if delta_row and delta_column:
                side_a = (current[0] + delta_row, current[1])
                side_b = (current[0], current[1] + delta_column)
                if not _is_free(grid, side_a, config) or not _is_free(grid, side_b, config):
                    continue
            step = grid.resolution_m * (math.sqrt(2.0) if delta_row and delta_column else 1.0)
            next_cost = current_cost + step
            if next_cost < costs.get(candidate, math.inf):
                costs[candidate] = next_cost
                heapq.heappush(queue, (next_cost, candidate))
    return costs


def extract_reachable_viewpoints(
    grid: OccupancyGrid2D,
    *,
    robot_pose_in_grid_frame: Pose2D,
    config: ReachableViewpointConfig = ReachableViewpointConfig(),
) -> tuple[ReachableViewpoint, ...]:
    """Return geometry-only free-space viewpoints; no goal or instruction is accepted."""

    raw_start = grid.world_to_cell(robot_pose_in_grid_frame.x_m, robot_pose_in_grid_frame.y_m)
    start = _nearest_free(grid, raw_start, config)
    if start is None:
        raise _fail("robot cannot bind to a current free cell", "NO_REACHABLE_ROBOT_CELL")
    reachable = _reachable_costs(grid, start, config)
    sector_width = 360.0 / config.angular_sectors
    ranked_by_sector: dict[int, tuple[tuple[float, float, Cell], dict[str, Any]]] = {}
    for cell, geodesic in reachable.items():
        if cell == start:
            continue
        grid_x, grid_y = grid.cell_to_world(cell)
        delta_x = grid_x - robot_pose_in_grid_frame.x_m
        delta_y = grid_y - robot_pose_in_grid_frame.y_m
        cosine = math.cos(robot_pose_in_grid_frame.yaw_rad)
        sine = math.sin(robot_pose_in_grid_frame.yaw_rad)
        forward = cosine * delta_x + sine * delta_y
        left = -sine * delta_x + cosine * delta_y
        relative_x = -left
        distance = math.hypot(relative_x, forward)
        if not config.minimum_distance_m <= distance <= config.maximum_distance_m:
            continue
        bearing = math.degrees(math.atan2(relative_x, forward))
        sector = min(
            config.angular_sectors - 1,
            int((bearing + 180.0) / sector_width),
        )
        # Prefer a useful bounded look-ahead distance, then the shorter safe path,
        # then the cell coordinate.  No instruction, goal, or oracle enters ranking.
        score = (abs(distance - config.preferred_distance_m), geodesic, cell)
        payload = {
            "relative_xz": (_clean(relative_x), _clean(forward)),
            "grid_xy": (_clean(grid_x), _clean(grid_y)),
            "candidate_xy_frame": grid.frame_id,
            "distance_m": _clean(distance),
            "geodesic_distance_m": _clean(geodesic),
            "bearing_deg": _clean(bearing),
        }
        prior = ranked_by_sector.get(sector)
        if prior is None or score < prior[0]:
            ranked_by_sector[sector] = (score, payload)

    selected = [item[1] for item in ranked_by_sector.values()]
    selected.sort(key=lambda item: (item["bearing_deg"], item["distance_m"], item["grid_xy"]))
    selected = selected[: config.maximum_candidates]
    if not selected:
        raise _fail(
            "current free-space component contains no bounded viewpoint candidate",
            "NO_REACHABLE_VIEWPOINT_CANDIDATES",
        )
    return tuple(
        ReachableViewpoint(candidate_id=index, **payload)
        for index, payload in enumerate(selected)
    )


def slow_planner_v1_compatibility() -> dict[str, Any]:
    """Machine-readable refusal to relabel viewpoints as frozen-v1 frontiers."""

    return {
        "protocol_version": 1,
        "status": "BLOCKED",
        "blocker_code": SLOW_PLANNER_V1_BLOCKER,
        "candidate_type": REACHABLE_VIEWPOINT_CANDIDATE_TYPE,
        "reason": (
            "SlowPlanner v1 accepts only candidate_frontiers/select_frontier; "
            "reachable free-space viewpoints are not Nav2 unknown-boundary frontiers"
        ),
        "conversion_to_candidate_frontiers_allowed": False,
        "online_control_allowed": False,
    }


def _content_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in value if key != "candidate_set_sha256"}


def viewpoint_content_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _content_payload(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_reachable_viewpoint_mapping(
    *,
    source_node: str,
    identity: LaneBSnapshotIdentity,
    ros_episode_id: str,
    captured_sim_time_s: float,
    valid_until_sim_time_s: float,
    robot_pose_in_map_frame: Pose2D,
    candidates: Sequence[ReachableViewpoint],
) -> dict[str, Any]:
    if not source_node.startswith("/"):
        raise _fail("source_node must be fully qualified", "INVALID_VIEWPOINT_SOURCE")
    if ros_episode_id != f"b::{identity.episode_id}":
        raise _fail("ROS episode identity mismatch", "INVALID_VIEWPOINT_IDENTITY")
    captured = _finite(captured_sim_time_s, "captured_sim_time_s")
    valid_until = _finite(valid_until_sim_time_s, "valid_until_sim_time_s")
    if captured <= 0.0 or valid_until < captured:
        raise _fail("viewpoint validity interval is invalid", "INVALID_VIEWPOINT_IDENTITY")
    if not candidates:
        raise _fail("at least one viewpoint is required", "NO_REACHABLE_VIEWPOINT_CANDIDATES")
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise _fail("candidate IDs must be unique", "INVALID_VIEWPOINT_CANDIDATE")
    candidate_frames = {candidate.candidate_xy_frame for candidate in candidates}
    if len(candidate_frames) != 1:
        raise _fail("candidate coordinate frames must agree", "INVALID_VIEWPOINT_CANDIDATE")
    grid_frame_id = next(iter(candidate_frames))
    payload: dict[str, Any] = {
        "schema_version": REACHABLE_VIEWPOINT_SCHEMA_VERSION,
        "kind": REACHABLE_VIEWPOINT_KIND,
        "source_node": source_node,
        "source": REACHABLE_VIEWPOINT_SOURCE,
        "candidate_type": REACHABLE_VIEWPOINT_CANDIDATE_TYPE,
        "ros_episode_id": ros_episode_id,
        "episode_id": identity.episode_id,
        "reset_id": identity.reset_id,
        "sequence_id": identity.sequence_id,
        "snapshot_id": identity.snapshot_id,
        "captured_sim_time_s": _clean(captured),
        "valid_until_sim_time_s": _clean(valid_until),
        "geometry_frame": "base_link",
        "grid_frame_id": grid_frame_id,
        "agent_pose_encoding": "map_xy_yaw_rad",
        "agent_pose": [
            _clean(robot_pose_in_map_frame.x_m),
            _clean(robot_pose_in_map_frame.y_m),
            _clean(robot_pose_in_map_frame.yaw_rad),
        ],
        "candidates": [candidate.to_mapping() for candidate in candidates],
        "slow_planner_v1_compatibility": slow_planner_v1_compatibility(),
        "authority": dict(_AUTHORITY),
        "oracle_inputs": dict(_ORACLE_INPUTS),
        "candidate_set_sha256": "",
    }
    payload["candidate_set_sha256"] = viewpoint_content_sha256(payload)
    validate_reachable_viewpoint_mapping(payload)
    return payload


def validate_reachable_viewpoint_mapping(value: Mapping[str, Any]) -> None:
    """Validate identity, type/source, zero authority, and canonical digest."""

    if not isinstance(value, Mapping) or set(value) != _MAPPING_KEYS:
        raise _fail("reachable-viewpoint keys differ", "INVALID_VIEWPOINT_SOURCE")
    if value.get("schema_version") != REACHABLE_VIEWPOINT_SCHEMA_VERSION or value.get("kind") != REACHABLE_VIEWPOINT_KIND:
        raise _fail("unsupported reachable-viewpoint schema", "INVALID_VIEWPOINT_SOURCE")
    if value.get("candidate_type") != REACHABLE_VIEWPOINT_CANDIDATE_TYPE or value.get("source") != REACHABLE_VIEWPOINT_SOURCE:
        raise _fail("viewpoint type/source mismatch", "INVALID_VIEWPOINT_SOURCE")
    if not str(value.get("source_node") or "").startswith("/"):
        raise _fail("source_node must be fully qualified", "INVALID_VIEWPOINT_SOURCE")
    if value.get("geometry_frame") != "base_link" or value.get("agent_pose_encoding") != "map_xy_yaw_rad":
        raise _fail("viewpoint frame encoding mismatch", "INVALID_VIEWPOINT_SOURCE")
    grid_frame_id = value.get("grid_frame_id")
    if not isinstance(grid_frame_id, str) or not grid_frame_id.strip():
        raise _fail("grid_frame_id is required", "INVALID_VIEWPOINT_SOURCE")
    identity = LaneBSnapshotIdentity(
        str(value.get("episode_id") or ""),
        value.get("reset_id"),
        value.get("sequence_id"),
    )
    if value.get("snapshot_id") != identity.snapshot_id or value.get("ros_episode_id") != f"b::{identity.episode_id}":
        raise _fail("viewpoint identity fields disagree", "INVALID_VIEWPOINT_IDENTITY")
    captured = _finite(value.get("captured_sim_time_s"), "captured_sim_time_s")
    valid_until = _finite(value.get("valid_until_sim_time_s"), "valid_until_sim_time_s")
    if captured <= 0.0 or valid_until < captured:
        raise _fail("viewpoint validity interval is invalid", "INVALID_VIEWPOINT_IDENTITY")
    pose = value.get("agent_pose")
    if not isinstance(pose, Sequence) or isinstance(pose, (str, bytes)) or len(pose) != 3:
        raise _fail("agent_pose encoding is invalid", "INVALID_VIEWPOINT_SOURCE")
    for index, item in enumerate(pose):
        _finite(item, f"agent_pose[{index}]")
    compatibility = value.get("slow_planner_v1_compatibility")
    if compatibility != slow_planner_v1_compatibility():
        raise _fail("SlowPlanner compatibility receipt mismatch", "INVALID_VIEWPOINT_SOURCE")
    authority = value.get("authority")
    if authority != _AUTHORITY:
        raise _fail("viewpoint source acquired forbidden authority", "INVALID_VIEWPOINT_AUTHORITY")
    if value.get("oracle_inputs") != _ORACLE_INPUTS:
        raise _fail("viewpoint source declares oracle input", "INVALID_VIEWPOINT_AUTHORITY")
    candidates = value.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
        raise _fail("viewpoint candidates must be a nonempty array", "INVALID_VIEWPOINT_CANDIDATE")
    parsed = []
    for item in candidates:
        if not isinstance(item, Mapping) or set(item) != _CANDIDATE_KEYS:
            raise _fail("viewpoint candidate is invalid or mislabeled", "INVALID_VIEWPOINT_CANDIDATE")
        parsed.append(
            ReachableViewpoint(
                candidate_id=item.get("candidate_id"),
                candidate_type=item.get("candidate_type"),
                source=item.get("source"),
                candidate_xy_frame=item.get("candidate_xy_frame"),
                relative_xz=tuple(item.get("relative_xz") or ()),
                grid_xy=tuple(item.get("grid_xy") or ()),
                distance_m=item.get("distance_m"),
                geodesic_distance_m=item.get("geodesic_distance_m"),
                bearing_deg=item.get("bearing_deg"),
            )
        )
    if len({item.candidate_id for item in parsed}) != len(parsed):
        raise _fail("viewpoint candidate IDs must be unique", "INVALID_VIEWPOINT_CANDIDATE")
    if any(item.candidate_xy_frame != grid_frame_id for item in parsed):
        raise _fail("candidate coordinate frame mismatch", "INVALID_VIEWPOINT_CANDIDATE")
    supplied_hash = str(value.get("candidate_set_sha256") or "")
    if len(supplied_hash) != 64 or supplied_hash != viewpoint_content_sha256(value):
        raise _fail("viewpoint content hash mismatch", "INVALID_VIEWPOINT_SOURCE")
