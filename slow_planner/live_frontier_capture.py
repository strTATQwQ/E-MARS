"""Coordinator-owned, capture-only producer for live Nav2 frontier snapshots.

The extraction boundary accepts only a current occupancy grid, robot poses and
the existing episode/reset/sequence identity.  It has no instruction, oracle,
goal, model, command or terminal-state input.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .base import CandidateFrontier
from .lane_b import LaneBSnapshotIdentity
from .live_frontier import (
    LIVE_FRONTIER_AGENT_POSE_ENCODING,
    LIVE_FRONTIER_GEOMETRY_FRAME,
    LIVE_FRONTIER_KIND,
    LIVE_FRONTIER_SCHEMA_VERSION,
    LiveNav2FrontierSnapshot,
    live_frontier_content_sha256,
)


Cell = tuple[int, int]
_CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = _CARDINAL + ((-1, -1), (-1, 1), (1, -1), (1, 1))


class LiveFrontierCaptureError(ValueError):
    """A capture input failed closed before any snapshot was published."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def _fail(message: str, code: str) -> LiveFrontierCaptureError:
    return LiveFrontierCaptureError(message, code=code)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise _fail(f"{name} must be numeric", "INVALID_CAPTURE_INPUT")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _fail(f"{name} must be numeric", "INVALID_CAPTURE_INPUT") from exc
    if not math.isfinite(number):
        raise _fail(f"{name} must be finite", "INVALID_CAPTURE_INPUT")
    return number


def _clean_float(value: float) -> float:
    rounded = round(float(value), 9)
    return 0.0 if rounded == 0.0 else rounded


def resolve_lane_topic(namespace: str, relative_topic: str) -> str:
    """Resolve one lane-relative topic while rejecting root-topic escape."""

    lane = str(namespace).rstrip("/")
    topic = str(relative_topic).strip("/")
    if not lane.startswith("/") or lane == "" or "//" in lane:
        raise _fail("namespace must be a non-root absolute lane", "INVALID_NAMESPACE")
    if not topic or relative_topic.startswith("/") or "//" in topic or ".." in topic.split("/"):
        raise _fail("topic must be lane-relative", "INVALID_NAMESPACE")
    return f"{lane}/{topic}"


@dataclass(frozen=True)
class OccupancyGrid2D:
    width: int
    height: int
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    origin_yaw_rad: float
    frame_id: str
    stamp_sim_time_s: float
    data: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or self.width <= 0:
            raise _fail("grid width must be positive", "INVALID_OCCUPANCY_GRID")
        if isinstance(self.height, bool) or self.height <= 0:
            raise _fail("grid height must be positive", "INVALID_OCCUPANCY_GRID")
        if _finite(self.resolution_m, "resolution_m") <= 0.0:
            raise _fail("grid resolution must be positive", "INVALID_OCCUPANCY_GRID")
        _finite(self.origin_x_m, "origin_x_m")
        _finite(self.origin_y_m, "origin_y_m")
        _finite(self.origin_yaw_rad, "origin_yaw_rad")
        if not str(self.frame_id).strip():
            raise _fail("grid frame_id is required", "INVALID_OCCUPANCY_GRID")
        if _finite(self.stamp_sim_time_s, "stamp_sim_time_s") <= 0.0:
            raise _fail("grid sim stamp must be positive", "INVALID_OCCUPANCY_GRID")
        if len(self.data) != self.width * self.height:
            raise _fail("grid data length mismatch", "INVALID_OCCUPANCY_GRID")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.data):
            raise _fail("grid cells must be integers", "INVALID_OCCUPANCY_GRID")

    def inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.height and 0 <= cell[1] < self.width

    def value(self, cell: Cell) -> int:
        return self.data[cell[0] * self.width + cell[1]]

    def world_to_cell(self, x_m: float, y_m: float) -> Cell:
        delta_x = float(x_m) - self.origin_x_m
        delta_y = float(y_m) - self.origin_y_m
        cosine = math.cos(self.origin_yaw_rad)
        sine = math.sin(self.origin_yaw_rad)
        local_x = cosine * delta_x + sine * delta_y
        local_y = -sine * delta_x + cosine * delta_y
        return math.floor(local_y / self.resolution_m), math.floor(local_x / self.resolution_m)

    def cell_to_world(self, cell: Cell) -> tuple[float, float]:
        local_x = (cell[1] + 0.5) * self.resolution_m
        local_y = (cell[0] + 0.5) * self.resolution_m
        cosine = math.cos(self.origin_yaw_rad)
        sine = math.sin(self.origin_yaw_rad)
        return (
            self.origin_x_m + cosine * local_x - sine * local_y,
            self.origin_y_m + sine * local_x + cosine * local_y,
        )


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        _finite(self.x_m, "pose.x_m")
        _finite(self.y_m, "pose.y_m")
        _finite(self.yaw_rad, "pose.yaw_rad")


@dataclass(frozen=True)
class CaptureIdentity:
    identity: LaneBSnapshotIdentity
    ros_episode_id: str
    captured_sim_time_s: float
    valid_until_sim_time_s: float

    def __post_init__(self) -> None:
        if self.ros_episode_id != f"b::{self.identity.episode_id}":
            raise _fail(
                "ROS episode identity must equal b::<episode>",
                "INVALID_CAPTURE_IDENTITY",
            )
        captured = _finite(self.captured_sim_time_s, "captured_sim_time_s")
        valid_until = _finite(self.valid_until_sim_time_s, "valid_until_sim_time_s")
        if captured <= 0.0 or valid_until < captured:
            raise _fail("identity sim-time validity is invalid", "INVALID_CAPTURE_IDENTITY")


@dataclass(frozen=True)
class FrontierCaptureConfig:
    free_max_value: int = 0
    unknown_value: int = -1
    nearest_free_radius_cells: int = 6
    minimum_cluster_cells: int = 2
    minimum_distance_m: float = 0.25
    maximum_distance_m: float = 12.0
    maximum_frontiers: int = 12
    maximum_metadata_age_s: float = 2.5
    maximum_costmap_age_s: float = 2.5
    maximum_odometry_age_s: float = 2.5
    snapshot_ttl_s: float = 2.5

    def __post_init__(self) -> None:
        integer_fields = (
            self.nearest_free_radius_cells,
            self.minimum_cluster_cells,
            self.maximum_frontiers,
        )
        if any(isinstance(value, bool) or value <= 0 for value in integer_fields):
            raise _fail("capture integer limits must be positive", "INVALID_CAPTURE_CONFIG")
        if self.free_max_value < 0 or self.unknown_value >= 0:
            raise _fail("free/unknown thresholds are invalid", "INVALID_CAPTURE_CONFIG")
        if not 0.0 <= self.minimum_distance_m < self.maximum_distance_m:
            raise _fail("frontier distance interval is invalid", "INVALID_CAPTURE_CONFIG")
        for name in (
            "maximum_metadata_age_s",
            "maximum_costmap_age_s",
            "maximum_odometry_age_s",
            "snapshot_ttl_s",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise _fail(f"{name} must be positive", "INVALID_CAPTURE_CONFIG")


def _is_free(grid: OccupancyGrid2D, cell: Cell, config: FrontierCaptureConfig) -> bool:
    value = grid.value(cell)
    return 0 <= value <= config.free_max_value


def _neighbors(grid: OccupancyGrid2D, cell: Cell, *, diagonal: bool) -> Iterable[Cell]:
    offsets = _EIGHT if diagonal else _CARDINAL
    for delta_row, delta_column in offsets:
        candidate = (cell[0] + delta_row, cell[1] + delta_column)
        if grid.inside(candidate):
            yield candidate


def _nearest_free(
    grid: OccupancyGrid2D,
    cell: Cell,
    config: FrontierCaptureConfig,
) -> Cell | None:
    candidates = []
    radius = config.nearest_free_radius_cells
    for delta_row in range(-radius, radius + 1):
        for delta_column in range(-radius, radius + 1):
            candidate = (cell[0] + delta_row, cell[1] + delta_column)
            if grid.inside(candidate) and _is_free(grid, candidate, config):
                candidates.append(
                    (delta_row * delta_row + delta_column * delta_column, candidate)
                )
    return min(candidates)[1] if candidates else None


def _reachable_free(
    grid: OccupancyGrid2D,
    start: Cell,
    config: FrontierCaptureConfig,
) -> dict[Cell, int]:
    distance = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in _neighbors(grid, current, diagonal=True):
            if neighbor in distance or not _is_free(grid, neighbor, config):
                continue
            delta_row = neighbor[0] - current[0]
            delta_column = neighbor[1] - current[1]
            if delta_row and delta_column:
                side_a = (current[0] + delta_row, current[1])
                side_b = (current[0], current[1] + delta_column)
                if not _is_free(grid, side_a, config) or not _is_free(grid, side_b, config):
                    continue
            distance[neighbor] = distance[current] + 1
            queue.append(neighbor)
    return distance


def _frontier_cells(
    grid: OccupancyGrid2D,
    reachable: Mapping[Cell, int],
    config: FrontierCaptureConfig,
) -> set[Cell]:
    return {
        cell
        for cell in reachable
        if any(
            grid.value(neighbor) == config.unknown_value
            for neighbor in _neighbors(grid, cell, diagonal=False)
        )
    }


def _clusters(grid: OccupancyGrid2D, cells: set[Cell]) -> list[set[Cell]]:
    remaining = set(cells)
    result = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        cluster = {seed}
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbor in _neighbors(grid, current, diagonal=True):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cluster.add(neighbor)
                    queue.append(neighbor)
        result.append(cluster)
    return result


def extract_reachable_frontiers(
    grid: OccupancyGrid2D,
    *,
    robot_pose_in_grid_frame: Pose2D,
    config: FrontierCaptureConfig = FrontierCaptureConfig(),
) -> tuple[CandidateFrontier, ...]:
    """Extract current reachable frontiers without any navigation goal input."""

    raw_start = grid.world_to_cell(
        robot_pose_in_grid_frame.x_m, robot_pose_in_grid_frame.y_m
    )
    start = _nearest_free(grid, raw_start, config)
    if start is None:
        raise _fail("robot cannot bind to a free costmap cell", "NO_REACHABLE_ROBOT_CELL")
    reachable = _reachable_free(grid, start, config)
    frontier_cells = _frontier_cells(grid, reachable, config)
    ranked: list[tuple[int, float, Cell, tuple[float, float]]] = []
    for cluster in _clusters(grid, frontier_cells):
        if len(cluster) < config.minimum_cluster_cells:
            continue
        centroid_row = sum(cell[0] for cell in cluster) / len(cluster)
        centroid_column = sum(cell[1] for cell in cluster) / len(cluster)
        representative = min(
            cluster,
            key=lambda cell: (
                (cell[0] - centroid_row) ** 2 + (cell[1] - centroid_column) ** 2,
                reachable[cell],
                cell,
            ),
        )
        world = grid.cell_to_world(representative)
        delta_x = world[0] - robot_pose_in_grid_frame.x_m
        delta_y = world[1] - robot_pose_in_grid_frame.y_m
        cosine = math.cos(robot_pose_in_grid_frame.yaw_rad)
        sine = math.sin(robot_pose_in_grid_frame.yaw_rad)
        forward = cosine * delta_x + sine * delta_y
        left = -sine * delta_x + cosine * delta_y
        relative_x = -left
        distance = math.hypot(relative_x, forward)
        if not config.minimum_distance_m <= distance <= config.maximum_distance_m:
            continue
        bearing = math.degrees(math.atan2(relative_x, forward))
        ranked.append(
            (
                reachable[representative],
                bearing,
                representative,
                (_clean_float(relative_x), _clean_float(forward)),
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    frontiers = []
    for frontier_id, (_, bearing, _cell, relative) in enumerate(
        ranked[: config.maximum_frontiers]
    ):
        frontiers.append(
            CandidateFrontier(
                frontier_id=frontier_id,
                relative_xz=relative,
                distance_m=_clean_float(math.hypot(*relative)),
                bearing_deg=_clean_float(bearing),
            )
        )
    return tuple(frontiers)


def build_live_frontier_mapping(
    *,
    source_node: str,
    identity: CaptureIdentity,
    robot_pose_in_map_frame: Pose2D,
    candidate_frontiers: Sequence[CandidateFrontier],
    valid_until_sim_time_s: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": LIVE_FRONTIER_SCHEMA_VERSION,
        "kind": LIVE_FRONTIER_KIND,
        "source_node": str(source_node),
        "ros_episode_id": identity.ros_episode_id,
        "episode_id": identity.identity.episode_id,
        "reset_id": identity.identity.reset_id,
        "sequence_id": identity.identity.sequence_id,
        "snapshot_id": identity.identity.snapshot_id,
        "captured_sim_time_s": _clean_float(identity.captured_sim_time_s),
        "valid_until_sim_time_s": _clean_float(valid_until_sim_time_s),
        "geometry_frame": LIVE_FRONTIER_GEOMETRY_FRAME,
        "agent_pose_encoding": LIVE_FRONTIER_AGENT_POSE_ENCODING,
        "agent_pose": [
            _clean_float(robot_pose_in_map_frame.x_m),
            _clean_float(robot_pose_in_map_frame.y_m),
            _clean_float(robot_pose_in_map_frame.yaw_rad),
        ],
        "candidate_frontiers": [frontier.to_mapping() for frontier in candidate_frontiers],
        "frontier_set_sha256": "",
    }
    payload["frontier_set_sha256"] = live_frontier_content_sha256(payload)
    LiveNav2FrontierSnapshot.from_mapping(payload)
    return payload


class AtomicLiveFrontierWriter:
    """Atomically replace or clear the single current snapshot file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    def publish(self, payload: Mapping[str, Any]) -> None:
        LiveNav2FrontierSnapshot.from_mapping(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


class LiveFrontierCaptureCoordinator:
    """Identity/freshness gate around the pure extractor and atomic writer."""

    def __init__(
        self,
        writer: AtomicLiveFrontierWriter,
        *,
        source_node: str,
        config: FrontierCaptureConfig = FrontierCaptureConfig(),
    ) -> None:
        if not str(source_node).startswith("/"):
            raise _fail("source_node must be fully qualified", "INVALID_NAMESPACE")
        self.writer = writer
        self.source_node = str(source_node)
        self.config = config
        self._current_identity: LaneBSnapshotIdentity | None = None
        self._published_identity: LaneBSnapshotIdentity | None = None
        self._published_payload: dict[str, Any] | None = None

    def clear(self) -> None:
        self.writer.clear()
        self._published_identity = None
        self._published_payload = None

    def clear_for_missing_identity(self) -> None:
        """Remove publishable state while retaining monotonic reset history."""

        self.clear()

    def observe_identity(self, identity: LaneBSnapshotIdentity) -> None:
        current = self._current_identity
        if current is not None:
            if identity.reset_id < current.reset_id:
                raise _fail("reset generation regressed", "STALE_CAPTURE_IDENTITY")
            if identity.reset_id == current.reset_id:
                if identity.episode_id != current.episode_id:
                    raise _fail("episode changed without reset", "STALE_CAPTURE_IDENTITY")
                if identity.sequence_id < current.sequence_id:
                    raise _fail("sequence regressed", "STALE_CAPTURE_IDENTITY")
            elif identity.sequence_id != 0:
                raise _fail("new reset must begin at sequence zero", "INVALID_CAPTURE_IDENTITY")
        if current != identity:
            self.clear()
            self._current_identity = identity

    @staticmethod
    def _require_fresh(name: str, stamp: float, now: float, maximum_age: float) -> None:
        value = _finite(stamp, f"{name}_sim_time_s")
        if value <= 0.0 or value > now or now - value > maximum_age:
            raise _fail(f"{name} is stale or from the future", f"STALE_{name.upper()}")

    def expire(self, *, sim_now_s: float) -> None:
        now = _finite(sim_now_s, "sim_now_s")
        if self._published_payload is None:
            return
        if now > float(self._published_payload["valid_until_sim_time_s"]):
            self.clear()

    def capture(
        self,
        *,
        identity: CaptureIdentity,
        grid: OccupancyGrid2D,
        robot_pose_in_grid_frame: Pose2D,
        robot_pose_in_map_frame: Pose2D,
        odometry_sim_time_s: float,
        sim_now_s: float,
    ) -> dict[str, Any]:
        now = _finite(sim_now_s, "sim_now_s")
        self.observe_identity(identity.identity)
        self.expire(sim_now_s=now)
        if (
            self._published_identity == identity.identity
            and self._published_payload is not None
            and self.writer.path.is_file()
        ):
            return dict(self._published_payload)
        try:
            self._require_fresh(
                "metadata",
                identity.captured_sim_time_s,
                now,
                self.config.maximum_metadata_age_s,
            )
            if now > identity.valid_until_sim_time_s:
                raise _fail("metadata validity expired", "STALE_METADATA")
            self._require_fresh(
                "costmap",
                grid.stamp_sim_time_s,
                now,
                self.config.maximum_costmap_age_s,
            )
            self._require_fresh(
                "odometry",
                odometry_sim_time_s,
                now,
                self.config.maximum_odometry_age_s,
            )
            frontiers = extract_reachable_frontiers(
                grid,
                robot_pose_in_grid_frame=robot_pose_in_grid_frame,
                config=self.config,
            )
            if not frontiers:
                raise _fail("no current reachable frontier", "NO_CURRENT_LEGAL_FRONTIERS")
            valid_until = min(
                identity.valid_until_sim_time_s,
                identity.captured_sim_time_s + self.config.snapshot_ttl_s,
                grid.stamp_sim_time_s + self.config.maximum_costmap_age_s,
                odometry_sim_time_s + self.config.maximum_odometry_age_s,
            )
            if valid_until < now:
                raise _fail("derived snapshot validity expired", "STALE_CAPTURE_INPUT")
            payload = build_live_frontier_mapping(
                source_node=self.source_node,
                identity=identity,
                robot_pose_in_map_frame=robot_pose_in_map_frame,
                candidate_frontiers=frontiers,
                valid_until_sim_time_s=valid_until,
            )
            self.writer.publish(payload)
        except LiveFrontierCaptureError:
            self.clear()
            raise
        self._published_identity = identity.identity
        self._published_payload = payload
        return dict(payload)
