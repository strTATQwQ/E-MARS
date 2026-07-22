"""Safety and coordinate adapter between OmniNav FastPolicy and Nav2.

The released FastPolicy predicts a cumulative two-dimensional waypoint and a
separate heading head.  Only the waypoint norm is used for distance; heading is
``atan2(sin, cos)``.  This module converts that result into ROS REP-103 local
coordinates (x forward, y left), then into the Nav2 ``map`` frame.

The adapter never snaps, rotates, or substitutes a rejected semantic goal.  A
costmap/path checker may accept or reject the exact transformed pose.  Nav2 is
updated only for the first goal, a changed semantic subgoal, a reached goal, or
an invalidated path.  Every decision and execution result can be appended to a
JSONL audit log.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Callable, Protocol, Sequence


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _angle_wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float
    stamp_s: float
    frame_id: str = "map"


@dataclass(frozen=True)
class FastWaypoint:
    """One metric FastPolicy waypoint expressed in the robot frame.

    ``raw_xy_m`` preserves the cumulative Fast head output after the released
    0.3 metric scale.  ``forward_m`` and ``left_m`` are derived from the
    independent heading head and are the only values used for transformation.
    """

    episode_id: str
    subgoal_id: str
    sequence_id: int
    stamp_s: float
    raw_xy_m: tuple[float, float]
    forward_m: float
    left_m: float
    heading_rad: float
    source_frame: str = "base_link"

    @classmethod
    def from_action_head(
        cls,
        *,
        episode_id: str,
        subgoal_id: str,
        sequence_id: int,
        stamp_s: float,
        waypoint_xy_m: Sequence[float],
        heading_sin_cos: Sequence[float],
    ) -> "FastWaypoint":
        if len(waypoint_xy_m) != 2 or len(heading_sin_cos) != 2:
            raise ValueError("waypoint_xy_m and heading_sin_cos must each have length 2")
        raw_x, raw_y = (float(value) for value in waypoint_xy_m)
        sin_value, cos_value = (float(value) for value in heading_sin_cos)
        radius = math.hypot(raw_x, raw_y)
        heading = math.atan2(sin_value, cos_value)
        return cls(
            episode_id=str(episode_id),
            subgoal_id=str(subgoal_id),
            sequence_id=int(sequence_id),
            stamp_s=float(stamp_s),
            raw_xy_m=(raw_x, raw_y),
            forward_m=radius * math.cos(heading),
            left_m=radius * math.sin(heading),
            heading_rad=heading,
        )


@dataclass(frozen=True)
class MapBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y


@dataclass(frozen=True)
class CostmapCheck:
    reachable: bool
    reason: str = "reachable"
    cell_cost: int | None = None
    path_cells: int | None = None


class CostmapChecker(Protocol):
    def check(self, start: Pose2D, goal: Pose2D) -> CostmapCheck: ...


@dataclass(frozen=True)
class ExecutionState:
    active_goal_id: str | None = None
    active_subgoal_id: str | None = None
    goal_reached: bool = False
    path_valid: bool = True


@dataclass(frozen=True)
class AdaptedGoal:
    goal_id: str
    episode_id: str
    subgoal_id: str
    pose: Pose2D
    raw_waypoint: FastWaypoint


@dataclass(frozen=True)
class AdapterDecision:
    accepted: bool
    send_goal: bool
    reason: str
    update_reason: str | None
    goal: AdaptedGoal | None
    costmap: CostmapCheck | None
    timestamp_s: float


@dataclass(frozen=True)
class AdapterConfig:
    target_frame: str = "map"
    max_waypoint_age_s: float = 0.75
    min_waypoint_m: float = 0.05
    max_waypoint_m: float = 3.0
    max_goal_jump_m: float = 2.0
    max_semantic_deviation_deg: float = 90.0
    goal_equivalence_m: float = 0.10
    bounds: MapBounds | None = None

    def __post_init__(self) -> None:
        if self.max_waypoint_age_s <= 0:
            raise ValueError("max_waypoint_age_s must be positive")
        if not 0 <= self.min_waypoint_m <= self.max_waypoint_m:
            raise ValueError("invalid waypoint distance limits")
        if self.max_goal_jump_m <= 0:
            raise ValueError("max_goal_jump_m must be positive")
        if not 0 <= self.max_semantic_deviation_deg <= 180:
            raise ValueError("max_semantic_deviation_deg must be in [0, 180]")


class GridCostmapChecker:
    """Connectivity check over an already-inflated Nav2-style occupancy grid."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        resolution_m: float,
        origin_x: float,
        origin_y: float,
        data: Sequence[int],
        lethal_cost: int = 65,
        allow_unknown: bool = False,
    ) -> None:
        if width <= 0 or height <= 0 or resolution_m <= 0:
            raise ValueError("invalid grid geometry")
        if len(data) != width * height:
            raise ValueError("costmap data length does not match width*height")
        self.width = int(width)
        self.height = int(height)
        self.resolution_m = float(resolution_m)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.data = tuple(int(value) for value in data)
        self.lethal_cost = int(lethal_cost)
        self.allow_unknown = bool(allow_unknown)

    def _cell(self, pose: Pose2D) -> tuple[int, int] | None:
        col = math.floor((pose.x - self.origin_x) / self.resolution_m)
        row = math.floor((pose.y - self.origin_y) / self.resolution_m)
        if not (0 <= col < self.width and 0 <= row < self.height):
            return None
        return int(col), int(row)

    def _cost(self, cell: tuple[int, int]) -> int:
        col, row = cell
        return self.data[row * self.width + col]

    def _traversable(self, cell: tuple[int, int]) -> bool:
        cost = self._cost(cell)
        return (cost < 0 and self.allow_unknown) or 0 <= cost < self.lethal_cost

    def check(self, start: Pose2D, goal: Pose2D) -> CostmapCheck:
        start_cell = self._cell(start)
        goal_cell = self._cell(goal)
        if start_cell is None:
            return CostmapCheck(False, "start_out_of_costmap")
        if goal_cell is None:
            return CostmapCheck(False, "goal_out_of_costmap")
        goal_cost = self._cost(goal_cell)
        if not self._traversable(goal_cell):
            reason = "goal_unknown" if goal_cost < 0 else "goal_lethal"
            return CostmapCheck(False, reason, cell_cost=goal_cost)
        if not self._traversable(start_cell):
            return CostmapCheck(False, "start_not_traversable", cell_cost=self._cost(start_cell))
        if start_cell == goal_cell:
            return CostmapCheck(True, cell_cost=goal_cost, path_cells=1)

        queue: deque[tuple[tuple[int, int], int]] = deque([(start_cell, 1)])
        seen = {start_cell}
        neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))
        while queue:
            (col, row), distance = queue.popleft()
            for dx, dy in neighbours:
                candidate = col + dx, row + dy
                if candidate in seen or not (0 <= candidate[0] < self.width and 0 <= candidate[1] < self.height):
                    continue
                seen.add(candidate)
                if not self._traversable(candidate):
                    continue
                if candidate == goal_cell:
                    return CostmapCheck(True, cell_cost=goal_cost, path_cells=distance + 1)
                queue.append((candidate, distance + 1))
        return CostmapCheck(False, "no_costmap_path", cell_cost=goal_cost)


class OmniNavNav2Adapter:
    def __init__(
        self,
        config: AdapterConfig,
        costmap_checker: CostmapChecker,
        *,
        audit_log: str | Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.costmap_checker = costmap_checker
        self.audit_log = Path(audit_log) if audit_log is not None else None
        self.clock = clock
        self._last_sequence: dict[str, int] = {}
        self._active_goal: AdaptedGoal | None = None
        self._last_sent_goal: AdaptedGoal | None = None

    @staticmethod
    def transform_to_map(robot_pose: Pose2D, waypoint: FastWaypoint, target_frame: str = "map") -> Pose2D:
        cos_yaw = math.cos(robot_pose.yaw)
        sin_yaw = math.sin(robot_pose.yaw)
        x = robot_pose.x + cos_yaw * waypoint.forward_m - sin_yaw * waypoint.left_m
        y = robot_pose.y + sin_yaw * waypoint.forward_m + cos_yaw * waypoint.left_m
        return Pose2D(
            x=x,
            y=y,
            yaw=_angle_wrap(robot_pose.yaw + waypoint.heading_rad),
            stamp_s=waypoint.stamp_s,
            frame_id=target_frame,
        )

    def consider(
        self,
        waypoint: FastWaypoint,
        robot_pose: Pose2D,
        execution: ExecutionState,
        *,
        semantic_bearing_rad: float | None = None,
    ) -> AdapterDecision:
        now = float(self.clock())
        transformed: Pose2D | None = None
        costmap: CostmapCheck | None = None

        def reject(reason: str) -> AdapterDecision:
            decision = AdapterDecision(False, False, reason, None, None, costmap, now)
            self._write_decision(decision, waypoint, robot_pose, transformed)
            return decision

        if not waypoint.episode_id or not waypoint.subgoal_id:
            return reject("missing_identity")
        if waypoint.source_frame != "base_link" or robot_pose.frame_id != self.config.target_frame:
            return reject("frame_mismatch")
        numeric = (*waypoint.raw_xy_m, waypoint.forward_m, waypoint.left_m, waypoint.heading_rad, waypoint.stamp_s)
        if not _finite(*numeric):
            return reject("non_finite_waypoint")
        age_s = now - waypoint.stamp_s
        if age_s < -0.05 or age_s > self.config.max_waypoint_age_s:
            return reject("stale_waypoint")
        previous_sequence = self._last_sequence.get(waypoint.episode_id, -1)
        if waypoint.sequence_id <= previous_sequence:
            return reject("non_monotonic_sequence")
        self._last_sequence[waypoint.episode_id] = waypoint.sequence_id
        distance = math.hypot(waypoint.forward_m, waypoint.left_m)
        if distance < self.config.min_waypoint_m:
            return reject("waypoint_too_short")
        if distance > self.config.max_waypoint_m:
            return reject("waypoint_too_far")
        if semantic_bearing_rad is not None:
            deviation = abs(_angle_wrap(waypoint.heading_rad - float(semantic_bearing_rad)))
            if deviation > math.radians(self.config.max_semantic_deviation_deg):
                return reject("semantic_direction_violation")

        transformed = self.transform_to_map(robot_pose, waypoint, self.config.target_frame)
        if not _finite(transformed.x, transformed.y, transformed.yaw):
            return reject("non_finite_transform")
        if self.config.bounds is not None and not self.config.bounds.contains(transformed.x, transformed.y):
            return reject("goal_out_of_bounds")
        if self._last_sent_goal is not None and self._last_sent_goal.subgoal_id == waypoint.subgoal_id:
            jump = math.hypot(
                transformed.x - self._last_sent_goal.pose.x,
                transformed.y - self._last_sent_goal.pose.y,
            )
            if jump > self.config.max_goal_jump_m:
                return reject("goal_jump")

        costmap = self.costmap_checker.check(robot_pose, transformed)
        if not costmap.reachable:
            return reject(f"costmap_rejected:{costmap.reason}")

        update_reason: str | None
        if self._active_goal is None:
            update_reason = "initial_goal"
        elif waypoint.subgoal_id != self._active_goal.subgoal_id:
            update_reason = "subgoal_changed"
        elif execution.goal_reached:
            update_reason = "goal_reached"
        elif not execution.path_valid:
            update_reason = "path_invalid"
        else:
            update_reason = None

        goal = AdaptedGoal(
            goal_id=f"{waypoint.episode_id}:{waypoint.subgoal_id}:{waypoint.sequence_id}",
            episode_id=waypoint.episode_id,
            subgoal_id=waypoint.subgoal_id,
            pose=transformed,
            raw_waypoint=waypoint,
        )
        if update_reason is None:
            decision = AdapterDecision(True, False, "active_goal_held", None, goal, costmap, now)
        else:
            self._active_goal = goal
            self._last_sent_goal = goal
            decision = AdapterDecision(True, True, "goal_ready", update_reason, goal, costmap, now)
        self._write_decision(decision, waypoint, robot_pose, transformed)
        return decision

    def record_execution(
        self,
        goal_id: str,
        *,
        result: str,
        detail: str = "",
        recoveries: int = 0,
        timestamp_s: float | None = None,
    ) -> None:
        if result not in {"accepted", "rejected", "succeeded", "aborted", "canceled"}:
            raise ValueError("unsupported Nav2 execution result")
        row = {
            "event": "nav2_execution",
            "timestamp_s": float(self.clock() if timestamp_s is None else timestamp_s),
            "goal_id": str(goal_id),
            "result": result,
            "detail": str(detail),
            "recoveries": int(recoveries),
        }
        self._append(row)
        if result in {"succeeded", "aborted", "canceled", "rejected"} and self._active_goal is not None:
            if self._active_goal.goal_id == goal_id:
                self._active_goal = None

    def reset_episode(self, episode_id: str) -> None:
        self._last_sequence.pop(str(episode_id), None)
        self._active_goal = None
        self._last_sent_goal = None

    def _write_decision(
        self,
        decision: AdapterDecision,
        waypoint: FastWaypoint,
        robot_pose: Pose2D,
        transformed: Pose2D | None,
    ) -> None:
        row = {
            "event": "adapter_decision",
            "timestamp_s": decision.timestamp_s,
            "accepted": decision.accepted,
            "send_goal": decision.send_goal,
            "reason": decision.reason,
            "update_reason": decision.update_reason,
            "original_waypoint": asdict(waypoint),
            "robot_pose": asdict(robot_pose),
            "transformed_goal": asdict(transformed) if transformed is not None else None,
            "costmap": asdict(decision.costmap) if decision.costmap is not None else None,
            "goal_id": decision.goal.goal_id if decision.goal is not None else None,
        }
        self._append(row)

    def _append(self, row: dict[str, object]) -> None:
        if self.audit_log is None:
            return
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
