from __future__ import annotations

import heapq
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


FAILURE_REASONS = {
    "timeout",
    "collision",
    "wrong_target",
    "wrong_direction",
    "target_not_visible",
    "stopped_too_early",
    "never_stopped",
    "stale_action_executed",
    "safety_stop",
    "parse_error",
    "no_progress",
    "forward_bias",
    "unsafe_stop",
    "left_workspace",
}


def distance_xy(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay = list(a)[:2]
    bx, by = list(b)[:2]
    return math.hypot(ax - bx, ay - by)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def bearing_deg(robot_pose: Iterable[float], target_xy: Iterable[float]) -> float:
    x, y, yaw = list(robot_pose)[:3]
    tx, ty = list(target_xy)[:2]
    return math.degrees(wrap_to_pi(math.atan2(ty - y, tx - x) - yaw))


def path_length(trajectory: list[dict[str, Any]]) -> float:
    total = 0.0
    last = None
    for row in trajectory:
        pose = row.get("pose") or [row.get("x", 0.0), row.get("y", 0.0), row.get("yaw", 0.0)]
        if last is not None:
            total += distance_xy(last, pose)
        last = pose
    return total


def shortest_path_length(
    scene: dict[str, Any],
    target_pose: Iterable[float],
    *,
    resolution_m: float = 0.10,
    robot_radius_m: float = 0.30,
) -> float:
    """Compute a deterministic 2-D static-obstacle A* reference path for SPL."""
    if resolution_m <= 0.0 or robot_radius_m < 0.0:
        raise ValueError("resolution_m must be positive and robot_radius_m non-negative")
    start = [float(value) for value in list(scene.get("robot_start_pose") or [0.0, 0.0])[:2]]
    target = [float(value) for value in list(target_pose)[:2]]
    bounds = list(scene.get("bounds") or [])
    if len(bounds) < 4:
        margin = 1.0
        bounds = [min(start[0], target[0]) - margin, min(start[1], target[1]) - margin,
                  max(start[0], target[0]) + margin, max(start[1], target[1]) + margin]
    xmin, ymin, xmax, ymax = (float(value) for value in bounds[:4])
    width = max(1, int(math.ceil((xmax - xmin) / resolution_m)) + 1)
    height = max(1, int(math.ceil((ymax - ymin) / resolution_m)) + 1)

    def cell(point: Iterable[float]) -> tuple[int, int]:
        x, y = list(point)[:2]
        return (
            max(0, min(width - 1, int(round((float(x) - xmin) / resolution_m)))),
            max(0, min(height - 1, int(round((float(y) - ymin) / resolution_m)))),
        )

    blocked: set[tuple[int, int]] = set()
    for obstacle in scene.get("obstacles", []):
        pose = list(obstacle.get("pose") or [0.0, 0.0])
        size = list(obstacle.get("size") or [])
        if len(size) >= 2:
            half_x = 0.5 * abs(float(size[0])) + robot_radius_m
            half_y = 0.5 * abs(float(size[1])) + robot_radius_m
        else:
            radius = obstacle_radius_xy(obstacle)
            half_x = half_y = radius + robot_radius_m
        x0 = max(0, int(math.floor((float(pose[0]) - half_x - xmin) / resolution_m)))
        x1 = min(width - 1, int(math.ceil((float(pose[0]) + half_x - xmin) / resolution_m)))
        y0 = max(0, int(math.floor((float(pose[1]) - half_y - ymin) / resolution_m)))
        y1 = min(height - 1, int(math.ceil((float(pose[1]) + half_y - ymin) / resolution_m)))
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                blocked.add((ix, iy))

    start_cell = cell(start)
    goal_cell = cell(target)
    blocked.discard(start_cell)
    blocked.discard(goal_cell)
    direct = distance_xy(start, target)
    queue: list[tuple[float, float, tuple[int, int]]] = [(direct, 0.0, start_cell)]
    best = {start_cell: 0.0}
    neighbours = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    )
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost > best.get(current, math.inf):
            continue
        if current == goal_cell:
            return max(direct, cost * resolution_m)
        for dx, dy, step_cost in neighbours:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height) or nxt in blocked:
                continue
            candidate = cost + step_cost
            if candidate >= best.get(nxt, math.inf):
                continue
            best[nxt] = candidate
            nx = xmin + nxt[0] * resolution_m
            ny = ymin + nxt[1] * resolution_m
            heuristic = math.hypot(target[0] - nx, target[1] - ny) / resolution_m
            heapq.heappush(queue, (candidate + heuristic, candidate, nxt))
    raise ValueError(f"no static-grid path from {start} to {target} in scene {scene.get('scene_id')!r}")


def object_visible(
    robot_pose: Iterable[float],
    obj: dict[str, Any],
    *,
    fov_deg: float = 90.0,
    max_range_m: float = 12.0,
    blockers: list[dict[str, Any]] | None = None,
) -> bool:
    pose = obj.get("pose", [0.0, 0.0, 0.0])
    dist = distance_xy(robot_pose, pose)
    if dist > max_range_m:
        return False
    if abs(bearing_deg(robot_pose, pose)) > fov_deg * 0.5:
        return False
    # Simple blocker approximation: if an obstacle center is almost on the ray and closer than the object, it blocks.
    for blocker in blockers or []:
        bpose = blocker.get("pose", [0.0, 0.0, 0.0])
        bdist = distance_xy(robot_pose, bpose)
        if bdist >= dist:
            continue
        if abs(bearing_deg(robot_pose, bpose) - bearing_deg(robot_pose, pose)) < 5.0:
            size = blocker.get("size", [0.5, 0.5, 0.5])
            if max(size[:2]) > 0.3:
                return False
    return True


def obstacle_radius_xy(obstacle: dict[str, Any], default: float = 0.25) -> float:
    size = obstacle.get("size")
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        try:
            return max(float(size[0]), float(size[1])) * 0.5
        except (TypeError, ValueError):
            return default
    try:
        return float(obstacle.get("radius", default))
    except (TypeError, ValueError):
        return default


def path_obstacle_status(
    robot_pose: Iterable[float],
    obstacles: list[dict[str, Any]] | None,
    *,
    stop_distance_m: float = 0.5,
    path_half_width_m: float = 0.15,
) -> dict[str, Any]:
    x, y, yaw = list(robot_pose)[:3]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    nearest_distance: float | None = None
    nearest_path_distance: float | None = None
    for obstacle in obstacles or []:
        pose = obstacle.get("pose", [0.0, 0.0, 0.0])
        dx = float(pose[0]) - x
        dy = float(pose[1]) - y
        dist = math.hypot(dx, dy)
        nearest_distance = dist if nearest_distance is None else min(nearest_distance, dist)
        radius = obstacle_radius_xy(obstacle)
        forward = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy
        if forward < -radius:
            continue
        longitudinal_clearance = max(0.0, forward - radius)
        lateral_clearance = max(0.0, abs(lateral) - radius)
        if longitudinal_clearance <= stop_distance_m and lateral_clearance <= path_half_width_m:
            nearest_path_distance = (
                longitudinal_clearance
                if nearest_path_distance is None
                else min(nearest_path_distance, longitudinal_clearance)
            )
    return {
        "local_costmap_clear": nearest_path_distance is None,
        "obstacle_distance_m": nearest_distance,
        "path_obstacle_distance_m": nearest_path_distance,
    }


@dataclass
class SuccessJudgeCore:
    success_hold_sec: float = 1.0
    stop_linear_threshold_mps: float = 0.05
    stop_angular_threshold_radps: float = 0.10
    workspace_margin_m: float = 0.5
    _stop_since: float | None = None
    _collision: bool = False

    def evaluate(
        self,
        task: dict[str, Any],
        scene: dict[str, Any],
        robot_pose: list[float],
        target: dict[str, Any],
        *,
        now_sec: float,
        cmd_vel: dict[str, float] | None = None,
        collision: bool = False,
        fatal_failsafe: bool = False,
    ) -> dict[str, Any]:
        success_cfg = task.get("success", {})
        dist = distance_xy(robot_pose, target.get("pose", [0.0, 0.0, 0.0]))
        visible = object_visible(robot_pose, target, blockers=scene.get("obstacles", []))
        lin = abs((cmd_vel or {}).get("linear_x", 0.0))
        ang = abs((cmd_vel or {}).get("angular_z", 0.0))
        stopped = lin <= self.stop_linear_threshold_mps and ang <= self.stop_angular_threshold_radps
        if stopped and self._stop_since is None:
            self._stop_since = now_sec
        if not stopped:
            self._stop_since = None
        stop_held = stopped and self._stop_since is not None and now_sec - self._stop_since >= self.success_hold_sec
        bounds = scene.get("bounds")
        left_workspace = False
        if bounds:
            xmin, ymin, xmax, ymax = bounds
            left_workspace = not (
                xmin - self.workspace_margin_m <= robot_pose[0] <= xmax + self.workspace_margin_m
                and ymin - self.workspace_margin_m <= robot_pose[1] <= ymax + self.workspace_margin_m
            )
        if collision:
            return self._status(False, True, "collision", dist, visible, now_sec)
        if fatal_failsafe:
            return self._status(False, True, "safety_stop", dist, visible, now_sec)
        if left_workspace:
            return self._status(False, True, "left_workspace", dist, visible, now_sec)
        if dist > success_cfg.get("distance_to_target_m", 2.0):
            return self._status(False, False, "distance_pending", dist, visible, now_sec)
        if success_cfg.get("target_visible", False) and not visible:
            return self._status(False, False, "target_not_visible", dist, visible, now_sec)
        if success_cfg.get("stop_required", True) and not stop_held:
            return self._status(False, False, "never_stopped", dist, visible, now_sec)
        return self._status(True, True, "success", dist, visible, now_sec)

    @staticmethod
    def _status(success: bool, done: bool, reason: str, dist: float, visible: bool, now_sec: float) -> dict[str, Any]:
        return {
            "success": success,
            "done": done,
            "reason": None if success else reason,
            "distance_to_target": round(dist, 3),
            "target_visible": bool(visible),
            "time_sec": round(now_sec, 3),
        }


@dataclass(order=True)
class DelayedMessage:
    release_time: float
    seq: int
    topic: str = field(compare=False)
    message: Any = field(compare=False)
    original_time: float = field(compare=False)


class DelayInjectorCore:
    def __init__(self, delay_sec: float = 0.0, drop_rate: float = 0.0, seed: int = 0):
        self.delay_sec = delay_sec
        self.drop_rate = drop_rate
        self.rng = random.Random(seed)
        self._queue: list[DelayedMessage] = []
        self._seq = 0
        self.records: list[dict[str, Any]] = []

    def enqueue(self, topic: str, message: Any, now: float) -> bool:
        if self.rng.random() < self.drop_rate:
            self.records.append({"topic": topic, "dropped": True, "delay_sec": None, "t": now})
            return False
        self._seq += 1
        heapq.heappush(self._queue, DelayedMessage(now + self.delay_sec, self._seq, topic, message, now))
        return True

    def drain(self, now: float) -> list[dict[str, Any]]:
        ready = []
        while self._queue and self._queue[0].release_time <= now:
            item = heapq.heappop(self._queue)
            delay = now - item.original_time
            record = {"topic": item.topic, "message": item.message, "delay_sec": delay, "dropped": False, "t": now}
            self.records.append(record)
            ready.append(record)
        return ready


@dataclass
class SafetyRecoveryEventCounter:
    """Counts continuous blocked/recovery stretches as events while preserving tick counts."""

    safety_block_ticks: int = 0
    safety_intervention_events: int = 0
    recovery_override_ticks: int = 0
    recovery_override_events: int = 0
    max_continuous_safety_block_sec: float = 0.0
    max_continuous_recovery_sec: float = 0.0
    _safety_active: bool = False
    _recovery_active: bool = False
    _safety_start_t: float | None = None
    _safety_last_t: float | None = None
    _recovery_start_t: float | None = None
    _recovery_last_t: float | None = None

    def update_safety(self, blocked: bool, timestamp_sec: float | None = None) -> None:
        timestamp = float(timestamp_sec or 0.0)
        if blocked:
            self.safety_block_ticks += 1
            if not self._safety_active:
                self.safety_intervention_events += 1
                self._safety_active = True
                self._safety_start_t = timestamp
            self._safety_last_t = timestamp
            self._refresh_safety_duration()
            return
        if self._safety_active:
            self._refresh_safety_duration()
        self._safety_active = False
        self._safety_start_t = None
        self._safety_last_t = None

    def update_recovery(self, active: bool, timestamp_sec: float | None = None) -> None:
        timestamp = float(timestamp_sec or 0.0)
        if active:
            self.recovery_override_ticks += 1
            if not self._recovery_active:
                self.recovery_override_events += 1
                self._recovery_active = True
                self._recovery_start_t = timestamp
            self._recovery_last_t = timestamp
            self._refresh_recovery_duration()
            return
        if self._recovery_active:
            self._refresh_recovery_duration()
        self._recovery_active = False
        self._recovery_start_t = None
        self._recovery_last_t = None

    def finalize(self) -> dict[str, Any]:
        self._refresh_safety_duration()
        self._refresh_recovery_duration()
        return {
            "safety_block_ticks": self.safety_block_ticks,
            "safety_intervention_events": self.safety_intervention_events,
            "recovery_override_ticks": self.recovery_override_ticks,
            "recovery_override_events": self.recovery_override_events,
            "episodes_with_recovery": 1 if self.recovery_override_ticks else 0,
            "max_continuous_safety_block_sec": round(self.max_continuous_safety_block_sec, 3),
            "max_continuous_recovery_sec": round(self.max_continuous_recovery_sec, 3),
        }

    def _refresh_safety_duration(self) -> None:
        if self._safety_start_t is None or self._safety_last_t is None:
            return
        self.max_continuous_safety_block_sec = max(
            self.max_continuous_safety_block_sec,
            max(0.0, self._safety_last_t - self._safety_start_t),
        )

    def _refresh_recovery_duration(self) -> None:
        if self._recovery_start_t is None or self._recovery_last_t is None:
            return
        self.max_continuous_recovery_sec = max(
            self.max_continuous_recovery_sec,
            max(0.0, self._recovery_last_t - self._recovery_start_t),
        )


def eventized_safety_recovery_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    counter = SafetyRecoveryEventCounter()
    for event in sorted(events, key=lambda row: float(row.get("t", 0.0) or 0.0)):
        t = float(event.get("t", 0.0) or 0.0)
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        name = str(event.get("event") or details.get("event_type") or "")
        result = str(details.get("result") or "")

        if name == "safety_local_status_json" or "local_costmap_clear" in details:
            blocked = (
                bool(details.get("estop"))
                or bool(details.get("robot_fallen_or_unstable"))
                or bool(details.get("collision"))
                or not bool(details.get("local_costmap_clear", True))
            )
            counter.update_safety(blocked, t)
        elif result in {"safety_stop", "failsafe", "stopped"} or name in {"safety_stop", "failsafe"}:
            counter.update_safety(True, t)
        elif result in {"accepted", "dry_run_zero", "stale_cmd", "missing_enable_env", "deadman_false"}:
            if str(details.get("event_type") or name) in {"safe_cmd_mux", "primitive_command"}:
                counter.update_safety(False, t)

        if "recovery_override" in details or "override_active" in details:
            counter.update_recovery(bool(details.get("recovery_override") or details.get("override_active")), t)
    return counter.finalize()


def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {}
    n = len(metrics)

    def mean(key: str) -> float:
        return sum(float(m.get(key, 0.0) or 0.0) for m in metrics) / n

    failures: dict[str, int] = {}
    for m in metrics:
        reason = m.get("failure_reason")
        if reason:
            failures[reason] = failures.get(reason, 0) + 1
    top_reason = max(failures.items(), key=lambda item: item[1])[0] if failures else None
    return {
        "episodes": n,
        "success_rate": sum(1 for m in metrics if m.get("success")) / n,
        "clean_success_rate": sum(1 for m in metrics if m.get("clean_success")) / n,
        "recovered_success_rate": sum(1 for m in metrics if m.get("recovered_success")) / n,
        "failure_rate": sum(1 for m in metrics if not m.get("success")) / n,
        "mean_mission_time": mean("mission_time_sec"),
        "mean_path_length": mean("path_length_m"),
        "mean_step_calls": mean("num_step_calls"),
        "mean_omninav_calls": mean("num_omninav_calls"),
        "mean_internnav_calls": mean("num_internnav_calls"),
        "mean_step_latency": mean("step_mean_latency_ms"),
        "mean_omninav_latency": mean("omninav_mean_latency_ms"),
        "mean_model_latency_s": mean("internnav_mean_latency_s"),
        "stale_action_rate": mean("stale_action_rate"),
        "recovery_override_rate": mean("recovery_override_rate"),
        "safety_stop_rate": mean("safety_stop_rate"),
        "collision_rate": mean("collision_rate"),
        "failure_top1": top_reason,
        "stale_discard_count": sum(int(m.get("num_stale_step_results", 0)) + int(m.get("num_stale_omninav_actions", 0)) for m in metrics),
        "unsafe_action_blocked_count": sum(int(m.get("unsafe_action_blocked_count", 0)) for m in metrics),
        "collision_count": sum(int(m.get("num_collisions", 0)) for m in metrics),
        "recovery_override_count": sum(int(m.get("recovery_override_count", 0)) for m in metrics),
        "safety_block_ticks": sum(int(m.get("safety_block_ticks", m.get("unsafe_action_blocked_count", 0)) or 0) for m in metrics),
        "safety_intervention_events": sum(int(m.get("safety_intervention_events", m.get("num_safety_stops", 0)) or 0) for m in metrics),
        "recovery_override_ticks": sum(int(m.get("recovery_override_ticks", m.get("recovery_override_count", 0)) or 0) for m in metrics),
        "recovery_override_events": sum(int(m.get("recovery_override_events", 1 if int(m.get("recovery_override_count", 0) or 0) else 0) or 0) for m in metrics),
        "episodes_with_recovery": sum(int(m.get("episodes_with_recovery", 1 if int(m.get("recovery_override_count", 0) or 0) else 0) or 0) for m in metrics),
        "max_continuous_safety_block_sec": max(float(m.get("max_continuous_safety_block_sec", 0.0) or 0.0) for m in metrics),
        "max_continuous_recovery_sec": max(float(m.get("max_continuous_recovery_sec", 0.0) or 0.0) for m in metrics),
        "internnav_call_count": sum(int(m.get("num_internnav_calls", 0)) for m in metrics),
        "forward_ratio": mean("forward_ratio"),
        "left_ratio": mean("left_ratio"),
        "right_ratio": mean("right_ratio"),
        "stop_ratio": mean("stop_ratio"),
        "action_entropy": mean("action_entropy"),
        "timeout_rate": sum(1 for m in metrics if m.get("failure_reason") == "timeout") / n,
    }


def classify_success(
    *,
    reached: bool,
    failure_reason: str | None = None,
    recovery_override_count: int = 0,
    stale_action_count: int = 0,
    collision_count: int = 0,
    safety_stop_count: int = 0,
    step_correction_count: int = 0,
    clean_recovery_limit: int = 0,
) -> dict[str, Any]:
    if not reached:
        return {
            "success": False,
            "clean_success": False,
            "recovered_success": False,
            "success_class": "failure",
            "failure_reason": failure_reason or "timeout",
        }
    recovered = (
        recovery_override_count > clean_recovery_limit
        or stale_action_count > 0
        or collision_count > 0
        or safety_stop_count > 0
        or step_correction_count > 0
    )
    return {
        "success": True,
        "clean_success": not recovered,
        "recovered_success": recovered,
        "success_class": "recovered_success" if recovered else "clean_success",
        "failure_reason": None,
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            if abs(value) <= 1.0:
                return f"{value:.2f}"
            return f"{value:.1f}"
        return "" if value is None else str(value)

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
