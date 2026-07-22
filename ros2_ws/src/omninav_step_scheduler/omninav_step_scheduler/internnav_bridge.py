from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any


ACTION_CODE_TO_NAME = {
    -1: "stop",
    0: "unknown",
    1: "forward",
    2: "left",
    3: "right",
}

ACTION_TO_PRIMITIVE = {
    "forward": "move_forward",
    "left": "turn_left",
    "right": "turn_right",
    "stop": "stop",
    "unknown": "stop",
}


def action_code_to_name(code: Any) -> str:
    try:
        return ACTION_CODE_TO_NAME.get(int(code), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def normalize_action_name(action: Any) -> str:
    if isinstance(action, str):
        value = action.strip().lower()
        aliases = {
            "move_forward": "forward",
            "forward": "forward",
            "left": "left",
            "turn_left": "left",
            "right": "right",
            "turn_right": "right",
            "stop": "stop",
            "stand_still": "stop",
            "unknown": "unknown",
        }
        return aliases.get(value, "unknown")
    return action_code_to_name(action)


def primitive_for_action(action: Any) -> str:
    return ACTION_TO_PRIMITIVE[normalize_action_name(action)]


def primitive_motion(action: Any, *, forward_distance_m: float = 0.35, turn_yaw_deg: float = 15.0) -> dict[str, Any]:
    name = normalize_action_name(action)
    primitive = primitive_for_action(name)
    if primitive == "move_forward":
        return {"primitive": primitive, "distance_m": float(forward_distance_m), "yaw_deg": 0.0}
    if primitive == "turn_left":
        return {"primitive": primitive, "distance_m": 0.0, "yaw_deg": abs(float(turn_yaw_deg))}
    if primitive == "turn_right":
        return {"primitive": primitive, "distance_m": 0.0, "yaw_deg": -abs(float(turn_yaw_deg))}
    return {"primitive": "stop", "distance_m": 0.0, "yaw_deg": 0.0}


def action_entropy(actions: list[str]) -> float:
    if not actions:
        return 0.0
    counts = Counter(actions)
    total = float(len(actions))
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values())


def planar_distance(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


@dataclass
class ProgressDecision:
    trigger: str | None
    reason: str | None
    forward_ratio: float
    same_action_ratio: float
    action_entropy: float
    progress_m: float
    target_improvement_m: float | None
    recovery_primitive: str | None = None


class InternNavProgressMonitor:
    def __init__(
        self,
        *,
        window_sec: float = 8.0,
        min_progress_m: float = 0.5,
        max_forward_without_goal_update: int = 6,
        max_same_action_ratio: float = 0.85,
    ):
        self.window_sec = float(window_sec)
        self.min_progress_m = float(min_progress_m)
        self.max_forward_without_goal_update = int(max_forward_without_goal_update)
        self.max_same_action_ratio = float(max_same_action_ratio)
        self.samples: deque[dict[str, Any]] = deque()
        self.forward_without_goal_update = 0
        self.last_target_distance: float | None = None
        self.recovery_index = 0

    def reset(self) -> None:
        self.samples.clear()
        self.forward_without_goal_update = 0
        self.last_target_distance = None
        self.recovery_index = 0

    def update(
        self,
        *,
        timestamp: float,
        pose: list[float],
        target_distance_m: float | None,
        model_action: str,
        applied_action: str,
    ) -> ProgressDecision:
        model_action = normalize_action_name(model_action)
        applied_action = normalize_action_name(applied_action)
        if target_distance_m is not None and self.last_target_distance is not None:
            if self.last_target_distance - target_distance_m > 0.05:
                self.forward_without_goal_update = 0
            elif model_action == "forward":
                self.forward_without_goal_update += 1
        elif model_action == "forward":
            self.forward_without_goal_update += 1
        if target_distance_m is not None:
            self.last_target_distance = float(target_distance_m)

        self.samples.append(
            {
                "timestamp": float(timestamp),
                "pose": [float(pose[0]), float(pose[1]), float(pose[2] if len(pose) > 2 else 0.0)],
                "target_distance_m": None if target_distance_m is None else float(target_distance_m),
                "model_action": model_action,
                "applied_action": applied_action,
            }
        )
        while self.samples and float(timestamp) - float(self.samples[0]["timestamp"]) > self.window_sec:
            self.samples.popleft()

        actions = [sample["model_action"] for sample in self.samples]
        counts = Counter(actions)
        forward_ratio = counts.get("forward", 0) / max(len(actions), 1)
        same_action_ratio = max(counts.values()) / max(len(actions), 1) if counts else 0.0
        entropy = action_entropy(actions)
        progress = planar_distance(self.samples[0]["pose"], self.samples[-1]["pose"]) if len(self.samples) >= 2 else 0.0

        first_target = self.samples[0].get("target_distance_m")
        last_target = self.samples[-1].get("target_distance_m")
        target_improvement = None
        if first_target is not None and last_target is not None:
            target_improvement = float(first_target) - float(last_target)

        elapsed = float(self.samples[-1]["timestamp"]) - float(self.samples[0]["timestamp"]) if len(self.samples) >= 2 else 0.0
        no_target_improvement = target_improvement is not None and target_improvement < 0.10
        unknown_or_bad_goal = target_improvement is None or no_target_improvement

        trigger = None
        reason = None
        if elapsed >= self.window_sec * 0.8 and progress < self.min_progress_m:
            trigger = "no_progress"
            reason = "window_progress_below_threshold"
        elif progress >= self.min_progress_m and no_target_improvement:
            trigger = "no_progress"
            reason = "target_distance_not_improving"
        elif (
            self.forward_without_goal_update >= self.max_forward_without_goal_update
            and forward_ratio >= self.max_same_action_ratio
            and same_action_ratio >= self.max_same_action_ratio
            and unknown_or_bad_goal
        ):
            trigger = "forward_bias"
            reason = "forward_without_goal_update"

        return ProgressDecision(
            trigger=trigger,
            reason=reason,
            forward_ratio=forward_ratio,
            same_action_ratio=same_action_ratio,
            action_entropy=entropy,
            progress_m=progress,
            target_improvement_m=target_improvement,
            recovery_primitive=self.next_recovery_primitive() if trigger else None,
        )

    def next_recovery_primitive(self) -> str:
        sequence = ["move_forward", "move_forward", "turn_left", "move_forward"]
        value = sequence[self.recovery_index % len(sequence)]
        self.recovery_index += 1
        return value
