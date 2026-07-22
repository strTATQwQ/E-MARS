from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from .config_loader import dump_data, load_data, object_by_id, scene_by_id, scene_by_type
from .metrics import (
    aggregate_metrics,
    append_jsonl,
    classify_success,
    distance_xy,
    eventized_safety_recovery_metrics,
    path_length,
    shortest_path_length,
)
from .reporting import write_summary


DEFAULT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_DIR = DEFAULT_ROOT / "configs"


def _latency(rng: random.Random, mean_ms: float, std_ms: float) -> float:
    return max(1.0, rng.gauss(mean_ms, std_ms))


class MockEpisodeSimulator:
    def __init__(
        self,
        mode_name: str,
        mode_cfg: dict[str, Any],
        benchmark_cfg: dict[str, Any],
        delay_profile_name: str,
        delay_cfg: dict[str, Any],
        *,
        seed: int,
        mock_models: bool,
        use_isaac: bool,
    ):
        self.mode_name = mode_name
        self.mode_cfg = mode_cfg
        self.benchmark_cfg = benchmark_cfg
        self.delay_profile_name = delay_profile_name
        self.delay_cfg = delay_cfg
        self.seed = seed
        self.mock_models = mock_models
        self.use_isaac = use_isaac

    def run_episode(
        self,
        task: dict[str, Any],
        scene: dict[str, Any],
        episode_dir: Path,
        episode_index: int,
    ) -> dict[str, Any]:
        rng = random.Random(f"{self.seed}:{self.mode_name}:{task['task_id']}:{episode_index}:{self.delay_profile_name}")
        target = object_by_id(scene, task["target_object"])
        start = scene.get("robot_start_pose", [0.0, 0.0, 0.0])
        target_pose = target.get("pose", [0.0, 0.0, 0.0])
        base_dist = distance_xy(start, target_pose)
        mode = self.mode_cfg
        latency_cfg = self.benchmark_cfg.get("latency_model", {})
        step_mean = latency_cfg.get("step_mean_ms", 1850) + 1000.0 * self.delay_cfg.get("step_response_delay_sec", 0.0)
        omni_mean = latency_cfg.get("omninav_mean_ms", 159) + 1000.0 * self.delay_cfg.get("action_delay_sec", 0.0)
        step_latency = _latency(rng, step_mean, latency_cfg.get("step_std_ms", 180))
        omni_latency = _latency(rng, omni_mean, latency_cfg.get("omninav_std_ms", 30))

        step_calls = self._step_calls(task, mode)
        nominal_time = base_dist / self.benchmark_cfg.get("mock_motion", {}).get("nominal_speed_mps", 0.35)
        pending_penalty = self._pending_penalty(mode.get("pending_policy", "none"), step_calls, step_latency)
        mission_time = max(4.0, nominal_time + pending_penalty + rng.uniform(-1.0, 2.5))
        if not mode.get("use_omninav", False):
            mission_time *= 1.25
        if mode.get("step_policy") == "roundrobin_1_1":
            mission_time *= 1.18
        omni_hz = 0.0 if not mode.get("use_omninav", False) else 1000.0 / max(omni_latency, 1.0)
        if mode.get("step_policy") == "roundrobin_1_1":
            omni_hz *= 0.55
        elif mode.get("step_policy") == "periodic_4_1":
            omni_hz *= 0.82
        omni_calls = int(mission_time * omni_hz)
        internnav_latency_s = max(0.015, _latency(rng, 40.0, 8.0) / 1000.0)
        internnav_calls = int(mission_time * 1.0) if mode.get("use_internnav", False) else 0
        if mode.get("use_internnav", False):
            omni_calls = 0

        stale_step = 0
        stale_omni = 0
        runtime_stale = 0
        timebase_error = 0
        stale_action_executed = 0
        unsafe_blocked = 0
        stale_executed = 0
        stale_cfg = self.benchmark_cfg.get("stale_gate", {})
        if step_latency / 1000.0 > stale_cfg.get("max_step_age_sec", 1.0):
            stale_step = max(0, step_calls - 1)
        if omni_latency / 1000.0 > stale_cfg.get("max_omninav_age_sec", 0.35):
            stale_omni = max(0, int(0.08 * omni_calls))
        if not mode.get("stale_gate_enabled", True):
            stale_executed = stale_step + stale_omni
            stale_step = 0
            stale_omni = 0

        dynamic_scene = bool(scene.get("dynamic_obstacles"))
        blocked_scene = bool(scene.get("blocked_paths"))
        if dynamic_scene and mode.get("pending_policy") in {"stop", "safe_scan", "auto"}:
            unsafe_blocked = 1

        success_score = self._success_score(task, scene, mode)
        success_score -= 0.08 * self.delay_cfg.get("drop_rate", 0.0) / 0.1
        if stale_executed:
            success_score -= 0.28
        if dynamic_scene and mode.get("pending_policy") == "move_slow":
            success_score -= 0.06
        if blocked_scene and not mode.get("use_step", False):
            success_score -= 0.18
        success = rng.random() < max(0.02, min(0.98, success_score))

        collisions = 0
        safety_stops = 0
        failure_reason = None
        if not success:
            if stale_executed:
                failure_reason = "stale_action_executed"
            elif mode.get("use_internnav", False) and rng.random() < 0.55:
                failure_reason = "forward_bias" if rng.random() < 0.5 else "no_progress"
            elif dynamic_scene and mode.get("pending_policy") == "move_slow":
                collisions = 1 if rng.random() < 0.35 else 0
                failure_reason = "collision" if collisions else "safety_stop"
            elif blocked_scene and not mode.get("use_step", False):
                failure_reason = "no_progress"
            elif task.get("task_type") in {"ambiguity", "relational_grounding"} and not mode.get("use_step", False):
                failure_reason = "wrong_target"
            elif mission_time > task.get("timeout_sec", 120) * 0.8:
                failure_reason = "timeout"
            else:
                failure_reason = rng.choice(["target_not_visible", "never_stopped", "no_progress"])
        if dynamic_scene and mode.get("pending_policy") in {"stop", "safe_scan", "auto"}:
            safety_stops = 1
        if failure_reason == "collision":
            collisions = max(collisions, 1)

        final_distance = rng.uniform(0.4, task.get("success", {}).get("distance_to_target_m", 2.0) * 0.9) if success else rng.uniform(2.2, 5.0)
        target_visible = bool(task.get("success", {}).get("target_visible", False))
        if not success and failure_reason == "target_not_visible":
            target_visible = False

        events = self._events(task, scene, mission_time, step_calls, omni_calls, step_latency, omni_latency, success, failure_reason)
        if internnav_calls:
            events.extend(self._internnav_events(mission_time, internnav_calls, internnav_latency_s))
            events = sorted(events, key=lambda e: e["t"])
        trajectory = self._trajectory(start, target_pose, mission_time, success)
        model_calls = self._model_calls(step_calls, omni_calls, step_latency, omni_latency)
        model_calls.extend(self._internnav_model_calls(internnav_calls, internnav_latency_s))
        safety_events = self._safety_events(dynamic_scene, safety_stops, unsafe_blocked, collisions)
        action_stats = self._internnav_action_stats(rng, internnav_calls, success)
        recovery_override_count = action_stats["recovery_override_count"]
        safety_ticks = int(safety_stops) + int(unsafe_blocked)
        eventized = {
            "safety_block_ticks": safety_ticks,
            "safety_intervention_events": 1 if safety_ticks else 0,
            "recovery_override_ticks": recovery_override_count,
            "recovery_override_events": 1 if recovery_override_count else 0,
            "episodes_with_recovery": 1 if recovery_override_count else 0,
            "max_continuous_safety_block_sec": round(0.05 * max(0, safety_ticks - 1), 3),
            "max_continuous_recovery_sec": round(0.5 * max(0, recovery_override_count - 1), 3),
        }
        step_correction_count = 1 if success and mode.get("use_step", False) and mode.get("use_internnav", False) and rng.random() < 0.35 else 0
        success_bits = classify_success(
            reached=success,
            failure_reason=failure_reason,
            recovery_override_count=recovery_override_count,
            stale_action_count=stale_step + stale_omni + stale_executed,
            collision_count=collisions,
            safety_stop_count=safety_stops,
            step_correction_count=step_correction_count,
        )
        metrics = {
            "episode_id": f"{self.mode_name}_{task['task_id']}_{episode_index:04d}",
            "mode": self.mode_name,
            "pending_policy": mode.get("pending_policy", "none"),
            "task_id": task["task_id"],
            "task_type": task.get("task_type"),
            "scene_id": scene.get("scene_id"),
            "scene_type": scene.get("scene_type"),
            "mock": self.mock_models,
            "use_isaac": self.use_isaac,
            "oracle_semantics": self.benchmark_cfg.get("oracle_semantics", True),
            "delay_profile": self.delay_profile_name,
            **success_bits,
            "mission_time_sec": round(mission_time, 3),
            "path_length_m": round(base_dist * (1.0 + rng.uniform(0.03, 0.28)), 3),
            "final_distance_to_target_m": round(final_distance, 3),
            "target_visible_at_completion": target_visible,
            "num_step_calls": step_calls,
            "num_step_multimodal_calls": max(0, min(step_calls, 1 if task.get("task_type") in {"ambiguity", "completion_verification"} else 0)),
            "num_omninav_calls": omni_calls,
            "num_internnav_calls": internnav_calls,
            "step_mean_latency_ms": round(step_latency, 1) if step_calls else 0.0,
            "omninav_mean_latency_ms": round(omni_latency, 1) if omni_calls else 0.0,
            "internnav_mean_latency_s": round(internnav_latency_s, 4) if internnav_calls else 0.0,
            "num_stop_and_think": step_calls if mode.get("pending_policy") == "stop" else 0,
            "num_move_while_thinking": step_calls if mode.get("pending_policy") == "move_slow" else 0,
            "num_safe_scan": step_calls if mode.get("pending_policy") == "safe_scan" else 0,
            "num_stale_step_results": stale_step,
            "num_stale_omninav_actions": stale_omni,
            "num_safety_stops": safety_stops,
            "num_collisions": collisions,
            "wrong_turn_count": 1 if failure_reason == "wrong_target" else 0,
            "early_stop_count": 1 if failure_reason == "stopped_too_early" else 0,
            "late_stop_count": 1 if failure_reason == "never_stopped" else 0,
            "average_cmd_age_ms": round(omni_latency * 0.55, 1) if omni_calls else 0.0,
            "stale_action_rate": (stale_step + stale_omni + stale_executed) / max(step_calls + omni_calls, 1),
            "safety_stop_rate": 1.0 if safety_stops else 0.0,
            "collision_rate": 1.0 if collisions else 0.0,
            "unsafe_action_blocked_count": unsafe_blocked,
            "recovery_override_count": recovery_override_count,
            "recovery_override_rate": recovery_override_count / max(internnav_calls, 1),
            "step_correction_count": step_correction_count,
            **eventized,
            **{k: v for k, v in action_stats.items() if k != "recovery_override_count"},
        }
        self._write_episode(episode_dir, task, scene, metrics, events, trajectory, model_calls, safety_events)
        return metrics

    def _success_score(self, task: dict[str, Any], scene: dict[str, Any], mode: dict[str, Any]) -> float:
        name = self.mode_name
        score = {
            "step_only": 0.58,
            "omninav_only": 0.62,
            "step_omninav_event": 0.84,
            "step_omninav_roundrobin_1_1": 0.72,
            "step_omninav_periodic_4_1": 0.78,
            "step_omninav_no_stale_gate": 0.70,
            "step_omninav_stop_only": 0.79,
            "step_omninav_move_while_thinking": 0.80,
            "step_omninav_safe_scan": 0.84,
            "internnav_only": 0.48,
            "step_internnav_event": 0.66,
            "step_internnav_periodic_4_1": 0.70,
            "step_internnav_roundrobin_1_1": 0.62,
        }.get(name, 0.65)
        task_type = task.get("task_type")
        if task_type == "simple_navigation" and mode.get("use_omninav"):
            score += 0.08
        if task_type in {"semantic_target", "relational_grounding", "ambiguity", "completion_verification"}:
            score += 0.08 if mode.get("use_step") else -0.16
        if mode.get("use_internnav") and task_type in {"turn_choice", "semantic_target", "relational_grounding"}:
            score -= 0.10
        if mode.get("use_internnav") and mode.get("use_step"):
            score += 0.12
        if task_type == "failure_recovery":
            score += 0.10 if mode.get("use_step") and (mode.get("use_omninav") or mode.get("use_internnav")) else -0.12
        if scene.get("dynamic_obstacles") and mode.get("pending_policy") == "safe_scan":
            score += 0.06
        return score

    def _step_calls(self, task: dict[str, Any], mode: dict[str, Any]) -> int:
        if not mode.get("use_step", False):
            return 0
        triggers = len(task.get("expected_step_triggers", []))
        policy = mode.get("step_policy")
        if policy == "every_decision":
            return max(3, triggers + 2)
        if policy == "roundrobin_1_1":
            return max(4, triggers + 3)
        if policy == "periodic_4_1":
            return max(2, triggers)
        return max(1, triggers)

    @staticmethod
    def _pending_penalty(policy: str, calls: int, latency_ms: float) -> float:
        if calls <= 0:
            return 0.0
        if policy == "stop":
            return calls * latency_ms / 1000.0
        if policy == "safe_scan":
            return calls * min(1.0, latency_ms / 2000.0)
        if policy == "move_slow":
            return calls * min(0.35, latency_ms / 6000.0)
        if policy == "auto":
            return calls * min(0.8, latency_ms / 3500.0)
        return 0.0

    def _events(
        self,
        task: dict[str, Any],
        scene: dict[str, Any],
        mission_time: float,
        step_calls: int,
        omni_calls: int,
        step_latency: float,
        omni_latency: float,
        success: bool,
        failure_reason: str | None,
    ) -> list[dict[str, Any]]:
        events = [
            {"t": 0.0, "event": "reset", "state": "resetting", "details": {"scene_id": scene.get("scene_id")}},
            {"t": 0.1, "event": "instruction", "state": "running", "details": {"instruction": task.get("instruction")}},
        ]
        events.extend(self._v8_mock_coverage_events(task, scene))
        for i in range(step_calls):
            t = round(0.5 + i * max(0.5, mission_time / max(step_calls, 1)), 3)
            events.append({"t": t, "event": "step_request", "latency_ms": None, "state": "step_pending", "details": {"trigger": task.get("expected_step_triggers", [None])[min(i, len(task.get("expected_step_triggers", [])) - 1)] if task.get("expected_step_triggers") else "periodic"}})
            events.append({"t": round(t + step_latency / 1000.0, 3), "event": "step_response", "latency_ms": round(step_latency, 1), "state": "step_ready", "details": {"stale_checked": self.mode_cfg.get("stale_gate_enabled", True)}})
        for i in range(min(omni_calls, 12)):
            t = round(0.3 + i * max(0.1, mission_time / max(min(omni_calls, 12), 1)), 3)
            events.append({"t": t, "event": "omninav_request", "state": "omninav_running", "details": {"i": i}})
            events.append({"t": round(t + omni_latency / 1000.0, 3), "event": "omninav_response", "latency_ms": round(omni_latency, 1), "state": "candidate_ready", "details": {"primitive": "forward_or_turn"}})
        events.append({"t": round(mission_time, 3), "event": "success" if success else "failure", "state": "done", "details": {"reason": failure_reason}})
        return sorted(events, key=lambda e: e["t"])

    def _v8_mock_coverage_events(self, task: dict[str, Any], scene: dict[str, Any]) -> list[dict[str, Any]]:
        v8_mode = str(self.mode_cfg.get("v8_mode") or "")
        if not v8_mode:
            return []
        task_type = str(task.get("task_type") or "")
        events: list[dict[str, Any]] = []
        if task_type in {"turn_choice", "turn_microbench"} and bool(self.mode_cfg.get("oracle_route_choice", False)):
            route = "left" if "left" in str(task.get("instruction", "")).lower() else "right"
            source = "deterministic_oracle" if "injection" in v8_mode else "forced_oracle"
            primitive = {
                "primitive": "follow_waypoint",
                "distance_m": 0.45,
                "yaw_deg": 35.0 if route == "left" else -35.0,
                "source": source,
                "route_choice": route,
                "request_id": f"mock_v8_route_{route}",
            }
            cmd = {
                "linear": {"x": 0.12, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.35 if route == "left" else -0.35},
            }
            decision = {
                "route_choice": route,
                "confidence": 1.0,
                "source": source,
                "episode_id": "",
                "task_id": task.get("task_id"),
                "visible_in_view": route,
            }
            events.extend(
                [
                    {
                        "t": 1.0,
                        "event": "metrics_event_jsonl",
                        "state": "running",
                        "details": {
                            "event_type": "route_oracle_coverage",
                            "near_intersection_detected": True,
                            "route_oracle_triggered": True,
                            "route_oracle_json_published": True,
                            "route_oracle_choice": route,
                            "route_oracle_source": source,
                        },
                    },
                    {"t": 1.01, "event": "oracle_route_choice_json", "state": "running", "details": decision},
                    {
                        "t": 1.05,
                        "event": "metrics_event_jsonl",
                        "state": "running",
                        "details": {
                            "event_type": "route_choice_bridge",
                            "result": "primitive_published",
                            "decision": decision,
                            "primitive": primitive,
                        },
                    },
                    {
                        "t": 1.10,
                        "event": "metrics_event_jsonl",
                        "state": "running",
                        "details": {
                            "event_type": "primitive_command",
                            "result": "accepted",
                            "action_type": "follow_waypoint",
                            "primitive": primitive,
                            "cmd_vel": cmd,
                        },
                    },
                    {
                        "t": 1.12,
                        "event": "metrics_event_jsonl",
                        "state": "running",
                        "details": {"event_type": "safe_cmd_mux", "result": "accepted", "cmd_vel": cmd, "dry_run": False},
                    },
                    {"t": 1.13, "event": "safe_cmd_vel", "state": "running", "details": cmd},
                ]
            )
        if task_type == "semantic_target" and bool(self.mode_cfg.get("semantic_stop_gate", False)):
            target = object_by_id(scene, task["target_object"])
            distance = round(distance_xy(scene.get("robot_start_pose", [0.0, 0.0, 0.0]), target.get("pose", [0.0, 0.0, 0.0])), 3)
            decision = {
                "stop": True,
                "confidence": 1.0,
                "source": "forced_oracle",
                "episode_id": "",
                "task_id": task.get("task_id"),
                "target_object": task.get("target_object"),
                "target_visible": True,
                "distance_to_target": min(distance, 2.4),
            }
            zero = {
                "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
            primitive = {"primitive": "stop", "source": "forced_oracle", "request_id": "mock_v8_stop"}
            events.extend(
                [
                    {
                        "t": 0.75,
                        "event": "isaac_episode_status",
                        "state": "running",
                        "details": {"target_visible": True, "distance_to_target": min(distance, 2.4), "done": False},
                    },
                    {
                        "t": 1.0,
                        "event": "metrics_event_jsonl",
                        "state": "running",
                        "details": {
                            "event_type": "semantic_stop_coverage",
                            "target_visible": True,
                            "distance_to_target": min(distance, 2.4),
                            "estimated_distance_ok": True,
                            "semantic_stop_oracle_triggered": True,
                            "semantic_stop_json_published": True,
                        },
                    },
                    {"t": 1.01, "event": "oracle_semantic_stop_json", "state": "running", "details": decision},
                    {
                        "t": 1.05,
                        "event": "metrics_event_jsonl",
                        "state": "running",
                        "details": {
                            "event_type": "semantic_stop_bridge",
                            "result": "primitive_published",
                            "decision": decision,
                            "primitive": primitive,
                        },
                    },
                    {
                        "t": 1.10,
                        "event": "metrics_event_jsonl",
                        "state": "running",
                        "details": {
                            "event_type": "primitive_command",
                            "result": "accepted",
                            "action_type": "stop",
                            "primitive": primitive,
                            "cmd_vel": zero,
                        },
                    },
                    {
                        "t": 1.12,
                        "event": "metrics_event_jsonl",
                        "state": "running",
                        "details": {"event_type": "safe_cmd_mux", "result": "accepted", "cmd_vel": zero, "dry_run": False},
                    },
                    {"t": 1.13, "event": "safe_cmd_vel", "state": "running", "details": zero},
                ]
            )
        return events

    @staticmethod
    def _internnav_events(mission_time: float, internnav_calls: int, latency_s: float) -> list[dict[str, Any]]:
        events = []
        for i in range(min(internnav_calls, 24)):
            t = round(0.25 + i * max(0.5, mission_time / max(min(internnav_calls, 24), 1)), 3)
            action = "forward" if i % 5 else "left"
            events.append(
                {
                    "t": t,
                    "event": "internnav_action",
                    "latency_ms": round(latency_s * 1000.0, 1),
                    "state": "internnav_running",
                    "details": {
                        "source": "internnav",
                        "event_type": "internnav_action",
                        "model_action": action,
                        "applied_action": action,
                        "primitive": "move_forward" if action == "forward" else "turn_left",
                        "recovery_override": False,
                    },
                }
            )
        return events

    @staticmethod
    def _trajectory(start: list[float], target: list[float], mission_time: float, success: bool) -> list[dict[str, Any]]:
        samples = max(4, int(mission_time // 2))
        end_fraction = 0.95 if success else 0.55
        rows = []
        for i in range(samples + 1):
            frac = end_fraction * i / samples
            x = start[0] + (target[0] - start[0]) * frac
            y = start[1] + (target[1] - start[1]) * frac
            yaw = math.atan2(target[1] - start[1], target[0] - start[0])
            rows.append({"t": round(mission_time * i / samples, 3), "pose": [round(x, 3), round(y, 3), round(yaw, 3)]})
        return rows

    @staticmethod
    def _model_calls(step_calls: int, omni_calls: int, step_latency: float, omni_latency: float) -> list[dict[str, Any]]:
        rows = []
        for i in range(step_calls):
            rows.append({"model": "step", "call_index": i, "latency_ms": round(step_latency, 1), "multimodal": i == 0})
        for i in range(omni_calls):
            rows.append({"model": "omninav", "call_index": i, "latency_ms": round(omni_latency, 1), "multimodal": False})
        return rows

    @staticmethod
    def _internnav_model_calls(internnav_calls: int, latency_s: float) -> list[dict[str, Any]]:
        return [
            {
                "model": "internnav",
                "request_id": f"internnav_{i:04d}",
                "latency_ms": round(latency_s * 1000.0, 3),
                "result": "response",
                "source": "internnav",
            }
            for i in range(internnav_calls)
        ]

    @staticmethod
    def _internnav_action_stats(rng: random.Random, internnav_calls: int, success: bool) -> dict[str, Any]:
        if internnav_calls <= 0:
            return {
                "forward_ratio": 0.0,
                "left_ratio": 0.0,
                "right_ratio": 0.0,
                "stop_ratio": 0.0,
                "action_entropy": 0.0,
                "recovery_override_count": 0,
            }
        forward = 0.72 + rng.uniform(-0.08, 0.08)
        left = 0.18 + rng.uniform(-0.04, 0.04)
        right = 0.06 + rng.uniform(-0.02, 0.02)
        stop = max(0.0, 1.0 - forward - left - right)
        total = forward + left + right + stop
        ratios = [max(0.0, x / total) for x in (forward, left, right, stop)]
        entropy = -sum(x * math.log(x, 2) for x in ratios if x > 0.0)
        recovery = 0 if success and rng.random() < 0.55 else max(1, int(0.08 * internnav_calls))
        return {
            "forward_ratio": round(ratios[0], 4),
            "left_ratio": round(ratios[1], 4),
            "right_ratio": round(ratios[2], 4),
            "stop_ratio": round(ratios[3], 4),
            "action_entropy": round(entropy, 4),
            "recovery_override_count": recovery,
        }

    @staticmethod
    def _safety_events(dynamic_scene: bool, safety_stops: int, unsafe_blocked: int, collisions: int) -> list[dict[str, Any]]:
        rows = []
        if dynamic_scene:
            rows.append({"t": 2.0, "event": "dynamic_obstacle", "details": "human dummy entered local zone"})
        for i in range(safety_stops):
            rows.append({"t": 2.2 + i, "event": "safety_stop", "details": "local status blocked motion"})
        for i in range(unsafe_blocked):
            rows.append({"t": 2.3 + i, "event": "unsafe_action_blocked", "details": "forbidden zone gate"})
        for i in range(collisions):
            rows.append({"t": 3.0 + i, "event": "collision", "details": "mock collision event"})
        return rows

    @staticmethod
    def _write_episode(
        episode_dir: Path,
        task: dict[str, Any],
        scene: dict[str, Any],
        metrics: dict[str, Any],
        events: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        model_calls: list[dict[str, Any]],
        safety_events: list[dict[str, Any]],
    ) -> None:
        episode_dir.mkdir(parents=True, exist_ok=True)
        dump_data({"mode": metrics["mode"], "mock": metrics["mock"], "delay_profile": metrics["delay_profile"]}, episode_dir / "config.yaml")
        dump_data(task, episode_dir / "task.yaml")
        dump_data(scene, episode_dir / "scene.yaml")
        dump_data(metrics, episode_dir / "metrics.json")
        append_jsonl(episode_dir / "events.jsonl", events)
        with (episode_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["t", "x", "y", "yaw"])
            writer.writeheader()
            for row in trajectory:
                pose = row["pose"]
                writer.writerow({"t": row["t"], "x": pose[0], "y": pose[1], "yaw": pose[2]})
        with (episode_dir / "model_calls.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["model", "call_index", "request_id", "latency_ms", "multimodal", "result", "source"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(model_calls)
        with (episode_dir / "safety_events.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["t", "event", "details"])
            writer.writeheader()
            writer.writerows(safety_events)
        (episode_dir / "frames").mkdir(exist_ok=True)
        (episode_dir / "summary.md").write_text(
            f"# Episode {metrics['episode_id']}\n\n"
            f"- success: {metrics['success']}\n"
            f"- failure_reason: {metrics['failure_reason']}\n"
            f"- mission_time_sec: {metrics['mission_time_sec']}\n",
            encoding="utf-8",
        )


def _safe_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return value if isinstance(value, dict) else {"value": value}


def _topic_to_event(topic: str) -> str:
    return topic.strip("/").replace("/", "_")


def _twist_dict(msg: Any) -> dict[str, Any]:
    return {
        "linear": {"x": float(msg.linear.x), "y": float(msg.linear.y), "z": float(msg.linear.z)},
        "angular": {"x": float(msg.angular.x), "y": float(msg.angular.y), "z": float(msg.angular.z)},
    }


def _odom_pose(msg: Any) -> list[float]:
    q = msg.pose.pose.orientation
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return [float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), math.atan2(siny_cosp, cosy_cosp)]


def _write_live_episode(
    episode_dir: Path,
    task: dict[str, Any],
    scene: dict[str, Any],
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    model_calls: list[dict[str, Any]],
    safety_events: list[dict[str, Any]],
) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    dump_data(
        {
            "mode": metrics["mode"],
            "mock": metrics["mock"],
            "delay_profile": metrics["delay_profile"],
            "use_isaac": True,
            "live_closed_loop": True,
        },
        episode_dir / "config.yaml",
    )
    dump_data(task, episode_dir / "task.yaml")
    dump_data(scene, episode_dir / "scene.yaml")
    dump_data(metrics, episode_dir / "metrics.json")
    append_jsonl(episode_dir / "events.jsonl", events)
    with (episode_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "x", "y", "yaw", "source"])
        writer.writeheader()
        for row in trajectory:
            pose = row["pose"]
            writer.writerow({"t": row["t"], "x": pose[0], "y": pose[1], "yaw": pose[2], "source": row.get("source", "")})
    with (episode_dir / "model_calls.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "request_id", "latency_ms", "result", "source", "multimodal",
                "vision_latency_ms", "model_latency_ms", "network_latency_ms", "roundtrip_latency_ms",
                "model_variant", "precision_mode", "action_head_trained", "peak_memory_mib", "fallback_used",
            ],
        )
        writer.writeheader()
        writer.writerows(model_calls)
    with (episode_dir / "safety_events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "event", "details"])
        writer.writeheader()
        writer.writerows(safety_events)
    (episode_dir / "frames").mkdir(exist_ok=True)
    (episode_dir / "summary.md").write_text(
        f"# Episode {metrics['episode_id']}\n\n"
        f"- live_closed_loop: true\n"
        f"- success: {metrics['success']}\n"
        f"- failure_reason: {metrics['failure_reason']}\n"
        f"- mission_time_sec: {metrics['mission_time_sec']}\n"
        f"- final_distance_to_target_m: {metrics['final_distance_to_target_m']}\n"
        f"- isaac_pose_source: {metrics.get('isaac_pose_source')}\n",
        encoding="utf-8",
    )


class LiveRosBenchmarkRunner:
    def __init__(self, args: argparse.Namespace, benchmark_cfg: dict[str, Any]):
        os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            from std_msgs.msg import String
        except Exception as exc:  # pragma: no cover - only exercised on ROS hosts
            raise RuntimeError("ROS2 Python packages are required for --use-isaac live closed-loop mode") from exc

        self.rclpy = rclpy
        self.Twist = Twist
        self.Odometry = Odometry
        self.Image = Image
        self.String = String
        self.args = args
        self.benchmark_cfg = benchmark_cfg
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False
        self.node = Node("isaac_vln_live_benchmark_runner")
        self.reset_pub = self.node.create_publisher(String, "/isaac/reset_episode", 10)
        self.scene_pub = self.node.create_publisher(String, "/isaac/set_scene", 10)
        self.instruction_pub = self.node.create_publisher(String, "/user_instruction", 10)
        self.mode_pub = self.node.create_publisher(String, "/benchmark/mode_json", 10)

        self.events: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []
        self.model_calls: list[dict[str, Any]] = []
        self.safety_events: list[dict[str, Any]] = []
        self.latest: dict[str, Any] = {}
        self.episode_started_monotonic = 0.0
        self.active_episode_id = ""
        self.active_task_id = ""

        string_topics = [
            "/isaac/ground_truth_pose",
            "/isaac/reset_ack_json",
            "/isaac/objects",
            "/isaac/episode_status",
            "/isaac/collision_event",
            "/scheduler/state",
            "/step/request_json",
            "/step/response_json",
            "/omninav/request_json",
            "/omninav/action_candidate_json",
            "/internnav/action_json",
            "/primitive/command_json",
            "/oracle/route_choice_json",
            "/oracle/semantic_stop_json",
            "/oracle/stop_json",
            "/mission/event_json",
            "/benchmark/delay_audit_json",
            "/benchmark/reset_stress_json",
            "/stress/raw/step/response_json",
            "/stress/raw/step/route_choice_json",
            "/stress/raw/step/semantic_stop_json",
            "/metrics/event_jsonl",
            "/perception/target_observation_json",
            "/perception/target_track_json",
            "/safety/local_status_json",
            "/semantic_summary_json",
        ]
        self._subs = [self.node.create_subscription(String, topic, lambda msg, t=topic: self.on_string(t, msg), 20) for topic in string_topics]
        self._subs.append(self.node.create_subscription(Twist, "/safe_cmd_vel", self.on_safe_cmd, 20))
        self._subs.append(self.node.create_subscription(Twist, "/cmd_vel_candidate", self.on_cmd_candidate, 20))
        self._subs.append(self.node.create_subscription(Odometry, "/odom", self.on_odom, 20))
        self._subs.append(self.node.create_subscription(Image, "/camera/front/image", self.on_image, 2))

    def close(self) -> None:
        self.node.destroy_node()
        if self._owns_rclpy and self.rclpy.ok():
            self.rclpy.shutdown()

    def now(self) -> float:
        return time.monotonic()

    def episode_t(self) -> float:
        if self.episode_started_monotonic <= 0.0:
            return 0.0
        return max(0.0, self.now() - self.episode_started_monotonic)

    def record(self, event: str, details: dict[str, Any] | None = None, *, latency_ms: float | None = None) -> None:
        payload = {
            "t": round(self.episode_t(), 6),
            "episode_id": self.active_episode_id,
            "event": event,
            "state": str(self.latest.get("/scheduler/state", "")),
            "details": details or {},
        }
        if latency_ms is not None:
            payload["latency_ms"] = round(latency_ms, 3)
        self.events.append(payload)

    def on_string(self, topic: str, msg: Any) -> None:
        data = _safe_json(msg.data)
        self.latest[topic] = data
        if not self.active_episode_id:
            return
        if topic == "/isaac/episode_status" and data.get("task_id") not in {None, self.active_task_id}:
            return
        if topic == "/isaac/ground_truth_pose":
            telemetry_episode_id = str(data.get("telemetry_episode_id") or "")
            if telemetry_episode_id and telemetry_episode_id != self.active_episode_id:
                return
        if topic == "/isaac/reset_ack_json" and str(data.get("episode_id") or "") != self.active_episode_id:
            return
        self.record(_topic_to_event(topic), data)
        if topic == "/isaac/ground_truth_pose":
            pose = data.get("pose") or data.get("robot_pose")
            if isinstance(pose, list) and len(pose) >= 3:
                self.trajectory.append(
                    {
                        "t": round(self.episode_t(), 6),
                        "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
                        "source": str(data.get("source", "ground_truth_pose")),
                    }
                )
        elif topic == "/isaac/collision_event":
            self.safety_events.append({"t": round(self.episode_t(), 6), "event": "collision", "details": json.dumps(data, sort_keys=True)})
        elif topic == "/step/response_json":
            self._record_model_call(self._model_call_from_payload("step", data))
        elif topic == "/omninav/action_candidate_json":
            self._record_model_call(self._model_call_from_payload("omninav", data))
        elif topic == "/internnav/action_json":
            self._record_model_call(self._model_call_from_payload("internnav", data))
        elif topic == "/metrics/event_jsonl":
            event_type = str(data.get("event_type", ""))
            if event_type == "step_http_response" and str(data.get("result") or "").startswith("accepted"):
                self._record_model_call(self._model_call_from_payload("step", data))
            if event_type in {"safe_cmd_mux", "step_pending_policy"} and data.get("result") in {"safety_stop", "failsafe"}:
                self.safety_events.append({"t": round(self.episode_t(), 6), "event": str(data.get("result")), "details": json.dumps(data, sort_keys=True)})

    def _record_model_call(self, call: dict[str, Any]) -> None:
        request_id = str(call.get("request_id") or "")
        if request_id and any(
            str(row.get("model")) == str(call.get("model"))
            and str(row.get("request_id") or "") == request_id
            for row in self.model_calls
        ):
            return
        self.model_calls.append(call)

    @staticmethod
    def _model_call_from_payload(model: str, data: dict[str, Any]) -> dict[str, Any]:
        t_req = data.get("timestamp_request")
        t_resp = data.get("timestamp_response")
        latency_ms = 0.0
        try:
            latency_ms = max(0.0, (float(t_resp) - float(t_req)) * 1000.0)
        except (TypeError, ValueError):
            latency_ms = float(data.get("latency_ms", 0.0) or 0.0)
            if latency_ms <= 0.0:
                latency_ms = float(data.get("latency_s", data.get("step_latency_sec", 0.0)) or 0.0) * 1000.0
        return {
            "model": model,
            "request_id": str(data.get("request_id", "")),
            "latency_ms": round(latency_ms, 3),
            "result": str(data.get("result", "response")),
            "source": str(data.get("source", "")),
            "multimodal": bool(data.get("multimodal", False)),
            "vision_latency_ms": float(data.get("vision_latency_ms", 0.0) or 0.0),
            "model_latency_ms": float(data.get("model_latency_ms", 0.0) or 0.0),
            "network_latency_ms": float(data.get("network_latency_ms", 0.0) or 0.0),
            "roundtrip_latency_ms": float(data.get("roundtrip_latency_ms", latency_ms) or latency_ms),
            "model_variant": str(data.get("model_variant") or ""),
            "precision_mode": str(data.get("precision_mode") or ""),
            "action_head_trained": data.get("action_head_trained") is True,
            "peak_memory_mib": float(data.get("peak_memory_mib", 0.0) or 0.0),
            "fallback_used": bool(data.get("fallback_used", False)),
        }

    def on_safe_cmd(self, msg: Any) -> None:
        self.latest["/safe_cmd_vel"] = _twist_dict(msg)
        if self.active_episode_id:
            self.record("safe_cmd_vel", _twist_dict(msg))

    def on_cmd_candidate(self, msg: Any) -> None:
        self.latest["/cmd_vel_candidate"] = _twist_dict(msg)
        if self.active_episode_id:
            self.record("cmd_vel_candidate", _twist_dict(msg))

    def on_odom(self, msg: Any) -> None:
        pose = _odom_pose(msg)
        self.latest["/odom"] = {"pose": pose, "t": self.now()}
        if self.active_episode_id:
            self.record("odom", {"pose": pose})
            if not self.trajectory or self.trajectory[-1].get("source") != "isaaclab_go2":
                self.trajectory.append({"t": round(self.episode_t(), 6), "pose": pose, "source": "odom"})

    def on_image(self, msg: Any) -> None:
        self.latest["/camera/front/image"] = {"width": int(msg.width), "height": int(msg.height), "t": self.now()}

    def spin_for(self, seconds: float) -> None:
        deadline = self.now() + max(0.0, seconds)
        while self.now() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=min(0.05, max(0.0, deadline - self.now())))

    def wait_for_ready(self, timeout_sec: float) -> None:
        required = ["/odom", "/camera/front/image", "/isaac/ground_truth_pose", "/isaac/episode_status"]
        deadline = self.now() + timeout_sec
        while self.now() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            missing = [topic for topic in required if topic not in self.latest]
            if not missing:
                return
        missing = [topic for topic in required if topic not in self.latest]
        raise TimeoutError(f"live Isaac benchmark topics not ready after {timeout_sec:.1f}s: {missing}")

    def wait_for_reset_ack(self, episode_id: str, timeout_sec: float) -> dict[str, Any]:
        deadline = self.now() + max(0.1, timeout_sec)
        while self.now() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            ack = self.latest.get("/isaac/reset_ack_json")
            if isinstance(ack, dict) and str(ack.get("episode_id") or "") == episode_id:
                return dict(ack)
        raise TimeoutError(f"Isaac reset ack not received for episode {episode_id!r} after {timeout_sec:.1f}s")

    def run_episode(
        self,
        *,
        mode_name: str,
        mode_cfg: dict[str, Any],
        delay_profile_name: str,
        task: dict[str, Any],
        scene: dict[str, Any],
        episode_dir: Path,
        episode_index: int,
    ) -> dict[str, Any]:
        self.events = []
        self.trajectory = []
        self.model_calls = []
        self.safety_events = []
        self.active_task_id = str(task["task_id"])
        self.active_episode_id = f"{mode_name}_{task['task_id']}_{episode_index:04d}"
        self.episode_started_monotonic = self.now()
        for topic in ("/isaac/reset_ack_json", "/isaac/ground_truth_pose", "/isaac/episode_status", "/odom"):
            self.latest.pop(topic, None)
        timeout_sec = float(getattr(self.args, "live_episode_timeout_sec", 0.0) or task.get("timeout_sec", 120))
        scheduler_timeout_sec = max(timeout_sec + 15.0, timeout_sec * 1.1)
        episode_mode_cfg = dict(mode_cfg)
        episode_mode_cfg.setdefault("task_timeout_sec", timeout_sec)
        episode_mode_cfg.setdefault("mission_timeout_margin_sec", max(5.0, scheduler_timeout_sec - timeout_sec))
        episode_mode_cfg.setdefault("mission_max_duration_sec", scheduler_timeout_sec)

        mode_payload = {
            "mode": mode_name,
            "mode_config": episode_mode_cfg,
            "episode_id": self.active_episode_id,
            "task_id": task["task_id"],
            "task_timeout_sec": timeout_sec,
            "mission_max_duration_sec": scheduler_timeout_sec,
        }
        mode_msg = self.String(data=json.dumps(mode_payload, ensure_ascii=False))
        for _ in range(3):
            self.mode_pub.publish(mode_msg)
            self.spin_for(0.1)
        self.scene_pub.publish(self.String(data=json.dumps(scene, ensure_ascii=False)))
        reset_msg = self.String(
            data=json.dumps(
                {
                    "episode_id": self.active_episode_id,
                    "episode_key": str(task.get("episode_key") or task["task_id"]),
                    "mode": mode_name,
                    "task_id": task["task_id"],
                    "scene_id": scene["scene_id"],
                    "random_seed": int(task.get("random_seed", self.args.seed)),
                },
                ensure_ascii=False,
            )
        )
        self.reset_pub.publish(reset_msg)
        self.record("reset", {"task_id": task["task_id"], "scene_id": scene["scene_id"], "mode": mode_name})
        reset_ack_timeout = float(getattr(self.args, "live_reset_ack_timeout_sec", 8.0))
        try:
            ack = self.wait_for_reset_ack(self.active_episode_id, reset_ack_timeout)
        except TimeoutError:
            # UDP reset/control is intentionally idempotent; one retry covers a lost datagram.
            self.reset_pub.publish(reset_msg)
            ack = self.wait_for_reset_ack(self.active_episode_id, reset_ack_timeout)
        self.events = []
        self.trajectory = []
        self.episode_started_monotonic = self.now()
        self.record("reset_ack", ack)
        self.spin_for(float(getattr(self.args, "live_settle_sec", 1.0)))
        for _ in range(2):
            self.mode_pub.publish(mode_msg)
            self.spin_for(0.1)
        self.trajectory = []
        instruction = {"mission_id": self.active_episode_id, "instruction": task["instruction"], "timestamp": time.time()}
        self.instruction_pub.publish(self.String(data=json.dumps(instruction, ensure_ascii=False)))
        self.record("instruction", {"instruction": task["instruction"]})

        final_status: dict[str, Any] | None = None
        deadline = self.now() + timeout_sec
        while self.now() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            status = self.latest.get("/isaac/episode_status")
            if isinstance(status, dict) and status.get("task_id") == task["task_id"] and status.get("done"):
                final_status = dict(status)
                break
        elapsed = self.episode_t()
        if final_status is None:
            status = self.latest.get("/isaac/episode_status")
            final_status = dict(status) if isinstance(status, dict) else {}
            final_status.update({"success": False, "done": True, "reason": "timeout", "time_sec": round(elapsed, 3)})
            self.record("failure", {"reason": "timeout"})
        elif final_status.get("success"):
            self.record("success", {"reason": None})
        else:
            self.record("failure", {"reason": final_status.get("reason")})

        metrics = self._episode_metrics(mode_name, mode_cfg, delay_profile_name, task, scene, final_status, elapsed)
        _write_live_episode(episode_dir, task, scene, metrics, self.events, self.trajectory, self.model_calls, self.safety_events)
        self.active_episode_id = ""
        self.active_task_id = ""
        return metrics

    def _episode_metrics(
        self,
        mode_name: str,
        mode_cfg: dict[str, Any],
        delay_profile_name: str,
        task: dict[str, Any],
        scene: dict[str, Any],
        status: dict[str, Any],
        elapsed: float,
    ) -> dict[str, Any]:
        mode_cfg = dict(mode_cfg or {})
        uses_step = bool(mode_cfg.get("use_step", "step" in mode_name))
        uses_omninav = bool(mode_cfg.get("use_omninav", "omninav" in mode_name))
        uses_internnav = bool(mode_cfg.get("use_internnav", "internnav" in mode_name))
        step_calls = [row for row in self.model_calls if row["model"] == "step"] if uses_step else []
        omni_calls = [row for row in self.model_calls if row["model"] == "omninav"] if uses_omninav else []
        internnav_calls = [row for row in self.model_calls if row["model"] == "internnav"] if uses_internnav else []
        collisions = sum(1 for row in self.safety_events if row.get("event") == "collision")
        safety_stops = sum(1 for row in self.safety_events if str(row.get("event")) in {"safety_stop", "failsafe"})
        stale_step = 0
        stale_omni = 0
        runtime_stale = 0
        timebase_error = 0
        stale_action_executed = 0
        unsafe_blocked = 0
        recovery_override_count = 0
        action_counts = {"forward": 0, "left": 0, "right": 0, "stop": 0, "unknown": 0}
        for event in self.events:
            details = event.get("details") or {}
            event_type = str(details.get("event_type") or details.get("type") or event.get("event"))
            model = str(details.get("model") or "")
            if "stale" in event_type:
                if model == "omninav" and uses_omninav:
                    stale_omni += 1
                elif uses_step:
                    stale_step += 1
                attribution = str(details.get("attribution") or details.get("reason") or "unknown")
                if attribution not in {"reset_cleanup", "old_response", "old_response_after_reset", "episode_mismatch"}:
                    runtime_stale += 1
            serialized_details = json.dumps(details, ensure_ascii=False).lower()
            if "timebase_error" in event_type or "timestamp_mismatch" in serialized_details:
                timebase_error += 1
            if "stale_action_executed" in event_type:
                stale_action_executed += 1
            if details.get("result") == "safety_stop":
                unsafe_blocked += 1
            if uses_internnav and (details.get("source") == "internnav" or event.get("event") == "internnav_action"):
                if details.get("recovery_override"):
                    recovery_override_count += 1
                action = str(details.get("model_action") or "unknown")
                action_counts[action if action in action_counts else "unknown"] += 1

        eventized = eventized_safety_recovery_metrics(self.events)
        recovery_override_count = int(eventized.get("recovery_override_ticks", recovery_override_count))
        safety_stops = int(eventized.get("safety_block_ticks", safety_stops))
        gt = self.latest.get("/isaac/ground_truth_pose") if isinstance(self.latest.get("/isaac/ground_truth_pose"), dict) else {}
        pose_source = str(gt.get("source", ""))
        step_latencies = [float(row.get("latency_ms", 0.0) or 0.0) for row in step_calls]
        omni_latencies = [float(row.get("latency_ms", 0.0) or 0.0) for row in omni_calls]
        decision_latencies = [float(row.get("roundtrip_latency_ms", row.get("latency_ms", 0.0)) or 0.0) for row in omni_calls]
        vision_latencies = [float(row.get("vision_latency_ms", 0.0) or 0.0) for row in omni_calls]
        model_latencies = [float(row.get("model_latency_ms", 0.0) or 0.0) for row in omni_calls]
        network_latencies = [float(row.get("network_latency_ms", 0.0) or 0.0) for row in omni_calls]
        internnav_latencies_s = [float(row.get("latency_ms", 0.0) or 0.0) / 1000.0 for row in internnav_calls]
        success = bool(status.get("success", False))
        reason = None if success else str(status.get("reason") or "unknown")
        success_bits = classify_success(
            reached=success,
            failure_reason=reason,
            recovery_override_count=recovery_override_count,
            stale_action_count=runtime_stale,
            collision_count=collisions,
            safety_stop_count=safety_stops,
            step_correction_count=1 if success and step_calls and internnav_calls else 0,
        )
        action_total = max(sum(action_counts.values()), 1)
        entropy = -sum((count / action_total) * math.log(count / action_total, 2) for count in action_counts.values() if count)
        path_length_m = path_length(self.trajectory)
        target = object_by_id(scene, task["target_object"])
        shortest_path_m = shortest_path_length(scene, target.get("pose", [0.0, 0.0, 0.0]))
        spl = (shortest_path_m / max(path_length_m, shortest_path_m)) if success else 0.0
        obstacle_distances: list[float] = []
        camera_sources: set[str] = set()
        for event in self.events:
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            value = details.get("obstacle_distance_m")
            if value is not None:
                try:
                    obstacle_distances.append(float(value))
                except (TypeError, ValueError):
                    pass
            if str(details.get("event_type") or "") == "camera_frame_source":
                camera_sources.add(str(details.get("source") or ""))
        model_variants = {str(row.get("model_variant") or "") for row in omni_calls if row.get("model_variant")}
        precision_modes = {str(row.get("precision_mode") or "") for row in omni_calls if row.get("precision_mode")}
        model_variant = next(iter(model_variants)) if len(model_variants) == 1 else ("mixed" if model_variants else "")
        precision_mode = next(iter(precision_modes)) if len(precision_modes) == 1 else ("mixed" if precision_modes else "")
        camera_source = next(iter(camera_sources)) if len(camera_sources) == 1 else ("mixed" if camera_sources else "")
        subgoals_expected = int(status.get("subgoals_expected", 0) or len(task.get("subgoals", [])))
        subgoals_completed = int(status.get("subgoals_completed", 0) or 0)
        if success and subgoals_expected:
            subgoals_completed = subgoals_expected
        subgoal_completion_rate = subgoals_completed / max(subgoals_expected, 1)
        fallback_count = sum(bool(row.get("fallback_used")) or "fallback" in str(row.get("source") or "").lower() for row in omni_calls)
        real_image_calls = sum(
            str(row.get("source") or "") in {
                "omninav_model_client:ros_image",
                "omninav_remote_model_client:isaac_zmq",
            }
            for row in omni_calls
        )
        return {
            "episode_id": self.active_episode_id,
            "episode_key": str(task.get("episode_key") or task["task_id"]),
            "mode": mode_name,
            "pending_policy": "live",
            "task_id": task["task_id"],
            "task_type": task.get("task_type"),
            "scene_id": scene.get("scene_id"),
            "scene_type": scene.get("scene_type"),
            "mock": bool(self.args.mock_models),
            "use_isaac": True,
            "live_closed_loop": True,
            "oracle_semantics": self.benchmark_cfg.get("oracle_semantics", True),
            "delay_profile": delay_profile_name,
            **success_bits,
            "mission_time_sec": round(float(status.get("time_sec", elapsed) or elapsed), 3),
            "path_length_m": round(path_length_m, 3),
            "path_length": round(path_length_m, 3),
            "shortest_path_length": round(shortest_path_m, 3),
            "shortest_path_basis": "static_grid_astar_0.10m_robot_radius_0.30m",
            "spl": round(spl, 6),
            "final_distance_to_target_m": float(status.get("distance_to_target", 0.0) or 0.0),
            "target_visible_at_completion": bool(status.get("target_visible", False)),
            "semantic_errors": int(status.get("semantic_errors", 0) or 0),
            "semantic_subgoals_completed": subgoals_completed,
            "semantic_subgoals_expected": subgoals_expected,
            "subgoal_completion_rate": round(subgoal_completion_rate, 6),
            "semantic_recovery_required": str(status.get("required_recovery") or ""),
            "semantic_recovery_observed": str(status.get("observed_recovery") or ""),
            "semantic_recovery_matched": bool(status.get("recovery_matched", True)),
            "instrumented_semantic_markers": bool(status.get("instrumented_semantic_markers", False)),
            "geometry_judge_is_oracle_only": bool(status.get("geometry_judge_is_oracle_only", False)),
            "qualification_evidence": bool(status.get("qualification_evidence", True)),
            "num_step_calls": len(step_calls),
            "num_step_multimodal_calls": sum(bool(row.get("multimodal")) for row in step_calls),
            "num_omninav_calls": len(omni_calls),
            "num_omninav_real_image_calls": real_image_calls,
            "num_omninav_fallback_calls": fallback_count,
            "fallback_count": fallback_count,
            "num_internnav_calls": len(internnav_calls),
            "step_mean_latency_ms": round(sum(step_latencies) / len(step_latencies), 3) if step_latencies else 0.0,
            "omninav_mean_latency_ms": round(sum(omni_latencies) / len(omni_latencies), 3) if omni_latencies else 0.0,
            "internnav_mean_latency_s": round(sum(internnav_latencies_s) / len(internnav_latencies_s), 4) if internnav_latencies_s else 0.0,
            "num_stop_and_think": sum(1 for e in self.events if (e.get("details") or {}).get("pending_mode") == "stop"),
            "num_move_while_thinking": sum(1 for e in self.events if (e.get("details") or {}).get("pending_mode") == "move_slow"),
            "num_safe_scan": sum(1 for e in self.events if (e.get("details") or {}).get("pending_mode") == "safe_scan"),
            "num_stale_step_results": stale_step,
            "num_stale_omninav_actions": stale_omni,
            "runtime_stale": runtime_stale,
            "timebase_error": timebase_error,
            "stale_action_executed": stale_action_executed,
            "num_safety_stops": safety_stops,
            "num_collisions": collisions,
            "collision": bool(collisions),
            "collision_count": collisions,
            "minimum_obstacle_distance": round(min(obstacle_distances), 3) if obstacle_distances else None,
            "emergency_stop_count": safety_stops,
            "timeout": reason == "timeout",
            "stuck": reason in {"no_progress", "forward_bias"},
            "arrive_false_positive": reason in {"stopped_too_early", "wrong_target"},
            "wrong_turn_count": 1 if reason == "wrong_target" else 0,
            "early_stop_count": 1 if reason == "stopped_too_early" else 0,
            "late_stop_count": 1 if reason == "never_stopped" else 0,
            "average_cmd_age_ms": 0.0,
            "stale_action_rate": (stale_step + stale_omni) / max(len(step_calls) + len(omni_calls), 1),
            "safety_stop_rate": 1.0 if safety_stops else 0.0,
            "collision_rate": 1.0 if collisions else 0.0,
            "unsafe_action_blocked_count": unsafe_blocked,
            "recovery_override_count": recovery_override_count,
            "recovery_override_rate": recovery_override_count / max(len(internnav_calls), 1),
            **eventized,
            "forward_ratio": action_counts["forward"] / action_total,
            "left_ratio": action_counts["left"] / action_total,
            "right_ratio": action_counts["right"] / action_total,
            "stop_ratio": action_counts["stop"] / action_total,
            "action_entropy": entropy,
            "decision_latency_ms": round(sum(decision_latencies) / len(decision_latencies), 3) if decision_latencies else 0.0,
            "vision_latency_ms": round(sum(vision_latencies) / len(vision_latencies), 3) if vision_latencies else 0.0,
            "model_latency_ms": round(sum(model_latencies) / len(model_latencies), 3) if model_latencies else 0.0,
            "network_latency_ms": round(sum(network_latencies) / len(network_latencies), 3) if network_latencies else 0.0,
            "control_frequency_hz": round(sum(e.get("event") == "safe_cmd_vel" for e in self.events) / max(elapsed, 1.0e-6), 3),
            "peak_memory": round(max((float(row.get("peak_memory_mib", 0.0) or 0.0) for row in omni_calls), default=0.0), 3),
            "random_seed": int(task.get("random_seed", self.args.seed)),
            "model_variant": model_variant,
            "precision_mode": precision_mode,
            "action_head_trained": bool(omni_calls) and all(row.get("action_head_trained") is True for row in omni_calls),
            "camera_source": camera_source,
            "isaac_headless": bool(getattr(self.args, "isaac_headless_evidence", False)),
            "isaac_pose_source": pose_source,
            "telemetry_seq": gt.get("telemetry_seq"),
            "telemetry_age_sec": gt.get("telemetry_age_sec"),
        }


def paired_mode_work_items(
    tasks: list[dict[str, Any]],
    modes: list[str],
    *,
    randomize_by_task: bool,
    seed: int,
) -> list[tuple[int, str, dict[str, Any]]]:
    if not randomize_by_task:
        return [(index, mode, task) for mode in modes for index, task in enumerate(tasks)]
    rng = random.Random(seed)
    items: list[tuple[int, str, dict[str, Any]]] = []
    for index, task in enumerate(tasks):
        task_modes = list(modes)
        rng.shuffle(task_modes)
        items.extend((index, mode, task) for mode in task_modes)
    return items


def mode_schedule_options(config: dict[str, Any], default_seed: int) -> tuple[bool, int]:
    nested = config.get("benchmark") if isinstance(config.get("benchmark"), dict) else {}
    randomize = bool(nested.get("randomize_mode_order_by_task", config.get("randomize_mode_order_by_task", False)))
    seed = int(nested.get("mode_order_seed", config.get("mode_order_seed", default_seed)))
    return randomize, seed


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.project_root or DEFAULT_ROOT).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)

    tasks_doc = load_data(args.tasks or root / "configs" / "tasks.yaml")
    scenes_doc = load_data(args.scenes or root / "configs" / "scenes.yaml")
    modes_doc = load_data(args.ablation_modes or root / "configs" / "ablation_modes.yaml")
    delay_doc = load_data(args.delay_profiles or root / "configs" / "delay_stress.yaml")
    benchmark_cfg = load_data(args.config or root / "configs" / "benchmark_default.yaml")
    benchmark_cfg["ros_domain_id"] = args.ros_domain_id
    benchmark_cfg["use_isaac"] = args.use_isaac
    benchmark_cfg["mock_models"] = args.mock_models
    benchmark_cfg["selected_modes"] = args.modes
    benchmark_cfg["num_episodes"] = args.num_episodes
    benchmark_cfg["live_closed_loop"] = bool(args.use_isaac)

    tasks = tasks_doc.get("tasks", [])[: args.num_episodes]
    selected_modes = args.modes or list(modes_doc.get("modes", {}).keys())
    randomize_mode_order, mode_order_seed = mode_schedule_options(benchmark_cfg, args.seed)
    work_items = paired_mode_work_items(
        tasks,
        selected_modes,
        randomize_by_task=randomize_mode_order,
        seed=mode_order_seed,
    )
    benchmark_cfg["effective_episode_schedule"] = [
        {"task_id": task.get("task_id"), "mode": mode_name, "task_index": index}
        for index, mode_name, task in work_items
    ]
    if args.delay_profile == "all":
        delay_profile_names = list(delay_doc.get("delay_profiles", {}).keys()) or ["none"]
    else:
        delay_profile_names = [args.delay_profile]
    benchmark_cfg["selected_delay_profiles"] = delay_profile_names
    all_metrics: list[dict[str, Any]] = []
    aggregate_events_path = output / "events.jsonl"
    if aggregate_events_path.exists():
        aggregate_events_path.unlink()

    if args.use_isaac:
        live = LiveRosBenchmarkRunner(args, benchmark_cfg)
        try:
            live.wait_for_ready(float(args.live_startup_timeout_sec))
            for delay_profile_name in delay_profile_names:
                for idx, mode_name, task in work_items:
                    if mode_name not in modes_doc.get("modes", {}):
                        raise KeyError(f"Unknown mode {mode_name!r}")
                    scene = scene_by_id(scenes_doc, task["scene_id"]) if task.get("scene_id") else scene_by_type(scenes_doc, task["scene_type"])
                    episode_dir = (
                        output / delay_profile_name / mode_name / task["task_id"]
                        if len(delay_profile_names) > 1
                        else output / mode_name / task["task_id"]
                    )
                    metrics = live.run_episode(
                        mode_name=mode_name,
                        mode_cfg=modes_doc["modes"][mode_name],
                        delay_profile_name=delay_profile_name,
                        task=task,
                        scene=scene,
                        episode_dir=episode_dir,
                        episode_index=idx,
                    )
                    all_metrics.append(metrics)
                    events_path = episode_dir / "events.jsonl"
                    if events_path.exists():
                        for line in events_path.read_text(encoding="utf-8").splitlines():
                            record = json.loads(line)
                            record["episode_id"] = metrics["episode_id"]
                            record["mode"] = mode_name
                            record["delay_profile"] = delay_profile_name
                            append_jsonl(aggregate_events_path, [record])
        finally:
            live.close()
    else:
        for delay_profile_name in delay_profile_names:
            delay_cfg = delay_doc.get("delay_profiles", {}).get(
                delay_profile_name,
                delay_doc.get("delay_profiles", {}).get("none", {}),
            )
            for mode_name in selected_modes:
                if mode_name not in modes_doc.get("modes", {}):
                    raise KeyError(f"Unknown mode {mode_name!r}")
                simulator = MockEpisodeSimulator(
                    mode_name,
                    modes_doc["modes"][mode_name],
                    benchmark_cfg,
                    delay_profile_name,
                    delay_cfg,
                    seed=args.seed,
                    mock_models=args.mock_models,
                    use_isaac=False,
                )
                for idx, task in enumerate(tasks):
                    scene = scene_by_id(scenes_doc, task["scene_id"]) if task.get("scene_id") else scene_by_type(scenes_doc, task["scene_type"])
                    episode_dir = (
                        output / delay_profile_name / mode_name / task["task_id"]
                        if len(delay_profile_names) > 1
                        else output / mode_name / task["task_id"]
                    )
                    metrics = simulator.run_episode(task, scene, episode_dir, idx)
                    all_metrics.append(metrics)
                    events_path = episode_dir / "events.jsonl"
                    if events_path.exists():
                        for line in events_path.read_text(encoding="utf-8").splitlines():
                            record = json.loads(line)
                            record["episode_id"] = metrics["episode_id"]
                            record["mode"] = mode_name
                            record["delay_profile"] = delay_profile_name
                            append_jsonl(aggregate_events_path, [record])

    by_mode = {}
    for mode_name in selected_modes:
        rows = [m for m in all_metrics if m["mode"] == mode_name]
        by_mode[mode_name] = aggregate_metrics(rows)
    run_metrics = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_dir": str(output),
        "aggregate_by_mode": by_mode,
        "episodes": all_metrics,
    }
    dump_data(run_metrics, output / "metrics.json")
    episodes_jsonl = output / "episodes.jsonl"
    if episodes_jsonl.exists():
        episodes_jsonl.unlink()
    append_jsonl(episodes_jsonl, all_metrics)
    dump_data(benchmark_cfg, output / "config.yaml")
    write_summary(output, all_metrics, benchmark_cfg)
    try:
        from .diagnostics import write_diagnostic_artifacts

        write_diagnostic_artifacts(output, benchmark_cfg)
    except Exception as exc:
        (output / "diagnostics_error.txt").write_text(repr(exc), encoding="utf-8")
    return run_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Isaac VLN benchmark episodes.")
    parser.add_argument("--modes", nargs="+", default=["step_only", "omninav_only", "step_omninav_event"])
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--ablation-modes", default=None)
    parser.add_argument("--delay-profiles", default=None)
    parser.add_argument("--delay-profile", default="none")
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--output", default="runs/mock_benchmark")
    parser.add_argument("--use-isaac", action="store_true")
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--ros-domain-id", type=int, default=int(os.environ.get("ROS_DOMAIN_ID", "0")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--live-startup-timeout-sec", type=float, default=20.0)
    parser.add_argument("--live-settle-sec", type=float, default=1.0)
    parser.add_argument("--live-reset-ack-timeout-sec", type=float, default=8.0)
    parser.add_argument("--live-episode-timeout-sec", type=float, default=0.0)
    parser.add_argument(
        "--isaac-headless-evidence",
        action="store_true",
        help="Set only after a strict process preflight proves the live Isaac process is headless.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metrics = run_benchmark(args)
    print(json.dumps({"output": metrics["output_dir"], "episodes": len(metrics["episodes"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
