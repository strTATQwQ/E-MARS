#!/usr/bin/env python3
"""Evaluator facades that trigger physics windows instead of flash actions."""

from __future__ import annotations

import json
import os
import hashlib
import math
from pathlib import Path
from typing import Any

from internvla_go2_controller.runtime import (
    reset_execution_identity,
    reset_obstacle_scenario,
    set_execution_identity,
    set_obstacle_scenario,
)
from internvla_ipc_agent_client import ROS2IPCAgentClient
from internvla_nav2_oracle_agent_client import Nav2OracleAgentClient
from t5_rgb_frame_recorder import T5RGBFrameRecorder


CONTINUOUS_TRIGGER = [{"action": [[0.0, 0.0, 0.0]], "ideal_flag": False}]
SAFE_STOP = [{"action": [0], "ideal_flag": True}]


def _continuous_evaluator_action(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate only a genuine model/Oracle STOP into evaluator termination."""

    terminal_stop = bool(response.get("stop", False)) and not bool(
        response.get("motion_observation_gate_only", False)
    )
    return SAFE_STOP if terminal_stop else CONTINUOUS_TRIGGER


def _continuous_execution_stop(response: dict[str, Any]) -> bool:
    """Preserve a control-plane safe-stop latch without terminating the task."""

    return bool(response.get("stop", False))


def _reference_route_yaw_world(episode: dict[str, Any]) -> float:
    """Return the start-to-1.5 m route chord in Isaac/map coordinates."""
    raw_path = episode.get("reference_path", [])
    if not isinstance(raw_path, list) or len(raw_path) < 2:
        raise RuntimeError("obstacle Oracle reference path is too short")
    points = [(float(point[0]), -float(point[2])) for point in raw_path]
    start_x, start_y = points[0]
    target_x, target_y = points[-1]
    for point_x, point_y in points[1:]:
        target_x, target_y = point_x, point_y
        if math.hypot(point_x - start_x, point_y - start_y) >= 1.5:
            break
    if math.hypot(target_x - start_x, target_y - start_y) < 0.25:
        raise RuntimeError("obstacle Oracle reference route chord is degenerate")
    return math.atan2(target_y - start_y, target_x - start_x)


def _scenario_manifest() -> tuple[dict[str, str], dict[str, str]]:
    manifest_path = os.environ.get("INTERNVLA_T3_SCENARIO_MANIFEST", "")
    if not manifest_path:
        return {}, {}
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    by_trajectory = {
        str(item["trajectory_id"]): str(item["scenario"])
        for item in manifest["scenarios"]
        if "trajectory_id" in item
    }
    by_instruction_digest = {
        str(item.get("instruction_runtime_sha256", item["instruction_sha256"])): str(
            item["scenario"]
        )
        for item in manifest["scenarios"]
        if "instruction_sha256" in item
    }
    return by_trajectory, by_instruction_digest


class ContinuousROS2IPCAgentClient(ROS2IPCAgentClient):
    def __init__(self, config: Any):
        super().__init__(config)
        self.full_rgb_recorder = T5RGBFrameRecorder.from_environment()
        self.capture_episode_id = str(self.handshake_response["episode_id"])
        self.capture_reset_generation = int(
            self.handshake_response["reset_generation"]
        )
        _, self.scenario_by_instruction_digest = _scenario_manifest()
        episode_id = str(self.handshake_response["episode_id"])
        reset_generation = int(self.handshake_response["reset_generation"])
        reset_execution_identity(episode_id, reset_generation)
        reset_obstacle_scenario(reset_generation)

    def step(self, obs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.full_rgb_recorder is not None:
            self.full_rgb_recorder.record_without_affecting_control(
                obs[0],
                episode_id=self.capture_episode_id,
                reset_generation=self.capture_reset_generation,
                # The model request has not been issued yet.  Do not guess its
                # execution sequence; the x86 source sequence and sim stamp
                # are the authoritative join keys for this pre-request frame.
                sequence_id=None,
            )
        result = super().step(obs)
        response = self.last_result
        if response is None or result == SAFE_STOP:
            if (
                getattr(self, "last_step_safe_stop_kind", None)
                == "unexpected_ipc_error"
            ):
                # The base client has already closed its IPC channel and the
                # ROS side has asserted safe-stop.  Do not translate an
                # unexpected transport/model failure into the evaluator's
                # official STOP action: that can be scored as success merely
                # because the robot happens to be inside the success radius.
                raise RuntimeError(
                    f"unexpected InternVLA IPC step failure: {self.last_error}"
                )
            return SAFE_STOP
        execution_stop = _continuous_execution_stop(response)
        evaluator_action = _continuous_evaluator_action(response)
        if self.scenario_by_instruction_digest:
            digest = hashlib.sha256(str(obs[0]["instruction"]).encode("utf-8")).hexdigest()
            scenario = self.scenario_by_instruction_digest.get(digest)
            if scenario is None:
                raise RuntimeError("obstacle scenario manifest does not cover instruction")
            set_obstacle_scenario(scenario, int(response["reset_generation"]))
        set_execution_identity(
            str(response["episode_id"]),
            int(response["reset_generation"]),
            int(response["sequence_id"]),
            stop=execution_stop,
        )
        # A gate-only safe-stop remains a CONTINUOUS_TRIGGER so Isaac can
        # advance to a fresh post-cancel observation.
        return evaluator_action

    def reset(self, reset_index: Any = None) -> None:
        response = super().reset(reset_index)
        self.capture_episode_id = str(response["episode_id"])
        self.capture_reset_generation = int(response["reset_generation"])
        reset_execution_identity(
            str(response["episode_id"]), int(response["reset_generation"])
        )
        reset_obstacle_scenario(int(response["reset_generation"]))


class ContinuousNav2OracleAgentClient(Nav2OracleAgentClient):
    def __init__(self, config: Any):
        super().__init__(config)
        self.scenario_by_trajectory, _ = _scenario_manifest()
        reset_execution_identity("oracle-episode-0", 0)
        reset_obstacle_scenario(0)

    def step(self, obs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = super().step(obs)
        response = self.last_result
        if response is None:
            return SAFE_STOP
        execution_stop = _continuous_execution_stop(response)
        generation = int(response["reset_generation"])
        if self.scenario_by_trajectory:
            episode = self.by_instruction[str(obs[0]["instruction"])]
            trajectory_id = str(episode.get("trajectory_id", ""))
            scenario = self.scenario_by_trajectory.get(trajectory_id)
            if scenario is None:
                raise RuntimeError("obstacle scenario manifest does not cover episode")
            set_obstacle_scenario(
                scenario,
                generation,
                route_yaw_world=_reference_route_yaw_world(episode),
            )
        set_execution_identity(
            str(response["episode_id"]),
            generation,
            int(response["sequence_id"]),
            stop=execution_stop,
        )
        return _continuous_evaluator_action(response)

    def reset(self, reset_index: Any = None) -> None:
        response = super().reset(reset_index)
        reset_execution_identity(
            str(response["episode_id"]), int(response["reset_generation"])
        )
        reset_obstacle_scenario(int(response["reset_generation"]))
