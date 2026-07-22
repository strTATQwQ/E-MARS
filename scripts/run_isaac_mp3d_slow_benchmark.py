#!/usr/bin/env python3
"""Run one scene of the frozen paired SlowPlanner benchmark in Isaac Sim."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True)
parser.add_argument("--manifest", required=True)
parser.add_argument("--scene", required=True)
parser.add_argument("--scene-usd", required=True)
parser.add_argument("--connectivity", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--run-id", required=True)
parser.add_argument("--expected-slow-model-variant", required=True)
parser.add_argument("--expected-slow-precision-mode", required=True)
parser.add_argument("--expected-slow-config-sha256", required=True)
parser.add_argument("--max-episodes", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import io
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from pxr import Gf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

from omninav_cosmos.contracts import NavigationRequest
from omninav_cosmos.transports.zmq_client import ZmqNavigationClient
from slow_benchmark.oracle_graph import MatterportGraph, habitat_pose_to_isaac, habitat_position_to_isaac
from slow_benchmark.control import heading_head_step_vector
from slow_benchmark.video import write_episode_video
from slow_planner.base import CandidateFrontier, OrderedImage, SlowPlannerRequest
from slow_planner.client import SlowPlannerClient


CAMERA_HEIGHT_M = 1.40


def load_config(path: str) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark config must be an object")
    return value


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("manifest rows must be objects")
                rows.append(value)
    return rows


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rgb_uint8(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = array * (255.0 if float(np.nanmax(array)) <= 1.01 else 1.0)
    return np.clip(array, 0, 255).astype(np.uint8)


def encode_jpeg(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=90, subsampling=0)
    return buffer.getvalue()


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class IsaacMp3DScene:
    def __init__(self, scene_usd: str, config: dict[str, Any], device: str) -> None:
        self.config = config
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=1.0 / float(config["controller"]["control_hz"]), device=device)
        )
        self.simulated_seconds = 0.0
        usd_cfg = sim_utils.UsdFileCfg(usd_path=str(Path(scene_usd).resolve()))
        prim = usd_cfg.func("/World/MP3D", usd_cfg)
        if not prim.IsValid():
            raise RuntimeError("failed to spawn MP3D scene")
        light_cfg = sim_utils.DistantLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/BenchmarkLight", light_cfg)
        sim_utils.create_prim("/World/Benchmark", "Xform")
        horizontal_aperture = 48.0
        horizontal_fov_rad = math.radians(float(config["camera"]["horizontal_fov_deg"]))
        if not 0.0 < horizontal_fov_rad < math.pi:
            raise ValueError("camera horizontal_fov_deg must be in (0, 180)")
        focal_length = horizontal_aperture / (2.0 * math.tan(horizontal_fov_rad / 2.0))
        camera_cfg = CameraCfg(
            prim_path="/World/Benchmark/Camera",
            update_period=0.0,
            height=int(config["camera"]["height"]),
            width=int(config["camera"]["width"]),
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                horizontal_aperture=horizontal_aperture,
                focus_distance=400.0,
                clipping_range=(0.05, 100.0),
            ),
        )
        self.camera = Camera(camera_cfg)
        capsule_cfg = sim_utils.CapsuleCfg(
            radius=float(config["controller"]["capsule_radius_m"]),
            height=float(config["controller"]["capsule_height_m"]),
            axis="Z",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.8), opacity=0.15),
        )
        self.capsule_prim = capsule_cfg.func("/World/Benchmark/RobotCapsule", capsule_cfg)
        self.sim.reset()
        self._validate_stage()

    def _validate_stage(self) -> None:
        stage = self.sim.stage
        if str(UsdGeom.GetStageUpAxis(stage)).upper() != "Z":
            raise RuntimeError("benchmark stage must be Z-up")
        if not math.isclose(float(UsdGeom.GetStageMetersPerUnit(stage)), 1.0, abs_tol=1.0e-9):
            raise RuntimeError("benchmark stage must use metersPerUnit=1")
        meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise RuntimeError("benchmark stage has no meshes")
        if not any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in meshes):
            raise RuntimeError("benchmark stage has no collision-enabled mesh")

    def set_capsule(self, camera_position: tuple[float, float, float]) -> None:
        capsule_height = float(self.config["controller"]["capsule_height_m"])
        floor_z = camera_position[2] - CAMERA_HEIGHT_M
        center = Gf.Vec3d(camera_position[0], camera_position[1], floor_z + capsule_height / 2.0)
        UsdGeom.XformCommonAPI(self.capsule_prim).SetTranslate(center)

    def render(self, position: tuple[float, float, float], yaw: float) -> tuple[bytes, np.ndarray]:
        self.set_capsule(position)
        target = (position[0] + math.cos(yaw), position[1] + math.sin(yaw), position[2])
        self.camera.set_world_poses_from_view(
            torch.tensor([position], dtype=torch.float32, device=self.sim.device),
            torch.tensor([target], dtype=torch.float32, device=self.sim.device),
        )
        for _ in range(2):
            self.sim.step()
            self.simulated_seconds += float(self.sim.get_physics_dt())
            self.camera.update(dt=self.sim.get_physics_dt())
        rgb = rgb_uint8(self.camera.data.output["rgb"])
        depth_value = self.camera.data.output["distance_to_image_plane"]
        depth = depth_value.detach().cpu().numpy() if hasattr(depth_value, "detach") else np.asarray(depth_value)
        if depth.ndim == 4:
            depth = depth[0, ..., 0]
        elif depth.ndim == 3:
            depth = depth[0]
        return encode_jpeg(rgb), depth

    @staticmethod
    def depth_guard(depth: np.ndarray, threshold_m: float) -> tuple[bool, float | None]:
        height, width = depth.shape
        region = depth[int(height * 0.42) : int(height * 0.88), int(width * 0.42) : int(width * 0.58)]
        finite = region[np.isfinite(region)]
        if not finite.size:
            return False, None
        percentile = float(np.percentile(finite, 5.0))
        return percentile < threshold_m, percentile


class EpisodeRunner:
    def __init__(
        self,
        scene: IsaacMp3DScene,
        graph: MatterportGraph,
        config: dict[str, Any],
        slow: SlowPlannerClient,
        fast: ZmqNavigationClient,
        output: Path,
        run_id: str,
    ) -> None:
        self.scene = scene
        self.graph = graph
        self.config = config
        self.slow = slow
        self.fast = fast
        self.output = output
        self.run_id = run_id
        self.episodes_path = output / "episodes.jsonl"
        self.latency_path = output / "latency.jsonl"
        self.video_config = config.get("evidence", {}).get("video", {})

    def _panorama(self, position, yaw, episode_id, snapshot_id):
        images = []
        for offset_deg in self.config["slow_planner"]["scan_headings_deg"]:
            image_yaw = normalize_angle(yaw + math.radians(float(offset_deg)))
            jpeg, _ = self.scene.render(position, image_yaw)
            images.append(
                OrderedImage(
                    view_id=f"heading_{int(offset_deg):03d}",
                    pose=(position[0], position[1], position[2], image_yaw),
                    jpeg=jpeg,
                    width=int(self.config["camera"]["width"]),
                    height=int(self.config["camera"]["height"]),
                )
            )
        return tuple(images)

    def _goal_positive(self, node_id: int, goal_node: int, radius: float) -> bool:
        distance = self.graph.shortest_distance(node_id, goal_node)
        line_of_sight = node_id == goal_node or goal_node in self.graph.nodes[node_id].neighbors
        return distance <= radius and line_of_sight

    def run(self, episode: dict[str, Any]) -> dict[str, Any]:
        episode_id = str(episode["benchmark_episode_id"])
        fast_state_episode_id = f"{self.run_id}::{episode_id}"
        _, yaw = habitat_pose_to_isaac(episode["start_position"], episode["start_rotation"])
        current_node = self.graph.nearest_node(episode["start_position"])
        goal = episode["goals"][0]
        goal_node = self.graph.nearest_node(goal["position"])
        shortest_distance = self.graph.shortest_distance(current_node, goal_node)
        current_position = self.graph.nodes[current_node].camera_position
        visited = [current_node]
        history: list[str] = []
        path_length = 0.0
        fast_steps = 0
        slow_decisions = 0
        collisions = 0
        fast_fallbacks = 0
        slow_fallbacks = 0
        target_tp = target_fp = target_fn = 0
        frontier_decisions = frontier_valid_decisions = frontier_geodesic_improved = 0
        repeated_frontier_decisions = 0
        frontier_geodesic_delta_sum = 0.0
        hold_seconds = 0.0
        failure_reason = "max_slow_decisions"
        success = False
        started = time.perf_counter()
        simulation_started = self.scene.simulated_seconds
        video_frames: list[bytes] = []
        self.fast.reset_episode(fast_state_episode_id)
        max_slow = int(self.config["episode"]["max_slow_decisions"])
        max_fast = int(self.config["episode"]["max_fast_steps"])
        wall_timeout = float(self.config["episode"]["timeout_wall_seconds"])
        simulation_timeout = float(self.config["episode"]["timeout_sim_seconds"])

        def timeout_reason() -> str:
            if time.perf_counter() - started >= wall_timeout:
                return "wall_timeout"
            if self.scene.simulated_seconds - simulation_started >= simulation_timeout:
                return "simulation_timeout"
            return ""

        for decision_index in range(max_slow):
            expired = timeout_reason()
            if expired:
                failure_reason = expired
                break
            slow_decisions += 1
            snapshot_id = f"{episode_id}:{decision_index}:{current_node}"
            geometries = self.graph.candidate_geometry(current_node, yaw)
            candidates = tuple(
                CandidateFrontier(
                    frontier_id=node_id,
                    relative_xz=(relative[0], relative[1]),
                    distance_m=distance,
                    bearing_deg=bearing,
                )
                for node_id, relative, distance, bearing, _ in geometries
            )
            images = self._panorama(current_position, yaw, episode_id, snapshot_id)
            if self.video_config.get("enabled") and images:
                video_frames.append(images[0].jpeg)
            request = SlowPlannerRequest(
                episode_id=episode_id,
                snapshot_id=snapshot_id,
                instruction=str(episode["instruction"]["instruction_text"]),
                ordered_images=images,
                candidate_frontiers=candidates,
                agent_pose=(current_position[0], current_position[1], current_position[2], yaw),
                visited_frontiers=tuple(visited),
                compact_history=tuple(history[-8:]),
            )
            slow_started = time.perf_counter()
            decision, metrics = self.slow.decide(request)
            slow_wall = time.perf_counter() - slow_started
            hold_seconds += slow_wall
            expired_after_slow = timeout_reason()
            positive = self._goal_positive(current_node, goal_node, float(goal["radius"]))
            if not expired_after_slow and decision.decision == "target_found":
                if positive:
                    target_tp += 1
                    success = True
                    failure_reason = ""
                else:
                    target_fp += 1
                    failure_reason = "false_target_found"
            elif not expired_after_slow and positive:
                target_fn += 1
            if decision.fallback_used:
                slow_fallbacks += 1
            current_goal_distance = self.graph.shortest_distance(current_node, goal_node)
            selected_frontier_valid = (
                decision.decision == "select_frontier"
                and decision.frontier_id is not None
                and decision.frontier_id in self.graph.nodes[current_node].neighbors
            )
            selected_goal_distance = (
                self.graph.shortest_distance(decision.frontier_id, goal_node) if selected_frontier_valid else None
            )
            geodesic_delta = (
                current_goal_distance - selected_goal_distance if selected_goal_distance is not None else None
            )
            repeated_frontier = bool(selected_frontier_valid and decision.frontier_id in visited)
            if decision.decision == "select_frontier":
                frontier_decisions += 1
                frontier_valid_decisions += int(selected_frontier_valid)
                repeated_frontier_decisions += int(repeated_frontier)
                if geodesic_delta is not None:
                    frontier_geodesic_delta_sum += geodesic_delta
                    frontier_geodesic_improved += int(geodesic_delta > 0.0)
            append_jsonl(
                self.latency_path,
                {
                    "run_id": self.run_id,
                    "episode_id": episode_id,
                    "scene_id": self.graph.scene_id,
                    "kind": "slow",
                    "index": decision_index,
                    "node_id": current_node,
                    "ground_truth_positive": positive,
                    "decision": decision.to_mapping(),
                    "evaluation": {
                        "frontier_valid": selected_frontier_valid,
                        "repeated_frontier": repeated_frontier,
                        "goal_geodesic_before_m": current_goal_distance,
                        "selected_frontier_goal_geodesic_m": selected_goal_distance,
                        "goal_geodesic_delta_m": geodesic_delta,
                    },
                    "metrics": metrics.to_mapping(),
                    "client_wall_ms": slow_wall * 1000.0,
                },
            )
            if expired_after_slow:
                failure_reason = expired_after_slow
                break
            if decision.decision == "target_found":
                break
            if decision.decision != "select_frontier" or decision.frontier_id is None:
                failure_reason = "slow_abstain"
                break
            target_node = decision.frontier_id
            if target_node not in self.graph.nodes[current_node].neighbors:
                failure_reason = "invalid_frontier_after_schema_gate"
                break

            edge_start = self.graph.nodes[current_node].camera_position
            edge_end = self.graph.nodes[target_node].camera_position
            edge = tuple(edge_end[index] - edge_start[index] for index in range(3))
            edge_length = math.sqrt(sum(value * value for value in edge))
            direction = tuple(value / edge_length for value in edge)
            subgoal_origin_yaw = yaw
            progress = 0.0
            edge_failed = False
            while progress + 1.0e-6 < edge_length:
                expired = timeout_reason()
                if expired:
                    failure_reason = expired
                    edge_failed = True
                    break
                if fast_steps >= max_fast:
                    failure_reason = "max_fast_steps"
                    edge_failed = True
                    break
                fast_steps += 1
                remaining = edge_length - progress
                front, front_depth = self.scene.render(current_position, yaw)
                left, left_depth = self.scene.render(current_position, normalize_angle(yaw + math.pi / 2.0))
                right, right_depth = self.scene.render(current_position, normalize_angle(yaw - math.pi / 2.0))
                video_stride = max(1, int(self.video_config.get("frame_stride_fast_steps", 10)))
                if self.video_config.get("enabled") and (fast_steps - 1) % video_stride == 0:
                    video_frames.append(front)
                dx = edge_end[0] - current_position[0]
                dy = edge_end[1] - current_position[1]
                forward = math.cos(yaw) * dx + math.sin(yaw) * dy
                left_target = -math.sin(yaw) * dx + math.cos(yaw) * dy
                fast_instruction = (
                    str(episode["instruction"]["instruction_text"])
                    + f"\nCurrent selected local waypoint: [{forward:.3f}, {left_target:.3f}] meters."
                )
                nav_request = NavigationRequest(
                    episode_id=fast_state_episode_id,
                    frame_id=fast_steps - 1,
                    timestamp=time.time(),
                    instruction=fast_instruction,
                    rgb_front=front,
                    rgb_left=left,
                    rgb_right=right,
                    agent_pose=(current_position[0], current_position[1], current_position[2], yaw),
                    last_action={
                        "subgoal_id": snapshot_id,
                        "subgoal_origin_pose": [
                            edge_start[0],
                            edge_start[1],
                            edge_start[2],
                            subgoal_origin_yaw,
                        ],
                        "selected_target_world": [edge_end[0], edge_end[1], edge_end[2]],
                    },
                    reset_episode=fast_steps == 1,
                )
                fast_started = time.perf_counter()
                nav_output = self.fast.infer(nav_request)
                client_ms = (time.perf_counter() - fast_started) * 1000.0
                max_progress = float(self.config["controller"]["max_linear_speed_mps"]) / float(
                    self.config["fast_policy"]["target_hz"]
                )
                fallback = bool(nav_output.safe_stop_reason or not nav_output.waypoints)
                model_projection = None
                model_heading_rad = None
                model_step_m = None
                waypoint_clamped = False
                if not fallback:
                    waypoint = nav_output.waypoints[0]
                    model_forward, model_left, model_heading_rad = heading_head_step_vector(
                        waypoint, nav_output.heading_sin_cos[0]
                    )
                    model_step_m = math.hypot(model_forward, model_left)
                    max_waypoint_m = float(self.config["fast_policy"]["max_waypoint_m"])
                    if model_step_m > max_waypoint_m:
                        scale = max_waypoint_m / model_step_m
                        model_forward *= scale
                        model_left *= scale
                        model_step_m = max_waypoint_m
                        waypoint_clamped = True
                    local_norm = max(math.hypot(forward, left_target), 1.0e-9)
                    model_projection = model_forward * forward / local_norm + model_left * left_target / local_norm
                    fallback = not math.isfinite(model_projection) or model_projection <= 0.005
                if fallback:
                    fast_fallbacks += 1
                    step_distance = min(remaining, max_progress)
                else:
                    step_distance = min(remaining, max_progress, float(model_projection))
                edge_yaw = math.atan2(direction[1], direction[0])
                edge_relative = normalize_angle(edge_yaw - yaw)
                guard_view = "back_unobserved"
                guard_depth = None
                if abs(edge_relative) <= math.pi / 4.0:
                    guard_view, guard_depth = "front", front_depth
                elif math.pi / 4.0 < edge_relative <= 3.0 * math.pi / 4.0:
                    guard_view, guard_depth = "left", left_depth
                elif -3.0 * math.pi / 4.0 <= edge_relative < -math.pi / 4.0:
                    guard_view, guard_depth = "right", right_depth
                if guard_depth is None:
                    breached, depth_p05 = False, None
                    step_distance = 0.0
                else:
                    guard_threshold = float(self.config["controller"]["capsule_radius_m"]) + 0.10
                    breached, depth_p05 = self.scene.depth_guard(guard_depth, guard_threshold)
                if breached:
                    collisions += 1
                    failure_reason = "depth_guard_breach"
                    edge_failed = True
                    step_distance = 0.0
                desired_turn = (
                    normalize_angle(edge_yaw - yaw)
                    if fallback or model_heading_rad is None
                    else model_heading_rad
                )
                max_turn = float(self.config["controller"]["max_yaw_rate_rps"]) / float(
                    self.config["fast_policy"]["target_hz"]
                )
                executed_turn = max(-max_turn, min(max_turn, desired_turn))
                yaw = normalize_angle(yaw + executed_turn)
                progress += step_distance
                path_length += step_distance
                current_position = tuple(edge_start[index] + direction[index] * progress for index in range(3))
                append_jsonl(
                    self.latency_path,
                    {
                        "run_id": self.run_id,
                        "episode_id": episode_id,
                        "scene_id": self.graph.scene_id,
                        "kind": "fast",
                        "index": fast_steps - 1,
                        "client_wall_ms": client_ms,
                        "model_latency_ms": nav_output.model_latency_ms,
                        "vision_latency_ms": nav_output.vision_latency_ms,
                        "server_total_latency_ms": nav_output.server_total_latency_ms,
                        "peak_memory_mib": nav_output.peak_memory_mib,
                        "model_variant": nav_output.model_variant,
                        "precision_mode": nav_output.precision_mode,
                        "action_head_trained": nav_output.action_head_trained,
                        "cache_hit": nav_output.cache_hit,
                        "safe_stop_reason": nav_output.safe_stop_reason,
                        "fallback": fallback,
                        "model_projection_m": model_projection,
                        "model_heading_rad": model_heading_rad,
                        "model_step_m": model_step_m,
                        "waypoint_clamped": waypoint_clamped,
                        "executed_step_m": step_distance,
                        "executed_turn_rad": executed_turn,
                        "depth_guard_breach": breached,
                        "depth_guard_view": guard_view,
                        "depth_p05_m": depth_p05,
                    },
                )
                if edge_failed:
                    break
            if edge_failed:
                break
            current_node = target_node
            current_position = self.graph.nodes[current_node].camera_position
            visited.append(current_node)
            history.append(f"node:{current_node}")
            if len(visited) >= int(self.config["episode"]["loop_window_decisions"]):
                window = visited[-int(self.config["episode"]["loop_window_decisions"]) :]
                repeats = max(window.count(value) for value in set(window))
                if repeats >= int(self.config["episode"]["loop_repeat_threshold"]):
                    failure_reason = "loop_detected"
                    break

        wall_seconds = time.perf_counter() - started
        simulation_seconds = self.scene.simulated_seconds - simulation_started
        active_wall_seconds = max(0.0, wall_seconds - hold_seconds)
        spl = float(success) * shortest_distance / max(shortest_distance, path_length, 1.0e-9)
        final_distance = self.graph.shortest_distance(current_node, goal_node)
        result = {
            "run_id": self.run_id,
            "episode_id": episode_id,
            "source_episode_id": episode["episode_id"],
            "scene_id": self.graph.scene_id,
            "success": success,
            "spl": spl,
            "shortest_path_m": shortest_distance,
            "executed_path_m": path_length,
            "final_geodesic_m": final_distance,
            "slow_decisions": slow_decisions,
            "fast_steps": fast_steps,
            "collisions": collisions,
            "slow_fallbacks": slow_fallbacks,
            "fast_fallbacks": fast_fallbacks,
            "frontier_decisions": frontier_decisions,
            "frontier_valid_decisions": frontier_valid_decisions,
            "frontier_geodesic_improved": frontier_geodesic_improved,
            "frontier_geodesic_delta_sum_m": frontier_geodesic_delta_sum,
            "repeated_frontier_decisions": repeated_frontier_decisions,
            "target_found_tp": target_tp,
            "target_found_fp": target_fp,
            "target_found_fn": target_fn,
            "hold_seconds": hold_seconds,
            "hold_fraction_wall": hold_seconds / max(wall_seconds, 1.0e-9),
            "wall_seconds": wall_seconds,
            "simulation_seconds": simulation_seconds,
            "fast_active_wall_seconds": active_wall_seconds,
            "fast_effective_control_hz_excluding_slow_hold": fast_steps / max(active_wall_seconds, 1.0e-9),
            "fast_control_hz_during_slow_hold": 0.0,
            "slow_calls_per_minute": slow_decisions * 60.0 / max(wall_seconds, 1.0e-9),
            "visited_nodes": visited,
            "failure_reason": failure_reason,
            "map_setting": self.config["map_setting"],
        }
        if self.video_config.get("enabled"):
            video = write_episode_video(
                self.output / "videos",
                episode_id,
                video_frames,
                fps=float(self.video_config.get("fps", 2.0)),
                codec=str(self.video_config.get("codec", "libx264")),
                quality=int(self.video_config.get("quality", 7)),
            )
            video["path"] = str(Path(video["path"]).relative_to(self.output))
            result["video"] = video
        append_jsonl(self.episodes_path, result)
        print(json.dumps(result, separators=(",", ":")), flush=True)
        return result


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def main() -> int:
    config = load_config(args_cli.config)
    if config.get("map_setting") != "oracle_map":
        raise ValueError("formal benchmark runner currently supports only declared oracle_map setting")
    if config.get("coordinate_transform", {}).get("status") != "validated_on_frozen_manifests_and_real_isaac_probe":
        raise ValueError("coordinate transform is not marked validated")
    if config.get("fast_policy", {}).get("safety_fallback") != "oracle_graph_bearing":
        raise ValueError("unexpected FastPolicy safety fallback")
    output = Path(args_cli.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase_path = output / "phases.jsonl"

    def phase(name: str, **details: Any) -> None:
        append_jsonl(
            phase_path,
            {"timestamp": time.time(), "phase": name, "scene_id": args_cli.scene, **details},
        )
        print(f"BENCHMARK_PHASE:{name}", flush=True)

    phase("main_start")
    episodes = [row for row in read_jsonl(args_cli.manifest) if row.get("scene_key") == args_cli.scene]
    if args_cli.max_episodes > 0:
        episodes = episodes[: args_cli.max_episodes]
    if not episodes:
        raise ValueError(f"manifest has no episodes for scene {args_cli.scene}")
    phase("manifest_loaded", episode_count=len(episodes))
    completed = set()
    episodes_path = output / "episodes.jsonl"
    if episodes_path.exists():
        completed = {row["episode_id"] for row in read_jsonl(str(episodes_path)) if row.get("run_id") == args_cli.run_id}

    graph = MatterportGraph.load(args_cli.connectivity, scene_id=args_cli.scene)
    phase("graph_loaded", node_count=len(graph.nodes))
    scene = IsaacMp3DScene(args_cli.scene_usd, config, str(args_cli.device))
    phase("isaac_scene_ready")
    slow = SlowPlannerClient(config["slow_planner"]["endpoint"], timeout_ms=int(config["slow_planner"]["timeout_ms"]))
    fast = ZmqNavigationClient(config["fast_policy"]["endpoint"], timeout_ms=60_000)
    phase("clients_created")
    slow_health = slow.health()
    phase("slow_health_received", ready=bool(slow_health.get("ready")))
    fast_health = fast.health()
    phase("fast_health_received", ready=bool(fast_health.get("ready")))
    if not slow_health.get("ok") or not slow_health.get("ready"):
        raise RuntimeError(f"slow planner is not ready: {slow_health}")
    if not fast_health.get("ready"):
        raise RuntimeError(f"fast policy is not ready: {fast_health}")
    expected_slow = {
        "model_variant": args_cli.expected_slow_model_variant,
        "precision_mode": args_cli.expected_slow_precision_mode,
        "service_config_sha256": args_cli.expected_slow_config_sha256,
        "max_new_tokens": int(config["slow_planner"]["output_token_limit"]),
        "max_retries": int(config["slow_planner"]["max_retries"]),
        "deterministic_decoding": True,
        "batch_size": 1,
        "processor_use_fast": False,
    }
    if args_cli.expected_slow_model_variant == "qwen25_baseline_bf16":
        expected_slow.update(
            independent_slow_instance=True,
            contains_omninav_action_head_but_never_calls_action_former=True,
        )
    elif args_cli.expected_slow_model_variant == "cosmos_reason2_32b_bf16":
        expected_slow["reasoning_budget"] = "disabled_by_compact_json_prompt"
    elif args_cli.expected_slow_model_variant == "step3_vl_10b_bf16":
        expected_slow.update(
            pacore=False,
            multi_crop=True,
            fix_mistral_regex=True,
            runtime_transformers_version="4.57.6",
            checkpoint_key_mapping="step3_flat_checkpoint_to_nested_v1",
            checkpoint_load_clean=True,
        )
    slow_mismatches = {
        key: {"expected": expected, "actual": slow_health.get(key)}
        for key, expected in expected_slow.items()
        if slow_health.get(key) != expected
    }
    if slow_mismatches:
        raise RuntimeError(f"slow planner health differs from frozen configuration: {slow_mismatches}")
    fast_config = config["fast_policy"]
    expected_fast = {
        "model_variant": str(fast_config["expected_model_variant"]),
        "precision_mode": str(fast_config["expected_precision_mode"]),
        "input_contract": str(fast_config["expected_input_contract"]),
        "action_head_trained": bool(fast_config["require_trained_action_head"]),
    }
    fast_mismatches = {
        key: {"expected": expected, "actual": fast_health.get(key)}
        for key, expected in expected_fast.items()
        if fast_health.get(key) != expected
    }
    if fast_mismatches:
        raise RuntimeError(f"FastPolicy health differs from frozen configuration: {fast_mismatches}")
    hardware = {
        "run_id": args_cli.run_id,
        "scene_id": args_cli.scene,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device": str(scene.sim.device),
        "requested_device": str(args_cli.device),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "slow_health": slow_health,
        "fast_health": fast_health,
    }
    hardware_name = f"hardware_{args_cli.scene}.json"
    (output / hardware_name).write_text(json.dumps(hardware, indent=2), encoding="utf-8")
    hardware_index_path = output / "hardware.json"
    hardware_index = {"schema_version": 1, "run_id": args_cli.run_id, "scene_files": []}
    if hardware_index_path.exists():
        hardware_index = json.loads(hardware_index_path.read_text(encoding="utf-8"))
    hardware_index["scene_files"] = sorted(set(hardware_index.get("scene_files", [])) | {hardware_name})
    hardware_index_path.write_text(json.dumps(hardware_index, indent=2), encoding="utf-8")
    precision = {
        "run_id": args_cli.run_id,
        "slow_model_variant": slow_health.get("model_variant"),
        "slow_precision_mode": slow_health.get("precision_mode"),
        "fast_model_variant": fast_health.get("model_variant"),
        "fast_precision_mode": fast_health.get("precision_mode"),
        "fast_action_head_required": True,
        "fast_episode_state_scope": "run_id_plus_canonical_episode_id",
        "video_evidence_required": bool(config.get("evidence", {}).get("video", {}).get("required")),
        "benchmark_config_sha256": sha256_file(Path(args_cli.config).resolve()),
        "slow_service_config_sha256": slow_health.get("service_config_sha256"),
        "slow_revision": slow_health.get("revision"),
        "device": str(scene.sim.device),
        "camera_source": config["camera"]["source"],
        "camera_resolution": [int(config["camera"]["width"]), int(config["camera"]["height"])],
        "camera_horizontal_fov_deg": float(config["camera"]["horizontal_fov_deg"]),
        "collision_definition": config["controller"]["collision_definition"],
        "synchronous_safe_hold": bool(config["slow_planner"]["synchronous_safe_hold"]),
        "map_setting": config["map_setting"],
    }
    precision_path = output / "precision_manifest.json"
    if precision_path.exists() and json.loads(precision_path.read_text(encoding="utf-8")) != precision:
        raise RuntimeError("precision manifest changed between scenes in the same run")
    precision_path.write_text(json.dumps(precision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    phase("run_started")
    runner = EpisodeRunner(scene, graph, config, slow, fast, output, args_cli.run_id)
    try:
        for episode in episodes:
            if episode["benchmark_episode_id"] in completed:
                continue
            runner.run(episode)
    finally:
        slow.close()
        fast.close()
    phase("run_finished")
    return 0


try:
    exit_code = main()
except BaseException as exc:
    output = Path(args_cli.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    append_jsonl(
        output / "phases.jsonl",
        {
            "timestamp": time.time(),
            "phase": "fatal_error",
            "scene_id": args_cli.scene,
            "error": f"{type(exc).__name__}:{exc}",
        },
    )
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
raise SystemExit(exit_code)
