#!/usr/bin/env python3
"""Runtime Go2 registration for frozen flash and T3 continuous execution."""

from __future__ import annotations

import math
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np


_T5_SIM_CLOCK_NS: int | None = None
_T5_CLOCK_SOCKET: socket.socket | None = None
_T5_CAMERA_SOURCE_SEQUENCE = 0


def _t5_continuation_bootstrap_identity_enabled() -> bool:
    continuation = os.environ.get(
        "INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET", ""
    )
    if continuation not in {"", "0", "1"}:
        raise RuntimeError("invalid T5 evaluator continuation identity")
    return bool(
        os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
        and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
        and os.environ.get("INTERNNAV_T5_LANE", "") in {"a", "b"}
        and continuation == "1"
    )


def _continuous_state_only_identity(
    identity: Any, bootstrap_generation: int
) -> tuple[str, int, int, bool]:
    """Bind continuation state-only samples to the real evaluator identity."""

    generation = int(bootstrap_generation)
    if generation < 0:
        raise RuntimeError("continuous state bootstrap generation is unset")
    if not _t5_continuation_bootstrap_identity_enabled():
        return f"bootstrap-episode-{generation}", generation, 0, True

    lane = os.environ.get("INTERNNAV_T5_LANE", "")
    episode_id = str(identity.episode_id)
    reset_generation = int(identity.reset_generation)
    sequence_id = int(identity.sequence_id)
    if (
        not episode_id.startswith(f"{lane}::")
        or reset_generation < generation
        or sequence_id < 0
    ):
        raise RuntimeError("continuation state-only has no current identity")
    return episode_id, reset_generation, sequence_id, True


def _next_t5_camera_source_sequence() -> int:
    global _T5_CAMERA_SOURCE_SEQUENCE
    if _T5_CAMERA_SOURCE_SEQUENCE == 0:
        seed_text = os.environ.get(
            "INTERNVLA_T5_CAMERA_SOURCE_SEQUENCE_START", ""
        )
        if seed_text:
            if not (
                os.environ.get("INTERNNAV_RUNTIME_POLICY", "")
                == "completion_sim"
                and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
                and os.environ.get("INTERNNAV_T5_LANE", "") in {"a", "b"}
            ):
                raise RuntimeError("T5 camera-sequence seed escaped completion_sim")
            try:
                seed = int(seed_text)
            except ValueError as error:
                raise RuntimeError("T5 camera-sequence seed is not an integer") from error
            if seed <= 0:
                raise RuntimeError("T5 camera-sequence seed must be positive")
            _T5_CAMERA_SOURCE_SEQUENCE = seed
    _T5_CAMERA_SOURCE_SEQUENCE += 1
    return _T5_CAMERA_SOURCE_SEQUENCE


def _attach_t5_camera_source_metadata(
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Stamp an actual x86 camera sample with the x86-owned sim clock."""

    t5_source_contract = (
        os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
        and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
        and os.environ.get("INTERNNAV_T5_LANE", "") in {"a", "b"}
    )
    if not t5_source_contract:
        return observation
    if "rgb" not in observation or "depth" not in observation:
        raise RuntimeError("T5 camera source frame is incomplete")
    source_stamp_ns = _T5_SIM_CLOCK_NS
    if source_stamp_ns is None or int(source_stamp_ns) <= 0:
        raise RuntimeError("T5 camera source frame has no simulation stamp")
    observation["camera_sensor_metadata"] = {
        "schema_version": 1,
        "source": "x86_isaac_pano_camera_0",
        "sequence": _next_t5_camera_source_sequence(),
        "sim_stamp_ns": int(source_stamp_ns),
    }
    return observation


def _advance_t5_sim_clock(delta_sec: float) -> None:
    """Advance the x86-owned T5 simulation clock by one physics step."""

    endpoint = os.environ.get("INTERNVLA_T5_CLOCK_UDP_ENDPOINT", "")
    if not endpoint:
        return
    host, separator, port_text = endpoint.rpartition(":")
    if not separator or host != "127.0.0.1" or not port_text.isdigit():
        raise RuntimeError("T5 clock endpoint must be 127.0.0.1:PORT")
    port = int(port_text)
    if not 1024 <= port <= 65535:
        raise RuntimeError("T5 clock port is outside [1024,65535]")
    global _T5_SIM_CLOCK_NS, _T5_CLOCK_SOCKET
    if _T5_SIM_CLOCK_NS is None:
        seed_text = os.environ.get("INTERNVLA_T5_SIM_CLOCK_START_NS", "")
        if seed_text:
            if not (
                os.environ.get("INTERNNAV_RUNTIME_POLICY", "")
                == "completion_sim"
                and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
                and os.environ.get("INTERNNAV_T5_LANE", "") in {"a", "b"}
            ):
                raise RuntimeError("T5 simulation-clock seed escaped completion_sim")
            try:
                seed_ns = int(seed_text)
            except ValueError as error:
                raise RuntimeError("T5 simulation-clock seed is not an integer") from error
            if seed_ns <= 0:
                raise RuntimeError("T5 simulation-clock seed must be positive")
            _T5_SIM_CLOCK_NS = seed_ns
        else:
            _T5_SIM_CLOCK_NS = time.time_ns()
    increment = int(round(float(delta_sec) * 1e9))
    if increment <= 0:
        raise RuntimeError("T5 simulation clock increment must be positive")
    _T5_SIM_CLOCK_NS += increment
    if _T5_CLOCK_SOCKET is None:
        _T5_CLOCK_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _T5_CLOCK_SOCKET.sendto(str(_T5_SIM_CLOCK_NS).encode("ascii"), (host, port))


JOINT_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]
COLLISION_SENSOR_BODIES = ("base", "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh")
STAND_JOINT_POSITIONS = np.asarray(
    [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
    dtype=np.float64,
)
CONTINUOUS_STAND_JOINT_POSITIONS = np.asarray(
    [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 0.8, 0.8, -1.5, -1.5, -1.5, -1.5],
    dtype=np.float64,
)


def _stop_root(robot: Any) -> None:
    articulation = robot.articulation.unwrap()
    try:
        velocity = np.asarray(articulation.get_linear_velocity(), dtype=np.float64)
    except Exception:
        # InternUtopia rebuilds the PhysX simulation view between episodes. A
        # hard reset can invalidate the SingleArticulation tensor handle before
        # the controller's reset barrier runs. Rebind once and retry; a second
        # failure propagates so execution cannot continue without a confirmed
        # physical stop.
        articulation.initialize()
        velocity = np.asarray(articulation.get_linear_velocity(), dtype=np.float64)
    vertical = float(velocity[2]) if velocity.shape == (3,) and np.isfinite(velocity[2]) else 0.0
    articulation.set_linear_velocity(np.asarray([0.0, 0.0, vertical], dtype=np.float64))
    articulation.set_angular_velocity(np.zeros(3, dtype=np.float64))


def _tilt(rotation_wxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = [float(value) for value in rotation_wxyz]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(pitch_argument)


def _integrate_t5_planar_pose(
    position_xyz: np.ndarray,
    rotation_wxyz: np.ndarray,
    linear_x: float,
    angular_z: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one already-limited body twist as a planar SE(2) pose."""

    position = np.asarray(position_xyz, dtype=np.float64)
    rotation = np.asarray(rotation_wxyz, dtype=np.float64)
    values = np.asarray([linear_x, angular_z, dt], dtype=np.float64)
    if position.shape != (3,) or rotation.shape != (4,):
        raise ValueError("planar integration requires xyz and wxyz inputs")
    if not np.isfinite(position).all() or not np.isfinite(rotation).all():
        raise ValueError("planar integration received non-finite pose")
    if not np.isfinite(values).all() or dt <= 0.0:
        raise ValueError("planar integration requires finite twist and positive dt")
    yaw = math.atan2(
        2.0 * (rotation[0] * rotation[3] + rotation[1] * rotation[2]),
        1.0 - 2.0 * (rotation[2] ** 2 + rotation[3] ** 2),
    )
    delta_yaw = float(angular_z) * float(dt)
    next_yaw = math.atan2(math.sin(yaw + delta_yaw), math.cos(yaw + delta_yaw))
    if abs(float(angular_z)) < 1.0e-9:
        delta_x = float(linear_x) * float(dt) * math.cos(yaw)
        delta_y = float(linear_x) * float(dt) * math.sin(yaw)
    else:
        radius = float(linear_x) / float(angular_z)
        delta_x = radius * (math.sin(yaw + delta_yaw) - math.sin(yaw))
        delta_y = -radius * (math.cos(yaw + delta_yaw) - math.cos(yaw))
    next_position = position.copy()
    next_position[0] += delta_x
    next_position[1] += delta_y
    next_rotation = np.asarray(
        [math.cos(next_yaw / 2.0), 0.0, 0.0, math.sin(next_yaw / 2.0)],
        dtype=np.float64,
    )
    return next_position, next_rotation


def install_go2_runtime() -> None:
    from internutopia.core.robot.articulation import ArticulationAction
    from internutopia.core.robot.articulation_subset import ArticulationSubset
    from internutopia.core.robot.controller import BaseController
    from internutopia.core.robot.robot import BaseRobot
    from internutopia_extension.robots.aliengo import AliengoRobot
    from internnav.configs.evaluator import ControllerCfg
    from internnav.configs.evaluator import vln_default_config
    from internnav.env.utils.internutopia_extension.tasks.vln_eval_task import (
        VLNEvalTask,
    )
    from internnav.env.utils.internutopia_extension.configs.controllers import (
        VlnMoveByFlashControllerCfg,
    )

    if getattr(vln_default_config, "_internnav_go2_installed", False):
        return

    original_get_rgb_depth = VLNEvalTask.get_rgb_depth

    def get_rgb_depth_with_source_metadata(self: Any) -> dict[str, Any]:
        observation = original_get_rgb_depth(self)
        return _attach_t5_camera_source_metadata(observation)

    VLNEvalTask.get_rgb_depth = get_rgb_depth_with_source_metadata

    def _author_diagnostic_course() -> None:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root = "/World/t3_continuous_diagnostics"
        if stage.GetPrimAtPath(root).IsValid():
            stage.RemovePrim(root)
        stage.DefinePrim(root, "Xform")

        def static_box(
            name: str,
            center: tuple[float, float, float],
            dimensions: tuple[float, float, float],
            color: tuple[float, float, float],
        ) -> None:
            cube = UsdGeom.Cube.Define(stage, f"{root}/{name}")
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*center))
            xform.AddScaleOp().Set(Gf.Vec3d(*dimensions))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        # Dataset [x, y, z] maps to the Isaac ground plane [x, -z].  This
        # isolated course is intentionally outside the MP3D scene bounds.
        static_box(
            "floor",
            (120.0, 101.0, -0.10),
            (50.0, 6.0, 0.20),
            (0.20, 0.24, 0.28),
        )
        static_box(
            "doorway_left",
            (139.35, 100.90, 0.40),
            (0.30, 0.30, 0.80),
            (0.85, 0.65, 0.08),
        )
        static_box(
            "doorway_right",
            (140.65, 100.90, 0.40),
            (0.30, 0.30, 0.80),
            (0.85, 0.65, 0.08),
        )

    class T3ObstacleManager:
        """Deterministic evaluation-only obstacle authoring; never moves the robot."""

        root_path = "/World/t3_obstacles"

        def __init__(self, robot: Any) -> None:
            self.robot = robot
            self.enabled = bool(os.environ.get("INTERNVLA_T3_SCENARIO_MANIFEST", ""))
            self.generation = -1
            self.scenario = ""
            self.elapsed = 0.0
            self.origin: tuple[float, float, float] | None = None
            self.origin_yaw = 0.0
            self.boxes: list[dict[str, Any]] = []
            self.sudden_created_elapsed: float | None = None
            self.scenario_started_wall = 0.0
            self.sudden_created_wall: float | None = None

        def _stage(self) -> Any:
            import omni.usd

            return omni.usd.get_context().get_stage()

        def _clear(self) -> None:
            if not self.enabled:
                return
            stage = self._stage()
            if stage.GetPrimAtPath(self.root_path).IsValid():
                stage.RemovePrim(self.root_path)
            self.boxes = []

        def reset(self) -> None:
            self._clear()
            self.generation = -1
            self.scenario = ""
            self.elapsed = 0.0
            self.origin = None
            self.sudden_created_elapsed = None
            self.scenario_started_wall = 0.0
            self.sudden_created_wall = None

        def _world_from_local(self, forward: float, lateral: float) -> tuple[float, float]:
            if self.origin is None:
                raise RuntimeError("obstacle origin is not initialized")
            cosine, sine = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
            return (
                self.origin[0] + cosine * forward - sine * lateral,
                self.origin[1] + sine * forward + cosine * lateral,
            )

        def _create_box(
            self,
            name: str,
            *,
            forward: float,
            lateral: float,
            dimensions: tuple[float, float, float],
            kinematic: bool = False,
        ) -> dict[str, Any]:
            from pxr import Gf, UsdGeom, UsdPhysics

            if self.origin is None:
                raise RuntimeError("obstacle origin is not initialized")
            stage = self._stage()
            stage.DefinePrim(self.root_path, "Xform")
            path = f"{self.root_path}/{name}"
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.08, 0.05)])
            xform = UsdGeom.Xformable(cube.GetPrim())
            translate = xform.AddTranslateOp()
            xform.AddRotateZOp().Set(math.degrees(self.origin_yaw))
            xform.AddScaleOp().Set(Gf.Vec3d(*dimensions))
            x, y = self._world_from_local(forward, lateral)
            z = self.origin[2] + dimensions[2] / 2.0
            translate.Set(Gf.Vec3d(x, y, z))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            if kinematic:
                body = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
                body.CreateKinematicEnabledAttr(True)
            item = {
                "name": name,
                "translate": translate,
                "world": [x, y, z],
                "yaw_world": self.origin_yaw,
                "dimensions": list(dimensions),
                "active": True,
            }
            self.boxes.append(item)
            return item

        def _initialize(
            self,
            name: str,
            generation: int,
            route_yaw_world: float | None,
        ) -> None:
            articulation = self.robot.articulation.unwrap()
            position, rotation = articulation.get_world_pose()
            position = np.asarray(position, dtype=np.float64)
            rotation = np.asarray(rotation, dtype=np.float64)
            self._clear()
            self.generation = generation
            self.scenario = name
            self.elapsed = 0.0
            self.sudden_created_elapsed = None
            self.scenario_started_wall = time.monotonic()
            self.sudden_created_wall = None
            self.origin_yaw = math.atan2(
                2.0 * (rotation[0] * rotation[3] + rotation[1] * rotation[2]),
                1.0 - 2.0 * (rotation[2] ** 2 + rotation[3] ** 2),
            )
            if name == "doorway" and route_yaw_world is not None:
                if not math.isfinite(route_yaw_world):
                    raise RuntimeError("doorway route yaw must be finite")
                self.origin_yaw = float(route_yaw_world)
            base_to_floor = 0.425
            self.origin = (
                float(position[0]),
                float(position[1]),
                float(position[2]) - base_to_floor,
            )
            stress_geometry = os.environ.get("INTERNVLA_T3_PHASE", "") == "stress"
            if name == "static_box":
                self._create_box(
                    "static_box", forward=1.35, lateral=0.0, dimensions=(0.40, 0.65, 0.70)
                )
            elif name == "dynamic_crossing":
                self._create_box(
                    "dynamic_crossing",
                    forward=1.40,
                    lateral=0.90,
                    dimensions=(0.35, 0.35, 0.65),
                    kinematic=True,
                )
            elif name == "doorway":
                doorway_lateral = 0.55 if stress_geometry else 1.15
                doorway_width = 0.40 if stress_geometry else 0.50
                self._create_box(
                    "doorway_left",
                    forward=0.75,
                    lateral=doorway_lateral,
                    dimensions=(0.40, doorway_width, 0.80),
                )
                self._create_box(
                    "doorway_right",
                    forward=0.75,
                    lateral=-doorway_lateral,
                    dimensions=(0.40, doorway_width, 0.80),
                )
            elif name == "narrow_corridor":
                corridor_lateral = 0.60 if stress_geometry else 0.90
                corridor_width = 0.30 if stress_geometry else 0.40
                self._create_box(
                    "corridor_left",
                    forward=1.80,
                    lateral=corridor_lateral,
                    dimensions=(2.50, corridor_width, 0.75),
                )
                self._create_box(
                    "corridor_right",
                    forward=1.80,
                    lateral=-corridor_lateral,
                    dimensions=(2.50, corridor_width, 0.75),
                )
            elif name != "sudden_blockage":
                raise RuntimeError(f"unknown T3 obstacle scenario: {name}")

        def _remove_box(self, item: dict[str, Any]) -> None:
            if not item["active"]:
                return
            self._stage().RemovePrim(f"{self.root_path}/{item['name']}")
            item["active"] = False

        def update(self, dt: float) -> list[list[float]]:
            if not self.enabled:
                return []
            from internvla_go2_controller.runtime import get_obstacle_scenario
            from pxr import Gf

            state = get_obstacle_scenario()
            if not state.name:
                if state.reset_generation != self.generation:
                    self._clear()
                    self.generation = state.reset_generation
                    self.scenario = ""
                return []
            if state.reset_generation != self.generation or state.name != self.scenario:
                self._initialize(
                    state.name,
                    state.reset_generation,
                    state.route_yaw_world,
                )
            self.elapsed += dt
            if self.scenario == "sudden_blockage" and not self.boxes:
                stress_geometry = os.environ.get("INTERNVLA_T3_PHASE", "") == "stress"
                should_create = self.elapsed >= 1.5
                if stress_geometry:
                    articulation = self.robot.articulation.unwrap()
                    velocity = np.asarray(
                        articulation.get_linear_velocity(), dtype=np.float64
                    )
                    moving = math.hypot(float(velocity[0]), float(velocity[1])) >= 0.03
                    wall_elapsed = time.monotonic() - self.scenario_started_wall
                    # The real policy can spend tens of seconds producing a
                    # turn or a local plan before translating.  Trigger the
                    # near-field challenge on actual motion using monotonic
                    # wall time (model waits do not advance simulator time).
                    # Retain a 45 s fallback so a stationary episode still
                    # acquires the obstacle before the 60 s stuck handler.
                    should_create = wall_elapsed >= 1.5 and (
                        moving or wall_elapsed >= 45.0
                    )
                if should_create and stress_geometry:
                    # Author the block from the robot's pose at appearance,
                    # not its episode-start pose.  The near face begins 0.45 m
                    # ahead: outside the 0.30 m footprint and inside the
                    # stress-only conservative 0.50 m stop polygon.
                    articulation = self.robot.articulation.unwrap()
                    position, rotation = articulation.get_world_pose()
                    position = np.asarray(position, dtype=np.float64)
                    rotation = np.asarray(rotation, dtype=np.float64)
                    self.origin_yaw = math.atan2(
                        2.0 * (rotation[0] * rotation[3] + rotation[1] * rotation[2]),
                        1.0 - 2.0 * (rotation[2] ** 2 + rotation[3] ** 2),
                    )
                    self.origin = (
                        float(position[0]),
                        float(position[1]),
                        float(position[2]) - 0.425,
                    )
                    self._create_box(
                        "sudden_block",
                        forward=0.55,
                        lateral=0.0,
                        dimensions=(0.20, 0.75, 0.70),
                    )
                    self.sudden_created_elapsed = self.elapsed
                    self.sudden_created_wall = time.monotonic()
                elif should_create:
                    self._create_box(
                        "sudden_block",
                        forward=0.75,
                        lateral=0.0,
                        dimensions=(0.35, 0.75, 0.70),
                    )
                    self.sudden_created_elapsed = self.elapsed
            elif self.scenario == "dynamic_crossing" and self.boxes:
                item = self.boxes[0]
                if self.elapsed >= 6.5:
                    self._remove_box(item)
                else:
                    lateral = max(-0.90, 0.90 - 0.35 * self.elapsed)
                    x, y = self._world_from_local(1.40, lateral)
                    item["world"][0:2] = [x, y]
                    item["translate"].Set(Gf.Vec3d(*item["world"]))
            elif (
                self.scenario == "sudden_blockage"
                and self.sudden_created_elapsed is not None
                and (
                    (
                        os.environ.get("INTERNVLA_T3_PHASE", "") == "stress"
                        and self.sudden_created_wall is not None
                        and time.monotonic() >= self.sudden_created_wall + 10.0
                    )
                    or (
                        os.environ.get("INTERNVLA_T3_PHASE", "") != "stress"
                        and self.elapsed >= self.sudden_created_elapsed + 3.0
                    )
                )
                and self.boxes
            ):
                self._remove_box(self.boxes[0])

            articulation = self.robot.articulation.unwrap()
            position, rotation = articulation.get_world_pose()
            yaw = math.atan2(
                2.0 * (rotation[0] * rotation[3] + rotation[1] * rotation[2]),
                1.0 - 2.0 * (rotation[2] ** 2 + rotation[3] ** 2),
            )
            cosine, sine = math.cos(yaw), math.sin(yaw)
            targets = []
            for item in self.boxes:
                if not item["active"]:
                    continue
                dx = float(item["world"][0]) - float(position[0])
                dy = float(item["world"][1]) - float(position[1])
                base_x = cosine * dx + sine * dy
                base_y = -sine * dx + cosine * dy
                base_z = float(item["world"][2]) - float(position[2])
                dimensions = item["dimensions"]
                half_sizes = [float(value) / 2.0 for value in dimensions]
                target_yaw = float(item["yaw_world"]) - yaw
                target_cosine = math.cos(target_yaw)
                target_sine = math.sin(target_yaw)

                # Only label a target as expected when its AABB intersects the
                # audited 640x480 pinhole camera frustum.  These labels are used
                # solely for scoring real depth-derived points; they never enter
                # the point cloud or costmap.
                observable = False
                # Corners alone are insufficient for a near, large obstacle:
                # all eight corners can lie outside the image while the box
                # spans the complete camera frustum.  Include face and body
                # centers in the conservative visibility intersection test.
                for sx in (-1.0, 0.0, 1.0):
                    for sy in (-1.0, 0.0, 1.0):
                        for sz in (-1.0, 0.0, 1.0):
                            local_x = sx * half_sizes[0]
                            local_y = sy * half_sizes[1]
                            relative_x = (
                                base_x
                                + target_cosine * local_x
                                - target_sine * local_y
                                - 0.2
                            )
                            relative_y = (
                                base_y
                                + target_sine * local_x
                                + target_cosine * local_y
                            )
                            relative_z = base_z + sz * half_sizes[2] - 0.2
                            camera_x = -relative_y
                            camera_y = 0.5 * relative_x + 0.8660254037844386 * relative_z
                            view_depth = (
                                0.8660254037844386 * relative_x - 0.5 * relative_z
                            )
                            if (
                                0.1 < view_depth <= 6.0
                                and abs(camera_x / view_depth) <= 320.0 / 585.0
                                and abs(camera_y / view_depth) <= 240.0 / 585.0
                            ):
                                observable = True
                                break
                        if observable:
                            break
                    if observable:
                        break
                if not observable:
                    continue
                targets.append(
                    [base_x, base_y, base_z, *half_sizes, target_yaw]
                )
            return targets

    @BaseController.register("Go2StandStillController")
    class Go2StandStillController(BaseController):
        """Actively holds the audited stable Go2 pose during warm-up and holds."""

        def __init__(self, config: Any, robot: Any, scene: Any) -> None:
            super().__init__(config=config, robot=robot, scene=scene)
            self.joint_subset = ArticulationSubset(robot.articulation, JOINT_NAMES)

        def action_to_control(self, action: list[Any] | np.ndarray) -> ArticulationAction:
            if len(action) != 0:
                raise ValueError("stand_still action must be empty")
            if execution_mode == "continuous":
                _advance_t5_sim_clock(
                    1.0 / float(os.environ.get("INTERNVLA_GO2_PHYSICS_HZ", "200"))
                )
            _stop_root(self.robot)
            if execution_mode == "continuous":
                continuous = self.robot.controllers.get("vln_dp_move_by_speed")
                if continuous is None:
                    raise RuntimeError("continuous controller is unavailable during state bootstrap")
                continuous.publish_state_only()
            return self.joint_subset.make_articulation_action(
                joint_positions=(
                    CONTINUOUS_STAND_JOINT_POSITIONS
                    if execution_mode == "continuous"
                    else STAND_JOINT_POSITIONS
                ),
                joint_velocities=None,
            )

        def get_obs(self) -> dict[str, bool]:
            return {"finished": True}

    @BaseController.register("Go2ContinuousController")
    class Go2ContinuousController(BaseController):
        """Paced physics controller consuming collision-monitored Nav2 velocity."""

        def __init__(self, config: Any, robot: Any, scene: Any) -> None:
            super().__init__(config=config, robot=robot, scene=scene)
            from internvla_go2_controller.runtime import (
                ControllerIPCClient,
                JerkLimitedTwist,
            )

            self.physics_hz = float(os.environ.get("INTERNVLA_GO2_PHYSICS_HZ", "200"))
            self.control_hz = float(os.environ.get("INTERNVLA_GO2_CONTROL_HZ", "40"))
            self.sensor_hz = float(os.environ.get("INTERNVLA_GO2_SENSOR_HZ", "10"))
            self.motion_profile = os.environ.get(
                "INTERNVLA_GO2_MOTION_PROFILE", "physics_root_velocity"
            )
            self.window_sec = float(os.environ.get("INTERNVLA_GO2_CONTROL_WINDOW_SEC", "0.5"))
            if not 20.0 <= self.control_hz <= 50.0 or self.physics_hz < self.control_hz:
                raise RuntimeError("invalid Go2 continuous controller frequency")
            if self.motion_profile not in {
                "physics_root_velocity",
                "t5_completion_planar_root_velocity",
            }:
                raise RuntimeError("invalid Go2 continuous motion profile")
            if not 1.0 <= self.sensor_hz <= self.control_hz:
                raise RuntimeError("invalid Go2 simulated sensor frequency")
            self.steps_per_control = int(round(self.physics_hz / self.control_hz))
            self.steps_per_window = int(round(self.physics_hz * self.window_sec))
            if self.steps_per_control <= 0 or self.steps_per_window < self.steps_per_control:
                raise RuntimeError("invalid continuous controller step window")
            self.dt = 1.0 / self.physics_hz
            self.control_dt = self.steps_per_control * self.dt
            self.depth_control_interval = int(round(self.control_hz / self.sensor_hz))
            effective_sensor_hz = self.control_hz / self.depth_control_interval
            if abs(effective_sensor_hz - self.sensor_hz) > 1e-9:
                raise RuntimeError(
                    "Go2 control frequency must be an integer multiple of sensor frequency"
                )
            self.joint_subset = ArticulationSubset(robot.articulation, JOINT_NAMES)
            self.ipc = ControllerIPCClient(
                os.environ.get(
                    "INTERNVLA_GO2_CONTROLLER_SOCKET",
                    "/tmp/internvla_go2_controller.sock",
                ),
                timeout_sec=0.25,
            )
            self.limiter = JerkLimitedTwist(
                max_linear=float(os.environ.get("INTERNVLA_GO2_MAX_LINEAR", "0.15")),
                max_angular=float(os.environ.get("INTERNVLA_GO2_MAX_ANGULAR", "0.6")),
                max_linear_acceleration=float(
                    os.environ.get("INTERNVLA_GO2_MAX_LINEAR_ACCEL", "0.4")
                ),
                max_angular_acceleration=float(
                    os.environ.get("INTERNVLA_GO2_MAX_ANGULAR_ACCEL", "1.2")
                ),
                max_linear_jerk=float(
                    os.environ.get("INTERNVLA_GO2_MAX_LINEAR_JERK", "1.5")
                ),
                max_angular_jerk=float(
                    os.environ.get("INTERNVLA_GO2_MAX_ANGULAR_JERK", "4.0")
                ),
            )
            # The Go2 asset is held in its audited standing joint pose while
            # planar motion is applied at the articulation root.  Foot friction
            # otherwise accumulates pitch/roll even at low forward speed.  Use
            # a velocity-level base stabilizer on every physics step: it never
            # writes a world pose, and therefore remains continuous and subject
            # to PhysX contacts between control updates.
            self.tilt_stabilization_gain = float(
                os.environ.get("INTERNVLA_GO2_TILT_STABILIZATION_GAIN", "12.0")
            )
            self.maximum_tilt_correction = float(
                os.environ.get("INTERNVLA_GO2_MAX_TILT_CORRECTION", "2.0")
            )
            self.height_stabilization_gain = float(
                os.environ.get("INTERNVLA_GO2_HEIGHT_STABILIZATION_GAIN", "8.0")
            )
            self.maximum_vertical_speed = float(
                os.environ.get("INTERNVLA_GO2_MAX_VERTICAL_SPEED", "0.6")
            )
            if (
                self.tilt_stabilization_gain <= 0.0
                or self.maximum_tilt_correction <= 0.0
                or self.height_stabilization_gain <= 0.0
                or self.maximum_vertical_speed <= 0.0
            ):
                raise RuntimeError("Go2 continuous stabilization gains must be positive")
            self.nominal_base_height: float | None = None
            self.step_in_window = 0
            self.control_update_index = 0
            self.window_finished = True
            self.window_started_wall = 0.0
            self.desired = (0.0, 0.0)
            self.applied = (0.0, 0.0)
            self.emergency_stop = True
            self.obstacles = T3ObstacleManager(robot)
            self.obstacle_targets: list[list[float]] = []
            self.bootstrap_generation = -1
            self.planar_position: np.ndarray | None = None
            self.planar_rotation: np.ndarray | None = None

        def set_bootstrap_generation(self, generation: int) -> None:
            self.bootstrap_generation = int(generation)

        def bind_persistent_bootstrap_generation(self) -> None:
            lane = os.environ.get("INTERNNAV_T5_LANE", "")
            if lane not in {"a", "b"}:
                raise RuntimeError("T5 bootstrap identity query requires a lane")
            identity = self.ipc.query_active_identity(
                expected_episode_prefix=f"{lane}::"
            )
            self.bootstrap_generation = int(identity.reset_generation)

        def reset_runtime(self) -> None:
            self.ipc.close()
            self.limiter.reset()
            self.step_in_window = 0
            self.control_update_index = 0
            self.window_finished = True
            self.desired = (0.0, 0.0)
            self.applied = (0.0, 0.0)
            self.emergency_stop = True
            self.nominal_base_height = None
            self.planar_position = None
            self.planar_rotation = None
            self.obstacles.reset()
            self.obstacle_targets = []
            _stop_root(self.robot)

        def _state(
            self,
        ) -> tuple[
            list[float],
            list[float],
            list[float],
            bool,
            bool,
            bool,
            float,
            list[list[str]],
        ]:
            articulation = self.robot.articulation.unwrap()
            position, rotation = articulation.get_world_pose()
            linear = np.asarray(articulation.get_linear_velocity(), dtype=np.float64)
            angular = np.asarray(articulation.get_angular_velocity(), dtype=np.float64)
            position = np.asarray(position, dtype=np.float64)
            rotation = np.asarray(rotation, dtype=np.float64)
            nan_detected = not (
                np.isfinite(position).all()
                and np.isfinite(rotation).all()
                and np.isfinite(linear).all()
                and np.isfinite(angular).all()
            )
            roll, pitch = _tilt(rotation) if not nan_detected else (math.inf, math.inf)
            ankle = float(self.robot.get_ankle_height()) if not nan_detected else math.inf
            relative_height = float(position[2]) - ankle if not nan_detected else -math.inf
            # Match the official InternNav evaluator thresholds.  The runtime
            # height calculation is conservative: evaluator bottom_z is ankle
            # height minus 0.03 m, so relative_height < 0.12 m is equivalent to
            # its configured 0.15 m fall threshold.
            fallen = (
                relative_height < 0.12
                or abs(roll) > math.radians(15.0)
                or abs(pitch) > math.radians(35.0)
            )
            collision_forces = []
            collision_pairs: list[list[str]] = []
            for sensor in getattr(self.robot, "_collision_sensors", []):
                frame = sensor.get_data()
                if bool(frame.get("in_contact", False)):
                    collision_forces.append(float(frame.get("force", 0.0)))
                    for contact in frame.get("contacts", []):
                        pair = [str(contact.get("body0", "")), str(contact.get("body1", ""))]
                        if pair not in collision_pairs and len(collision_pairs) < 64:
                            collision_pairs.append(pair)
            physical_collision = bool(collision_forces)
            maximum_collision_force = max(collision_forces, default=0.0)
            pose = [
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(rotation[0]),
                float(rotation[1]),
                float(rotation[2]),
                float(rotation[3]),
            ]
            return (
                pose,
                linear.tolist(),
                angular.tolist(),
                fallen,
                nan_detected,
                physical_collision,
                maximum_collision_force,
                collision_pairs,
            )

        def _sample_depth(self) -> dict[str, Any]:
            if self.control_update_index % self.depth_control_interval:
                return {}
            sensor = self.robot.sensors.get("pano_camera_0")
            if sensor is None:
                return {}
            value = sensor.get_data().get("depth")
            if value is None:
                return {}
            depth = np.asarray(value, dtype=np.float32)
            if depth.shape != (480, 640):
                return {}
            row_stride = 10
            column_stride = 10
            sample = np.nan_to_num(
                depth[::row_stride, ::column_stride],
                copy=True,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            valid_depth_count = int(
                np.count_nonzero((sample > 0.1) & (sample <= 6.0))
            )
            # The official wrapper substitutes uniform [0, 256] noise while
            # its render product is being rebuilt.  That placeholder has only
            # about 2.3% samples in the controller's valid depth range.
            if valid_depth_count < int(0.10 * sample.size):
                return {}
            articulation = self.robot.articulation.unwrap()
            base_position, base_rotation = articulation.get_world_pose()
            base_position = np.asarray(base_position, dtype=np.float64)
            w, x, y, z = [float(value) for value in base_rotation]
            rotation = np.asarray(
                [
                    [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                    [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                    [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
                ],
                dtype=np.float64,
            )
            link_centers = []
            prefix = self.robot.config.prim_path.rstrip("/") + "/"
            for prim_path, rigid_body in self.robot._rigid_body_map.items():
                if not prim_path.startswith(prefix):
                    continue
                try:
                    link_position, _ = rigid_body.get_pose()
                    center = rotation.T @ (
                        np.asarray(link_position, dtype=np.float64) - base_position
                    )
                except Exception:
                    continue
                if np.isfinite(center).all():
                    link_centers.append(
                        {
                            "name": prim_path[len(prefix) :],
                            "center_base": [float(value) for value in center],
                        }
                    )
            return {
                "depth_height": int(sample.shape[0]),
                "depth_width": int(sample.shape[1]),
                "depth_row_stride": row_stride,
                "depth_column_stride": column_stride,
                "depth_values": sample.reshape(-1).astype(float).tolist(),
                "obstacle_targets_base": self.obstacle_targets,
                "robot_link_centers_base": link_centers,
                "support_plane_world_z": float(self.robot.get_ankle_height()) + 0.10,
            }

        def _update_command(self, *, state_only: bool = False) -> None:
            from internvla_go2_controller.runtime import get_execution_identity

            identity = get_execution_identity()
            episode_id = identity.episode_id
            reset_generation = identity.reset_generation
            sequence_id = identity.sequence_id
            agent_stop = identity.stop
            if state_only:
                (
                    episode_id,
                    reset_generation,
                    sequence_id,
                    agent_stop,
                ) = _continuous_state_only_identity(
                    identity, self.bootstrap_generation
                )
            (
                pose,
                linear,
                angular,
                fallen,
                nan_detected,
                physical_collision,
                maximum_collision_force,
                collision_pairs,
            ) = self._state()
            request: dict[str, Any] = {
                "schema_version": 1,
                "operation": "update",
                "episode_id": episode_id,
                "reset_generation": reset_generation,
                "sequence_id": sequence_id,
                "pose_wxyz": pose,
                "linear_velocity": linear,
                "angular_velocity": angular,
                "fallen": fallen,
                "nan_detected": nan_detected,
                "physical_collision": physical_collision,
                "maximum_collision_force": maximum_collision_force,
                "collision_pairs": collision_pairs,
                "agent_stop": agent_stop,
                "state_only": state_only,
                "obstacle_scenario": self.obstacles.scenario,
            }
            request.update(self._sample_depth())
            try:
                response = self.ipc.exchange(request)
                self.desired = (
                    float(response["linear_x"]),
                    float(response["angular_z"]),
                )
                self.emergency_stop = bool(response.get("emergency_stop", False))
            except Exception:
                self.desired = (0.0, 0.0)
                self.emergency_stop = True
                if state_only:
                    raise
            self.control_update_index += 1

        def publish_state_only(self) -> None:
            self.desired = (0.0, 0.0)
            self.emergency_stop = True
            position, _ = self.robot.articulation.unwrap().get_world_pose()
            settled_height = float(position[2])
            if math.isfinite(settled_height):
                # State-only samples span the official warm-up.  Refreshing the
                # target here freezes the final settled height when motion
                # begins, while adapting independently after every reset.
                self.nominal_base_height = settled_height
            self._update_command(state_only=True)

        def _apply_root_velocity(self, linear_x: float, angular_z: float) -> None:
            articulation = self.robot.articulation.unwrap()
            position, rotation = articulation.get_world_pose()
            position = np.asarray(position, dtype=np.float64)
            rotation = np.asarray(rotation, dtype=np.float64)
            yaw = math.atan2(
                2.0 * (rotation[0] * rotation[3] + rotation[1] * rotation[2]),
                1.0 - 2.0 * (rotation[2] ** 2 + rotation[3] ** 2),
            )
            if self.nominal_base_height is None:
                self.nominal_base_height = float(position[2])
            height_error = self.nominal_base_height - float(position[2])
            vertical = max(
                -self.maximum_vertical_speed,
                min(
                    self.maximum_vertical_speed,
                    self.height_stabilization_gain * height_error,
                ),
            )
            articulation.set_linear_velocity(
                np.asarray(
                    [math.cos(yaw) * linear_x, math.sin(yaw) * linear_x, vertical],
                    dtype=np.float64,
                )
            )
            w, x, y, z = [float(value) for value in rotation]
            body_z_world = np.asarray(
                [
                    2.0 * (x * z + w * y),
                    2.0 * (y * z - w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
                dtype=np.float64,
            )
            # cross(body_z_world, world_z) is the shortest world-frame
            # angular correction that brings the base vertical axis upright.
            tilt_correction = self.tilt_stabilization_gain * np.asarray(
                [body_z_world[1], -body_z_world[0]], dtype=np.float64
            )
            correction_norm = float(np.linalg.norm(tilt_correction))
            if correction_norm > self.maximum_tilt_correction:
                tilt_correction *= self.maximum_tilt_correction / correction_norm
            articulation.set_angular_velocity(
                np.asarray(
                    [tilt_correction[0], tilt_correction[1], angular_z],
                    dtype=np.float64,
                )
            )

        def _apply_t5_planar_root_pose(
            self, linear_x: float, angular_z: float
        ) -> None:
            """Apply the T5-only 20 Hz navigation kinematic deviation."""

            articulation = self.robot.articulation.unwrap()
            if self.planar_position is None or self.planar_rotation is None:
                position, rotation = articulation.get_world_pose()
                self.planar_position = np.asarray(position, dtype=np.float64)
                self.planar_rotation = np.asarray(rotation, dtype=np.float64)
            next_position, next_rotation = _integrate_t5_planar_pose(
                self.planar_position,
                self.planar_rotation,
                linear_x,
                angular_z,
                self.dt,
            )
            self.planar_position = next_position
            self.planar_rotation = next_rotation
            # Prevent PhysX from adding a second displacement. The live root
            # still carries the footprint, collision bodies and LiDAR frame.
            articulation.set_linear_velocity(np.zeros(3, dtype=np.float64))
            articulation.set_angular_velocity(np.zeros(3, dtype=np.float64))
            articulation.set_world_pose(next_position, next_rotation)

        def action_to_control(self, action: list[Any] | np.ndarray) -> ArticulationAction:
            if len(action) != 3:
                raise ValueError("continuous trigger must contain three elements")
            if self.window_finished:
                self.step_in_window = 0
                self.window_finished = False
                self.window_started_wall = time.monotonic()
            self.obstacle_targets = self.obstacles.update(self.dt)
            if self.step_in_window % self.steps_per_control == 0:
                self._update_command()
                limited = self.limiter.step(
                    self.desired[0],
                    self.desired[1],
                    self.control_dt,
                    emergency_stop=self.emergency_stop,
                )
                self.applied = (limited.linear_x, limited.angular_z)
            if self.motion_profile == "t5_completion_planar_root_velocity":
                self._apply_t5_planar_root_pose(self.applied[0], self.applied[1])
            else:
                self._apply_root_velocity(self.applied[0], self.applied[1])
            self.step_in_window += 1
            _advance_t5_sim_clock(self.dt)
            target_wall = self.window_started_wall + self.step_in_window * self.dt
            remaining = target_wall - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            return self.joint_subset.make_articulation_action(
                joint_positions=CONTINUOUS_STAND_JOINT_POSITIONS,
                joint_velocities=None,
            )

        def get_obs(self) -> dict[str, bool]:
            finished = self.step_in_window >= self.steps_per_window
            if finished:
                self.window_finished = True
            return {"finished": finished}

    execution_mode = os.environ.get("INTERNVLA_GO2_EXECUTION_MODE", "flash")
    if execution_mode not in {"flash", "continuous"}:
        raise RuntimeError("INTERNVLA_GO2_EXECUTION_MODE must be flash or continuous")
    continuous_reset_generation = -1

    @BaseRobot.register("VLNGo2Robot")
    class VLNGo2Robot(AliengoRobot):
        def __init__(self, config: Any, scene: Any):
            super().__init__(config, scene)
            self.current_action = None

        def set_up_to_scene(self, scene: Any) -> None:
            if execution_mode == "continuous":
                if os.environ.get("INTERNVLA_T3_PHASE", "") == "diagnostics":
                    _author_diagnostic_course()
                # Contact.create applies PhysxContactReportAPI to each parent
                # body. Author those USD changes before InternUtopia creates
                # the episode's PhysX tensor view; doing this in post_reset
                # invalidates the freshly-created articulation handle.
                from isaacsim.sensors.experimental.physics import Contact

                self._collision_sensor_authoring = []
                self._collision_sensor_bodies = list(COLLISION_SENSOR_BODIES)
                for body in COLLISION_SENSOR_BODIES:
                    sensor_path = self.config.prim_path + f"/{body}/t3_contact_sensor"
                    self._collision_sensor_authoring.append(
                        Contact.create(
                            sensor_path,
                            min_threshold=1.0,
                            max_threshold=100000.0,
                            radius=-1.0,
                        )
                    )
            super().set_up_to_scene(scene)

        def post_reset(self) -> None:
            nonlocal continuous_reset_generation
            super().post_reset()
            self._robot_feet = [
                self._rigid_body_map[self.config.prim_path + f"/{leg}_foot"]
                for leg in ("FL", "FR", "RL", "RR")
            ]
            if execution_mode == "continuous":
                from isaacsim.sensors.experimental.physics import ContactSensor

                for sensor in getattr(self, "_collision_sensors", []):
                    sensor.reset()
                self._collision_sensors = []
                authored = getattr(self, "_collision_sensor_authoring", [])
                if len(authored) != len(COLLISION_SENSOR_BODIES):
                    raise RuntimeError("Go2 collision sensors were not authored before physics setup")
                for sensor in authored:
                    runtime_sensor = ContactSensor(sensor)
                    runtime_sensor.add_raw_contact_data_to_frame()
                    self._collision_sensors.append(runtime_sensor)
                continuous = self.controllers["vln_dp_move_by_speed"]
                if _t5_continuation_bootstrap_identity_enabled():
                    # A replacement evaluator starts with no trustworthy local
                    # generation.  Keep state-only publishing disabled until the
                    # persistent DGX bridge identifies its active reset.
                    continuous.set_bootstrap_generation(-1)
                else:
                    continuous_reset_generation += 1
                    continuous.set_bootstrap_generation(continuous_reset_generation)
            for controller in self.controllers.values():
                reset = getattr(controller, "reset_runtime", None)
                if reset is not None:
                    reset()
            if (
                execution_mode == "continuous"
                and _t5_continuation_bootstrap_identity_enabled()
            ):
                self.controllers[
                    "vln_dp_move_by_speed"
                ].bind_persistent_bootstrap_generation()
            audit_path = os.environ.get("INTERNVLA_GO2_RUNTIME_AUDIT", "")
            if audit_path:
                position, rotation = self.articulation.get_world_pose()
                audit = {
                    "schema_version": 1,
                    "execution_mode": execution_mode,
                    "base_frame": "base_link",
                    "asset_base_prim": "base",
                    "initial_position": [float(value) for value in position],
                    "initial_rotation_wxyz": [float(value) for value in rotation],
                    "minimum_foot_height": self.get_ankle_height(),
                    "base_to_minimum_foot_height": (
                        float(position[2]) - self.get_ankle_height()
                    ),
                    "camera_prim": "base/internvla_camera",
                    "camera_translation_from_base": [0.2, 0.0, 0.2],
                    "diagnostic_course_authored": (
                        os.environ.get("INTERNVLA_T3_PHASE", "") == "diagnostics"
                    ),
                    "nominal_camera_world_height": float(position[2]) + 0.2,
                    "rigid_body_count": len(self._rigid_body_map),
                    "foot_collision_bodies": [
                        "FL_foot",
                        "FR_foot",
                        "RL_foot",
                        "RR_foot",
                    ],
                    "collision_sensor_bodies": list(
                        getattr(self, "_collision_sensor_bodies", [])
                    ),
                    "motion_api": (
                        (
                            "articulation_root_velocity_feedback_stabilized_physics_steps"
                            if self.controllers[
                                "vln_dp_move_by_speed"
                            ].motion_profile
                            == "physics_root_velocity"
                            else "t5_completion_planar_root_pose_20hz_held_joints"
                        )
                        if execution_mode == "continuous"
                        else "move_by_flash"
                    ),
                    "motion_profile": (
                        self.controllers[
                            "vln_dp_move_by_speed"
                        ].motion_profile
                        if execution_mode == "continuous"
                        else None
                    ),
                    "physics_hz": (
                        float(self.controllers["vln_dp_move_by_speed"].physics_hz)
                        if execution_mode == "continuous"
                        else None
                    ),
                    "control_hz": (
                        float(self.controllers["vln_dp_move_by_speed"].control_hz)
                        if execution_mode == "continuous"
                        else None
                    ),
                    "sensor_hz": (
                        float(self.controllers["vln_dp_move_by_speed"].sensor_hz)
                        if execution_mode == "continuous"
                        else None
                    ),
                    "navigation_only_motion_deviation": (
                        self.controllers[
                            "vln_dp_move_by_speed"
                        ].motion_profile
                        == "t5_completion_planar_root_velocity"
                        if execution_mode == "continuous"
                        else False
                    ),
                    "base_stabilization": (
                        {
                            "writes_world_pose": (
                                self.controllers[
                                    "vln_dp_move_by_speed"
                                ].motion_profile
                                == "t5_completion_planar_root_velocity"
                            ),
                            "tilt_gain_per_sec": float(
                                self.controllers[
                                    "vln_dp_move_by_speed"
                                ].tilt_stabilization_gain
                            ),
                            "maximum_tilt_correction_rad_per_sec": float(
                                self.controllers[
                                    "vln_dp_move_by_speed"
                                ].maximum_tilt_correction
                            ),
                            "height_gain_per_sec": float(
                                self.controllers[
                                    "vln_dp_move_by_speed"
                                ].height_stabilization_gain
                            ),
                            "maximum_vertical_speed_mps": float(
                                self.controllers[
                                    "vln_dp_move_by_speed"
                                ].maximum_vertical_speed
                            ),
                            "fall_roll_threshold_deg": 15.0,
                            "fall_pitch_threshold_deg": 35.0,
                        }
                        if execution_mode == "continuous"
                        else None
                    ),
                }
                target = Path(audit_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(audit, sort_keys=True) + "\n")

        def cleanup(self) -> None:
            for sensor in getattr(self, "_collision_sensors", []):
                sensor.reset()
            self._collision_sensors = []
            super().cleanup()

        def get_ankle_height(self) -> float:
            return float(min(foot.get_pose()[0][2] for foot in self._robot_feet))

        def apply_action(self, action: dict[str, Any]) -> None:
            self.current_action = action
            if execution_mode == "continuous" and "move_by_flash" in action:
                raise RuntimeError("move_by_flash is forbidden in continuous mode")
            if "stop" in action:
                _stop_root(self)
            else:
                super().apply_action(action)
            if "topdown_camera_500" in self.sensors:
                robot_pos = self.articulation.get_world_pose()[0]
                self.sensors["topdown_camera_500"].set_world_pose(
                    [robot_pos[0], robot_pos[1], robot_pos[2] + 0.75],
                    np.array([-0.70710678, 0.0, 0.0, 0.70710678]),
                )

    original_get_config = vln_default_config.get_config

    def get_go2_config(evaluator_cfg: Any) -> Any:
        if execution_mode == "continuous":
            evaluator_cfg.task.robot_flash = False
            evaluator_cfg.task.one_step_stand_still = True
        final_cfg = original_get_config(evaluator_cfg)
        wrapper = os.environ.get("INTERNVLA_GO2_WRAPPER_USD", "")
        if not wrapper or not os.path.isfile(wrapper):
            raise FileNotFoundError("INTERNVLA_GO2_WRAPPER_USD is missing")
        robot = final_cfg.task.robot
        robot.robot_settings["type"] = "VLNGo2Robot"
        robot.robot_settings["usd_path"] = wrapper
        robot.robot_settings["position"] = (0.0, 0.0, 0.42)
        for sensor in robot.sensors:
            settings = sensor.sensor_settings
            if settings.get("name") == "pano_camera_0":
                settings["prim_path"] = "base/internvla_camera"
        robot.sensors = [
            sensor
            for sensor in robot.sensors
            if sensor.sensor_settings.get("name") != "tp_pointcloud"
        ]
        stand = ControllerCfg(
            controller_settings={
                "name": "stand_still",
                "type": "Go2StandStillController",
            }
        )
        if execution_mode == "continuous":
            robot.controllers = [
                stand,
                ControllerCfg(
                    controller_settings={
                        "name": "vln_dp_move_by_speed",
                        "type": "Go2ContinuousController",
                    }
                ),
            ]
            final_cfg.task.robot_flash = False
            final_cfg.task.one_step_stand_still = True
        else:
            flash = VlnMoveByFlashControllerCfg(name="move_by_flash").model_dump()
            robot.controllers = [
                stand,
                ControllerCfg(controller_settings=flash),
            ]
        final_cfg.task.task_settings["fall_height_threshold"] = 0.15
        final_cfg.task.task_settings["robot_ankle_height"] = 0.03
        final_cfg.dataset.dataset_settings["robot_offset"] = np.array([0.0, 0.0, 0.42])
        return final_cfg

    vln_default_config.get_config = get_go2_config
    vln_default_config._internnav_go2_installed = True
