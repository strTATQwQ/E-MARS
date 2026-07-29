# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Run Unitree Go2 waypoint navigation inside an Isaac Sim warehouse USD scene.

This script uses the Isaac Lab Go2 flat locomotion policy exported as TorchScript.
It converts world-frame waypoint error into the Go2 velocity command observation
and lets the learned low-level policy produce joint actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import time
import traceback
import zlib
from pathlib import Path

from isaaclab.app import AppLauncher
import numpy as np

from low_speed_gait_servo import LowSpeedGaitServo, LowSpeedMotionGuard, LowSpeedYawServo
from ideal_kinematic_base import IdealKinematicBase

DEFAULT_ASSET_ROOT = "/home/song/isaacsim_assets/Assets/Isaac/6.0"
DEFAULT_WAREHOUSE_USD = (
    DEFAULT_ASSET_ROOT + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
)
DEFAULT_POLICY = (
    DEFAULT_ASSET_ROOT
    + "/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/exported/policy.pt"
)
DEFAULT_TRACE = "/home/song/isaac_projects/go2_warehouse_waypoint_trace.csv"
DEFAULT_WAYPOINTS = "0.6,0.0;1.2,0.1;1.8,0.1;2.2,0.0"


def parse_waypoints(text: str) -> list[tuple[float, float]]:
    waypoints: list[tuple[float, float]] = []
    for raw in text.split(";"):
        item = raw.strip()
        if not item:
            continue
        pieces = [piece.strip() for piece in item.split(",")]
        if len(pieces) != 2:
            raise argparse.ArgumentTypeError(f"Invalid waypoint '{item}', expected x,y")
        waypoints.append((float(pieces[0]), float(pieces[1])))
    if not waypoints:
        raise argparse.ArgumentTypeError("At least one waypoint is required")
    return waypoints


def parse_vec3(text: str) -> tuple[float, float, float]:
    pieces = [piece.strip() for piece in text.split(",")]
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError(f"Invalid vector '{text}', expected x,y,z")
    return (float(pieces[0]), float(pieces[1]), float(pieces[2]))


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


parser = argparse.ArgumentParser(description="Go2 waypoint navigation in a warehouse USD scene.")
parser.add_argument("--checkpoint", default=DEFAULT_POLICY, help="TorchScript policy path exported from RSL-RL.")
parser.add_argument("--warehouse_usd", default=DEFAULT_WAREHOUSE_USD, help="Warehouse USD path.")
parser.add_argument("--flat_ground", action="store_true", help="Use an empty plane instead of loading the warehouse USD.")
parser.add_argument("--waypoints", type=parse_waypoints, default=parse_waypoints(DEFAULT_WAYPOINTS))
parser.add_argument("--trace_csv", default=DEFAULT_TRACE, help="Path to write the robot trajectory CSV.")
parser.add_argument("--trace_every", type=int, default=1, help="Write one trace row every N steps; 0 disables trace rows.")
parser.add_argument("--perf_every", type=int, default=0, help="Log averaged loop timing every N steps; 0 disables.")
parser.add_argument("--render_interval", type=int, default=0, help="Render every N physics steps; 0 keeps the task default.")
parser.add_argument("--physics_dt", type=float, default=0.0, help="Physics timestep in seconds; 0 keeps the task default.")
parser.add_argument("--decimation", type=int, default=0, help="Physics steps per control step; 0 keeps the task default.")
parser.add_argument("--max_steps", type=int, default=1200, help="Maximum environment steps.")
parser.add_argument("--reached_radius", type=float, default=0.35, help="Waypoint success radius in meters.")
parser.add_argument("--linear_speed", type=float, default=0.55, help="Maximum XY command speed in m/s.")
parser.add_argument(
    "--min_stable_linear_command",
    type=float,
    default=0.45,
    help="Map nonzero low-speed ROS commands to this locomotion-policy gait floor; 0 disables.",
)
parser.add_argument(
    "--low_speed_gait_servo",
    action="store_true",
    help="Use measured-speed hysteresis to pulse the locomotion policy for <=0.20 m/s requests.",
)
parser.add_argument("--low_speed_gait_command", type=float, default=0.45)
parser.add_argument("--low_speed_gait_on_below", type=float, default=0.07)
parser.add_argument("--low_speed_gait_off_above", type=float, default=0.15)
parser.add_argument("--low_speed_actual_hard_limit", type=float, default=0.20)
parser.add_argument("--low_speed_total_hard_limit", type=float, default=0.20)
parser.add_argument("--low_speed_velocity_filter_alpha", type=float, default=0.12)
parser.add_argument(
    "--low_speed_gait_coast_requested",
    action="store_true",
    help="Return to the bounded ROS request between gait pulses instead of dropping policy input to zero.",
)
parser.add_argument("--low_speed_yaw_servo", action="store_true")
parser.add_argument("--low_speed_motion_guard", action="store_true")
parser.add_argument("--low_speed_guard_speed_trigger", type=float, default=0.18)
parser.add_argument("--low_speed_guard_yaw_trigger", type=float, default=0.20)
parser.add_argument("--low_speed_guard_speed_release", type=float, default=0.08)
parser.add_argument("--low_speed_guard_yaw_release", type=float, default=0.08)
parser.add_argument("--low_speed_guard_release_steps", type=int, default=25)
parser.add_argument("--low_speed_yaw_gait_command", type=float, default=0.30)
parser.add_argument("--low_speed_yaw_on_below", type=float, default=0.03)
parser.add_argument("--low_speed_yaw_off_above", type=float, default=0.12)
parser.add_argument("--low_speed_yaw_actual_hard_limit", type=float, default=0.24)
parser.add_argument("--low_speed_yaw_filter_alpha", type=float, default=0.12)
parser.add_argument("--low_speed_yaw_pulse_steps", type=int, default=3)
parser.add_argument("--low_speed_yaw_positive_pulse_steps", type=int, default=12)
parser.add_argument("--low_speed_yaw_cooldown_steps", type=int, default=8)
parser.add_argument("--yaw_kp", type=float, default=1.6, help="Yaw-rate proportional gain.")
parser.add_argument("--max_yaw_rate", type=float, default=1.2, help="Maximum yaw-rate command in rad/s.")
parser.add_argument(
    "--min_stable_yaw_command",
    type=float,
    default=0.0,
    help="Map nonzero low-rate ROS yaw commands to this locomotion-policy yaw floor; 0 disables.",
)
parser.add_argument("--start_x", type=float, default=0.0)
parser.add_argument("--start_y", type=float, default=0.0)
parser.add_argument("--start_yaw", type=float, default=0.0)
parser.add_argument("--save_stage", default="", help="Optional path to save the composed USD stage after setup.")
parser.add_argument("--real_time", action="store_true", help="Throttle simulation to roughly real time.")
parser.add_argument("--hold_open", action="store_true", help="Keep the app open after navigation finishes.")
parser.add_argument("--loop_waypoints", action="store_true", help="Repeat the waypoint route until max_steps or stop.")
parser.add_argument("--loop_reset", action="store_true", help="Reset the robot to the start pose after each waypoint loop.")
parser.add_argument("--hold_seconds", type=float, default=0.0, help="Keep the app open for this many seconds after navigation finishes.")
parser.add_argument(
    "--control_mode",
    choices=("waypoint", "ros_twist"),
    default="waypoint",
    help="Use built-in waypoint navigation or realtime Twist commands from the ROS UDP bridge.",
)
parser.add_argument("--cmd_udp_host", default="127.0.0.1", help="UDP bind host for ros_twist control.")
parser.add_argument("--cmd_udp_port", type=int, default=15002, help="UDP bind port for ros_twist control.")
parser.add_argument("--cmd_timeout", type=float, default=0.45, help="Stop if no Twist command arrives for this many seconds.")
parser.add_argument(
    "--benchmark_udp_host",
    default="127.0.0.1",
    help="UDP host for benchmark telemetry packets; empty disables telemetry.",
)
parser.add_argument("--benchmark_udp_port", type=int, default=15010, help="UDP port for benchmark telemetry packets.")
parser.add_argument("--benchmark_publish_hz", type=float, default=20.0, help="Benchmark telemetry publish rate.")
parser.add_argument(
    "--camera_udp_host",
    default="127.0.0.1",
    help="UDP host for front camera render-product packets; empty disables camera streaming.",
)
parser.add_argument("--camera_udp_port", type=int, default=15012, help="UDP port for front camera packets.")
parser.add_argument("--camera_publish_hz", type=float, default=5.0, help="Front camera publish rate.")
parser.add_argument("--camera_width", type=int, default=256, help="Front camera render width.")
parser.add_argument("--camera_height", type=int, default=144, help="Front camera render height.")
parser.add_argument("--disable_camera_stream", action="store_true", help="Do not attach or stream the Isaac camera sensor.")
parser.add_argument(
    "--third_person_record_dir",
    default="",
    help="Optional directory for a fixed third-person RGB PNG sequence and timeline.",
)
parser.add_argument("--third_person_record_fps", type=float, default=25.0)
parser.add_argument("--third_person_record_duration_sec", type=float, default=0.0)
parser.add_argument("--third_person_camera_width", type=int, default=960)
parser.add_argument("--third_person_camera_height", type=int, default=540)
parser.add_argument(
    "--third_person_camera_eye",
    type=parse_vec3,
    default=parse_vec3("2.6,-3.2,1.45"),
)
parser.add_argument(
    "--third_person_camera_target",
    type=parse_vec3,
    default=parse_vec3("1.1,0.0,0.35"),
)
parser.add_argument(
    "--control_udp_host",
    default="127.0.0.1",
    help="UDP bind host for benchmark control packets; empty disables control.",
)
parser.add_argument("--control_udp_port", type=int, default=15011, help="UDP port for benchmark control packets.")
parser.add_argument(
    "--disable_base_contact_termination",
    action="store_true",
    help="Do not reset the env when IsaacLab's base-contact termination fires.",
)
parser.add_argument(
    "--ideal_kinematic_base",
    action="store_true",
    help="Isaac development only: integrate safe body commands without using the learned locomotion policy.",
)
parser.add_argument("--ideal_root_height", type=float, default=0.40)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
import torch
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics


import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import read_file
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg_PLAY
from isaaclab_tasks.utils.hydra import resolve_presets


def add_waypoint_markers(waypoints: list[tuple[float, float]]) -> None:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    root = UsdGeom.Xform.Define(stage, "/World/Waypoints")
    root.GetPrim().SetActive(True)
    colors = [(0.0, 0.8, 1.0), (1.0, 0.65, 0.0), (0.25, 1.0, 0.25), (1.0, 0.2, 0.2)]
    for idx, (x_pos, y_pos) in enumerate(waypoints):
        sphere = UsdGeom.Sphere.Define(stage, f"/World/Waypoints/wp_{idx:02d}")
        sphere.CreateRadiusAttr(0.16)
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*colors[idx % len(colors)])])
        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(x_pos, y_pos, 0.18))


def sync_benchmark_visuals(
    objects: list[dict],
    obstacles: list[dict],
    dynamic_obstacles: list[dict] | None = None,
    *,
    scene_brightness: float = 1.0,
    background_usd: str = "",
    background_pose: list[float] | None = None,
    background_scale: list[float] | None = None,
    background_visual_only: bool = True,
) -> int:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return 0
    root_path = "/World/BenchmarkVisuals"
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)
    if background_usd and os.path.isfile(background_usd):
        background = stage.DefinePrim(f"{root_path}/Mp3dBackground", "Xform")
        background.GetReferences().AddReference(background_usd)
        pose = list(background_pose or [0.0, 0.0, 0.0])
        scale = list(background_scale or [1.0, 1.0, 1.0])
        if len(pose) >= 3 and len(scale) >= 3:
            common_xform = UsdGeom.XformCommonAPI(background)
            common_xform.SetTranslate(Gf.Vec3d(*(float(value) for value in pose[:3])))
            common_xform.SetScale(Gf.Vec3f(*(float(value) for value in scale[:3])))
        if background_visual_only:
            for prim in Usd.PrimRange(background):
                collision = UsdPhysics.CollisionAPI(prim)
                if collision:
                    collision.CreateCollisionEnabledAttr(False)
    dome = UsdLux.DomeLight.Define(stage, f"{root_path}/VisualDomeLight")
    dome.CreateIntensityAttr(1200.0 * max(0.25, min(1.5, float(scene_brightness))))
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    count = 0
    dynamic_visuals: list[dict] = []
    for value in dynamic_obstacles or []:
        item = dict(value)
        path_points = list(item.get("path") or [])
        if path_points:
            item["pose"] = [float(path_points[0][0]), float(path_points[0][1]), 0.9]
        radius_value = float(item.get("radius", 0.45))
        item.setdefault("size", [2.0 * radius_value, 2.0 * radius_value, 1.8])
        item["dynamic"] = True
        dynamic_visuals.append(item)
    obstacle_visuals = list(obstacles) + dynamic_visuals
    for index, item in enumerate(list(objects) + obstacle_visuals):
        pose = list(item.get("pose") or [0.0, 0.0, 0.5])
        if len(pose) < 3:
            continue
        object_id = "".join(ch if ch.isalnum() else "_" for ch in str(item.get("id") or f"object_{index}"))
        object_class = str(item.get("class") or "obstacle").lower()
        path = f"{root_path}/{object_id}"
        color_name = str(item.get("color") or ("gray" if item in obstacle_visuals else "white")).lower()
        color = {
            "red": (0.9, 0.04, 0.03),
            "blue": (0.03, 0.15, 0.9),
            "green": (0.03, 0.7, 0.12),
            "yellow": (0.95, 0.75, 0.02),
            "orange": (0.95, 0.28, 0.02),
            "purple": (0.55, 0.08, 0.75),
            "black": (0.03, 0.03, 0.03),
            "silver": (0.65, 0.68, 0.72),
            "gray": (0.35, 0.38, 0.42),
            "white": (0.9, 0.9, 0.9),
        }.get(color_name, (0.8, 0.8, 0.8))
        brightness = max(0.2, min(1.5, float(item.get("brightness", 1.0))))
        color = tuple(max(0.0, min(1.0, component * brightness)) for component in color)
        radius = max(0.12, float(item.get("radius", 0.35)))
        size = list(item.get("size") or [])
        usd_path = str(item.get("usd_path") or "")
        if object_class == "fire_hydrant":
            root = UsdGeom.Xform.Define(stage, path)
            UsdGeom.Xformable(root.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2]))
            )
            central = UsdGeom.Cylinder.Define(stage, f"{path}/central")
            central.CreateRadiusAttr(0.24)
            central.CreateHeightAttr(0.85)
            central.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdGeom.Xformable(central.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.55))
            cap = UsdGeom.Sphere.Define(stage, f"{path}/cap")
            cap.CreateRadiusAttr(0.32)
            cap.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdGeom.Xformable(cap.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.02))
            for nozzle_index, y_pos in enumerate((-0.34, 0.34)):
                nozzle = UsdGeom.Cube.Define(stage, f"{path}/nozzle_{nozzle_index}")
                nozzle.CreateSizeAttr(1.0)
                nozzle.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                nozzle_xform = UsdGeom.Xformable(nozzle.GetPrim())
                nozzle_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, y_pos, 0.72))
                nozzle_xform.AddScaleOp().Set(Gf.Vec3d(0.22, 0.22, 0.18))
            count += 1
            continue
        if object_class == "fire_extinguisher":
            root = UsdGeom.Xform.Define(stage, path)
            UsdGeom.Xformable(root.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2]) - 0.4)
            )
            bottle = UsdGeom.Cylinder.Define(stage, f"{path}/red_pressure_bottle")
            bottle.CreateRadiusAttr(0.19)
            bottle.CreateHeightAttr(0.72)
            bottle.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdGeom.Xformable(bottle.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.42))
            shoulder = UsdGeom.Sphere.Define(stage, f"{path}/rounded_shoulder")
            shoulder.CreateRadiusAttr(0.19)
            shoulder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            shoulder_xform = UsdGeom.Xformable(shoulder.GetPrim())
            shoulder_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.75))
            shoulder_xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 0.55))
            valve = UsdGeom.Cylinder.Define(stage, f"{path}/black_valve")
            valve.CreateRadiusAttr(0.065)
            valve.CreateHeightAttr(0.18)
            valve.CreateDisplayColorAttr([Gf.Vec3f(0.03, 0.03, 0.03)])
            UsdGeom.Xformable(valve.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.91))
            handle = UsdGeom.Cube.Define(stage, f"{path}/black_handle")
            handle.CreateSizeAttr(1.0)
            handle.CreateDisplayColorAttr([Gf.Vec3f(0.03, 0.03, 0.03)])
            handle_xform = UsdGeom.Xformable(handle.GetPrim())
            handle_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.08, 1.04))
            handle_xform.AddScaleOp().Set(Gf.Vec3d(0.08, 0.26, 0.035))
            hose = UsdGeom.Cube.Define(stage, f"{path}/black_hose")
            hose.CreateSizeAttr(1.0)
            hose.CreateDisplayColorAttr([Gf.Vec3f(0.02, 0.02, 0.02)])
            hose_xform = UsdGeom.Xformable(hose.GetPrim())
            hose_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.23, 0.72))
            hose_xform.AddScaleOp().Set(Gf.Vec3d(0.035, 0.035, 0.42))
            label = UsdGeom.Cube.Define(stage, f"{path}/white_label")
            label.CreateSizeAttr(1.0)
            label.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.92, 0.88)])
            label_xform = UsdGeom.Xformable(label.GetPrim())
            label_xform.AddTranslateOp().Set(Gf.Vec3d(-0.18, 0.0, 0.48))
            label_xform.AddScaleOp().Set(Gf.Vec3d(0.015, 0.12, 0.18))
            count += 1
            continue
        if usd_path and os.path.isfile(usd_path):
            prim = stage.DefinePrim(path, "Xform")
            prim.GetReferences().AddReference(usd_path)
            reference_scale = list(item.get("scale") or [1.0, 1.0, 1.0])
            common_xform = UsdGeom.XformCommonAPI(prim)
            common_xform.SetTranslate(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
            common_xform.SetScale(Gf.Vec3f(*(float(value) for value in reference_scale[:3])))
            count += 1
            continue
        if object_class in {"cylinder", "hydrant", "human_dummy", "dynamic_obstacle"}:
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateRadiusAttr(max(0.12, radius * 0.45))
            geom.CreateHeightAttr(max(0.6, radius * 2.0))
            scale = (1.0, 1.0, 1.0)
        elif object_class in {"cone", "traffic_cone"}:
            geom = UsdGeom.Cone.Define(stage, path)
            geom.CreateRadiusAttr(max(0.15, radius * 0.65))
            geom.CreateHeightAttr(max(0.5, radius * 1.8))
            scale = (1.0, 1.0, 1.0)
        else:
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.0)
            if object_class in {"exit_sign", "sign"}:
                scale = (max(0.45, radius * 1.4), 0.10, max(0.35, radius * 0.8))
            elif len(size) >= 3:
                scale = tuple(max(0.1, float(value)) for value in size[:3])
            else:
                scale = (radius * 1.2, radius * 1.2, radius * 1.2)
        geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        xform = UsdGeom.Xformable(geom.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
        xform.AddScaleOp().Set(Gf.Vec3d(*scale))
        if item in obstacle_visuals:
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        count += 1
    return count


def update_benchmark_dynamic_visuals(dynamic_obstacles: list[dict], elapsed_sec: float) -> int:
    """Move simplified rendered dynamic obstacles along deterministic ping-pong paths."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return 0
    updated = 0
    for index, item in enumerate(dynamic_obstacles):
        path_points = list(item.get("path") or [])
        if len(path_points) < 2:
            continue
        p0, p1 = path_points[0], path_points[1]
        dx = float(p1[0]) - float(p0[0])
        dy = float(p1[1]) - float(p0[1])
        segment = max(1.0e-6, math.hypot(dx, dy))
        phase = (max(0.0, float(elapsed_sec)) * max(0.01, float(item.get("speed_mps", 0.2))) / segment) % 2.0
        fraction = phase if phase <= 1.0 else 2.0 - phase
        x_pos = float(p0[0]) + dx * fraction
        y_pos = float(p0[1]) + dy * fraction
        object_id = "".join(ch if ch.isalnum() else "_" for ch in str(item.get("id") or f"dynamic_{index}"))
        prim = stage.GetPrimAtPath(f"/World/BenchmarkVisuals/{object_id}")
        if not prim.IsValid():
            continue
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        translate = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        if translate is not None:
            translate.Set(Gf.Vec3d(x_pos, y_pos, 0.9))
            updated += 1
    return updated


def configure_env() -> UnitreeGo2FlatEnvCfg_PLAY:
    env_cfg = UnitreeGo2FlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 1.0
    if args_cli.physics_dt > 0.0:
        env_cfg.sim.dt = args_cli.physics_dt
    if args_cli.decimation > 0:
        env_cfg.decimation = args_cli.decimation
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = max(30.0, args_cli.max_steps * step_dt + 5.0)
    env_cfg.curriculum = None
    if args_cli.flat_ground:
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            env_spacing=1.0,
        )
    else:
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=os.path.abspath(args_cli.warehouse_usd),
            env_spacing=1.0,
        )
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.ranges.heading = None
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
    env_cfg.observations.policy.enable_corruption = False
    if not args_cli.disable_camera_stream and args_cli.camera_udp_host and args_cli.camera_udp_port > 0:
        env_cfg.scene.front_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
            update_period=1.0 / max(0.1, args_cli.camera_publish_hz),
            height=max(16, int(args_cli.camera_height)),
            width=max(16, int(args_cli.camera_width)),
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 20.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.35, 0.0, 0.20),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )
    if args_cli.third_person_record_dir:
        if args_cli.third_person_record_fps <= 0.0:
            raise ValueError("third-person recording FPS must be positive")
        if args_cli.third_person_record_duration_sec <= 0.0:
            raise ValueError("third-person recording duration must be positive")
        env_cfg.scene.third_person_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/third_person_camera",
            update_period=1.0 / args_cli.third_person_record_fps,
            height=max(16, int(args_cli.third_person_camera_height)),
            width=max(16, int(args_cli.third_person_camera_width)),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=24.0,
                clipping_range=(0.05, 30.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 1.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
        )
    if args_cli.disable_base_contact_termination:
        env_cfg.terminations.base_contact = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    for reward_name in (
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "lin_vel_z_l2",
        "ang_vel_xy_l2",
        "dof_torques_l2",
        "dof_acc_l2",
        "action_rate_l2",
        "feet_air_time",
        "flat_orientation_l2",
        "dof_pos_limits",
    ):
        if hasattr(env_cfg.rewards, reward_name):
            setattr(env_cfg.rewards, reward_name, None)
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (args_cli.start_x, args_cli.start_x),
        "y": (args_cli.start_y, args_cli.start_y),
        "yaw": (args_cli.start_yaw, args_cli.start_yaw),
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.sim.device = args_cli.device
    if args_cli.render_interval > 0:
        env_cfg.sim.render_interval = max(args_cli.render_interval, env_cfg.decimation)
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False
    return resolve_presets(env_cfg)


def compute_velocity_command(robot, target_xy: tuple[float, float]) -> tuple[torch.Tensor, dict[str, float]]:
    pos = robot.data.root_pos_w.torch[0]
    heading = float(robot.data.heading_w.torch[0].item())
    dx = target_xy[0] - float(pos[0].item())
    dy = target_xy[1] - float(pos[1].item())
    dist = math.hypot(dx, dy)
    target_heading = math.atan2(dy, dx) if dist > 1.0e-6 else heading
    heading_error = wrap_to_pi(target_heading - heading)

    speed = min(args_cli.linear_speed, max(0.08, dist * 0.7))
    if abs(heading_error) > 1.2:
        speed *= 0.25
    elif abs(heading_error) > 0.8:
        speed *= 0.55

    if dist < args_cli.reached_radius:
        speed = 0.0

    vx_w = speed * math.cos(target_heading)
    vy_w = speed * math.sin(target_heading)
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    vx_b = cos_h * vx_w + sin_h * vy_w
    vy_b = -sin_h * vx_w + cos_h * vy_w
    yaw_rate = max(-args_cli.max_yaw_rate, min(args_cli.max_yaw_rate, args_cli.yaw_kp * heading_error))
    if dist < args_cli.reached_radius:
        yaw_rate = 0.0

    cmd = torch.tensor([[vx_b, vy_b, yaw_rate]], dtype=torch.float32, device=robot.device)
    info = {
        "x": float(pos[0].item()),
        "y": float(pos[1].item()),
        "z": float(pos[2].item()),
        "heading": heading,
        "target_x": target_xy[0],
        "target_y": target_xy[1],
        "dist": dist,
        "heading_error": heading_error,
        "cmd_vx": vx_b,
        "cmd_vy": vy_b,
        "cmd_yaw": yaw_rate,
    }
    return cmd, info


class UdpTwistReceiver:
    """Receive velocity commands from the local ROS-to-UDP bridge."""

    def __init__(self, host: str, port: int, timeout_s: float, device: str):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.device = device
        self.last_cmd = (0.0, 0.0, 0.0)
        self.last_seq = -1
        self.last_source = "none"
        self.last_rx_time = 0.0
        self.cmd_tensor = torch.zeros((1, 3), dtype=torch.float32, device=device)
        self.gait_servo = (
            LowSpeedGaitServo(
                gait_command_mps=args_cli.low_speed_gait_command,
                on_below_mps=args_cli.low_speed_gait_on_below,
                off_above_mps=args_cli.low_speed_gait_off_above,
                hard_limit_mps=args_cli.low_speed_actual_hard_limit,
                total_hard_limit_mps=args_cli.low_speed_total_hard_limit,
                filter_alpha=args_cli.low_speed_velocity_filter_alpha,
                coast_at_requested=args_cli.low_speed_gait_coast_requested,
            )
            if args_cli.low_speed_gait_servo
            else None
        )
        self.yaw_servo = (
            LowSpeedYawServo(
                gait_command_radps=args_cli.low_speed_yaw_gait_command,
                on_below_radps=args_cli.low_speed_yaw_on_below,
                off_above_radps=args_cli.low_speed_yaw_off_above,
                hard_limit_radps=args_cli.low_speed_yaw_actual_hard_limit,
                filter_alpha=args_cli.low_speed_yaw_filter_alpha,
                pulse_steps=args_cli.low_speed_yaw_pulse_steps,
                positive_pulse_steps=args_cli.low_speed_yaw_positive_pulse_steps,
                cooldown_steps=args_cli.low_speed_yaw_cooldown_steps,
            )
            if args_cli.low_speed_yaw_servo
            else None
        )
        self.motion_guard = (
            LowSpeedMotionGuard(
                speed_trigger_mps=args_cli.low_speed_guard_speed_trigger,
                yaw_trigger_radps=args_cli.low_speed_guard_yaw_trigger,
                speed_release_mps=args_cli.low_speed_guard_speed_release,
                yaw_release_radps=args_cli.low_speed_guard_yaw_release,
                release_steps=args_cli.low_speed_guard_release_steps,
            )
            if args_cli.low_speed_motion_guard
            else None
        )
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

    def close(self) -> None:
        self.sock.close()

    def reset_control_state(self) -> None:
        self.last_cmd = (0.0, 0.0, 0.0)
        self.last_seq = -1
        self.last_source = "reset"
        self.last_rx_time = 0.0
        self.cmd_tensor.zero_()
        if self.gait_servo is not None:
            self.gait_servo.reset()
        if self.yaw_servo is not None:
            self.yaw_servo.reset()
        if self.motion_guard is not None:
            self.motion_guard.reset()

    def _drain_socket(self) -> None:
        while True:
            try:
                payload, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                return
            try:
                data = json.loads(payload.decode("utf-8"))
                vx = float(data.get("vx", 0.0))
                vy = float(data.get("vy", 0.0))
                wz = float(data.get("wz", 0.0))
                self.last_cmd = (vx, vy, wz)
                self.last_seq = int(data.get("seq", self.last_seq + 1))
                self.last_source = f"{addr[0]}:{addr[1]}"
                self.last_rx_time = time.time()
            except Exception as exc:
                print(f"CMD_UDP_BAD_PACKET {type(exc).__name__}: {exc!r}", flush=True)

    def command(
        self,
        actual_velocity_xy: tuple[float, float] = (0.0, 0.0),
        actual_yaw_rate: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        self._drain_socket()
        now = time.time()
        age = now - self.last_rx_time if self.last_rx_time > 0.0 else float("inf")
        stale = age > self.timeout_s
        requested_vx, requested_vy, requested_wz = (0.0, 0.0, 0.0) if stale else self.last_cmd
        vx, vy, wz = requested_vx, requested_vy, requested_wz
        vx = max(-args_cli.linear_speed, min(args_cli.linear_speed, vx))
        vy = max(-args_cli.linear_speed, min(args_cli.linear_speed, vy))
        wz = max(-args_cli.max_yaw_rate, min(args_cli.max_yaw_rate, wz))
        xy_speed = math.hypot(vx, vy)
        gait_floor = min(max(0.0, args_cli.min_stable_linear_command), args_cli.linear_speed)
        if 0.0 < xy_speed < gait_floor:
            scale = gait_floor / xy_speed
            vx *= scale
            vy *= scale
        servo_active = False
        servo_reason = "disabled"
        actual_vx, actual_vy = actual_velocity_xy
        actual_speed = math.hypot(actual_vx, actual_vy)
        actual_along = 0.0
        actual_lateral = 0.0
        effective_along_hard = 0.0
        filtered_speed = 0.0
        filtered_along = 0.0
        filtered_lateral = 0.0
        if self.gait_servo is not None:
            servo = self.gait_servo.update(vx, vy, actual_vx, actual_vy)
            vx, vy = servo.policy_vx, servo.policy_vy
            servo_active = servo.active
            servo_reason = servo.reason
            actual_speed = servo.actual_speed_mps
            actual_along = servo.actual_along_request_mps
            actual_lateral = servo.actual_lateral_request_mps
            effective_along_hard = servo.effective_along_hard_limit_mps
            filtered_speed = servo.filtered_speed_mps
            filtered_along = servo.filtered_along_request_mps
            filtered_lateral = servo.filtered_lateral_request_mps
        yaw_floor = min(max(0.0, args_cli.min_stable_yaw_command), args_cli.max_yaw_rate)
        if xy_speed <= 1e-4 and 0.0 < abs(wz) < yaw_floor:
            wz = math.copysign(yaw_floor, wz)
        yaw_servo_active = False
        yaw_servo_reason = "disabled"
        filtered_yaw_rate = 0.0
        effective_yaw_hard = 0.0
        if self.yaw_servo is not None:
            yaw_servo = self.yaw_servo.update(wz, actual_yaw_rate)
            wz = yaw_servo.policy_wz
            yaw_servo_active = yaw_servo.active
            yaw_servo_reason = yaw_servo.reason
            filtered_yaw_rate = yaw_servo.filtered_wz
            effective_yaw_hard = yaw_servo.effective_hard_limit
        guard_active = False
        guard_reason = "disabled"
        guard_stable_steps = 0
        if self.motion_guard is not None:
            guard = self.motion_guard.update(actual_speed, actual_yaw_rate)
            guard_active = guard.active
            guard_reason = guard.reason
            guard_stable_steps = guard.stable_steps
            if guard_active:
                vx = 0.0
                vy = 0.0
                wz = 0.0
        self.cmd_tensor[0, 0] = vx
        self.cmd_tensor[0, 1] = vy
        self.cmd_tensor[0, 2] = wz
        info = {
            "x": math.nan,
            "y": math.nan,
            "z": math.nan,
            "heading": math.nan,
            "target_x": math.nan,
            "target_y": math.nan,
            "dist": math.nan,
            "heading_error": math.nan,
            "cmd_vx": vx,
            "cmd_vy": vy,
            "cmd_yaw": wz,
            "requested_vx": requested_vx,
            "requested_vy": requested_vy,
            "requested_yaw": requested_wz,
            "gait_floor": gait_floor,
            "yaw_floor": yaw_floor,
            "low_speed_servo_enabled": 1.0 if self.gait_servo is not None else 0.0,
            "low_speed_servo_active": 1.0 if servo_active else 0.0,
            "low_speed_servo_reason": servo_reason,
            "low_speed_yaw_servo_enabled": 1.0 if self.yaw_servo is not None else 0.0,
            "low_speed_yaw_servo_active": 1.0 if yaw_servo_active else 0.0,
            "low_speed_yaw_servo_reason": yaw_servo_reason,
            "actual_yaw_rate": actual_yaw_rate,
            "filtered_yaw_rate": filtered_yaw_rate,
            "effective_yaw_hard_limit": effective_yaw_hard,
            "low_speed_motion_guard_enabled": 1.0 if self.motion_guard is not None else 0.0,
            "low_speed_motion_guard_active": 1.0 if guard_active else 0.0,
            "low_speed_motion_guard_reason": guard_reason,
            "low_speed_motion_guard_stable_steps": guard_stable_steps,
            "actual_body_vx": actual_vx,
            "actual_body_vy": actual_vy,
            "actual_body_speed": actual_speed,
            "actual_along_request": actual_along,
            "actual_lateral_request": actual_lateral,
            "effective_along_hard_limit": effective_along_hard,
            "filtered_body_speed": filtered_speed,
            "filtered_along_request": filtered_along,
            "filtered_lateral_request": filtered_lateral,
            "cmd_age": age,
            "cmd_stale": 1.0 if stale else 0.0,
        }
        return self.cmd_tensor, info


def sample_robot_pose(
    robot,
    info: dict[str, float],
    ideal_state: IdealKinematicBase | None = None,
) -> dict[str, float]:
    if ideal_state is not None:
        sampled = dict(info)
        sampled.update(
            ideal_state.telemetry(
                vx_body=float(info.get("cmd_vx", 0.0)),
                vy_body=float(info.get("cmd_vy", 0.0)),
                wz=float(info.get("cmd_yaw", 0.0)),
            )
        )
        return sampled
    pos = robot.data.root_pos_w.torch[0].detach().cpu().tolist()
    heading = float(robot.data.heading_w.torch[0].detach().cpu().item())
    linear_velocity = robot.data.root_lin_vel_w.torch[0].detach().cpu().tolist()
    angular_velocity = robot.data.root_ang_vel_w.torch[0].detach().cpu().tolist()
    sampled = dict(info)
    sampled["x"] = float(pos[0])
    sampled["y"] = float(pos[1])
    sampled["z"] = float(pos[2])
    sampled["heading"] = heading
    sampled["linear_velocity"] = [float(value) for value in linear_velocity]
    sampled["angular_velocity"] = [float(value) for value in angular_velocity]
    return sampled


def apply_benchmark_reset_pose(robot, requested_pose) -> bool:
    return False


def write_ideal_kinematic_state(robot, state: IdealKinematicBase, command) -> None:
    half_yaw = 0.5 * state.yaw
    root_pose = torch.tensor(
        [[state.x, state.y, state.z, math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]],
        dtype=torch.float32,
        device=robot.device,
    )
    velocity = state.world_velocity(
        vx_body=float(command[0, 0].item()),
        vy_body=float(command[0, 1].item()),
        wz=float(command[0, 2].item()),
    )
    root_velocity = torch.tensor([velocity], dtype=torch.float32, device=robot.device)
    default_joint_pos = getattr(robot.data.default_joint_pos, "torch", robot.data.default_joint_pos).clone()
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_velocity)
    robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos))


class UdpBenchmarkTelemetry:
    """Send IsaacLab Go2 root pose telemetry to the ROS2 benchmark adapter."""

    def __init__(self, host: str, port: int, publish_hz: float, *, source: str = "isaaclab_go2"):
        self.host = host
        self.port = port
        self.period = 1.0 / max(0.1, publish_hz)
        self.last_send = 0.0
        self.seq = 0
        self.source = source
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        self.sock.close()

    def maybe_send(
        self,
        *,
        robot,
        info: dict[str, float],
        step: int,
        event: str = "pose",
        metadata: dict | None = None,
        ideal_state: IdealKinematicBase | None = None,
    ) -> None:
        now_wall = time.time()
        if event == "pose" and now_wall - self.last_send < self.period:
            return
        self.last_send = now_wall
        self.seq += 1
        sampled = sample_robot_pose(robot, info, ideal_state=ideal_state)
        payload = {
            "seq": self.seq,
            "timestamp": now_wall,
            "event": event,
            "source": self.source,
            "step": int(step),
            "pose": [sampled["x"], sampled["y"], sampled["heading"]],
            "z": sampled["z"],
            "linear_velocity": sampled["linear_velocity"],
            "angular_velocity": sampled["angular_velocity"],
            "cmd": {
                "vx": float(sampled.get("cmd_vx", 0.0)),
                "vy": float(sampled.get("cmd_vy", 0.0)),
                "wz": float(sampled.get("cmd_yaw", 0.0)),
                "age": float(sampled.get("cmd_age", 0.0)) if not math.isinf(float(sampled.get("cmd_age", 0.0))) else None,
                "stale": bool(sampled.get("cmd_stale", 0.0)),
                "requested_vx": float(sampled.get("requested_vx", sampled.get("cmd_vx", 0.0))),
                "requested_vy": float(sampled.get("requested_vy", sampled.get("cmd_vy", 0.0))),
                "requested_wz": float(sampled.get("requested_yaw", sampled.get("cmd_yaw", 0.0))),
                "gait_floor": float(sampled.get("gait_floor", 0.0)),
                "yaw_floor": float(sampled.get("yaw_floor", 0.0)),
                "low_speed_servo_enabled": bool(sampled.get("low_speed_servo_enabled", 0.0)),
                "low_speed_servo_active": bool(sampled.get("low_speed_servo_active", 0.0)),
                "low_speed_servo_reason": str(sampled.get("low_speed_servo_reason", "disabled")),
                "low_speed_yaw_servo_enabled": bool(sampled.get("low_speed_yaw_servo_enabled", 0.0)),
                "low_speed_yaw_servo_active": bool(sampled.get("low_speed_yaw_servo_active", 0.0)),
                "low_speed_yaw_servo_reason": str(sampled.get("low_speed_yaw_servo_reason", "disabled")),
                "actual_yaw_rate": float(sampled.get("actual_yaw_rate", 0.0)),
                "filtered_yaw_rate": float(sampled.get("filtered_yaw_rate", 0.0)),
                "effective_yaw_hard_limit": float(sampled.get("effective_yaw_hard_limit", 0.0)),
                "low_speed_motion_guard_enabled": bool(sampled.get("low_speed_motion_guard_enabled", 0.0)),
                "low_speed_motion_guard_active": bool(sampled.get("low_speed_motion_guard_active", 0.0)),
                "low_speed_motion_guard_reason": str(sampled.get("low_speed_motion_guard_reason", "disabled")),
                "low_speed_motion_guard_stable_steps": int(sampled.get("low_speed_motion_guard_stable_steps", 0)),
                "actual_body_vx": float(sampled.get("actual_body_vx", 0.0)),
                "actual_body_vy": float(sampled.get("actual_body_vy", 0.0)),
                "actual_body_speed": float(sampled.get("actual_body_speed", 0.0)),
                "actual_along_request": float(sampled.get("actual_along_request", 0.0)),
                "actual_lateral_request": float(sampled.get("actual_lateral_request", 0.0)),
                "effective_along_hard_limit": float(sampled.get("effective_along_hard_limit", 0.0)),
                "filtered_body_speed": float(sampled.get("filtered_body_speed", 0.0)),
                "filtered_along_request": float(sampled.get("filtered_along_request", 0.0)),
                "filtered_lateral_request": float(sampled.get("filtered_lateral_request", 0.0)),
                "ideal_kinematic_base": bool(args_cli.ideal_kinematic_base),
            },
        }
        if metadata:
            payload.update(
                {
                    "episode_id": str(metadata.get("episode_id", "")),
                    "task_id": str(metadata.get("task_id", "")),
                    "scene_id": str(metadata.get("scene_id", "")),
                    "reset_requested_at": metadata.get("timestamp"),
                }
            )
        self.sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), (self.host, self.port))


class UdpBenchmarkCameraTelemetry:
    """Send Isaac render-product RGB/depth frames to the ROS2 benchmark adapter."""

    MAGIC = "isaac_cam_v1"
    CHUNK_BYTES = 60000

    def __init__(self, host: str, port: int, publish_hz: float):
        self.host = host
        self.port = port
        self.period = 1.0 / max(0.1, publish_hz)
        self.last_send = 0.0
        self.last_error_log = 0.0
        self.seq = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        self.sock.close()

    @staticmethod
    def _to_numpy(value):
        tensor = getattr(value, "torch", value)
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    @classmethod
    def _rgb_bytes(cls, camera) -> tuple[bytes, int, int]:
        rgb = cls._to_numpy(camera.data.output["rgb"])
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.ndim != 3:
            raise ValueError(f"unexpected RGB shape {rgb.shape}")
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.float32, copy=False)
            if float(np.nanmax(rgb)) <= 1.5:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        else:
            rgb = np.ascontiguousarray(rgb)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        return rgb.tobytes(order="C"), width, height

    @classmethod
    def _depth_bytes(cls, camera) -> tuple[bytes, int, int]:
        depth = cls._to_numpy(camera.data.output["distance_to_image_plane"])
        if depth.ndim == 4:
            depth = depth[0]
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"unexpected depth shape {depth.shape}")
        depth = depth.astype(np.float32, copy=False)
        depth = np.nan_to_num(depth, nan=0.0, posinf=20.0, neginf=0.0)
        depth_mm = (np.clip(depth, 0.0, 20.0) * 1000.0).astype(np.uint16)
        height, width = int(depth_mm.shape[0]), int(depth_mm.shape[1])
        return np.ascontiguousarray(depth_mm).tobytes(order="C"), width, height

    def _send_frame(
        self,
        *,
        raw: bytes,
        kind: str,
        width: int,
        height: int,
        encoding: str,
        step: int,
        timestamp: float,
    ) -> None:
        encoded = zlib.compress(raw, level=1)
        chunks = max(1, math.ceil(len(encoded) / self.CHUNK_BYTES))
        for chunk_index in range(chunks):
            start = chunk_index * self.CHUNK_BYTES
            end = min(len(encoded), start + self.CHUNK_BYTES)
            header = {
                "magic": self.MAGIC,
                "seq": self.seq,
                "kind": kind,
                "width": width,
                "height": height,
                "encoding": encoding,
                "frame_id": "isaac_front_camera",
                "step": int(step),
                "chunk": chunk_index,
                "chunks": chunks,
                "timestamp": timestamp,
                "compression": "zlib",
                "raw_bytes": len(raw),
                "focal_length_mm": 24.0,
                "horizontal_aperture_mm": 20.955,
                "camera_translation_base_m": [0.35, 0.0, 0.20],
                "camera_rotation_base_xyzw": [-0.5, 0.5, -0.5, 0.5],
                "optical_convention": "ros",
            }
            packet = json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\n" + encoded[start:end]
            self.sock.sendto(packet, (self.host, self.port))

    def maybe_send(self, *, camera, step: int) -> None:
        now_wall = time.time()
        if now_wall - self.last_send < self.period:
            return
        self.last_send = now_wall
        try:
            rgb_raw, rgb_width, rgb_height = self._rgb_bytes(camera)
            depth_raw, depth_width, depth_height = self._depth_bytes(camera)
            self.seq += 1
            self._send_frame(
                raw=rgb_raw,
                kind="rgb8",
                width=rgb_width,
                height=rgb_height,
                encoding="rgb8",
                step=step,
                timestamp=now_wall,
            )
            self._send_frame(
                raw=depth_raw,
                kind="depth16",
                width=depth_width,
                height=depth_height,
                encoding="16UC1",
                step=step,
                timestamp=now_wall,
            )
        except Exception as exc:
            if now_wall - self.last_error_log > 2.0:
                self.last_error_log = now_wall
                print(f"CAMERA_TELEMETRY_EXCEPTION {type(exc).__name__}: {exc!r}", flush=True)


class UdpBenchmarkControlReceiver:
    """Receive reset/control events from the ROS2 benchmark adapter."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

    def close(self) -> None:
        self.sock.close()

    def drain(self) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        while True:
            try:
                payload, addr = self.sock.recvfrom(8192)
            except BlockingIOError:
                return events
            try:
                data = json.loads(payload.decode("utf-8"))
                if isinstance(data, dict):
                    data["_source"] = f"{addr[0]}:{addr[1]}"
                    events.append(data)
            except Exception as exc:
                print(f"BENCHMARK_CONTROL_BAD_PACKET {type(exc).__name__}: {exc!r}", flush=True)


def main() -> int:
    checkpoint = os.path.abspath(args_cli.checkpoint)
    warehouse_usd = os.path.abspath(args_cli.warehouse_usd)
    if not os.path.exists(checkpoint):
        print(f"POLICY_MISSING {checkpoint}")
        return 2
    if not os.path.exists(warehouse_usd):
        print(f"WAREHOUSE_MISSING {warehouse_usd}")
        return 2

    print(f"WAREHOUSE_USD {warehouse_usd}", flush=True)
    print(f"POLICY {checkpoint}", flush=True)
    print(f"CONTROL_MODE {args_cli.control_mode}", flush=True)
    print(f"LOCOMOTION_FIDELITY {'ideal_kinematic' if args_cli.ideal_kinematic_base else 'learned_policy'}", flush=True)
    if args_cli.ideal_kinematic_base and args_cli.control_mode != "ros_twist":
        print("IDEAL_KINEMATIC_REQUIRES_ROS_TWIST", flush=True)
        return 2
    print(
        f"BASE_CONTACT_TERMINATION_ENABLED {not args_cli.disable_base_contact_termination}",
        flush=True,
    )
    if args_cli.control_mode == "waypoint":
        print("WAYPOINTS " + ";".join(f"{x:.3f},{y:.3f}" for x, y in args_cli.waypoints), flush=True)

    print("LOAD_POLICY_BEGIN", flush=True)
    file = read_file(checkpoint)
    policy = torch.jit.load(file, map_location=args_cli.device)
    policy.eval()
    print("LOAD_POLICY_DONE", flush=True)

    print("CREATE_ENV_BEGIN", flush=True)
    try:
        env = ManagerBasedRLEnv(cfg=configure_env())
    except BaseException as exc:
        print(f"CREATE_ENV_EXCEPTION {type(exc).__name__}: {exc!r}", flush=True)
        return 2
    print("CREATE_ENV_DONE", flush=True)
    robot = env.scene["robot"]
    front_camera = None
    third_person_camera = None
    if not args_cli.disable_camera_stream and args_cli.camera_udp_host and args_cli.camera_udp_port > 0:
        try:
            front_camera = env.scene["front_camera"]
            print(
                f"CAMERA_READY name=front_camera width={args_cli.camera_width} "
                f"height={args_cli.camera_height} hz={args_cli.camera_publish_hz:.1f}",
                flush=True,
            )
        except Exception as exc:
            print(f"CAMERA_UNAVAILABLE {type(exc).__name__}: {exc!r}", flush=True)
    if args_cli.third_person_record_dir:
        third_person_camera = env.scene["third_person_camera"]
        print(
            "THIRD_PERSON_CAMERA_READY "
            f"width={args_cli.third_person_camera_width} "
            f"height={args_cli.third_person_camera_height} "
            f"fps={args_cli.third_person_record_fps:.3f} "
            f"eye={args_cli.third_person_camera_eye} "
            f"target={args_cli.third_person_camera_target}",
            flush=True,
        )
    if args_cli.control_mode == "waypoint":
        add_waypoint_markers(args_cli.waypoints)
        print("MARKERS_DONE", flush=True)

    if args_cli.save_stage:
        stage = omni.usd.get_context().get_stage()
        save_path = os.path.abspath(args_cli.save_stage)
        try:
            saved = bool(stage and stage.GetRootLayer().Export(save_path))
        except Exception as exc:
            saved = False
            print(f"SAVE_STAGE_EXCEPTION {type(exc).__name__}: {exc!r}", flush=True)
        print(f"SAVE_STAGE {save_path} {saved}", flush=True)

    trace_path = Path(args_cli.trace_csv)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_file = trace_path.open("w", newline="")
    writer = csv.DictWriter(
        trace_file,
        fieldnames=[
            "step",
            "waypoint_index",
            "x",
            "y",
            "z",
            "heading",
            "target_x",
            "target_y",
            "dist",
            "heading_error",
            "cmd_vx",
            "cmd_vy",
            "cmd_yaw",
            "cmd_age",
            "cmd_stale",
        ],
    )
    writer.writeheader()

    obs, _ = env.reset()
    third_person_root = None
    third_person_manifest = None
    third_person_frame_count = 0
    third_person_next_sim_sec = 0.0
    third_person_recording_completed = False
    if third_person_camera is not None:
        third_person_root = Path(args_cli.third_person_record_dir).expanduser().resolve()
        third_person_root.mkdir(parents=True, exist_ok=False)
        third_person_manifest = (third_person_root / "frames.jsonl").open("w", encoding="utf-8")
        eye = torch.tensor(
            [list(args_cli.third_person_camera_eye)], dtype=torch.float32, device=robot.device
        )
        target = torch.tensor(
            [list(args_cli.third_person_camera_target)], dtype=torch.float32, device=robot.device
        )
        third_person_camera.set_world_poses_from_view(eye, target)
        env.sim.render()
        third_person_next_sim_sec = 1.0 / args_cli.third_person_record_fps
    ideal_state = None
    if args_cli.ideal_kinematic_base:
        ideal_state = IdealKinematicBase(
            x=args_cli.start_x,
            y=args_cli.start_y,
            yaw=args_cli.start_yaw,
            z=args_cli.ideal_root_height,
        )
        zero_command = torch.zeros((1, 3), dtype=torch.float32, device=robot.device)
        write_ideal_kinematic_state(robot, ideal_state, zero_command)
    waypoint_index = 0
    reached_count = 0
    success = args_cli.control_mode == "ros_twist"
    twist_receiver = None
    if args_cli.control_mode == "ros_twist":
        twist_receiver = UdpTwistReceiver(args_cli.cmd_udp_host, args_cli.cmd_udp_port, args_cli.cmd_timeout, robot.device)
        print(
            f"TELEOP_READY udp={args_cli.cmd_udp_host}:{args_cli.cmd_udp_port} "
            f"timeout={args_cli.cmd_timeout:.2f} max_linear={args_cli.linear_speed:.2f} "
            f"min_stable_linear={args_cli.min_stable_linear_command:.2f} max_yaw={args_cli.max_yaw_rate:.2f} "
            f"min_stable_yaw={args_cli.min_stable_yaw_command:.2f}",
            flush=True,
        )
    benchmark_telemetry = None
    if args_cli.benchmark_udp_host and args_cli.benchmark_udp_port > 0:
        benchmark_telemetry = UdpBenchmarkTelemetry(
            args_cli.benchmark_udp_host,
            args_cli.benchmark_udp_port,
            args_cli.benchmark_publish_hz,
            source="isaaclab_go2_ideal_kinematic" if args_cli.ideal_kinematic_base else "isaaclab_go2",
        )
        print(
            f"BENCHMARK_TELEMETRY_READY udp={args_cli.benchmark_udp_host}:{args_cli.benchmark_udp_port} "
            f"hz={args_cli.benchmark_publish_hz:.1f}",
            flush=True,
        )
    benchmark_camera = None
    if front_camera is not None and args_cli.camera_udp_host and args_cli.camera_udp_port > 0:
        benchmark_camera = UdpBenchmarkCameraTelemetry(
            args_cli.camera_udp_host,
            args_cli.camera_udp_port,
            args_cli.camera_publish_hz,
        )
        print(
            f"CAMERA_TELEMETRY_READY udp={args_cli.camera_udp_host}:{args_cli.camera_udp_port} "
            f"hz={args_cli.camera_publish_hz:.1f}",
            flush=True,
        )
    benchmark_control = None
    if args_cli.control_udp_host and args_cli.control_udp_port > 0:
        benchmark_control = UdpBenchmarkControlReceiver(args_cli.control_udp_host, args_cli.control_udp_port)
        print(
            f"BENCHMARK_CONTROL_READY udp={args_cli.control_udp_host}:{args_cli.control_udp_port}",
            flush=True,
        )
    print("NAV_READY", flush=True)
    perf_acc = {"command": 0.0, "obs": 0.0, "policy": 0.0, "env_step": 0.0, "io": 0.0, "sleep": 0.0}
    perf_count = 0
    perf_last_wall = time.perf_counter()

    control_loop_steps = int(args_cli.max_steps)
    if args_cli.control_mode == "ros_twist" and args_cli.hold_open:
        control_loop_steps = max(control_loop_steps, 1_000_000_000)
    print(f"CONTROL_LOOP_STEPS {control_loop_steps}", flush=True)
    loop_failed = False
    active_dynamic_obstacles: list[dict] = []
    dynamic_episode_started = time.monotonic()
    try:
        for step in range(control_loop_steps):
            step_start = time.time()
            perf_t0 = time.perf_counter()
            if benchmark_control is not None:
                reset_requested = False
                for control_event in benchmark_control.drain():
                    event_name = str(control_event.get("event", ""))
                    if event_name != "reset":
                        print(
                            f"BENCHMARK_CONTROL_IGNORED step={step} event={event_name}",
                            flush=True,
                        )
                        continue
                    env.command_manager.get_command("base_velocity")[:] = 0.0
                    if twist_receiver is not None:
                        twist_receiver.reset_control_state()
                    obs, _ = env.reset()
                    visual_count = sync_benchmark_visuals(
                        list(control_event.get("objects") or []),
                        list(control_event.get("obstacles") or []),
                        list(control_event.get("dynamic_obstacles") or []),
                        scene_brightness=float(control_event.get("brightness") or 1.0),
                        background_usd=str(control_event.get("background_usd") or ""),
                        background_pose=list(control_event.get("background_pose") or [0.0, 0.0, 0.0]),
                        background_scale=list(control_event.get("background_scale") or [1.0, 1.0, 1.0]),
                        background_visual_only=bool(control_event.get("background_visual_only", True)),
                    )
                    active_dynamic_obstacles = [dict(value) for value in control_event.get("dynamic_obstacles") or []]
                    dynamic_episode_started = time.monotonic()
                    if ideal_state is not None:
                        ideal_state.reset(control_event.get("pose"))
                        write_ideal_kinematic_state(
                            robot,
                            ideal_state,
                            torch.zeros((1, 3), dtype=torch.float32, device=robot.device),
                        )
                        reset_pose_applied = True
                    else:
                        reset_pose_applied = apply_benchmark_reset_pose(robot, control_event.get("pose"))
                    reset_requested = True
                    task_id = str(control_event.get("task_id", ""))
                    scene_id = str(control_event.get("scene_id", ""))
                    print(
                        f"BENCHMARK_RESET step={step} task={task_id} scene={scene_id} "
                        f"pose_applied={reset_pose_applied} visuals={visual_count} pose={control_event.get('pose')}",
                        flush=True,
                    )
                    if benchmark_telemetry is not None:
                        benchmark_telemetry.maybe_send(
                            robot=robot,
                            info={"cmd_vx": 0.0, "cmd_vy": 0.0, "cmd_yaw": 0.0},
                            step=step,
                            event="reset",
                            metadata=control_event,
                            ideal_state=ideal_state,
                        )
                if reset_requested:
                    continue
            if active_dynamic_obstacles:
                update_benchmark_dynamic_visuals(
                    active_dynamic_obstacles,
                    time.monotonic() - dynamic_episode_started,
                )
            target = None
            if args_cli.control_mode == "waypoint":
                target = args_cli.waypoints[waypoint_index]
                cmd, info = compute_velocity_command(robot, target)
            else:
                assert twist_receiver is not None
                world_velocity = robot.data.root_lin_vel_w.torch[0].detach().cpu().tolist()
                heading = float(robot.data.heading_w.torch[0].detach().cpu().item())
                cos_heading = math.cos(heading)
                sin_heading = math.sin(heading)
                actual_body_velocity = (
                    cos_heading * float(world_velocity[0]) + sin_heading * float(world_velocity[1]),
                    -sin_heading * float(world_velocity[0]) + cos_heading * float(world_velocity[1]),
                )
                actual_yaw_rate = float(robot.data.root_ang_vel_w.torch[0, 2].detach().cpu().item())
                cmd, info = twist_receiver.command(actual_body_velocity, actual_yaw_rate)
            perf_t1 = time.perf_counter()

            env.command_manager.get_command("base_velocity")[:] = cmd
            if ideal_state is None:
                obs = env.observation_manager.compute(update_history=False)
            perf_t2 = time.perf_counter()

            if args_cli.control_mode == "waypoint" and info["dist"] < args_cli.reached_radius:
                assert target is not None
                print(
                    "WAYPOINT_REACHED "
                    f"index={waypoint_index} step={step} "
                    f"pos=({info['x']:.3f},{info['y']:.3f}) "
                    f"target=({target[0]:.3f},{target[1]:.3f}) dist={info['dist']:.3f}",
                    flush=True,
                )
                waypoint_index += 1
                reached_count = waypoint_index
                if waypoint_index >= len(args_cli.waypoints):
                    if args_cli.loop_waypoints:
                        print(f"WAYPOINT_LOOP_COMPLETE step={step}", flush=True)
                        waypoint_index = 0
                        reached_count = 0
                        if args_cli.loop_reset:
                            env.command_manager.get_command("base_velocity")[:] = 0.0
                            obs, _ = env.reset()
                            print(f"WAYPOINT_LOOP_RESET step={step}", flush=True)
                    else:
                        success = True
                        break
                continue

            if ideal_state is not None:
                ideal_state.integrate(
                    vx_body=float(cmd[0, 0].item()),
                    vy_body=float(cmd[0, 1].item()),
                    wz=float(cmd[0, 2].item()),
                    dt=env.step_dt,
                )
                action = torch.zeros((1, 12), dtype=torch.float32, device=robot.device)
            else:
                with torch.inference_mode():
                    action = policy(obs["policy"])
            perf_t3 = time.perf_counter()
            obs, _, terminated, truncated, _ = env.step(action)
            if ideal_state is not None:
                write_ideal_kinematic_state(robot, ideal_state, cmd)
            perf_t4 = time.perf_counter()
            if benchmark_telemetry is not None:
                benchmark_telemetry.maybe_send(
                    robot=robot,
                    info=info,
                    step=step,
                    ideal_state=ideal_state,
                )
            if benchmark_camera is not None:
                benchmark_camera.maybe_send(camera=front_camera, step=step)
            if third_person_camera is not None:
                sim_elapsed_sec = (step + 1) * float(env.step_dt)
                if sim_elapsed_sec + 1.0e-9 >= third_person_next_sim_sec:
                    rgb_raw, rgb_width, rgb_height = UdpBenchmarkCameraTelemetry._rgb_bytes(
                        third_person_camera
                    )
                    frame_name = f"third_person_{third_person_frame_count:06d}.png"
                    assert third_person_root is not None
                    assert third_person_manifest is not None
                    Image.frombytes("RGB", (rgb_width, rgb_height), rgb_raw).save(
                        third_person_root / frame_name,
                        format="PNG",
                    )
                    third_person_manifest.write(
                        json.dumps(
                            {
                                "frame_index": third_person_frame_count,
                                "path": frame_name,
                                "sim_time_sec": sim_elapsed_sec,
                                "physics_fidelity": "isaaclab_go2_learned_policy",
                                "ideal_kinematic_base": False,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    third_person_manifest.flush()
                    third_person_frame_count += 1
                    third_person_next_sim_sec = (
                        third_person_frame_count + 1
                    ) / args_cli.third_person_record_fps
                if sim_elapsed_sec + 1.0e-9 >= args_cli.third_person_record_duration_sec:
                    third_person_recording_completed = True
                    success = True
                    print(
                        "THIRD_PERSON_RECORD_COMPLETE "
                        f"sim_duration_sec={sim_elapsed_sec:.6f} "
                        f"frames={third_person_frame_count}",
                        flush=True,
                    )
                    break

            if args_cli.trace_every > 0 and step % args_cli.trace_every == 0:
                trace_info = info
                if args_cli.control_mode == "ros_twist" and math.isnan(trace_info["x"]):
                    trace_info = sample_robot_pose(robot, trace_info, ideal_state=ideal_state)
                row = {"step": step, "waypoint_index": waypoint_index, **trace_info}
                writer.writerow(row)
            if step % 50 == 0:
                if args_cli.control_mode == "waypoint":
                    assert target is not None
                    print(
                        "NAV_STEP "
                        f"step={step} wp={waypoint_index}/{len(args_cli.waypoints)} "
                        f"pos=({info['x']:.3f},{info['y']:.3f},{info['heading']:.3f}) "
                        f"target=({target[0]:.3f},{target[1]:.3f}) dist={info['dist']:.3f} "
                        f"cmd=({info['cmd_vx']:.3f},{info['cmd_vy']:.3f},{info['cmd_yaw']:.3f})",
                        flush=True,
                    )
                else:
                    info = sample_robot_pose(robot, info, ideal_state=ideal_state)
                    age_text = "inf" if math.isinf(info["cmd_age"]) else f"{info['cmd_age']:.2f}"
                    print(
                        "TELEOP_STEP "
                        f"step={step} pos=({info['x']:.3f},{info['y']:.3f},{info['heading']:.3f}) "
                        f"cmd=({info['cmd_vx']:.3f},{info['cmd_vy']:.3f},{info['cmd_yaw']:.3f}) "
                        f"age={age_text}s stale={bool(info['cmd_stale'])}",
                        flush=True,
                    )
            if bool(terminated[0].item() or truncated[0].item()):
                if args_cli.control_mode == "ros_twist":
                    print(
                        f"TELEOP_RESET step={step} terminated={bool(terminated[0].item())} "
                        f"truncated={bool(truncated[0].item())}",
                        flush=True,
                    )
                    if benchmark_telemetry is not None:
                        benchmark_telemetry.maybe_send(
                            robot=robot,
                            info=info,
                            step=step,
                            event="reset",
                            ideal_state=ideal_state,
                        )
                    env.command_manager.get_command("base_velocity")[:] = 0.0
                    obs, _ = env.reset()
                    if ideal_state is not None:
                        ideal_state.reset([args_cli.start_x, args_cli.start_y, args_cli.start_yaw])
                        write_ideal_kinematic_state(
                            robot,
                            ideal_state,
                            torch.zeros((1, 3), dtype=torch.float32, device=robot.device),
                        )
                    continue
                print(
                    f"NAV_ABORT step={step} terminated={bool(terminated[0].item())} "
                    f"truncated={bool(truncated[0].item())}",
                    flush=True,
                )
                break
            perf_t5 = time.perf_counter()
            sleep_s = 0.0
            if args_cli.real_time:
                sleep_s = max(0.0, env.step_dt - (time.time() - step_start))
                time.sleep(sleep_s)
            perf_t6 = time.perf_counter()
            if args_cli.perf_every > 0:
                perf_acc["command"] += perf_t1 - perf_t0
                perf_acc["obs"] += perf_t2 - perf_t1
                perf_acc["policy"] += perf_t3 - perf_t2
                perf_acc["env_step"] += perf_t4 - perf_t3
                perf_acc["io"] += perf_t5 - perf_t4
                perf_acc["sleep"] += perf_t6 - perf_t5
                perf_count += 1
                if step > 0 and step % args_cli.perf_every == 0:
                    wall = max(1.0e-9, perf_t6 - perf_last_wall)
                    print(
                        "PERF_STEP "
                        f"step={step} hz={perf_count / wall:.1f} "
                        f"command_ms={1000.0 * perf_acc['command'] / perf_count:.2f} "
                        f"obs_ms={1000.0 * perf_acc['obs'] / perf_count:.2f} "
                        f"policy_ms={1000.0 * perf_acc['policy'] / perf_count:.2f} "
                        f"env_step_ms={1000.0 * perf_acc['env_step'] / perf_count:.2f} "
                        f"io_ms={1000.0 * perf_acc['io'] / perf_count:.2f} "
                        f"sleep_ms={1000.0 * perf_acc['sleep'] / perf_count:.2f}",
                        flush=True,
                    )
                    for key in perf_acc:
                        perf_acc[key] = 0.0
                    perf_count = 0
                    perf_last_wall = perf_t6
    except Exception:
        loop_failed = True
        print("CONTROL_LOOP_EXCEPTION", flush=True)
        traceback.print_exc()
        raise
    finally:
        trace_file.close()
        final_pos = robot.data.root_pos_w.torch[0].detach().cpu().tolist()
        if third_person_manifest is not None:
            third_person_manifest.close()
        if third_person_root is not None:
            summary = {
                "schema_version": 1,
                "status": "PASS" if third_person_recording_completed else "INCOMPLETE",
                "frame_count": third_person_frame_count,
                "fps": args_cli.third_person_record_fps,
                "requested_duration_sec": args_cli.third_person_record_duration_sec,
                "video_duration_sec": (
                    third_person_frame_count / args_cli.third_person_record_fps
                ),
                "camera_eye": list(args_cli.third_person_camera_eye),
                "camera_target": list(args_cli.third_person_camera_target),
                "resolution": [
                    args_cli.third_person_camera_width,
                    args_cli.third_person_camera_height,
                ],
                "locomotion_fidelity": "isaaclab_go2_learned_policy",
                "ideal_kinematic_base": False,
                "physics_dt_sec": float(env.physics_dt),
                "control_dt_sec": float(env.step_dt),
                "final_root_position": [float(value) for value in final_pos],
            }
            (third_person_root / "capture_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        hold_forever = bool(args_cli.hold_open and not loop_failed)
        hold_until = time.time() + max(0.0, args_cli.hold_seconds)
        if hold_forever or args_cli.hold_seconds > 0.0:
            print("HOLD_OPEN_BEGIN", flush=True)
            try:
                while simulation_app.is_running() and (hold_forever or time.time() < hold_until):
                    env.sim.render()
                    time.sleep(1.0 / 30.0)
            except KeyboardInterrupt:
                pass
            print("HOLD_OPEN_END", flush=True)
        if twist_receiver is not None:
            twist_receiver.close()
        if benchmark_control is not None:
            benchmark_control.close()
        if benchmark_camera is not None:
            benchmark_camera.close()
        if benchmark_telemetry is not None:
            benchmark_telemetry.maybe_send(
                robot=robot,
                info={"cmd_vx": 0.0, "cmd_vy": 0.0, "cmd_yaw": 0.0},
                step=-1,
                event="shutdown",
                ideal_state=ideal_state,
            )
            benchmark_telemetry.close()
        env.close()

    print(
        "NAV_DONE "
        f"success={success} reached={reached_count}/{len(args_cli.waypoints)} "
        f"trace={trace_path} final=({final_pos[0]:.3f},{final_pos[1]:.3f},{final_pos[2]:.3f})",
        flush=True,
    )
    return 0 if success else 3


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
