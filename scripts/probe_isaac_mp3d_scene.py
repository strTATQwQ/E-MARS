#!/usr/bin/env python3
"""Render a frozen MP3D collision USD at official Matterport viewpoints."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene-usd", required=True)
parser.add_argument("--connectivity", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--viewpoints", type=int, default=3)
parser.add_argument("--node-id", type=int)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

from slow_benchmark.oracle_graph import MatterportGraph


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb_to_uint8(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.01 else 1.0
        array = array * scale
    return np.clip(array, 0, 255).astype(np.uint8)


def main() -> int:
    output = Path(args_cli.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_path = Path(args_cli.scene_usd).expanduser().resolve()
    graph = MatterportGraph.load(args_cli.connectivity)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 20.0, device=str(args_cli.device)))
    usd_cfg = sim_utils.UsdFileCfg(usd_path=str(scene_path))
    scene_prim = usd_cfg.func("/World/MP3D", usd_cfg)
    if not scene_prim.IsValid():
        raise RuntimeError("MP3D scene prim is invalid")
    light_cfg = sim_utils.DistantLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/BenchmarkLight", light_cfg)
    sim_utils.create_prim("/World/Benchmark", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Benchmark/Camera",
        update_period=0.0,
        height=360,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=48.0,
            focus_distance=400.0,
            clipping_range=(0.05, 100.0),
        ),
    )
    camera = Camera(camera_cfg)
    sim.reset()

    stage = sim.stage
    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    meshes = 0
    collision_meshes = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            meshes += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_meshes += 1

    render_rows = []
    if args_cli.node_id is not None:
        if args_cli.node_id not in graph.nodes:
            raise ValueError(f"node_id={args_cli.node_id} is absent from connectivity graph")
        node_ids = [args_cli.node_id]
    else:
        node_ids = sorted(graph.nodes)[: max(1, args_cli.viewpoints)]
    for node_id in node_ids:
        position = graph.nodes[node_id].camera_position
        for heading_deg in (0.0, 90.0, 180.0, 270.0):
            yaw = math.radians(heading_deg)
            target = (position[0] + math.cos(yaw), position[1] + math.sin(yaw), position[2])
            camera.set_world_poses_from_view(
                torch.tensor([position], dtype=torch.float32, device=sim.device),
                torch.tensor([target], dtype=torch.float32, device=sim.device),
            )
            for _ in range(3):
                sim.step()
                camera.update(dt=sim.get_physics_dt())
            rgb = _rgb_to_uint8(camera.data.output["rgb"])
            depth_value = camera.data.output["distance_to_image_plane"]
            depth = depth_value.detach().cpu().numpy() if hasattr(depth_value, "detach") else np.asarray(depth_value)
            if depth.ndim == 4:
                depth = depth[0, ..., 0]
            elif depth.ndim == 3:
                depth = depth[0]
            finite = depth[np.isfinite(depth)]
            filename = f"node_{node_id:04d}_heading_{int(heading_deg):03d}.jpg"
            Image.fromarray(rgb, mode="RGB").save(output / filename, quality=92, subsampling=0)
            render_rows.append(
                {
                    "node_id": node_id,
                    "image_id": graph.nodes[node_id].image_id,
                    "camera_position": list(position),
                    "heading_deg": heading_deg,
                    "rgb_file": filename,
                    "rgb_sha256": _sha256(output / filename),
                    "depth_finite_fraction": float(np.isfinite(depth).mean()),
                    "depth_min_m": float(finite.min()) if finite.size else None,
                    "depth_median_m": float(np.median(finite)) if finite.size else None,
                }
            )

    report = {
        "schema_version": 1,
        "scene_usd": str(scene_path),
        "scene_usd_sha256": _sha256(scene_path),
        "connectivity": str(Path(args_cli.connectivity).resolve()),
        "scene_id": graph.scene_id,
        "up_axis": str(up_axis),
        "meters_per_unit": float(meters_per_unit),
        "mesh_count": meshes,
        "collision_mesh_count": collision_meshes,
        "camera": {"width": 640, "height": 360, "data_types": camera_cfg.data_types},
        "renders": render_rows,
    }
    (output / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
