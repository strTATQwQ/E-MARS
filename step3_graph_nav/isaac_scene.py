from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _rgb_uint8(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = array * (255.0 if float(np.nanmax(array)) <= 1.01 else 1.0)
    return np.clip(array, 0, 255).astype(np.uint8)


def _encode_jpeg(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=92, subsampling=0)
    return buffer.getvalue()


class IsaacCandidateScene:
    """One real Isaac render-product camera over one frozen MP3D USD."""

    def __init__(self, scene_usd: str | Path, config: dict[str, Any], device: str) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera, CameraCfg

        self.config = config
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 20.0, device=device))
        usd_cfg = sim_utils.UsdFileCfg(usd_path=str(Path(scene_usd).expanduser().resolve()))
        prim = usd_cfg.func("/World/MP3D", usd_cfg)
        if not prim.IsValid():
            raise RuntimeError("failed to spawn MP3D scene")
        light_cfg = sim_utils.DistantLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/GraphNavLight", light_cfg)
        sim_utils.create_prim("/World/GraphNav", "Xform")
        camera_cfg = config["camera"]
        horizontal_aperture = 48.0
        fov = math.radians(float(camera_cfg["horizontal_fov_deg"]))
        focal_length = horizontal_aperture / (2.0 * math.tan(fov / 2.0))
        self.camera = Camera(
            CameraCfg(
                prim_path="/World/GraphNav/Camera",
                update_period=0.0,
                height=int(camera_cfg["height"]),
                width=int(camera_cfg["width"]),
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=focal_length,
                    horizontal_aperture=horizontal_aperture,
                    focus_distance=400.0,
                    clipping_range=(0.05, 100.0),
                ),
            )
        )
        self.sim.reset()
        self._validate_stage()

    def _validate_stage(self) -> None:
        from pxr import UsdGeom, UsdPhysics

        stage = self.sim.stage
        if str(UsdGeom.GetStageUpAxis(stage)).upper() != "Z":
            raise RuntimeError("MP3D stage must be Z-up")
        if not math.isclose(float(UsdGeom.GetStageMetersPerUnit(stage)), 1.0, abs_tol=1.0e-9):
            raise RuntimeError("MP3D stage must use metersPerUnit=1")
        meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise RuntimeError("MP3D stage has no meshes")
        if not any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in meshes):
            raise RuntimeError("MP3D stage has no collision-enabled mesh")

    def stage_stats(self) -> dict[str, Any]:
        from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade

        meshes = [prim for prim in self.sim.stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
        materials = [prim for prim in self.sim.stage.Traverse() if prim.IsA(UsdShade.Material)]
        bound_material_meshes = 0
        for prim in meshes:
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            bound_material_meshes += int(bool(material and material.GetPrim().IsValid()))
        asset_references = []
        for prim in self.sim.stage.Traverse():
            for attribute in prim.GetAttributes():
                if attribute.GetTypeName() == Sdf.ValueTypeNames.Asset:
                    value = attribute.Get()
                    if value and getattr(value, "path", ""):
                        asset_references.append(str(value.path))
        return {
            "up_axis": str(UsdGeom.GetStageUpAxis(self.sim.stage)),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(self.sim.stage)),
            "mesh_count": len(meshes),
            "collision_mesh_count": sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in meshes),
            "material_count": len(materials),
            "bound_material_mesh_count": bound_material_meshes,
            "asset_reference_count": len(asset_references),
        }

    def render(self, position: tuple[float, float, float], yaw_rad: float) -> tuple[bytes, np.ndarray, np.ndarray]:
        import torch

        target = (
            position[0] + math.cos(yaw_rad),
            position[1] + math.sin(yaw_rad),
            position[2],
        )
        self.camera.set_world_poses_from_view(
            torch.tensor([position], dtype=torch.float32, device=self.sim.device),
            torch.tensor([target], dtype=torch.float32, device=self.sim.device),
        )
        for _ in range(3):
            self.sim.step()
            self.camera.update(dt=self.sim.get_physics_dt())
        rgb = _rgb_uint8(self.camera.data.output["rgb"])
        value = self.camera.data.output["distance_to_image_plane"]
        depth = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        if depth.ndim == 4:
            depth = depth[0, ..., 0]
        elif depth.ndim == 3:
            depth = depth[0]
        return _encode_jpeg(rgb), rgb, depth
