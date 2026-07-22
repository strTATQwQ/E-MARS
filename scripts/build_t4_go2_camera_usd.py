#!/usr/bin/env python3
"""Build a T4 Go2 camera wrapper without modifying the frozen T3 builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


BASE_HEIGHT_ABOVE_SUPPORT_M = 0.42
HORIZONTAL_APERTURE_MM = 20.954999923706055


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def camera_orientation(pitch_down_deg: float) -> tuple[float, float, float, float]:
    # Camera local axes are +X right, +Y up, -Z view. This quaternion keeps
    # camera right on base -Y and points -Z along base +X with the requested
    # downward pitch. It reproduces the frozen T3 quaternion at 30 degrees.
    alpha = math.radians((90.0 - pitch_down_deg) / 2.0)
    scale = math.sqrt(0.5)
    return (
        scale * math.cos(alpha),
        scale * math.sin(alpha),
        -scale * math.sin(alpha),
        -scale * math.cos(alpha),
    )


def define_camera(
    stage: Usd.Stage,
    path: str,
    translation: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float | None = None,
) -> None:
    camera = UsdGeom.Camera.Define(stage, path)
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    w, x, y, z = orientation_wxyz
    xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
    camera.GetFocalLengthAttr().Set(focal_length)
    camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
    if vertical_aperture is not None:
        camera.GetVerticalApertureAttr().Set(vertical_aperture)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1_000_000.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    camera_height = env_float("INTERNVLA_T4_CAMERA_HEIGHT_M", 0.62)
    camera_forward = env_float("INTERNVLA_T4_CAMERA_FORWARD_M", 0.20)
    pitch_down = env_float("INTERNVLA_T4_CAMERA_PITCH_DOWN_DEG", 30.0)
    hfov = env_float("INTERNVLA_T4_CAMERA_HFOV_DEG", 90.0)
    vfov_raw = os.environ.get("INTERNVLA_T4_CAMERA_VFOV_DEG")
    vfov = float(vfov_raw) if vfov_raw is not None else None
    camera_model = os.environ.get("INTERNVLA_T4_CAMERA_MODEL", "generic_rgbd")
    if not 0.50 <= camera_height <= 1.35:
        raise ValueError("camera height must be in [0.50, 1.35] m")
    if not 0.0 <= camera_forward <= 0.40:
        raise ValueError("camera forward offset must be in [0.0, 0.40] m")
    if not 0.0 <= pitch_down <= 60.0:
        raise ValueError("camera downward pitch must be in [0, 60] degrees")
    if not 60.0 <= hfov <= 120.0:
        raise ValueError("camera HFOV must be in [60, 120] degrees")
    if vfov is not None and not (30.0 <= vfov <= 90.0 and math.isfinite(vfov)):
        raise ValueError("camera VFOV must be finite and in [30, 90] degrees")

    translation = (
        camera_forward,
        0.0,
        camera_height - BASE_HEIGHT_ABOVE_SUPPORT_M,
    )
    orientation = camera_orientation(pitch_down)
    focal_length = HORIZONTAL_APERTURE_MM / (
        2.0 * math.tan(math.radians(hfov) / 2.0)
    )
    vertical_aperture = (
        2.0 * focal_length * math.tan(math.radians(vfov) / 2.0)
        if vfov is not None
        else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/go2_description")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(str(source))
    define_camera(
        stage,
        "/go2_description/base/internvla_camera",
        translation,
        orientation,
        focal_length,
        HORIZONTAL_APERTURE_MM,
        vertical_aperture,
    )
    define_camera(
        stage,
        "/go2_description/topdown_camera_500",
        (0.0, 0.0, 0.72),
        (0.0, 0.0, 0.0, 1.0),
        18.14756202697754,
        200.0,
    )
    stage.GetRootLayer().Save()

    payload = {
        "schema_version": 1,
        "source_asset": "NVIDIA Isaac Sim 6 Unitree Go2",
        "source_sha256": sha256(source),
        "wrapper_sha256": sha256(output),
        "camera_prim_path": "base/internvla_camera",
        "camera_resolution": [640, 480],
        "camera_model": camera_model,
        "camera_height_above_support_m": camera_height,
        "base_height_above_support_m": BASE_HEIGHT_ABOVE_SUPPORT_M,
        "camera_translation_from_base_m": list(translation),
        "camera_orientation_wxyz": list(orientation),
        "pitch_down_deg": pitch_down,
        "hfov_deg": hfov,
        "vfov_deg": vfov,
        "focal_length_mm": focal_length,
        "horizontal_aperture_mm": HORIZONTAL_APERTURE_MM,
        "vertical_aperture_mm": vertical_aperture,
        "mount": "rigid_fixed_extrinsics",
        "posture_compensation": "continuous_base_roll_pitch_feedback_stabilizer",
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
