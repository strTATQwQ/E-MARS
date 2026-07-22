#!/usr/bin/env python3
"""Build the frozen T4 D435i RGB/depth mount plus the odometry stereo rig."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pxr import Usd, UsdGeom

from build_t4_go2_camera_usd import (
    BASE_HEIGHT_ABOVE_SUPPORT_M,
    HORIZONTAL_APERTURE_MM,
    camera_orientation,
    define_camera,
    env_float,
    sha256,
)


def focal_length(hfov_deg: float) -> float:
    return HORIZONTAL_APERTURE_MM / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def vertical_aperture(focal_length_mm: float, vfov_deg: float) -> float:
    return 2.0 * focal_length_mm * math.tan(math.radians(vfov_deg) / 2.0)


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
    camera_pitch = env_float("INTERNVLA_T4_CAMERA_PITCH_DOWN_DEG", 20.0)
    camera_hfov = env_float("INTERNVLA_T4_CAMERA_HFOV_DEG", 69.4)
    camera_vfov = env_float("INTERNVLA_T4_CAMERA_VFOV_DEG", 42.5)
    depth_height = env_float("INTERNVLA_T4_DEPTH_HEIGHT_M", 0.62)
    depth_forward = env_float("INTERNVLA_T4_DEPTH_FORWARD_M", 0.20)
    depth_pitch = env_float("INTERNVLA_T4_DEPTH_PITCH_DOWN_DEG", 20.0)
    depth_hfov = env_float("INTERNVLA_T4_DEPTH_HFOV_DEG", 87.0)
    depth_vfov = env_float("INTERNVLA_T4_DEPTH_VFOV_DEG", 58.0)
    stereo_height = env_float("INTERNVLA_T4_STEREO_HEIGHT_M", 0.62)
    stereo_forward = env_float("INTERNVLA_T4_STEREO_FORWARD_M", 0.20)
    stereo_pitch = env_float("INTERNVLA_T4_STEREO_PITCH_DOWN_DEG", 10.0)
    stereo_hfov = env_float("INTERNVLA_T4_STEREO_HFOV_DEG", 90.0)
    stereo_baseline = env_float("INTERNVLA_T4_STEREO_BASELINE_M", 0.12)
    if not (
        0.50 <= camera_height <= 1.35
        and 0.0 <= camera_forward <= 0.40
        and 0.0 <= camera_pitch <= 60.0
        and 60.0 <= camera_hfov <= 120.0
        and 30.0 <= camera_vfov <= 90.0
        and 0.50 <= depth_height <= 1.35
        and 0.0 <= depth_forward <= 0.40
        and 0.0 <= depth_pitch <= 60.0
        and 60.0 <= depth_hfov <= 120.0
        and 30.0 <= depth_vfov <= 90.0
        and 0.50 <= stereo_height <= 1.20
        and 0.0 <= stereo_forward <= 0.40
        and 0.0 <= stereo_pitch <= 30.0
        and 60.0 <= stereo_hfov <= 120.0
        and 0.06 <= stereo_baseline <= 0.30
    ):
        raise ValueError("invalid T4 sensor-rig calibration")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/go2_description")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(str(source))
    semantic_translation = (
        camera_forward,
        0.0,
        camera_height - BASE_HEIGHT_ABOVE_SUPPORT_M,
    )
    semantic_focal_length = focal_length(camera_hfov)
    define_camera(
        stage,
        "/go2_description/base/internvla_camera",
        semantic_translation,
        camera_orientation(camera_pitch),
        semantic_focal_length,
        HORIZONTAL_APERTURE_MM,
        vertical_aperture(semantic_focal_length, camera_vfov),
    )
    depth_translation = (
        depth_forward,
        0.0,
        depth_height - BASE_HEIGHT_ABOVE_SUPPORT_M,
    )
    depth_focal_length = focal_length(depth_hfov)
    define_camera(
        stage,
        "/go2_description/base/t4_d435i_depth",
        depth_translation,
        camera_orientation(depth_pitch),
        depth_focal_length,
        HORIZONTAL_APERTURE_MM,
        vertical_aperture(depth_focal_length, depth_vfov),
    )
    for side, lateral in (
        ("left", stereo_baseline / 2.0),
        ("right", -stereo_baseline / 2.0),
    ):
        define_camera(
            stage,
            f"/go2_description/base/t4_stereo_{side}",
            (
                stereo_forward,
                lateral,
                stereo_height - BASE_HEIGHT_ABOVE_SUPPORT_M,
            ),
            camera_orientation(stereo_pitch),
            focal_length(stereo_hfov),
            HORIZONTAL_APERTURE_MM,
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
        "semantic_camera": {
            "prim_path": "base/internvla_camera",
            "resolution": [640, 480],
            "translation_from_base_m": list(semantic_translation),
            "height_above_support_m": camera_height,
            "pitch_down_deg": camera_pitch,
            "hfov_deg": camera_hfov,
            "vfov_deg": camera_vfov,
        },
        "depth_camera": {
            "camera_model": "Intel RealSense D435i depth imager",
            "prim_path": "base/t4_d435i_depth",
            "resolution": [640, 480],
            "translation_from_base_m": list(depth_translation),
            "height_above_support_m": depth_height,
            "pitch_down_deg": depth_pitch,
            "hfov_deg": depth_hfov,
            "vfov_deg": depth_vfov,
            "minimum_depth_m": 0.28,
        },
        "odometry_stereo": {
            "left_prim_path": "base/t4_stereo_left",
            "right_prim_path": "base/t4_stereo_right",
            "resolution": [320, 240],
            "baseline_m": stereo_baseline,
            "forward_m": stereo_forward,
            "height_above_support_m": stereo_height,
            "pitch_down_deg": stereo_pitch,
            "hfov_deg": stereo_hfov,
            "rectified": True,
            "mount": "rigid_fixed_extrinsics",
        },
        "posture_compensation": "continuous_base_roll_pitch_feedback_stabilizer",
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
