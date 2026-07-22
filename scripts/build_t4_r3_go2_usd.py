#!/usr/bin/env python3
"""Build the T4.2-R3 Go2 wrapper with explicit auxiliary sensor frames."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

from build_t4_go2_camera_usd import (
    HORIZONTAL_APERTURE_MM,
    camera_orientation,
    define_camera,
    sha256,
)
from build_t4_sensor_go2_usd import focal_length, vertical_aperture
from t5_revc_sensor_contract import (
    DEFAULT_CONTRACT as DEFAULT_REVC_CONTRACT,
    camera_orientation_wxyz,
    camera_translation_base_m,
    load_revc_contract,
    t5_revc_feature_scope,
)


def _xform(stage: Usd.Stage, path: str, translation: tuple[float, float, float]) -> None:
    prim = UsdGeom.Xform.Define(stage, path)
    prim.AddTranslateOp().Set(Gf.Vec3d(*translation))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    base_builder = Path(__file__).with_name("build_t4_sensor_go2_usd.py")
    revc_scope = t5_revc_feature_scope()
    revc_enabled = revc_scope is not None
    revc_config_path: Path | None = None
    revc_contract: dict[str, object] | None = None
    if revc_enabled:
        revc_config_path = Path(
            os.environ.get(
                "INTERNVLA_T5_REVC_CAMERA_CONFIG", str(DEFAULT_REVC_CONTRACT)
            )
        ).expanduser().resolve()
        revc_contract = load_revc_contract(revc_config_path)

    with tempfile.TemporaryDirectory(prefix="t4_r3_usd_") as temporary:
        base_manifest = Path(temporary) / "base_manifest.json"
        subprocess.run(
            [
                sys.executable,
                str(base_builder),
                "--source",
                str(source),
                "--output",
                str(output),
                "--manifest",
                str(base_manifest),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        payload = json.loads(base_manifest.read_text(encoding="utf-8"))

    lidar_translation = (0.25, 0.0, 0.18)
    imu_translation = (0.0, 0.0, 0.0)
    front_translation = (0.29, 0.0, -0.06)
    front_pitch = float(os.environ.get("INTERNVLA_T4_R3_FRONT_RGB_PITCH_DEG", "8"))
    front_hfov = float(os.environ.get("INTERNVLA_T4_R3_FRONT_RGB_HFOV_DEG", "120"))
    front_vfov = float(os.environ.get("INTERNVLA_T4_R3_FRONT_RGB_VFOV_DEG", "75"))
    front_clipping = (0.20, 1_000_000.0)
    if not (0.0 <= front_pitch <= 25.0 and 90.0 <= front_hfov <= 130.0):
        raise ValueError("invalid Go2 front-RGB calibration")

    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"cannot reopen wrapper: {output}")
    _xform(stage, "/go2_description/base/go2_l1_lidar", lidar_translation)
    _xform(stage, "/go2_description/base/go2_imu", imu_translation)
    front_focal = focal_length(front_hfov)
    define_camera(
        stage,
        "/go2_description/base/go2_front_rgb",
        front_translation,
        camera_orientation(front_pitch),
        front_focal,
        HORIZONTAL_APERTURE_MM,
        vertical_aperture(front_focal, front_vfov),
    )
    front_camera = UsdGeom.Camera(
        stage.GetPrimAtPath("/go2_description/base/go2_front_rgb")
    )
    front_camera.GetClippingRangeAttr().Set(Gf.Vec2f(*front_clipping))
    revc_camera_manifest: list[dict[str, object]] = []
    if revc_enabled:
        assert revc_contract is not None
        revc_optics = revc_contract["optics"]
        revc_focal = focal_length(float(revc_optics["hfov_deg"]))
        revc_vertical_aperture = HORIZONTAL_APERTURE_MM * (
            float(revc_optics["resolution"][1])
            / float(revc_optics["resolution"][0])
        )
        for camera in revc_contract["cameras"]:
            translation = camera_translation_base_m(revc_contract, camera)
            orientation = camera_orientation_wxyz(
                float(revc_optics["pitch_down_deg"]), float(camera["yaw_deg"])
            )
            define_camera(
                stage,
                f"/go2_description/{camera['prim_path']}",
                translation,
                orientation,
                revc_focal,
                HORIZONTAL_APERTURE_MM,
                revc_vertical_aperture,
            )
            revc_camera_manifest.append(
                {
                    "identity": camera["identity"],
                    "sensor_name": camera["sensor_name"],
                    "prim_path": camera["prim_path"],
                    "frame_id": camera["frame_id"],
                    "translation_from_base_m": list(translation),
                    "orientation_wxyz": list(orientation),
                    "yaw_deg": camera["yaw_deg"],
                }
            )
    stage.GetRootLayer().Save()

    payload.update(
        {
            "schema_version": 2,
            "wrapper_sha256": sha256(output),
            "go2_4d_lidar": {
                "simulation_backend": "PhysX scene-query rays",
                "prim_path": "base/go2_l1_lidar",
                "frame_id": "go2_l1_lidar",
                "translation_from_base_m": list(lidar_translation),
                "azimuth_samples": 180,
                "elevation_channels": 8,
                "elevation_degrees": [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
                "range_m": [0.10, 12.0],
            },
            "go2_front_rgb": {
                "role": "low-mounted visual recording and review only",
                "is_depth_source": False,
                "prim_path": "base/go2_front_rgb",
                "frame_id": "go2_front_rgb_optical_frame",
                "translation_from_base_m": list(front_translation),
                "resolution": [320, 240],
                "ros_publish_resolution": [160, 120],
                "pitch_down_deg": front_pitch,
                "hfov_deg": front_hfov,
                "vfov_deg": front_vfov,
                "clipping_range_m": list(front_clipping),
            },
            "imu": {
                "prim_path": "base/go2_imu",
                "frame_id": "go2_imu_link",
                "translation_from_base_m": list(imu_translation),
                "simulation_source": "articulation orientation and angular velocity",
            },
        }
    )
    if revc_enabled:
        assert revc_contract is not None
        assert revc_config_path is not None
        assert revc_scope is not None
        payload["schema_version"] = 3
        payload["imu"] = {
            **payload["imu"],
            "simulation_source": (
                "articulation orientation/angular velocity plus two-normal-sample "
                "simulation-stamp finite-difference linear acceleration"
            ),
            "linear_acceleration_contract": (
                "internnav-t5-sim-imu-linear-acceleration-v1"
            ),
            "lio_backend_implemented": False,
        }
        payload["revc_four_camera"] = {
            "enabled": True,
            "contract_id": revc_contract["contract_id"],
            "contract_sha256": sha256(revc_config_path),
            "scope": revc_contract["scope"],
            "camera_order": revc_contract["camera_order"],
            "resolution": revc_contract["optics"]["resolution"],
            "hfov_deg": revc_contract["optics"]["hfov_deg"],
            "pitch_down_deg": revc_contract["optics"]["pitch_down_deg"],
            "capture": revc_contract["snapshot"]["capture"],
            "external_preview_max_hz": revc_contract["snapshot"][
                "external_preview_max_hz"
            ],
            "render_barrier": "replicator_step_pause_timeline_wait_for_render",
            "cameras": revc_camera_manifest,
            "cuvslam_stereo_is_separate": True,
            "frozen_scope": {
                "lane": revc_scope.lane,
                "identity_prefix": revc_scope.identity_prefix,
                "result_root": str(revc_scope.result_root),
            },
        }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
