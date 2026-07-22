#!/usr/bin/env python3
"""Build a thin Go2 USD wrapper with the frozen InternVLA camera prims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _camera(
    stage: Usd.Stage,
    path: str,
    translation: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    *,
    focal_length: float,
    horizontal_aperture: float,
) -> None:
    camera = UsdGeom.Camera.Define(stage, path)
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    w, x, y, z = orientation_wxyz
    xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
    camera.GetFocalLengthAttr().Set(focal_length)
    camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
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
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/go2_description")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(str(source))

    # Exact transform and intrinsics of the official H1 1.25 m/down-30 camera,
    # attached at the Go2 base so the sensor follows flash motion.
    _camera(
        stage,
        "/go2_description/base/internvla_camera",
        (0.2, 0.0, 0.2),
        (0.6123724356957947, 0.3535533905932736, -0.3535533905932738, -0.6123724356957944),
        focal_length=10.477499961853027,
        horizontal_aperture=20.954999923706055,
    )
    _camera(
        stage,
        "/go2_description/topdown_camera_500",
        (0.0, 0.0, 0.72),
        (0.0, 0.0, 0.0, 1.0),
        focal_length=18.14756202697754,
        horizontal_aperture=200.0,
    )
    stage.GetRootLayer().Save()
    payload = {
        "schema_version": 1,
        "source_asset": "NVIDIA Isaac Sim 6 Unitree Go2",
        "source_sha256": _sha256(source),
        "wrapper_sha256": _sha256(output),
        "camera_prim_path": "base/internvla_camera",
        "camera_resolution": [640, 480],
        "camera_transform_source": "official H1 h1_1_25_down_30",
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
