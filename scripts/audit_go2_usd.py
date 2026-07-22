#!/usr/bin/env python3
"""Print a compact, deterministic structural audit of a Go2 USD asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", type=Path)
    args = parser.parse_args()
    stage = Usd.Stage.Open(str(args.usd.resolve()))
    if stage is None:
        raise RuntimeError(f"cannot open USD: {args.usd}")
    print(f"default_prim={stage.GetDefaultPrim().GetPath()}")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        schemas = set(prim.GetAppliedSchemas())
        if (
            prim.IsA(UsdGeom.Camera)
            or UsdPhysics.ArticulationRootAPI.Get(stage, prim.GetPath())
            or path.lower().endswith(("/base", "/base_link", "/trunk"))
            or "ArticulationRootAPI" in schemas
        ):
            print(
                f"prim={path} type={prim.GetTypeName()} "
                f"camera={prim.IsA(UsdGeom.Camera)} schemas={sorted(schemas)}"
            )
            if prim.IsA(UsdGeom.Camera):
                xform = UsdGeom.Xformable(prim)
                print(
                    "  xform_ops="
                    + repr([(op.GetOpName(), op.Get()) for op in xform.GetOrderedXformOps()])
                )
                camera = UsdGeom.Camera(prim)
                print(
                    f"  focal_length={camera.GetFocalLengthAttr().Get()} "
                    f"horizontal_aperture={camera.GetHorizontalApertureAttr().Get()}"
                )


if __name__ == "__main__":
    main()
