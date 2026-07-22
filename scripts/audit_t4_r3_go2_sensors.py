#!/usr/bin/env python3
"""Audit sensor-related prims in the composed Isaac Go2 asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pxr import Usd


TOKENS = ("sensor", "lidar", "camera", "imu", "head", "radar")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    asset = args.asset.expanduser().resolve()
    if not asset.is_file():
        raise FileNotFoundError(asset)
    stage = Usd.Stage.Open(str(asset), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open {asset}")
    related = []
    cameras = []
    lidars = []
    imus = []
    for prim in stage.TraverseAll():
        path = prim.GetPath().pathString
        lowered = path.lower()
        type_name = prim.GetTypeName()
        item = {
            "path": path,
            "type_name": type_name,
            "applied_schemas": list(prim.GetAppliedSchemas()),
            "attributes": sorted(attribute.GetName() for attribute in prim.GetAttributes()),
        }
        if any(token in lowered for token in TOKENS):
            related.append(item)
        if type_name == "Camera" or "camera" in lowered:
            cameras.append(path)
        if "lidar" in lowered or "lidarsensor" in type_name.lower():
            lidars.append(path)
        if "imu" in lowered or "imusensor" in type_name.lower():
            imus.append(path)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "asset_sha256": sha256(asset),
        "default_prim": (
            stage.GetDefaultPrim().GetPath().pathString
            if stage.GetDefaultPrim().IsValid()
            else None
        ),
        "prim_count": sum(1 for _ in stage.TraverseAll()),
        "camera_paths": sorted(set(cameras)),
        "lidar_paths": sorted(set(lidars)),
        "imu_paths": sorted(set(imus)),
        "sensor_related_prims": related,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
