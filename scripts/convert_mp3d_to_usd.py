#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_source(asset_root: Path, scene_id: str) -> Path:
    values = sorted(
        path for path in (asset_root / scene_id).rglob("isaacsim_*.usd") if "non_metric" not in path.name
    )
    if len(values) != 1:
        raise RuntimeError(f"scene {scene_id} has {len(values)} metric source USD files")
    return values[0]


def has_nonzero_area_triangle(mesh: Any, *, epsilon: float = 1.0e-12) -> bool:
    points = mesh.GetPointsAttr().Get() or []
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    offset = 0
    for count_value in counts:
        count = int(count_value)
        face = indices[offset : offset + count]
        offset += count
        if count < 3:
            continue
        anchor = points[int(face[0])]
        for index in range(1, count - 1):
            first = points[int(face[index])]
            second = points[int(face[index + 1])]
            ax, ay, az = (float(first[i] - anchor[i]) for i in range(3))
            bx, by, bz = (float(second[i] - anchor[i]) for i in range(3))
            cx, cy, cz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
            if cx * cx + cy * cy + cz * cz > (2.0 * epsilon) ** 2:
                return True
    return False


def create_collision_wrapper(source: Path, target: Path) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(target))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    scene = stage.DefinePrim("/World/Scene", "Xform")
    scene.GetReferences().AddReference(str(source.resolve()))
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().Save()
    stage.Reload()
    meshes = collisions = materials = bindings = 0
    excluded_degenerate_meshes: list[str] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            meshes += 1
            if has_nonzero_area_triangle(UsdGeom.Mesh(prim)):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
                collisions += 1
            else:
                # Disable the zero-area prim in this wrapper layer.  Omitting
                # CollisionAPI or authoring collisionEnabled=false still lets
                # PhysX traverse it while merging a static parent hierarchy.
                # A zero-area mesh has no visual surface to preserve.
                prim.SetActive(False)
                excluded_degenerate_meshes.append(str(prim.GetPath()))
            relation = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
            if relation and relation.GetTargets():
                bindings += 1
        if prim.IsA(UsdShade.Material):
            materials += 1
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(target))
    if not reopened:
        raise RuntimeError(f"cannot reopen generated USD {target}")
    enabled = sum(
        1
        for prim in reopened.Traverse()
        if prim.IsA(UsdGeom.Mesh)
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
    )
    if meshes <= 0 or enabled != collisions:
        raise RuntimeError(
            f"collision wrapper validation failed: meshes={meshes}, expected_enabled={collisions}, enabled={enabled}"
        )
    return {
        "source_usd": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "output_usd": str(target.resolve()),
        "output_sha256": sha256_file(target),
        "up_axis": str(UsdGeom.GetStageUpAxis(reopened)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(reopened)),
        "mesh_count": meshes,
        "collision_mesh_count": enabled,
        "collision_excluded_degenerate_count": len(excluded_degenerate_meshes),
        "collision_excluded_degenerate_meshes": excluded_degenerate_meshes,
        "material_count": materials,
        "bound_mesh_count": bindings,
        "collision_approximation": "none_static_triangle_mesh",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create metric, collision-enabled wrappers without modifying MP3D source assets.")
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--scene-splits", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--roles", nargs="+", default=["conversion_debug_scenes", "prompt_validation_scenes", "nvfp4_calibration_scenes", "formal_test_scenes"])
    args = parser.parse_args()
    asset_root = Path(args.asset_root)
    output_root = Path(args.output_root)
    splits = json.loads(Path(args.scene_splits).read_text(encoding="utf-8"))
    scene_ids = sorted({scene for role in args.roles for scene in splits[role]})
    rows = []
    failures = []
    for scene_id in scene_ids:
        try:
            source = metric_source(asset_root, scene_id)
            row = create_collision_wrapper(source, output_root / scene_id / "scene_collision.usd")
            row["scene_id"] = scene_id
            rows.append(row)
            print(json.dumps({"scene_id": scene_id, "status": "ok", "mesh_count": row["mesh_count"]}))
        except Exception as exc:
            failures.append({"scene_id": scene_id, "error": f"{type(exc).__name__}:{exc}"})
            print(json.dumps({"scene_id": scene_id, "status": "failed", "error": failures[-1]["error"]}))
    manifest = {
        "schema_version": 2,
        "source_asset_root": str(asset_root.resolve()),
        "output_root": str(output_root.resolve()),
        "source_data_modified": False,
        "scenes": rows,
        "failures": failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "converted": len(rows), "failed": len(failures)}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
