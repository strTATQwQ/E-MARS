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


def obj_bounds(path: Path) -> dict[str, Any] | None:
    minima = [float("inf")] * 3
    maxima = [float("-inf")] * 3
    vertices = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) < 4:
                continue
            values = [float(fields[index]) for index in range(1, 4)]
            minima = [min(current, value) for current, value in zip(minima, values)]
            maxima = [max(current, value) for current, value in zip(maxima, values)]
            vertices += 1
    if not vertices:
        return None
    center = [(low + high) / 2.0 for low, high in zip(minima, maxima)]
    return {
        "vertices": vertices,
        "min": [round(value, 6) for value in minima],
        "max": [round(value, 6) for value in maxima],
        "center": [round(value, 6) for value in center],
        "extent": [round(high - low, 6) for low, high in zip(minima, maxima)],
    }


def directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def build_catalog(root: Path, archive_sha256_path: Path | None = None) -> dict[str, Any]:
    candidates = sorted(
        path
        for path in root.rglob("isaacsim_*.usd")
        if path.is_file() and not path.name.endswith("_non_metric.usd")
    )
    scenes = []
    for usd_path in candidates:
        relative = usd_path.relative_to(root)
        obj_path = usd_path.with_suffix(".obj")
        scenes.append(
            {
                "scene_id": relative.parts[0],
                "mesh_id": usd_path.stem.removeprefix("isaacsim_"),
                "usd_path": relative.as_posix(),
                "usd_bytes": usd_path.stat().st_size,
                "usd_sha256": sha256_file(usd_path),
                "obj_path": obj_path.relative_to(root).as_posix() if obj_path.is_file() else None,
                "obj_bounds": obj_bounds(obj_path) if obj_path.is_file() else None,
            }
        )
    archive_sha256 = None
    if archive_sha256_path and archive_sha256_path.is_file():
        archive_sha256 = archive_sha256_path.read_text(encoding="utf-8").split()[0]
    return {
        "schema_version": 1,
        "asset_root": str(root.resolve()),
        "archive_sha256": archive_sha256,
        "directory_size_bytes": directory_size(root),
        "scene_count": len(scenes),
        "scene_entry_rule": "isaacsim_*.usd excluding *_non_metric.usd",
        "scenes": scenes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Index verified metric MP3D-PE Isaac scene roots.")
    parser.add_argument("root")
    parser.add_argument("--archive-sha256", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"missing asset root: {root}")
    output = Path(args.output) if args.output else root / "scene_catalog.json"
    catalog = build_catalog(root, Path(args.archive_sha256) if args.archive_sha256 else None)
    output.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    (root / "scene_root_candidates.txt").write_text(
        "\n".join(row["usd_path"] for row in catalog["scenes"]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: catalog[key] for key in ("asset_root", "archive_sha256", "directory_size_bytes", "scene_count")}, indent=2))
    return 0 if catalog["scene_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
