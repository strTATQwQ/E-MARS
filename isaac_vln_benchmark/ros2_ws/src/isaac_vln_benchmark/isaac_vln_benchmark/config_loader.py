from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


def resolve_data_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("isaac_vln_benchmark"))
        for fallback in (share / candidate, share / "configs" / candidate.name):
            if fallback.exists():
                return fallback
    except Exception:
        pass
    return candidate


def load_data(path: str | Path) -> Any:
    """Load JSON-compatible YAML, using PyYAML when available."""
    path = resolve_data_path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def dump_data(data: Any, path: str | Path) -> None:
    """Write deterministic JSON, which is also valid YAML 1.2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scene_by_type(scenes_doc: dict[str, Any], scene_type: str) -> dict[str, Any]:
    for scene in scenes_doc.get("scenes", []):
        if scene.get("scene_type") == scene_type:
            return scene
    raise KeyError(f"No scene with scene_type={scene_type!r}")


def scene_by_id(scenes_doc: dict[str, Any], scene_id: str) -> dict[str, Any]:
    for scene in scenes_doc.get("scenes", []):
        if scene.get("scene_id") == scene_id:
            return scene
    raise KeyError(f"No scene with scene_id={scene_id!r}")


def object_by_id(scene: dict[str, Any], object_id: str) -> dict[str, Any]:
    for obj in scene.get("objects", []):
        if obj.get("id") == object_id:
            return obj
    raise KeyError(f"No object {object_id!r} in scene {scene.get('scene_id')!r}")


def normalize_scene_to_robot_origin(scene: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(scene)
    start = list(normalized.get("robot_start_pose") or [0.0, 0.0, 0.0])
    sx, sy, syaw = float(start[0]), float(start[1]), float(start[2])
    cos_yaw = math.cos(-syaw)
    sin_yaw = math.sin(-syaw)

    def transform_xy(values: Any) -> Any:
        if not isinstance(values, list) or len(values) < 2:
            return values
        transformed = list(values)
        dx = float(transformed[0]) - sx
        dy = float(transformed[1]) - sy
        transformed[0] = cos_yaw * dx - sin_yaw * dy
        transformed[1] = sin_yaw * dx + cos_yaw * dy
        return transformed

    normalized["source_robot_start_pose"] = start
    normalized["robot_start_pose"] = [0.0, 0.0, 0.0]
    bounds = normalized.get("bounds")
    if isinstance(bounds, list) and len(bounds) >= 4:
        corners = [
            transform_xy([float(bounds[0]), float(bounds[1])]),
            transform_xy([float(bounds[0]), float(bounds[3])]),
            transform_xy([float(bounds[2]), float(bounds[1])]),
            transform_xy([float(bounds[2]), float(bounds[3])]),
        ]
        normalized["bounds"] = [
            min(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[0] for point in corners),
            max(point[1] for point in corners),
        ]
    for key in ("objects", "obstacles"):
        for item in normalized.get(key, []):
            if isinstance(item, dict) and "pose" in item:
                item["pose"] = transform_xy(item["pose"])
    for item in normalized.get("dynamic_obstacles", []):
        if not isinstance(item, dict):
            continue
        if "pose" in item:
            item["pose"] = transform_xy(item["pose"])
        if isinstance(item.get("path"), list):
            item["path"] = [transform_xy(point) for point in item["path"]]
    for zone in normalized.get("semantic_zones", []):
        if isinstance(zone, dict) and "center" in zone:
            zone["center"] = transform_xy(zone["center"])
    return normalized
