from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


def _finite3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be finite")
    return result  # type: ignore[return-value]


def habitat_position_to_isaac(position: Sequence[float]) -> tuple[float, float, float]:
    """Map Habitat Y-up coordinates to the metric MP3D USD Z-up frame."""
    x, y, z = _finite3(position, "Habitat position")
    return (x, -z, y)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def habitat_pose_to_isaac(
    position: Sequence[float], rotation_xyzw: Sequence[float]
) -> tuple[tuple[float, float, float], float]:
    """Return Isaac position and planar yaw for an R2R Habitat start pose."""
    if len(rotation_xyzw) != 4:
        raise ValueError("Habitat rotation must be an xyzw quaternion")
    qx, qy, qz, qw = (float(value) for value in rotation_xyzw)
    if not all(math.isfinite(value) for value in (qx, qy, qz, qw)):
        raise ValueError("Habitat rotation must be finite")
    if abs(qx) > 1.0e-5 or abs(qz) > 1.0e-5:
        raise ValueError("R2R benchmark expects a yaw-only Habitat quaternion")
    norm = math.hypot(qy, qw)
    if norm < 1.0e-8:
        raise ValueError("Habitat rotation quaternion has zero norm")
    habitat_yaw = 2.0 * math.atan2(qy / norm, qw / norm)
    # Habitat's zero-yaw camera looks along -Z. After the +90 degree X
    # basis change that direction is Isaac +Y, hence the pi/2 offset.
    return habitat_position_to_isaac(position), normalize_angle(math.pi / 2.0 + habitat_yaw)


@dataclass(frozen=True)
class ViewpointNode:
    """One included viewpoint; node_id is its stable source-list index."""

    node_id: int
    image_id: str
    camera_position: tuple[float, float, float]
    neighbors: tuple[int, ...]


class MatterportGraph:
    """Frozen oracle graph from the official Matterport connectivity files."""

    def __init__(self, scene_id: str, nodes: Iterable[ViewpointNode]) -> None:
        self.scene_id = str(scene_id)
        self.nodes = {node.node_id: node for node in nodes}
        if not self.nodes:
            raise ValueError(f"scene {scene_id!r} has no included viewpoints")
        for node in self.nodes.values():
            invalid = sorted(set(node.neighbors) - self.nodes.keys())
            if invalid:
                raise ValueError(f"node {node.node_id} has invalid neighbors {invalid}")

    @classmethod
    def load(cls, path: str | Path, *, scene_id: str | None = None) -> "MatterportGraph":
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError("connectivity file must contain a list")
        included = {
            index
            for index, row in enumerate(rows)
            if isinstance(row, dict) and bool(row.get("included"))
        }
        nodes: list[ViewpointNode] = []
        for index in sorted(included):
            row = rows[index]
            pose = row.get("pose")
            unobstructed = row.get("unobstructed")
            if not isinstance(pose, list) or len(pose) != 16:
                raise ValueError(f"node {index} has invalid 4x4 pose")
            if not isinstance(unobstructed, list) or len(unobstructed) != len(rows):
                raise ValueError(f"node {index} has invalid unobstructed array")
            position = _finite3((pose[3], pose[7], pose[11]), f"node {index} position")
            neighbors = tuple(
                candidate
                for candidate in sorted(included)
                if candidate != index and bool(unobstructed[candidate])
            )
            nodes.append(
                ViewpointNode(
                    node_id=index,
                    image_id=str(row.get("image_id") or index),
                    camera_position=position,
                    neighbors=neighbors,
                )
            )
        inferred_scene = scene_id or source.name.removesuffix("_connectivity.json")
        graph = cls(inferred_scene, nodes)
        graph._validate_reciprocal_edges()
        return graph

    def _validate_reciprocal_edges(self) -> None:
        for node in self.nodes.values():
            for neighbor in node.neighbors:
                if node.node_id not in self.nodes[neighbor].neighbors:
                    raise ValueError(f"edge {node.node_id}->{neighbor} is not reciprocal")

    def nearest_node(
        self,
        habitat_position: Sequence[float],
        *,
        nominal_camera_height_m: float = 1.40,
    ) -> int:
        """Snap an R2R floor point to a viewpoint without mixing floors."""
        floor = habitat_position_to_isaac(habitat_position)
        target = (floor[0], floor[1], floor[2] + nominal_camera_height_m)
        return min(
            self.nodes,
            key=lambda node_id: (
                math.dist(target, self.nodes[node_id].camera_position),
                node_id,
            ),
        )

    def edge_distance(self, source: int, target: int) -> float:
        if target not in self.nodes[source].neighbors:
            raise ValueError(f"{source}->{target} is not an edge")
        return math.dist(self.nodes[source].camera_position, self.nodes[target].camera_position)

    def shortest_path(self, source: int, target: int) -> tuple[int, ...]:
        if source not in self.nodes or target not in self.nodes:
            raise KeyError("shortest-path endpoint is not in the graph")
        queue: list[tuple[float, int]] = [(0.0, source)]
        distance = {source: 0.0}
        previous: dict[int, int] = {}
        while queue:
            cost, node_id = heapq.heappop(queue)
            if cost != distance.get(node_id):
                continue
            if node_id == target:
                break
            for neighbor in self.nodes[node_id].neighbors:
                candidate = cost + self.edge_distance(node_id, neighbor)
                if candidate < distance.get(neighbor, math.inf):
                    distance[neighbor] = candidate
                    previous[neighbor] = node_id
                    heapq.heappush(queue, (candidate, neighbor))
        if target not in distance:
            raise ValueError(f"no graph path from {source} to {target}")
        path = [target]
        while path[-1] != source:
            path.append(previous[path[-1]])
        return tuple(reversed(path))

    def shortest_distance(self, source: int, target: int) -> float:
        path = self.shortest_path(source, target)
        return sum(self.edge_distance(a, b) for a, b in zip(path, path[1:]))

    def candidate_geometry(
        self, node_id: int, yaw_rad: float
    ) -> tuple[tuple[int, tuple[float, float], float, float, float], ...]:
        """Return stable ID, local forward/left, range, bearing and dz."""
        origin = self.nodes[node_id].camera_position
        rows = []
        for neighbor in self.nodes[node_id].neighbors:
            target = self.nodes[neighbor].camera_position
            dx, dy, dz = (target[index] - origin[index] for index in range(3))
            forward = math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy
            left = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
            bearing = math.degrees(math.atan2(left, forward))
            rows.append((neighbor, (forward, left), math.sqrt(dx * dx + dy * dy + dz * dz), bearing, dz))
        return tuple(sorted(rows, key=lambda row: row[0]))
