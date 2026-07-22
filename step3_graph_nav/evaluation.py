from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from slow_benchmark.oracle_graph import MatterportGraph, habitat_pose_to_isaac, normalize_angle


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: int
    target_viewpoint_id: int
    relative_heading_deg: float
    graph_distance_m: float
    absolute_yaw_rad: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target_viewpoint_id": self.target_viewpoint_id,
            "relative_heading_deg": self.relative_heading_deg,
            "graph_distance_m": self.graph_distance_m,
            "absolute_yaw_rad": self.absolute_yaw_rad,
        }


def candidate_specs(graph: MatterportGraph, node_id: int, yaw_rad: float) -> tuple[CandidateSpec, ...]:
    origin = graph.nodes[node_id].camera_position
    rows = []
    for candidate_id, target_id in enumerate(sorted(graph.nodes[node_id].neighbors)):
        target = graph.nodes[target_id].camera_position
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        absolute_yaw = math.atan2(dy, dx)
        relative = math.degrees(normalize_angle(absolute_yaw - yaw_rad))
        rows.append(
            CandidateSpec(
                candidate_id=candidate_id,
                target_viewpoint_id=target_id,
                relative_heading_deg=relative,
                graph_distance_m=graph.edge_distance(node_id, target_id),
                absolute_yaw_rad=absolute_yaw,
            )
        )
    return tuple(rows)


@dataclass
class GraphEpisodeState:
    graph: MatterportGraph
    episode_id: str
    instruction: str
    current_node: int
    goal_node: int
    goal_radius_m: float
    yaw_rad: float
    shortest_path_m: float
    visited_nodes: list[int] = field(default_factory=list)
    executed_path_m: float = 0.0
    stop_requested: bool = False
    success: bool = False
    stop_fp: int = 0
    stop_fn: int = 0

    @classmethod
    def from_episode(cls, graph: MatterportGraph, episode: dict[str, Any]) -> "GraphEpisodeState":
        episode_id = str(episode.get("benchmark_episode_id") or episode.get("episode_id"))
        _, yaw = habitat_pose_to_isaac(episode["start_position"], episode["start_rotation"])
        start_node = graph.nearest_node(episode["start_position"])
        goal = episode["goals"][0]
        goal_node = graph.nearest_node(goal["position"])
        return cls(
            graph=graph,
            episode_id=episode_id,
            instruction=str(episode["instruction"]["instruction_text"]),
            current_node=start_node,
            goal_node=goal_node,
            goal_radius_m=float(goal["radius"]),
            yaw_rad=yaw,
            shortest_path_m=graph.shortest_distance(start_node, goal_node),
            visited_nodes=[start_node],
        )

    @property
    def ne_m(self) -> float:
        return self.graph.shortest_distance(self.current_node, self.goal_node)

    @property
    def goal_positive(self) -> bool:
        return self.ne_m <= self.goal_radius_m + 1.0e-9

    @property
    def oracle_success(self) -> bool:
        return any(
            self.graph.shortest_distance(node_id, self.goal_node) <= self.goal_radius_m + 1.0e-9
            for node_id in self.visited_nodes
        )

    def move(self, spec: CandidateSpec) -> None:
        if self.stop_requested:
            raise RuntimeError("cannot move after STOP")
        if spec.target_viewpoint_id not in self.graph.nodes[self.current_node].neighbors:
            raise ValueError("selected target is not a current graph neighbor")
        if self.goal_positive:
            self.stop_fn += 1
        self.executed_path_m += self.graph.edge_distance(self.current_node, spec.target_viewpoint_id)
        self.current_node = spec.target_viewpoint_id
        self.yaw_rad = spec.absolute_yaw_rad
        self.visited_nodes.append(self.current_node)

    def stop(self) -> None:
        if self.stop_requested:
            raise RuntimeError("STOP already requested")
        self.stop_requested = True
        self.success = self.goal_positive
        if not self.success:
            self.stop_fp += 1

    def result(self, *, calls: int, failure_reason: str, wall_seconds: float) -> dict[str, Any]:
        if not self.success and self.oracle_success and not self.stop_requested and self.stop_fn == 0:
            self.stop_fn += 1
        spl = float(self.success) * self.shortest_path_m / max(
            self.shortest_path_m, self.executed_path_m, 1.0e-9
        )
        return {
            "episode_id": self.episode_id,
            "scene_id": self.graph.scene_id,
            "success": self.success,
            "oracle_success": self.oracle_success,
            "spl": spl,
            "ne_m": self.ne_m,
            "shortest_path_m": self.shortest_path_m,
            "executed_path_m": self.executed_path_m,
            "stop_requested": self.stop_requested,
            "stop_fp": self.stop_fp,
            "stop_fn": self.stop_fn,
            "model_calls": calls,
            "visited_nodes": list(self.visited_nodes),
            "failure_reason": failure_reason,
            "wall_seconds": wall_seconds,
            "map_setting": "oracle_connectivity_graph",
            "rgb_source": "isaac_render_product",
        }
