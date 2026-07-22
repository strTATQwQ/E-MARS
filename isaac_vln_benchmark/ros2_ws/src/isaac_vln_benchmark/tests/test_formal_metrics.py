from __future__ import annotations

from isaac_vln_benchmark.config_loader import scene_by_id
from isaac_vln_benchmark.metrics import shortest_path_length


def test_scene_by_id_selects_exact_variant():
    document = {"scenes": [{"scene_id": "a", "scene_type": "same"}, {"scene_id": "b", "scene_type": "same"}]}
    assert scene_by_id(document, "b")["scene_id"] == "b"


def test_static_grid_shortest_path_detours_around_barrier():
    scene = {
        "scene_id": "detour",
        "robot_start_pose": [0.0, 0.0, 0.0],
        "bounds": [-1.0, -2.0, 5.0, 2.0],
        "obstacles": [{"pose": [2.0, 0.0, 0.5], "size": [0.5, 2.0, 1.0]}],
    }
    length = shortest_path_length(scene, [4.0, 0.0, 0.0], resolution_m=0.1, robot_radius_m=0.2)
    assert length > 4.0
