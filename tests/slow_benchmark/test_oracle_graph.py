import json
import math

import pytest

from slow_benchmark.oracle_graph import MatterportGraph, habitat_pose_to_isaac


def _row(image_id, xyz, included, links):
    pose = [1.0, 0.0, 0.0, xyz[0], 0.0, 1.0, 0.0, xyz[1], 0.0, 0.0, 1.0, xyz[2], 0.0, 0.0, 0.0, 1.0]
    return {"image_id": image_id, "included": included, "pose": pose, "unobstructed": links}


def test_load_snap_path_and_local_geometry(tmp_path):
    rows = [
        _row("a", (0.0, 0.0, 1.4), True, [False, True, False, False]),
        _row("b", (2.0, 0.0, 1.4), True, [True, False, False, True]),
        _row("excluded", (99.0, 99.0, 99.0), False, [False] * 4),
        _row("c", (2.0, 3.0, 2.4), True, [False, True, False, False]),
    ]
    path = tmp_path / "scene_connectivity.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    graph = MatterportGraph.load(path)
    assert tuple(graph.nodes) == (0, 1, 3)
    assert graph.nearest_node((0.0, 0.0, 0.0)) == 0
    assert graph.shortest_path(0, 3) == (0, 1, 3)
    assert graph.shortest_distance(0, 3) == pytest.approx(2.0 + math.sqrt(10.0))
    candidate = graph.candidate_geometry(0, 0.0)[0]
    assert candidate[0] == 1
    assert candidate[1] == pytest.approx((2.0, 0.0))
    assert candidate[3] == pytest.approx(0.0)


def test_habitat_pose_basis_change():
    position, yaw = habitat_pose_to_isaac((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    assert position == (1.0, -3.0, 2.0)
    assert yaw == pytest.approx(math.pi / 2.0)


def test_non_reciprocal_connectivity_is_rejected(tmp_path):
    rows = [
        _row("a", (0.0, 0.0, 1.4), True, [False, True]),
        _row("b", (1.0, 0.0, 1.4), True, [False, False]),
    ]
    path = tmp_path / "bad_connectivity.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="not reciprocal"):
        MatterportGraph.load(path)
