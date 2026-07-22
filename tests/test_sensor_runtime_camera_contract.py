from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sensor_runtime.camera_contract import matrices_from_fov, require_camera_matrices


def test_front_camera_info_locks_area_downsample_intrinsics_and_all_matrices() -> None:
    expected = matrices_from_fov(160, 120, 120.0, 75.0)
    assert expected["k"][2] == 79.5 and expected["k"][5] == 59.5
    assert expected["p"][2] == 79.5 and expected["p"][6] == 59.5
    require_camera_matrices(expected, expected)
    for name, index, delta in (("k", 2, 0.25), ("p", 6, -0.25), ("k", 0, 0.1)):
        tampered = {key: list(value) for key, value in expected.items()}
        tampered[name][index] += delta
        with pytest.raises(ValueError, match="matrix"):
            require_camera_matrices(tampered, expected)


def test_bridge_multithreaded_control_callbacks_take_the_shared_rlock() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "go2_sensor_bridge/go2_sensor_bridge/bridge_node.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Go2SensorBridge"
        for node in node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in ("_ingest", "_on_identity", "_watchdog", "_snapshot", "_summary", "_fail"):
        function = functions[name]
        assert any(
            isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Attribute)
                and item.context_expr.attr == "_lock"
                for item in node.items
            )
            for node in ast.walk(function)
        ), name
