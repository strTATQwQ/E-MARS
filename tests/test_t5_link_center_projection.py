from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SENSOR_BRIDGE = (
    ROOT / "internvla_t4_sensors/internvla_t4_sensors/sensor_bridge_node.py"
)


def _load_projection() -> tuple[tuple[str, ...], Any]:
    tree = ast.parse(SENSOR_BRIDGE.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_BRIDGE_LINK_CENTER_NAMES"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name == "_project_bridge_link_centers"
        )
    ]
    namespace: dict[str, Any] = {"math": math}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(SENSOR_BRIDGE), "exec"),
        namespace,
    )
    return namespace["_BRIDGE_LINK_CENTER_NAMES"], namespace[
        "_project_bridge_link_centers"
    ]


def _center(name: str, value: float) -> dict[str, object]:
    return {"name": name, "center_base": [value, value + 0.1, value + 0.2]}


def test_navigation_fast_nineteen_centers_project_to_frozen_thirteen() -> None:
    names, project = _load_projection()
    extras = ("FL_hip", "FR_hip", "RL_hip", "RR_hip", "Head_upper", "Head_lower")
    raw_names = (
        "base",
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "FL_foot",
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "FR_foot",
        "Head_upper",
        "Head_lower",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
        "RL_foot",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
        "RR_foot",
    )
    raw = [_center(name, float(index)) for index, name in enumerate(raw_names)]

    projected = project(raw)

    assert len(raw) == 19
    assert len(names) == len(projected) == 13
    assert all(name not in names for name in extras)
    values_by_name = {
        str(item["name"]): tuple(float(value) for value in item["center_base"])
        for item in raw
    }
    assert projected == [values_by_name[name] for name in names]


def test_projection_is_stable_when_articulation_iteration_order_changes() -> None:
    names, project = _load_projection()
    raw = [_center(name, float(index)) for index, name in enumerate(reversed(names))]
    values_by_name = {
        str(item["name"]): tuple(float(value) for value in item["center_base"])
        for item in raw
    }

    assert project(raw) == [values_by_name[name] for name in names]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nonfinite"])
def test_projection_fails_closed_when_required_center_evidence_is_invalid(
    defect: str,
) -> None:
    names, project = _load_projection()
    raw = [_center(name, float(index)) for index, name in enumerate(names)]
    if defect == "missing":
        raw.pop()
    elif defect == "duplicate":
        raw.append(_center(names[-1], 100.0))
    else:
        raw[0]["center_base"] = [math.nan, 0.0, 0.0]

    with pytest.raises(ValueError):
        project(raw)


def test_projection_only_changes_bridge_publication_not_depth_self_filter_input() -> None:
    source = SENSOR_BRIDGE.read_text(encoding="utf-8")
    publish = source[source.index("    def _publish_r3_sensors(") : source.index(
        "    def _publish_metric_depth("
    )]
    depth = source[source.index("    def _publish_depth(") : source.index(
        "    def _publish_pointcloud("
    )]

    assert "bridge_centers = _project_bridge_link_centers(centers)" in publish
    assert "for center in bridge_centers:" in publish
    assert 'raw_link_centers = request.get("robot_link_centers_base", [])' in depth
    assert "for item in raw_link_centers:" in depth
