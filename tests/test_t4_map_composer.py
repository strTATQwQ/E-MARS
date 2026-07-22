from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from t4_completion.map.composer import ComposeRequest, compose_bundle, compose_plan
from t4_completion.map.contract import ContractError, load_and_validate


ROOT = Path(__file__).resolve().parents[1]


def test_default_and_active_failure_resolve_to_same_shadow_core() -> None:
    configs = load_and_validate()
    default = compose_plan(ComposeRequest(), configs)
    failed = compose_plan(
        ComposeRequest(requested_mode="active", nvblox_health="failed"), configs
    )
    assert default["effective_mode"] == "shadow"
    assert failed["effective_mode"] == "shadow"
    assert failed["fallback_applied"] is True
    assert failed["core_nav_sha256"] == default["core_nav_sha256"]
    assert default["default_path"] == {
        "global_map": "static",
        "local_costmap": "lidar_voxel",
        "lidar_topic": "/go2/lidar/points_base",
        "nvblox_mode": "shadow",
    }
    assert default["nvblox_process_started"] is False
    assert failed["nvblox_process_started"] is False
    assert "nvblox_mapper" not in {
        process["role"] for process in default["processes"]
    }


def test_active_ready_is_opt_in_and_keeps_lidar_beside_nvblox() -> None:
    plan = compose_plan(
        ComposeRequest(requested_mode="active", nvblox_health="ready")
    )
    assert plan["effective_mode"] == "active"
    assert plan["fallback_applied"] is False
    assert plan["core_nav_sha256"] == plan["fallback_core_nav_sha256"]
    assert plan["active_runtime_fallback"]["effective_mode_after"] == "shadow"
    assert plan["active_runtime_fallback"]["preserved_layers"] == [
        "voxel_layer",
        "inflation_layer",
    ]
    assert plan["nvblox_process_started"] is True
    nvblox = next(
        process for process in plan["processes"] if process["role"] == "nvblox_mapper"
    )
    assert nvblox["required"] is True


@pytest.mark.parametrize("target", ("real_go2", "hardware_motion", "isaac_host"))
def test_non_simulation_target_is_rejected_without_output(
    tmp_path: Path, target: str
) -> None:
    output = tmp_path / "must not exist"
    with pytest.raises(ContractError):
        compose_bundle(output, ComposeRequest(target=target))
    assert not output.exists()


def test_bundle_is_fresh_deterministic_and_ros_consumable(tmp_path: Path) -> None:
    first = tmp_path / "first bundle"
    second = tmp_path / "second bundle"
    request = ComposeRequest(requested_mode="active", nvblox_health="ready")
    compose_bundle(first, request)
    compose_bundle(second, request)
    names = {
        "nav2_params.yaml",
        "nvblox_params.yaml",
        "smoke_static_map.yaml",
        "launch_plan.json",
        "config_validation.json",
    }
    assert {path.name for path in first.iterdir()} == names
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    plan = json.loads((first / "launch_plan.json").read_text(encoding="utf-8"))
    nav2 = yaml.safe_load((first / "nav2_params.yaml").read_text(encoding="utf-8"))
    nvblox = yaml.safe_load(
        (first / "nvblox_params.yaml").read_text(encoding="utf-8")
    )
    assert plan["online_executed_by_composer"] is False
    assert nav2["local_costmap"]["local_costmap"]["ros__parameters"]["plugins"] == [
        "voxel_layer",
        "nvblox_layer",
        "inflation_layer",
    ]
    assert nvblox["/**"]["ros__parameters"]["use_lidar"] is True
    with pytest.raises(FileExistsError):
        compose_bundle(first, request)


def test_cli_runs_from_unrelated_space_path(tmp_path: Path) -> None:
    cwd = tmp_path / "unrelated cwd with spaces"
    cwd.mkdir()
    output = tmp_path / "rendered bundle with spaces"
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/t4_map_compose.py"),
            "--output-dir",
            str(output),
            "--nvblox-mode",
            "active",
            "--nvblox-health",
            "failed",
        ],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS"
    assert payload["effective_mode"] == "shadow"
