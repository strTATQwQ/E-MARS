from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = (
    ROOT / "internvla_t4_sensors/internvla_t4_sensors/sensor_bridge_node.py"
)
LAUNCH = ROOT / "internvla_t4_sensors/launch/t4_cuvslam.launch.py"
T5_CONFIG = ROOT / "configs/internnav_t5/cuvslam_shadow.yaml"


def test_r3_overlay_freezes_explicit_stereo_opt_in_for_isaac_kit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runtime.py"
    manifest = tmp_path / "manifest.json"
    env = os.environ.copy()
    env["INTERNVLA_T4_ENABLE_STEREO_ODOMETRY"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_t4_r3_sensor_runtime_overlay.py"),
            "--source",
            str(ROOT / "scripts/internnav_go2_runtime.py"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    generated = output.read_text(encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert 'os.environ.get("INTERNVLA_T4_ENABLE_STEREO_ODOMETRY"' not in generated
    assert "'1' != \"1\"" in generated
    assert "'1' == \"1\"" in generated
    assert payload["frozen_geometry_sources"]["stereo_odometry"] is True
    assert "frozen_stereo_odometry_enable_switch" in payload["changes"]


def _load_stereo_feed_resolver() -> Any:
    tree = ast.parse(BRIDGE.read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_stereo_feed_enabled"
    )
    namespace: dict[str, Any] = {"os": os}
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(BRIDGE), "exec"),
        namespace,
    )
    return namespace["_stereo_feed_enabled"]


def test_empty_lane_preserves_frozen_t4_stereo_behavior() -> None:
    enabled = _load_stereo_feed_resolver()

    assert enabled(
        requested=False,
        pose_source="external_odometry",
        lane_identity_prefix="",
    )
    assert not enabled(
        requested=True,
        pose_source="ground_truth",
        lane_identity_prefix="",
    )


def test_t5_shadow_requires_explicit_switch_and_exact_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled = _load_stereo_feed_resolver()
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNNAV_T5_ID_PREFIX", "a::")

    assert not enabled(
        requested=False,
        pose_source="external_odometry",
        lane_identity_prefix="a::",
    )
    assert enabled(
        requested=True,
        pose_source="ground_truth",
        lane_identity_prefix="a::",
    )

    monkeypatch.setenv("INTERNNAV_T5_ID_PREFIX", "b::")
    with pytest.raises(RuntimeError, match="exact completion_sim"):
        enabled(
            requested=True,
            pose_source="ground_truth",
            lane_identity_prefix="a::",
        )


def test_bridge_uses_stereo_switch_without_changing_odometry_authority() -> None:
    source = BRIDGE.read_text(encoding="utf-8")

    assert 'self.declare_parameter("enable_stereo_feed", False)' in source
    assert "if self.enable_stereo_feed:" in source
    assert "if not self.enable_stereo_feed:" in source
    assert (
        'if self.pose_source == "external_odometry":\n'
        "            self.create_subscription("
    ) in source
    assert '"stereo_feed_enabled": self.enable_stereo_feed' in source


def test_cuvslam_launch_keeps_false_default_and_accepts_sim_time_override() -> None:
    source = LAUNCH.read_text(encoding="utf-8")

    assert 'LaunchConfiguration("use_sim_time")' in source
    assert "ParameterValue(use_sim_time, value_type=bool)" in source
    assert 'DeclareLaunchArgument(\n                "use_sim_time",' in source
    assert 'default_value="false"' in source
    assert '"use_sim_time": use_sim_time_parameter' in source
    assert '"use_sim_time": False' not in source


def test_cuvslam_launch_isolates_t5_tf_without_changing_empty_t4_namespace() -> None:
    source = LAUNCH.read_text(encoding="utf-8")
    supervisor = (
        ROOT
        / "internvla_t4_sensors/internvla_t4_sensors/odometry_supervisor_node.py"
    ).read_text(encoding="utf-8")

    assert 'node_namespace = LaunchConfiguration("namespace")' in source
    assert "namespace=node_namespace" in source
    assert '("/tf", "tf")' in source
    assert '("/tf_static", "tf_static")' in source
    container = source.split("container = ComposableNodeContainer(", 1)[1]
    assert "namespace=node_namespace" in container
    assert 'remappings=[("/tf", "tf"), ("/tf_static", "tf_static")]' in container
    assert '"namespace",\n                default_value=""' in source
    assert 'Odometry, "visual_slam/tracking/odometry"' in supervisor
    assert 'command.append(f"namespace:={namespace}")' in supervisor


def test_t5_shadow_config_is_gt_pose_with_explicit_stereo_and_sim_time() -> None:
    config = yaml.safe_load(T5_CONFIG.read_text(encoding="utf-8"))
    bridge = config["/**/internvla_go2_controller_bridge"]["ros__parameters"]
    visual_slam = config["/**/visual_slam_node"]["ros__parameters"]

    assert bridge == {
        "enable_stereo_feed": True,
        "pose_source": "ground_truth",
        "use_sim_time": True,
    }
    assert visual_slam == {"use_sim_time": True}
