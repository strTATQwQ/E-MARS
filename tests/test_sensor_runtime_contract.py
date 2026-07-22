from __future__ import annotations

import ast
import math
from pathlib import Path

from sensor_runtime.contract import (
    DIAGNOSTIC_GEOMETRY,
    DIAGNOSTIC_LIGHT,
    FILTER_CONTRACT,
    contract_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_frozen_filter_contract_is_exact_and_machine_readable() -> None:
    value = contract_payload()["filter_contract"]
    assert value["d435i_color"] == {
        "prim_suffix": "/base/internvla_camera",
        "resolution": [640, 480],
        "hfov_deg": 69.4,
        "vfov_deg": 42.5,
    }
    depth = value["d435i_depth"]
    assert depth["prim_suffix"] == "/base/t4_d435i_depth"
    assert (depth["hfov_deg"], depth["vfov_deg"]) == (87.0, 58.0)
    assert (depth["minimum_depth_m"], depth["maximum_depth_m"]) == (0.28, 6.0)
    assert depth["minimum_forward_base_m_exclusive"] == 0.12
    assert depth["support_plane_margin_m"] == 0.08
    assert depth["base_half_extents_m"] == [0.34, 0.18, 0.14]
    assert depth["dynamic_link_radii_m"] == {"thigh": 0.10, "calf": 0.09, "foot": 0.08}
    assert depth["required_link_centers"] == 13
    assert depth["pointcloud_reduction"] == "4x4_nearest_valid_after_full_resolution_filter"
    assert depth["pointcloud_tile_stride"] == 4
    assert depth["pointcloud_resolution"] == [160, 120]
    assert depth["pointcloud_max_points"] == 19200
    assert depth["pointcloud_safety_rule"] == "retain_nearest_accepted_depth_in_every_angular_tile"
    assert value["front_rgb"]["publish_resolution"] == [160, 120]


def test_isaac_backend_uses_independent_color_depth_and_actual_render_metadata() -> None:
    source = _source("sensor_runtime/isaac_backend.py")
    pipeline_source = _source("sensor_runtime/render_pipeline.py")
    tree = ast.parse(source)
    assert 'prim_path="/World/Go2/base/internvla_camera"' in source
    assert 'prim_path="/World/Go2/base/t4_d435i_depth"' in source
    assert 'rgb = self._rgb8(color_frame.get("rgb")' in source
    assert 'depth_frame.get("distance_to_image_plane")' in source
    assert 'depth_frame.get("rgba")' not in source
    assert '"rendering_frame"' in source and '"rendering_time"' in source
    assert '"referenceTimeNumerator"' in source
    assert '"referenceTimeDenominator"' in source
    assert "SingleRenderPipeline" in source
    assert "_negotiate_render_latency()" in source
    assert "render_latency_negotiated_before_workload" in source
    assert source.index(
        "self._render_latency_negotiation = self._negotiate_render_latency()"
    ) < source.index("self._pace_origin = time.monotonic()")
    assert "camera render metadata did not advance across one-render ticks" in pipeline_source
    assert "camera frame and delayed SafeStep timestamp are not identical" in pipeline_source
    assert "self.world.step(render=False)" in source
    assert "self.world.render()" in source
    assert "apply_action" in source and "ArticulationAction" in source
    assert "set_joint_position_targets" not in source
    assert source.count("frequency=-1") == 3
    assert "get_current_stage" in source and "GetPrimAtPath(path).IsValid()" in source
    assert 'f"/World/Go2/{name}"' in source
    assert '/World/Go2/base/{name}' not in source
    assert "eval.py" not in source
    # Three distinct Camera constructor calls remain present in the AST.
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Camera"]
    assert len(calls) == 3


def test_safe_stop_reset_and_close_do_not_increment_physics_coverage() -> None:
    source = _source("sensor_runtime/isaac_backend.py")
    assert "_apply_safe_stop(count_physics_step=False)" in source
    assert "_apply_safe_stop(count_physics_step=True)" in source
    assert source.count("_apply_safe_stop(count_physics_step=False)") >= 2
    assert "render-only synchronization advanced physics time" in source


def test_camera_info_full_contract_and_front_publish_resolution_are_locked() -> None:
    sidecar = _source("sensor_runtime/ros_sidecar.py")
    recorder = _source("sensor_runtime/downstream_recorder.py")
    bridge = _source("go2_sensor_bridge/go2_sensor_bridge/bridge_node.py")
    assert "message.d = [0.0] * 5" in sidecar
    assert "message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]" in sidecar
    assert "message.p = [fx, 0.0, cx" in sidecar
    assert 'width=160, height=120' in recorder
    assert 'int(message.width) != 160' in bridge
    assert '"tf_lookup": lookup' in recorder and "lookup_transform" in recorder
    assert "TransformListener" not in recorder
    assert "ingest_observed_transforms" in recorder
    assert '"downstream_pending_failure.json"' in recorder
    assert 'row["tf_can_transform"] = availability' in recorder
    assert "actual downstream batch incomplete for 0.35 seconds" in recorder


def test_bridge_is_ordered_finite_and_pairs_front_before_publish() -> None:
    source = _source("go2_sensor_bridge/go2_sensor_bridge/bridge_node.py")
    assert '"front", "front_info"' in source
    assert "stamp = min(self._pending)" in source
    assert "self._publish_complete(stamp, entry)" in source
    assert "all(math.isfinite(value) for value in center)" in source
    assert "self.front_pub.publish(parts[\"front\"])" in source
    assert "self._last_processed != target or self._pending" in source
    assert "new identity arrived before prior batch was complete" not in source


def test_diagnostic_geometry_is_footprint_safe_and_visible_to_depth_and_lidar() -> None:
    camera = (0.2, 0.0, 0.62)
    lidar = (0.25, 0.0, 0.60)
    jointly_visible = 0
    for item in DIAGNOSTIC_GEOMETRY:
        x, y, z = item["position_world_m"]
        sx, sy, sz = item["size_m"]
        assert x - sx / 2 > 0.36 or abs(y) - sy / 2 > 0.20
        horizontal = math.hypot(x - camera[0], y - camera[1])
        camera_down = math.degrees(math.atan2(camera[2] - z, horizontal))
        in_depth = abs(camera_down - 20.0) <= 58.0 / 2.0 and abs(math.degrees(math.atan2(y, x - camera[0]))) <= 87.0 / 2.0
        lidar_horizontal = math.hypot(x - lidar[0], y - lidar[1])
        low, high = z - sz / 2, z + sz / 2
        lidar_elevations = [math.degrees(math.atan2(value - lidar[2], lidar_horizontal)) for value in (low, high)]
        in_lidar = max(lidar_elevations) >= -15.0 and min(lidar_elevations) <= 20.0
        jointly_visible += int(in_depth and in_lidar)
    assert jointly_visible >= 1
    source = _source("sensor_runtime/isaac_backend.py")
    assert "FixedCuboid" in source and "DIAGNOSTIC_GEOMETRY" in source
    assert DIAGNOSTIC_LIGHT["type"] == "DomeLight"
    assert DIAGNOSTIC_LIGHT["intensity"] > 0
    assert contract_payload()["diagnostic_light"] == DIAGNOSTIC_LIGHT
    assert "UsdLux.DomeLight.Define" in source
