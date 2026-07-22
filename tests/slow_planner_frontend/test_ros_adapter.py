from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from slow_planner_frontend.ros_adapter import (
    RosAdapterConfig,
    atomic_write_json,
    project_lidar_cloud,
    project_lidar_imu,
    project_lidar_state,
    project_low_state,
    project_sport_state,
)


def test_config_requires_absolute_topics_and_bounded_preview(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_RESULTS", str(tmp_path))
    value = {
        "frontend": {
            "ros_adapter": {
                "enabled": True,
                "preview_hz": 1.0,
                "state_publish_hz": 2.0,
                "stale_after_s": 3.0,
                "jpeg_quality": 80,
                "paths": {
                    "state_path": "${ROS_RESULTS}/state.json",
                    "camera_manifest_path": "${ROS_RESULTS}/cameras.json",
                    "camera_dir": "${ROS_RESULTS}/cameras",
                },
                "topics": {
                    "go2_front": "/frontvideostream",
                    "low_state": "/lowstate",
                    "sport_mode_state": "/sportmodestate",
                    "lidar_state": "/utlidar/lidar_state",
                    "lidar_imu": "/utlidar/imu",
                    "lidar_cloud": "/utlidar/cloud",
                    "odometry": "/utlidar/robot_odom",
                    "d435_color": "/check/d435/color/image_raw",
                    "d435_depth": "/check/d435/depth/image_rect_raw",
                },
            }
        }
    }
    config = RosAdapterConfig.from_mapping(value)
    assert config.preview_hz == 1.0
    assert config.control_plane is None
    assert config.state_path == (tmp_path / "state.json").resolve()
    value["frontend"]["ros_adapter"]["topics"]["low_state"] = "lowstate"
    with pytest.raises(ValueError, match="absolute"):
        RosAdapterConfig.from_mapping(value)


def test_deployed_config_uses_videohub_rpc_and_stable_usb_paths(
    tmp_path, monkeypatch
):
    import yaml
    from pathlib import Path

    monkeypatch.setenv("T5_LANE_B_RESULTS", str(tmp_path))
    value = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs/frontend.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = RosAdapterConfig.from_mapping(value)
    assert config.go2_camera_source == "videohub_rpc"
    assert config.usb_camera_enabled is True
    assert config.usb_camera_capture_hz == 5.0
    assert config.usb_camera_horizontal_flip is True
    assert tuple(config.usb_camera_devices) == (
        "front_left",
        "front",
        "front_right",
        "rear",
    )
    assert all("/dev/v4l/by-path/" in path for path in config.usb_camera_devices.values())


def test_strict_real_config_enables_only_high_level_control_topics(
    tmp_path, monkeypatch
):
    import yaml
    from pathlib import Path

    monkeypatch.setenv("T5_LANE_B_RESULTS", str(tmp_path))
    value = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs/strict_real_go2.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = RosAdapterConfig.from_mapping(value)
    assert config.control_plane is not None
    assert config.control_plane.instruction_topic == "/user_instruction"
    assert config.control_plane.canonical_instruction_topic == (
        "/internvla/mission/canonical"
    )
    assert config.control_plane.navigation_dispatch_enabled is False
    assert config.control_plane.motion_bridge_enabled is False


def test_atomic_json_replaces_complete_document(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"schema_version": 1, "panel_state": "ready"})
    assert json.loads(path.read_text()) == {
        "panel_state": "ready",
        "schema_version": 1,
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_projects_only_bounded_go2_status_fields():
    message = SimpleNamespace(
        bms_state=SimpleNamespace(soc=86, status=1, cycle=12, current=-100),
        imu_state=SimpleNamespace(
            rpy=[1, 2, 3],
            quaternion=[0, 0, 0, 1],
            gyroscope=[4, 5, 6],
            accelerometer=[7, 8, 9],
            temperature=42,
        ),
        motor_state=[SimpleNamespace(temperature=50, lost=0)] * 20,
        power_v=30.1,
        power_a=1.2,
        foot_force=[1, 2, 3, 4],
        tick=99,
        raw_text="must-not-leak",
    )
    projected = project_low_state(message)
    assert projected["battery"]["soc"] == 86
    assert projected["motors"]["count"] == 20
    assert "raw_text" not in json.dumps(projected)

    sport = project_sport_state(
        SimpleNamespace(
            error_code=0,
            mode=1,
            gait_type=2,
            progress=0.5,
            position=[1, 2, 3],
            velocity=[0.1, 0.2, 0.3],
            yaw_speed=0.4,
            body_height=0.3,
            foot_raise_height=0.1,
            range_obstacle=[1, 2, 3, 4],
        )
    )
    assert sport["motion"]["yaw_speed"] == 0.4

    lidar = project_lidar_state(
        SimpleNamespace(
            error_state=0,
            cloud_frequency=15.0,
            cloud_packet_loss_rate=0.01,
            cloud_size=100,
            imu_frequency=250.0,
            imu_packet_loss_rate=0.0,
        )
    )
    assert lidar["lidar"]["cloud_frequency"] == 15.0

    lidar_imu = project_lidar_imu(
        SimpleNamespace(
            header=SimpleNamespace(frame_id="utlidar_imu"),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            angular_velocity=SimpleNamespace(x=0.1, y=0.2, z=0.3),
            linear_acceleration=SimpleNamespace(x=1.0, y=2.0, z=9.8),
        )
    )
    assert lidar_imu["imu"]["source"] == "utlidar"
    assert lidar_imu["imu"]["rpy"] == [0.0, 0.0, 0.0]

    cloud = project_lidar_cloud(
        SimpleNamespace(
            header=SimpleNamespace(frame_id="utlidar_lidar"),
            width=1440,
            height=1,
            point_step=24,
            row_step=34560,
        )
    )
    assert cloud["lidar"]["point_count"] == 1440


def test_module_import_does_not_require_ros_or_cv_runtime():
    # rclpy/cv_bridge/cv2 imports must remain lazy so Uvicorn and offline tests
    # never join or initialize the ROS graph.
    import slow_planner_frontend.ros_adapter as module

    assert callable(module.main)
