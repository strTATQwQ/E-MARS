"""Frozen 01R producer contract.

Acceptance limits live here rather than in argv or the environment.  Online
entrypoints select one named profile; they cannot replace individual limits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class SensorProfile:
    name: str
    duration_sec: float
    minimum_effective_sec: float
    capture_hz: float
    minimum_rate_hz: float
    ready_deadline_sec: float
    reset_interval_sec: float
    minimum_active_resets: int
    minimum_post_ready_records: int


PROFILES = MappingProxyType(
    {
        "bootstrap": SensorProfile(
            name="bootstrap",
            duration_sec=60.0,
            minimum_effective_sec=50.0,
            capture_hz=20.0,
            minimum_rate_hz=10.0,
            ready_deadline_sec=20.0,
            reset_interval_sec=30.0,
            minimum_active_resets=1,
            minimum_post_ready_records=100,
        ),
        "soak": SensorProfile(
            name="soak",
            duration_sec=600.0,
            minimum_effective_sec=590.0,
            capture_hz=20.0,
            minimum_rate_hz=10.0,
            ready_deadline_sec=20.0,
            reset_interval_sec=120.0,
            minimum_active_resets=4,
            minimum_post_ready_records=1000,
        ),
        "completion_sim": SensorProfile(
            name="completion_sim",
            duration_sec=60.0,
            minimum_effective_sec=5.0,
            capture_hz=20.0,
            minimum_rate_hz=1.0,
            ready_deadline_sec=20.0,
            reset_interval_sec=30.0,
            minimum_active_resets=0,
            minimum_post_ready_records=1,
        ),
        "completion_sim_map": SensorProfile(
            name="completion_sim_map",
            duration_sec=60.0,
            minimum_effective_sec=5.0,
            capture_hz=20.0,
            minimum_rate_hz=1.0,
            ready_deadline_sec=20.0,
            reset_interval_sec=30.0,
            minimum_active_resets=0,
            minimum_post_ready_records=1,
        ),
    }
)

PHYSICS_HZ = 200.0
MAX_FRESHNESS_SEC = 0.35
MAX_GAP_SEC = 0.35
P95_GAP_SEC = 0.20
MIN_VALID_DEPTH_RATIO = 0.10
MIN_VALID_DEPTH_POINTS = 640 * 480 // 10
DEPTH_CLOUD_TILE_STRIDE = 4
DEPTH_CLOUD_WIDTH = 640 // DEPTH_CLOUD_TILE_STRIDE
DEPTH_CLOUD_HEIGHT = 480 // DEPTH_CLOUD_TILE_STRIDE
REQUIRED_STREAMS = ("d435i_rgb", "d435i_depth", "lidar", "pose", "tf")
CAUSAL_ORDER = (
    "/clock",
    "/internnav/sensor_generation",
    "/tf_static:first_in_generation",
    "/tf",
    "sensors",
    "/odom",
    "/go2/imu/data",
    "audit",
)

TOPICS = MappingProxyType(
    {
        "d435i_rgb": "/go2/d435i/color/image_raw",
        "d435i_rgb_info": "/go2/d435i/color/camera_info",
        "d435i_depth": "/go2/d435i/depth/image_rect",
        "d435i_depth_info": "/go2/d435i/depth/camera_info",
        "d435i_points": "/go2/depth/points",
        "lidar_raw": "/internvla_t4/go2/lidar_raw",
        "lidar": "/go2/lidar/points",
        "lidar_base": "/go2/lidar/points_base",
        "front_rgb_raw": "/internvla_t4/go2/front_rgb_raw",
        "front_rgb": "/go2/front_rgb/image_raw",
        "imu_raw": "/internvla_t4/go2/imu_raw",
        "imu": "/go2/imu/data",
        "safety": "/go2/safety/points",
        "odom": "/odom",
        "tf": "/tf",
        "tf_static": "/tf_static",
        "clock": "/clock",
        "generation": "/internnav/sensor_generation",
    }
)

FRAMES = MappingProxyType(
    {
        "base": "base_link",
        "odom": "odom",
        "d435i_color": "go2_d435i_color_optical_frame",
        "d435i_depth": "go2_depth_optical_frame",
        "lidar": "go2_l1_lidar",
        "imu": "go2_imu_link",
        "front_rgb": "go2_front_rgb_optical_frame",
    }
)

FILTER_CONTRACT = MappingProxyType(
    {
        "d435i_color": {
            "prim_suffix": "/base/internvla_camera",
            "resolution": [640, 480],
            "hfov_deg": 69.4,
            "vfov_deg": 42.5,
        },
        "d435i_depth": {
            "prim_suffix": "/base/t4_d435i_depth",
            "resolution": [640, 480],
            "hfov_deg": 87.0,
            "vfov_deg": 58.0,
            "minimum_depth_m": 0.28,
            "maximum_depth_m": 6.0,
            "translation_from_base_m": [0.20, 0.0, 0.20],
            "pitch_down_deg": 20.0,
            "minimum_forward_base_m_exclusive": 0.12,
            "support_plane_margin_m": 0.08,
            "base_half_extents_m": [0.34, 0.18, 0.14],
            "dynamic_link_radii_m": {"thigh": 0.10, "calf": 0.09, "foot": 0.08},
            "required_link_centers": 13,
            "pointcloud_reduction": "4x4_nearest_valid_after_full_resolution_filter",
            "pointcloud_tile_stride": DEPTH_CLOUD_TILE_STRIDE,
            "pointcloud_resolution": [DEPTH_CLOUD_WIDTH, DEPTH_CLOUD_HEIGHT],
            "pointcloud_max_points": DEPTH_CLOUD_WIDTH * DEPTH_CLOUD_HEIGHT,
            "pointcloud_safety_rule": "retain_nearest_accepted_depth_in_every_angular_tile",
        },
        "front_rgb": {
            "prim_suffix": "/base/go2_front_rgb",
            "render_resolution": [320, 240],
            "publish_resolution": [160, 120],
            "hfov_deg": 120.0,
            "vfov_deg": 75.0,
            "translation_from_base_m": [0.29, 0.0, -0.06],
            "pitch_down_deg": 8.0,
            "is_depth_source": False,
        },
        "lidar": {
            "minimum_range_m": 0.10,
            "maximum_range_m": 12.0,
            "minimum_height_base_m": -0.55,
            "maximum_height_base_m": 2.0,
            "base_half_extents_m": [0.36, 0.20, 0.16],
            "dynamic_link_radius_m": 0.12,
            "requires_same_stamp_identity_centers_depth_tf_clock": True,
        },
    }
)

DIAGNOSTIC_GEOMETRY = (
    {
        "name": "near_depth_lidar_box",
        "prim_path": "/World/Diagnostic/near_depth_lidar_box",
        "position_world_m": [1.0, 0.0, 0.15],
        "size_m": [0.30, 0.40, 0.30],
    },
    {
        "name": "far_diagnostic_wall",
        "prim_path": "/World/Diagnostic/far_diagnostic_wall",
        "position_world_m": [2.0, 0.0, 0.50],
        "size_m": [0.10, 2.00, 1.00],
    },
)

DIAGNOSTIC_LIGHT = {
    "name": "sensor_diagnostic_dome",
    "prim_path": "/World/Diagnostic/sensor_diagnostic_dome",
    "type": "DomeLight",
    "intensity": 1200.0,
    "color_temperature_kelvin": 6500.0,
}


def contract_payload() -> dict[str, Any]:
    """Return the canonical, machine-readable frozen producer contract."""

    return {
        "schema_version": 2,
        "worker": "01R",
        "model_free": True,
        "navigation_lifecycle_dependency": False,
        "clock_driver": "independent_monotonic_bounded_loop",
        "physics_hz": PHYSICS_HZ,
        "profiles": {name: asdict(value) for name, value in PROFILES.items()},
        "required_streams": list(REQUIRED_STREAMS),
        "same_positive_sim_timestamp": True,
        "latest_only_capacity": 1,
        "reset_policy": "strictly_increasing_generation_atomic_clear_continuous_world_articulation_state",
        "reset_kind": "continuous_world_articulation_state",
        "safe_stop": {
            "linear_x": 0.0,
            "angular_z": 0.0,
            "emergency_stop": True,
            "applied_every_physics_step": True,
        },
        "freshness_sec": MAX_FRESHNESS_SEC,
        "gap_limits_sec": {"p95_exclusive": P95_GAP_SEC, "max_exclusive": MAX_GAP_SEC},
        "sensor_qos": "best_effort keep_last=4",
        "tf_static_qos": "reliable transient_local keep_last=1",
        "causal_order": list(CAUSAL_ORDER),
        "topics": dict(TOPICS),
        "frames": dict(FRAMES),
        "filter_contract": dict(FILTER_CONTRACT),
        "minimum_valid_depth_ratio": MIN_VALID_DEPTH_RATIO,
        "minimum_valid_depth_points": MIN_VALID_DEPTH_POINTS,
        "diagnostic_geometry": [dict(item) for item in DIAGNOSTIC_GEOMETRY],
        "diagnostic_light": dict(DIAGNOSTIC_LIGHT),
        "rgb_content_gate": {
            "finite": True,
            "nonzero_count_minimum": 1,
            "dynamic_range_minimum_exclusive": 0,
            "reject_black_or_constant": True,
        },
        "process_lifecycle": {
            "ledger": "atomic_immediately_after_spawn_inside_signal_blocked_helper",
            "early_exit": "per_role_poll_to_atomic_first_failure",
            "cleanup": "ledger_pgid_term_wait_kill_remeasure_and_socket_zero",
            "outer_owner_liveness": "bind_mount_shared_exclusive_flock_until_inner_supervisor_zero_probe",
            "outer_launch": "managed_same_pgid_descendant_cleanup_for_all_spawned_roles",
        },
    }


def canonical_contract_bytes() -> bytes:
    return (
        json.dumps(contract_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


CONTRACT_SHA256 = hashlib.sha256(canonical_contract_bytes()).hexdigest()
