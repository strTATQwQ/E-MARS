#!/usr/bin/env python3
"""Extend the T4 runtime with PhysX LiDAR, front RGB, D435i RGB, and IMU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from t5_revc_sensor_contract import (
    DEFAULT_CONTRACT as DEFAULT_REVC_CONTRACT,
    load_revc_contract,
    t5_revc_feature_scope,
)


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one R3 runtime token, found {count}: {old!r}")
    return text.replace(old, new)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    base_builder = Path(__file__).with_name("build_t4_sensor_runtime_overlay.py")
    d435i_enabled = os.environ.get("INTERNVLA_T4_R3_ENABLE_D435I", "1") == "1"
    lidar_enabled = os.environ.get("INTERNVLA_T4_R3_ENABLE_LIDAR", "1") == "1"
    lidar_ray_count = int(os.environ.get("INTERNVLA_T4_R3_LIDAR_RAY_COUNT", "1440"))
    if lidar_ray_count not in {0, 360, 720, 1440}:
        raise RuntimeError("unsupported T4 R3 LiDAR ray count")
    if lidar_enabled != (lidar_ray_count > 0):
        raise RuntimeError("LiDAR enable and ray-count settings disagree")
    lidar_width = lidar_ray_count // 8 if lidar_enabled else 0
    rgb_ipc_enabled = os.environ.get("INTERNVLA_T4_R3_ENABLE_RGB_IPC", "1") == "1"
    fault_profile = os.environ.get("INTERNNAV_T5_FAULT_INJECTION_PROFILE", "off")
    if fault_profile not in {"off", "completion_sim_minimal_v1"}:
        raise RuntimeError("unsupported T5 fault injection profile")
    fault_injection_enabled = fault_profile == "completion_sim_minimal_v1"
    if fault_injection_enabled and (
        os.environ.get("INTERNNAV_RUNTIME_POLICY") != "completion_sim"
        or os.environ.get("INTERNNAV_SIMULATION_TARGET") != "isaac"
        or os.environ.get("INTERNNAV_T5_LANE") not in {"a", "b"}
    ):
        raise RuntimeError(
            "fault overlay is restricted to an isolated T5 completion_sim lane"
        )
    stereo_odometry_override = os.environ.get(
        "INTERNVLA_T4_ENABLE_STEREO_ODOMETRY"
    )
    if stereo_odometry_override not in {None, "0", "1"}:
        raise RuntimeError("stereo odometry enable must be 0 or 1")
    stereo_odometry_enabled = stereo_odometry_override == "1"
    t5_revc_scope = t5_revc_feature_scope()
    t5_sensor_extensions_enabled = t5_revc_scope is not None
    revc_config_path: Path | None = None
    revc_contract: dict[str, object] | None = None
    imu_contract_path: Path | None = None
    imu_contract: dict[str, object] | None = None
    if t5_sensor_extensions_enabled:
        revc_config_path = Path(
            os.environ.get(
                "INTERNVLA_T5_REVC_CAMERA_CONFIG", str(DEFAULT_REVC_CONTRACT)
            )
        ).expanduser().resolve()
        revc_contract = load_revc_contract(revc_config_path)
        imu_contract_path = (
            Path(__file__).resolve().parents[1]
            / "configs/internnav_t5/simulated_imu_linear_acceleration.json"
        )
        imu_contract = json.loads(imu_contract_path.read_text(encoding="utf-8"))
        if (
            imu_contract.get("contract_id")
            != "internnav-t5-sim-imu-linear-acceleration-v1"
            or imu_contract.get("purpose") != "sensor_contract_precondition_only"
            or imu_contract.get("lio_backend_implemented") is not False
            or imu_contract.get("derivative_clock")
            != "consecutive_normal_sample_sim_stamps"
            or imu_contract.get("invalid_sample_behavior")
            != "clear_baseline_and_require_two_new_consecutive_normal_samples"
            or imu_contract.get("published_sample_stamp_key")
            != "go2_imu_linear_acceleration_sample_sim_stamp_ns"
        ):
            raise RuntimeError("invalid simulated IMU linear-acceleration contract")
    with tempfile.TemporaryDirectory(prefix="t4_r3_runtime_") as temporary:
        base_output = Path(temporary) / "base_overlay.py"
        base_manifest = Path(temporary) / "base_manifest.json"
        subprocess.run(
            [
                sys.executable,
                str(base_builder),
                "--source",
                str(source),
                "--output",
                str(base_output),
                "--manifest",
                str(base_manifest),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        text = base_output.read_text(encoding="utf-8")
        inherited = json.loads(base_manifest.read_text(encoding="utf-8"))

    if fault_injection_enabled:
        text = _replace_once(
            text,
            "import time\n",
            "import time\n"
            "from internvla_ros2.fault_injection import FaultControlReader\n",
        )

    if t5_sensor_extensions_enabled:
        text = _replace_once(
            text,
            "import base64\nimport math\nimport json",
            "import base64\nimport hashlib\nimport math\nimport json\nimport struct\n"
            "from numbers import Integral, Real",
        )
        text = _replace_once(
            text,
            "    from internnav.env.utils.internutopia_extension.configs.sensors import VLNCameraCfg\n",
            "    from internnav.env.utils.internutopia_extension.configs.sensors import VLNCameraCfg\n"
            "    from internnav.env.utils.internutopia_extension.sensors.vln_camera import VLNCamera\n",
        )
        text = _replace_once(
            text,
            "    original_get_rgb_depth = VLNEvalTask.get_rgb_depth\n",
            '''    original_restore_sensor_info = VLNCamera.restore_sensor_info

    def restore_sensor_info_with_revc_reference_time(self: Any) -> None:
        original_restore_sensor_info(self)
        if not str(self.config.name).startswith("t5_revc_"):
            return
        import omni.replicator.core as rep

        camera = getattr(self, "_camera", None)
        render_product = getattr(camera, "rp", None)
        annotators = getattr(camera, "rp_annotators", None)
        if render_product is None or not isinstance(annotators, dict):
            raise RuntimeError(
                f"Rev-C sensor {self.config.name} has no Replicator render product"
            )
        reference_time = rep.AnnotatorRegistry.get_annotator("ReferenceTime")
        reference_time.attach(render_product)
        # IsaacsimCamera.cleanup() detaches every entry in this mapping, so the
        # T5-only annotator follows the camera across episode resets without
        # leaking a render-product attachment.
        annotators["_t5_revc_reference_time"] = reference_time

    VLNCamera.restore_sensor_info = restore_sensor_info_with_revc_reference_time

    original_get_rgb_depth = VLNEvalTask.get_rgb_depth
''',
        )
        text = _replace_once(
            text,
            "            self.obstacle_targets: list[list[float]] = []\n"
            "            self.bootstrap_generation = -1",
            "            self.obstacle_targets: list[list[float]] = []\n"
            "            self.bootstrap_generation = -1\n"
            "            self._imu_previous_normal_sample = None\n"
            "            self._revc_last_request_id = \"\"\n"
            "            self._revc_last_preview_monotonic = -math.inf",
        )
        text = _replace_once(
            text,
            "            self.obstacles.reset()\n"
            "            self.obstacle_targets = []\n"
            "            _stop_root(self.robot)",
            "            self.obstacles.reset()\n"
            "            self.obstacle_targets = []\n"
            "            self._imu_previous_normal_sample = None\n"
            "            self._revc_last_request_id = \"\"\n"
            "            self._revc_last_preview_monotonic = -math.inf\n"
            "            _stop_root(self.robot)",
        )

    methods = '''        def _sample_go2_lidar(self, *, state_only: bool = False) -> dict[str, Any]:
            if os.environ.get("INTERNVLA_T4_R3_ENABLE_LIDAR", "1") != "1":
                return {}
            # State-only bootstrap has no valid episode identity; defer the
            # synchronous ray batch until a normal control update can use it.
            if state_only:
                return {}
            if self.control_update_index % self.depth_control_interval:
                return {}
            import omni.physx

            articulation = self.robot.articulation.unwrap()
            position, rotation = articulation.get_world_pose()
            position = np.asarray(position, dtype=np.float64)
            rotation = np.asarray(rotation, dtype=np.float64)
            if position.shape != (3,) or rotation.shape != (4,):
                return {}

            def rotate_wxyz(vector: np.ndarray) -> np.ndarray:
                w, x, y, z = [float(value) for value in rotation]
                qvec = np.asarray([x, y, z], dtype=np.float64)
                twice_cross = 2.0 * np.cross(qvec, vector)
                return vector + w * twice_cross + np.cross(qvec, twice_cross)

            origin_base = np.asarray([0.25, 0.0, 0.18], dtype=np.float64)
            origin_world = position + rotate_wxyz(origin_base)
            interface = omni.physx.get_physx_scene_query_interface()
            width = 180
            elevations = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
            minimum_range = 0.10
            maximum_range = 12.0
            points: list[list[float] | None] = []
            for elevation_deg in elevations:
                elevation = math.radians(elevation_deg)
                horizontal = math.cos(elevation)
                for column in range(width):
                    azimuth = 2.0 * math.pi * column / width
                    direction_base = np.asarray(
                        [
                            horizontal * math.cos(azimuth),
                            horizontal * math.sin(azimuth),
                            math.sin(elevation),
                        ],
                        dtype=np.float64,
                    )
                    direction_world = rotate_wxyz(direction_base)
                    hit = interface.raycast_closest(
                        tuple(float(value) for value in origin_world),
                        tuple(float(value) for value in direction_world),
                        maximum_range,
                    )
                    if not bool(hit.get("hit", False)):
                        points.append(None)
                        continue
                    distance = float(hit.get("distance", math.inf))
                    if not math.isfinite(distance) or distance < minimum_range:
                        points.append(None)
                        continue
                    # PointCloud2 is expressed in the LiDAR frame.  The fixed
                    # base_link extrinsic is represented only by TF.
                    point = distance * direction_base
                    points.append([float(value) for value in point])
            return {
                "go2_lidar_width": width,
                "go2_lidar_height": len(elevations),
                "go2_lidar_points_lidar": points,
                "go2_lidar_capture_wall_time_unix": time.time(),
                "go2_lidar_minimum_range_m": minimum_range,
                "go2_lidar_maximum_range_m": maximum_range,
            }

        def _sample_r3_images_and_imu(self, *, state_only: bool = False) -> dict[str, Any]:
            # State-only controller handshakes must remain small.  Images are
            # observational data and are sent on the next normal update.
            if (
                state_only
                or os.environ.get("INTERNVLA_T4_R3_ENABLE_RGB_IPC", "1") != "1"
                or self.control_update_index % self.depth_control_interval
            ):
                return {}
            payload: dict[str, Any] = {}
            for payload_name, sensor_name in (
                ("d435i_rgb", "t4_d435i_depth"),
                ("go2_front_rgb", "go2_front_rgb"),
            ):
                sensor = self.robot.sensors.get(sensor_name)
                if sensor is None:
                    continue
                rgba = np.asarray(sensor.get_data().get("rgba"))
                if rgba.ndim != 3 or rgba.shape[2] < 3:
                    continue
                rgb = np.ascontiguousarray(rgba[:, :, :3].astype(np.uint8))
                # The controller IPC bound is a safety invariant.  Keep both
                # review-only RGB streams at 160x120 instead of relaxing it.
                source_height, source_width = rgb.shape[:2]
                row_index = np.linspace(0, source_height - 1, 120, dtype=np.int64)
                column_index = np.linspace(0, source_width - 1, 160, dtype=np.int64)
                rgb = np.ascontiguousarray(rgb[row_index][:, column_index, :])
                height, width = rgb.shape[:2]
                compressed = zlib.compress(rgb.tobytes(order="C"), level=1)
                payload[f"{payload_name}_width"] = int(width)
                payload[f"{payload_name}_height"] = int(height)
                payload[f"{payload_name}_rgb8_zlib_b64"] = base64.b64encode(compressed).decode("ascii")
            return payload

'''
    feature_methods = '''        def _sample_simulated_imu_linear_acceleration(
            self,
            linear_velocity_world: list[float],
            pose_wxyz: list[float],
            sim_stamp_ns: int | None,
            *,
            state_only: bool = False,
        ) -> dict[str, Any]:
            unavailable = {
                "go2_imu_linear_acceleration_available": False,
                "go2_imu_linear_acceleration_contract": (
                    "internnav-t5-sim-imu-linear-acceleration-v1"
                ),
            }
            # Every gap in the normal-sample sequence invalidates the finite-
            # difference baseline.  The next legal sample must seed a fresh
            # pair and remain unavailable.
            if state_only:
                self._imu_previous_normal_sample = None
                return unavailable
            if (
                not isinstance(linear_velocity_world, list)
                or len(linear_velocity_world) != 3
                or not isinstance(pose_wxyz, list)
                or len(pose_wxyz) != 7
            ):
                self._imu_previous_normal_sample = None
                return unavailable
            try:
                current = np.asarray(linear_velocity_world, dtype=np.float64)
                rotation = np.asarray(pose_wxyz[3:7], dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                self._imu_previous_normal_sample = None
                return unavailable
            if (
                not isinstance(sim_stamp_ns, int)
                or isinstance(sim_stamp_ns, bool)
                or sim_stamp_ns <= 0
                or current.shape != (3,)
                or rotation.shape != (4,)
                or not np.isfinite(current).all()
                or not np.isfinite(rotation).all()
            ):
                self._imu_previous_normal_sample = None
                return unavailable
            rotation_norm = float(np.linalg.norm(rotation))
            if not math.isfinite(rotation_norm) or rotation_norm <= 1.0e-9:
                self._imu_previous_normal_sample = None
                return unavailable
            previous = self._imu_previous_normal_sample
            if previous is None:
                self._imu_previous_normal_sample = (int(sim_stamp_ns), current.copy())
                return unavailable
            previous_stamp_ns, previous_velocity = previous
            dt = (int(sim_stamp_ns) - int(previous_stamp_ns)) / 1_000_000_000.0
            if (
                not math.isfinite(dt)
                or dt <= 0.0
                or previous_velocity.shape != (3,)
                or not np.isfinite(previous_velocity).all()
            ):
                self._imu_previous_normal_sample = None
                return unavailable
            rotation = rotation / rotation_norm
            acceleration_world = (current - previous_velocity) / dt
            # ROS sensor_msgs/Imu carries accelerometer specific force.  With
            # ENU gravity (0, 0, -g), a stationary level sensor reads +g on Z.
            specific_force_world = acceleration_world - np.asarray(
                [0.0, 0.0, -9.80665], dtype=np.float64
            )
            w, x, y, z = [float(value) for value in rotation]
            conjugate_vector = np.asarray([-x, -y, -z], dtype=np.float64)
            twice_cross = 2.0 * np.cross(conjugate_vector, specific_force_world)
            specific_force_body = (
                specific_force_world
                + w * twice_cross
                + np.cross(conjugate_vector, twice_cross)
            )
            if not np.isfinite(specific_force_body).all():
                self._imu_previous_normal_sample = None
                return unavailable
            self._imu_previous_normal_sample = (int(sim_stamp_ns), current.copy())
            return {
                "go2_imu_linear_acceleration_available": True,
                "go2_imu_linear_acceleration_mps2": specific_force_body.tolist(),
                "go2_imu_linear_acceleration_sample_dt_sec": dt,
                "go2_imu_linear_acceleration_sample_sim_stamp_ns": int(sim_stamp_ns),
                "go2_imu_linear_acceleration_contract": (
                    "internnav-t5-sim-imu-linear-acceleration-v1"
                ),
                "go2_imu_linear_acceleration_source": (
                    "sim_time_finite_difference_articulation_velocity"
                ),
            }

        @staticmethod
        def _encode_revc_png(rgb: np.ndarray) -> bytes:
            height, width, channels = rgb.shape
            if channels != 3 or rgb.dtype != np.uint8:
                raise ValueError("Rev-C PNG input must be RGB8")

            def chunk(kind: bytes, data: bytes) -> bytes:
                checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
                return (
                    struct.pack(">I", len(data))
                    + kind
                    + data
                    + struct.pack(">I", checksum)
                )

            scanlines = b"".join(
                b"\\x00" + np.ascontiguousarray(row).tobytes(order="C")
                for row in rgb
            )
            return (
                b"\\x89PNG\\r\\n\\x1a\\n"
                + chunk(
                    b"IHDR",
                    struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
                )
                + chunk(b"IDAT", zlib.compress(scanlines, level=3))
                + chunk(b"IEND", b"")
            )

        @staticmethod
        def _revc_render_metadata(
            frame: dict[str, Any], sensor_name: str
        ) -> tuple[int, int, float]:
            if set(frame).isdisjoint({"rendering_frame", "rendering_time"}):
                raise RuntimeError(
                    f"Rev-C sensor {sensor_name} lacks real render metadata"
                )
            if "rendering_frame" not in frame or "rendering_time" not in frame:
                raise RuntimeError(
                    f"Rev-C sensor {sensor_name} render metadata is incomplete"
                )
            identity = frame["rendering_frame"]
            if not isinstance(identity, dict) or set(identity) != {
                "referenceTimeNumerator",
                "referenceTimeDenominator",
            }:
                raise RuntimeError(
                    f"Rev-C sensor {sensor_name} ReferenceTime schema is invalid"
                )
            numerator = identity["referenceTimeNumerator"]
            denominator = identity["referenceTimeDenominator"]
            if (
                isinstance(numerator, bool)
                or isinstance(denominator, bool)
                or not isinstance(numerator, Integral)
                or not isinstance(denominator, Integral)
                or numerator <= 0
                or denominator <= 0
                or numerator > 2**53
                or denominator > 2**53
            ):
                raise RuntimeError(
                    f"Rev-C sensor {sensor_name} ReferenceTime value is invalid"
                )
            render_time = frame["rendering_time"]
            if (
                isinstance(render_time, bool)
                or not isinstance(render_time, Real)
                or not math.isfinite(float(render_time))
                or float(render_time) < 0.0
                or not math.isclose(
                    float(render_time),
                    int(numerator) / int(denominator),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ):
                raise RuntimeError(
                    f"Rev-C sensor {sensor_name} rendering_time is invalid"
                )
            return int(numerator), int(denominator), float(render_time)

        def _revc_reference_time_metadata(
            self, sensor: Any, sensor_name: str
        ) -> tuple[int, int, float]:
            camera = getattr(sensor, "_camera", None)
            annotators = getattr(camera, "rp_annotators", None)
            if not isinstance(annotators, dict):
                raise RuntimeError(
                    f"Rev-C sensor {sensor_name} has no Replicator annotators"
                )
            reference_time = annotators.get("_t5_revc_reference_time")
            if reference_time is None:
                raise RuntimeError(
                    f"Rev-C sensor {sensor_name} lacks real ReferenceTime annotator"
                )
            identity = reference_time.get_data()
            if not isinstance(identity, dict):
                raise RuntimeError(
                    f"Rev-C sensor {sensor_name} returned no ReferenceTime mapping"
                )
            numerator = identity.get("referenceTimeNumerator")
            denominator = identity.get("referenceTimeDenominator")
            if (
                isinstance(numerator, bool)
                or isinstance(denominator, bool)
                or not isinstance(numerator, Integral)
                or not isinstance(denominator, Integral)
                or numerator <= 0
                or denominator <= 0
            ):
                raise RuntimeError(
                    f"Rev-C sensor {sensor_name} ReferenceTime value is invalid"
                )
            from isaacsim.core.simulation_manager import SimulationManager

            rendering_time = (
                SimulationManager._simulation_manager_interface
                .get_simulation_time_at_time((int(numerator), int(denominator)))
            )
            return self._revc_render_metadata(
                {
                    "rendering_frame": identity,
                    "rendering_time": rendering_time,
                },
                sensor_name,
            )

        @staticmethod
        def _revc_require_pause_stable_render(
            before: tuple[int, int, float],
            after: tuple[int, int, float],
            sensor_name: str,
        ) -> None:
            # ReferenceTime identifies simulation time, not a unique render
            # submission.  A synchronous pause_timeline render completes a new
            # image without advancing that time, so equality is required here.
            # rep.orchestrator.step(wait_for_render=True) is the completion
            # barrier; the real post-barrier ReferenceTime binds all four
            # annotator payloads to the same paused simulation state.
            if after != before:
                raise RuntimeError(
                    f"Rev-C sensor {sensor_name} ReferenceTime changed at paused "
                    "render barrier"
                )

        def _sample_revc_snapshot(
            self,
            *,
            episode_id: str,
            reset_generation: int,
            sequence_id: int,
            sim_stamp_ns: int | None,
            state_only: bool = False,
        ) -> dict[str, Any]:
            if state_only:
                return {}
            expected_lane = __T5_REVC_LANE__
            expected_prefix = __T5_REVC_IDENTITY_PREFIX__
            try:
                result_root = Path(__T5_REVC_RESULT_ROOT__).resolve()
            except (OSError, RuntimeError) as error:
                return {
                    "revc_snapshot_status": "WARN_LANE_SCOPE_MISMATCH",
                    "revc_snapshot_error": str(error)[:240],
                }

            def scoped_path(candidate: Path, label: str) -> Path:
                resolved = candidate.expanduser().resolve()
                if not resolved.is_relative_to(result_root):
                    raise RuntimeError(
                        f"Rev-C {label} escapes the frozen lane result root"
                    )
                return resolved

            runtime_result_root_text = os.environ.get("INTERNVLA_T4_RESULT_ROOT", "")
            try:
                runtime_result_root = (
                    Path(runtime_result_root_text).expanduser().resolve()
                    if runtime_result_root_text
                    else None
                )
            except (OSError, RuntimeError) as error:
                return {
                    "revc_snapshot_status": "WARN_LANE_SCOPE_MISMATCH",
                    "revc_snapshot_error": str(error)[:240],
                }
            if (
                os.environ.get("INTERNNAV_T5_LANE", "") != expected_lane
                or os.environ.get("INTERNNAV_T5_ID_PREFIX", "") != expected_prefix
                or runtime_result_root != result_root
                or not str(episode_id).startswith(expected_prefix)
            ):
                return {"revc_snapshot_status": "WARN_LANE_SCOPE_MISMATCH"}
            if (
                not isinstance(sim_stamp_ns, int)
                or isinstance(sim_stamp_ns, bool)
                or sim_stamp_ns <= 0
            ):
                return {"revc_snapshot_status": "WARN_INVALID_SIM_STAMP"}
            try:
                request_path = scoped_path(
                    Path(
                        os.environ.get(
                            "INTERNVLA_T5_REVC_SNAPSHOT_REQUEST_PATH",
                            str(result_root / "revc_snapshot.request.json"),
                        )
                    ),
                    "request path",
                )
                ack_path = scoped_path(
                    Path(
                        os.environ.get(
                            "INTERNVLA_T5_REVC_SNAPSHOT_ACK_PATH",
                            str(result_root / "revc_snapshot.ack.json"),
                        )
                    ),
                    "ack path",
                )
            except (OSError, RuntimeError) as error:
                return {
                    "revc_snapshot_status": "WARN_PATH_SCOPE_MISMATCH",
                    "revc_snapshot_error": str(error)[:240],
                }
            if (
                request_path.name != "revc_snapshot.request.json"
                or ack_path.name != "revc_snapshot.ack.json"
            ):
                return {"revc_snapshot_status": "WARN_PATH_SCOPE_MISMATCH"}
            if not request_path.is_file():
                return {}
            try:
                claim_path = scoped_path(
                    request_path.with_name(
                        f".{request_path.name}.inflight.{os.getpid()}."
                        f"{self.control_update_index}"
                    ),
                    "claimed request path",
                )
            except (OSError, RuntimeError) as error:
                return {
                    "revc_snapshot_status": "WARN_PATH_SCOPE_MISMATCH",
                    "revc_snapshot_error": str(error)[:240],
                }
            try:
                request_path.rename(claim_path)
            except FileNotFoundError:
                return {}
            except OSError as error:
                return {
                    "revc_snapshot_status": "WARN_REQUEST_CLAIM_FAILED",
                    "revc_snapshot_error": str(error)[:240],
                }

            ack_written = False
            request_id = ""

            def write_ack(status: str, **extra: Any) -> None:
                nonlocal ack_written
                ack = {
                    "schema_version": 1,
                    "status": status,
                    "request_id": request_id,
                    "episode_id": str(episode_id),
                    "reset_generation": int(reset_generation),
                    "sequence_id": int(sequence_id),
                    "wall_time_unix": time.time(),
                }
                ack.update(extra)
                ack_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_ack = scoped_path(
                    ack_path.with_name(f".{ack_path.name}.tmp.{os.getpid()}"),
                    "temporary ack path",
                )
                temporary_ack.write_text(
                    json.dumps(ack, indent=2, sort_keys=True) + "\\n",
                    encoding="utf-8",
                )
                temporary_ack.replace(ack_path)
                ack_written = True

            try:
                snapshot_request = json.loads(claim_path.read_text(encoding="utf-8"))
                request_id = str(snapshot_request.get("request_id", ""))
                if (
                    snapshot_request.get("schema_version") != 1
                    or not request_id
                    or len(request_id) > 160
                    or not request_id.startswith(expected_prefix)
                ):
                    write_ack("WARN_INVALID_REQUEST_ID")
                    return {"revc_snapshot_status": "WARN_INVALID_REQUEST_ID"}
                if request_id == self._revc_last_request_id:
                    write_ack("DUPLICATE_REQUEST")
                    return {
                        "revc_snapshot_status": "DUPLICATE_REQUEST",
                        "revc_snapshot_request_id": request_id,
                    }
                required_identity = {"episode_id", "reset_generation", "sequence_id"}
                if not required_identity.issubset(snapshot_request):
                    write_ack("WARN_MISSING_EXECUTION_IDENTITY")
                    return {
                        "revc_snapshot_status": "WARN_MISSING_EXECUTION_IDENTITY",
                        "revc_snapshot_request_id": request_id,
                    }
                expected_identity = {
                    "episode_id": str(episode_id),
                    "reset_generation": int(reset_generation),
                    "sequence_id": int(sequence_id),
                }
                for field, active_value in expected_identity.items():
                    requested_value = snapshot_request.get(field)
                    if (
                        type(requested_value) is not type(active_value)
                        or requested_value != active_value
                    ):
                        write_ack(
                            "IDENTITY_MISMATCH",
                            mismatched_field=field,
                            requested_value=requested_value,
                            active_value=active_value,
                        )
                        return {
                            "revc_snapshot_status": "IDENTITY_MISMATCH",
                            "revc_snapshot_request_id": request_id,
                        }
                contract = __REVC_CONTRACT__
                preview_hz = float(contract["snapshot"]["external_preview_max_hz"])
                now_monotonic = time.monotonic()
                if now_monotonic - self._revc_last_preview_monotonic < 1.0 / preview_hz:
                    write_ack("RATE_LIMITED")
                    return {
                        "revc_snapshot_status": "RATE_LIMITED",
                        "revc_snapshot_request_id": request_id,
                    }

                expected_order = list(contract["camera_order"])
                sensors: list[tuple[int, dict[str, Any], Any]] = []
                pre_render_metadata: dict[str, tuple[int, int, float]] = {}
                for camera_index, camera in enumerate(contract["cameras"]):
                    if camera["identity"] != expected_order[camera_index]:
                        raise RuntimeError("Rev-C camera order drift at capture")
                    sensor = self.robot.sensors.get(camera["sensor_name"])
                    if sensor is None:
                        raise RuntimeError(
                            f"missing Rev-C sensor {camera['sensor_name']}"
                        )
                    before_frame = sensor.get_data()
                    if not isinstance(before_frame, dict):
                        raise RuntimeError("Rev-C sensor returned no frame mapping")
                    pre_render_metadata[camera["identity"]] = (
                        self._revc_reference_time_metadata(
                            sensor, camera["sensor_name"]
                        )
                    )
                    sensors.append((camera_index, camera, sensor))

                # The synchronous barrier completes a new render while the
                # timeline stays paused.  ReferenceTime therefore must remain
                # stable for each sensor and must agree across all four post-
                # barrier payloads.  The local barrier label is audit data,
                # never frame evidence.
                sim_stamp_before_ns = globals().get("_T5_SIM_CLOCK_NS")
                if sim_stamp_before_ns != sim_stamp_ns:
                    raise RuntimeError(
                        "Rev-C request sim stamp changed before render barrier"
                    )
                import omni.replicator.core as rep

                rep.orchestrator.step(
                    rt_subframes=0,
                    pause_timeline=True,
                    wait_for_render=True,
                )
                sim_stamp_after_ns = globals().get("_T5_SIM_CLOCK_NS")
                if sim_stamp_after_ns != sim_stamp_before_ns:
                    raise RuntimeError("Rev-C render barrier advanced simulation time")
                post_render_frames: list[
                    tuple[int, dict[str, Any], dict[str, Any], tuple[int, int, float]]
                ] = []
                for camera_index, camera, sensor in sensors:
                    frame = sensor.get_data()
                    if not isinstance(frame, dict):
                        raise RuntimeError("Rev-C sensor returned no frame mapping")
                    frame_identity = self._revc_reference_time_metadata(
                        sensor, camera["sensor_name"]
                    )
                    self._revc_require_pause_stable_render(
                        pre_render_metadata[camera["identity"]],
                        frame_identity,
                        camera["sensor_name"],
                    )
                    post_render_frames.append(
                        (camera_index, camera, frame, frame_identity)
                    )

                actual_identities = [item[3] for item in post_render_frames]
                if len(set(actual_identities)) != 1:
                    raise RuntimeError("Rev-C cameras are from different render frames")

                captured_frames: list[
                    tuple[int, dict[str, Any], np.ndarray, tuple[int, int, float]]
                ] = []
                for camera_index, camera, frame, frame_identity in post_render_frames:
                    rgba = np.asarray(frame.get("rgba"))
                    if rgba.shape != (480, 640, 4):
                        raise RuntimeError(
                            f"invalid Rev-C frame shape for {camera['identity']}: "
                            f"{rgba.shape!r}"
                    )
                    rgb = np.ascontiguousarray(rgba[:, :, :3].astype(np.uint8))
                    captured_frames.append(
                        (camera_index, camera, rgb, frame_identity)
                    )

                numerator, denominator, render_time = actual_identities[0]
                render_identity_source = (
                    "replicator_annotator:ReferenceTime+SimulationManager"
                )
                render_identity_value = {
                    "referenceTimeNumerator": numerator,
                    "referenceTimeDenominator": denominator,
                    "rendering_time": render_time,
                }
                render_barrier_id = (
                    f"{sim_stamp_before_ns}:{numerator}/{denominator}"
                )
                snapshot_key = hashlib.sha256(
                    (
                        f"{request_id}|{episode_id}|{reset_generation}|"
                        f"{sequence_id}|{render_barrier_id}"
                    ).encode("utf-8")
                ).hexdigest()[:20]
                snapshot_root = scoped_path(
                    result_root / "revc_snapshots", "snapshot root"
                )
                snapshot_dir = scoped_path(
                    snapshot_root / snapshot_key, "snapshot directory"
                )
                if snapshot_dir.parent != snapshot_root:
                    raise RuntimeError("Rev-C snapshot directory scope is invalid")
                same_render_tick = sim_stamp_after_ns == sim_stamp_before_ns
                if not same_render_tick:
                    raise RuntimeError("Rev-C same-render-tick proof failed")

                # Validate all frames before producing any success sidecar.
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                camera_metadata: list[dict[str, Any]] = []
                for camera_index, camera, rgb, frame_identity in captured_frames:
                    png = self._encode_revc_png(rgb)
                    image_path = scoped_path(
                        snapshot_dir
                        / f"{camera_index:02d}_{camera['identity']}.png",
                        "snapshot image path",
                    )
                    if image_path.parent != snapshot_dir:
                        raise RuntimeError("Rev-C snapshot image scope is invalid")
                    temporary_image = scoped_path(
                        image_path.with_suffix(".png.tmp"),
                        "temporary snapshot image path",
                    )
                    temporary_image.write_bytes(png)
                    temporary_image.replace(image_path)
                    camera_metadata.append(
                        {
                            "identity": camera["identity"],
                            "order_index": camera_index,
                            "sensor_name": camera["sensor_name"],
                            "frame_id": camera["frame_id"],
                            "prim_path": camera["prim_path"],
                            "position_F_M_mm": camera["position_F_M_mm"],
                            "yaw_deg": camera["yaw_deg"],
                            "pitch_down_deg": contract["optics"]["pitch_down_deg"],
                            "resolution": contract["optics"]["resolution"],
                            "hfov_deg": contract["optics"]["hfov_deg"],
                            "render_identity": {
                                "referenceTimeNumerator": frame_identity[0],
                                "referenceTimeDenominator": frame_identity[1],
                                "rendering_time": frame_identity[2],
                            },
                            "encoding": "png_rgb8",
                            "path": image_path.relative_to(result_root).as_posix(),
                            "sha256": hashlib.sha256(png).hexdigest(),
                            "bytes": len(png),
                        }
                    )
                sidecar = {
                    "schema_version": 1,
                    "contract_id": contract["contract_id"],
                    "contract_sha256": "__REVC_CONTRACT_SHA__",
                    "scope": contract["scope"],
                    "request_id": request_id,
                    "episode_id": str(episode_id),
                    "reset_generation": int(reset_generation),
                    "sequence_id": int(sequence_id),
                    "render_barrier_id": render_barrier_id,
                    "render_identity_source": render_identity_source,
                    "render_identity_value": render_identity_value,
                    "sim_stamp_before_ns": sim_stamp_before_ns,
                    "sim_stamp_after_ns": sim_stamp_after_ns,
                    "wall_time_unix": time.time(),
                    "same_render_tick": same_render_tick,
                    "camera_order": expected_order,
                    "external_preview_max_hz": preview_hz,
                    "cuvslam_stereo_is_separate": True,
                    "cameras": camera_metadata,
                }
                sidecar_path = scoped_path(
                    snapshot_dir / "snapshot.json", "snapshot sidecar path"
                )
                if sidecar_path.parent != snapshot_dir:
                    raise RuntimeError("Rev-C snapshot sidecar scope is invalid")
                temporary_sidecar = scoped_path(
                    sidecar_path.with_suffix(".json.tmp"),
                    "temporary snapshot sidecar path",
                )
                temporary_sidecar.write_text(
                    json.dumps(sidecar, indent=2, sort_keys=True) + "\\n",
                    encoding="utf-8",
                )
                temporary_sidecar.replace(sidecar_path)
                self._revc_last_request_id = request_id
                self._revc_last_preview_monotonic = now_monotonic
                relative_sidecar = sidecar_path.relative_to(result_root).as_posix()
                write_ack(
                    "CAPTURED",
                    render_barrier_id=render_barrier_id,
                    render_identity_source=render_identity_source,
                    render_identity_value=render_identity_value,
                    sidecar=relative_sidecar,
                )
                return {
                    "revc_snapshot_status": "CAPTURED",
                    "revc_snapshot_request_id": request_id,
                    "revc_snapshot_render_barrier_id": render_barrier_id,
                    "revc_snapshot_sidecar": relative_sidecar,
                }
            except Exception as error:
                # Snapshot/preview is an observational path.  A malformed
                # request or capture failure must not stop bounded navigation.
                try:
                    write_ack("WARN_CAPTURE_FAILED", error=str(error)[:240])
                except Exception as ack_error:
                    return {
                        "revc_snapshot_status": "WARN_ACK_FAILED",
                        "revc_snapshot_error": (
                            f"{error}; ack={ack_error}"
                        )[:240],
                    }
                return {
                    "revc_snapshot_status": "WARN_CAPTURE_FAILED",
                    "revc_snapshot_error": str(error)[:240],
                }
            finally:
                if ack_written:
                    claim_path.unlink(missing_ok=True)

'''
    if t5_sensor_extensions_enabled:
        assert revc_contract is not None
        assert revc_config_path is not None
        assert t5_revc_scope is not None
        feature_methods = feature_methods.replace(
            "__REVC_CONTRACT__", repr(revc_contract)
        )
        feature_methods = feature_methods.replace(
            "__REVC_CONTRACT_SHA__", _sha256(revc_config_path)
        )
        feature_methods = feature_methods.replace(
            "__T5_REVC_LANE__", repr(t5_revc_scope.lane)
        )
        feature_methods = feature_methods.replace(
            "__T5_REVC_IDENTITY_PREFIX__", repr(t5_revc_scope.identity_prefix)
        )
        feature_methods = feature_methods.replace(
            "__T5_REVC_RESULT_ROOT__", repr(str(t5_revc_scope.result_root))
        )
        methods += feature_methods
    text = _replace_once(
        text,
        "        def _sample_stereo(self) -> dict[str, Any]:\n",
        methods + "        def _sample_stereo(self) -> dict[str, Any]:\n",
    )
    fault_prefix = ""
    fault_sensor_guard = ""
    if fault_injection_enabled:
        fault_prefix = (
            "            t5_fault_control = getattr(self, \"_t5_fault_control\", None)\n"
            "            if t5_fault_control is None:\n"
            "                t5_fault_control = FaultControlReader(\"isaac_controller\")\n"
            "                self._t5_fault_control = t5_fault_control\n"
            "            t5_fault_snapshot = t5_fault_control.read()\n"
            "            t5_sensor_outage = t5_fault_snapshot.event_for(\n"
            "                \"lidar_depth_short_outage\"\n"
            "            ) is not None\n"
            "            t5_network_outage = t5_fault_snapshot.event_for(\n"
            "                \"network_short_outage_and_recovery\"\n"
            "            ) is not None\n"
        )
        fault_sensor_guard = " and not t5_sensor_outage"
    request_replacement = fault_prefix
    request_replacement += (
        "            if not state_only:\n"
        "                if os.environ.get(\"INTERNVLA_T4_R3_ENABLE_D435I\", \"1\") == \"1\""
        + fault_sensor_guard
        + ":\n"
    )
    request_replacement += (
        "                    request.update(self._sample_depth())\n"
        "                request.update(self._sample_stereo())\n"
    )
    request_replacement += (
        "            if not t5_sensor_outage:\n"
        "                request.update(self._sample_go2_lidar(state_only=state_only))\n"
        if fault_injection_enabled
        else "            request.update(self._sample_go2_lidar(state_only=state_only))\n"
    )
    request_replacement += (
        "            request.update(self._sample_r3_images_and_imu(state_only=state_only))\n"
    )
    if t5_sensor_extensions_enabled:
        request_replacement += (
            "            t5_revc_sim_stamp_ns = globals().get(\"_T5_SIM_CLOCK_NS\")\n"
            "            request[\"t5_revc_sim_sample_stamp_ns\"] = t5_revc_sim_stamp_ns\n"
            "            request.update(\n"
            "                self._sample_simulated_imu_linear_acceleration(\n"
            "                    linear, pose, t5_revc_sim_stamp_ns,\n"
            "                    state_only=state_only\n"
            "                )\n"
            "            )\n"
            "            request.update(\n"
            "                self._sample_revc_snapshot(\n"
            "                    episode_id=episode_id,\n"
            "                    reset_generation=reset_generation,\n"
            "                    sequence_id=sequence_id,\n"
            "                    sim_stamp_ns=t5_revc_sim_stamp_ns,\n"
            "                    state_only=state_only,\n"
            "                )\n"
            "            )\n"
        )
    text = _replace_once(
        text,
        "            if not state_only:\n"
        "                request.update(self._sample_depth())\n"
        "                request.update(self._sample_stereo())\n",
        request_replacement,
    )
    if fault_injection_enabled:
        text = _replace_once(
            text,
            "                response = self.ipc.exchange(request)\n",
            "                if t5_network_outage:\n"
            "                    raise ConnectionError(\n"
            "                        \"injected T5 lane network data-plane outage\"\n"
            "                    )\n"
            "                response = self.ipc.exchange(request)\n",
        )
        text = _replace_once(
            text,
            "            except Exception:\n"
            "                self.desired = (0.0, 0.0)\n"
            "                self.emergency_stop = True\n"
            "                if state_only:\n"
            "                    raise\n",
            "            except Exception as error:\n"
            "                self.desired = (0.0, 0.0)\n"
            "                self.emergency_stop = True\n"
            "                if state_only and not (\n"
            "                    t5_network_outage\n"
            "                    and isinstance(error, ConnectionError)\n"
            "                ):\n"
            "                    raise\n",
        )
    sensor_block = '''        robot.sensors.append(
            SensorCfg(
                sensor_settings=VLNCameraCfg(
                    name="go2_front_rgb",
                    prim_path="base/go2_front_rgb",
                    enable=True,
                    resolution=[320, 240],
                ).model_dump(),
            )
        )
'''
    if not rgb_ipc_enabled:
        sensor_block = ""
    revc_sensor_block = ""
    if t5_sensor_extensions_enabled:
        assert revc_contract is not None
        for camera in revc_contract["cameras"]:
            revc_sensor_block += (
                "        robot.sensors.append(\n"
                "            SensorCfg(\n"
                "                sensor_settings=VLNCameraCfg(\n"
                f"                    name={camera['sensor_name']!r},\n"
                f"                    prim_path={camera['prim_path']!r},\n"
                "                    enable=True,\n"
                "                    resolution=[640, 480],\n"
                "                ).model_dump(),\n"
                "            )\n"
                "        )\n"
            )
    text = _replace_once(
        text,
        "        if os.environ.get(\"INTERNVLA_T4_ENABLE_STEREO_ODOMETRY\", \"0\") == \"1\":\n",
        sensor_block
        + revc_sensor_block
        + "        if os.environ.get(\"INTERNVLA_T4_ENABLE_STEREO_ODOMETRY\", \"0\") == \"1\":\n",
    )
    # Freeze the A/B/C/D geometry-source choice at overlay generation time.
    # The Isaac application may sanitize its environment during Kit startup;
    # a generated constant is therefore more reproducible than late getenv.
    text = _replace_once(
        text,
        'os.environ.get("INTERNVLA_T4_R3_ENABLE_LIDAR", "1") != "1"',
        f'{"1" if lidar_enabled else "0"!r} != "1"',
    )
    text = _replace_once(text, "            width = 180", f"            width = {lidar_width}")
    text = _replace_once(
        text,
        'os.environ.get("INTERNVLA_T4_R3_ENABLE_D435I", "1") == "1"',
        f'{"1" if d435i_enabled else "0"!r} == "1"',
    )
    text = _replace_once(
        text,
        'os.environ.get("INTERNVLA_T4_R3_ENABLE_RGB_IPC", "1") != "1"',
        f'{"1" if rgb_ipc_enabled else "0"!r} != "1"',
    )
    # T5 explicitly exports this switch before generating the overlay.  Freeze
    # that opt-in because Isaac Kit may sanitize the environment before the
    # generated runtime constructs or samples the stereo sensors.  An absent
    # switch preserves the byte-for-byte frozen T4 runtime.
    if stereo_odometry_override is not None:
        text = _replace_once(
            text,
            'os.environ.get("INTERNVLA_T4_ENABLE_STEREO_ODOMETRY", "0") != "1"',
            f'{"1" if stereo_odometry_enabled else "0"!r} != "1"',
        )
        text = _replace_once(
            text,
            'os.environ.get("INTERNVLA_T4_ENABLE_STEREO_ODOMETRY", "0") == "1"',
            f'{"1" if stereo_odometry_enabled else "0"!r} == "1"',
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    payload = {
        "schema_version": 2,
        "status": "PASS",
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(output),
        "inherited_overlay_sha256": inherited["output_sha256"],
        "changes": inherited["changes"]
        + [
            "physx_scene_query_go2_4d_lidar",
            "standard_ros_payload_for_go2_front_rgb",
            "standard_ros_payload_for_d435i_rgb",
            "articulation_imu_payload",
            "independent_d435i_lidar_enable_switches",
            "independent_bounded_rgb_ipc_switch",
        ],
        "map_or_obstacle_truth_used_for_lidar": False,
        "frozen_geometry_sources": {
            "d435i": d435i_enabled,
            "lidar": lidar_enabled,
            "lidar_ray_count": lidar_ray_count,
            "lidar_width": lidar_width,
            "rgb_ipc": rgb_ipc_enabled,
        },
        "motion_control_semantics_changed": False,
    }
    if fault_injection_enabled:
        payload["fault_injection_profile"] = fault_profile
        payload["changes"].append(
            "completion_sim_lidar_depth_and_network_outage_hooks"
        )
    if stereo_odometry_override is not None:
        payload["changes"].append("frozen_stereo_odometry_enable_switch")
        payload["frozen_geometry_sources"]["stereo_odometry"] = (
            stereo_odometry_enabled
        )
    if t5_sensor_extensions_enabled:
        assert revc_contract is not None
        assert revc_config_path is not None
        assert imu_contract is not None
        assert imu_contract_path is not None
        assert t5_revc_scope is not None
        payload["changes"].extend(
            [
                "on_demand_verified_render_barrier_revc_four_camera_snapshot",
                "two_sim_stamp_imu_linear_acceleration_precondition",
            ]
        )
        payload["revc_four_camera"] = {
            "enabled": True,
            "contract_id": revc_contract["contract_id"],
            "contract_sha256": _sha256(revc_config_path),
            "camera_order": revc_contract["camera_order"],
            "snapshot_capture": revc_contract["snapshot"]["capture"],
            "external_preview_max_hz": revc_contract["snapshot"][
                "external_preview_max_hz"
            ],
            "resolution": revc_contract["optics"]["resolution"],
            "cuvslam_stereo_is_separate": True,
            "render_barrier": "replicator_step_pause_timeline_wait_for_render",
            "frozen_scope": {
                "lane": t5_revc_scope.lane,
                "identity_prefix": t5_revc_scope.identity_prefix,
                "result_root": str(t5_revc_scope.result_root),
            },
        }
        payload["simulated_imu_linear_acceleration"] = {
            "contract_id": imu_contract["contract_id"],
            "contract_sha256": _sha256(imu_contract_path),
            "purpose": imu_contract["purpose"],
            "lio_backend_implemented": False,
            "derivative_clock": imu_contract["derivative_clock"],
            "published_sample_stamp_key": imu_contract[
                "published_sample_stamp_key"
            ],
        }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
