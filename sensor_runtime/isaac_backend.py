"""Standalone Isaac Sim backend for the model-free 01R workload.

This module intentionally does not import InternNav's evaluator.  A World,
Go2 articulation, two render cameras, PhysX rays, and the outer physics loop
are owned directly by this backend.
"""

from __future__ import annotations

import math
import time
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

from .contract import DIAGNOSTIC_GEOMETRY, DIAGNOSTIC_LIGHT, PHYSICS_HZ
from .image_contract import downsample_rgb_2x2, rgb_content_evidence
from .lidar_contract import frozen_lidar_local_directions, raycast_frozen_lidar
from .pointcloud import valid_depth_evidence
from .render_pipeline import (
    RenderClockCalibration,
    RenderSnapshot,
    SingleRenderPipeline,
    bounded_zero_frame_resync,
    calibrate_render_clock,
)
from .workload import CaptureNotReady, SafeStep


JOINT_NAMES = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)
STAND_POSITIONS = (0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 0.8, 0.8, -1.5, -1.5, -1.5, -1.5)


class IsaacModelFreeBackend:
    """Real physics/render stepping with a hard-coded zero-motion controller."""

    def __init__(
        self,
        wrapper_usd: Path,
        *,
        render_stamp_deviation_tolerance_ns: int = 0,
        render_resync_limit: int = 1,
    ) -> None:
        if type(render_resync_limit) is not int or not 1 <= render_resync_limit <= 4:
            raise ValueError("render resync limit must be an integer within [1, 4]")
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import UsdGeom, UsdLux

        try:
            from isaacsim.sensors.camera import Camera
        except ImportError:  # Isaac 6 package spelling
            from isaacsim.sensors.camera.camera import Camera

        if not wrapper_usd.is_file():
            raise FileNotFoundError(wrapper_usd)
        self.np = np
        self._ArticulationAction = ArticulationAction
        self._simulation_manager_interface = (
            SimulationManager._simulation_manager_interface
        )
        self.world = World(
            physics_dt=1.0 / PHYSICS_HZ,
            rendering_dt=1.0 / 20.0,
            stage_units_in_meters=1.0,
        )
        stage = get_current_stage()
        self.diagnostic_light = UsdLux.DomeLight.Define(
            stage, DIAGNOSTIC_LIGHT["prim_path"]
        )
        self.diagnostic_light.CreateIntensityAttr(DIAGNOSTIC_LIGHT["intensity"])
        self.diagnostic_light.CreateEnableColorTemperatureAttr(True)
        self.diagnostic_light.CreateColorTemperatureAttr(
            DIAGNOSTIC_LIGHT["color_temperature_kelvin"]
        )
        self.world.scene.add_default_ground_plane()
        self.diagnostic_geometry = []
        for item in DIAGNOSTIC_GEOMETRY:
            self.diagnostic_geometry.append(
                self.world.scene.add(
                    FixedCuboid(
                        prim_path=item["prim_path"],
                        name=item["name"],
                        position=np.asarray(item["position_world_m"], dtype=np.float64),
                        size=1.0,
                        scale=np.asarray(item["size_m"], dtype=np.float64),
                        color=np.asarray([0.85, 0.25, 0.10], dtype=np.float32),
                    )
                )
            )
        add_reference_to_stage(str(wrapper_usd), "/World/Go2")
        self.articulation = self.world.scene.add(
            SingleArticulation(
                prim_path="/World/Go2",
                name="sensor_soak_go2",
                position=np.asarray([0.0, 0.0, 0.42], dtype=np.float64),
            )
        )
        self.color_camera = Camera(
            prim_path="/World/Go2/base/internvla_camera",
            name="sensor_soak_d435i_color",
            resolution=(640, 480),
            frequency=-1,
        )
        self.depth_camera = Camera(
            prim_path="/World/Go2/base/t4_d435i_depth",
            name="sensor_soak_d435i",
            resolution=(640, 480),
            frequency=-1,
        )
        self.front_camera = Camera(
            prim_path="/World/Go2/base/go2_front_rgb",
            name="sensor_soak_front_rgb",
            resolution=(320, 240),
            frequency=-1,
        )
        self.world.reset()
        self.color_camera.initialize()
        self.depth_camera.initialize()
        self.front_camera.initialize()
        self.depth_camera.add_distance_to_image_plane_to_frame()
        front_camera_schema = UsdGeom.Camera(
            stage.GetPrimAtPath("/World/Go2/base/go2_front_rgb")
        )
        front_clipping = front_camera_schema.GetClippingRangeAttr().Get()
        self._front_clipping_range = (
            float(front_clipping[0]),
            float(front_clipping[1]),
        )
        if not (
            math.isclose(self._front_clipping_range[0], 0.20, abs_tol=1e-6)
            and math.isclose(
                self._front_clipping_range[1], 1_000_000.0, abs_tol=1e-3
            )
        ):
            raise RuntimeError("Go2 front RGB clipping range differs from frozen asset")
        link_paths = {
            name: f"/World/Go2/{name}"
            for leg in ("FL", "FR", "RL", "RR")
            for name in (f"{leg}_thigh", f"{leg}_calf", f"{leg}_foot")
        }
        missing = [path for path in link_paths.values() if not stage.GetPrimAtPath(path).IsValid()]
        if missing:
            raise RuntimeError(f"Go2 articulation lacks frozen rigid link prim(s): {missing}")
        self._filter_links = {
            name: SingleRigidPrim(prim_path=path, name=f"filter_{name}")
            for name, path in link_paths.items()
        }
        for link in self._filter_links.values():
            link.initialize()
        self._dof_indices = np.asarray(
            [self.articulation.get_dof_index(name) for name in JOINT_NAMES], dtype=np.int64
        )
        if (self._dof_indices < 0).any():
            raise RuntimeError("Go2 articulation lacks a frozen stand joint")
        self._stand = np.asarray(STAND_POSITIONS, dtype=np.float64)
        self._zeros = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._lidar_local_directions = frozen_lidar_local_directions()
        self._lidar_distances = np.empty(8 * 180, dtype=np.float32)
        self._generation = -1
        self._generation_step = 0
        self._global_safe_steps = 0
        self._render_reference_denominator: int | None = None
        self._render_pipeline = SingleRenderPipeline(
            int(PHYSICS_HZ / 20.0),
            max_stamp_deviation_ns=render_stamp_deviation_tolerance_ns,
        )
        self._pending_capture_snapshot: RenderSnapshot | None = None
        self._pending_capture_metadata: dict[str, tuple[int, int, float]] | None = None
        self._render_only_drain_total = 0
        self._render_only_drain_max = 0
        self._last_render_drain_count = 0
        self._startup_render_only_drain_count = 0
        self._render_resync_limit = render_resync_limit
        self._render_resync_event_count = 0
        self._render_resync_drop_count = 0
        self._render_resync_extra_drain_total = 0
        self._physics_latch_total = 0
        self._physical_step_total = 0
        self._stamp_world_origin = float(self.world.current_time)
        self._render_clock: RenderClockCalibration | None = None
        self._render_clock_seed_physical_step: int | None = None
        self._apply_safe_stop(count_physics_step=False)
        prewarm_physical_times = tuple(
            self._advance_one_physics_step()
            for _ in range(self._render_pipeline.lag_steps + 1)
        )
        prewarm_metadata = self._render_until_current()
        prewarm_id, _prewarm_denominator, prewarm_time = prewarm_metadata["color"]
        self._render_clock = calibrate_render_clock(
            origin_time=self._stamp_world_origin,
            physical_times=prewarm_physical_times,
            render_time=prewarm_time,
            lag_steps=self._render_pipeline.lag_steps,
        )
        self._render_clock_seed_physical_step = self._physical_step_total
        for warm_stamp_ns in self._render_clock.warm_stamps_ns:
            self._render_pipeline.warm_stamp(warm_stamp_ns)
        prewarm_snapshot = self._capture_render_snapshot(
            generation=-1,
            stamp_ns=self._render_clock.seed_stamp_ns,
            world_time=prewarm_time,
        )
        self._render_pipeline.seed(prewarm_snapshot, prewarm_id, prewarm_time)
        self._render_latency_negotiation = self._negotiate_render_latency()
        self._pace_origin = time.monotonic()

    @staticmethod
    def _rgb8(value: Any, expected: tuple[int, int]) -> Any:
        import numpy as np

        array = np.asarray(value)
        if array.ndim != 3 or tuple(array.shape[:2]) != expected or array.shape[2] < 3:
            raise CaptureNotReady(f"unexpected rendered RGB shape: {array.shape}")
        rgb = array[:, :, :3]
        if not np.isfinite(rgb).all():
            raise CaptureNotReady("rendered RGB contains NaN or infinity before conversion")
        if np.issubdtype(rgb.dtype, np.floating) and float(np.nanmax(rgb)) <= 1.0:
            rgb = rgb * 255.0
        return np.ascontiguousarray(np.nan_to_num(rgb).clip(0, 255), dtype=np.uint8)

    def _diagnostic_light_evidence(self) -> dict[str, Any]:
        return {
            "name": DIAGNOSTIC_LIGHT["name"],
            "prim_path": DIAGNOSTIC_LIGHT["prim_path"],
            "type": str(self.diagnostic_light.GetPrim().GetTypeName()),
            "intensity": float(self.diagnostic_light.GetIntensityAttr().Get()),
            "temperature_enabled": bool(
                self.diagnostic_light.GetEnableColorTemperatureAttr().Get()
            ),
            "color_temperature_kelvin": float(
                self.diagnostic_light.GetColorTemperatureAttr().Get()
            ),
        }

    @staticmethod
    def _optical_quaternion(pitch_down_deg: float) -> list[float]:
        alpha = math.radians((90.0 - pitch_down_deg) / 2.0)
        scale = math.sqrt(0.5)
        return [
            scale * math.sin(alpha),
            -scale * math.cos(alpha),
            scale * math.cos(alpha),
            -scale * math.sin(alpha),
        ]

    @staticmethod
    def _rotate(rotation: Any, vector: Any) -> Any:
        import numpy as np

        w, x, y, z = [float(value) for value in rotation]
        q = np.asarray([x, y, z], dtype=np.float64)
        twice = 2.0 * np.cross(q, vector)
        return vector + w * twice + np.cross(q, twice)

    @staticmethod
    def _rotation_matrix(rotation: Any) -> Any:
        import numpy as np

        w, x, y, z = [float(value) for value in rotation]
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _link_centers_base(self, base_position: Any, base_rotation: Any) -> list[dict[str, Any]]:
        rotation = self._rotation_matrix(base_rotation)
        centers: list[dict[str, Any]] = [{"name": "base", "center_base": [0.0, 0.0, 0.0]}]
        for name, link in self._filter_links.items():
            position, _ = link.get_world_pose()
            center = rotation.T @ (self.np.asarray(position, dtype=self.np.float64) - base_position)
            if not self.np.isfinite(center).all():
                raise RuntimeError(f"non-finite dynamic self-filter link center: {name}")
            centers.append({"name": name, "center_base": [float(value) for value in center]})
        if len(centers) != 13:
            raise RuntimeError("self-filter requires base plus all thigh/calf/foot centers")
        return centers

    def _support_plane_world_z(self) -> float:
        foot_heights = []
        for name, link in self._filter_links.items():
            if name.endswith("_foot"):
                position, _ = link.get_world_pose()
                foot_heights.append(float(position[2]))
        if len(foot_heights) != 4 or not all(math.isfinite(value) for value in foot_heights):
            raise RuntimeError("support plane requires four finite foot centers")
        # This workload creates a known default ground plane at world z=0.
        # Foot centers are only a finite/contact sanity probe; adding their
        # collision radius here would double-count the frozen +0.08 m filter
        # margin and erase real low obstacles.
        if not -0.20 <= min(foot_heights) <= 0.30:
            raise RuntimeError("Go2 foot centers are inconsistent with the default support plane")
        return 0.0

    def _apply_safe_stop(self, *, count_physics_step: bool) -> None:
        self.articulation.set_linear_velocity(
            self.np.zeros(3, dtype=self.np.float64)
        )
        self.articulation.set_angular_velocity(
            self.np.zeros(3, dtype=self.np.float64)
        )
        self.articulation.apply_action(
            self._ArticulationAction(
                joint_positions=self._stand,
                joint_velocities=self._zeros,
                joint_indices=self._dof_indices,
            )
        )
        if count_physics_step:
            self._global_safe_steps += 1

    def _render_metadata(
        self, frame: Mapping[str, Any], camera_name: str
    ) -> tuple[int, int, float]:
        if "rendering_frame" not in frame or "rendering_time" not in frame:
            raise RuntimeError(f"{camera_name} frame lacks real render metadata")
        raw_identity = frame["rendering_frame"]
        if not isinstance(raw_identity, Mapping) or set(raw_identity) != {
            "referenceTimeNumerator",
            "referenceTimeDenominator",
        }:
            raise RuntimeError(f"{camera_name} ReferenceTime schema is invalid")
        numerator = raw_identity["referenceTimeNumerator"]
        denominator = raw_identity["referenceTimeDenominator"]
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
            raise RuntimeError(f"{camera_name} ReferenceTime value is invalid")
        numerator = int(numerator)
        denominator = int(denominator)
        if self._render_reference_denominator is None:
            self._render_reference_denominator = denominator
        elif denominator != self._render_reference_denominator:
            raise RuntimeError("camera ReferenceTime denominator changed")
        render_time = float(frame["rendering_time"])
        if (
            not math.isfinite(render_time)
            or render_time < 0.0
            or not math.isclose(
                render_time,
                numerator / denominator,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise RuntimeError(f"{camera_name} frame has invalid render metadata")
        return numerator, denominator, render_time

    def _camera_frames(self) -> dict[str, Mapping[str, Any]]:
        return {
            "color": self.color_camera.get_current_frame(),
            "depth": self.depth_camera.get_current_frame(),
            "front": self.front_camera.get_current_frame(),
        }

    def _read_current_render_metadata(
        self, frames: Mapping[str, Mapping[str, Any]] | None = None
    ) -> dict[str, tuple[int, int, float]]:
        selected = self._camera_frames() if frames is None else frames
        metadata = {
            name: self._render_metadata(frame, name)
            for name, frame in selected.items()
        }
        if set(metadata) != {"color", "depth", "front"}:
            raise RuntimeError("camera render set differs from the frozen three-camera set")
        if len(set(metadata.values())) != 1:
            raise RuntimeError(
                "color/depth/front cameras do not share one actual render identity/time"
            )
        return metadata

    def _refresh_camera_from_completed_render(self, camera: Any) -> None:
        camera._og_controller.evaluate_sync(graph_id=camera._sdg_graph_pipeline)
        frame_number = camera._fabric_time_annotator.get_data()
        if not isinstance(frame_number, Mapping) or set(frame_number) != {
            "referenceTimeNumerator",
            "referenceTimeDenominator",
        }:
            raise RuntimeError("completed render lacks Isaac ReferenceTime")
        numerator = frame_number["referenceTimeNumerator"]
        denominator = frame_number["referenceTimeDenominator"]
        current_time = self._simulation_manager_interface.get_simulation_time_at_time(
            (numerator, denominator)
        )
        if not isinstance(camera._current_frame, dict):
            raise RuntimeError("Isaac camera current frame is not mutable")
        camera._current_frame["rendering_frame"] = dict(frame_number)
        camera._current_frame["rendering_time"] = current_time
        for key in tuple(camera._current_frame):
            if key in {"rendering_time", "rendering_frame"}:
                continue
            annotator = camera._custom_annotators.get(key)
            if annotator is None:
                raise RuntimeError(f"Isaac camera annotator is missing for {key}")
            camera._current_frame[key] = annotator.get_data()
        camera._previous_time = current_time
        camera._elapsed_time = 0

    def _advance_one_physics_step(self) -> float:
        before = float(self.world.current_time)
        attempts: list[float] = []
        for latch_count in range(4):
            self.world.step(render=False)
            after = float(self.world.current_time)
            attempts.append(after)
            delta = after - before
            if math.isclose(
                delta, 1.0 / PHYSICS_HZ, rel_tol=0.0, abs_tol=1e-6
            ):
                self._physics_latch_total += latch_count
                self._physical_step_total += 1
                return after
            if abs(delta) > 1e-9:
                raise RuntimeError(
                    f"one safe-stop step advanced invalid physics time: {attempts}"
                )
        raise RuntimeError(f"timeline did not advance one physics step: {attempts}")

    def _negotiate_render_latency(self) -> dict[str, Any]:
        """Freeze zero/one-frame camera latency before the workload starts."""

        for _ in range(self._render_pipeline.lag_steps):
            self._advance_one_physics_step()
            self._render_pipeline.begin_step(self._physical_stamp_ns())
        requested = self._capture_render_snapshot(
            generation=-1,
            stamp_ns=self._physical_stamp_ns(),
            world_time=self._reference_world_time(),
        )
        metadata = self._render_once()
        render_id, _denominator, render_time = metadata["color"]
        returned = self._render_pipeline.bind(
            requested,
            render_id,
            render_time,
            self._stamp_for_world_time(render_time),
            -1,
        )
        mode = self._render_pipeline.render_latency_mode
        if returned is None or mode not in {
            SingleRenderPipeline.ZERO_FRAME,
            SingleRenderPipeline.ONE_FRAME,
        }:
            raise RuntimeError("render latency negotiation did not freeze one exact mode")
        return {
            "mode": mode,
            "seed_stamp_ns": self._render_clock.seed_stamp_ns,
            "requested_stamp_ns": requested.stamp_ns,
            "returned_stamp_ns": returned.stamp_ns,
            "render_stamp_ns": self._stamp_for_world_time(render_time),
            "render_id": render_id,
        }

    def _physical_stamp_ns(self) -> int:
        if (
            self._render_clock is None
            or self._render_clock_seed_physical_step is None
        ):
            raise RuntimeError("camera reference clock was not calibrated")
        return self._render_clock.stamp_for_physical_step(
            self._physical_step_total,
            self._render_clock_seed_physical_step,
        )

    def _reference_world_time(self) -> float:
        if (
            self._render_clock is None
            or self._render_clock_seed_physical_step is None
        ):
            raise RuntimeError("camera reference clock was not calibrated")
        return self._render_clock.reference_time_for_physical_step(
            self._physical_step_total,
            self._render_clock_seed_physical_step,
        )

    def _stamp_for_world_time(self, world_time: float) -> int:
        elapsed = world_time - self._stamp_world_origin
        stamp_ns = int(round(elapsed * 1_000_000_000))
        if stamp_ns <= 0:
            raise RuntimeError("physical simulation timestamp is not positive")
        return stamp_ns

    def _render_until_current(self) -> dict[str, tuple[int, int, float]]:
        target_time = float(self.world.current_time)
        attempts: list[Any] = []
        for drain_count in range(1, 5):
            before_render = float(self.world.current_time)
            self.world.render()
            for camera in (
                self.color_camera,
                self.depth_camera,
                self.front_camera,
            ):
                self._refresh_camera_from_completed_render(camera)
            after_render = float(self.world.current_time)
            if not math.isclose(
                after_render, before_render, rel_tol=0.0, abs_tol=1e-9
            ):
                raise RuntimeError("render-only synchronization advanced physics time")
            try:
                metadata = self._read_current_render_metadata()
            except RuntimeError as exc:
                attempts.append(str(exc))
                continue
            _render_id, _denominator, render_time = metadata["color"]
            attempts.append(render_time)
            if abs(render_time - target_time) <= 1.0 / PHYSICS_HZ + 1e-9:
                self._startup_render_only_drain_count = drain_count
                return metadata
        raise RuntimeError(
            "render-only drain did not synchronize camera time: "
            f"target={target_time}, attempts={attempts}"
        )

    def _render_once(self) -> dict[str, tuple[int, int, float]]:
        before_render = float(self.world.current_time)
        self.world.render()
        for camera in (
            self.color_camera,
            self.depth_camera,
            self.front_camera,
        ):
            self._refresh_camera_from_completed_render(camera)
        after_render = float(self.world.current_time)
        if not math.isclose(
            after_render, before_render, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RuntimeError("single render-only synchronization advanced physics time")
        metadata = self._read_current_render_metadata()
        self._render_only_drain_total += 1
        self._render_only_drain_max = max(self._render_only_drain_max, 1)
        self._last_render_drain_count = 1
        return metadata

    def _render_for_runtime_step(
        self,
    ) -> dict[str, tuple[int, int, float]] | None:
        """Bound completion-only catch-up without advancing physics.

        A negotiated zero-frame camera can transiently expose the previous
        20 Hz render under startup load.  Strict evidence still returns that
        first observation to the exact binder (and fails).  Completion may
        perform at most three additional render-only drains; if none reaches
        the current reference state, this camera capture is dropped while the
        safe-stop physics loop continues.
        """

        metadata = self._render_once()
        if (
            self._render_resync_limit == 1
            or self._render_pipeline.render_latency_mode
            != SingleRenderPipeline.ZERO_FRAME
        ):
            return metadata
        selected, drain_count = bounded_zero_frame_resync(
            metadata,
            render_once=self._render_once,
            target_time=self._reference_world_time(),
            limit=self._render_resync_limit,
        )
        if drain_count > 1 or selected is None:
            self._render_resync_event_count += 1
            self._render_resync_extra_drain_total += drain_count - 1
        if selected is None:
            self._render_resync_drop_count += 1
        self._last_render_drain_count = drain_count
        self._render_only_drain_max = max(self._render_only_drain_max, drain_count)
        return selected

    def reset(self, generation: int) -> None:
        if generation != self._generation + 1:
            raise RuntimeError("Isaac reset generation must increase by one")
        self._generation = generation
        # The World and render products are intentionally not reset here.
        # Resetting only articulation state keeps the generation boundary well
        # inside the 0.35 s downstream gap budget and preserves global sim time.
        self.articulation.set_world_pose(
            position=self.np.asarray([0.0, 0.0, 0.42], dtype=self.np.float64),
            orientation=self.np.asarray([1.0, 0.0, 0.0, 0.0], dtype=self.np.float64),
        )
        self.articulation.set_joint_positions(self._stand, self._dof_indices)
        self.articulation.set_joint_velocities(self._zeros, self._dof_indices)
        for expected, geometry in zip(DIAGNOSTIC_GEOMETRY, self.diagnostic_geometry):
            observed, _ = geometry.get_world_pose()
            if not self.np.allclose(
                self.np.asarray(observed, dtype=self.np.float64),
                self.np.asarray(expected["position_world_m"], dtype=self.np.float64),
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError("diagnostic collision geometry moved during articulation reset")
        light = self._diagnostic_light_evidence()
        if light != {
            **DIAGNOSTIC_LIGHT,
            "temperature_enabled": True,
        }:
            raise RuntimeError("diagnostic DomeLight contract changed during articulation reset")
        self._generation_step = 0
        self._pace_origin = time.monotonic()
        self._apply_safe_stop(count_physics_step=False)

    def step_safe_stop(self) -> SafeStep:
        self._apply_safe_stop(count_physics_step=True)
        self._generation_step += 1
        render_requested = self._generation_step % int(PHYSICS_HZ / 20.0) == 0
        self._advance_one_physics_step()
        physical_stamp_ns = self._physical_stamp_ns()
        stamp_ns = self._render_pipeline.begin_step(physical_stamp_ns)
        rendered = False
        if render_requested:
            requested = self._capture_render_snapshot(
                generation=self._generation,
                stamp_ns=physical_stamp_ns,
                world_time=self._reference_world_time(),
            )
            metadata = self._render_for_runtime_step()
            if metadata is None:
                self._pending_capture_snapshot = None
                self._pending_capture_metadata = None
            else:
                render_id, _denominator, render_time = metadata["color"]
                returned = self._render_pipeline.bind(
                    requested,
                    render_id,
                    render_time,
                    self._stamp_for_world_time(render_time),
                    self._generation,
                )
                if returned is not None:
                    self._pending_capture_snapshot = returned
                    self._pending_capture_metadata = metadata
                    rendered = True
                else:
                    self._pending_capture_snapshot = None
                    self._pending_capture_metadata = None
        target = self._pace_origin + self._generation_step / PHYSICS_HZ
        remaining = target - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        return SafeStep(
            sim_stamp_ns=stamp_ns,
            physics_step=self._generation_step,
            applied_step_count=self._global_safe_steps,
            rendered=rendered,
            render_id=self._render_pipeline.last_render_id,
            render_generation=self._render_pipeline.last_render_generation,
        )

    def _lidar(self, position: Any, rotation: Any) -> Any:
        import omni.physx

        np = self.np
        query = omni.physx.get_physx_scene_query_interface()
        origin_local = np.asarray([0.25, 0.0, 0.18], dtype=np.float64)
        rotation_matrix = self._rotation_matrix(rotation)
        origin = position + rotation_matrix @ origin_local
        return raycast_frozen_lidar(
            self._lidar_local_directions,
            rotation_matrix,
            origin,
            query.raycast_closest,
            self._lidar_distances,
        )

    def _capture_render_snapshot(
        self,
        *,
        generation: int,
        stamp_ns: int,
        world_time: float,
    ) -> RenderSnapshot:
        snapshot_started_ns = time.perf_counter_ns()
        current_world_time = float(self.world.current_time)
        reference_world_time = self._reference_world_time()
        if (
            not math.isfinite(world_time)
            or abs(world_time - reference_world_time) > 1e-9
            or stamp_ns != self._physical_stamp_ns()
        ):
            raise RuntimeError("render snapshot time differs from the current physical state")
        if (
            self._render_clock is None
            or abs(current_world_time - world_time) > 1.0 / PHYSICS_HZ + 1e-9
        ):
            raise RuntimeError("render snapshot camera reference offset changed")
        np = self.np
        position, rotation = self.articulation.get_world_pose()
        position = np.asarray(position, dtype=np.float64).copy()
        rotation = np.asarray(rotation, dtype=np.float64).copy()
        linear = np.asarray(self.articulation.get_linear_velocity(), dtype=np.float64).copy()
        angular = np.asarray(self.articulation.get_angular_velocity(), dtype=np.float64).copy()
        if not all(np.isfinite(value).all() for value in (position, rotation, linear, angular)):
            raise RuntimeError("non-finite Go2 pose/velocity")
        lidar_started_ns = time.perf_counter_ns()
        lidar_points = self._lidar(position, rotation)
        lidar_elapsed_sec = (time.perf_counter_ns() - lidar_started_ns) / 1e9
        link_centers = self._link_centers_base(position, rotation)
        support_plane = self._support_plane_world_z()
        snapshot_elapsed_sec = (time.perf_counter_ns() - snapshot_started_ns) / 1e9
        if (
            not math.isfinite(lidar_elapsed_sec)
            or lidar_elapsed_sec < 0.0
            or not math.isfinite(snapshot_elapsed_sec)
            or snapshot_elapsed_sec < lidar_elapsed_sec
        ):
            raise RuntimeError("physical render-state timing evidence is invalid")
        return RenderSnapshot(
            generation=generation,
            stamp_ns=stamp_ns,
            world_time=world_time,
            state={
                "position": position,
                "rotation_wxyz": rotation,
                "linear_velocity": linear,
                "angular_velocity": angular,
                "lidar_points": lidar_points,
                "link_centers_base": link_centers,
                "support_plane_world_z": support_plane,
                "lidar_query_sec": lidar_elapsed_sec,
                "state_snapshot_sec": snapshot_elapsed_sec,
            },
        )

    def capture(self, step: SafeStep) -> Mapping[str, Any]:
        capture_started_ns = time.perf_counter_ns()
        snapshot = self._pending_capture_snapshot
        if (
            not step.rendered
            or snapshot is None
            or step.render_id != self._render_pipeline.last_render_id
            or step.render_generation != self._generation
            or snapshot.generation != self._generation
            or snapshot.stamp_ns != step.sim_stamp_ns
            or step.sim_stamp_ns != self._render_pipeline.last_output_stamp_ns
        ):
            raise RuntimeError("camera capture does not match the latest current-generation render")
        stamp_ns = step.sim_stamp_ns
        np = self.np
        position = np.asarray(snapshot.state["position"], dtype=np.float64)
        rotation = np.asarray(snapshot.state["rotation_wxyz"], dtype=np.float64)
        linear = np.asarray(snapshot.state["linear_velocity"], dtype=np.float64)
        angular = np.asarray(snapshot.state["angular_velocity"], dtype=np.float64)
        camera_started_ns = time.perf_counter_ns()
        color_frame = self.color_camera.get_current_frame()
        depth_frame = self.depth_camera.get_current_frame()
        front_frame = self.front_camera.get_current_frame()
        observed_metadata = {
            "color": self._render_metadata(color_frame, "color"),
            "depth": self._render_metadata(depth_frame, "depth"),
            "front": self._render_metadata(front_frame, "front"),
        }
        if observed_metadata != self._pending_capture_metadata:
            raise RuntimeError("camera frame changed or regressed between render tick and capture")
        if any(value[0] != step.render_id for value in observed_metadata.values()):
            raise RuntimeError("camera frame does not match SafeStep render identity")
        depth = np.asarray(depth_frame.get("distance_to_image_plane"), dtype=np.float32)
        if depth.shape != (480, 640):
            raise RuntimeError(f"unexpected D435i depth shape: {depth.shape}")
        valid_depth_count, valid_depth_ratio, valid_depth_ready = valid_depth_evidence(depth)
        if not valid_depth_ready:
            raise CaptureNotReady(
                f"current render depth content below frozen 10% gate: {valid_depth_count}/307200"
            )
        depth = np.ascontiguousarray(np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0))
        rgb = self._rgb8(color_frame.get("rgb"), (480, 640))
        front = self._rgb8(front_frame.get("rgb"), (240, 320))
        d435i_rgb_content = rgb_content_evidence(rgb)
        front_publish = downsample_rgb_2x2(front)
        front_rgb_content = rgb_content_evidence(front_publish)
        if d435i_rgb_content["valid"] is not True or front_rgb_content["valid"] is not True:
            raise CaptureNotReady("current render RGB is black or constant despite frozen DomeLight")
        camera_elapsed_sec = (time.perf_counter_ns() - camera_started_ns) / 1e9
        color_fx = 640.0 / (2.0 * math.tan(math.radians(69.4) / 2.0))
        color_fy = 480.0 / (2.0 * math.tan(math.radians(42.5) / 2.0))
        depth_fx = 640.0 / (2.0 * math.tan(math.radians(87.0) / 2.0))
        depth_fy = 480.0 / (2.0 * math.tan(math.radians(58.0) / 2.0))
        front_fx = 320.0 / (2.0 * math.tan(math.radians(120.0) / 2.0))
        front_fy = 240.0 / (2.0 * math.tan(math.radians(75.0) / 2.0))
        fixed = [
            {"parent": "base_link", "child": "go2_l1_lidar", "translation": [0.25, 0.0, 0.18], "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            {"parent": "base_link", "child": "go2_imu_link", "translation": [0.0, 0.0, 0.0], "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            {"parent": "base_link", "child": "go2_depth_optical_frame", "translation": [0.20, 0.0, 0.20], "rotation_wxyz": self._optical_quaternion(20.0)},
            {"parent": "base_link", "child": "go2_d435i_color_optical_frame", "translation": [0.20, 0.0, 0.20], "rotation_wxyz": self._optical_quaternion(20.0)},
            {"parent": "base_link", "child": "go2_front_rgb_optical_frame", "translation": [0.29, 0.0, -0.06], "rotation_wxyz": self._optical_quaternion(8.0)},
        ]
        lidar_points = np.asarray(snapshot.state["lidar_points"], dtype=np.float32)
        lidar_elapsed_sec = float(snapshot.state["lidar_query_sec"])
        link_centers = list(snapshot.state["link_centers_base"])
        support_plane = float(snapshot.state["support_plane_world_z"])
        state_snapshot_sec = float(snapshot.state["state_snapshot_sec"])
        capture_elapsed_sec = (
            state_snapshot_sec
            + (time.perf_counter_ns() - capture_started_ns) / 1e9
        )
        timing = {
            "clock": "monotonic_perf_counter",
            "camera_read_content_sec": camera_elapsed_sec,
            "lidar_query_sec": lidar_elapsed_sec,
            "total_capture_sec": capture_elapsed_sec,
            "lidar_raycast_count": 8 * 180,
            "physics_step_sec": 1.0 / PHYSICS_HZ,
            "render_reference_denominator": self._render_reference_denominator,
            "last_render_only_drain_count": self._last_render_drain_count,
            "maximum_render_only_drain_count": self._render_only_drain_max,
            "total_render_only_drain_count": self._render_only_drain_total,
            "startup_render_only_drain_count": self._startup_render_only_drain_count,
            "render_resync_limit": self._render_resync_limit,
            "render_resync_event_count": self._render_resync_event_count,
            "render_resync_drop_count": self._render_resync_drop_count,
            "render_resync_extra_drain_total": self._render_resync_extra_drain_total,
            "render_pipeline_lag_steps": self._render_pipeline.lag_steps,
            "render_latency_mode": self._render_pipeline.render_latency_mode,
            "render_latency_negotiated_before_workload": True,
            "render_latency_negotiation": dict(self._render_latency_negotiation),
            "render_stamp_deviation_tolerance_ns": self._render_pipeline.max_stamp_deviation_ns,
            "render_stamp_deviation_count": self._render_pipeline.stamp_deviation_count,
            "render_stamp_deviation_maximum_ns": self._render_pipeline.maximum_stamp_deviation_ns,
            "render_stamp_deviation_last_ns": self._render_pipeline.last_stamp_deviation_ns,
            "render_reference_lag_sec": self._render_clock.reference_lag_sec,
            "reference_clock_source": "seed_integer_ns_plus_global_physics_step_count",
            "reference_clock_seed_stamp_ns": self._render_clock.seed_stamp_ns,
            "reference_clock_seed_physical_step": self._render_clock_seed_physical_step,
            "physics_step_ns": self._render_clock.physics_step_ns,
            "physical_step_total": self._physical_step_total,
            "state_snapshot_sec": state_snapshot_sec,
            "zero_delta_physics_latch_count": self._physics_latch_total,
            "front_clipping_range_m": list(self._front_clipping_range),
        }
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (camera_elapsed_sec, lidar_elapsed_sec, capture_elapsed_sec)
        ) or capture_elapsed_sec < max(camera_elapsed_sec, lidar_elapsed_sec):
            raise RuntimeError("capture stage timing evidence is invalid")
        return {
            "d435i_rgb": {
                "rgb8": rgb,
                "front_rgb8": front,
                "d435i_fx": color_fx,
                "d435i_fy": color_fy,
                "camera_prim": "/World/Go2/base/internvla_camera",
                "hfov_deg": 69.4,
                "vfov_deg": 42.5,
                "render_id": step.render_id,
                "render_generation": step.render_generation,
                "d435i_content_evidence": d435i_rgb_content,
                "front_publish_content_evidence": front_rgb_content,
                "front_fx": front_fx,
                "front_fy": front_fy,
            },
            "d435i_depth": {
                "depth_m": depth,
                "fx": depth_fx,
                "fy": depth_fy,
                "cx": 319.5,
                "cy": 239.5,
                "pitch_down_deg": 20.0,
                "translation_from_base_m": [0.20, 0.0, 0.20],
                "support_plane_world_z": support_plane,
                "camera_prim": "/World/Go2/base/t4_d435i_depth",
                "hfov_deg": 87.0,
                "vfov_deg": 58.0,
                "link_centers_base": link_centers,
                "valid_in_range_count": valid_depth_count,
                "valid_in_range_ratio": valid_depth_ratio,
                "diagnostic_geometry_ids": [item["name"] for item in DIAGNOSTIC_GEOMETRY],
                "diagnostic_light_evidence": self._diagnostic_light_evidence(),
                "render_id": step.render_id,
                "render_generation": step.render_generation,
                "capture_timing": timing,
            },
            "lidar": {"points_lidar": lidar_points},
            "pose": {
                "position": position,
                "rotation_wxyz": rotation,
                "linear_velocity": linear,
                "angular_velocity": angular,
            },
            "tf": {
                "base_translation": position,
                "base_rotation_wxyz": rotation,
                "fixed": fixed,
            },
        }

    def runtime_evidence(self) -> dict[str, Any]:
        return {
            "render_resync_limit": self._render_resync_limit,
            "render_resync_event_count": self._render_resync_event_count,
            "render_resync_drop_count": self._render_resync_drop_count,
            "render_resync_extra_drain_total": self._render_resync_extra_drain_total,
            "render_only_drain_total": self._render_only_drain_total,
            "render_only_drain_max": self._render_only_drain_max,
        }

    def close(self) -> None:
        errors = []
        try:
            self._apply_safe_stop(count_physics_step=False)
        except BaseException as exc:
            errors.append(f"safe_stop: {type(exc).__name__}: {exc}")
        try:
            self.world.stop()
        except BaseException as exc:
            errors.append(f"world_stop: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))
