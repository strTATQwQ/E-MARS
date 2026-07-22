"""Trace Nvblox distance slices before and after Nav2 inflation.

Stage B is reconstructed cell-for-cell from the official
NvbloxCostmapLayer algorithm.  The reconstruction uses the same slice, TF,
costmap geometry, nearest-cell lookup, binary threshold, and unknown policy as
the plugin.  Stage C is the received Nav2 costmap_raw message.
"""

from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import rclpy
from nav2_msgs.msg import Costmap
from nav2_msgs.srv import GetCostmap
from nvblox_msgs.msg import DistanceMapSlice
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Int32, String
from tf2_ros import Buffer, TransformException, TransformListener


FREE_SPACE = np.uint8(0)
LETHAL_OBSTACLE = np.uint8(254)
NO_INFORMATION = np.uint8(255)


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _yaw(quaternion: Any) -> float:
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    quantiles = np.percentile(finite, [0, 5, 25, 50, 75, 95, 100])
    return dict(
        zip(
            ("min", "p05", "p25", "p50", "p75", "p95", "max"),
            (float(value) for value in quantiles),
            strict=True,
        )
    )


def _cost_counts(costs: np.ndarray) -> dict[str, int]:
    flat = np.asarray(costs, dtype=np.uint8).reshape(-1)
    free = int(np.count_nonzero(flat == FREE_SPACE))
    lethal = int(np.count_nonzero(flat == LETHAL_OBSTACLE))
    unknown = int(np.count_nonzero(flat == NO_INFORMATION))
    inflated = int(np.count_nonzero((flat > FREE_SPACE) & (flat < LETHAL_OBSTACLE)))
    return {
        "cell_count": int(flat.size),
        "free_count": free,
        "occupied_count": lethal + inflated,
        "lethal_count": lethal,
        "inflated_count": inflated,
        "unknown_count": unknown,
    }


def _classify_cost(value: int | None) -> str:
    if value is None:
        return "out_of_bounds"
    if value == 0:
        return "free"
    if value == 255:
        return "unknown"
    if value == 254:
        return "lethal"
    return "inflated"


def _nearest_integer(values: np.ndarray) -> np.ndarray:
    """Match Eigen's round-away-from-zero behavior used by the plugin."""
    bounded = np.clip(values, -1.0e9, 1.0e9)
    return np.where(
        bounded >= 0.0,
        np.floor(bounded + 0.5),
        np.ceil(bounded - 0.5),
    ).astype(np.int64)


class CostmapStageTracer(Node):
    """Persist raw slice, pre-inflation, and final costmap evidence."""

    def __init__(self) -> None:
        super().__init__("internvla_t4_costmap_stage_tracer")
        self.declare_parameter("result_dir", "")
        self.declare_parameter("robot_radius_m", 0.30)
        self.declare_parameter("footprint_padding_m", 0.02)
        self.declare_parameter("slice_height_m", 0.15)
        self.declare_parameter("slice_min_height_m", 0.15)
        self.declare_parameter("slice_max_height_m", 0.65)
        self.declare_parameter("binary_conversion", True)
        self.declare_parameter("occupied_distance_threshold_m", 0.0)
        self.declare_parameter("service_startup_delay_sec", 20.0)
        self.declare_parameter("pre_inflation_layer_name", "nvblox_layer")
        self.result_dir = Path(str(self.get_parameter("result_dir").value)).resolve()
        if not str(self.result_dir):
            raise RuntimeError("result_dir is required")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.result_dir / "nvblox_distance_slice_stage_a.jsonl"
        self.stage_path = self.result_dir / "costmap_stage_records.jsonl"
        self.summary_path = self.result_dir / "costmap_stage_summary.json"
        self.service_stage_path = (
            self.result_dir / "costmap_service_stage_records.jsonl"
        )
        for path in (
            self.raw_path,
            self.stage_path,
            self.service_stage_path,
            self.summary_path,
        ):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite stage evidence: {path}")

        self.robot_radius_m = float(self.get_parameter("robot_radius_m").value)
        self.padding_m = float(self.get_parameter("footprint_padding_m").value)
        self.slice_height_m = float(self.get_parameter("slice_height_m").value)
        self.slice_min_height_m = float(
            self.get_parameter("slice_min_height_m").value
        )
        self.slice_max_height_m = float(
            self.get_parameter("slice_max_height_m").value
        )
        self.binary_conversion = bool(self.get_parameter("binary_conversion").value)
        self.distance_threshold_m = float(
            self.get_parameter("occupied_distance_threshold_m").value
        )
        self.service_startup_delay_sec = float(
            self.get_parameter("service_startup_delay_sec").value
        )
        self.pre_inflation_layer_name = str(
            self.get_parameter("pre_inflation_layer_name").value
        )
        if self.pre_inflation_layer_name not in {
            "nvblox_layer",
            "footprint_clearing_layer",
        }:
            raise RuntimeError("unsupported pre-inflation layer service")
        if not self.binary_conversion:
            raise RuntimeError("R2 tracer currently audits the frozen binary conversion only")

        self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._lock = threading.RLock()
        self._latest_slice: DistanceMapSlice | None = None
        self._last_raw_write_ns = 0
        self._generation = -1
        self._records = {
            "raw_slice": 0,
            "local": 0,
            "global": 0,
            "local_grid": 0,
            "global_grid": 0,
            "service_local": 0,
            "service_global": 0,
            "skipped_before_first_slice": 0,
            "skipped_service_unavailable": 0,
            "errors": 0,
        }
        self._consecutive_final_free = {
            "local": 0,
            "global": 0,
            "local_grid": 0,
            "global_grid": 0,
            "service_local": 0,
            "service_global": 0,
        }
        self._maximum_consecutive_final_free = dict(self._consecutive_final_free)
        self._sequence = 0
        self._started_monotonic = time.monotonic()

        self.create_subscription(
            DistanceMapSlice,
            "/nvblox_node/static_map_slice",
            self._on_slice,
            10,
        )
        self._service_clients = {
            "local": {
                "b": self.create_client(
                    GetCostmap,
                    f"/local_costmap/get_{self.pre_inflation_layer_name}",
                ),
                "c": self.create_client(
                    GetCostmap,
                    "/local_costmap/get_costmap",
                ),
            },
            "global": {
                "b": self.create_client(
                    GetCostmap,
                    f"/global_costmap/get_{self.pre_inflation_layer_name}",
                ),
                "c": self.create_client(
                    GetCostmap,
                    "/global_costmap/get_costmap",
                ),
            },
        }
        self._service_batch = 0
        self._service_pending = {"local": False, "global": False}
        self._service_responses: dict[
            tuple[str, int], dict[str, Costmap]
        ] = {}
        self.create_timer(1.0, self._poll_costmap_services)
        self.create_subscription(
            Int32,
            "/internvla_t4/map_reset_generation",
            self._on_reset,
            10,
        )
        self.clearing_status_path = self.result_dir / "footprint_clearing_status.jsonl"
        if self.clearing_status_path.exists():
            raise FileExistsError(self.clearing_status_path)
        for logical in ("local", "global"):
            self.create_subscription(
                String,
                f"/{logical}_costmap/footprint_clearing_layer/status",
                lambda message, logical=logical: self._on_clearing_status(
                    message, logical
                ),
                10,
            )
        self._write_summary()

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        record = {
            "schema_version": 1,
            **payload,
            "wall_time_unix": time.time(),
        }
        with self._lock:
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                )

    def _write_summary(self) -> None:
        payload = {
            "schema_version": 1,
            "status": "CAPTURING",
            "generation": self._generation,
            "records": self._records,
            "maximum_consecutive_final_free_updates": (
                self._maximum_consecutive_final_free
            ),
            "conversion_contract": {
                "implementation": "official_nvblox_costmap_layer_exact_reconstruction",
                "actual_pre_inflation_layer_service": self.pre_inflation_layer_name,
                "convert_to_binary_costmap": self.binary_conversion,
                "occupied_if_distance_m_le": self.distance_threshold_m,
                "free_if_distance_m_gt": self.distance_threshold_m,
                "unknown_or_out_of_bounds": 255,
                "known_occupied": 254,
                "known_free": 0,
                "slice_lookup": "nearest_index_round_away_from_zero",
            },
            "frozen_geometry": {
                "robot_radius_m": self.robot_radius_m,
                "footprint_padding_m": self.padding_m,
                "effective_footprint_radius_m": self.robot_radius_m + self.padding_m,
                "slice_height_m": self.slice_height_m,
                "slice_min_height_m": self.slice_min_height_m,
                "slice_max_height_m": self.slice_max_height_m,
                "service_startup_delay_sec": self.service_startup_delay_sec,
            },
            "wall_time_unix": time.time(),
        }
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.summary_path)

    def _on_reset(self, message: Int32) -> None:
        with self._lock:
            self._generation = int(message.data)
            self._latest_slice = None
            self._consecutive_final_free = {
                "local": 0,
                "global": 0,
                "local_grid": 0,
                "global_grid": 0,
                "service_local": 0,
                "service_global": 0,
            }
            self._write_summary()

    def _on_clearing_status(self, message: String, logical: str) -> None:
        try:
            payload = json.loads(str(message.data))
        except (TypeError, ValueError, json.JSONDecodeError):
            with self._lock:
                self._records["errors"] += 1
            return
        payload.update(
            {
                "costmap_name": logical,
                "wall_time_unix": time.time(),
            }
        )
        with self.clearing_status_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")

    def _poll_costmap_services(self) -> None:
        """Request C, then request B with C's exact geometry specification."""
        if time.monotonic() - self._started_monotonic < self.service_startup_delay_sec:
            with self._lock:
                self._records["skipped_service_unavailable"] += 2
            return
        for logical, clients in self._service_clients.items():
            with self._lock:
                pending = self._service_pending[logical]
            if pending:
                continue
            if not all(client.service_is_ready() for client in clients.values()):
                with self._lock:
                    self._records["skipped_service_unavailable"] += 1
                continue
            with self._lock:
                self._service_batch += 1
                batch = self._service_batch
                self._service_pending[logical] = True
                self._service_responses[(logical, batch)] = {}
            future = clients["c"].call_async(GetCostmap.Request())
            future.add_done_callback(
                lambda completed, logical=logical, batch=batch: (
                    self._on_service_response(logical, "c", batch, completed)
                )
            )

    def _on_service_response(
        self,
        logical: str,
        stage: str,
        batch: int,
        future: Any,
    ) -> None:
        try:
            response = future.result()
        except Exception as error:  # rclpy service exceptions are runtime-specific
            with self._lock:
                self._service_pending[logical] = False
                self._service_responses.pop((logical, batch), None)
            self._record_error(
                f"service_{logical}_{stage}",
                f"GetCostmap call failed: {error}",
            )
            return
        pair: dict[str, Costmap] | None = None
        with self._lock:
            key = (logical, batch)
            if key not in self._service_responses:
                return
            self._service_responses[key][stage] = response.map
        if stage == "c":
            request = GetCostmap.Request()
            request.specs = response.map.metadata
            future_b = self._service_clients[logical]["b"].call_async(request)
            future_b.add_done_callback(
                lambda completed, logical=logical, batch=batch: (
                    self._on_service_response(logical, "b", batch, completed)
                )
            )
            return
        with self._lock:
            key = (logical, batch)
            if key not in self._service_responses:
                return
            if set(self._service_responses[key]) == {"b", "c"}:
                pair = self._service_responses.pop(key)
                self._service_pending[logical] = False
        if pair is not None:
            self._record_service_pair(logical, batch, pair["b"], pair["c"])

    @staticmethod
    def _costmap_metadata(message: Costmap) -> dict[str, Any]:
        return {
            "frame": message.header.frame_id,
            "header_stamp_ns": _stamp_ns(message.header.stamp),
            "map_load_time_ns": _stamp_ns(message.metadata.map_load_time),
            "update_time_ns": _stamp_ns(message.metadata.update_time),
            "layer": message.metadata.layer,
            "resolution_m": float(message.metadata.resolution),
            "width": int(message.metadata.size_x),
            "height": int(message.metadata.size_y),
            "origin_xyz_m": [
                float(message.metadata.origin.position.x),
                float(message.metadata.origin.position.y),
                float(message.metadata.origin.position.z),
            ],
            "origin_yaw_rad": _yaw(message.metadata.origin.orientation),
        }

    def _record_service_pair(
        self,
        logical: str,
        batch: int,
        stage_b_message: Costmap,
        stage_c_message: Costmap,
    ) -> None:
        with self._lock:
            slice_message = self._latest_slice
            generation = self._generation
        if slice_message is None:
            return
        b_width = int(stage_b_message.metadata.size_x)
        b_height = int(stage_b_message.metadata.size_y)
        c_width = int(stage_c_message.metadata.size_x)
        c_height = int(stage_c_message.metadata.size_y)
        if not b_width or not b_height:
            with self._lock:
                self._records["skipped_service_unavailable"] += 1
            return
        if (b_width, b_height) != (c_width, c_height):
            self._record_error(
                f"service_{logical}",
                "Stage B/C service dimensions are mismatched: "
                f"B={b_width}x{b_height}, C={c_width}x{c_height}",
            )
            return
        stage_b = np.asarray(stage_b_message.data, dtype=np.uint8)
        stage_c = np.asarray(stage_c_message.data, dtype=np.uint8)
        if stage_b.size != b_width * b_height or stage_c.size != c_width * c_height:
            self._record_error(f"service_{logical}", "Stage B/C data length mismatch")
            return
        stage_b = stage_b.reshape((b_height, b_width))
        stage_c = stage_c.reshape((c_height, c_width))
        try:
            reconstruction, backing_distance, transform_evidence = (
                self._pre_inflation(stage_c_message, slice_message)
            )
            robot_x, robot_y, robot_yaw, robot_tf_basis = self._robot_pose(
                stage_c_message.header.frame_id,
                stage_c_message.header.stamp,
            )
        except TransformException as error:
            self._record_error(f"service_{logical}", f"TF lookup failed: {error}")
            return
        resolution = float(stage_c_message.metadata.resolution)
        origin_x = float(stage_c_message.metadata.origin.position.x)
        origin_y = float(stage_c_message.metadata.origin.position.y)
        stage_b_spatial = self._spatial_evidence(
            stage_b,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
        )
        stage_c_spatial = self._spatial_evidence(
            stage_c,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
        )
        b_to_c_different = stage_b != stage_c
        reconstruction_different = reconstruction != stage_b
        stream = f"service_{logical}"
        stable_final_free = (
            _cost_counts(stage_c)["free_count"] > 0
            and stage_c_spatial["connected_free_cell_count"] > 0
        )
        with self._lock:
            if stable_final_free:
                self._consecutive_final_free[stream] += 1
            else:
                self._consecutive_final_free[stream] = 0
            self._maximum_consecutive_final_free[stream] = max(
                self._maximum_consecutive_final_free[stream],
                self._consecutive_final_free[stream],
            )
            consecutive = self._consecutive_final_free[stream]
        known_backing = backing_distance[np.isfinite(backing_distance)]
        known_backing = known_backing[
            ~np.isclose(
                known_backing,
                float(slice_message.unknown_value),
                rtol=0.0,
                atol=1.0e-6,
            )
        ]
        self._append(
            self.service_stage_path,
            {
                "event": "actual_costmap_service_stage_pair",
                "batch": batch,
                "stream": stream,
                "generation": generation,
                "stage_b_metadata": self._costmap_metadata(stage_b_message),
                "stage_c_metadata": self._costmap_metadata(stage_c_message),
                "slice_metadata": {
                    "frame": slice_message.header.frame_id,
                    "stamp_ns": _stamp_ns(slice_message.header.stamp),
                    "resolution_m": float(slice_message.resolution),
                    "origin_xyz_m": [
                        float(slice_message.origin.x),
                        float(slice_message.origin.y),
                        float(slice_message.origin.z),
                    ],
                    "slice_height_m": self.slice_height_m,
                    "slice_min_height_m": self.slice_min_height_m,
                    "slice_max_height_m": self.slice_max_height_m,
                },
                "conversion": {
                    "convert_to_binary_costmap": self.binary_conversion,
                    "occupied_if_distance_m_le": self.distance_threshold_m,
                    "free_if_distance_m_gt": self.distance_threshold_m,
                    "unknown_policy": "slice_unknown_or_out_of_bounds_to_255",
                },
                "transform": transform_evidence,
                "robot_tf_basis": robot_tf_basis,
                "backing_slice_distance_m": _percentiles(known_backing),
                "stage_b_actual_pre_inflation": {
                    "provenance": (
                        f"/{logical}_costmap/get_{self.pre_inflation_layer_name}"
                    ),
                    "counts": _cost_counts(stage_b),
                    "spatial": stage_b_spatial,
                },
                "stage_c_actual_final": {
                    "provenance": f"/{logical}_costmap/get_costmap",
                    "counts": _cost_counts(stage_c),
                    "spatial": stage_c_spatial,
                    "consecutive_updates_with_connected_free": consecutive,
                },
                "stage_b_to_c_cellwise": {
                    "equal_count": int(np.count_nonzero(~b_to_c_different)),
                    "different_count": int(np.count_nonzero(b_to_c_different)),
                    "b_free_to_c_free_count": int(
                        np.count_nonzero(
                            (stage_b == FREE_SPACE) & (stage_c == FREE_SPACE)
                        )
                    ),
                    "b_free_to_c_inflated_count": int(
                        np.count_nonzero(
                            (stage_b == FREE_SPACE)
                            & (stage_c > FREE_SPACE)
                            & (stage_c < LETHAL_OBSTACLE)
                        )
                    ),
                    "b_free_to_c_lethal_count": int(
                        np.count_nonzero(
                            (stage_b == FREE_SPACE) & (stage_c == LETHAL_OBSTACLE)
                        )
                    ),
                    "b_free_to_c_unknown_count": int(
                        np.count_nonzero(
                            (stage_b == FREE_SPACE) & (stage_c == NO_INFORMATION)
                        )
                    ),
                },
                "reconstruction_cross_check": {
                    "equal_count": int(np.count_nonzero(~reconstruction_different)),
                    "different_count": int(np.count_nonzero(reconstruction_different)),
                },
            },
        )
        with self._lock:
            self._records[stream] += 1
            self._write_summary()

    def _lookup_transform(
        self, target_frame: str, source_frame: str, stamp: Any
    ) -> Any:
        return self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time.from_msg(stamp),
            timeout=Duration(seconds=0.25),
        )

    def _robot_pose(
        self, frame: str, stamp: Any
    ) -> tuple[float, float, float, str]:
        try:
            transform = self._lookup_transform(frame, "base_link", stamp)
            basis = "message_stamp"
        except TransformException:
            transform = self.tf_buffer.lookup_transform(
                frame,
                "base_link",
                Time(),
                timeout=Duration(seconds=0.25),
            )
            basis = "latest_fallback"
        return (
            float(transform.transform.translation.x),
            float(transform.transform.translation.y),
            _yaw(transform.transform.rotation),
            basis,
        )

    @staticmethod
    def _world_to_cell(
        x: float,
        y: float,
        origin_x: float,
        origin_y: float,
        resolution: float,
        width: int,
        height: int,
    ) -> tuple[int, int] | None:
        column = math.floor((x - origin_x) / resolution)
        row = math.floor((y - origin_y) / resolution)
        if column < 0 or row < 0 or column >= width or row >= height:
            return None
        return column, row

    def _spatial_evidence(
        self,
        costs: np.ndarray,
        *,
        resolution: float,
        origin_x: float,
        origin_y: float,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> dict[str, Any]:
        height, width = costs.shape
        robot_cell = self._world_to_cell(
            robot_x,
            robot_y,
            origin_x,
            origin_y,
            resolution,
            width,
            height,
        )
        center_cost: int | None = None
        if robot_cell is not None:
            center_cost = int(costs[robot_cell[1], robot_cell[0]])

        footprint_counts = {
            "cell_count": 0,
            "free_count": 0,
            "inflated_count": 0,
            "lethal_count": 0,
            "unknown_count": 0,
            "out_of_bounds_count": 0,
        }
        effective_radius = self.robot_radius_m + self.padding_m
        min_column = math.floor((robot_x - effective_radius - origin_x) / resolution)
        max_column = math.floor((robot_x + effective_radius - origin_x) / resolution)
        min_row = math.floor((robot_y - effective_radius - origin_y) / resolution)
        max_row = math.floor((robot_y + effective_radius - origin_y) / resolution)
        for row in range(min_row, max_row + 1):
            for column in range(min_column, max_column + 1):
                cell_x = origin_x + (column + 0.5) * resolution
                cell_y = origin_y + (row + 0.5) * resolution
                if math.hypot(cell_x - robot_x, cell_y - robot_y) > effective_radius:
                    continue
                footprint_counts["cell_count"] += 1
                if column < 0 or row < 0 or column >= width or row >= height:
                    footprint_counts["out_of_bounds_count"] += 1
                    continue
                value = int(costs[row, column])
                key = _classify_cost(value) + "_count"
                footprint_counts[key] += 1
        footprint_counts["all_traversable"] = (
            footprint_counts["cell_count"] > 0
            and footprint_counts["lethal_count"] == 0
            and footprint_counts["unknown_count"] == 0
            and footprint_counts["out_of_bounds_count"] == 0
        )

        free = costs == FREE_SPACE
        connected = np.zeros((height, width), dtype=np.bool_)
        if robot_cell is not None and free[robot_cell[1], robot_cell[0]]:
            queue: deque[tuple[int, int]] = deque([robot_cell])
            connected[robot_cell[1], robot_cell[0]] = True
            while queue:
                column, row = queue.popleft()
                for next_column, next_row in (
                    (column - 1, row),
                    (column + 1, row),
                    (column, row - 1),
                    (column, row + 1),
                ):
                    if (
                        0 <= next_column < width
                        and 0 <= next_row < height
                        and free[next_row, next_column]
                        and not connected[next_row, next_column]
                    ):
                        connected[next_row, next_column] = True
                        queue.append((next_column, next_row))

        forward: dict[str, Any] = {}
        for distance in (0.5, 1.0, 1.5):
            target_x = robot_x + distance * math.cos(robot_yaw)
            target_y = robot_y + distance * math.sin(robot_yaw)
            target_cell = self._world_to_cell(
                target_x,
                target_y,
                origin_x,
                origin_y,
                resolution,
                width,
                height,
            )
            value = (
                int(costs[target_cell[1], target_cell[0]])
                if target_cell is not None
                else None
            )
            forward[f"{distance:.1f}_m"] = {
                "world_xy": [target_x, target_y],
                "cell": list(target_cell) if target_cell is not None else None,
                "cost": value,
                "class": _classify_cost(value),
                "connected_free_from_robot": bool(
                    target_cell is not None
                    and connected[target_cell[1], target_cell[0]]
                ),
            }

        return {
            "robot_pose_xy_yaw": [robot_x, robot_y, robot_yaw],
            "robot_cell": list(robot_cell) if robot_cell is not None else None,
            "robot_cell_cost": center_cost,
            "robot_cell_class": _classify_cost(center_cost),
            "footprint": footprint_counts,
            "connected_free_cell_count": int(np.count_nonzero(connected)),
            "connected_free_area_m2": float(
                np.count_nonzero(connected) * resolution * resolution
            ),
            "forward": forward,
        }

    def _on_slice(self, message: DistanceMapSlice) -> None:
        now_ns = self.get_clock().now().nanoseconds
        with self._lock:
            self._latest_slice = message
            if now_ns - self._last_raw_write_ns < 500_000_000:
                return
            self._last_raw_write_ns = now_ns
            generation = self._generation
        values = np.asarray(message.data, dtype=np.float32)
        expected = int(message.width) * int(message.height)
        if values.size != expected:
            self._record_error(
                "raw_slice",
                f"data length {values.size} does not match {expected}",
            )
            return
        unknown = (~np.isfinite(values)) | np.isclose(
            values,
            float(message.unknown_value),
            rtol=0.0,
            atol=1.0e-6,
        )
        known = ~unknown
        binary = np.full(values.shape, NO_INFORMATION, dtype=np.uint8)
        binary[known & (values <= self.distance_threshold_m)] = LETHAL_OBSTACLE
        binary[known & (values > self.distance_threshold_m)] = FREE_SPACE
        robot_evidence: dict[str, Any] | None = None
        robot_tf_error: str | None = None
        tf_basis: str | None = None
        try:
            robot_x, robot_y, robot_yaw, tf_basis = self._robot_pose(
                message.header.frame_id,
                message.header.stamp,
            )
            robot_evidence = self._spatial_evidence(
                binary.reshape((int(message.height), int(message.width))),
                resolution=float(message.resolution),
                origin_x=float(message.origin.x) - 0.5 * float(message.resolution),
                origin_y=float(message.origin.y) - 0.5 * float(message.resolution),
                robot_x=robot_x,
                robot_y=robot_y,
                robot_yaw=robot_yaw,
            )
        except TransformException as error:
            robot_tf_error = str(error)
        self._append(
            self.raw_path,
            {
                "stage": "A_raw_nvblox_distance_slice",
                "generation": generation,
                "slice_stamp_ns": _stamp_ns(message.header.stamp),
                "frame": message.header.frame_id,
                "resolution_m": float(message.resolution),
                "origin_xyz_m": [
                    float(message.origin.x),
                    float(message.origin.y),
                    float(message.origin.z),
                ],
                "width": int(message.width),
                "height": int(message.height),
                "slice_height_m": self.slice_height_m,
                "slice_min_height_m": self.slice_min_height_m,
                "slice_max_height_m": self.slice_max_height_m,
                "unknown_value": float(message.unknown_value),
                "free_positive_count": int(
                    np.count_nonzero(known & (values > self.distance_threshold_m))
                ),
                "occupied_nonpositive_count": int(
                    np.count_nonzero(known & (values <= self.distance_threshold_m))
                ),
                "unknown_count": int(np.count_nonzero(unknown)),
                "known_distance_m": _percentiles(values[known]),
                "conversion": {
                    "convert_to_binary_costmap": self.binary_conversion,
                    "occupied_if_distance_m_le": self.distance_threshold_m,
                    "unknown_policy": "preserve_as_255",
                },
                "robot_tf_basis": tf_basis,
                "robot_tf_error": robot_tf_error,
                "spatial": robot_evidence,
            },
        )
        with self._lock:
            self._records["raw_slice"] += 1
            self._write_summary()

    def _record_error(self, stream: str, error: str) -> None:
        self._append(
            self.stage_path,
            {
                "event": "trace_error",
                "stream": stream,
                "generation": self._generation,
                "error": error,
            },
        )
        with self._lock:
            self._records["errors"] += 1
            self._write_summary()

    def _pre_inflation(
        self,
        message: Costmap,
        slice_message: DistanceMapSlice,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        width = int(message.metadata.size_x)
        height = int(message.metadata.size_y)
        resolution = float(message.metadata.resolution)
        origin_x = float(message.metadata.origin.position.x)
        origin_y = float(message.metadata.origin.position.y)
        columns = np.tile(np.arange(width, dtype=np.float64), height)
        rows = np.repeat(np.arange(height, dtype=np.float64), width)
        x_global = origin_x + (columns + 0.5) * resolution
        y_global = origin_y + (rows + 0.5) * resolution

        transform = self._lookup_transform(
            message.header.frame_id,
            slice_message.header.frame_id,
            slice_message.header.stamp,
        )
        translation_x = float(transform.transform.translation.x)
        translation_y = float(transform.transform.translation.y)
        transform_yaw = _yaw(transform.transform.rotation)
        cosine = math.cos(transform_yaw)
        sine = math.sin(transform_yaw)
        delta_x = x_global - translation_x
        delta_y = y_global - translation_y
        x_slice = cosine * delta_x + sine * delta_y
        y_slice = -sine * delta_x + cosine * delta_y
        slice_resolution = float(slice_message.resolution)
        scaled_columns = (
            x_slice - float(slice_message.origin.x)
        ) / slice_resolution
        scaled_rows = (
            y_slice - float(slice_message.origin.y)
        ) / slice_resolution
        finite_scaled = np.isfinite(scaled_columns) & np.isfinite(scaled_rows)
        slice_columns = _nearest_integer(
            np.where(finite_scaled, scaled_columns, -1.0e12)
        )
        slice_rows = _nearest_integer(
            np.where(finite_scaled, scaled_rows, -1.0e12)
        )
        in_bounds = (
            finite_scaled
            & (slice_columns >= 0)
            & (slice_rows >= 0)
            & (slice_columns < int(slice_message.width))
            & (slice_rows < int(slice_message.height))
        )
        slice_values = np.asarray(slice_message.data, dtype=np.float32)
        backing_distance = np.full(columns.shape, np.nan, dtype=np.float32)
        backing_indices = (
            slice_rows[in_bounds] * int(slice_message.width) + slice_columns[in_bounds]
        )
        backing_distance[in_bounds] = slice_values[backing_indices]
        unknown = (~np.isfinite(backing_distance)) | np.isclose(
            backing_distance,
            float(slice_message.unknown_value),
            rtol=0.0,
            atol=1.0e-6,
        )
        known = in_bounds & ~unknown
        output = np.full(columns.shape, NO_INFORMATION, dtype=np.uint8)
        output[
            known & (backing_distance <= self.distance_threshold_m)
        ] = LETHAL_OBSTACLE
        output[known & (backing_distance > self.distance_threshold_m)] = FREE_SPACE
        transform_evidence = {
            "target_costmap_frame": message.header.frame_id,
            "source_slice_frame": slice_message.header.frame_id,
            "lookup_stamp_ns": _stamp_ns(slice_message.header.stamp),
            "translation_xy_m": [translation_x, translation_y],
            "yaw_rad": transform_yaw,
        }
        return (
            output.reshape((height, width)),
            backing_distance.reshape((height, width)),
            transform_evidence,
        )

    def _on_costmap(self, stream: str, message: Costmap) -> None:
        with self._lock:
            slice_message = self._latest_slice
            generation = self._generation
        if slice_message is None:
            # Nav2 starts publishing its all-unknown master costmaps before the
            # first depth-integrated Nvblox slice.  This is an expected reset
            # barrier, not a tracing failure and not Stage B evidence.
            with self._lock:
                self._records["skipped_before_first_slice"] += 1
                skipped = self._records["skipped_before_first_slice"]
                if skipped == 1 or skipped % 25 == 0:
                    self._write_summary()
            return
        width = int(message.metadata.size_x)
        height = int(message.metadata.size_y)
        final = np.asarray(message.data, dtype=np.uint8)
        if final.size != width * height:
            self._record_error(
                stream,
                f"costmap length {final.size} does not match {width * height}",
            )
            return
        final = final.reshape((height, width))
        try:
            pre_inflation, backing_distance, transform_evidence = self._pre_inflation(
                message,
                slice_message,
            )
            robot_x, robot_y, robot_yaw, robot_tf_basis = self._robot_pose(
                message.header.frame_id,
                message.header.stamp,
            )
        except TransformException as error:
            self._record_error(stream, f"TF lookup failed: {error}")
            return
        origin_x = float(message.metadata.origin.position.x)
        origin_y = float(message.metadata.origin.position.y)
        resolution = float(message.metadata.resolution)
        stage_b_spatial = self._spatial_evidence(
            pre_inflation,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
        )
        stage_c_spatial = self._spatial_evidence(
            final,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
        )
        stage_b_counts = _cost_counts(pre_inflation)
        stage_c_counts = _cost_counts(final)
        different = pre_inflation != final
        cellwise_comparison = {
            "equal_count": int(np.count_nonzero(~different)),
            "different_count": int(np.count_nonzero(different)),
            "equal_ratio": float(np.count_nonzero(~different) / final.size),
            "b_free_to_c_free_count": int(
                np.count_nonzero(
                    (pre_inflation == FREE_SPACE) & (final == FREE_SPACE)
                )
            ),
            "b_free_to_c_inflated_count": int(
                np.count_nonzero(
                    (pre_inflation == FREE_SPACE)
                    & (final > FREE_SPACE)
                    & (final < LETHAL_OBSTACLE)
                )
            ),
            "b_free_to_c_lethal_count": int(
                np.count_nonzero(
                    (pre_inflation == FREE_SPACE) & (final == LETHAL_OBSTACLE)
                )
            ),
            "b_free_to_c_unknown_count": int(
                np.count_nonzero(
                    (pre_inflation == FREE_SPACE) & (final == NO_INFORMATION)
                )
            ),
        }
        known_backing = backing_distance[np.isfinite(backing_distance)]
        slice_unknown = float(slice_message.unknown_value)
        known_backing = known_backing[
            ~np.isclose(known_backing, slice_unknown, rtol=0.0, atol=1.0e-6)
        ]
        stable_final_free = (
            stage_c_counts["free_count"] > 0
            and stage_c_spatial["connected_free_cell_count"] > 0
        )
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            if stable_final_free:
                self._consecutive_final_free[stream] += 1
            else:
                self._consecutive_final_free[stream] = 0
            self._maximum_consecutive_final_free[stream] = max(
                self._maximum_consecutive_final_free[stream],
                self._consecutive_final_free[stream],
            )
            consecutive = self._consecutive_final_free[stream]
        metadata = {
            "frame": message.header.frame_id,
            "header_stamp_ns": _stamp_ns(message.header.stamp),
            "map_load_time_ns": _stamp_ns(message.metadata.map_load_time),
            "update_time_ns": _stamp_ns(message.metadata.update_time),
            "layer": message.metadata.layer,
            "resolution_m": resolution,
            "width": width,
            "height": height,
            "origin_xyz_m": [
                origin_x,
                origin_y,
                float(message.metadata.origin.position.z),
            ],
            "origin_yaw_rad": _yaw(message.metadata.origin.orientation),
            "slice_stamp_ns": _stamp_ns(slice_message.header.stamp),
            "slice_age_at_costmap_header_sec": (
                _stamp_ns(message.header.stamp) - _stamp_ns(slice_message.header.stamp)
            )
            / 1.0e9,
            "slice_height_m": self.slice_height_m,
            "slice_min_height_m": self.slice_min_height_m,
            "slice_max_height_m": self.slice_max_height_m,
            "robot_tf_basis": robot_tf_basis,
            "transform": transform_evidence,
        }
        self._append(
            self.stage_path,
            {
                "event": "costmap_stage_pair",
                "sequence": sequence,
                "stream": stream,
                "generation": generation,
                "metadata": metadata,
                "conversion": {
                    "convert_to_binary_costmap": self.binary_conversion,
                    "occupied_if_distance_m_le": self.distance_threshold_m,
                    "free_if_distance_m_gt": self.distance_threshold_m,
                    "unknown_policy": "slice_unknown_or_out_of_bounds_to_255",
                },
                "backing_slice_distance_m": _percentiles(known_backing),
                "stage_b_to_c_cellwise": cellwise_comparison,
                "stage_b_pre_inflation": {
                    "provenance": "official_plugin_exact_reconstruction",
                    "counts": stage_b_counts,
                    "spatial": stage_b_spatial,
                },
                "stage_c_final_after_inflation": {
                    "provenance": "received_nav2_costmap_raw",
                    "counts": stage_c_counts,
                    "cost_percentiles_known": _percentiles(
                        final[final != NO_INFORMATION]
                    ),
                    "spatial": stage_c_spatial,
                    "consecutive_updates_with_connected_free": consecutive,
                },
            },
        )
        with self._lock:
            self._records[stream] += 1
            self._write_summary()

    def close(self) -> None:
        with self._lock:
            self._write_summary()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CostmapStageTracer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
