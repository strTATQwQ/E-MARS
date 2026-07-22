"""Episode-isolated Nvblox lifecycle with fail-safe motion inhibition."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import rclpy
from nav2_msgs.srv import ClearEntireCostmap
from nvblox_msgs.msg import DistanceMapSlice
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Int32


class NvbloxSupervisor(Node):
    """Own exactly one Nvblox process and replace it at every episode reset."""

    def __init__(self) -> None:
        super().__init__("internvla_t4_nvblox_supervisor")
        self.declare_parameter("result_dir", "")
        self.declare_parameter("nvblox_params_file", "")
        self.result_dir = Path(str(self.get_parameter("result_dir").value)).resolve()
        self.params_file = Path(
            str(self.get_parameter("nvblox_params_file").value)
        ).resolve()
        if not str(self.result_dir) or not self.params_file.is_file():
            raise RuntimeError("result_dir and nvblox_params_file are required")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.result_dir / "nvblox_resets.jsonl"
        self.slice_audit_path = self.result_dir / "nvblox_slice_classes.jsonl"
        self.log_path = self.result_dir / "nvblox.log"
        if (
            self.audit_path.exists()
            or self.slice_audit_path.exists()
            or self.log_path.exists()
        ):
            raise FileExistsError("refusing to append Nvblox supervisor evidence")
        self._log_stream = self.log_path.open("ab", buffering=0)
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._pending_generation: int | None = None
        self._worker: threading.Thread | None = None
        self._planned_stop = False
        self._fatal_child_exit = False
        self._start_index = 0
        self._last_generation = -1
        self._awaiting_slice_generation: int | None = None
        self._child_started_monotonic = 0.0
        self._child_started_ros_ns = 0
        self._pre_epoch_slice_reject_count = 0
        self._map_pollution_count = 0
        self._last_slice_audit_monotonic = 0.0
        self.motion_enabled_publisher = self.create_publisher(
            Bool, "/internvla/nav2_motion_enabled", 10
        )
        self.stop_publisher = self.create_publisher(Bool, "/internvla/stop", 10)
        self.map_ready_publisher = self.create_publisher(
            Int32, "/internvla_t4/map_ready_generation", 10
        )
        self.create_subscription(
            Int32,
            "/internvla_t4/map_reset_generation",
            self._on_reset,
            10,
        )
        self.create_subscription(
            DistanceMapSlice,
            "/nvblox_node/static_map_slice",
            self._on_map_slice,
            10,
        )
        self.clear_clients = [
            self.create_client(
                ClearEntireCostmap,
                "/local_costmap/clear_entirely_local_costmap",
            ),
            self.create_client(
                ClearEntireCostmap,
                "/global_costmap/clear_entirely_global_costmap",
            ),
        ]
        self._start_child(reason="supervisor_start")
        self.create_timer(0.10, self._tick)

    def _append(self, payload: dict[str, object]) -> None:
        payload = {"schema_version": 1, **payload, "wall_time_unix": time.time()}
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")

    def _command(self) -> list[str]:
        return [
            "ros2",
            "run",
            "nvblox_ros",
            "nvblox_node",
            "--ros-args",
            "--params-file",
            str(self.params_file),
            "-r",
            "camera_0/depth/image:=/go2/d435i/depth/image_rect",
            "-r",
            "camera_0/depth/camera_info:=/go2/d435i/depth/camera_info",
            "-r",
            "pointcloud:=/go2/lidar/points",
        ]

    def _start_child(self, *, reason: str) -> None:
        process = subprocess.Popen(
            self._command(),
            stdin=subprocess.DEVNULL,
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        with self._lock:
            self._process = process
            self._planned_stop = False
            self._start_index += 1
            self._child_started_monotonic = time.monotonic()
            self._child_started_ros_ns = self.get_clock().now().nanoseconds
            self._awaiting_slice_generation = self._last_generation
        self._append(
            {
                "event": "start",
                "reason": reason,
                "start_index": self._start_index,
                "pid": process.pid,
                "generation": self._last_generation,
            }
        )

    def _stop_child(self, *, reason: str) -> int | None:
        with self._lock:
            process = self._process
            self._planned_stop = True
        if process is None:
            return None
        started = time.monotonic()
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2.0)
        code = process.returncode
        with self._lock:
            self._process = None
        self._append(
            {
                "event": "stop",
                "reason": reason,
                "pid": process.pid,
                "exit_code": code,
                "duration_sec": time.monotonic() - started,
                "generation": self._last_generation,
            }
        )
        return code

    def _on_reset(self, message: Int32) -> None:
        generation = int(message.data)
        with self._lock:
            if generation <= self._last_generation:
                return
            self._pending_generation = generation

    def _on_map_slice(self, message: DistanceMapSlice) -> None:
        header = getattr(message, "header", None)
        stamp = getattr(header, "stamp", None)
        stamp_ns = (
            int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            if stamp is not None
            else 0
        )
        with self._lock:
            generation = self._awaiting_slice_generation
            active_generation = self._last_generation
            child_age = time.monotonic() - self._child_started_monotonic
            child_started_ros_ns = self._child_started_ros_ns
            audit_due = (
                active_generation >= 0
                and time.monotonic() - self._last_slice_audit_monotonic >= 0.5
            )
            if audit_due:
                self._last_slice_audit_monotonic = time.monotonic()
        if audit_due:
            unknown_value = float(message.unknown_value)
            data = [float(value) for value in message.data]
            unknown_count = sum(
                not math.isfinite(value)
                or math.isclose(value, unknown_value, rel_tol=0.0, abs_tol=1.0e-6)
                for value in data
            )
            known = [
                value
                for value in data
                if math.isfinite(value)
                and not math.isclose(
                    value, unknown_value, rel_tol=0.0, abs_tol=1.0e-6
                )
            ]
            positive = [value for value in known if value > 0.0]
            occupied = [value for value in known if value <= 0.0]
            record = {
                "schema_version": 1,
                "generation": active_generation,
                "slice_stamp_ns": stamp_ns or None,
                "width": int(message.width),
                "height": int(message.height),
                "resolution_m": float(message.resolution),
                "unknown_value": unknown_value,
                "unknown_count": unknown_count,
                "free_positive_count": len(positive),
                "occupied_nonpositive_count": len(occupied),
                "known_min_distance_m": min(known) if known else None,
                "known_max_distance_m": max(known) if known else None,
                "wall_time_unix": time.time(),
            }
            with self.slice_audit_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        with self._lock:
            if generation is None or generation < 0 or child_age < 0.10:
                return
            if stamp_ns and stamp_ns < child_started_ros_ns:
                self._pre_epoch_slice_reject_count += 1
                rejected_count = self._pre_epoch_slice_reject_count
                self._append(
                    {
                        "event": "pre_epoch_slice_rejected",
                        "generation": generation,
                        "slice_stamp_ns": stamp_ns,
                        "generation_epoch_ros_ns": child_started_ros_ns,
                        "reject_count": rejected_count,
                    }
                )
                return
            self._awaiting_slice_generation = None
        ready = Int32()
        ready.data = generation
        self.map_ready_publisher.publish(ready)
        self._append(
            {
                "event": "first_map_slice",
                "generation": generation,
                "child_age_sec": child_age,
                "map_ready_published": True,
                "slice_stamp_ns": stamp_ns or None,
                "generation_epoch_ros_ns": child_started_ros_ns,
                "map_pollution_count": self._map_pollution_count,
            }
        )

    def _reset_worker(self, generation: int) -> None:
        started = time.monotonic()
        disabled = Bool()
        disabled.data = False
        self.motion_enabled_publisher.publish(disabled)
        self._stop_child(reason=f"episode_reset_{generation}")
        self._last_generation = generation
        self._start_child(reason=f"episode_reset_{generation}")
        clear_requested = []
        for client in self.clear_clients:
            available = client.wait_for_service(timeout_sec=1.5)
            clear_requested.append(bool(available))
            if available:
                client.call_async(ClearEntireCostmap.Request())
        self._append(
            {
                "event": "reset_complete",
                "generation": generation,
                "duration_sec": time.monotonic() - started,
                "clear_costmap_requests": clear_requested,
                "new_pid": self._process.pid if self._process is not None else None,
            }
        )

    def _tick(self) -> None:
        with self._lock:
            process = self._process
            planned = self._planned_stop
            worker_running = self._worker is not None and self._worker.is_alive()
            pending = self._pending_generation
            if pending is not None and not worker_running:
                self._pending_generation = None
                self._worker = threading.Thread(
                    target=self._reset_worker,
                    args=(pending,),
                    daemon=True,
                )
                self._worker.start()
                worker_running = True
        if (
            process is not None
            and process.poll() is not None
            and not planned
            and not worker_running
            and not self._fatal_child_exit
        ):
            self._fatal_child_exit = True
            stop = Bool()
            stop.data = True
            disabled = Bool()
            disabled.data = False
            self.stop_publisher.publish(stop)
            self.motion_enabled_publisher.publish(disabled)
            self._append(
                {
                    "event": "fatal_unexpected_exit",
                    "pid": process.pid,
                    "exit_code": process.returncode,
                    "generation": self._last_generation,
                    "fail_safe_stop_published": True,
                }
            )

    def close(self) -> None:
        self._stop_child(reason="supervisor_shutdown")
        self._log_stream.close()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = NvbloxSupervisor()
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
