"""T5 completion_sim-only supervisor for sensor-fed Nvblox.

This node owns one real ``nvblox_node`` child.  Shadow mode records fused
sensor/slice evidence without navigation authority.  Active-local mode starts
the Nav2 layer disabled and enables it only after the pure sim-time readiness
gate passes; child exit, stale input, or reset returns to static+LiDAR.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import rclpy
from nav_msgs.msg import Odometry
from nav2_msgs.msg import Costmap
from nav2_msgs.srv import GetCostmap
from nvblox_msgs.msg import DistanceMapSlice
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Int32, String

from .t5_nvblox_runtime import NvbloxReadinessState, T5NvbloxContractError


def _stamp_ns(message: Any) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class T5NvbloxSupervisor(Node):
    def __init__(self) -> None:
        super().__init__("internvla_t5_nvblox_supervisor")
        self.declare_parameter("mode", "shadow")
        self.declare_parameter("result_dir", "")
        self.declare_parameter("nvblox_params_file", "")
        self.declare_parameter("minimum_consecutive_slices", 10)
        self.declare_parameter("stale_after_sim_sec", 2.5)
        self.declare_parameter(
            "local_costmap_node", "local_costmap/local_costmap"
        )
        self.declare_parameter("pose_source", "")

        if os.environ.get("INTERNNAV_RUNTIME_POLICY") != "completion_sim":
            raise RuntimeError("T5 Nvblox supervisor requires completion_sim")
        if os.environ.get("INTERNNAV_SIMULATION_TARGET") != "isaac":
            raise RuntimeError("T5 Nvblox supervisor rejects non-Isaac targets")
        mode = str(self.get_parameter("mode").value)
        result_value = str(self.get_parameter("result_dir").value)
        params_value = str(self.get_parameter("nvblox_params_file").value)
        result_dir = Path(result_value).resolve()
        params_file = Path(params_value).resolve()
        if mode not in {"shadow", "active_local_gt"}:
            raise RuntimeError("T5 Nvblox runtime mode must be shadow or active_local_gt")
        if not result_value or not params_file.is_file():
            raise RuntimeError("result_dir and nvblox_params_file are required")
        stale_sec = float(self.get_parameter("stale_after_sim_sec").value)
        minimum_slices = int(self.get_parameter("minimum_consecutive_slices").value)
        pose_source = str(self.get_parameter("pose_source").value)
        if not math.isfinite(stale_sec) or stale_sec <= 0.0:
            raise RuntimeError("stale_after_sim_sec must be positive")
        if pose_source != "isaac_ground_truth":
            raise RuntimeError("T5 Nvblox GT profile requires isaac_ground_truth pose")

        namespace = self.get_namespace().rstrip("/")
        if namespace not in {"/t5/lane_a", "/t5/lane_b"}:
            raise RuntimeError("T5 Nvblox supervisor requires an isolated Lane namespace")
        layer_leaf = str(self.get_parameter("local_costmap_node").value).strip("/")
        if layer_leaf != "local_costmap/local_costmap":
            raise RuntimeError(
                "local_costmap_node must remain local_costmap/local_costmap"
            )

        result_dir.mkdir(parents=True, exist_ok=False)
        self.result_dir = result_dir
        self.params_file = params_file
        self.mode = mode
        self.pose_source = pose_source
        self.namespace = namespace
        self.layer_node = namespace + "/" + layer_leaf
        self.events_path = result_dir / "nvblox_runtime.jsonl"
        self.slices_path = result_dir / "nvblox_slice_classes.jsonl"
        self.costmap_evidence_path = result_dir / "nvblox_costmap_evidence.jsonl"
        self.status_path = result_dir / "nvblox_runtime_status.json"
        self.ready_path = result_dir / "nvblox_ready.json"
        self.child_log = (result_dir / "nvblox_node.log").open("xb", buffering=0)
        self._lock = threading.RLock()
        self._layer_lock = threading.Lock()
        self._child: Optional[subprocess.Popen[bytes]] = None
        self._reset_worker: Optional[threading.Thread] = None
        self._pending_generation: Optional[int] = None
        self._closed = False
        self._last_status_wall = 0.0
        self._ready_written = False
        self._activation_attempted_for_slice_ns = 0
        self._last_degraded_reason: Optional[str] = None
        self._costmap_evidence_window_generation: Optional[int] = None
        self._last_credited_slice_stamp_ns = 0
        self._last_credited_costmap_update_ns = 0
        self._consecutive_costmap_layer_updates = 0
        self._maximum_consecutive_costmap_layer_updates = 0
        self._costmap_request_pending = False
        self._last_costmap_poll_sim_ns = 0
        self.state = NvbloxReadinessState(
            mode=mode,
            minimum_consecutive_slices=minimum_slices,
            stale_after_ns=int(stale_sec * 1_000_000_000),
        )
        self.health_publisher = self.create_publisher(String, "~/health", 10)
        self.create_subscription(
            Image,
            "/go2/d435i/depth/image_rect",
            lambda message: self._on_sensor("depth", message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            "/go2/d435i/depth/camera_info",
            lambda message: self._on_sensor("depth_camera_info", message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            "/go2/lidar/points",
            lambda message: self._on_sensor("lidar", message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            "/odom",
            lambda message: self._on_sensor("ground_truth_odometry", message),
            10,
        )
        self.create_subscription(
            DistanceMapSlice,
            namespace + "/nvblox_node/static_map_slice",
            self._on_slice,
            10,
        )
        self.create_subscription(
            Costmap,
            namespace + "/local_costmap/costmap_raw",
            self._on_local_costmap,
            10,
        )
        self._costmap_client = self.create_client(
            GetCostmap,
            namespace + "/local_costmap/get_costmap",
        )
        self.create_subscription(
            Int32,
            "/internvla_t4/map_reset_generation",
            self._on_reset,
            10,
        )
        self._start_child("supervisor_start")
        self.create_timer(0.1, self._tick)

    def _append(self, path: Path, payload: Mapping[str, Any]) -> None:
        row = dict(payload)
        row.update(
            {
                "schema_version": 1,
                "mode": self.mode,
                "wall_time_unix": time.time(),
            }
        )
        with self._lock:
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    def _event(self, event: str, **values: Any) -> None:
        self._append(self.events_path, {"event": event, **values})

    def _atomic_write(self, path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_name(
            ".{}.{}.tmp".format(path.name, os.getpid())
        )
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(str(temporary), str(path))

    def _now_sim_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _command(self) -> list[str]:
        return [
            "ros2",
            "run",
            "nvblox_ros",
            "nvblox_node",
            "--ros-args",
            "-r",
            "__ns:={}".format(self.namespace),
            "-r",
            "/tf:=tf",
            "-r",
            "/tf_static:=tf_static",
            "--params-file",
            str(self.params_file),
            "-r",
            "camera_0/depth/image:=/go2/d435i/depth/image_rect",
            "-r",
            "camera_0/depth/camera_info:=/go2/d435i/depth/camera_info",
            "-r",
            "pointcloud:=/go2/lidar/points",
        ]

    def _start_child(self, reason: str) -> None:
        child = subprocess.Popen(
            self._command(),
            stdin=subprocess.DEVNULL,
            stdout=self.child_log,
            stderr=subprocess.STDOUT,
            # The Lane runner is the one setsid leader. Keep the mapper in the
            # supervisor PGID so its bounded outer cleanup also catches the
            # child if this Python process itself becomes unresponsive.
            start_new_session=False,
            close_fds=True,
        )
        with self._lock:
            self._child = child
            self.state.child_restarted(self._now_sim_ns())
        self._event(
            "nvblox_child_start",
            reason=reason,
            pid=child.pid,
            process_epoch_sim_ns=self.state.process_epoch_sim_ns,
            command=self._command(),
        )

    def _stop_child(self, reason: str) -> None:
        with self._lock:
            child = self._child
        if child is None:
            return
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2.0)
        self._event(
            "nvblox_child_stop", reason=reason, pid=child.pid, exit_code=child.returncode
        )
        with self._lock:
            self._child = None

    def _invoke_layer_parameter(self, enabled: bool) -> tuple[bool, str]:
        command = [
            "ros2",
            "param",
            "set",
            self.layer_node,
            "nvblox_layer.enabled",
            "true" if enabled else "false",
        ]
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3.0,
                check=False,
            )
            return (
                completed.returncode == 0
                and "successful" in completed.stdout.lower(),
                completed.stdout[-2000:],
            )
        except subprocess.TimeoutExpired as exc:
            return False, "timeout: {}".format(exc)

    def _reset_pending(self) -> bool:
        return self._pending_generation is not None or (
            self._reset_worker is not None and self._reset_worker.is_alive()
        )

    def _set_layer(self, enabled: bool, reason: str) -> bool:
        if self.mode != "active_local_gt":
            return not enabled
        with self._layer_lock:
            if enabled:
                with self._lock:
                    reset_pending = self._reset_pending()
                    still_ready = self.state.ready(self._now_sim_ns())
                if reset_pending or not still_ready:
                    self._event(
                        "nvblox_layer_set_skipped",
                        enabled=True,
                        reason="reset_pending_or_readiness_changed",
                    )
                    return False
            success, output = self._invoke_layer_parameter(enabled)
            self._event(
                "nvblox_layer_set",
                enabled=enabled,
                reason=reason,
                success=success,
                output=output,
            )
            if success:
                started_evidence_generation: Optional[int] = None
                with self._lock:
                    reset_arrived_during_enable = enabled and self._reset_pending()
                    if enabled and not reset_arrived_during_enable:
                        self.state.mark_layer_enabled()
                        if (
                            self._costmap_evidence_window_generation
                            != self.state.generation
                        ):
                            self._costmap_evidence_window_generation = (
                                self.state.generation
                            )
                            self._last_credited_slice_stamp_ns = 0
                            self._last_credited_costmap_update_ns = 0
                            self._consecutive_costmap_layer_updates = 0
                            self._last_costmap_poll_sim_ns = 0
                            started_evidence_generation = self.state.generation
                    elif not enabled:
                        self.state.mark_layer_disabled(reason)
                        self._consecutive_costmap_layer_updates = 0
                if started_evidence_generation is not None:
                    self._append(
                        self.costmap_evidence_path,
                        {
                            "event": "ready_window_start",
                            "generation": started_evidence_generation,
                            "sim_time_ns": self._now_sim_ns(),
                        },
                    )
                if reset_arrived_during_enable:
                    rollback_success, rollback_output = self._invoke_layer_parameter(False)
                    self._event(
                        "nvblox_layer_set",
                        enabled=False,
                        reason="reset_arrived_during_enable",
                        success=rollback_success,
                        output=rollback_output,
                    )
                    if rollback_success:
                        with self._lock:
                            self.state.mark_layer_disabled(
                                "reset_arrived_during_enable"
                            )
                    return False
            return success

    def _on_sensor(self, name: str, message: Any) -> None:
        with self._lock:
            self.state.observe_sensor(name, _stamp_ns(message))

    def _on_slice(self, message: DistanceMapSlice) -> None:
        stamp_ns = _stamp_ns(message)
        unknown_value = float(message.unknown_value)
        values = [float(value) for value in message.data]
        dimensions_valid = (
            int(message.width) > 0
            and int(message.height) > 0
            and len(values) == int(message.width) * int(message.height)
        )
        if dimensions_valid:
            unknown_count = sum(
                not math.isfinite(value)
                or math.isclose(value, unknown_value, rel_tol=0.0, abs_tol=1.0e-6)
                for value in values
            )
            known = [
                value
                for value in values
                if math.isfinite(value)
                and not math.isclose(
                    value, unknown_value, rel_tol=0.0, abs_tol=1.0e-6
                )
            ]
            free_count = sum(value > 0.0 for value in known)
            occupied_count = sum(value <= 0.0 for value in known)
        else:
            unknown_count = free_count = occupied_count = -1
        with self._lock:
            action = self.state.observe_slice(
                stamp_ns,
                self._now_sim_ns(),
                unknown_count,
                free_count,
                occupied_count,
            )
            snapshot = self.state.snapshot(self._now_sim_ns(), self._child_alive())
        self._append(
            self.slices_path,
            {
                "slice_stamp_ns": stamp_ns or None,
                "width": int(message.width),
                "height": int(message.height),
                "dimensions_valid": dimensions_valid,
                "unknown_count": unknown_count,
                "free_positive_count": free_count,
                "occupied_nonpositive_count": occupied_count,
                "action": action,
                "generation": snapshot["generation"],
                "consecutive_valid_slices": snapshot["consecutive_valid_slices"],
            },
        )
        if action == "ENABLE_LAYER" and stamp_ns != self._activation_attempted_for_slice_ns:
            self._activation_attempted_for_slice_ns = stamp_ns
            self._set_layer(True, "fresh_fused_inputs_and_ten_valid_slices")
        self._publish_ready_if_needed()

    def _on_local_costmap(self, message: Costmap) -> None:
        self._record_local_costmap(message, "costmap_raw")

    def _record_local_costmap(self, message: Costmap, source: str) -> None:
        """Count fresh, distinct costmap states backed by a new Nvblox slice."""

        if self.mode != "active_local_gt":
            return
        now_ns = self._now_sim_ns()
        update_time = message.metadata.update_time
        costmap_update_ns = (
            int(update_time.sec) * 1_000_000_000 + int(update_time.nanosec)
        )
        costmap_stamp_ns = costmap_update_ns or _stamp_ns(message)
        with self._lock:
            snapshot = self.state.snapshot(now_ns, self._child_alive())
            generation = int(snapshot["generation"])
            slice_stamp_ns = int(snapshot["last_slice_stamp_ns"] or 0)
            if self._costmap_evidence_window_generation != generation:
                return
            if not snapshot["ready"] or not snapshot["layer_enabled"]:
                self._consecutive_costmap_layer_updates = 0
                return
            if slice_stamp_ns <= self._last_credited_slice_stamp_ns:
                return
            if costmap_stamp_ns <= self._last_credited_costmap_update_ns:
                return
            dimensions_valid = (
                int(message.metadata.size_x) > 0
                and int(message.metadata.size_y) > 0
                and len(message.data)
                == int(message.metadata.size_x) * int(message.metadata.size_y)
            )
            slice_age_ns = costmap_stamp_ns - slice_stamp_ns
            if (
                not dimensions_valid
                or costmap_stamp_ns <= 0
                or slice_age_ns < 0
                or slice_age_ns > self.state.stale_after_ns
            ):
                return
            self._last_credited_slice_stamp_ns = slice_stamp_ns
            self._last_credited_costmap_update_ns = costmap_stamp_ns
            self._consecutive_costmap_layer_updates += 1
            self._maximum_consecutive_costmap_layer_updates = max(
                self._maximum_consecutive_costmap_layer_updates,
                self._consecutive_costmap_layer_updates,
            )
            consecutive = self._consecutive_costmap_layer_updates
            maximum = self._maximum_consecutive_costmap_layer_updates
        self._append(
            self.costmap_evidence_path,
            {
                "event": "nvblox_backed_costmap_update",
                "source": source,
                "generation": generation,
                "sim_time_ns": now_ns,
                "costmap_stamp_ns": costmap_stamp_ns,
                "slice_stamp_ns": slice_stamp_ns,
                "slice_age_sec": slice_age_ns / 1_000_000_000.0,
                "costmap_layer": str(message.metadata.layer),
                "width": int(message.metadata.size_x),
                "height": int(message.metadata.size_y),
                "consecutive_updates": consecutive,
                "maximum_consecutive_updates": maximum,
            },
        )

    def _on_costmap_service(self, future: Any) -> None:
        with self._lock:
            self._costmap_request_pending = False
        try:
            response = future.result()
            if response is None:
                raise RuntimeError("GetCostmap returned no response")
            self._record_local_costmap(response.map, "get_costmap_service")
        except BaseException as exc:
            self._append(
                self.costmap_evidence_path,
                {
                    "event": "costmap_service_error",
                    "sim_time_ns": self._now_sim_ns(),
                    "error": repr(exc)[:1024],
                },
            )

    def _poll_local_costmap(self) -> None:
        if self.mode != "active_local_gt" or not self._costmap_client.service_is_ready():
            return
        now_ns = self._now_sim_ns()
        with self._lock:
            if self._costmap_evidence_window_generation != self.state.generation:
                return
            if self._costmap_request_pending:
                return
            if now_ns - self._last_costmap_poll_sim_ns < 500_000_000:
                return
            self._last_costmap_poll_sim_ns = now_ns
            self._costmap_request_pending = True
        future = self._costmap_client.call_async(GetCostmap.Request())
        future.add_done_callback(self._on_costmap_service)

    def _on_reset(self, message: Int32) -> None:
        generation = int(message.data)
        ended_generation: Optional[int] = None
        with self._lock:
            if generation <= self.state.generation:
                return
            ended_generation = self._costmap_evidence_window_generation
            self._costmap_evidence_window_generation = None
            self._last_credited_slice_stamp_ns = 0
            self._last_credited_costmap_update_ns = 0
            self._consecutive_costmap_layer_updates = 0
            self._last_costmap_poll_sim_ns = 0
            self._pending_generation = generation
        if ended_generation is not None:
            self._append(
                self.costmap_evidence_path,
                {
                    "event": "ready_window_end",
                    "generation": ended_generation,
                    "next_generation": generation,
                    "sim_time_ns": self._now_sim_ns(),
                    "reason": "episode_reset",
                },
            )

    def _reset(self, generation: int) -> None:
        if self._closed:
            return
        if self.mode == "active_local_gt" and self.state.layer_enabled:
            # Do not restart the mapper until Nav2 has acknowledged fallback.
            while not self._set_layer(False, "episode_reset"):
                if self._closed:
                    return
                time.sleep(0.2)
        self._stop_child("episode_reset_{}".format(generation))
        if self._closed:
            return
        with self._lock:
            self.state.reset(generation, self._now_sim_ns())
            self._ready_written = False
            self._activation_attempted_for_slice_ns = 0
        try:
            self.ready_path.unlink()
        except FileNotFoundError:
            pass
        self._start_child("episode_reset_{}".format(generation))
        self._event("episode_reset_complete", generation=generation)

    def _child_alive(self) -> bool:
        child = self._child
        return child is not None and child.poll() is None

    def _publish_ready_if_needed(self) -> None:
        now_ns = self._now_sim_ns()
        child_alive = self._child_alive()
        with self._lock:
            snapshot = dict(self.state.snapshot(now_ns, child_alive))
        if not snapshot["ready"]:
            return
        if self.mode == "active_local_gt" and not snapshot["layer_enabled"]:
            return
        status = "SHADOW_SENSOR_FED_READY" if self.mode == "shadow" else "ACTIVE_LOCAL_READY"
        payload = {
            "schema_version": 1,
            "status": status,
            "runtime_policy": "completion_sim",
            "target": "isaac_simulation_only",
            "namespace": self.namespace,
            "pose_source": self.pose_source,
            "real_nvblox_node": True,
            "depth_and_lidar_required": True,
            "navigation_authority": self.mode == "active_local_gt",
            "state": snapshot,
            "wall_time_unix": time.time(),
        }
        if not self._ready_written:
            self._atomic_write(self.ready_path, payload)
            self._ready_written = True
            self._event("runtime_ready", readiness_status=status)

    def _tick(self) -> None:
        with self._lock:
            pending = self._pending_generation
            worker_alive = self._reset_worker is not None and self._reset_worker.is_alive()
            if pending is not None and not worker_alive:
                self._pending_generation = None
                self._reset_worker = threading.Thread(
                    target=self._reset, args=(pending,), daemon=True
                )
                self._reset_worker.start()
                worker_alive = True
            child_alive = self._child_alive()
            action = None if worker_alive else self.state.tick(self._now_sim_ns(), child_alive)
            fallback_reason = self.state.fallback_reason
        if action == "DISABLE_LAYER":
            self._ready_written = False
            try:
                self.ready_path.unlink()
            except FileNotFoundError:
                pass
            self._set_layer(False, fallback_reason or "runtime_degraded")
        elif action == "DEGRADED" and fallback_reason != self._last_degraded_reason:
            self._ready_written = False
            try:
                self.ready_path.unlink()
            except FileNotFoundError:
                pass
            self._event("runtime_degraded", reason=fallback_reason)
            self._last_degraded_reason = fallback_reason
        self._publish_ready_if_needed()
        self._poll_local_costmap()
        if time.monotonic() - self._last_status_wall >= 0.5:
            self._last_status_wall = time.monotonic()
            with self._lock:
                snapshot = dict(
                    self.state.snapshot(self._now_sim_ns(), self._child_alive())
                )
            runtime_ready = snapshot["ready"] and (
                self.mode == "shadow" or snapshot["layer_enabled"]
            )
            with self._lock:
                evidence_window_generation = (
                    self._costmap_evidence_window_generation
                )
                maximum_costmap_updates = (
                    self._maximum_consecutive_costmap_layer_updates
                )
            if (
                self.mode == "active_local_gt"
                and evidence_window_generation == snapshot["generation"]
            ):
                self._append(
                    self.costmap_evidence_path,
                    {
                        "event": "ready_window_sample",
                        "generation": snapshot["generation"],
                        "sim_time_ns": self._now_sim_ns(),
                        "ready": runtime_ready,
                        "layer_enabled": snapshot["layer_enabled"],
                        "maximum_consecutive_updates": maximum_costmap_updates,
                    },
                )
            payload = {
                "schema_version": 1,
                "status": "READY" if runtime_ready else "WAITING_OR_FALLBACK",
                "runtime_policy": "completion_sim",
                "target": "isaac_simulation_only",
                "namespace": self.namespace,
                "pose_source": self.pose_source,
                "state": snapshot,
                "wall_time_unix": time.time(),
            }
            self._atomic_write(self.status_path, payload)
            message = String()
            message.data = json.dumps(payload, sort_keys=True, allow_nan=False)
            self.health_publisher.publish(message)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.mode == "active_local_gt" and self.state.layer_enabled:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if self._set_layer(False, "supervisor_shutdown"):
                    break
                time.sleep(0.2)
        worker = self._reset_worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=8.0)
        self._stop_child("supervisor_shutdown")
        self.child_log.close()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[T5NvbloxSupervisor] = None
    try:
        node = T5NvbloxSupervisor()
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
    except (RuntimeError, T5NvbloxContractError):
        if node is not None:
            node.close()
        raise
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
