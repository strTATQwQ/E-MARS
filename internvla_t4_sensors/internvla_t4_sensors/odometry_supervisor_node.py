"""Episode-isolated cuVSLAM lifecycle, freshness audit, and fail-safe stop."""

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
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Int32


class OdometrySupervisor(Node):
    def __init__(self) -> None:
        super().__init__("internvla_t4_odometry_supervisor")
        self.declare_parameter("result_dir", "")
        self.declare_parameter("odometry_timeout_sec", 0.30)
        self.declare_parameter("shadow_only", False)
        self.declare_parameter("launch_use_sim_time", False)
        raw_result_dir = str(self.get_parameter("result_dir").value)
        if not raw_result_dir:
            raise RuntimeError("result_dir is required")
        self.result_dir = Path(raw_result_dir).resolve()
        self.timeout = float(self.get_parameter("odometry_timeout_sec").value)
        self.shadow_only = bool(self.get_parameter("shadow_only").value)
        self.launch_use_sim_time = bool(
            self.get_parameter("launch_use_sim_time").value
        )
        if not 0.10 <= self.timeout <= 1.0:
            raise RuntimeError("odometry_timeout_sec is outside the safety envelope")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.result_dir / "odometry_supervisor_records.jsonl"
        self.log_path = self.result_dir / "cuvslam.log"
        self.shadow_sample_path = self.result_dir / "cuvslam_shadow_samples.jsonl"
        if (
            self.audit_path.exists()
            or self.log_path.exists()
            or (self.shadow_only and self.shadow_sample_path.exists())
        ):
            raise FileExistsError("refusing to append odometry evidence")
        self._log = self.log_path.open("ab", buffering=0)
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._worker: threading.Thread | None = None
        self._pending_generation: int | None = None
        self._generation = -1
        self._planned_stop = False
        self._fatal = False
        self._last_odom_monotonic = 0.0
        self._last_odom_sim_sec = 0.0
        self._first_odom_monotonic = 0.0
        self._odom_count = 0
        self._stale_latched = False
        self._motion_enabled = False
        self._start_index = 0
        self._latest_truth: Odometry | None = None
        self._truth_anchor: tuple[float, float, float] | None = None
        self._estimate_anchor: tuple[float, float, float] | None = None
        self.stop_publisher = self.create_publisher(Bool, "/internvla/stop", 10)
        self.motion_publisher = self.create_publisher(
            Bool, "/internvla/nav2_motion_enabled", 10
        )
        self.create_subscription(
            Int32, "/internvla_t4/map_reset_generation", self._on_reset, 10
        )
        self.create_subscription(
            Odometry, "visual_slam/tracking/odometry", self._on_odom, 20
        )
        if self.shadow_only:
            self.create_subscription(Odometry, "/odom", self._on_truth, 20)
        self.create_subscription(
            Bool, "/internvla/nav2_motion_enabled", self._on_motion, 20
        )
        self._start_child("supervisor_start")
        self.create_timer(0.05, self._tick)

    def _append(self, payload: dict[str, object]) -> None:
        record = {
            "schema_version": 1,
            "reset_generation": self._generation,
            "sim_time_ns": self.get_clock().now().nanoseconds,
            **payload,
            "wall_time_unix": time.time(),
        }
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def _command(self) -> list[str]:
        command = [
            "ros2",
            "launch",
            "internvla_t4_sensors",
            "t4_cuvslam.launch.py",
        ]
        if self.launch_use_sim_time:
            command.append("use_sim_time:=true")
        namespace = self.get_namespace()
        if namespace not in {"", "/"}:
            command.append(f"namespace:={namespace}")
        return command

    def _start_child(self, reason: str) -> None:
        child_environment = None
        if self.shadow_only:
            # The T5 supervisor itself is namespaced, while the frozen
            # cuVSLAM topics are explicitly rooted.  Do not let the inherited
            # lane namespace silently move the child output topic.
            child_environment = os.environ.copy()
            child_environment.pop("ROS_NAMESPACE", None)
        process = subprocess.Popen(
            self._command(),
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=child_environment,
        )
        with self._lock:
            self._process = process
            self._planned_stop = False
            self._start_index += 1
            self._last_odom_monotonic = 0.0
            self._last_odom_sim_sec = 0.0
            self._first_odom_monotonic = 0.0
            self._odom_count = 0
            self._stale_latched = False
            self._latest_truth = None
            self._truth_anchor = None
            self._estimate_anchor = None
        self._append(
            {
                "event": "start",
                "reason": reason,
                "pid": process.pid,
                "start_index": self._start_index,
            }
        )

    def _stop_child(self, reason: str) -> None:
        with self._lock:
            process = self._process
            self._planned_stop = True
        if process is None:
            return
        if process.poll() is None:
            for sig, wait_sec in ((signal.SIGINT, 5.0), (signal.SIGTERM, 3.0)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=wait_sec)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2.0)
        self._append(
            {
                "event": "stop",
                "reason": reason,
                "pid": process.pid,
                "exit_code": process.returncode,
                "odom_count": self._odom_count,
            }
        )
        with self._lock:
            self._process = None

    def _on_reset(self, message: Int32) -> None:
        generation = int(message.data)
        with self._lock:
            if generation <= self._generation:
                return
            self._pending_generation = generation

    def _on_odom(self, message: Odometry) -> None:
        now = time.monotonic()
        stamp_sec = float(message.header.stamp.sec) + float(
            message.header.stamp.nanosec
        ) / 1e9
        ros_now = self.get_clock().now().nanoseconds / 1e9
        with self._lock:
            if self._first_odom_monotonic == 0.0:
                self._first_odom_monotonic = now
                self._append(
                    {
                        "event": "tracking_acquired",
                        "pid": self._process.pid if self._process else None,
                        "tf_age_sec": max(0.0, ros_now - stamp_sec),
                    }
                )
            self._last_odom_monotonic = now
            self._last_odom_sim_sec = stamp_sec
            self._odom_count += 1
            if self._stale_latched:
                self._append({"event": "tracking_recovered", "odom_count": self._odom_count})
                self._stale_latched = False
            if self.shadow_only:
                self._record_shadow_sample(message, ros_now)

    def _on_truth(self, message: Odometry) -> None:
        with self._lock:
            self._latest_truth = message

    @staticmethod
    def _yaw(message: Odometry) -> float:
        orientation = message.pose.pose.orientation
        return math.atan2(
            2.0
            * (
                float(orientation.w) * float(orientation.z)
                + float(orientation.x) * float(orientation.y)
            ),
            1.0
            - 2.0
            * (
                float(orientation.y) * float(orientation.y)
                + float(orientation.z) * float(orientation.z)
            ),
        )

    @staticmethod
    def _stamp_ns(message: Odometry) -> int:
        return int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )

    def _record_shadow_sample(self, estimate: Odometry, ros_now: float) -> None:
        truth = self._latest_truth
        if truth is None or self._generation < 0:
            return
        truth_stamp_ns = self._stamp_ns(truth)
        estimate_stamp_ns = self._stamp_ns(estimate)
        if truth_stamp_ns <= 0 or estimate_stamp_ns <= 0:
            return
        truth_pose = truth.pose.pose.position
        estimate_pose = estimate.pose.pose.position
        truth_yaw = self._yaw(truth)
        estimate_yaw = self._yaw(estimate)
        if self._truth_anchor is None or self._estimate_anchor is None:
            self._truth_anchor = (
                float(truth_pose.x),
                float(truth_pose.y),
                truth_yaw,
            )
            self._estimate_anchor = (
                float(estimate_pose.x),
                float(estimate_pose.y),
                estimate_yaw,
            )
        truth_anchor = self._truth_anchor
        estimate_anchor = self._estimate_anchor
        relative_x = float(estimate_pose.x) - estimate_anchor[0]
        relative_y = float(estimate_pose.y) - estimate_anchor[1]
        alignment_yaw = truth_anchor[2] - estimate_anchor[2]
        cosine, sine = math.cos(alignment_yaw), math.sin(alignment_yaw)
        aligned_x = truth_anchor[0] + cosine * relative_x - sine * relative_y
        aligned_y = truth_anchor[1] + sine * relative_x + cosine * relative_y
        aligned_yaw = truth_anchor[2] + math.atan2(
            math.sin(estimate_yaw - estimate_anchor[2]),
            math.cos(estimate_yaw - estimate_anchor[2]),
        )
        yaw_error = abs(
            math.atan2(
                math.sin(aligned_yaw - truth_yaw),
                math.cos(aligned_yaw - truth_yaw),
            )
        )
        record = {
            "schema_version": 1,
            "reset_generation": self._generation,
            "source_start_index": self._start_index,
            "truth_stamp_ns": truth_stamp_ns,
            "estimate_stamp_ns": estimate_stamp_ns,
            "pair_skew_ms": abs(truth_stamp_ns - estimate_stamp_ns) / 1e6,
            "truth_xy_yaw": [
                float(truth_pose.x),
                float(truth_pose.y),
                truth_yaw,
            ],
            "estimated_xy_yaw": [aligned_x, aligned_y, aligned_yaw],
            "xy_error_m": math.hypot(
                aligned_x - float(truth_pose.x),
                aligned_y - float(truth_pose.y),
            ),
            "yaw_error_rad": yaw_error,
            "tf_age_sec": max(0.0, ros_now - estimate_stamp_ns / 1e9),
            "truth_usage": "shadow_scoring_only_gt_remains_navigation_authority",
            "wall_time_unix": time.time(),
        }
        with self.shadow_sample_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def _on_motion(self, message: Bool) -> None:
        with self._lock:
            self._motion_enabled = bool(message.data)

    def _restart(self, generation: int) -> None:
        disabled = Bool()
        disabled.data = False
        if not self.shadow_only:
            self.motion_publisher.publish(disabled)
        with self._lock:
            self._motion_enabled = False
        self._stop_child(f"episode_reset_{generation}")
        self._generation = generation
        self._start_child(f"episode_reset_{generation}")

    def _publish_fail_safe(self, event: str, details: dict[str, object]) -> None:
        stop = Bool()
        stop.data = True
        disabled = Bool()
        disabled.data = False
        if not self.shadow_only:
            self.stop_publisher.publish(stop)
            self.motion_publisher.publish(disabled)
        self._append(
            {
                "event": event,
                "fail_safe_stop_published": not self.shadow_only,
                "shadow_only": self.shadow_only,
                **details,
            }
        )

    def _tick(self) -> None:
        with self._lock:
            worker_running = self._worker is not None and self._worker.is_alive()
            if self._pending_generation is not None and not worker_running:
                generation = self._pending_generation
                self._pending_generation = None
                self._worker = threading.Thread(
                    target=self._restart, args=(generation,), daemon=True
                )
                self._worker.start()
                worker_running = True
            process = self._process
            planned = self._planned_stop
            last_odom = self._last_odom_monotonic
            last_odom_sim = self._last_odom_sim_sec
            stale = self._stale_latched
            motion_enabled = self._motion_enabled
        if (
            process is not None
            and process.poll() is not None
            and not planned
            and not worker_running
            and not self._fatal
        ):
            self._fatal = True
            self._publish_fail_safe(
                "fatal_unexpected_exit",
                {"pid": process.pid, "exit_code": process.returncode},
            )
            return
        odometry_age = (
            self.get_clock().now().nanoseconds / 1e9 - last_odom_sim
            if self.shadow_only and last_odom_sim
            else time.monotonic() - last_odom
        )
        if motion_enabled and last_odom and odometry_age > self.timeout and not stale:
            with self._lock:
                self._stale_latched = True
            self._publish_fail_safe(
                "tracking_lost",
                {
                    "odometry_age_sec": odometry_age,
                    "freshness_clock": "sim" if self.shadow_only else "wall",
                },
            )

    def close(self) -> None:
        self._stop_child("supervisor_shutdown")
        self._log.close()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OdometrySupervisor()
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
