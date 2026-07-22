"""ROS 2 telemetry sidecar for vla-nav-panel.

The process deliberately runs outside Uvicorn/Step3.  It creates subscriptions
only, projects a small allowlisted state, and atomically publishes JSON/JPEG
files for :mod:`slow_planner_frontend.state` to consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


ROS_CAMERA_VIEWS = ("go2_front", "d435_color", "d435_depth")
USB_CAMERA_VIEWS = ("front_left", "front", "front_right", "rear")
CAMERA_VIEWS = (*USB_CAMERA_VIEWS, *ROS_CAMERA_VIEWS)
TOPIC_TYPES = {
    "go2_front": "unitree_go/msg/Go2FrontVideoData",
    "low_state": "unitree_go/msg/LowState",
    "sport_mode_state": "unitree_go/msg/SportModeState",
    "lidar_state": "unitree_go/msg/LidarState",
    "lidar_imu": "sensor_msgs/msg/Imu",
    "lidar_cloud": "sensor_msgs/msg/PointCloud2",
    "odometry": "nav_msgs/msg/Odometry",
    "d435_color": "sensor_msgs/msg/Image",
    "d435_depth": "sensor_msgs/msg/Image",
}


def _expand_path(value: Any) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value or "")))
    if not expanded or "$" in expanded or "%" in expanded:
        raise ValueError(f"unresolved ROS adapter path: {value!r}")
    return Path(expanded).resolve()


@dataclass(frozen=True)
class RosAdapterConfig:
    node_name: str
    preview_hz: float
    state_publish_hz: float
    stale_after_s: float
    jpeg_quality: int
    go2_h264_decoder: str
    go2_camera_source: str
    usb_camera_enabled: bool
    usb_camera_capture_hz: float
    usb_camera_width: int
    usb_camera_height: int
    usb_camera_input_format: str
    usb_camera_horizontal_flip: bool
    usb_camera_devices: dict[str, str]
    state_path: Path
    camera_manifest_path: Path
    camera_dir: Path
    topics: dict[str, str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RosAdapterConfig":
        frontend = value.get("frontend", value)
        if not isinstance(frontend, Mapping):
            raise ValueError("frontend config must be an object")
        raw = frontend.get("ros_adapter") or {}
        if not isinstance(raw, Mapping) or not bool(raw.get("enabled", False)):
            raise ValueError("frontend.ros_adapter must be enabled")
        paths = raw.get("paths") or {}
        topics = raw.get("topics") or {}
        if not isinstance(paths, Mapping) or not isinstance(topics, Mapping):
            raise ValueError("ROS adapter paths and topics must be objects")
        missing_topics = [key for key in TOPIC_TYPES if not str(topics.get(key, ""))]
        if missing_topics:
            raise ValueError(f"ROS adapter topics missing: {missing_topics}")
        resolved_topics = {key: str(topics[key]) for key in TOPIC_TYPES}
        if any(not topic.startswith("/") for topic in resolved_topics.values()):
            raise ValueError("ROS adapter topic names must be absolute")
        preview_hz = float(raw.get("preview_hz", 1.0))
        state_publish_hz = float(raw.get("state_publish_hz", 2.0))
        stale_after_s = float(raw.get("stale_after_s", 3.0))
        jpeg_quality = int(raw.get("jpeg_quality", 80))
        go2_camera_source = str(
            raw.get("go2_camera_source") or "videohub_rpc"
        ).strip()
        if go2_camera_source not in {"videohub_rpc", "h264_topic"}:
            raise ValueError(
                "go2_camera_source must be videohub_rpc or h264_topic"
            )
        usb_raw = raw.get("usb_cameras") or {}
        if not isinstance(usb_raw, Mapping):
            raise ValueError("frontend.ros_adapter.usb_cameras must be an object")
        usb_enabled = bool(usb_raw.get("enabled", False))
        usb_devices_raw = usb_raw.get("views") or {}
        if not isinstance(usb_devices_raw, Mapping):
            raise ValueError("usb_cameras.views must be an object")
        usb_devices = {
            view_id: str(usb_devices_raw.get(view_id) or "").strip()
            for view_id in USB_CAMERA_VIEWS
        }
        if usb_enabled:
            missing_usb = [key for key, path in usb_devices.items() if not path]
            if missing_usb:
                raise ValueError(f"usb_cameras.views missing: {missing_usb}")
            if any(not path.startswith("/dev/") for path in usb_devices.values()):
                raise ValueError("USB camera device paths must be under /dev")
        usb_capture_hz = float(usb_raw.get("capture_hz", 5.0))
        usb_width = int(usb_raw.get("width", 640))
        usb_height = int(usb_raw.get("height", 480))
        usb_input_format = str(usb_raw.get("input_format") or "mjpeg").strip()
        usb_horizontal_flip = bool(usb_raw.get("horizontal_flip", False))
        if not 0.1 <= preview_hz <= 5.0:
            raise ValueError("preview_hz must be in [0.1, 5.0]")
        if not 0.2 <= state_publish_hz <= 10.0:
            raise ValueError("state_publish_hz must be in [0.2, 10.0]")
        if not 0.5 <= stale_after_s <= 30.0:
            raise ValueError("stale_after_s must be in [0.5, 30.0]")
        if not 40 <= jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be in [40, 95]")
        if not 1.0 <= usb_capture_hz <= 15.0:
            raise ValueError("usb_cameras.capture_hz must be in [1, 15]")
        if not 160 <= usb_width <= 1920 or not 120 <= usb_height <= 1080:
            raise ValueError("USB camera dimensions are out of bounds")
        if usb_input_format not in {"mjpeg", "yuyv422"}:
            raise ValueError("usb_cameras.input_format must be mjpeg or yuyv422")
        return cls(
            node_name=str(raw.get("node_name") or "vla_nav_panel_ros_adapter"),
            preview_hz=preview_hz,
            state_publish_hz=state_publish_hz,
            stale_after_s=stale_after_s,
            jpeg_quality=jpeg_quality,
            go2_h264_decoder=str(raw.get("go2_h264_decoder") or "ffmpeg"),
            go2_camera_source=go2_camera_source,
            usb_camera_enabled=usb_enabled,
            usb_camera_capture_hz=usb_capture_hz,
            usb_camera_width=usb_width,
            usb_camera_height=usb_height,
            usb_camera_input_format=usb_input_format,
            usb_camera_horizontal_flip=usb_horizontal_flip,
            usb_camera_devices=usb_devices,
            state_path=_expand_path(paths.get("state_path")),
            camera_manifest_path=_expand_path(paths.get("camera_manifest_path")),
            camera_dir=_expand_path(paths.get("camera_dir")),
            topics=resolved_topics,
        )


def load_config(path: Path) -> RosAdapterConfig:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("frontend config must be an object")
    return RosAdapterConfig.from_mapping(value)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    atomic_write_bytes(path, payload)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _vector(value: Any, limit: int) -> list[float]:
    try:
        return [_finite(item) for item in list(value)[:limit]]
    except TypeError:
        return []


def _message_stamp_s(message: Any) -> float:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        stamp = getattr(message, "stamp", None)
    if stamp is not None and hasattr(stamp, "sec"):
        return max(0.0, _finite(stamp.sec) + _finite(stamp.nanosec) / 1e9)
    raw = _finite(getattr(message, "time_frame", 0))
    if raw > 1e17:
        return raw / 1e9
    if raw > 1e14:
        return raw / 1e6
    if raw > 1e11:
        return raw / 1e3
    return max(0.0, raw)


def project_low_state(message: Any) -> dict[str, Any]:
    bms = getattr(message, "bms_state", None)
    imu = getattr(message, "imu_state", None)
    motors = list(getattr(message, "motor_state", ()) or ())[:20]
    return {
        "battery": {
            "soc": int(getattr(bms, "soc", 0)),
            "status": int(getattr(bms, "status", 0)),
            "cycle": int(getattr(bms, "cycle", 0)),
            "current_ma": int(getattr(bms, "current", 0)),
            "power_v": _finite(getattr(message, "power_v", 0.0)),
            "power_a": _finite(getattr(message, "power_a", 0.0)),
        },
        "imu": {
            "rpy": _vector(getattr(imu, "rpy", ()), 3),
            "quaternion": _vector(getattr(imu, "quaternion", ()), 4),
            "gyroscope": _vector(getattr(imu, "gyroscope", ()), 3),
            "accelerometer": _vector(getattr(imu, "accelerometer", ()), 3),
            "temperature_c": int(getattr(imu, "temperature", 0)),
        },
        "motors": {
            "count": len(motors),
            "max_temperature_c": max(
                (int(getattr(motor, "temperature", 0)) for motor in motors),
                default=0,
            ),
            "lost_total": sum(int(getattr(motor, "lost", 0)) for motor in motors),
        },
        "foot_force": [int(value) for value in list(getattr(message, "foot_force", ()))[:4]],
        "tick": int(getattr(message, "tick", 0)),
    }


def project_sport_state(message: Any) -> dict[str, Any]:
    return {
        "motion": {
            "error_code": int(getattr(message, "error_code", 0)),
            "mode": int(getattr(message, "mode", 0)),
            "gait_type": int(getattr(message, "gait_type", 0)),
            "progress": _finite(getattr(message, "progress", 0.0)),
            "position": _vector(getattr(message, "position", ()), 3),
            "velocity": _vector(getattr(message, "velocity", ()), 3),
            "yaw_speed": _finite(getattr(message, "yaw_speed", 0.0)),
            "body_height": _finite(getattr(message, "body_height", 0.0)),
            "foot_raise_height": _finite(
                getattr(message, "foot_raise_height", 0.0)
            ),
            "range_obstacle": _vector(getattr(message, "range_obstacle", ()), 4),
        }
    }


def project_lidar_state(message: Any) -> dict[str, Any]:
    return {
        "lidar": {
            "error_state": int(getattr(message, "error_state", 0)),
            "cloud_frequency": _finite(getattr(message, "cloud_frequency", 0.0)),
            "cloud_packet_loss_rate": _finite(
                getattr(message, "cloud_packet_loss_rate", 0.0)
            ),
            "cloud_size": int(getattr(message, "cloud_size", 0)),
            "imu_frequency": _finite(getattr(message, "imu_frequency", 0.0)),
            "imu_packet_loss_rate": _finite(
                getattr(message, "imu_packet_loss_rate", 0.0)
            ),
        }
    }


def project_lidar_imu(message: Any) -> dict[str, Any]:
    orientation = getattr(message, "orientation", None)
    angular = getattr(message, "angular_velocity", None)
    linear = getattr(message, "linear_acceleration", None)
    quaternion = [_finite(getattr(orientation, key, 0.0)) for key in "xyzw"]
    x, y, z, w = quaternion
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return {
        "imu": {
            "source": "utlidar",
            "frame_id": str(
                getattr(getattr(message, "header", None), "frame_id", "")
            ),
            "rpy": [roll, pitch, yaw],
            "quaternion": quaternion,
            "gyroscope": [_finite(getattr(angular, key, 0.0)) for key in "xyz"],
            "accelerometer": [_finite(getattr(linear, key, 0.0)) for key in "xyz"],
        }
    }


def project_lidar_cloud(message: Any) -> dict[str, Any]:
    width = max(0, int(getattr(message, "width", 0)))
    height = max(0, int(getattr(message, "height", 0)))
    return {
        "lidar": {
            "frame_id": str(getattr(getattr(message, "header", None), "frame_id", "")),
            "point_count": width * height,
            "width": width,
            "height": height,
            "point_step": max(0, int(getattr(message, "point_step", 0))),
            "row_step": max(0, int(getattr(message, "row_step", 0))),
        }
    }


def project_odometry(message: Any) -> dict[str, Any]:
    pose = getattr(getattr(message, "pose", None), "pose", None)
    twist = getattr(getattr(message, "twist", None), "twist", None)
    position = getattr(pose, "position", None)
    orientation = getattr(pose, "orientation", None)
    linear = getattr(twist, "linear", None)
    angular = getattr(twist, "angular", None)
    return {
        "odometry": {
            "position": [_finite(getattr(position, key, 0.0)) for key in "xyz"],
            "orientation_xyzw": [
                _finite(getattr(orientation, key, 0.0)) for key in "xyzw"
            ],
            "linear_velocity": [_finite(getattr(linear, key, 0.0)) for key in "xyz"],
            "angular_velocity": [_finite(getattr(angular, key, 0.0)) for key in "xyz"],
        }
    }


class H264JpegDecoder:
    """Persistent bounded ffmpeg pipe for Unitree's Annex-B fragments."""

    def __init__(
        self,
        executable: str,
        preview_hz: float,
        callback: Callable[[bytes], None],
    ) -> None:
        self.executable = executable
        self.preview_hz = preview_hz
        self.callback = callback
        self.status = "not_started"
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=64)
        self._process: subprocess.Popen[bytes] | None = None
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        resolved = shutil.which(self.executable)
        if resolved is None:
            self.status = "decoder_unavailable"
            return
        command = [
            resolved,
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-vf",
            f"fps={self.preview_hz},scale='min(640,iw)':-2",
            "-f",
            "image2pipe",
            "-c:v",
            "mjpeg",
            "-q:v",
            "5",
            "pipe:1",
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.status = "running"
        self._threads = [
            threading.Thread(target=self._write_loop, daemon=True),
            threading.Thread(target=self._read_loop, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def feed(self, payload: bytes) -> None:
        if self.status != "running" or not payload:
            return
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(payload)
            except queue.Empty:
                return

    def _write_loop(self) -> None:
        assert self._process is not None and self._process.stdin is not None
        try:
            while True:
                payload = self._queue.get()
                if payload is None:
                    break
                self._process.stdin.write(payload)
                self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.status = "decoder_failed"

    def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        buffer = bytearray()
        try:
            while self._process.poll() is None:
                chunk = self._process.stdout.read(65_536)
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9", max(start + 2, 0))
                    if start < 0 or end < 0:
                        if len(buffer) > 8_000_000:
                            del buffer[:-2]
                        break
                    frame = bytes(buffer[start : end + 2])
                    del buffer[: end + 2]
                    self.callback(frame)
        except OSError:
            self.status = "decoder_failed"

    def stop(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        process = self._process
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        self.status = "stopped"


class V4l2CameraPoller:
    """Cycle USB cameras so one bandwidth-constrained hub has one active stream."""

    def __init__(
        self,
        *,
        executable: str,
        devices: Mapping[str, str],
        capture_hz: float,
        width: int,
        height: int,
        input_format: str,
        horizontal_flip: bool,
        callback: Callable[[str, bytes], None],
    ) -> None:
        self.executable = executable
        self.devices = dict(devices)
        self.capture_hz = capture_hz
        self.width = width
        self.height = height
        self.input_format = input_format
        self.horizontal_flip = horizontal_flip
        self.callback = callback
        self._statuses = {view_id: "not_started" for view_id in self.devices}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None

    def status(self, view_id: str) -> str:
        with self._lock:
            return self._statuses.get(view_id, "unknown")

    def _set_status(self, view_id: str, status: str) -> None:
        with self._lock:
            self._statuses[view_id] = status

    def start(self) -> None:
        resolved = shutil.which(self.executable)
        if resolved is None:
            for view_id in self.devices:
                self._set_status(view_id, "capture_unavailable")
            return
        for view_id, device in self.devices.items():
            if not Path(device).exists():
                self._set_status(view_id, "device_missing")
        self._thread = threading.Thread(
            target=self._run, args=(resolved,), daemon=True
        )
        self._thread.start()

    def _command(self, resolved: str, device: str) -> list[str]:
        command = [
            resolved,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-f",
            "v4l2",
            "-input_format",
            self.input_format,
            "-framerate",
            str(self.capture_hz),
            "-video_size",
            f"{self.width}x{self.height}",
            "-i",
            device,
            "-frames:v",
            "1",
            "-an",
        ]
        if self.horizontal_flip:
            command.extend(["-vf", "hflip", "-c:v", "mjpeg", "-q:v", "5"])
        elif self.input_format == "mjpeg":
            command.extend(["-c:v", "copy"])
        else:
            command.extend(["-c:v", "mjpeg", "-q:v", "5"])
        command.extend(["-f", "image2pipe", "pipe:1"])
        return command

    def _run(self, resolved: str) -> None:
        while not self._stop.is_set():
            for view_id, device in self.devices.items():
                if self._stop.is_set():
                    return
                if not Path(device).exists():
                    self._set_status(view_id, "device_missing")
                    continue
                self._set_status(view_id, "capturing")
                process = subprocess.Popen(
                    self._command(resolved, device),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                with self._lock:
                    self._process = process
                try:
                    stdout, _stderr = process.communicate(timeout=4)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        stdout, _stderr = process.communicate(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, _stderr = process.communicate()
                    self._set_status(view_id, "capture_timeout")
                    continue
                finally:
                    with self._lock:
                        if self._process is process:
                            self._process = None
                start = stdout.find(b"\xff\xd8")
                end = stdout.rfind(b"\xff\xd9")
                if process.returncode == 0 and start >= 0 and end > start:
                    self._set_status(view_id, "live")
                    self.callback(view_id, stdout[start : end + 2])
                elif not self._stop.is_set():
                    self._set_status(view_id, "capture_failed")

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            for view_id in self._statuses:
                self._statuses[view_id] = "stopped"


class RosTelemetrySidecar:
    def __init__(self, config: RosAdapterConfig) -> None:
        self.config = config
        now = time.monotonic()
        self._topics = {
            key: {
                "topic": config.topics[key],
                "type": TOPIC_TYPES[key],
                "received": False,
                "count": 0,
                "age_s": None,
            }
            for key in TOPIC_TYPES
        }
        if config.usb_camera_enabled:
            for view_id, device in config.usb_camera_devices.items():
                self._topics[f"usb_{view_id}"] = {
                    "topic": device,
                    "type": "v4l2/mjpeg",
                    "received": False,
                    "count": 0,
                    "age_s": None,
                    "capture_status": "not_started",
                }
        self._last_received_mono = {key: now - 1e9 for key in self._topics}
        self._last_preview_mono = {key: now - 1e9 for key in CAMERA_VIEWS}
        self._robot: dict[str, Any] = {}
        self._camera_rows: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._manifest_lock = threading.Lock()
        self._bridge: Any = None
        self._cv2: Any = None
        self._latest_go2_stamp = 0.0
        self._latest_go2_wall = 0.0
        self._go2_rpc_pending_id: int | None = None
        self._go2_rpc_requested_mono = now - 1e9
        self._go2_rpc_requested_wall = 0.0
        self._decoder = H264JpegDecoder(
            config.go2_h264_decoder, config.preview_hz, self._on_go2_jpeg
        )
        self._usb_poller = (
            V4l2CameraPoller(
                executable="ffmpeg",
                devices=config.usb_camera_devices,
                capture_hz=config.usb_camera_capture_hz,
                width=config.usb_camera_width,
                height=config.usb_camera_height,
                input_format=config.usb_camera_input_format,
                horizontal_flip=config.usb_camera_horizontal_flip,
                callback=self._on_usb_jpeg,
            )
            if config.usb_camera_enabled
            else None
        )

    def _merge_robot(self, projection: Mapping[str, Any]) -> None:
        for key, value in projection.items():
            current = self._robot.get(key)
            if isinstance(current, dict) and isinstance(value, Mapping):
                current.update(value)
            else:
                self._robot[key] = dict(value) if isinstance(value, Mapping) else value

    def _mark(
        self,
        key: str,
        message: Any,
        *,
        width: int | None = None,
        height: int | None = None,
        encoding: str | None = None,
    ) -> None:
        now_mono = time.monotonic()
        row = self._topics[key]
        row["received"] = True
        row["count"] = int(row["count"]) + 1
        row["stamp_s"] = _message_stamp_s(message)
        if width is not None:
            row["width"] = width
        if height is not None:
            row["height"] = height
        if encoding is not None:
            row["encoding"] = encoding
        self._last_received_mono[key] = now_mono

    def on_low_state(self, message: Any) -> None:
        with self._lock:
            self._mark("low_state", message)
            self._merge_robot(project_low_state(message))

    def on_sport_state(self, message: Any) -> None:
        with self._lock:
            self._mark("sport_mode_state", message)
            self._merge_robot(project_sport_state(message))

    def on_lidar_state(self, message: Any) -> None:
        with self._lock:
            self._mark("lidar_state", message)
            self._merge_robot(project_lidar_state(message))

    def on_lidar_imu(self, message: Any) -> None:
        with self._lock:
            self._mark("lidar_imu", message)
            self._merge_robot(project_lidar_imu(message))

    def on_lidar_cloud(self, message: Any) -> None:
        with self._lock:
            self._mark(
                "lidar_cloud",
                message,
                width=int(getattr(message, "width", 0)),
                height=int(getattr(message, "height", 0)),
            )
            self._merge_robot(project_lidar_cloud(message))

    def on_odometry(self, message: Any) -> None:
        with self._lock:
            self._mark("odometry", message)
            self._merge_robot(project_odometry(message))

    def on_go2_video(self, message: Any) -> None:
        payload = b""
        if self.config.go2_camera_source == "h264_topic":
            payload = bytes(getattr(message, "video720p", ()) or ())
            if not payload:
                payload = bytes(getattr(message, "video360p", ()) or ())
        with self._lock:
            self._mark("go2_front", message)
            self._latest_go2_stamp = _message_stamp_s(message)
            self._latest_go2_wall = time.time()
            self._topics["go2_front"]["camera_source"] = (
                self.config.go2_camera_source
            )
            if self.config.go2_camera_source == "h264_topic":
                self._topics["go2_front"]["decoder_status"] = self._decoder.status
        if self.config.go2_camera_source == "h264_topic":
            self._decoder.feed(payload)

    def request_go2_camera(self, publisher: Any, request_type: Any) -> None:
        if self.config.go2_camera_source != "videohub_rpc":
            return
        if publisher.get_subscription_count() < 1:
            return
        now_mono = time.monotonic()
        interval_s = 1.0 / self.config.preview_hz
        if now_mono - self._go2_rpc_requested_mono < interval_s:
            return
        if (
            self._go2_rpc_pending_id is not None
            and now_mono - self._go2_rpc_requested_mono < 3.0
        ):
            return
        if self._go2_rpc_pending_id is not None:
            with self._lock:
                self._topics["go2_front"]["preview_status"] = "rpc_timeout"
        request = request_type()
        request.header.identity.id = time.monotonic_ns()
        request.header.identity.api_id = 1001
        request.header.lease.id = 0
        request.header.policy.priority = 0
        request.header.policy.noreply = False
        request.parameter = ""
        request.binary = []
        publisher.publish(request)
        self._go2_rpc_pending_id = request.header.identity.id
        self._go2_rpc_requested_mono = now_mono
        self._go2_rpc_requested_wall = time.time()

    def on_go2_camera_response(self, message: Any) -> None:
        identity = getattr(getattr(message, "header", None), "identity", None)
        status = getattr(getattr(message, "header", None), "status", None)
        response_id = int(getattr(identity, "id", 0))
        if response_id != self._go2_rpc_pending_id:
            return
        self._go2_rpc_pending_id = None
        status_code = int(getattr(status, "code", -1))
        if status_code != 0:
            with self._lock:
                self._topics["go2_front"]["preview_status"] = (
                    f"rpc_error_{status_code}"
                )
            return
        binary = getattr(message, "binary", ()) or ()
        try:
            jpeg = bytes(bytearray(int(value) & 0xFF for value in binary))
        except (TypeError, ValueError):
            return
        if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
            with self._lock:
                self._topics["go2_front"]["preview_status"] = "rpc_invalid_jpeg"
            return
        cv2 = self._cv2
        if cv2 is None:
            return
        import numpy as np

        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        now_wall = time.time()
        with self._lock:
            stamp = self._latest_go2_stamp or now_wall
            topic_row = self._topics["go2_front"]
            topic_row.update(
                {
                    "received": True,
                    "count": int(topic_row["count"]) + 1,
                    "stamp_s": stamp,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "encoding": "jpeg",
                    "camera_source": "videohub_rpc",
                    "preview_status": "live",
                    "preview_rpc_latency_s": max(
                        0.0, now_wall - self._go2_rpc_requested_wall
                    ),
                }
            )
            self._last_received_mono["go2_front"] = time.monotonic()
        self._publish_camera(
            "go2_front",
            jpeg,
            stamp,
            now_wall,
            int(image.shape[1]),
            int(image.shape[0]),
            "videohub-jpeg",
            source_topic="/api/videohub/response",
        )

    def _on_usb_jpeg(self, view_id: str, jpeg: bytes) -> None:
        if not self._preview_due(view_id):
            return
        now_wall = time.time()
        key = f"usb_{view_id}"
        with self._lock:
            row = self._topics[key]
            row["received"] = True
            row["count"] = int(row["count"]) + 1
            row["stamp_s"] = now_wall
            row["width"] = self.config.usb_camera_width
            row["height"] = self.config.usb_camera_height
            row["encoding"] = "mjpeg"
            row["capture_status"] = (
                self._usb_poller.status(view_id)
                if self._usb_poller is not None
                else "disabled"
            )
            self._last_received_mono[key] = time.monotonic()
        self._publish_camera(
            view_id,
            jpeg,
            now_wall,
            now_wall,
            self.config.usb_camera_width,
            self.config.usb_camera_height,
            (
                "v4l2-mjpeg-hflip"
                if self.config.usb_camera_horizontal_flip
                else "v4l2-mjpeg"
            ),
            source_topic=self.config.usb_camera_devices[view_id],
        )

    def _on_go2_jpeg(self, jpeg: bytes) -> None:
        cv2 = self._cv2
        if cv2 is None:
            return
        import numpy as np

        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        with self._lock:
            stamp = self._latest_go2_stamp
            wall = self._latest_go2_wall or time.time()
        self._publish_camera(
            "go2_front",
            jpeg,
            stamp,
            wall,
            int(image.shape[1]),
            int(image.shape[0]),
            "h264->jpeg",
        )

    def _preview_due(self, key: str) -> bool:
        now = time.monotonic()
        if now - self._last_preview_mono[key] < 1.0 / self.config.preview_hz:
            return False
        self._last_preview_mono[key] = now
        return True

    def on_d435_color(self, message: Any) -> None:
        with self._lock:
            self._mark(
                "d435_color",
                message,
                width=int(getattr(message, "width", 0)),
                height=int(getattr(message, "height", 0)),
                encoding=str(getattr(message, "encoding", "")),
            )
        if not self._preview_due("d435_color"):
            return
        image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        ok, encoded = self._cv2.imencode(
            ".jpg", image, [self._cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        )
        if ok:
            self._publish_camera(
                "d435_color",
                bytes(encoded),
                _message_stamp_s(message),
                time.time(),
                int(image.shape[1]),
                int(image.shape[0]),
                "rgb8->jpeg",
            )

    def on_d435_depth(self, message: Any) -> None:
        encoding = str(getattr(message, "encoding", ""))
        with self._lock:
            self._mark(
                "d435_depth",
                message,
                width=int(getattr(message, "width", 0)),
                height=int(getattr(message, "height", 0)),
                encoding=encoding,
            )
        if not self._preview_due("d435_depth"):
            return
        import numpy as np

        depth = self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        depth_m = depth.astype(np.float32)
        if encoding.upper() in {"16UC1", "MONO16", "Z16"}:
            depth_m *= 0.001
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        clipped = np.clip(depth_m, 0.15, 5.0)
        normalized = ((clipped - 0.15) * (255.0 / 4.85)).astype(np.uint8)
        colored = self._cv2.applyColorMap(255 - normalized, self._cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        ok, encoded_jpeg = self._cv2.imencode(
            ".jpg",
            colored,
            [self._cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        if ok:
            self._publish_camera(
                "d435_depth",
                bytes(encoded_jpeg),
                _message_stamp_s(message),
                time.time(),
                int(colored.shape[1]),
                int(colored.shape[0]),
                f"{encoding}->turbo-jpeg",
            )

    def _publish_camera(
        self,
        view_id: str,
        jpeg: bytes,
        stamp_s: float,
        received_wall_time_s: float,
        width: int,
        height: int,
        encoding: str,
        *,
        source_topic: str | None = None,
    ) -> None:
        path = self.config.camera_dir / f"{view_id}.jpg"
        row = {
            "view_id": view_id,
            "source_topic": source_topic or self.config.topics[view_id],
            "stamp_s": max(0.0, stamp_s),
            "received_wall_time_s": received_wall_time_s,
            "width": width,
            "height": height,
            "encoding": encoding,
            "jpeg_path": str(path),
            "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
        }
        with self._manifest_lock:
            atomic_write_bytes(path, jpeg)
            with self._lock:
                self._camera_rows[view_id] = row
                manifest = {
                    "schema_version": 1,
                    "kind": "vla_nav_panel_ros_cameras",
                    "updated_wall_time_s": time.time(),
                    "cameras": [
                        dict(self._camera_rows[key])
                        for key in CAMERA_VIEWS
                        if key in self._camera_rows
                    ],
                }
            atomic_write_json(self.config.camera_manifest_path, manifest)

    def publish_state(self) -> None:
        now_mono = time.monotonic()
        with self._lock:
            if self._usb_poller is not None:
                for view_id in USB_CAMERA_VIEWS:
                    self._topics[f"usb_{view_id}"]["capture_status"] = (
                        self._usb_poller.status(view_id)
                    )
            topics = {}
            for key, current in self._topics.items():
                row = dict(current)
                age_s = max(0.0, now_mono - self._last_received_mono[key])
                row["age_s"] = age_s if row["received"] else None
                row["fresh"] = bool(row["received"] and age_s <= self.config.stale_after_s)
                topics[key] = row
            required = (
                "go2_front",
                "lidar_state",
                "lidar_imu",
                "lidar_cloud",
                "d435_color",
                "d435_depth",
                *(
                    tuple(f"usb_{view_id}" for view_id in USB_CAMERA_VIEWS)
                    if self.config.usb_camera_enabled
                    else ()
                ),
            )
            ready = all(bool(topics[key]["fresh"]) for key in required)
            any_received = any(bool(row["received"]) for row in topics.values())
            record = {
                "schema_version": 1,
                "kind": "vla_nav_panel_ros_state",
                "ready": ready,
                "status": "live" if ready else ("partial" if any_received else "waiting"),
                "updated_wall_time_s": time.time(),
                "node": {
                    "name": self.config.node_name,
                    "rmw": os.environ.get("RMW_IMPLEMENTATION", ""),
                    "domain_id": int(os.environ.get("ROS_DOMAIN_ID", "0")),
                },
                "topics": topics,
                "robot": dict(self._robot),
            }
        atomic_write_json(self.config.state_path, record)

    def run(self) -> None:
        import cv2
        import rclpy
        from cv_bridge import CvBridge
        from nav_msgs.msg import Odometry
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, Imu, PointCloud2
        from unitree_api.msg import Request, Response
        from unitree_go.msg import Go2FrontVideoData, LidarState, LowState, SportModeState

        self._cv2 = cv2
        self._bridge = CvBridge()
        context = Context()
        rclpy.init(context=context)
        node = Node(
            self.config.node_name,
            context=context,
            enable_rosout=False,
            start_parameter_services=False,
            enable_logger_service=False,
        )
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        subscriptions = [
            (LowState, "low_state", self.on_low_state),
            (SportModeState, "sport_mode_state", self.on_sport_state),
            (LidarState, "lidar_state", self.on_lidar_state),
            (Imu, "lidar_imu", self.on_lidar_imu),
            (PointCloud2, "lidar_cloud", self.on_lidar_cloud),
            (Odometry, "odometry", self.on_odometry),
            (Image, "d435_color", self.on_d435_color),
            (Image, "d435_depth", self.on_d435_depth),
        ]
        if self.config.go2_camera_source == "h264_topic":
            subscriptions.insert(
                0, (Go2FrontVideoData, "go2_front", self.on_go2_video)
            )
        # Intentionally the only ROS entities created by this process.
        handles = [
            node.create_subscription(
                message_type,
                self.config.topics[key],
                callback,
                qos_profile_sensor_data,
            )
            for message_type, key, callback in subscriptions
        ]
        go2_camera_publisher = None
        if self.config.go2_camera_source == "videohub_rpc":
            handles.append(
                node.create_subscription(
                    Response,
                    "/api/videohub/response",
                    self.on_go2_camera_response,
                    10,
                )
            )
            go2_camera_publisher = node.create_publisher(
                Request, "/api/videohub/request", 10
            )
        else:
            self._decoder.start()
        if self._usb_poller is not None:
            self._usb_poller.start()
        next_publish = 0.0
        try:
            while context.ok():
                executor.spin_once(timeout_sec=0.1)
                now = time.monotonic()
                if go2_camera_publisher is not None:
                    self.request_go2_camera(go2_camera_publisher, Request)
                if now >= next_publish:
                    self.publish_state()
                    next_publish = now + 1.0 / self.config.state_publish_hz
        except KeyboardInterrupt:
            pass
        finally:
            if self.config.go2_camera_source == "h264_topic":
                self._decoder.stop()
            if self._usb_poller is not None:
                self._usb_poller.stop()
            del handles
            executor.remove_node(node)
            node.destroy_node()
            rclpy.shutdown(context=context)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the vla-nav-panel ROS sidecar.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    RosTelemetrySidecar(load_config(Path(args.config).resolve())).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
