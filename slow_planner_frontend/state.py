from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    LANE_ID,
    PLANNER_METRIC_FIELDS,
    REV_C_IMAGE_SIZE,
    REV_C_VIEW_ORDER,
    ValidatedRevCSnapshot,
    parse_lane_b_snapshot_id,
    snapshot_content_sha256,
)


ROS_LIVE_CAMERA_VIEW_ORDER = ("go2_front", "d435_color", "d435_depth")
ROS_CAMERA_VIEW_ORDER = (*REV_C_VIEW_ORDER, *ROS_LIVE_CAMERA_VIEW_ORDER)
_ROS_TOPIC_KEYS = frozenset(
    {
        "go2_front",
        "low_state",
        "sport_mode_state",
        "lidar_state",
        "lidar_imu",
        "lidar_cloud",
        "odometry",
        "d435_color",
        "d435_depth",
        "usb_front_left",
        "usb_front",
        "usb_front_right",
        "usb_rear",
    }
)

_PRIVATE_KEYS = frozenset(
    {
        "analysis",
        "chain_of_thought",
        "chainofthought",
        "correction",
        "cot",
        "hidden_reasoning",
        "hiddenreasoning",
        "model_raw_text",
        "model_output",
        "prompt",
        "raw_generation",
        "rawgeneration",
        "raw_output",
        "rawoutput",
        "raw_text",
        "rawtext",
        "reasoning",
        "thoughts",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "snapshot_id",
        "mode",
        "source_decision",
        "intent",
        "frontier_id",
        "target_relative_xz",
        "confidence",
        "scene_summary",
        "target_evidence",
        "blocked_directions",
        "recommended_frontier",
        "target_found",
        "abstain",
        "fallback_used",
        "fallback_reason",
        "requires_arrival_confirmation",
        "requires_internvla_fallback",
        "fallback_owner",
        "fallback_candidate",
        "motion_authority",
    }
)
_SNAPSHOT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "lane_id",
        "episode_id",
        "reset_id",
        "sequence_id",
        "snapshot_id",
        "snapshot_sim_stamp_s",
        "written_wall_time_s",
        "config_sha256",
        "view_order",
        "inter_camera_skew_s",
        "max_frame_age_s",
        "max_inter_camera_skew_s",
        "snapshot_content_sha256",
    }
)
_SNAPSHOT_IMAGE_FIELDS = frozenset(
    {
        "view_id",
        "source_frame_id",
        "sim_stamp_s",
        "age_s",
        "width",
        "height",
        "pose",
        "extrinsic_sha256",
        "jpeg_sha256",
    }
)
_PUBLIC_REASON_RE = re.compile(r"^[A-Za-z0-9_.:-]{0,128}$")
_PUBLIC_SEMANTIC_RE = re.compile(r"^[^\x00-\x1f\x7f<>`]{0,160}$")


def _normalized_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def sanitize_public(value: Any) -> Any:
    """Recursively remove model generations and hidden-reasoning fields."""

    if isinstance(value, Mapping):
        return {
            str(key): sanitize_public(item)
            for key, item in value.items()
            if _normalized_key(key) not in _PRIVATE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_public(item) for item in value]
    if isinstance(value, bytes):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _require_sha256(value: Any, name: str) -> str:
    normalized = str(value or "")
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _require_finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(
            f"{name} must be finite" + (" and non-negative" if nonnegative else "")
        )
    return number


def _require_int(value: Any, name: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _public_snapshot_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    public = {
        key: sanitize_public(value)
        for key, value in record.items()
        if key in _SNAPSHOT_FIELDS
    }
    rows = record.get("images") or []
    public["images"] = [
        {
            key: sanitize_public(value)
            for key, value in row.items()
            if key in _SNAPSHOT_IMAGE_FIELDS
        }
        for row in rows
        if isinstance(row, Mapping)
    ]
    return public


def _expand_path(value: str | os.PathLike[str]) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if re.search(
        r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*|%[A-Za-z_][A-Za-z0-9_]*%", expanded
    ):
        raise ValueError(
            f"frontend path contains an unresolved environment variable: {value!r}"
        )
    return Path(expanded).resolve()


@dataclass(frozen=True)
class FrontendConfig:
    host: str
    port: int
    poll_interval_s: float
    snapshot_sidecar_path: Path
    decision_log_path: Path
    health_path: Path
    gpu_telemetry_path: Path
    camera_dir: Path
    ros_state_path: Path | None
    ros_camera_manifest_path: Path | None
    ros_camera_dir: Path | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrontendConfig":
        frontend = value.get("frontend", value)
        if not isinstance(frontend, Mapping):
            raise ValueError("frontend config must be an object")
        paths = frontend.get("paths") or {}
        if not isinstance(paths, Mapping):
            raise ValueError("frontend.paths must be an object")
        host = str(frontend.get("host", "127.0.0.1"))
        port = int(frontend.get("port", 8300))
        poll_interval_s = float(frontend.get("poll_interval_s", 0.25))
        if not host:
            raise ValueError("frontend.host is required")
        if not 1 <= port <= 65535:
            raise ValueError("frontend.port must be in [1, 65535]")
        if not 0.05 <= poll_interval_s <= 10.0:
            raise ValueError("frontend.poll_interval_s must be in [0.05, 10.0]")

        required = (
            "snapshot_sidecar_path",
            "decision_log_path",
            "health_path",
            "gpu_telemetry_path",
            "camera_dir",
        )
        missing = [name for name in required if not str(paths.get(name, "")).strip()]
        if missing:
            raise ValueError(f"frontend paths missing required values: {missing}")
        ros_adapter = frontend.get("ros_adapter") or {}
        if not isinstance(ros_adapter, Mapping):
            raise ValueError("frontend.ros_adapter must be an object")
        ros_enabled = bool(ros_adapter.get("enabled", False))
        ros_paths = ros_adapter.get("paths") or {}
        if not isinstance(ros_paths, Mapping):
            raise ValueError("frontend.ros_adapter.paths must be an object")
        ros_required = ("state_path", "camera_manifest_path", "camera_dir")
        if ros_enabled:
            ros_missing = [
                name for name in ros_required if not str(ros_paths.get(name, "")).strip()
            ]
            if ros_missing:
                raise ValueError(
                    f"frontend.ros_adapter paths missing required values: {ros_missing}"
                )
        return cls(
            host=host,
            port=port,
            poll_interval_s=poll_interval_s,
            snapshot_sidecar_path=_expand_path(paths["snapshot_sidecar_path"]),
            decision_log_path=_expand_path(paths["decision_log_path"]),
            health_path=_expand_path(paths["health_path"]),
            gpu_telemetry_path=_expand_path(paths["gpu_telemetry_path"]),
            camera_dir=_expand_path(paths["camera_dir"]),
            ros_state_path=(
                _expand_path(ros_paths["state_path"]) if ros_enabled else None
            ),
            ros_camera_manifest_path=(
                _expand_path(ros_paths["camera_manifest_path"])
                if ros_enabled
                else None
            ),
            ros_camera_dir=(
                _expand_path(ros_paths["camera_dir"]) if ros_enabled else None
            ),
        )


@dataclass(frozen=True)
class CameraFrame:
    view_id: str
    jpeg: bytes
    snapshot_id: str
    sim_stamp_s: float
    age_s: float
    extrinsic_sha256: str
    source_frame_id: str
    width: int
    height: int

    def metadata(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "snapshot_id": self.snapshot_id,
            "sim_stamp_s": self.sim_stamp_s,
            "age_s": self.age_s,
            "extrinsic_sha256": self.extrinsic_sha256,
            "source_frame_id": self.source_frame_id,
            "width": self.width,
            "height": self.height,
            "available": bool(self.jpeg),
            "url": f"/api/v1/cameras/{self.view_id}.jpg",
        }


@dataclass(frozen=True)
class RosCameraFrame:
    view_id: str
    jpeg: bytes
    stamp_s: float
    received_wall_time_s: float
    source_topic: str
    width: int
    height: int
    encoding: str

    def metadata(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "stamp_s": self.stamp_s,
            "age_s": max(0.0, time.time() - self.received_wall_time_s),
            "source_topic": self.source_topic,
            "width": self.width,
            "height": self.height,
            "encoding": self.encoding,
            "available": bool(self.jpeg),
            "url": f"/api/v1/cameras/{self.view_id}.jpg",
        }


class FrontendStateStore:
    """Thread-safe, in-memory projection of files produced by the Lane B runtime."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._version = 0
        self._updated_wall_time_s = 0.0
        self._health: dict[str, Any] = {"ready": False, "status": "waiting_for_runtime"}
        self._gpu: dict[str, Any] = {}
        self._latency: dict[str, Any] = {}
        self._decision: dict[str, Any] = {}
        self._snapshot: dict[str, Any] = {}
        self._cameras: dict[str, CameraFrame] = {}
        self._ros: dict[str, Any] = {
            "ready": False,
            "status": "waiting_for_ros_adapter",
        }
        self._robot: dict[str, Any] = {}
        self._ros_cameras: dict[str, RosCameraFrame] = {}
        self._ingest_warnings: list[str] = []

    def _touch(self) -> None:
        self._version += 1
        self._updated_wall_time_s = time.time()

    def publish_health(self, value: Mapping[str, Any]) -> None:
        allowed = {
            "ready",
            "status",
            "model_variant",
            "precision_mode",
            "protocol_version",
            "revision",
            "service_config_sha256",
            "max_retries",
            "max_request_age_s",
            "batch_size",
            "pacore",
            "multi_crop",
            "fix_mistral_regex",
            "runtime_transformers_version",
            "checkpoint_key_mapping",
            "checkpoint_load_clean",
            "checkpoint_loading_counts",
            "parameter_count",
            "parameter_bytes",
            "parameter_dtype_counts",
            "uptime_s",
        }
        with self._lock:
            self._health = dict(
                sanitize_public(
                    {key: item for key, item in value.items() if key in allowed}
                )
            )
            self._touch()

    def publish_gpu(self, value: Mapping[str, Any]) -> None:
        allowed = {
            "timestamp",
            "gpu_index",
            "gpu_name",
            "utilization_percent",
            "gpu_util_percent",
            "memory_used_mib",
            "used_memory_mib",
            "memory_total_mib",
            "temperature_c",
            "power_w",
            "unified_memory_total_mib",
            "unified_memory_available_mib",
            "system_swap_used_mib",
            "step3_process_rss_mib",
            "step3_process_swap_mib",
        }
        with self._lock:
            self._gpu = dict(
                sanitize_public(
                    {key: item for key, item in value.items() if key in allowed}
                )
            )
            self._touch()

    def publish_decision(self, value: Mapping[str, Any]) -> None:
        sanitized = dict(sanitize_public(value))
        if sanitized.get("kind") not in (None, "lane_b_planner_decision"):
            raise ValueError("frontend rejects an unknown decision record kind")
        decision = sanitized.get("decision")
        if not isinstance(decision, Mapping):
            raise ValueError("frontend decision record must contain a decision object")
        public_decision = {
            key: item for key, item in decision.items() if key in _DECISION_FIELDS
        }
        mode = public_decision.get("mode")
        if mode is not None and mode not in {"bounded_advisor", "direct_high_level"}:
            raise ValueError("frontend rejects an unknown planner mode")
        intent = public_decision.get("intent")
        if intent is not None and intent not in {
            "frontier_advice",
            "frontier_goal_candidate",
            "relative_target_safe_hold_candidate",
            "INTERNVLA_FALLBACK_REQUIRED",
        }:
            raise ValueError("frontend rejects an unknown planner intent")
        fallback_reason = str(public_decision.get("fallback_reason") or "")
        if not _PUBLIC_REASON_RE.fullmatch(fallback_reason):
            raise ValueError("frontend rejects an unstructured fallback reason")
        public_decision["fallback_reason"] = fallback_reason
        scene_summary = str(public_decision.get("scene_summary") or "")
        if not _PUBLIC_SEMANTIC_RE.fullmatch(scene_summary):
            raise ValueError("frontend rejects an unsafe scene summary")
        public_decision["scene_summary"] = scene_summary
        for field, maximum, allowed in (
            ("target_evidence", 2, None),
            ("blocked_directions", 4, set(REV_C_VIEW_ORDER)),
        ):
            items = public_decision.get(field) or []
            if (
                not isinstance(items, list)
                or len(items) > maximum
                or any(
                    not isinstance(item, str)
                    or not _PUBLIC_SEMANTIC_RE.fullmatch(item)
                    for item in items
                )
                or len(set(items)) != len(items)
                or (allowed is not None and any(item not in allowed for item in items))
            ):
                raise ValueError(f"frontend rejects invalid {field}")
            public_decision[field] = items
        recommended = public_decision.get("recommended_frontier")
        if recommended is not None and (
            isinstance(recommended, bool) or not isinstance(recommended, int)
        ):
            raise ValueError("frontend rejects invalid recommended_frontier")
        for field in ("target_found", "abstain"):
            if field in public_decision and not isinstance(
                public_decision[field], bool
            ):
                raise ValueError(f"frontend rejects non-boolean {field}")
        if public_decision.get("motion_authority") not in (None, "none"):
            raise ValueError("frontend rejects a decision with motion authority")
        if public_decision.get("fallback_owner") not in (
            None,
            "coordinator_frozen_internvla_candidate",
        ):
            raise ValueError("frontend rejects an unknown fallback owner")
        if public_decision.get("fallback_candidate") not in (None, "a1+b1+c1"):
            raise ValueError("frontend rejects an unknown fallback candidate")
        metrics = sanitized.get("metrics")
        public_metrics = (
            {
                key: item
                for key, item in metrics.items()
                if key in PLANNER_METRIC_FIELDS
            }
            if isinstance(metrics, Mapping)
            else {}
        )
        public = {
            "kind": "lane_b_planner_decision",
            "published_wall_time_s": sanitized.get("published_wall_time_s"),
            "decision": public_decision,
            "metrics": public_metrics,
        }
        with self._lock:
            self._decision = public
            self._latency = public_metrics
            self._touch()

    def publish_validated_snapshot(self, snapshot: ValidatedRevCSnapshot) -> None:
        record = snapshot.sidecar_record()
        frames = {
            frame.image.view_id: CameraFrame(
                view_id=frame.image.view_id,
                jpeg=frame.image.jpeg,
                snapshot_id=snapshot.identity.snapshot_id,
                sim_stamp_s=frame.sim_stamp_s,
                age_s=age_s,
                extrinsic_sha256=frame.extrinsic_sha256,
                source_frame_id=frame.source_frame_id,
                width=frame.image.width,
                height=frame.image.height,
            )
            for frame, age_s in zip(snapshot.frames, snapshot.frame_ages_s)
        }
        with self._lock:
            self._snapshot = _public_snapshot_projection(record)
            self._cameras = frames
            self._touch()

    def publish_snapshot_record(
        self, value: Mapping[str, Any], camera_dir: Path
    ) -> None:
        record = dict(value)
        if (
            type(record.get("schema_version")) is not int
            or record.get("schema_version") != 1
        ):
            raise ValueError("frontend rejects an unsupported snapshot schema")
        if record.get("kind") != "lane_b_rev_c_snapshot":
            raise ValueError("frontend rejects an unknown snapshot record kind")
        if record.get("lane_id") != LANE_ID:
            raise ValueError("frontend rejects non-Lane-B snapshot records")
        identity = parse_lane_b_snapshot_id(str(record.get("snapshot_id") or ""))
        reset_id = _require_int(record.get("reset_id"), "reset_id", nonnegative=True)
        sequence_id = _require_int(
            record.get("sequence_id"), "sequence_id", nonnegative=True
        )
        if (
            record.get("episode_id") != identity.episode_id
            or reset_id != identity.reset_id
            or sequence_id != identity.sequence_id
        ):
            raise ValueError("frontend rejects mismatched snapshot identity fields")
        _require_sha256(record.get("config_sha256"), "snapshot config_sha256")
        declared_content_hash = _require_sha256(
            record.get("snapshot_content_sha256"), "snapshot_content_sha256"
        )
        if snapshot_content_sha256(record) != declared_content_hash:
            raise ValueError("frontend rejects a mismatched snapshot content hash")
        if tuple(record.get("view_order") or ()) != REV_C_VIEW_ORDER:
            raise ValueError(
                "frontend rejects snapshot records with a non-Rev-C view order"
            )
        image_rows = record.get("images") or []
        if not isinstance(image_rows, list):
            raise ValueError("snapshot images must be an array")
        if len(image_rows) != len(REV_C_VIEW_ORDER):
            raise ValueError("snapshot must contain exactly four image rows")
        if any(not isinstance(row, Mapping) for row in image_rows):
            raise ValueError("snapshot image rows must be objects")
        if (
            tuple(str(row.get("view_id") or "") for row in image_rows)
            != REV_C_VIEW_ORDER
        ):
            raise ValueError("snapshot image rows must use the fixed Rev-C order")

        frames: dict[str, CameraFrame] = {}
        camera_root = camera_dir.resolve()
        expected_width, expected_height = REV_C_IMAGE_SIZE
        for view_id, row in zip(REV_C_VIEW_ORDER, image_rows):
            width = _require_int(
                row.get("width"), f"image[{view_id}].width", nonnegative=True
            )
            height = _require_int(
                row.get("height"), f"image[{view_id}].height", nonnegative=True
            )
            if (width, height) != (expected_width, expected_height):
                raise ValueError(
                    f"snapshot image[{view_id}] must be {expected_width}x{expected_height}"
                )
            extrinsic_sha256 = _require_sha256(
                row.get("extrinsic_sha256"), f"image[{view_id}].extrinsic_sha256"
            )
            jpeg_sha256 = _require_sha256(
                row.get("jpeg_sha256"), f"image[{view_id}].jpeg_sha256"
            )
            source_frame_id = str(row.get("source_frame_id") or "")
            if not source_frame_id:
                raise ValueError(
                    f"snapshot image[{view_id}] source_frame_id is required"
                )
            sim_stamp_s = _require_finite(
                row.get("sim_stamp_s"),
                f"image[{view_id}].sim_stamp_s",
                nonnegative=True,
            )
            age_s = _require_finite(
                row.get("age_s"), f"image[{view_id}].age_s", nonnegative=True
            )
            pose = row.get("pose")
            if not isinstance(pose, (list, tuple)) or not pose:
                raise ValueError(f"snapshot image[{view_id}] pose is required")
            for index, component in enumerate(pose):
                _require_finite(component, f"image[{view_id}].pose[{index}]")
            raw_jpeg_path = str(row.get("jpeg_path") or "")
            if not raw_jpeg_path:
                raise ValueError(f"snapshot image[{view_id}] is missing jpeg_path")
            jpeg_path = Path(raw_jpeg_path).resolve()
            if (
                not jpeg_path.is_relative_to(camera_root)
                or jpeg_path.name != f"{view_id}.jpg"
            ):
                raise ValueError(
                    f"snapshot image[{view_id}] jpeg_path escapes the configured camera root"
                )
            if not jpeg_path.is_file():
                raise ValueError(f"snapshot image[{view_id}] JPEG is missing")
            jpeg = jpeg_path.read_bytes()
            if hashlib.sha256(jpeg).hexdigest() != jpeg_sha256:
                raise ValueError(f"snapshot image[{view_id}] JPEG hash mismatch")
            frames[view_id] = CameraFrame(
                view_id=view_id,
                jpeg=jpeg,
                snapshot_id=identity.snapshot_id,
                sim_stamp_s=sim_stamp_s,
                age_s=age_s,
                extrinsic_sha256=extrinsic_sha256,
                source_frame_id=source_frame_id,
                width=width,
                height=height,
            )
        with self._lock:
            self._snapshot = _public_snapshot_projection(record)
            self._cameras = frames
            self._touch()

    def publish_ros_state(self, value: Mapping[str, Any]) -> None:
        record = dict(value)
        if record.get("schema_version") != 1:
            raise ValueError("frontend rejects an unsupported ROS state schema")
        if record.get("kind") != "vla_nav_panel_ros_state":
            raise ValueError("frontend rejects an unknown ROS state kind")
        if len(json.dumps(record, separators=(",", ":"), default=str)) > 65_536:
            raise ValueError("frontend rejects an oversized ROS state record")
        topics = record.get("topics") or {}
        robot = record.get("robot") or {}
        node = record.get("node") or {}
        if not isinstance(topics, Mapping) or not isinstance(robot, Mapping):
            raise ValueError("ROS state topics and robot fields must be objects")
        if not isinstance(node, Mapping):
            raise ValueError("ROS state node field must be an object")
        public_topics = {
            str(key): sanitize_public(item)
            for key, item in topics.items()
            if key in _ROS_TOPIC_KEYS and isinstance(item, Mapping)
        }
        public_ros = {
            "ready": bool(record.get("ready", False)),
            "status": str(record.get("status") or "waiting_for_topics")[:128],
            "updated_wall_time_s": record.get("updated_wall_time_s"),
            "node": sanitize_public(
                {
                    key: item
                    for key, item in node.items()
                    if key in {"name", "rmw", "domain_id"}
                }
            ),
            "topics": public_topics,
        }
        with self._lock:
            self._ros = dict(public_ros)
            self._robot = dict(sanitize_public(robot))
            self._touch()

    def publish_ros_camera_manifest(
        self, value: Mapping[str, Any], camera_dir: Path
    ) -> None:
        record = dict(value)
        if record.get("schema_version") != 1:
            raise ValueError("frontend rejects an unsupported ROS camera schema")
        if record.get("kind") != "vla_nav_panel_ros_cameras":
            raise ValueError("frontend rejects an unknown ROS camera kind")
        rows = record.get("cameras") or []
        if not isinstance(rows, list) or len(rows) > len(ROS_CAMERA_VIEW_ORDER):
            raise ValueError("frontend rejects invalid ROS camera rows")
        camera_root = camera_dir.resolve()
        frames: dict[str, RosCameraFrame] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("ROS camera rows must be objects")
            view_id = str(row.get("view_id") or "")
            if view_id not in ROS_CAMERA_VIEW_ORDER or view_id in frames:
                raise ValueError("frontend rejects an unknown or duplicate ROS camera")
            raw_path = str(row.get("jpeg_path") or "")
            jpeg_path = Path(raw_path).resolve()
            if (
                not raw_path
                or not jpeg_path.is_relative_to(camera_root)
                or jpeg_path.name != f"{view_id}.jpg"
            ):
                raise ValueError("ROS camera JPEG escapes the configured camera root")
            jpeg = jpeg_path.read_bytes()
            expected_sha = _require_sha256(
                row.get("jpeg_sha256"), f"ROS camera[{view_id}].jpeg_sha256"
            )
            if hashlib.sha256(jpeg).hexdigest() != expected_sha:
                raise ValueError(f"ROS camera[{view_id}] JPEG hash mismatch")
            width = _require_int(
                row.get("width"), f"ROS camera[{view_id}].width", nonnegative=True
            )
            height = _require_int(
                row.get("height"), f"ROS camera[{view_id}].height", nonnegative=True
            )
            if not 1 <= width <= 4096 or not 1 <= height <= 4096:
                raise ValueError("ROS camera dimensions are out of bounds")
            source_topic = str(row.get("source_topic") or "")
            if not source_topic.startswith("/") or len(source_topic) > 256:
                raise ValueError("ROS camera source_topic is invalid")
            frames[view_id] = RosCameraFrame(
                view_id=view_id,
                jpeg=jpeg,
                stamp_s=_require_finite(
                    row.get("stamp_s"),
                    f"ROS camera[{view_id}].stamp_s",
                    nonnegative=True,
                ),
                received_wall_time_s=_require_finite(
                    row.get("received_wall_time_s"),
                    f"ROS camera[{view_id}].received_wall_time_s",
                    nonnegative=True,
                ),
                source_topic=source_topic,
                width=width,
                height=height,
                encoding=str(row.get("encoding") or "jpeg")[:32],
            )
        with self._lock:
            self._ros_cameras = frames
            self._touch()

    def publish_ingest_warning(self, warning: str) -> None:
        normalized = str(warning).strip()
        if not normalized:
            return
        with self._lock:
            if not self._ingest_warnings or self._ingest_warnings[-1] != normalized:
                self._ingest_warnings = (self._ingest_warnings + [normalized])[-8:]
                self._touch()

    def camera(self, view_id: str) -> CameraFrame | RosCameraFrame | None:
        if view_id not in (*REV_C_VIEW_ORDER, *ROS_CAMERA_VIEW_ORDER):
            return None
        with self._lock:
            return self._cameras.get(view_id) or self._ros_cameras.get(view_id)

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "lane_id": LANE_ID,
                "version": self._version,
                "updated_wall_time_s": self._updated_wall_time_s,
                "health": sanitize_public(self._health),
                "gpu": sanitize_public(self._gpu),
                "ros": sanitize_public(self._ros),
                "ingest_warnings": list(self._ingest_warnings),
            }

    def _decision_matches_snapshot(self) -> bool:
        decision = self._decision.get("decision")
        if not isinstance(decision, Mapping) or not self._snapshot:
            return False
        return (
            decision.get("snapshot_id") == self._snapshot.get("snapshot_id")
            and decision.get("episode_id") == self._snapshot.get("episode_id")
        )

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            decision_matches_snapshot = self._decision_matches_snapshot()
            warnings = list(self._ingest_warnings)
            if self._decision and not decision_matches_snapshot:
                warnings = (warnings + ["decision_snapshot_identity_mismatch"])[-8:]
            cameras = [
                self._cameras[view_id].metadata()
                if view_id in self._cameras
                else self._ros_cameras[view_id].metadata()
                if view_id in self._ros_cameras
                else {
                    "view_id": view_id,
                    "available": False,
                    "url": f"/api/v1/cameras/{view_id}.jpg",
                }
                for view_id in REV_C_VIEW_ORDER
            ]
            live_cameras = [
                self._ros_cameras[view_id].metadata()
                if view_id in self._ros_cameras
                else {
                    "view_id": view_id,
                    "available": False,
                    "url": f"/api/v1/cameras/{view_id}.jpg",
                }
                for view_id in ROS_LIVE_CAMERA_VIEW_ORDER
            ]
            return {
                "schema_version": 1,
                "lane_id": LANE_ID,
                "version": self._version,
                "updated_wall_time_s": self._updated_wall_time_s,
                "health": sanitize_public(self._health),
                "gpu": sanitize_public(self._gpu),
                "latency": sanitize_public(
                    self._latency if decision_matches_snapshot else {}
                ),
                "decision": sanitize_public(
                    self._decision if decision_matches_snapshot else {}
                ),
                "snapshot": sanitize_public(self._snapshot),
                "cameras": cameras,
                "ros": sanitize_public(self._ros),
                "robot": sanitize_public(self._robot),
                "live_cameras": live_cameras,
                "ingest_warnings": warnings,
            }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_last_json_line(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        buffer = bytearray()
        while position > 0:
            position -= 1
            handle.seek(position)
            byte = handle.read(1)
            if byte == b"\n" and buffer:
                break
            if byte != b"\n":
                buffer.extend(byte)
        if not buffer:
            raise ValueError(f"{path} has no JSONL records")
        line = bytes(reversed(buffer)).decode("utf-8")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError(f"last record in {path} must be an object")
    return value


class FilesystemStateSource:
    """Project runtime artifacts into the operator panel state store."""

    def __init__(self, config: FrontendConfig) -> None:
        self.config = config
        self._mtimes: dict[Path, int] = {}

    def _changed(self, path: Path) -> bool:
        if not path.is_file():
            return False
        mtime = path.stat().st_mtime_ns
        if self._mtimes.get(path) == mtime:
            return False
        self._mtimes[path] = mtime
        return True

    def refresh(self, store: FrontendStateStore) -> None:
        readers: list[tuple[Path, Any]] = [
            (
                self.config.snapshot_sidecar_path,
                lambda path: store.publish_snapshot_record(
                    _read_last_json_line(path), self.config.camera_dir
                ),
            ),
            (
                self.config.decision_log_path,
                lambda path: store.publish_decision(_read_last_json_line(path)),
            ),
            (
                self.config.health_path,
                lambda path: store.publish_health(_read_json(path)),
            ),
            (
                self.config.gpu_telemetry_path,
                lambda path: store.publish_gpu(_read_json(path)),
            ),
        ]
        if self.config.ros_state_path is not None:
            readers.append(
                (
                    self.config.ros_state_path,
                    lambda path: store.publish_ros_state(_read_json(path)),
                )
            )
        if (
            self.config.ros_camera_manifest_path is not None
            and self.config.ros_camera_dir is not None
        ):
            readers.append(
                (
                    self.config.ros_camera_manifest_path,
                    lambda path: store.publish_ros_camera_manifest(
                        _read_json(path), self.config.ros_camera_dir
                    ),
                )
            )
        for path, publish in readers:
            try:
                if self._changed(path):
                    publish(path)
            except Exception as exc:
                store.publish_ingest_warning(f"{path.name}:{type(exc).__name__}:{exc}")
