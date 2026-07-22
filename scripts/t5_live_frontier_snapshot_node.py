#!/usr/bin/env python3
"""Capture reachable Nav2 frontiers into the frozen private JSON schema.

This process is read-only on ROS: it creates subscriptions and a TF listener,
then atomically replaces one file.  It has no navigation or model publisher.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slow_planner.lane_b import LaneBSnapshotIdentity  # noqa: E402
from slow_planner.live_frontier_capture import (  # noqa: E402
    AtomicLiveFrontierWriter,
    CaptureIdentity,
    FrontierCaptureConfig,
    LiveFrontierCaptureCoordinator,
    LiveFrontierCaptureError,
    OccupancyGrid2D,
    Pose2D,
    resolve_lane_topic,
)


# The frozen InternVLA client publishes these two observations with absolute
# ROS names.  Lane isolation is provided by the per-lane ROS_DOMAIN_ID; the
# Nav2 costmaps remain namespace-relative.  Keeping this exception here (and
# out of the shared frontier schema) mirrors the production graph without
# adding a relay or publisher to this capture-only sidecar.
_ROOT_SCOPED_PRODUCTION_INPUTS = frozenset({"metadata", "odometry"})


def _resolve_runtime_topics(
    namespace: str, topics: dict[str, Any]
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for name, configured in topics.items():
        relative = str(configured)
        # Always run the lane-relative resolver first so the checked-in config
        # cannot use an absolute path or escape through '..'.
        lane_topic = resolve_lane_topic(namespace, relative)
        if name in _ROOT_SCOPED_PRODUCTION_INPUTS:
            resolved[name] = f"/{relative.strip('/')}"
        else:
            resolved[name] = lane_topic
    return resolved


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _status(kind: str, status: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": kind,
        "status": status,
        "motion_authority": "none",
        "terminal_stop_authority": "none",
        "goal_authority": "none",
        "model_request_authority": "none",
        "recorded_wall_time_s": time.time(),
        **fields,
    }


def _stamp_s(message: Any) -> float:
    return float(message.sec) + float(message.nanosec) / 1_000_000_000.0


def _yaw(quaternion: Any) -> float:
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _load_config(path: Path) -> tuple[FrontierCaptureConfig, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != "t5_live_frontier_capture":
        raise ValueError("capture config kind is invalid")
    extraction = value.get("extraction")
    if not isinstance(extraction, dict):
        raise ValueError("capture extraction config is missing")
    config = FrontierCaptureConfig(**extraction)
    return config, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "internnav_t5" / "live_frontier_capture.json",
    )
    parser.add_argument("--namespace", default="/b")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--dependency-check-only", action="store_true")
    args = parser.parse_args()
    if not args.dependency_check_only and args.output is None:
        parser.error("--output is required outside dependency-check mode")

    try:
        config, raw_config = _load_config(args.config.resolve())
        topics = raw_config["topics"]
        frames = raw_config["frames"]
        resolved_topics = _resolve_runtime_topics(args.namespace, topics)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        _atomic_json(
            args.status_output,
            _status(
                "t5_live_frontier_capture_dependency_status",
                "BLOCKED",
                blocker_code="INVALID_CAPTURE_CONFIG",
                detail=f"{type(exc).__name__}: {exc}"[:512],
            ),
        )
        return 75

    try:
        import rclpy
        from internvla_ros2_msgs.msg import ObservationMetadata
        from nav_msgs.msg import OccupancyGrid, Odometry
        from rclpy.duration import Duration
        from rclpy.executors import ExternalShutdownException
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
            qos_profile_sensor_data,
        )
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformException, TransformListener
    except ImportError as exc:
        _atomic_json(
            args.status_output,
            _status(
                "t5_live_frontier_capture_dependency_status",
                "BLOCKED",
                blocker_code="ROS_RUNTIME_DEPENDENCY_MISSING",
                detail=f"{type(exc).__name__}: {exc}"[:512],
                namespace=args.namespace,
                resolved_topics=resolved_topics,
            ),
        )
        return 75

    if args.dependency_check_only:
        _atomic_json(
            args.status_output,
            _status(
                "t5_live_frontier_capture_dependency_status",
                "PASS",
                blocker_code=None,
                namespace=args.namespace,
                resolved_topics=resolved_topics,
            ),
        )
        return 0

    assert args.output is not None

    class LiveFrontierCaptureNode(Node):
        def __init__(self) -> None:
            super().__init__(
                "nav2_live_frontier_capture",
                namespace=args.namespace,
                cli_args=[
                    "--ros-args",
                    "-p",
                    "use_sim_time:=true",
                    "-r",
                    "/tf:=tf",
                    "-r",
                    "/tf_static:=tf_static",
                ],
                use_global_arguments=False,
                parameter_overrides=[],
            )
            self._writer = AtomicLiveFrontierWriter(args.output)
            self._capture = LiveFrontierCaptureCoordinator(
                self._writer,
                source_node=self.get_fully_qualified_name(),
                config=config,
            )
            self._metadata: CaptureIdentity | None = None
            self._odometry: Any | None = None
            self._costmaps: dict[str, Any] = {}
            self._identity_epoch: tuple[str, int] | None = None
            self._snapshot_ready_count = 0
            self._last_ready_snapshot_id: str | None = None
            self._last_status_key: tuple[Any, ...] | None = None
            self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
            self._tf_listener = TransformListener(
                self._tf_buffer, self, spin_thread=False
            )
            reliable = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=4,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.create_subscription(
                ObservationMetadata,
                resolved_topics["metadata"],
                self._on_metadata,
                reliable,
            )
            self.create_subscription(
                Odometry,
                resolved_topics["odometry"],
                self._on_odometry,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                OccupancyGrid,
                resolved_topics["local_costmap"],
                lambda message: self._on_costmap("local_costmap", message),
                reliable,
            )
            self.create_subscription(
                OccupancyGrid,
                resolved_topics["global_costmap"],
                lambda message: self._on_costmap("global_costmap", message),
                reliable,
            )
            self.create_timer(0.1, self._tick)
            self._record_status("WAITING_FOR_IDENTITY", blocker_code="MISSING_IDENTITY")

        def _record_status(self, status: str, **fields: Any) -> None:
            fields = {
                "snapshot_ready_count": self._snapshot_ready_count,
                "identity_epoch": (
                    {
                        "episode_id": self._identity_epoch[0],
                        "reset_id": self._identity_epoch[1],
                    }
                    if self._identity_epoch is not None
                    else None
                ),
                **fields,
            }
            key = (status, json.dumps(fields, sort_keys=True, default=str))
            if key == self._last_status_key:
                return
            self._last_status_key = key
            _atomic_json(
                args.status_output,
                _status(
                    "t5_live_frontier_capture_runtime_status",
                    status,
                    namespace=args.namespace,
                    resolved_topics=resolved_topics,
                    **fields,
                ),
            )

        def _on_metadata(self, message: Any) -> None:
            try:
                ros_episode_id = str(message.episode_id)
                if not ros_episode_id.startswith("b::") or ros_episode_id.count("::") != 1:
                    raise LiveFrontierCaptureError(
                        "metadata episode is not Lane B identity",
                        code="INVALID_CAPTURE_IDENTITY",
                    )
                identity = LaneBSnapshotIdentity(
                    ros_episode_id.split("::", 1)[1],
                    int(message.reset_generation),
                    int(message.sequence_id),
                )
                captured = _stamp_s(message.sim_stamp)
                current = CaptureIdentity(
                    identity=identity,
                    ros_episode_id=ros_episode_id,
                    captured_sim_time_s=captured,
                    valid_until_sim_time_s=_stamp_s(message.valid_until),
                )
                epoch = (identity.episode_id, identity.reset_id)
                if epoch != self._identity_epoch:
                    # Sensor messages and TF do not carry the reset identity.
                    # A new episode/reset therefore opens a hard barrier: only
                    # inputs observed after its metadata may be rebound.
                    self._clear_sensor_inputs()
                self._capture.observe_identity(identity)
                self._identity_epoch = epoch
                self._metadata = current
            except (LiveFrontierCaptureError, TypeError, ValueError) as exc:
                # A regressed sequence/reset must never leave the last
                # identity's file publishable.  Keep the coordinator's
                # monotonic history, but remove all current publishable state.
                self._metadata = None
                self._capture.clear_for_missing_identity()
                self._clear_sensor_inputs()
                self._record_status(
                    "BLOCKED",
                    blocker_code=getattr(exc, "code", "INVALID_CAPTURE_IDENTITY"),
                    detail=str(exc)[:512],
                )
                return
            self._attempt_capture()

        def _clear_sensor_inputs(self) -> None:
            self._odometry = None
            self._costmaps.clear()
            self._tf_buffer.clear()

        def _on_odometry(self, message: Any) -> None:
            self._odometry = message
            self._attempt_capture()

        def _on_costmap(self, name: str, message: Any) -> None:
            self._costmaps[name] = message
            self._attempt_capture()

        @staticmethod
        def _pose_from_transform(transform: Any) -> Pose2D:
            return Pose2D(
                float(transform.transform.translation.x),
                float(transform.transform.translation.y),
                _yaw(transform.transform.rotation),
            )

        @staticmethod
        def _grid(message: Any) -> OccupancyGrid2D:
            return OccupancyGrid2D(
                width=int(message.info.width),
                height=int(message.info.height),
                resolution_m=float(message.info.resolution),
                origin_x_m=float(message.info.origin.position.x),
                origin_y_m=float(message.info.origin.position.y),
                origin_yaw_rad=_yaw(message.info.origin.orientation),
                frame_id=str(message.header.frame_id),
                stamp_sim_time_s=_stamp_s(message.header.stamp),
                data=tuple(int(value) for value in message.data),
            )

        def _attempt_capture(self) -> None:
            now = float(self.get_clock().now().nanoseconds) / 1_000_000_000.0
            self._capture.expire(sim_now_s=now)
            if self._metadata is None:
                self._capture.clear_for_missing_identity()
                return
            if self._odometry is None or not self._costmaps:
                self._capture.clear()
                self._record_status(
                    "WAITING_FOR_INPUT",
                    blocker_code=(
                        "MISSING_ODOMETRY" if self._odometry is None else "MISSING_COSTMAP"
                    ),
                    snapshot_id=self._metadata.identity.snapshot_id,
                )
                return
            identity_time = Time(
                nanoseconds=round(self._metadata.captured_sim_time_s * 1_000_000_000)
            )
            errors = []
            for source in ("local_costmap", "global_costmap"):
                message = self._costmaps.get(source)
                if message is None:
                    continue
                try:
                    grid = self._grid(message)
                    grid_transform = self._tf_buffer.lookup_transform(
                        grid.frame_id,
                        str(frames["base"]),
                        identity_time,
                    )
                    map_transform = self._tf_buffer.lookup_transform(
                        str(frames["map"]),
                        str(frames["base"]),
                        identity_time,
                    )
                    payload = self._capture.capture(
                        identity=self._metadata,
                        grid=grid,
                        robot_pose_in_grid_frame=self._pose_from_transform(grid_transform),
                        robot_pose_in_map_frame=self._pose_from_transform(map_transform),
                        odometry_sim_time_s=_stamp_s(self._odometry.header.stamp),
                        sim_now_s=now,
                    )
                except (LiveFrontierCaptureError, TransformException, TypeError, ValueError) as exc:
                    errors.append(
                        {
                            "source": source,
                            "code": getattr(exc, "code", "TF_OR_GRID_INVALID"),
                            "detail": str(exc)[:256],
                        }
                    )
                    continue
                snapshot_id = str(payload["snapshot_id"])
                if snapshot_id != self._last_ready_snapshot_id:
                    self._snapshot_ready_count += 1
                    self._last_ready_snapshot_id = snapshot_id
                self._record_status(
                    "SNAPSHOT_READY",
                    blocker_code=None,
                    source_costmap=resolved_topics[source],
                    snapshot_id=snapshot_id,
                    frontier_set_sha256=payload["frontier_set_sha256"],
                    candidate_frontier_count=len(payload["candidate_frontiers"]),
                )
                return
            self._capture.clear()
            self._record_status(
                "BLOCKED",
                blocker_code=(errors[-1]["code"] if errors else "MISSING_COSTMAP"),
                snapshot_id=self._metadata.identity.snapshot_id,
                source_errors=errors,
            )

        def _tick(self) -> None:
            self._attempt_capture()

        def close(self) -> None:
            self._capture.clear()

    rclpy.init()
    node = LiveFrontierCaptureNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Supervised SIGINT/SIGTERM is a normal sidecar shutdown.  The finally
        # block still clears the current snapshot before the process exits.
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
