from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


FORBIDDEN_CONTROLLER_FIELDS = frozenset(
    {
        "target_pose",
        "nav_target_pose",
        "expected_branch",
        "branch_polygon",
        "branch_polygons",
        "branch_membership",
        "oracle_visibility",
        "oracle_context",
    }
)


def wrap_to_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def instruction_visual_target(instruction: str) -> str:
    """Extract a visual noun phrase from the user instruction only."""

    text = " ".join(str(instruction or "").strip().lower().split())
    for pattern in (
        r"\btoward(?:s)?\s+the\s+(.+?)(?:[.,;]|$)",
        r"\bapproach\s+the\s+(.+?)(?:\s+and\s+stop|[.,;]|$)",
        r"\bfind\s+the\s+(.+?)(?:\s+and\s+stop|[.,;]|$)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def payload_oracle_fields(payload: Any, *, path: str = "") -> list[str]:
    """Return forbidden controller-input fields, including nested occurrences."""

    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{path}.{key}" if path else str(key)
            if str(key) in FORBIDDEN_CONTROLLER_FIELDS:
                found.append(child)
            found.extend(payload_oracle_fields(value, path=child))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(payload_oracle_fields(value, path=f"{path}[{index}]"))
    return found


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    frame_id: str = "isaac_front_camera"

    @classmethod
    def from_camera_info(cls, payload: dict[str, Any]) -> "CameraIntrinsics":
        matrix = payload.get("k") or payload.get("K") or []
        if not isinstance(matrix, list) or len(matrix) < 9:
            raise ValueError("camera info requires a 3x3 K matrix")
        width = int(payload.get("width", 0))
        height = int(payload.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError("camera info width and height must be positive")
        fx, fy = float(matrix[0]), float(matrix[4])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        return cls(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=float(matrix[2]),
            cy=float(matrix[5]),
            frame_id=str(payload.get("frame_id") or "isaac_front_camera"),
        )


def project_depth_pixel(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> dict[str, Any]:
    """Project an optical-frame pixel into camera and planar base coordinates."""

    depth = float(depth_m)
    if not math.isfinite(depth) or depth <= 0.0:
        raise ValueError("depth must be finite and positive")
    camera_x_right = (float(u) - intrinsics.cx) * depth / intrinsics.fx
    camera_y_down = (float(v) - intrinsics.cy) * depth / intrinsics.fy
    bearing = math.atan2(-camera_x_right, depth)
    return {
        "point_camera_xyz_m": [camera_x_right, camera_y_down, depth],
        "point_base_xy_m": [depth, -camera_x_right],
        "bearing_rad": bearing,
        "distance_m": math.hypot(depth, camera_x_right),
    }


def normalized_bbox_center(bbox: list[Any], intrinsics: CameraIntrinsics) -> tuple[float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox_xyxy_norm requires four values")
    x0, y0, x1, y1 = [float(value) for value in bbox]
    if not (0.0 <= x0 <= x1 <= 1.0 and 0.0 <= y0 <= y1 <= 1.0):
        raise ValueError("normalized bbox must be inside [0, 1]")
    return ((x0 + x1) * 0.5 * intrinsics.width, (y0 + y1) * 0.5 * intrinsics.height)


def validate_target_observation(payload: dict[str, Any]) -> dict[str, Any]:
    forbidden = payload_oracle_fields(payload)
    if forbidden:
        raise ValueError(f"oracle fields are forbidden in sensor observation: {forbidden}")
    result = dict(payload)
    result["episode_id"] = str(payload.get("episode_id") or "")
    result["target_id"] = str(payload.get("target_id") or "").strip().lower()
    result["frame_seq"] = int(payload.get("frame_seq"))
    result["source_stamp_sec"] = float(payload.get("source_stamp_sec"))
    result["visible"] = bool(payload.get("visible", False))
    result["confidence"] = clamp(float(payload.get("confidence", 0.0)), 0.0, 1.0)
    if result["visible"]:
        result["distance_m"] = float(payload.get("distance_m"))
        result["bearing_rad"] = float(payload.get("bearing_rad"))
        if result["distance_m"] <= 0.0 or not math.isfinite(result["distance_m"]):
            raise ValueError("visible observation requires positive finite distance_m")
        if not math.isfinite(result["bearing_rad"]):
            raise ValueError("visible observation requires finite bearing_rad")
    return result


@dataclass
class SpatialTargetTrackState:
    required_hits: int = 2
    min_confidence: float = 0.60
    retain_confidence: float = 0.25
    max_misses: int = 2
    max_age_sec: float = 3.0
    max_distance_jump_m: float = 1.0
    max_bearing_jump_deg: float = 25.0
    episode_id: str = ""
    target_id: str = ""
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    visible: bool = False
    confidence: float = 0.0
    distance_m: float | None = None
    bearing_rad: float | None = None
    observer_x_m: float | None = None
    observer_y_m: float | None = None
    observer_yaw_rad: float | None = None
    world_bearing_rad: float | None = None
    world_x_m: float | None = None
    world_y_m: float | None = None
    frame_seq: int | None = None
    first_seen_time: float | None = None
    last_seen_time: float | None = None
    lost_count: int = 0
    reacquired_count: int = 0
    duplicate_frames: int = 0
    out_of_order_frames: int = 0
    discontinuity_rejections: int = 0
    provisional_switches: int = 0
    selected_candidate_index: int | None = None
    bbox_xyxy_norm: list[float] | None = None
    mask_sha256: str = ""
    label: str = ""
    candidate_source: str = ""
    source_frames: list[int] = field(default_factory=list)

    def reset(self, *, episode_id: str, target_id: str) -> None:
        self.episode_id = str(episode_id or "")
        self.target_id = str(target_id or "").strip().lower()
        self.hits = 0
        self.misses = 0
        self.confirmed = False
        self.visible = False
        self.confidence = 0.0
        self.distance_m = None
        self.bearing_rad = None
        self.observer_x_m = None
        self.observer_y_m = None
        self.observer_yaw_rad = None
        self.world_bearing_rad = None
        self.world_x_m = None
        self.world_y_m = None
        self.frame_seq = None
        self.first_seen_time = None
        self.last_seen_time = None
        self.lost_count = 0
        self.reacquired_count = 0
        self.duplicate_frames = 0
        self.out_of_order_frames = 0
        self.discontinuity_rejections = 0
        self.provisional_switches = 0
        self.selected_candidate_index = None
        self.bbox_xyxy_norm = None
        self.mask_sha256 = ""
        self.label = ""
        self.candidate_source = ""
        self.source_frames = []

    def update(self, observation: dict[str, Any], *, timestamp: float) -> dict[str, Any]:
        value = validate_target_observation(self._select_candidate(observation))
        episode_id = value["episode_id"]
        target_id = value["target_id"]
        if episode_id != self.episode_id or target_id != self.target_id:
            self.reset(episode_id=episode_id, target_id=target_id)

        frame_seq = int(value["frame_seq"])
        if self.frame_seq is not None and frame_seq == self.frame_seq:
            self.duplicate_frames += 1
            return self.summary(timestamp=timestamp, update_result="duplicate_discarded")
        if self.frame_seq is not None and frame_seq < self.frame_seq:
            self.out_of_order_frames += 1
            return self.summary(timestamp=timestamp, update_result="out_of_order_discarded")
        self.frame_seq = frame_seq
        self.source_frames.append(frame_seq)
        self.source_frames = self.source_frames[-32:]

        was_confirmed = self.confirmed
        confidence_threshold = self.retain_confidence if self.confirmed else self.min_confidence
        accepted = bool(value["visible"] and value["confidence"] >= confidence_threshold)
        observer_x, observer_y, observer_yaw = self._observer_pose(value)
        observed_world_bearing = (
            wrap_to_pi(observer_yaw + float(value["bearing_rad"]))
            if accepted and observer_yaw is not None
            else None
        )
        observed_world_xy = (
            (
                observer_x + float(value["distance_m"]) * math.cos(observed_world_bearing),
                observer_y + float(value["distance_m"]) * math.sin(observed_world_bearing),
            )
            if accepted
            and observer_x is not None
            and observer_y is not None
            and observed_world_bearing is not None
            else None
        )
        if accepted and self.last_seen_time is not None and self.distance_m is not None and self.bearing_rad is not None:
            if observed_world_xy is not None and self.world_x_m is not None and self.world_y_m is not None:
                distance_jump = math.hypot(
                    observed_world_xy[0] - self.world_x_m,
                    observed_world_xy[1] - self.world_y_m,
                )
                bearing_jump = 0.0
            else:
                distance_jump = abs(float(value["distance_m"]) - self.distance_m)
            if observed_world_xy is None and observed_world_bearing is not None and self.world_bearing_rad is not None:
                bearing_jump = abs(math.degrees(wrap_to_pi(observed_world_bearing - self.world_bearing_rad)))
            elif observed_world_xy is None:
                bearing_jump = abs(math.degrees(wrap_to_pi(float(value["bearing_rad"]) - self.bearing_rad)))
            if distance_jump > self.max_distance_jump_m or bearing_jump > self.max_bearing_jump_deg:
                if self.confirmed:
                    accepted = False
                    self.discontinuity_rejections += 1
                else:
                    self.hits = 0
                    self.misses = 0
                    self.provisional_switches += 1
        self.visible = accepted
        self.confidence = value["confidence"]
        update_result = "miss"
        if accepted:
            self.hits += 1
            self.misses = 0
            self.distance_m = float(value["distance_m"])
            self.bearing_rad = float(value["bearing_rad"])
            self.observer_x_m = observer_x
            self.observer_y_m = observer_y
            self.observer_yaw_rad = observer_yaw
            self.world_bearing_rad = observed_world_bearing
            if observed_world_xy is not None:
                self.world_x_m, self.world_y_m = observed_world_xy
            self.selected_candidate_index = (
                int(value["selected_candidate_index"])
                if value.get("selected_candidate_index") is not None
                else None
            )
            bbox = value.get("bbox_xyxy_norm")
            self.bbox_xyxy_norm = [float(item) for item in bbox] if isinstance(bbox, list) else None
            self.mask_sha256 = str(value.get("mask_sha256") or "")
            self.label = str(value.get("label") or "")
            self.candidate_source = str(value.get("candidate_source") or "")
            if self.first_seen_time is None:
                self.first_seen_time = float(timestamp)
            self.last_seen_time = float(timestamp)
            if self.hits >= max(2, int(self.required_hits)):
                self.confirmed = True
            if self.confirmed and not was_confirmed and self.lost_count > 0:
                self.reacquired_count += 1
                update_result = "reacquired"
            else:
                update_result = "confirmed" if self.confirmed else "hit"
        else:
            self.hits = 0
            self.misses += 1
            if self.confirmed and self.misses > self.max_misses:
                self.confirmed = False
                self.lost_count += 1
                update_result = "lost"
        return self.summary(timestamp=timestamp, update_result=update_result)

    def summary(self, *, timestamp: float, update_result: str = "snapshot") -> dict[str, Any]:
        age = None if self.last_seen_time is None else max(0.0, float(timestamp) - self.last_seen_time)
        fresh = bool(age is not None and age <= self.max_age_sec)
        return {
            "episode_id": self.episode_id,
            "target_id": self.target_id,
            "hits": self.hits,
            "misses": self.misses,
            "confirmed": bool(self.confirmed and fresh),
            "visible": bool(self.visible and fresh),
            "confidence": round(self.confidence, 4),
            "distance_m": self.distance_m,
            "bearing_rad": self.bearing_rad,
            "observer_x_m": self.observer_x_m,
            "observer_y_m": self.observer_y_m,
            "observer_yaw_rad": self.observer_yaw_rad,
            "world_bearing_rad": self.world_bearing_rad,
            "world_x_m": self.world_x_m,
            "world_y_m": self.world_y_m,
            "frame_seq": self.frame_seq,
            "first_seen_time": self.first_seen_time,
            "last_seen_time": self.last_seen_time,
            "age_sec": None if age is None else round(age, 4),
            "fresh": fresh,
            "lost_count": self.lost_count,
            "reacquired_count": self.reacquired_count,
            "duplicate_frames": self.duplicate_frames,
            "out_of_order_frames": self.out_of_order_frames,
            "discontinuity_rejections": self.discontinuity_rejections,
            "provisional_switches": self.provisional_switches,
            "selected_candidate_index": self.selected_candidate_index,
            "bbox_xyxy_norm": self.bbox_xyxy_norm,
            "mask_sha256": self.mask_sha256,
            "label": self.label,
            "candidate_source": self.candidate_source,
            "source_frames": list(self.source_frames),
            "update_result": update_result,
            "source": "actual_sensor_spatial_track",
        }

    def _select_candidate(self, observation: dict[str, Any]) -> dict[str, Any]:
        payload = dict(observation)
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            return payload
        valid = [row for row in candidates if isinstance(row, dict) and row.get("visible")]
        if not valid:
            return payload
        observer_x, observer_y, observer_yaw = self._observer_pose(payload)

        def cost(candidate: dict[str, Any]) -> float:
            confidence = clamp(float(candidate.get("confidence", 0.0)), 0.0, 1.0)
            if not self.confirmed or self.distance_m is None or self.bearing_rad is None:
                return -confidence
            if (
                observer_x is not None
                and observer_y is not None
                and observer_yaw is not None
                and self.world_x_m is not None
                and self.world_y_m is not None
            ):
                candidate_world_bearing = wrap_to_pi(observer_yaw + float(candidate["bearing_rad"]))
                candidate_world_x = observer_x + float(candidate["distance_m"]) * math.cos(candidate_world_bearing)
                candidate_world_y = observer_y + float(candidate["distance_m"]) * math.sin(candidate_world_bearing)
                world_jump = math.hypot(candidate_world_x - self.world_x_m, candidate_world_y - self.world_y_m)
                return world_jump / max(0.1, self.max_distance_jump_m) - 0.25 * confidence
            distance_jump = abs(float(candidate["distance_m"]) - self.distance_m)
            if observer_yaw is not None and self.world_bearing_rad is not None:
                candidate_world = wrap_to_pi(observer_yaw + float(candidate["bearing_rad"]))
                bearing_jump_deg = abs(math.degrees(wrap_to_pi(candidate_world - self.world_bearing_rad)))
            else:
                bearing_jump_deg = abs(
                    math.degrees(wrap_to_pi(float(candidate["bearing_rad"]) - self.bearing_rad))
                )
            return (
                distance_jump / max(0.1, self.max_distance_jump_m)
                + bearing_jump_deg / max(1.0, self.max_bearing_jump_deg)
                - 0.25 * confidence
            )

        selected = min(valid, key=cost)
        payload.update(selected)
        payload["selected_candidate_index"] = selected.get("candidate_index")
        return payload

    @staticmethod
    def _observer_pose(payload: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
        pose = payload.get("observer_pose")
        if isinstance(pose, list) and len(pose) >= 3:
            try:
                return float(pose[0]), float(pose[1]), float(pose[2])
            except (TypeError, ValueError):
                pass
        try:
            yaw = payload.get("observer_yaw_rad")
            return None, None, float(yaw) if yaw is not None else None
        except (TypeError, ValueError):
            return None, None, None


def semantic_track_primitive(
    track: dict[str, Any],
    *,
    max_linear_mps: float = 0.20,
    max_yaw_radps: float = 0.30,
    stop_distance_m: float = 2.0,
    yaw_start_deg: float = 15.0,
    yaw_stop_deg: float = 8.0,
) -> dict[str, Any]:
    if not (track.get("confirmed") and track.get("fresh") and track.get("visible")):
        return {"primitive": "stop", "phase": "track_unavailable", "reason": "unconfirmed_or_stale_track"}
    distance = float(track["distance_m"])
    bearing = float(track["bearing_rad"])
    if distance <= float(stop_distance_m):
        return {
            "primitive": "stop",
            "phase": "await_step_stop",
            "reason": "confirmed_track_inside_stop_threshold",
            "distance_m": distance,
            "bearing_rad": bearing,
        }
    abs_deg = abs(math.degrees(bearing))
    orienting = abs_deg > float(yaw_stop_deg)
    linear = 0.0 if orienting else min(float(max_linear_mps), 0.20) * max(0.0, math.cos(bearing))
    angular = clamp(0.8 * bearing, -abs(float(max_yaw_radps)), abs(float(max_yaw_radps)))
    if abs_deg >= float(yaw_start_deg):
        linear = 0.0
    return {
        "primitive": "target_relative_approach",
        "phase": "orient" if linear == 0.0 else "approach",
        "linear_x_mps": round(linear, 4),
        "angular_z_radps": round(angular, 4),
        "max_linear_x_mps": min(abs(float(max_linear_mps)), 0.20),
        "max_yaw_rate_radps": abs(float(max_yaw_radps)),
        "distance_m": round(distance, 4),
        "bearing_rad": round(bearing, 6),
        "source": "sensor_only_semantic_planner",
    }


@dataclass
class SensorRouteController:
    turn_angle_deg: float = 75.0
    post_turn_forward_m: float = 1.0
    max_linear_mps: float = 0.20
    max_yaw_radps: float = 0.30
    yaw_tolerance_deg: float = 8.0
    approach_trigger_distance_m: float = 3.3
    max_approach_distance_m: float = 3.4
    target_stop_distance_m: float = 1.6
    min_turn_angle_deg: float = 30.0
    max_turn_angle_deg: float = 60.0
    obstacle_bypass_trigger_m: float = 1.85
    obstacle_bypass_turn_deg: float = 40.0
    obstacle_bypass_forward_m: float = 1.4
    obstacle_bypass_speed_mps: float = 0.15
    obstacle_bypass_min_advance_clearance_m: float = 1.35
    choice: str = ""
    episode_id: str = ""
    phase: str = "idle"
    start_pose: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    phase_start_pose: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    desired_yaw: float = 0.0
    avoidance_phase: str = ""
    avoidance_sign: float = 0.0
    avoidance_yaw: float = 0.0
    avoidance_start_pose: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    avoidance_cycles: int = 0
    avoidance_turn_extensions: int = 0

    def reset(self) -> None:
        self.choice = ""
        self.episode_id = ""
        self.phase = "idle"
        self.start_pose = [0.0, 0.0, 0.0]
        self.phase_start_pose = [0.0, 0.0, 0.0]
        self.desired_yaw = 0.0
        self.avoidance_phase = ""
        self.avoidance_sign = 0.0
        self.avoidance_yaw = 0.0
        self.avoidance_start_pose = [0.0, 0.0, 0.0]
        self.avoidance_cycles = 0
        self.avoidance_turn_extensions = 0

    def start(self, decision: dict[str, Any], pose: list[float]) -> None:
        if payload_oracle_fields(decision):
            raise ValueError("route controller decision contains oracle fields")
        choice = str(decision.get("route_choice") or "").lower()
        if choice not in {"left", "right", "front", "scan", "stop"}:
            raise ValueError(f"unsupported route choice: {choice}")
        self.choice = choice
        self.episode_id = str(decision.get("episode_id") or "")
        self.start_pose = [float(value) for value in pose[:3]]
        self.phase_start_pose = list(self.start_pose)
        sign = 1.0 if choice == "left" else -1.0 if choice == "right" else 0.0
        self.desired_yaw = wrap_to_pi(self.start_pose[2] + math.radians(sign * self.turn_angle_deg))
        self.avoidance_phase = ""
        self.avoidance_sign = 0.0
        self.avoidance_yaw = 0.0
        self.avoidance_start_pose = list(self.start_pose)
        self.avoidance_cycles = 0
        self.avoidance_turn_extensions = 0
        self.phase = (
            "approach_intersection"
            if choice in {"left", "right"}
            else "advance_into_branch"
            if choice == "front"
            else choice
        )

    def command(
        self,
        pose: list[float],
        free_space: dict[str, Any] | None = None,
        target_track: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = [float(value) for value in pose[:3]]
        sectors: dict[str, float | None] = {"left": None, "front": None, "right": None}
        if isinstance(free_space, dict):
            for name in sectors:
                try:
                    sectors[name] = float(free_space.get(name))
                except (TypeError, ValueError):
                    pass
        track = target_track if isinstance(target_track, dict) else {}
        track_valid = bool(track.get("confirmed") and track.get("fresh") and track.get("visible"))
        try:
            landmark_distance = float(track.get("distance_m")) if track_valid else None
            landmark_bearing = float(track.get("bearing_rad")) if track_valid else None
        except (TypeError, ValueError):
            landmark_distance, landmark_bearing, track_valid = None, None, False
        if self.phase in {"idle", "stop", "verify_branch", "done"}:
            return {"primitive": "stop", "phase": self.phase, "source": "sensor_only_route_planner"}
        if self.phase == "scan":
            return {"primitive": "look_around", "phase": "scan", "source": "sensor_only_route_planner"}
        choice_sign = 1.0 if self.choice == "left" else -1.0 if self.choice == "right" else 0.0
        if (
            self.phase == "approach_intersection"
            and track_valid
            and choice_sign
            and landmark_bearing * choice_sign < -math.radians(5.0)
        ):
            return {
                "primitive": "stop",
                "phase": "landmark_choice_mismatch",
                "reason": "visual_bearing_disagrees_with_step_route_choice",
                "landmark_bearing_rad": landmark_bearing,
                "source": "sensor_only_route_planner",
            }
        approach_distance = math.hypot(current[0] - self.start_pose[0], current[1] - self.start_pose[1])
        if self.phase == "approach_intersection":
            avoidance = self._avoidance_command(current, sectors)
            if avoidance is not None:
                avoidance["approach_distance_m"] = round(approach_distance, 4)
                return avoidance
            reached_visual_window = bool(
                track_valid and landmark_distance is not None and landmark_distance <= self.approach_trigger_distance_m
            )
            reached_odom_limit = approach_distance >= self.max_approach_distance_m
            if reached_visual_window or reached_odom_limit:
                turn_angle = abs(float(landmark_bearing or 0.0))
                turn_angle = clamp(
                    turn_angle,
                    math.radians(self.min_turn_angle_deg),
                    math.radians(self.max_turn_angle_deg),
                )
                self.desired_yaw = wrap_to_pi(current[2] + choice_sign * turn_angle)
                self.phase = "rotate_to_branch"
                self.phase_start_pose = list(current)
            else:
                front = sectors["front"]
                if front is not None and front < self.obstacle_bypass_trigger_m:
                    left = sectors["left"] or 0.0
                    right = sectors["right"] or 0.0
                    self.avoidance_sign = 1.0 if left > right else -1.0
                    self.avoidance_yaw = wrap_to_pi(
                        current[2] + self.avoidance_sign * math.radians(self.obstacle_bypass_turn_deg)
                    )
                    self.avoidance_start_pose = list(current)
                    self.avoidance_phase = "rotate_out"
                    self.avoidance_cycles += 1
                    self.avoidance_turn_extensions = 0
                    avoidance = self._avoidance_command(current, sectors)
                    if avoidance is not None:
                        avoidance["approach_distance_m"] = round(approach_distance, 4)
                        return avoidance
                return {
                    "primitive": "enter_branch",
                    "phase": "approach_intersection",
                    "linear_x_mps": min(abs(self.max_linear_mps), 0.20),
                    "angular_z_radps": round(clamp(0.8 * wrap_to_pi(self.start_pose[2] - current[2]), -0.18, 0.18), 4),
                    "free_space_m": sectors,
                    "landmark_distance_m": landmark_distance,
                    "landmark_bearing_rad": landmark_bearing,
                    "approach_distance_m": round(approach_distance, 4),
                    "source": "sensor_only_route_planner",
                }
        yaw_error = wrap_to_pi(self.desired_yaw - current[2])
        if self.phase == "rotate_to_branch" and abs(math.degrees(yaw_error)) <= self.yaw_tolerance_deg:
            self.phase = "advance_into_branch"
            self.phase_start_pose = list(current)
        moved = math.hypot(current[0] - self.phase_start_pose[0], current[1] - self.phase_start_pose[1])
        if self.phase == "advance_into_branch":
            if track_valid and landmark_distance is not None and landmark_distance <= self.target_stop_distance_m:
                self.phase = "verify_branch"
                return {
                    "primitive": "stop",
                    "phase": "verify_branch",
                    "reason": "actual_mask_depth_inside_route_target_threshold",
                    "landmark_distance_m": landmark_distance,
                    "source": "sensor_only_route_planner",
                }
            if moved >= self.post_turn_forward_m:
                return {
                    "primitive": "stop",
                    "phase": "route_progress_exhausted",
                    "reason": "target_not_reached_within_sensor_route_horizon",
                    "post_turn_distance_m": round(moved, 4),
                    "source": "sensor_only_route_planner",
                }
            if not track_valid:
                center_error = wrap_to_pi(self.desired_yaw - current[2])
                if abs(math.degrees(center_error)) <= self.yaw_tolerance_deg:
                    return {
                        "primitive": "stop",
                        "phase": "await_landmark_reacquire",
                        "reason": "hold_branch_heading_for_fresh_track",
                        "yaw_error_rad": round(center_error, 6),
                        "source": "sensor_only_route_planner",
                    }
                return {
                    "primitive": "enter_branch",
                    "phase": "reacquire_landmark",
                    "reason": "return_to_branch_heading_for_fresh_track",
                    "linear_x_mps": 0.0,
                    "angular_z_radps": round(
                        clamp(0.8 * center_error, -abs(self.max_yaw_radps), abs(self.max_yaw_radps)), 4
                    ),
                    "yaw_error_rad": round(center_error, 6),
                    "source": "sensor_only_route_planner",
                }
            yaw_error = float(landmark_bearing or 0.0)
        linear = 0.0 if self.phase == "rotate_to_branch" else min(abs(self.max_linear_mps), 0.20)
        if self.phase == "advance_into_branch" and abs(math.degrees(yaw_error)) > 18.0:
            linear = 0.06
        angular = clamp(0.9 * yaw_error, -abs(self.max_yaw_radps), abs(self.max_yaw_radps))
        return {
            "primitive": "enter_branch",
            "phase": self.phase,
            "route_choice": self.choice,
            "linear_x_mps": round(linear, 4),
            "angular_z_radps": round(angular, 4),
            "max_linear_x_mps": min(abs(self.max_linear_mps), 0.20),
            "max_yaw_rate_radps": abs(self.max_yaw_radps),
            "yaw_error_rad": round(yaw_error, 6),
            "post_turn_distance_m": round(moved, 4),
            "free_space_m": sectors,
            "landmark_distance_m": landmark_distance,
            "landmark_bearing_rad": landmark_bearing,
            "source": "sensor_only_route_planner",
        }

    def _avoidance_command(
        self,
        current: list[float],
        sectors: dict[str, float | None],
    ) -> dict[str, Any] | None:
        if not self.avoidance_phase:
            return None
        common = {
            "primitive": "enter_branch",
            "reason": "depth_gap_short_horizon_bypass",
            "free_space_m": sectors,
            "avoidance_sign": self.avoidance_sign,
            "avoidance_cycle": self.avoidance_cycles,
            "source": "sensor_only_route_planner",
        }
        if self.avoidance_phase == "rotate_out":
            yaw_error = wrap_to_pi(self.avoidance_yaw - current[2])
            if abs(math.degrees(yaw_error)) <= self.yaw_tolerance_deg:
                front = sectors.get("front")
                if (
                    front is not None
                    and front < self.obstacle_bypass_min_advance_clearance_m
                    and self.avoidance_turn_extensions < 3
                ):
                    self.avoidance_yaw = wrap_to_pi(
                        self.avoidance_yaw + self.avoidance_sign * math.radians(10.0)
                    )
                    self.avoidance_turn_extensions += 1
                    yaw_error = wrap_to_pi(self.avoidance_yaw - current[2])
                    return common | {
                        "phase": "avoid_obstacle_rotate",
                        "linear_x_mps": 0.0,
                        "angular_z_radps": round(
                            clamp(0.9 * yaw_error, -abs(self.max_yaw_radps), abs(self.max_yaw_radps)), 4
                        ),
                        "yaw_error_rad": round(yaw_error, 6),
                        "turn_extension": self.avoidance_turn_extensions,
                    }
                self.avoidance_phase = "advance"
                self.avoidance_start_pose = list(current)
            else:
                return common | {
                    "phase": "avoid_obstacle_rotate",
                    "linear_x_mps": 0.0,
                    "angular_z_radps": round(
                        clamp(0.9 * yaw_error, -abs(self.max_yaw_radps), abs(self.max_yaw_radps)), 4
                    ),
                    "yaw_error_rad": round(yaw_error, 6),
                }
        if self.avoidance_phase == "advance":
            moved = math.hypot(
                current[0] - self.avoidance_start_pose[0],
                current[1] - self.avoidance_start_pose[1],
            )
            if moved >= self.obstacle_bypass_forward_m:
                self.avoidance_phase = "rejoin"
                self.avoidance_yaw = self.start_pose[2]
            else:
                yaw_error = wrap_to_pi(self.avoidance_yaw - current[2])
                return common | {
                    "phase": "avoid_obstacle_advance",
                    "linear_x_mps": min(abs(self.obstacle_bypass_speed_mps), 0.20),
                    "angular_z_radps": round(clamp(0.7 * yaw_error, -0.16, 0.16), 4),
                    "bypass_distance_m": round(moved, 4),
                    "bypass_target_distance_m": round(self.obstacle_bypass_forward_m, 4),
                }
        if self.avoidance_phase == "rejoin":
            yaw_error = wrap_to_pi(self.avoidance_yaw - current[2])
            if abs(math.degrees(yaw_error)) <= self.yaw_tolerance_deg:
                self.avoidance_phase = ""
                self.avoidance_sign = 0.0
                self.avoidance_turn_extensions = 0
                return None
            return common | {
                "phase": "avoid_obstacle_rejoin",
                "linear_x_mps": 0.0,
                "angular_z_radps": round(
                    clamp(0.9 * yaw_error, -abs(self.max_yaw_radps), abs(self.max_yaw_radps)), 4
                ),
                "yaw_error_rad": round(yaw_error, 6),
            }
        return None
