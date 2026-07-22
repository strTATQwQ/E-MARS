from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from .config_loader import normalize_scene_to_robot_origin
from .semantic_navigation_benchmark import load_task_set


COMPLETION_EVENT = {
    "find": "target_track_confirmed",
    "pass": "landmark_passed",
    "enter": "region_entered",
    "approach": "target_within_stop_distance",
    "verify": "completion_verified",
    "ask": "clarification_received",
}


class SemanticMarkerJudgeCore:
    """Oracle-only evaluator for the instrumented semantic-marker upper bound."""

    def __init__(
        self,
        task: dict[str, Any],
        scene: dict[str, Any],
        episode_id: str,
        *,
        no_progress_timeout_sec: float = 8.0,
    ) -> None:
        self.no_progress_timeout_sec = float(no_progress_timeout_sec)
        self.reset(task, scene, episode_id)

    def reset(self, task: dict[str, Any], scene: dict[str, Any], episode_id: str) -> None:
        self.task = dict(task)
        self.scene = normalize_scene_to_robot_origin(scene)
        self.episode_id = str(episode_id)
        self.objects = {str(obj.get("id")): dict(obj) for obj in self.scene.get("objects", [])}
        runtime = task.get("semantic_runtime") if isinstance(task.get("semantic_runtime"), dict) else {}
        self.completion_policy = str(runtime.get("judge_completion_policy") or "proximity_v1")
        self.pass_reach_distance_m = float(runtime.get("pass_reach_distance_m", 1.20))
        self.pass_cross_track_max_m = float(runtime.get("pass_cross_track_max_m", 1.50))
        self.enter_reach_distance_m = float(runtime.get("enter_reach_distance_m", 1.00))
        self.enter_cross_track_max_m = float(runtime.get("enter_cross_track_max_m", 1.25))
        self.object_ids = {
            int(index): str(object_id)
            for index, object_id in (runtime.get("subgoal_object_ids") or {}).items()
            if str(index).isdigit()
        }
        self.active: dict[str, Any] | None = None
        self.completed_indices: list[int] = []
        self.recovered_indices: list[int] = []
        self.visible_hits = 0
        self.injected = False
        self.awaiting_recovery = False
        self.required_recovery = str((task.get("judge") or {}).get("required_recovery") or "")
        self.active_required_recovery = self.required_recovery
        self.observed_recovery = ""
        self.recovery_matched = not bool(self.required_recovery)
        self.recovery_attempted_indices: set[int] = set()
        self.semantic_errors = 0
        self.collision = False
        self.last_distance_m: float | None = None
        self.last_bearing_deg: float | None = None
        self.best_distance_m: float | None = None
        self.best_route_progress_m: float | None = None
        self.last_route_progress_m: float | None = None
        self.last_cross_track_m: float | None = None
        self.passed_completion_plane = False
        self.last_progress_monotonic = time.monotonic()
        self.started = time.monotonic()

    def accept_subgoal(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if str(payload.get("episode_id") or self.episode_id) != self.episode_id:
            self.semantic_errors += 1
            return None
        try:
            index = int(payload.get("subgoal_index"))
        except (TypeError, ValueError):
            self.semantic_errors += 1
            return None
        if index < 0 or index >= len(self.task.get("oracle_plan", [])):
            self.semantic_errors += 1
            return None
        expected = len(self.completed_indices)
        if index != expected:
            self.semantic_errors += 1
            return None
        expected_subgoal = dict(self.task.get("oracle_plan", [])[index])
        if str(payload.get("subgoal_type") or "").lower() != str(
            expected_subgoal.get("subgoal_type") or ""
        ).lower():
            self.semantic_errors += 1
            return None
        if not _semantic_target_matches(
            str(payload.get("target") or ""),
            str(expected_subgoal.get("target") or ""),
        ):
            self.semantic_errors += 1
            return None
        self.active = dict(payload) | {"subgoal_index": index}
        self.active_required_recovery = str(payload.get("recovery") or self.required_recovery or "stop")
        self.visible_hits = 0
        self.best_distance_m = None
        self.best_route_progress_m = None
        self.last_route_progress_m = None
        self.last_cross_track_m = None
        self.passed_completion_plane = False
        self.last_progress_monotonic = time.monotonic()
        judge = self.task.get("judge") if isinstance(self.task.get("judge"), dict) else {}
        injected_event = str(judge.get("injected_event") or "")
        if injected_event and not self.injected:
            self.injected = True
            self.awaiting_recovery = True
            return self._event(injected_event, index=index, injected=True)
        return None

    def observe_recovery(self, payload: dict[str, Any]) -> None:
        if not self.awaiting_recovery:
            return
        recovery = str(payload.get("recovery") or "")
        self.observed_recovery = recovery
        self.recovery_matched = recovery == self.active_required_recovery
        if not self.recovery_matched:
            self.semantic_errors += 1

    def observe_event(self, payload: dict[str, Any]) -> None:
        if str(payload.get("type") or "") != "semantic_recovery_completed" or not self.awaiting_recovery:
            return
        if self.active is None:
            self.semantic_errors += 1
            return
        recovery = str(payload.get("recovery") or "")
        if recovery != self.active_required_recovery or not self.recovery_matched:
            self.semantic_errors += 1
            return
        index = int(self.active["subgoal_index"])
        if index not in self.recovered_indices:
            self.recovered_indices.append(index)
        self.awaiting_recovery = False
        self.visible_hits = 0
        self.best_distance_m = None
        self.best_route_progress_m = None
        self.last_route_progress_m = None
        self.last_cross_track_m = None
        self.passed_completion_plane = False
        self.last_progress_monotonic = time.monotonic()

    def update(
        self,
        pose: list[float],
        *,
        linear_x: float = 0.0,
        angular_z: float = 0.0,
    ) -> dict[str, Any] | None:
        if self.active is None or self.awaiting_recovery:
            return None
        index = int(self.active["subgoal_index"])
        subgoal_type = str(self.active.get("subgoal_type") or "")
        object_id = self.object_ids.get(index, "")
        target = self.objects.get(object_id)
        distance = None
        bearing = None
        visible = False
        route_progress = None
        cross_track = None
        completion_plane = False
        if target is not None:
            target_pose = list(target.get("pose") or [0.0, 0.0, 0.0])
            dx = float(target_pose[0]) - float(pose[0])
            dy = float(target_pose[1]) - float(pose[1])
            distance = math.hypot(dx, dy)
            bearing = math.degrees(_wrap(math.atan2(dy, dx) - float(pose[2])))
            visible = distance <= 12.0 and abs(bearing) <= 45.0
            self.last_distance_m = distance
            self.last_bearing_deg = bearing
            if self.best_distance_m is None or distance <= self.best_distance_m - 0.15:
                self.best_distance_m = distance
                self.last_progress_monotonic = time.monotonic()
            route_progress, cross_track, completion_plane = self._route_geometry(index, pose, target_pose)
            self.last_route_progress_m = route_progress
            self.last_cross_track_m = cross_track
            self.passed_completion_plane = completion_plane
            if self.best_route_progress_m is None or route_progress >= self.best_route_progress_m + 0.15:
                self.best_route_progress_m = route_progress
                self.last_progress_monotonic = time.monotonic()

        completed, completion_method = self._completion_state(
            subgoal_type,
            distance=distance,
            visible=visible,
            completion_plane=completion_plane,
            cross_track=cross_track,
            linear_x=linear_x,
            angular_z=angular_z,
        )
        if completed and self.completion_policy == "route_progress_v2":
            return self._finish_subgoal(
                index,
                subgoal_type,
                object_id=object_id,
                distance=distance,
                bearing=bearing,
                completion_method=completion_method,
                route_progress=route_progress,
                cross_track=cross_track,
            )

        if (
            subgoal_type in {"pass", "enter", "approach"}
            and index not in self.recovery_attempted_indices
            and time.monotonic() - self.last_progress_monotonic >= self.no_progress_timeout_sec
        ):
            self.recovery_attempted_indices.add(index)
            self.awaiting_recovery = True
            self.recovery_matched = False
            return self._event(
                "no_progress_timeout",
                index=index,
                recovery=self.active_required_recovery,
                distance_m=None if distance is None else round(distance, 4),
                best_distance_m=None if self.best_distance_m is None else round(self.best_distance_m, 4),
                route_progress_m=None if route_progress is None else round(route_progress, 4),
                cross_track_m=None if cross_track is None else round(cross_track, 4),
                oracle_judge_only=True,
            )

        if not completed:
            return None
        return self._finish_subgoal(
            index,
            subgoal_type,
            object_id=object_id,
            distance=distance,
            bearing=bearing,
            completion_method=completion_method,
            route_progress=route_progress,
            cross_track=cross_track,
        )

    def _completion_state(
        self,
        subgoal_type: str,
        *,
        distance: float | None,
        visible: bool,
        completion_plane: bool,
        cross_track: float | None,
        linear_x: float,
        angular_z: float,
    ) -> tuple[bool, str]:
        if subgoal_type == "find":
            self.visible_hits = self.visible_hits + 1 if visible else 0
            return self.visible_hits >= 2, "two_visible_frames"
        if subgoal_type == "pass":
            if distance is not None and distance <= self.pass_reach_distance_m:
                return True, "pass_marker_reached"
            crossed = bool(
                self.completion_policy == "route_progress_v2"
                and completion_plane
                and cross_track is not None
                and cross_track <= self.pass_cross_track_max_m
            )
            return crossed, "pass_completion_plane"
        if subgoal_type == "enter":
            if distance is not None and distance <= self.enter_reach_distance_m:
                return True, "enter_marker_reached"
            crossed = bool(
                self.completion_policy == "route_progress_v2"
                and completion_plane
                and cross_track is not None
                and cross_track <= self.enter_cross_track_max_m
            )
            return crossed, "enter_completion_plane"
        if subgoal_type == "approach":
            stop_distance = float((self.task.get("success") or {}).get("distance_to_target_m", 2.0))
            return distance is not None and distance <= stop_distance, "stop_distance_reached"
        if subgoal_type == "verify":
            stop_distance = float((self.task.get("success") or {}).get("distance_to_target_m", 2.0))
            stopped = abs(float(linear_x)) <= 0.03 and abs(float(angular_z)) <= 0.03
            return distance is not None and distance <= stop_distance and stopped, "near_and_stopped"
        if subgoal_type == "ask":
            stopped = abs(float(linear_x)) <= 0.03 and abs(float(angular_z)) <= 0.03
            return stopped, "stationary_clarification"
        self.semantic_errors += 1
        return False, "unsupported_subgoal"

    def _finish_subgoal(
        self,
        index: int,
        subgoal_type: str,
        *,
        object_id: str,
        distance: float | None,
        bearing: float | None,
        completion_method: str,
        route_progress: float | None,
        cross_track: float | None,
    ) -> dict[str, Any]:
        self.completed_indices.append(index)
        self.active = None
        self.visible_hits = 0
        self.best_distance_m = None
        self.best_route_progress_m = None
        return self._event(
            COMPLETION_EVENT[subgoal_type],
            index=index,
            object_id=object_id,
            distance_m=None if distance is None else round(distance, 4),
            bearing_deg=None if bearing is None else round(bearing, 3),
            completion_method=completion_method,
            route_progress_m=None if route_progress is None else round(route_progress, 4),
            cross_track_m=None if cross_track is None else round(cross_track, 4),
            oracle_judge_only=True,
        )

    def _route_geometry(
        self,
        index: int,
        pose: list[float],
        target_pose: list[float],
    ) -> tuple[float, float, bool]:
        current_id = self.object_ids.get(index, "")
        origin = list(self.scene.get("robot_start_pose") or [0.0, 0.0, 0.0])
        for previous in reversed(range(index)):
            previous_id = self.object_ids.get(previous, "")
            if previous_id and previous_id != current_id and previous_id in self.objects:
                origin = list(self.objects[previous_id].get("pose") or origin)
                break
        dx = float(target_pose[0]) - float(origin[0])
        dy = float(target_pose[1]) - float(origin[1])
        route_length = math.hypot(dx, dy)
        if route_length <= 1e-6:
            dx, dy, route_length = 1.0, 0.0, 1.0
        ux, uy = dx / route_length, dy / route_length
        from_origin_x = float(pose[0]) - float(origin[0])
        from_origin_y = float(pose[1]) - float(origin[1])
        progress = from_origin_x * ux + from_origin_y * uy
        cross_track = abs(from_origin_x * (-uy) + from_origin_y * ux)
        return progress, cross_track, progress >= route_length

    def status(self, *, linear_x: float = 0.0, angular_z: float = 0.0) -> dict[str, Any]:
        expected = len(self.task.get("oracle_plan", []))
        all_subgoals = len(self.completed_indices) == expected
        stopped = abs(float(linear_x)) <= 0.03 and abs(float(angular_z)) <= 0.03
        success = bool(all_subgoals and stopped and not self.collision and self.semantic_errors == 0)
        return {
            "task_id": self.task.get("task_id"),
            "task_type": "semantic_navigation",
            "category": self.task.get("category"),
            "episode_id": self.episode_id,
            "success": success,
            "clean": success and self.semantic_errors == 0,
            "done": success or self.collision,
            "reason": "success" if success else ("collision" if self.collision else "running"),
            "completed_subgoal_indices": list(self.completed_indices),
            "recovered_subgoal_indices": list(self.recovered_indices),
            "subgoals_completed": len(self.completed_indices),
            "subgoals_expected": expected,
            "active_subgoal_index": None if self.active is None else self.active.get("subgoal_index"),
            "active_subgoal_type": None if self.active is None else self.active.get("subgoal_type"),
            "required_recovery": self.required_recovery or self.active_required_recovery,
            "observed_recovery": self.observed_recovery,
            "recovery_matched": self.recovery_matched,
            "awaiting_recovery": self.awaiting_recovery,
            "semantic_errors": self.semantic_errors,
            "collision": int(self.collision),
            "distance_to_active_marker_m": None
            if self.last_distance_m is None
            else round(self.last_distance_m, 4),
            "bearing_to_active_marker_deg": None
            if self.last_bearing_deg is None
            else round(self.last_bearing_deg, 3),
            "route_progress_m": None
            if self.last_route_progress_m is None
            else round(self.last_route_progress_m, 4),
            "cross_track_m": None
            if self.last_cross_track_m is None
            else round(self.last_cross_track_m, 4),
            "passed_completion_plane": self.passed_completion_plane,
            "judge_completion_policy": self.completion_policy,
            "time_sec": round(time.monotonic() - self.started, 3),
            "instrumented_semantic_markers": True,
            "geometry_judge_is_oracle_only": True,
            "qualification_evidence": False,
        }

    def _event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        return {
            "type": event_type,
            "episode_id": self.episode_id,
            "task_id": self.task.get("task_id"),
            "source": "semantic_marker_judge",
            **fields,
        }


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _semantic_target_matches(predicted: str, expected: str) -> bool:
    ignored = {"nearest", "the", "a", "an", "marker", "target", "operator", "alternative"}
    expected_tokens = {
        token for token in str(expected).lower().replace("-", " ").split() if token not in ignored
    }
    predicted_tokens = set(str(predicted).lower().replace("-", " ").split())
    return not expected_tokens or expected_tokens.issubset(predicted_tokens)


def _load_document(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def main(args=None) -> None:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import String

    class SemanticMarkerJudgeNode(Node):
        def __init__(self) -> None:
            super().__init__("semantic_marker_judge")
            self.declare_parameter("task_file", "")
            self.declare_parameter("scenes_file", "")
            task_file = str(self.get_parameter("task_file").value)
            scenes_file = str(self.get_parameter("scenes_file").value)
            self.tasks = {
                str(task["task_id"]): task
                for task in load_task_set(task_file, require_full=False)["tasks"]
            }
            scenes_doc = _load_document(scenes_file)
            self.scenes = {str(scene["scene_id"]): scene for scene in scenes_doc.get("scenes", [])}
            self.core: SemanticMarkerJudgeCore | None = None
            self.pose = [0.0, 0.0, 0.0]
            self.linear_x = 0.0
            self.angular_z = 0.0
            self.event_pub = self.create_publisher(String, "/mission/event_json", 10)
            self.status_pub = self.create_publisher(String, "/semantic_navigation/status_json", 10)
            self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
            self.create_subscription(String, "/benchmark/mode_json", self.on_mode, 10)
            self.create_subscription(String, "/semantic_executive/accepted_subgoal_json", self.on_subgoal, 10)
            self.create_subscription(String, "/semantic_executive/recovery_json", self.on_recovery, 10)
            self.create_subscription(String, "/mission/event_json", self.on_event, 20)
            self.create_subscription(String, "/isaac/ground_truth_pose", self.on_pose, 10)
            self.create_subscription(Twist, "/safe_cmd_vel", self.on_cmd, 10)
            self.create_subscription(String, "/isaac/collision_event", self.on_collision, 10)
            self.create_timer(0.2, self.tick)

        def on_mode(self, msg: Any) -> None:
            payload = _safe_json(msg.data)
            task_id = str(payload.get("task_id") or "")
            episode_id = str(payload.get("episode_id") or "")
            task = self.tasks.get(task_id)
            if task is None:
                self.core = None
                return
            runtime = task.get("semantic_runtime") if isinstance(task.get("semantic_runtime"), dict) else {}
            scene = self.scenes.get(str(runtime.get("scene_id") or ""))
            if scene is None:
                self.core = None
                return
            self.core = SemanticMarkerJudgeCore(task, scene, episode_id)
            self.pose = list(scene.get("robot_start_pose") or [0.0, 0.0, 0.0])
            self.linear_x = 0.0
            self.angular_z = 0.0
            self.publish_metric("semantic_marker_judge_reset", result="ready", task_id=task_id)

        def on_subgoal(self, msg: Any) -> None:
            if self.core is None:
                return
            event = self.core.accept_subgoal(_safe_json(msg.data))
            if event:
                self.publish_event(event)

        def on_recovery(self, msg: Any) -> None:
            if self.core is not None:
                self.core.observe_recovery(_safe_json(msg.data))

        def on_event(self, msg: Any) -> None:
            payload = _safe_json(msg.data)
            if self.core is not None and str(payload.get("source") or "") != "semantic_marker_judge":
                self.core.observe_event(payload)

        def on_pose(self, msg: Any) -> None:
            payload = _safe_json(msg.data)
            pose = payload.get("pose") or payload.get("robot_pose")
            if isinstance(pose, list) and len(pose) >= 3:
                self.pose = [float(pose[0]), float(pose[1]), float(pose[2])]

        def on_cmd(self, msg: Any) -> None:
            self.linear_x = float(msg.linear.x)
            self.angular_z = float(msg.angular.z)

        def on_collision(self, _msg: Any) -> None:
            if self.core is not None:
                self.core.collision = True

        def tick(self) -> None:
            if self.core is None:
                return
            event = self.core.update(
                self.pose,
                linear_x=self.linear_x,
                angular_z=self.angular_z,
            )
            if event:
                self.publish_event(event)
            status = self.core.status(linear_x=self.linear_x, angular_z=self.angular_z)
            self.status_pub.publish(String(data=json.dumps(status, ensure_ascii=False)))

        def publish_event(self, event: dict[str, Any]) -> None:
            payload = dict(event) | {"timestamp": time.time()}
            self.event_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            self.publish_metric(
                "semantic_marker_judge_event",
                result="published",
                semantic_event_type=payload.get("type"),
                subgoal_index=payload.get("index"),
            )

        def publish_metric(self, event_type: str, **fields: Any) -> None:
            payload = {"event": event_type, "timestamp": time.time(), "qualification_evidence": False, **fields}
            self.metric_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    rclpy.init(args=args)
    node = SemanticMarkerJudgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


if __name__ == "__main__":
    main()
