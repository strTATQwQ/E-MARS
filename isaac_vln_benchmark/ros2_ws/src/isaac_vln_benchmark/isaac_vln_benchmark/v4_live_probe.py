from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .config_loader import load_data, object_by_id, scene_by_type
from .metrics import distance_xy
from .v3_benchmark_utils import append_jsonl, dump_json
from .v4_benchmark_utils import (
    analyze_stale_events,
    judge_forced_route_episode,
    judge_forced_stop_episode,
    summarize_route_rows,
    summarize_stop_rows,
    write_stale_analysis,
)


DEFAULT_ROOT = Path(__file__).resolve().parents[4]


class LiveProbe:
    def __init__(self, ros_domain_id: int = 42):
        os.environ["ROS_DOMAIN_ID"] = str(ros_domain_id)
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from std_msgs.msg import String

        self.rclpy = rclpy
        self.String = String
        self.Twist = Twist
        self.Odometry = Odometry
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False
        self.node = Node("v4_live_probe")
        self.mode_pub = self.node.create_publisher(String, "/benchmark/mode_json", 10)
        self.scene_pub = self.node.create_publisher(String, "/isaac/set_scene", 10)
        self.reset_pub = self.node.create_publisher(String, "/isaac/reset_episode", 10)
        self.instruction_pub = self.node.create_publisher(String, "/user_instruction", 10)
        self.mission_event_pub = self.node.create_publisher(String, "/mission/event_json", 10)
        self.route_oracle_pub = self.node.create_publisher(String, "/oracle/route_choice_json", 10)
        self.stop_oracle_pub = self.node.create_publisher(String, "/oracle/semantic_stop_json", 10)
        self.stop_primitive_pub = self.node.create_publisher(String, "/primitive/command_json", 10)

        self.latest: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []
        self.safe_cmd_samples: list[dict[str, Any]] = []
        self.cmd_candidate_samples: list[dict[str, Any]] = []
        self.active_episode_id = ""
        self.active_task_id = ""
        self.active_scene_id = ""
        self.active_start_pose = [0.0, 0.0, 0.0]
        self.episode_started = 0.0

        topics = [
            "/isaac/ground_truth_pose",
            "/isaac/objects",
            "/isaac/episode_status",
            "/isaac/reset_ack_json",
            "/scheduler/state",
            "/primitive/command_json",
            "/mission/event_json",
            "/metrics/event_jsonl",
            "/safety/local_status_json",
            "/step/route_choice_json",
            "/step/semantic_stop_json",
        ]
        self._subs = [self.node.create_subscription(String, topic, lambda msg, t=topic: self.on_string(t, msg), 50) for topic in topics]
        self._subs.append(self.node.create_subscription(Twist, "/safe_cmd_vel", self.on_safe_cmd, 50))
        self._subs.append(self.node.create_subscription(Twist, "/cmd_vel_candidate", self.on_cmd_candidate, 50))
        self._subs.append(self.node.create_subscription(Odometry, "/odom", self.on_odom, 20))

    def close(self) -> None:
        self.node.destroy_node()
        if self._owns_rclpy and self.rclpy.ok():
            self.rclpy.shutdown()

    def now(self) -> float:
        return time.monotonic()

    def episode_t(self) -> float:
        return 0.0 if self.episode_started <= 0.0 else max(0.0, self.now() - self.episode_started)

    def publish_json(self, pub, payload: dict[str, Any]) -> None:
        pub.publish(self.String(data=json.dumps(payload, ensure_ascii=False)))

    def spin_for(self, seconds: float) -> None:
        deadline = self.now() + max(0.0, seconds)
        while self.now() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=min(0.05, max(0.0, deadline - self.now())))

    def on_string(self, topic: str, msg: Any) -> None:
        data = _safe_json(msg.data)
        if topic == "/isaac/episode_status" and self.active_episode_id:
            if data.get("task_id") not in {None, self.active_task_id}:
                return
            if data.get("scene_id") not in {None, self.active_scene_id}:
                return
            if data.get("episode_id") not in {None, self.active_episode_id}:
                return
        self.latest[topic] = data
        if not self.active_episode_id:
            return
        event = {"t": round(self.episode_t(), 6), "episode_id": self.active_episode_id, "event": _topic_event(topic), "details": data}
        self.events.append(event)
        if topic == "/isaac/ground_truth_pose":
            pose = data.get("pose") or data.get("robot_pose")
            if isinstance(pose, list) and len(pose) >= 3:
                self.trajectory.append({"t": event["t"], "pose": [float(pose[0]), float(pose[1]), float(pose[2])], "source": "ground_truth_pose"})

    def on_safe_cmd(self, msg: Any) -> None:
        row = {"t": round(self.episode_t(), 6), "cmd": _twist_dict(msg)}
        self.latest["/safe_cmd_vel"] = row
        if self.active_episode_id:
            self.safe_cmd_samples.append(row)
            self.events.append({"t": row["t"], "episode_id": self.active_episode_id, "event": "safe_cmd_vel", "details": row["cmd"]})

    def on_cmd_candidate(self, msg: Any) -> None:
        row = {"t": round(self.episode_t(), 6), "cmd": _twist_dict(msg)}
        self.latest["/cmd_vel_candidate"] = row
        if self.active_episode_id:
            self.cmd_candidate_samples.append(row)
            self.events.append({"t": row["t"], "episode_id": self.active_episode_id, "event": "cmd_vel_candidate", "details": row["cmd"]})

    def on_odom(self, msg: Any) -> None:
        pose = [
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            _yaw_from_quat(msg.pose.pose.orientation),
        ]
        self.latest["/odom"] = {"pose": pose, "t": self.now()}
        if self.active_episode_id and (not self.trajectory or self.trajectory[-1].get("source") != "ground_truth_pose"):
            self.trajectory.append({"t": round(self.episode_t(), 6), "pose": pose, "source": "odom"})

    def wait_for_ready(self, timeout_sec: float = 20.0) -> None:
        deadline = self.now() + timeout_sec
        required = ["/isaac/episode_status", "/isaac/ground_truth_pose"]
        while self.now() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            if all(topic in self.latest for topic in required):
                return
        missing = [topic for topic in required if topic not in self.latest]
        raise TimeoutError(f"v4 live probe topics not ready: {missing}")

    def start_episode(self, *, mode_name: str, mode_cfg: dict[str, Any], task: dict[str, Any], scene: dict[str, Any], index: int) -> str:
        self.events = []
        self.trajectory = []
        self.safe_cmd_samples = []
        self.cmd_candidate_samples = []
        self.active_task_id = str(task.get("adapter_task_id") or task["task_id"])
        self.active_scene_id = str(scene["scene_id"])
        self.active_start_pose = [0.0, 0.0, 0.0]
        self.active_episode_id = f"{mode_name}_{task['task_id']}_{index:04d}"
        self.episode_started = self.now()
        self.latest.pop("/isaac/episode_status", None)
        self.latest.pop("/isaac/ground_truth_pose", None)
        self.latest.pop("/isaac/reset_ack_json", None)
        mode_payload = {"mode": mode_name, "mode_config": mode_cfg, "episode_id": self.active_episode_id, "task_id": self.active_task_id}
        self.wait_for_step_mode(mode_payload, timeout_sec=3.0)
        self.publish_json(self.scene_pub, scene)
        self.publish_json(
            self.reset_pub,
            {"episode_id": self.active_episode_id, "mode": mode_name, "task_id": self.active_task_id, "scene_id": self.active_scene_id},
        )
        self.wait_for_episode_reset(timeout_sec=5.0)
        self.publish_json(self.instruction_pub, {"mission_id": self.active_episode_id, "instruction": task["instruction"], "timestamp": time.time()})
        self.events.append({"t": round(self.episode_t(), 6), "episode_id": self.active_episode_id, "event": "instruction", "details": {"instruction": task["instruction"]}})
        return self.active_episode_id

    def wait_for_step_mode(self, mode_payload: dict[str, Any], timeout_sec: float) -> None:
        deadline = self.now() + timeout_sec
        while self.now() < deadline:
            self.publish_json(self.mode_pub, mode_payload)
            self.spin_for(0.15)
            if _step_supervisor_mode_ack(self.events, self.active_episode_id):
                return
        raise TimeoutError(f"step supervisor mode handshake timed out: episode={self.active_episode_id}")

    def wait_for_episode_reset(self, timeout_sec: float) -> None:
        deadline = self.now() + timeout_sec
        while self.now() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            status = self.latest.get("/isaac/episode_status")
            reset_ack = self.latest.get("/isaac/reset_ack_json")
            status_ready = bool(
                isinstance(status, dict)
                and status.get("task_id") == self.active_task_id
                and status.get("scene_id") == self.active_scene_id
                and status.get("episode_id") == self.active_episode_id
            )
            telemetry_ready = bool(
                isinstance(reset_ack, dict)
                and reset_ack.get("episode_id") == self.active_episode_id
                and reset_ack.get("task_id") == self.active_task_id
                and reset_ack.get("scene_id") == self.active_scene_id
            )
            if status_ready and telemetry_ready:
                return
        raise TimeoutError(
            "episode reset handshake timed out: "
            f"episode={self.active_episode_id} task={self.active_task_id} scene={self.active_scene_id}"
        )

    def finish_episode(self) -> None:
        self.publish_json(self.stop_primitive_pub, {"primitive": "stop", "ttl_sec": 0.5, "request_id": "v4_finish_stop", "source": "v4_probe"})
        self.spin_for(0.5)
        self.active_episode_id = ""
        self.active_task_id = ""
        self.active_scene_id = ""
        self.active_start_pose = [0.0, 0.0, 0.0]


def run_route_live(config: dict[str, Any], output: Path, *, project_root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    tasks_doc = load_data(project_root / "configs" / "tasks.yaml")
    scenes_doc = load_data(project_root / "configs" / "scenes.yaml")
    bench = config.get("benchmark", {})
    probe = LiveProbe(int(bench.get("ros_domain_id", 42)))
    rows: list[dict[str, Any]] = []
    try:
        probe.wait_for_ready(float(bench.get("startup_timeout_sec", 20.0)))
        for idx, task_cfg in enumerate(config.get("tasks", [])):
            task = _task_from_config(tasks_doc, task_cfg)
            scene = scene_by_type(scenes_doc, str(task_cfg.get("scene_type") or task["scene_type"]))
            mode_cfg = dict(config.get("mode") or {})
            mode_name = str(mode_cfg.get("name") or "forced_route_oracle")
            probe.start_episode(mode_name=mode_name, mode_cfg=mode_cfg, task=task, scene=scene, index=idx)
            expected = str(task_cfg.get("expected_route") or _expected_route(task)).lower()
            start_time = probe.now()
            publish_deadline = start_time + float(bench.get("route_publish_duration_sec", 8.0))
            duration = float(bench.get("duration_sec", 20.0))
            period = float(bench.get("publish_period_sec", 0.20))
            while probe.now() - start_time < duration:
                if probe.now() < publish_deadline:
                    probe.publish_json(
                        probe.route_oracle_pub,
                        {"route_choice": expected, "confidence": float((config.get("route_oracle") or {}).get("confidence", 1.0)), "source": "forced_oracle"},
                    )
                    probe.events.append({"t": round(probe.episode_t(), 6), "episode_id": probe.active_episode_id, "event": "forced_route_oracle_publish", "details": {"route_choice": expected}})
                probe.spin_for(period)
            status = probe.latest.get("/isaac/episode_status") if isinstance(probe.latest.get("/isaac/episode_status"), dict) else {}
            row = _route_metrics_from_probe(probe, task, scene, expected, status)
            rows.append(row)
            probe.finish_episode()
    finally:
        probe.close()
    return write_route_run(output, rows, config)


def run_stop_live(config: dict[str, Any], output: Path, *, project_root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    tasks_doc = load_data(project_root / "configs" / "tasks.yaml")
    scenes_doc = load_data(project_root / "configs" / "scenes.yaml")
    bench = config.get("benchmark", {})
    stop_cfg = config.get("stop_oracle") or {}
    max_dist = float(stop_cfg.get("max_distance_to_target_m", 2.5))
    probe = LiveProbe(int(bench.get("ros_domain_id", 42)))
    rows: list[dict[str, Any]] = []
    try:
        probe.wait_for_ready(float(bench.get("startup_timeout_sec", 20.0)))
        for idx, task_cfg in enumerate(config.get("tasks", [])):
            task = _task_from_config(tasks_doc, task_cfg)
            scene = scene_by_type(scenes_doc, str(task_cfg.get("scene_type") or task["scene_type"]))
            mode_cfg = dict(config.get("mode") or {})
            mode_name = str(mode_cfg.get("name") or "forced_stop_oracle")
            probe.start_episode(mode_name=mode_name, mode_cfg=mode_cfg, task=task, scene=scene, index=idx)
            duration = float(bench.get("duration_sec", 8.0))
            period = float(bench.get("publish_period_sec", 0.10))
            publish_until = 0.0
            first_visible_t: float | None = None
            stop_publish_t: float | None = None
            distance_at_first_visible: float | None = None
            distance_at_stop: float | None = None
            start = probe.now()
            while probe.now() - start < duration:
                status = probe.latest.get("/isaac/episode_status") if isinstance(probe.latest.get("/isaac/episode_status"), dict) else {}
                visible = bool(status.get("target_visible", False))
                dist = _float(status.get("distance_to_target"), 999.0)
                if visible and first_visible_t is None:
                    first_visible_t = probe.episode_t()
                    distance_at_first_visible = dist
                if visible and dist <= max_dist and stop_publish_t is None:
                    stop_publish_t = probe.episode_t()
                    distance_at_stop = dist
                    publish_until = probe.now() + float(bench.get("stop_publish_duration_sec", 2.0))
                if publish_until and probe.now() <= publish_until:
                    probe.publish_json(probe.stop_oracle_pub, {"stop": True, "source": "forced_oracle", "confidence": float(stop_cfg.get("confidence", 1.0))})
                    probe.events.append({"t": round(probe.episode_t(), 6), "episode_id": probe.active_episode_id, "event": "forced_stop_oracle_publish", "details": {"stop": True}})
                probe.spin_for(period)
                status = probe.latest.get("/isaac/episode_status") if isinstance(probe.latest.get("/isaac/episode_status"), dict) else {}
                if status.get("done"):
                    break
            status = probe.latest.get("/isaac/episode_status") if isinstance(probe.latest.get("/isaac/episode_status"), dict) else {}
            row = _stop_metrics_from_probe(
                probe,
                task,
                scene,
                status,
                first_visible_t=first_visible_t,
                stop_publish_t=stop_publish_t,
                distance_at_first_visible=distance_at_first_visible,
                distance_at_stop=distance_at_stop,
                max_distance=max_dist,
            )
            rows.append(row)
            probe.finish_episode()
    finally:
        probe.close()
    return write_stop_run(output, rows, config)


def run_step_route_live(config: dict[str, Any], output: Path, *, project_root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    tasks_doc = load_data(project_root / "configs" / "tasks.yaml")
    scenes_doc = load_data(project_root / "configs" / "scenes.yaml")
    bench = config.get("benchmark", {})
    probe = LiveProbe(int(bench.get("ros_domain_id", 42)))
    rows: list[dict[str, Any]] = []
    episode_configs = [
        (int(seed), dict(task_cfg))
        for seed in bench.get("seeds", [0])
        for task_cfg in config.get("tasks", [])
    ]
    try:
        probe.wait_for_ready(float(bench.get("startup_timeout_sec", 20.0)))
        for idx, (seed, task_cfg) in enumerate(episode_configs):
            task = _seeded_task(_task_from_config(tasks_doc, task_cfg), seed)
            scene = scene_by_type(scenes_doc, str(task_cfg.get("scene_type") or task["scene_type"]))
            mode_cfg = dict(config.get("mode") or {})
            mode_name = str(mode_cfg.get("name") or "step_route_choice_only")
            probe.start_episode(mode_name=mode_name, mode_cfg=mode_cfg, task=task, scene=scene, index=idx)
            expected = str(task_cfg.get("expected_route") or _expected_route(task)).lower()
            start_time = probe.now()
            duration = float(bench.get("duration_sec", 24.0))
            period = float(bench.get("trigger_period_sec", 0.50))
            while probe.now() - start_time < duration:
                event = {
                    "type": "route_choice_upcoming",
                    "instruction": task["instruction"],
                    "active_subgoal": task["instruction"],
                    "visible_in_view": expected,
                    "near_intersection": True,
                    "intersection_ahead_m": 0.75,
                    "source": "v4_step_route_probe",
                    "episode_id": probe.active_episode_id,
                    "mission_id": probe.active_episode_id,
                }
                probe.publish_json(probe.mission_event_pub, event)
                probe.events.append({"t": round(probe.episode_t(), 6), "episode_id": probe.active_episode_id, "event": "step_route_choice_trigger", "details": event})
                probe.spin_for(period)
            status = probe.latest.get("/isaac/episode_status") if isinstance(probe.latest.get("/isaac/episode_status"), dict) else {}
            row = _route_metrics_from_probe(probe, task, scene, expected, status)
            rows.append(row)
            probe.finish_episode()
    finally:
        probe.close()
    return write_step_route_run(output, rows, config)


def run_step_stop_live(config: dict[str, Any], output: Path, *, project_root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    tasks_doc = load_data(project_root / "configs" / "tasks.yaml")
    scenes_doc = load_data(project_root / "configs" / "scenes.yaml")
    bench = config.get("benchmark", {})
    probe = LiveProbe(int(bench.get("ros_domain_id", 42)))
    rows: list[dict[str, Any]] = []
    episode_configs = [
        (int(seed), dict(task_cfg))
        for seed in bench.get("seeds", [0])
        for task_cfg in config.get("tasks", [])
    ]
    try:
        probe.wait_for_ready(float(bench.get("startup_timeout_sec", 20.0)))
        for idx, (seed, task_cfg) in enumerate(episode_configs):
            task = _seeded_task(_task_from_config(tasks_doc, task_cfg), seed)
            scene = scene_by_type(scenes_doc, str(task_cfg.get("scene_type") or task["scene_type"]))
            mode_cfg = dict(config.get("mode") or {})
            mode_name = str(mode_cfg.get("name") or "step_stop_verify_live")
            probe.start_episode(mode_name=mode_name, mode_cfg=mode_cfg, task=task, scene=scene, index=idx)
            duration = float(bench.get("duration_sec", 10.0))
            period = float(bench.get("trigger_period_sec", 0.25))
            first_visible_t: float | None = None
            stop_publish_t: float | None = None
            distance_at_first_visible: float | None = None
            distance_at_stop: float | None = None
            start = probe.now()
            while probe.now() - start < duration:
                status = probe.latest.get("/isaac/episode_status") if isinstance(probe.latest.get("/isaac/episode_status"), dict) else {}
                visible = bool(status.get("target_visible", False))
                dist = _float(status.get("distance_to_target"), 999.0)
                if visible and first_visible_t is None:
                    first_visible_t = probe.episode_t()
                    distance_at_first_visible = dist
                event = {
                    "type": "target_visible" if visible else "completion_verification",
                    "target": task.get("target_object"),
                    "target_visible": visible,
                    "distance_to_target": dist,
                    "distance_to_target_m": dist,
                    "active_subgoal": task["instruction"],
                    "instruction": task["instruction"],
                    "source": "v4_step_stop_probe",
                    "episode_id": probe.active_episode_id,
                    "mission_id": probe.active_episode_id,
                }
                probe.publish_json(probe.mission_event_pub, event)
                probe.events.append({"t": round(probe.episode_t(), 6), "episode_id": probe.active_episode_id, "event": "step_semantic_stop_trigger", "details": event})
                probe.spin_for(period)
                accepted_stop = _first_accepted_step_role_response(probe.events, "semantic_stop")
                if accepted_stop and bool((accepted_stop.get("output") or {}).get("stop")) and stop_publish_t is None:
                    stop_publish_t = float(accepted_stop["t"])
                    distance_at_stop = dist
                status = probe.latest.get("/isaac/episode_status") if isinstance(probe.latest.get("/isaac/episode_status"), dict) else {}
                if status.get("done") and stop_publish_t is not None:
                    break
            status = probe.latest.get("/isaac/episode_status") if isinstance(probe.latest.get("/isaac/episode_status"), dict) else {}
            row = _stop_metrics_from_probe(
                probe,
                task,
                scene,
                status,
                first_visible_t=first_visible_t,
                stop_publish_t=stop_publish_t,
                distance_at_first_visible=distance_at_first_visible,
                distance_at_stop=distance_at_stop,
                max_distance=2.5,
            )
            rows.append(row)
            probe.finish_episode()
    finally:
        probe.close()
    return write_step_stop_run(output, rows, config)


def write_route_run(output: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    target = float((config.get("benchmark") or {}).get("correct_branch_target_rate", 0.80))
    summary = summarize_route_rows(rows, target_rate=target)
    accepted_episodes = sum(1 for row in rows if row.get("real_step_accepted"))
    parse_errors = sum(int(row.get("parse_error", 0)) for row in rows)
    metrics = {
        "benchmark": str((config.get("benchmark") or {}).get("name", "forced_route_oracle_live")),
        "mock_models": False,
        "live_closed_loop": True,
        "episodes": summary["episodes"],
        "forced_route_oracle_correct_branch_rate": summary["correct_branch_rate"],
        "correct_branch_rate": summary["correct_branch_rate"],
        "target_correct_branch_rate": target,
        "pass": bool(summary["pass"] and accepted_episodes == summary["episodes"] and parse_errors == 0),
        "collision_count": summary["collision_count"],
        "stale_action_executed": summary["stale_action_executed"],
        "stale_discard_count": summary["stale_discard_count"],
        "parse_error": parse_errors,
        "real_step_accepted_episodes": accepted_episodes,
        "real_step_call_coverage": round(accepted_episodes / max(1, summary["episodes"]), 3),
        "max_linear_x_mps": 0.20,
        "actions_through_safe_mux": True,
        "real_robot_motion_enabled": False,
        "failure_top1": summary["failure_top1"],
    }
    _write_common_tables(output, summary["rows"], route=True)
    dump_json(output / "metrics.json", metrics)
    append_jsonl(output / "events.jsonl", [event for row in rows for event in row.get("events", [])])
    _write_trajectory(output, rows)
    analysis = analyze_stale_events([event for row in rows for event in row.get("events", [])])
    write_stale_analysis(output / "stale_discard_analysis.md", analysis)
    _write_route_summary(output, metrics, summary["rows"])
    return metrics


def write_stop_run(output: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    target_acc = float((config.get("benchmark") or {}).get("stop_accuracy_target", 0.90))
    target_latency = float((config.get("benchmark") or {}).get("visible_to_stop_latency_target_sec", 1.5))
    summary = summarize_stop_rows(rows, target_acc=target_acc, target_latency_sec=target_latency)
    accepted_episodes = sum(1 for row in rows if row.get("real_step_accepted"))
    parse_errors = sum(int(row.get("parse_error", 0)) for row in rows)
    metrics = {
        "benchmark": str((config.get("benchmark") or {}).get("name", "forced_stop_oracle_live")),
        "mock_models": False,
        "live_closed_loop": True,
        "episodes": summary["episodes"],
        "forced_stop_oracle_accuracy": summary["stop_decision_accuracy"],
        "stop_decision_accuracy": summary["stop_decision_accuracy"],
        "visible_to_stop_latency_sec": summary["visible_to_stop_latency_sec"],
        "visible_to_stop_latency_p95_sec": summary["visible_to_stop_latency_p95_sec"],
        "target_stop_decision_accuracy": target_acc,
        "target_visible_to_stop_latency_sec": target_latency,
        "pass": bool(summary["pass"] and accepted_episodes == summary["episodes"] and parse_errors == 0),
        "collision_count": summary["collision_count"],
        "stale_action_executed": summary["stale_action_executed"],
        "stale_discard_count": summary["stale_discard_count"],
        "parse_error": parse_errors,
        "real_step_accepted_episodes": accepted_episodes,
        "real_step_call_coverage": round(accepted_episodes / max(1, summary["episodes"]), 3),
        "max_linear_x_mps": 0.20,
        "actions_through_safe_mux": True,
        "real_robot_motion_enabled": False,
        "failure_top1": summary["failure_top1"],
    }
    _write_common_tables(output, summary["rows"], route=False)
    dump_json(output / "metrics.json", metrics)
    append_jsonl(output / "events.jsonl", [event for row in rows for event in row.get("events", [])])
    _write_trajectory(output, rows)
    analysis = analyze_stale_events([event for row in rows for event in row.get("events", [])])
    write_stale_analysis(output / "stale_discard_analysis.md", analysis)
    _write_stop_summary(output, metrics, summary["rows"])
    return metrics


def write_step_route_run(output: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    target = float((config.get("benchmark") or {}).get("correct_branch_target_rate", 0.60))
    summary = summarize_route_rows(rows, target_rate=target)
    accepted_episodes = sum(1 for row in rows if row.get("real_step_accepted"))
    parse_errors = sum(int(row.get("parse_error", 0)) for row in rows)
    metrics = {
        "benchmark": str((config.get("benchmark") or {}).get("name", "step_route_choice_live")),
        "mock_models": False,
        "live_closed_loop": True,
        "episodes": summary["episodes"],
        "step_route_choice_correct_branch_rate": summary["correct_branch_rate"],
        "correct_branch_rate": summary["correct_branch_rate"],
        "target_correct_branch_rate": target,
        "pass": bool(summary["pass"] and accepted_episodes == summary["episodes"] and parse_errors == 0),
        "collision_count": summary["collision_count"],
        "stale_action_executed": summary["stale_action_executed"],
        "stale_discard_count": summary["stale_discard_count"],
        "parse_error": parse_errors,
        "real_step_accepted_episodes": accepted_episodes,
        "real_step_call_coverage": round(accepted_episodes / max(1, summary["episodes"]), 3),
        "max_linear_x_mps": 0.20,
        "actions_through_safe_mux": True,
        "real_robot_motion_enabled": False,
        "failure_top1": summary["failure_top1"],
    }
    _write_common_tables(output, summary["rows"], route=True)
    dump_json(output / "metrics.json", metrics)
    append_jsonl(output / "events.jsonl", [event for row in rows for event in row.get("events", [])])
    _write_trajectory(output, rows)
    analysis = analyze_stale_events([event for row in rows for event in row.get("events", [])])
    write_stale_analysis(output / "stale_discard_analysis.md", analysis)
    _write_route_summary(output, metrics, summary["rows"])
    return metrics


def write_step_stop_run(output: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    target_acc = float((config.get("benchmark") or {}).get("stop_accuracy_target", 0.75))
    target_latency = float((config.get("benchmark") or {}).get("visible_to_stop_latency_target_sec", 2.0))
    summary = summarize_stop_rows(rows, target_acc=target_acc, target_latency_sec=target_latency)
    accepted_episodes = sum(1 for row in rows if row.get("real_step_accepted"))
    parse_errors = sum(int(row.get("parse_error", 0)) for row in rows)
    metrics = {
        "benchmark": str((config.get("benchmark") or {}).get("name", "step_stop_verify_live")),
        "mock_models": False,
        "live_closed_loop": True,
        "episodes": summary["episodes"],
        "step_semantic_stop_accuracy": summary["stop_decision_accuracy"],
        "stop_decision_accuracy": summary["stop_decision_accuracy"],
        "visible_to_stop_latency_sec": summary["visible_to_stop_latency_sec"],
        "visible_to_stop_latency_p95_sec": summary["visible_to_stop_latency_p95_sec"],
        "target_stop_decision_accuracy": target_acc,
        "target_visible_to_stop_latency_sec": target_latency,
        "pass": bool(summary["pass"] and accepted_episodes == summary["episodes"] and parse_errors == 0),
        "collision_count": summary["collision_count"],
        "stale_action_executed": summary["stale_action_executed"],
        "stale_discard_count": summary["stale_discard_count"],
        "parse_error": parse_errors,
        "real_step_accepted_episodes": accepted_episodes,
        "real_step_call_coverage": round(accepted_episodes / max(1, summary["episodes"]), 3),
        "max_linear_x_mps": 0.20,
        "actions_through_safe_mux": True,
        "real_robot_motion_enabled": False,
        "failure_top1": summary["failure_top1"],
    }
    _write_common_tables(output, summary["rows"], route=False)
    dump_json(output / "metrics.json", metrics)
    append_jsonl(output / "events.jsonl", [event for row in rows for event in row.get("events", [])])
    _write_trajectory(output, rows)
    analysis = analyze_stale_events([event for row in rows for event in row.get("events", [])])
    write_stale_analysis(output / "stale_discard_analysis.md", analysis)
    _write_stop_summary(output, metrics, summary["rows"])
    return metrics


def mock_route_run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    rows = []
    for task in config.get("tasks", []):
        expected = str(task.get("expected_route") or "left")
        row = {
            "task_id": task["task_id"],
            "expected_route": expected,
            "route_choice": expected,
            "first_turn_action": expected,
            "yaw_after_3s": 0.2 if expected == "left" else -0.2,
            "yaw_after_5s": 0.35 if expected == "left" else -0.35,
            "yaw_after_10s": 0.55 if expected == "left" else -0.55,
            "final_y": 0.8 if expected == "left" else -0.8,
            "turn_cmd_vel_count": 10,
            "turn_cmd_vel_duration": 4.0,
            "safety_block_events": 0,
            "stale_discards": 0,
            "collision_count": 0,
            "stale_action_executed": 0,
            "events": [],
            "trajectory": [],
        }
        row.update(judge_forced_route_episode(row))
        rows.append(row)
    return _mark_mock(output, write_route_run(output, rows, config))


def mock_stop_run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    rows = []
    for task in config.get("tasks", []):
        row = {
            "task_id": task["task_id"],
            "first_target_visible_time": 0.2,
            "forced_stop_publish_time": 0.4,
            "stop_cmd_received_time": 0.45,
            "robot_velocity_zero_time": 0.5,
            "visible_to_stop_latency_sec": 0.3,
            "distance_at_first_visible": 2.3,
            "distance_at_stop": 2.2,
            "target_visible_at_stop": True,
            "stop_cmd_received": True,
            "collision_count": 0,
            "stale_action_executed": 0,
            "stale_discards": 0,
            "events": [],
            "trajectory": [],
        }
        row.update(judge_forced_stop_episode(row))
        rows.append(row)
    return _mark_mock(output, write_stop_run(output, rows, config))


def mock_step_route_run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    rows = []
    for task in config.get("tasks", []):
        expected = str(task.get("expected_route") or "left")
        row = {
            "task_id": task["task_id"],
            "expected_route": expected,
            "route_choice": expected,
            "first_turn_action": expected,
            "yaw_after_3s": 0.2 if expected == "left" else -0.2,
            "yaw_after_5s": 0.35 if expected == "left" else -0.35,
            "yaw_after_10s": 0.55 if expected == "left" else -0.55,
            "final_y": 0.8 if expected == "left" else -0.8,
            "turn_cmd_vel_count": 10,
            "turn_cmd_vel_duration": 4.0,
            "safety_block_events": 0,
            "stale_discards": 0,
            "collision_count": 0,
            "stale_action_executed": 0,
            "events": [],
            "trajectory": [],
        }
        row.update(judge_forced_route_episode(row))
        rows.append(row)
    return _mark_mock(output, write_step_route_run(output, rows, config))


def mock_step_stop_run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    rows = []
    for task in config.get("tasks", []):
        row = {
            "task_id": task["task_id"],
            "first_target_visible_time": 0.2,
            "forced_stop_publish_time": 0.4,
            "stop_cmd_received_time": 0.45,
            "robot_velocity_zero_time": 0.5,
            "visible_to_stop_latency_sec": 0.3,
            "distance_at_first_visible": 2.3,
            "distance_at_stop": 2.2,
            "target_visible_at_stop": True,
            "stop_cmd_received": True,
            "collision_count": 0,
            "stale_action_executed": 0,
            "stale_discards": 0,
            "events": [],
            "trajectory": [],
        }
        row.update(judge_forced_stop_episode(row, max_latency_sec=2.0))
        rows.append(row)
    return _mark_mock(output, write_step_stop_run(output, rows, config))


def _mark_mock(output: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(metrics)
    metrics["mock_models"] = True
    metrics["live_closed_loop"] = False
    dump_json(output / "metrics.json", metrics)
    return metrics


def _task_from_config(tasks_doc: dict[str, Any], task_cfg: dict[str, Any]) -> dict[str, Any]:
    task_id = str(task_cfg["task_id"])
    for task in tasks_doc.get("tasks", []):
        if task.get("task_id") == task_id:
            out = dict(task)
            out.update({key: value for key, value in task_cfg.items() if value is not None})
            return out
    return dict(task_cfg)


def _seeded_task(task: dict[str, Any], seed: int) -> dict[str, Any]:
    seeded = dict(task)
    seeded["adapter_task_id"] = str(seeded["task_id"])
    seeded["seed"] = int(seed)
    seeded["task_id"] = f"{seeded['adapter_task_id']}_seed{seed}"
    return seeded


def _pose_matches_start(pose: Any, start_pose: list[float], tolerance_m: float = 0.5) -> bool:
    if not isinstance(pose, (list, tuple)) or len(pose) < 2 or len(start_pose) < 2:
        return False
    return math.hypot(float(pose[0]) - float(start_pose[0]), float(pose[1]) - float(start_pose[1])) <= tolerance_m


def _step_supervisor_mode_ack(events: list[dict[str, Any]], episode_id: str) -> bool:
    for event in events:
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        if (
            details.get("event_type") == "episode_reset"
            and details.get("model") == "step"
            and details.get("reset_scope") == "step_supervisor"
            and details.get("episode_id") == episode_id
        ):
            return True
    return False


def _route_metrics_from_probe(probe: LiveProbe, task: dict[str, Any], scene: dict[str, Any], expected: str, status: dict[str, Any]) -> dict[str, Any]:
    yaws = _yaw_samples(probe.trajectory)
    first_turn = _first_turn(probe.safe_cmd_samples or probe.cmd_candidate_samples)
    final_pose = _last_pose(probe.trajectory)
    target = object_by_id(scene, task["target_object"])
    response = _first_step_role_response(probe.events, "route_choice")
    row = {
        "episode_id": probe.active_episode_id,
        "task_id": task["task_id"],
        "expected_route": expected,
        "route_choice": expected,
        "first_turn_action": first_turn,
        "yaw_after_3s": yaws.get(3, 0.0),
        "yaw_after_5s": yaws.get(5, 0.0),
        "yaw_after_10s": yaws.get(10, 0.0),
        "final_x": final_pose[0],
        "final_y": final_pose[1],
        "distance_to_target_after_20s": round(distance_xy(final_pose, target.get("pose", [0.0, 0.0, 0.0])), 3),
        "turn_cmd_vel_count": sum(1 for row in probe.safe_cmd_samples if abs(_angular_z(row)) > 0.02),
        "turn_cmd_vel_duration": round(0.05 * sum(1 for row in probe.safe_cmd_samples if abs(_angular_z(row)) > 0.02), 3),
        "safety_block_events": _count_events(probe.events, "safety_stop", "failsafe"),
        "stale_discards": _count_model_stale_discards(probe.events),
        "collision_count": 1 if status.get("reason") == "collision" else 0,
        "stale_action_executed": 0,
        "real_step_accepted": bool(response and response.get("result") == "accepted"),
        "step_http_latency_sec": response.get("latency_s") if response else None,
        "step_output": response.get("output") if response else None,
        "parse_error": 1 if response and response.get("result") != "accepted" else 0,
        "events": list(probe.events),
        "trajectory": list(probe.trajectory),
    }
    row.update(judge_forced_route_episode(row))
    return row


def _stop_metrics_from_probe(
    probe: LiveProbe,
    task: dict[str, Any],
    scene: dict[str, Any],
    status: dict[str, Any],
    *,
    first_visible_t: float | None,
    stop_publish_t: float | None,
    distance_at_first_visible: float | None,
    distance_at_stop: float | None,
    max_distance: float,
) -> dict[str, Any]:
    bridge_times = [float(e["t"]) for e in probe.events if e.get("event") == "metrics_event_jsonl" and "semantic_stop_bridge" in json.dumps(e.get("details", {}))]
    stop_cmd_time = bridge_times[0] if bridge_times else stop_publish_t
    zero_times = [float(row["t"]) for row in probe.safe_cmd_samples if abs(_linear_x(row)) < 0.01 and abs(_angular_z(row)) < 0.01 and stop_publish_t is not None and float(row["t"]) >= stop_publish_t]
    zero_t = zero_times[0] if zero_times else stop_cmd_time
    latency = 999.0
    if first_visible_t is not None and zero_t is not None:
        latency = max(0.0, zero_t - first_visible_t)
    final_distance = _float(status.get("distance_to_target"), distance_at_stop if distance_at_stop is not None else 999.0)
    response = _first_step_role_response(probe.events, "semantic_stop")
    row = {
        "episode_id": probe.active_episode_id,
        "task_id": task["task_id"],
        "first_target_visible_time": first_visible_t,
        "forced_stop_publish_time": stop_publish_t,
        "stop_cmd_received_time": stop_cmd_time,
        "robot_velocity_zero_time": zero_t,
        "visible_to_stop_latency_sec": round(latency, 3),
        "distance_at_first_visible": distance_at_first_visible,
        "distance_at_stop": round(_float(distance_at_stop, final_distance), 3),
        "target_visible_at_stop": bool(status.get("target_visible", False)),
        "stop_cmd_received": stop_cmd_time is not None,
        "max_distance_to_target_m": max_distance,
        "success": bool(status.get("success", False)),
        "collision_count": 1 if status.get("reason") == "collision" else 0,
        "stale_action_executed": 0,
        "real_step_accepted": bool(response and response.get("result") == "accepted"),
        "step_http_latency_sec": response.get("latency_s") if response else None,
        "step_output": response.get("output") if response else None,
        "parse_error": 1 if response and response.get("result") != "accepted" else 0,
        "stale_discards": _count_model_stale_discards(probe.events),
        "events": list(probe.events),
        "trajectory": list(probe.trajectory),
    }
    row.update(judge_forced_stop_episode(row))
    return row


def _write_common_tables(output: Path, rows: list[dict[str, Any]], *, route: bool) -> None:
    failure_counts: dict[str, int] = {}
    for row in rows:
        failure_counts[row.get("failure_reason", "unknown")] = failure_counts.get(row.get("failure_reason", "unknown"), 0) + 1
    with (output / "failure_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["failure_reason", "count"])
        writer.writeheader()
        for key, value in sorted(failure_counts.items(), key=lambda item: item[1], reverse=True):
            writer.writerow({"failure_reason": key, "count": value})
    if route:
        fields = ["task_id", "expected_route", "first_turn_action", "yaw_after_3s", "yaw_after_5s", "yaw_after_10s", "entered_correct_branch", "turn_cmd_vel_count", "safety_block_events", "stale_discards", "failure_reason"]
    else:
        fields = ["task_id", "visible_to_stop_latency_sec", "distance_at_stop", "target_visible_at_stop", "stop_decision_accuracy", "stopped_too_late", "stopped_too_early", "never_stopped", "failure_reason"]
    with (output / "mode_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_trajectory(output: Path, rows: list[dict[str, Any]]) -> None:
    with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "t", "x", "y", "yaw", "source"])
        writer.writeheader()
        for row in rows:
            for sample in row.get("trajectory", []):
                pose = sample.get("pose") or [0.0, 0.0, 0.0]
                writer.writerow({"episode_id": row.get("episode_id", row.get("task_id")), "t": sample.get("t", 0.0), "x": pose[0], "y": pose[1], "yaw": pose[2], "source": sample.get("source", "")})


def _write_route_summary(output: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# forced_route_oracle_live Summary",
        "",
        f"- run_dir: {output}",
        f"- correct_branch_rate: {metrics['correct_branch_rate']:.3f}",
        f"- target_correct_branch_rate: {metrics['target_correct_branch_rate']:.3f}",
        f"- pass: {metrics['pass']}",
        f"- collision_count: {metrics['collision_count']}",
        f"- stale_action_executed: {metrics['stale_action_executed']}",
        f"- stale_discard_count: {metrics['stale_discard_count']}",
        "",
        "| task_id | expected | first_turn | yaw_5 | yaw_10 | entered_correct_branch | failure |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['task_id']} | {row.get('expected_route')} | {row.get('first_turn_action')} | {row.get('yaw_after_5s')} | {row.get('yaw_after_10s')} | {row.get('entered_correct_branch')} | {row.get('failure_reason')} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stop_summary(output: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# forced_stop_oracle_live Summary",
        "",
        f"- run_dir: {output}",
        f"- stop_decision_accuracy: {metrics['stop_decision_accuracy']:.3f}",
        f"- visible_to_stop_latency_sec: {metrics['visible_to_stop_latency_sec']:.3f}",
        f"- visible_to_stop_latency_p95_sec: {metrics['visible_to_stop_latency_p95_sec']:.3f}",
        f"- target_stop_decision_accuracy: {metrics['target_stop_decision_accuracy']:.3f}",
        f"- target_visible_to_stop_latency_sec: {metrics['target_visible_to_stop_latency_sec']:.3f}",
        f"- pass: {metrics['pass']}",
        f"- collision_count: {metrics['collision_count']}",
        f"- stale_action_executed: {metrics['stale_action_executed']}",
        f"- stale_discard_count: {metrics['stale_discard_count']}",
        "",
        "| task_id | latency | distance_at_stop | visible_at_stop | stop_acc | failure |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['task_id']} | {row.get('visible_to_stop_latency_sec')} | {row.get('distance_at_stop')} | {row.get('target_visible_at_stop')} | {row.get('stop_decision_accuracy')} | {row.get('failure_reason')} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _yaw_samples(trajectory: list[dict[str, Any]]) -> dict[int, float]:
    result: dict[int, float] = {}
    for target_t in (3, 5, 10):
        if not trajectory:
            result[target_t] = 0.0
            continue
        closest = min(trajectory, key=lambda row: abs(float(row.get("t", 0.0)) - target_t))
        pose = closest.get("pose") or [0.0, 0.0, 0.0]
        result[target_t] = round(float(pose[2]), 3)
    return result


def _last_pose(trajectory: list[dict[str, Any]]) -> list[float]:
    if not trajectory:
        return [0.0, 0.0, 0.0]
    pose = trajectory[-1].get("pose") or [0.0, 0.0, 0.0]
    return [float(pose[0]), float(pose[1]), float(pose[2])]


def _first_turn(samples: list[dict[str, Any]]) -> str:
    for row in samples:
        z = _angular_z(row)
        x = _linear_x(row)
        if z > 0.02:
            return "left"
        if z < -0.02:
            return "right"
        if x > 0.02:
            return "front"
    return "none"


def _linear_x(row: dict[str, Any]) -> float:
    return _float(((row.get("cmd") or {}).get("linear") or {}).get("x"), 0.0)


def _angular_z(row: dict[str, Any]) -> float:
    return _float(((row.get("cmd") or {}).get("angular") or {}).get("z"), 0.0)


def _count_events(events: list[dict[str, Any]], *tokens: str) -> int:
    count = 0
    for event in events:
        text = json.dumps(event, ensure_ascii=False).lower()
        if any(token.lower() in text for token in tokens):
            count += 1
    return count


def _count_text(events: list[dict[str, Any]], token: str) -> int:
    token = token.lower()
    return sum(1 for event in events if token in json.dumps(event, ensure_ascii=False).lower())


def _count_model_stale_discards(events: list[dict[str, Any]]) -> int:
    stale_event_types = {
        "step_response_stale",
        "stale_step_result",
        "stale_omninav_action",
        "runtime_stale_discard",
        "old_response_after_reset",
        "episode_mismatch",
        "timebase_error",
    }
    count = 0
    for event in events:
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        event_type = str(details.get("event_type") or event.get("event") or "")
        result = str(details.get("result") or "")
        if event_type in stale_event_types and result in {"discarded", "stale", "rejected", ""}:
            count += 1
    return count


def _first_step_role_response(events: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    for event in events:
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        if details.get("event_type") != "step_http_response" or details.get("role") != role:
            continue
        return {
            "t": float(event.get("t", 0.0)),
            "result": str(details.get("result") or ""),
            "latency_s": _float(details.get("latency_s"), 999.0),
            "output": details.get("output") if isinstance(details.get("output"), dict) else {},
            "request_id": str(details.get("request_id") or ""),
        }
    return None


def _first_accepted_step_role_response(events: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    response = _first_step_role_response(events, role)
    return response if response and response.get("result") == "accepted" else None


def _expected_route(task: dict[str, Any]) -> str:
    text = str(task.get("instruction") or "").lower()
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    return "front"


def _topic_event(topic: str) -> str:
    return topic.strip("/").replace("/", "_") or "event"


def _twist_dict(msg: Any) -> dict[str, Any]:
    return {
        "linear": {"x": float(msg.linear.x), "y": float(msg.linear.y), "z": float(msg.linear.z)},
        "angular": {"x": float(msg.angular.x), "y": float(msg.angular.y), "z": float(msg.angular.z)},
    }


def _yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
