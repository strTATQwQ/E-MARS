#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_ROOT = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler"
BENCHMARK_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(SCHEDULER_ROOT) not in sys.path:
    sys.path.insert(0, str(SCHEDULER_ROOT))
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from isaac_vln_benchmark.perception_planning_suite import (
    audit_step_visible_request,
    reset_payload_for_visual_case,
)
from omninav_step_scheduler.step_roles import build_route_choice_prompt, build_semantic_stop_prompt


def jpeg_visual_stats(jpeg_bytes: bytes) -> dict[str, float | bool]:
    from PIL import Image, ImageFilter, ImageStat

    image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    upper = image.crop((0, 0, image.width, max(1, image.height // 2))).convert("L")
    luminance_stddev = float(ImageStat.Stat(upper).stddev[0])
    edge_mean = float(ImageStat.Stat(upper.filter(ImageFilter.FIND_EDGES)).mean[0])
    return {
        "upper_half_luminance_stddev": round(luminance_stddev, 6),
        "upper_half_edge_mean": round(edge_mean, 6),
        "background_visual_detected": luminance_stddev >= 18.0 or edge_mean >= 5.0,
    }


def evaluate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if isinstance(row.get("response"), dict)]
    observations = [
        response
        for row in rows
        for response in (row.get("observation_responses") or [row.get("response")])
        if isinstance(response, dict)
    ]
    fresh_images = [
        response
        for response in observations
        if bool(response.get("multimodal"))
        and isinstance(response.get("image_snapshot"), dict)
        and float(response["image_snapshot"].get("age_sec", 99.0)) <= 0.75
    ]
    route_rows = [row for row in accepted if row.get("role") == "route_choice"]
    stop_rows = [row for row in accepted if row.get("role") == "semantic_stop"]
    route_correct = sum(row["response"].get("route_choice") == row.get("expected") for row in route_rows)
    semantic_correct = sum(bool(row["response"].get("target_visible")) == bool(row.get("expected")) for row in stop_rows)
    track_confirmed = sum(bool((row["response"].get("track") or {}).get("confirmed")) for row in accepted)
    positive_stop_rows = [row for row in stop_rows if bool(row.get("expected"))]
    positive_tracks_confirmed = sum(
        bool((row["response"].get("track") or {}).get("confirmed")) for row in positive_stop_rows
    )
    route_primitive_correct = 0
    route_safe_cmd_correct = 0
    for row in route_rows:
        expected = str(row.get("expected") or "")
        primitives = row.get("primitives_after_response") or []
        if any(
            primitive.get("primitive") == "follow_waypoint"
            and primitive.get("route_choice") == expected
            for primitive in primitives
        ):
            route_primitive_correct += 1
        angular_values = [float(cmd.get("angular_z", 0.0)) for cmd in row.get("safe_cmds_after_response") or []]
        if expected == "left" and any(value > 0.01 for value in angular_values):
            route_safe_cmd_correct += 1
        if expected == "right" and any(value < -0.01 for value in angular_values):
            route_safe_cmd_correct += 1
    semantic_chain_correct = 0
    for row in stop_rows:
        stop_published = any(
            primitive.get("primitive") == "stop" for primitive in row.get("primitives_after_response") or []
        )
        if stop_published == bool(row.get("expected")):
            semantic_chain_correct += 1
    planning_rows = [row for row in route_rows if bool(row.get("route_planning_required"))]
    route_planning_correct = sum(bool(row.get("entered_correct_branch")) for row in planning_rows)
    approach_rows = [row for row in stop_rows if bool(row.get("semantic_approach_required"))]
    semantic_approach_triggered = sum(bool(row.get("approach_trigger_reached")) for row in approach_rows)
    semantic_approach_success = sum(bool(row.get("semantic_success")) for row in approach_rows)
    failures = []
    if len(accepted) != len(rows):
        failures.append("missing Step response")
    if len(fresh_images) != len(observations):
        failures.append("missing or stale multimodal image")
    if route_correct != len(route_rows):
        failures.append("route recognition mismatch")
    if semantic_correct != len(stop_rows):
        failures.append("semantic target presence mismatch")
    if positive_tracks_confirmed != len(positive_stop_rows):
        failures.append("positive semantic target was not confirmed across frames")
    if route_primitive_correct != len(route_rows):
        failures.append("route decision did not produce the expected scoped primitive")
    if route_safe_cmd_correct != len(route_rows):
        failures.append("route primitive did not reach safe_mux with the expected yaw sign")
    if semantic_chain_correct != len(stop_rows):
        failures.append("semantic track did not produce the expected stop/no-stop primitive")
    if route_planning_correct != len(planning_rows):
        failures.append("Step route decision did not enter the correct branch")
    if semantic_approach_triggered != len(approach_rows):
        failures.append("target-relative planner did not reach the configured visible stop trigger")
    if semantic_approach_success != len(approach_rows):
        failures.append("tracked semantic stop did not satisfy the Isaac success judge")
    return {
        "schema_version": 1,
        "pass": not failures,
        "qualification_evidence": False,
        "locomotion_fidelity": "ideal_kinematic",
        "requests": len(rows),
        "step_calls": len(observations),
        "responses": len(accepted),
        "fresh_multimodal_images": len(fresh_images),
        "route_correct": route_correct,
        "route_total": len(route_rows),
        "semantic_correct": semantic_correct,
        "semantic_total": len(stop_rows),
        "track_confirmed_responses": track_confirmed,
        "positive_semantic_tracks_confirmed": positive_tracks_confirmed,
        "route_primitive_correct": route_primitive_correct,
        "route_safe_cmd_correct": route_safe_cmd_correct,
        "semantic_chain_correct": semantic_chain_correct,
        "route_planning_correct": route_planning_correct,
        "route_planning_total": len(planning_rows),
        "semantic_approach_triggered": semantic_approach_triggered,
        "semantic_approach_success": semantic_approach_success,
        "semantic_approach_total": len(approach_rows),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled real-Step visual route/target recognition micro cases.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--endpoint", default="http://10.100.100.128:8096/v1/chat/completions")
    parser.add_argument("--model", default="Step-3.7-flash-IQ3_XXS-00001-of-00002.gguf")
    parser.add_argument("--no-horizontal-flip", action="store_false", dest="horizontal_flip")
    parser.add_argument("--no-vertical-flip", action="store_false", dest="vertical_flip")
    parser.add_argument("--case-group", choices=("all", "route", "semantic"), default="all")
    parser.add_argument("--case-name", action="append", default=[])
    parser.add_argument("--case-manifest", default="")
    parser.add_argument("--route-planning-timeout-sec", type=float, default=0.0)
    parser.add_argument("--semantic-approach-timeout-sec", type=float, default=0.0)
    parser.add_argument("--semantic-trigger-distance-m", type=float, default=2.0)
    parser.add_argument("--background-settle-sec", type=float, default=8.0)
    parser.set_defaults(horizontal_flip=True, vertical_flip=True)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    class VisualMicroNode(Node):
        def __init__(self) -> None:
            super().__init__("step_visual_perception_micro")
            self.image_count = 0
            self.latest_image = None
            self.responses: dict[str, dict[str, Any]] = {}
            self.primitives: list[dict[str, Any]] = []
            self.safe_cmds: list[dict[str, float]] = []
            self.robot_states: list[dict[str, Any]] = []
            self.metrics: list[dict[str, Any]] = []
            self.statuses: list[dict[str, Any]] = []
            self.reset_acks: list[dict[str, Any]] = []
            self.mode_pub = self.create_publisher(String, "/benchmark/mode_json", 10)
            self.reset_pub = self.create_publisher(String, "/isaac/reset_episode", 10)
            self.instruction_pub = self.create_publisher(String, "/user_instruction", 10)
            self.request_pub = self.create_publisher(String, "/step/request_json", 10)
            self.create_subscription(Image, "/camera/front/image", self.on_image, 2)
            self.create_subscription(String, "/step/route_choice_json", self.on_response, 10)
            self.create_subscription(String, "/step/semantic_stop_json", self.on_response, 10)
            self.create_subscription(String, "/primitive/command_json", self.on_primitive, 20)
            self.create_subscription(Twist, "/safe_cmd_vel", self.on_safe_cmd, 20)
            self.create_subscription(String, "/robot_state_json", self.on_robot_state, 20)
            self.create_subscription(String, "/metrics/event_jsonl", self.on_metric, 100)
            self.create_subscription(String, "/isaac/episode_status", self.on_status, 20)
            self.create_subscription(String, "/isaac/reset_ack_json", self.on_reset_ack, 20)

        def on_image(self, msg) -> None:
            self.image_count += 1
            self.latest_image = msg

        def on_response(self, msg) -> None:
            payload = safe_json(msg.data)
            request_id = str(payload.get("request_id") or "")
            if request_id:
                self.responses[request_id] = payload

        def on_primitive(self, msg) -> None:
            payload = safe_json(msg.data)
            payload["_received_monotonic"] = time.monotonic()
            self.primitives.append(payload)

        def on_safe_cmd(self, msg) -> None:
            self.safe_cmds.append(
                {
                    "timestamp": time.monotonic(),
                    "linear_x": float(msg.linear.x),
                    "angular_z": float(msg.angular.z),
                }
            )

        def on_robot_state(self, msg) -> None:
            payload = safe_json(msg.data)
            if isinstance(payload.get("pose"), list):
                self.robot_states.append(payload)

        def on_metric(self, msg) -> None:
            self.metrics.append(safe_json(msg.data))

        def on_status(self, msg) -> None:
            payload = safe_json(msg.data)
            payload["_received_monotonic"] = time.monotonic()
            self.statuses.append(payload)

        def on_reset_ack(self, msg) -> None:
            self.reset_acks.append(safe_json(msg.data))

        def spin_until(self, predicate, timeout: float) -> bool:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                if predicate():
                    return True
            return False

        def publish_json(self, pub, payload: dict[str, Any]) -> None:
            pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    cases = [
        {
            "name": "route_left_red_sign",
            "role": "route_choice",
            "task_id": "turn_001",
            "instruction": "At the intersection, turn left toward the red sign.",
            "target": "red exit sign",
            "expected": "left",
        },
        {
            "name": "route_right_blue_box",
            "role": "route_choice",
            "task_id": "turn_002",
            "instruction": "At the intersection, turn right toward the blue box.",
            "target": "blue box",
            "expected": "right",
        },
        {
            "name": "semantic_fire_extinguisher",
            "role": "semantic_stop",
            "task_id": "semantic_001",
            "instruction": "Go to the fire extinguisher near the exit sign.",
            "target": "fire extinguisher",
            "expected": True,
            "distance_m": 1.9,
        },
        {
            "name": "semantic_absent_yellow_hydrant",
            "role": "semantic_stop",
            "task_id": "semantic_001",
            "instruction": "Stop near the yellow hydrant.",
            "target": "yellow hydrant",
            "expected": False,
            "distance_m": 1.9,
        },
    ]
    if args.case_manifest:
        manifest_value = json.loads(Path(args.case_manifest).read_text(encoding="utf-8"))
        if not isinstance(manifest_value, list) or not all(isinstance(row, dict) for row in manifest_value):
            raise SystemExit("--case-manifest must contain a JSON array of case objects")
        cases = [dict(row) for row in manifest_value]
    for case_index, case in enumerate(cases):
        case.setdefault("name", str(case.get("case_id") or f"case_{case_index + 1:03d}"))
        case.setdefault("task_id", "turn_001" if case.get("role") == "route_choice" else "semantic_001")
    if args.case_group != "all":
        wanted_role = "route_choice" if args.case_group == "route" else "semantic_stop"
        cases = [case for case in cases if case["role"] == wanted_role]
    if args.case_name:
        wanted_names = set(args.case_name)
        cases = [case for case in cases if case["name"] in wanted_names]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = VisualMicroNode()
    rows = []
    try:
        node.spin_until(
            lambda: node.request_pub.get_subscription_count() > 0
            and node.reset_pub.get_subscription_count() > 0,
            5.0,
        )
        for index, case in enumerate(cases):
            episode_id = f"visual_micro_{case['name']}_{index:02d}"
            request_id = f"step_visual_{case['name']}_{index:02d}"
            mode = {
                "mode": "omninav_step_route_stop_v12_screen",
                "episode_id": episode_id,
                "mode_config": {
                    "use_step": True,
                    "use_omninav": False,
                    "step_roles_only": True,
                    "external_step_triggers_only": True,
                    "stale_gate_enabled": True,
                    "controller_decision_source": "step",
                },
            }
            if case["role"] == "route_choice" and args.route_planning_timeout_sec > 0.0:
                mode["mode_config"].update(
                    {
                        "v10_mode": "geometric_watchdog",
                        "controller_decision_source": "step",
                        "route_controller_enabled": True,
                        "target_controller_enabled": False,
                        "branch_pre_entry_distance_m": 0.25,
                        "branch_centerline_lookahead_m": 1.0,
                        "branch_target_yaw_deg": 60.0,
                        "branch_post_turn_forward_m": 1.6,
                        "branch_max_enter_forward_m": 2.0,
                        "branch_forward_speed_mps": 0.25,
                        "branch_advance_orient_linear_x_mps": 0.0,
                        "branch_rotate_linear_x_mps": 0.0,
                        "branch_max_yaw_rate_radps": 0.35,
                        "branch_rotate_stop_error_deg": 20.0,
                        "branch_pre_entry_lateral_tolerance_m": 0.9,
                        "branch_advance_timeout_sec": 90.0,
                        "branch_rotate_timeout_sec": 40.0,
                        "branch_enter_timeout_sec": 24.0,
                        "branch_bypass_enabled": True,
                        "branch_bypass_offset_y_m": 2.4,
                        "branch_bypass_clearance_y_m": 1.4,
                        "branch_bypass_release_after_x_m": 0.8,
                        "branch_bypass_waypoint_margin_x_m": 0.2,
                        "branch_bypass_waypoint_reached_m": 0.3,
                        "branch_advance_yaw_start_deg": 6.0,
                        "branch_advance_yaw_stop_deg": 3.0,
                    }
                )
            if (
                case["role"] == "semantic_stop"
                and bool(case.get("expected"))
                and args.semantic_approach_timeout_sec > 0.0
            ):
                mode["mode_config"].update(
                    {
                        "v10_mode": "geometric_watchdog",
                        "controller_decision_source": "step",
                        "route_controller_enabled": False,
                        "target_controller_enabled": True,
                        "semantic_stop_requires_step": True,
                        "target_controller": "waypoint",
                        "target_bypass_enabled": True,
                        "target_bypass_offset_y_m": 1.1,
                        "target_bypass_clearance_y_m": 0.8,
                        "target_bypass_release_after_x_m": 0.8,
                        "target_bypass_waypoint_margin_x_m": 0.2,
                        "target_bypass_waypoint_reached_m": 0.3,
                        "target_coverage_threshold_m": 2.5,
                        "target_stop_threshold_m": 1.95,
                        "target_max_linear_x_mps": 0.20,
                        "target_max_yaw_rate_radps": 0.30,
                        "target_yaw_start_deg": 15.0,
                        "target_yaw_stop_deg": 8.0,
                        "target_max_phase_duration_sec": 120.0,
                    }
                )
            status_start = len(node.statuses)
            reset_ack_start = len(node.reset_acks)
            case_primitive_start = len(node.primitives)
            case_safe_cmd_start = len(node.safe_cmds)
            case_state_start = len(node.robot_states)
            case_metric_start = len(node.metrics)
            reset = reset_payload_for_visual_case(case, episode_id=episode_id)
            reset_ack_received = False
            for _ in range(3):
                node.publish_json(node.mode_pub, mode)
                node.publish_json(node.reset_pub, reset)
                reset_ack_received = node.spin_until(
                    lambda: any(
                        str(ack.get("episode_id") or "") == episode_id
                        for ack in node.reset_acks[reset_ack_start:]
                    ),
                    1.0,
                )
                if reset_ack_received:
                    break
            node.publish_json(node.mode_pub, mode)
            node.spin_until(lambda: False, 0.1)
            # Drop mode-before-reset telemetry. The adapter can publish one old
            # scene sample carrying the new episode id before reset is applied.
            status_start = len(node.statuses)
            case_primitive_start = len(node.primitives)
            case_safe_cmd_start = len(node.safe_cmds)
            case_state_start = len(node.robot_states)
            case_metric_start = len(node.metrics)
            node.publish_json(
                node.instruction_pub,
                {"instruction": case["instruction"], "mission_id": episode_id},
            )
            background_settle_sec = (
                max(0.0, args.background_settle_sec) if reset.get("background_usd") else 0.0
            )
            if background_settle_sec > 0.0:
                node.spin_until(lambda: False, background_settle_sec)
            post_reset_image_count = node.image_count
            post_reset_images_ready = node.spin_until(
                lambda: node.image_count >= post_reset_image_count + 3,
                12.0 if reset.get("background_usd") else 3.0,
            )
            pose_before = (
                list(node.robot_states[-1].get("pose") or []) if node.robot_states else []
            )
            approach_trigger_reached = False
            trigger_status = None
            coverage_status = None
            distance_m = case.get("distance_m")
            if (
                case["role"] == "semantic_stop"
                and bool(case.get("expected"))
                and args.semantic_approach_timeout_sec > 0.0
            ):
                approach_trigger_reached = node.spin_until(
                    lambda: any(
                        str(status.get("episode_id") or "") == episode_id
                        and bool(status.get("target_visible"))
                        and float(status.get("distance_to_target", 999.0))
                        <= args.semantic_trigger_distance_m
                        for status in node.statuses[status_start:]
                    ),
                    args.semantic_approach_timeout_sec,
                )
                matches = [
                    status
                    for status in node.statuses[status_start:]
                    if str(status.get("episode_id") or "") == episode_id
                    and bool(status.get("target_visible"))
                    and float(status.get("distance_to_target", 999.0))
                    <= args.semantic_trigger_distance_m
                ]
                trigger_status = matches[-1] if matches else None
                coverage_matches = [
                    status
                    for status in node.statuses[status_start:]
                    if str(status.get("episode_id") or "") == episode_id
                    and bool(status.get("target_visible"))
                    and float(status.get("distance_to_target", 999.0)) <= 2.5
                ]
                coverage_status = coverage_matches[0] if coverage_matches else None
                if trigger_status is not None:
                    distance_m = float(trigger_status["distance_to_target"])
            event = {"type": "visual_decision_due", "target": case["target"]}
            if case["role"] == "route_choice":
                prompt = build_route_choice_prompt(
                    instruction=case["instruction"],
                    active_subgoal=case["target"],
                    robot_state={},
                    semantic_summary={},
                    safety_status={},
                    event=event,
                )
            else:
                prompt = build_semantic_stop_prompt(
                    instruction=case["instruction"],
                    active_subgoal=case["target"],
                    robot_state={},
                    semantic_summary={},
                    safety_status={},
                    event=event,
                )
            observation_count = 2 if case["role"] == "semantic_stop" else 1
            observation_responses = []
            request_audits = []
            request_ids = []
            image_visual_stats: dict[str, Any] = {}
            total_attempts = 0
            image_ready = True
            primitive_start = len(node.primitives)
            safe_cmd_start = len(node.safe_cmds)
            state_start = len(node.robot_states)
            metric_start = len(node.metrics)
            for observation_index in range(observation_count):
                baseline_images = node.image_count
                current_image_ready = node.spin_until(lambda: node.image_count >= baseline_images + 2, 5.0)
                image_ready = image_ready and current_image_ready
                if observation_index == 0 and node.latest_image is not None:
                    from omninav_step_scheduler.step_http_client_node import ros_image_to_data_url
                    import base64

                    data_url = ros_image_to_data_url(
                        node.latest_image,
                        max_width=640,
                        jpeg_quality=90,
                        horizontal_flip=args.horizontal_flip,
                        vertical_flip=args.vertical_flip,
                    )
                    jpeg_bytes = base64.b64decode(data_url.split(",", 1)[1])
                    (output / f"{index:02d}_{case['name']}.jpg").write_bytes(jpeg_bytes)
                    image_visual_stats = jpeg_visual_stats(jpeg_bytes)
                observation_request_id = (
                    request_id if observation_count == 1 else f"{request_id}_obs{observation_index + 1}"
                )
                request_ids.append(observation_request_id)
                stamp = node.get_clock().now().nanoseconds / 1.0e9
                request = {
                    "request_id": observation_request_id,
                    "episode_id": episode_id,
                    "mission_id": episode_id,
                    "timestamp_request": stamp,
                    "source_stamp_sec": stamp,
                    "created_ros_time_sec": stamp,
                    "clock_domain": "ros_system",
                    "role": case["role"],
                    "multimodal": True,
                    "pose_at_request": [0.0, 0.0, 0.0],
                    "pending_mode": "stop",
                    "instruction": case["instruction"],
                    "active_subgoal": case["target"],
                    "target": case["target"],
                    "visual_ground_truth_hidden": True,
                    "endpoint": args.endpoint,
                    "model": args.model,
                    "max_tokens": 40,
                    "temperature": 0.0,
                    "prompt": prompt,
                }
                request_audits.append(audit_step_visible_request(request))
                attempts = 0
                received = False
                while attempts < 3 and not received:
                    attempts += 1
                    total_attempts += 1
                    node.publish_json(node.request_pub, request)
                    received = node.spin_until(
                        lambda request_id=observation_request_id: request_id in node.responses,
                        min(args.timeout_sec, 7.0),
                    )
                response = node.responses.get(observation_request_id)
                if isinstance(response, dict):
                    observation_responses.append(response)
            response = observation_responses[-1] if observation_responses else None
            expect_primitive = case["role"] == "route_choice" or bool(case.get("expected"))
            if expect_primitive:
                node.spin_until(
                    lambda: any(
                        str(primitive.get("request_id") or "") in request_ids
                        for primitive in node.primitives[primitive_start:]
                    ),
                    1.5,
                )
            entered_correct_branch = False
            if case["role"] == "route_choice" and args.route_planning_timeout_sec > 0.0:
                entered_correct_branch = node.spin_until(
                    lambda: any(
                        metric.get("event_type") == "v10_branch_controller_trace"
                        and str(metric.get("episode_id") or "") == episode_id
                        and bool(metric.get("entered_correct_branch"))
                        for metric in node.metrics[metric_start:]
                    ),
                    args.route_planning_timeout_sec,
                )
            semantic_success = False
            if (
                case["role"] == "semantic_stop"
                and args.semantic_approach_timeout_sec > 0.0
                and bool(case.get("expected"))
            ):
                semantic_success = node.spin_until(
                    lambda: any(
                        str(status.get("episode_id") or "") == episode_id
                        and bool(status.get("success"))
                        for status in node.statuses[status_start:]
                    ),
                    5.0,
                )
            node.spin_until(lambda: False, 1.0)
            scoped_primitives = [
                primitive
                for primitive in node.primitives[primitive_start:]
                if str(primitive.get("request_id") or "") in request_ids
            ]
            scoped_safe_cmds = list(node.safe_cmds[safe_cmd_start:])
            case_primitives = list(node.primitives[case_primitive_start:])
            case_safe_cmds = list(node.safe_cmds[case_safe_cmd_start:])
            scoped_statuses = [
                status
                for status in node.statuses[status_start:]
                if str(status.get("episode_id") or "") == episode_id
            ]
            distance_samples = []
            for status in scoped_statuses:
                if status.get("distance_to_target") is None:
                    continue
                distance = float(status["distance_to_target"])
                if not distance_samples or abs(distance - distance_samples[-1]) > 1.0e-6:
                    distance_samples.append(distance)
            approach_steps = [
                {"distance_before": before, "distance_after": after}
                for before, after in zip(distance_samples, distance_samples[1:])
            ]
            stop_primitive_times = [
                float(primitive["_received_monotonic"])
                for primitive in scoped_primitives
                if primitive.get("primitive") == "stop"
                and primitive.get("_received_monotonic") is not None
            ]
            stop_primitive_time = min(stop_primitive_times) if stop_primitive_times else None
            safe_stop_times = [
                float(cmd["timestamp"])
                for cmd in scoped_safe_cmds
                if stop_primitive_time is not None
                and float(cmd["timestamp"]) >= stop_primitive_time
                and abs(float(cmd.get("linear_x", 0.0))) <= 1.0e-3
                and abs(float(cmd.get("angular_z", 0.0))) <= 1.0e-3
            ]
            safe_stop_time = min(safe_stop_times) if safe_stop_times else None
            trigger_received_time = (
                float(trigger_status["_received_monotonic"])
                if trigger_status and trigger_status.get("_received_monotonic") is not None
                else None
            )
            coverage_received_time = (
                float(coverage_status["_received_monotonic"])
                if coverage_status and coverage_status.get("_received_monotonic") is not None
                else None
            )
            threshold_to_safe_stop_sec = (
                max(0.0, safe_stop_time - stop_primitive_time)
                if safe_stop_time is not None and stop_primitive_time is not None
                else None
            )
            trigger_to_safe_stop_sec = (
                max(0.0, safe_stop_time - trigger_received_time)
                if safe_stop_time is not None and trigger_received_time is not None
                else None
            )
            coverage_to_stop_sec = (
                max(0.0, safe_stop_time - coverage_received_time)
                if safe_stop_time is not None and coverage_received_time is not None
                else None
            )
            pose_after = (
                list(node.robot_states[-1].get("pose") or []) if node.robot_states else []
            )
            controller_trace = [
                metric
                for metric in node.metrics[metric_start:]
                if metric.get("event_type") == "v10_branch_controller_trace"
                and str(metric.get("episode_id") or "") == episode_id
            ]
            step_http_events = [
                metric
                for metric in node.metrics[metric_start:]
                if metric.get("event_type") == "step_http_response"
                and str(metric.get("request_id") or "") in request_ids
            ]
            trajectory = [
                state.get("pose")
                for state in node.robot_states[state_start:]
                if isinstance(state.get("pose"), list)
            ]
            full_trajectory = [
                state.get("pose")
                for state in node.robot_states[case_state_start:]
                if isinstance(state.get("pose"), list)
            ]
            case_metrics = list(node.metrics[case_metric_start:])
            rows.append(
                {
                    **case,
                    "episode_id": episode_id,
                    "request_id": request_id,
                    "request_ids": request_ids,
                    "camera_ready": image_ready,
                    "reset_ack_received": reset_ack_received,
                    "post_reset_images_ready": post_reset_images_ready,
                    "background_usd": reset.get("background_usd"),
                    "background_visual_only": bool(reset.get("background_visual_only", False)),
                    "background_settle_sec": background_settle_sec,
                    "image_visual_stats": image_visual_stats,
                    "response_received": len(observation_responses) == observation_count,
                    "publish_attempts": total_attempts,
                    "response": response,
                    "observation_responses": observation_responses,
                    "request_audits": request_audits,
                    "step_http_events": step_http_events,
                    "primitives_after_response": scoped_primitives,
                    "safe_cmds_after_response": scoped_safe_cmds,
                    "case_primitives": case_primitives,
                    "case_safe_cmds": case_safe_cmds,
                    "case_metrics": case_metrics,
                    "pose_before": pose_before,
                    "pose_after": pose_after,
                    "route_planning_required": bool(
                        case["role"] == "route_choice" and args.route_planning_timeout_sec > 0.0
                    ),
                    "entered_correct_branch": entered_correct_branch,
                    "controller_phases": list(dict.fromkeys(str(row.get("phase") or "") for row in controller_trace)),
                    "controller_trace": controller_trace,
                    "trajectory": trajectory,
                    "full_trajectory": full_trajectory,
                    "approach_steps": approach_steps,
                    "threshold_to_safe_stop_sec": threshold_to_safe_stop_sec,
                    "trigger_to_safe_stop_sec": trigger_to_safe_stop_sec,
                    "coverage_to_stop_sec": coverage_to_stop_sec,
                    "semantic_approach_required": bool(
                        case["role"] == "semantic_stop"
                        and bool(case.get("expected"))
                        and args.semantic_approach_timeout_sec > 0.0
                    ),
                    "approach_trigger_reached": approach_trigger_reached,
                    "trigger_status": trigger_status,
                    "coverage_status": coverage_status,
                    "distance_m_used": distance_m,
                    "semantic_trigger_distance_m": args.semantic_trigger_distance_m,
                    "semantic_success": semantic_success,
                }
            )
            time.sleep(0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    summary = evaluate_results(rows)
    (output / "cases.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["pass"] else 2


def safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


if __name__ == "__main__":
    raise SystemExit(main())
