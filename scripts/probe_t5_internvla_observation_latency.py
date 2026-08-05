#!/usr/bin/env python3
"""Measure InternVLA inference without publishing a navigation command or STOP.

The probe owns only the three typed observation publishers.  It sends Step
goals directly to the model action server and records the returned latency,
but deliberately never creates the client-side NavigationCommand,
``/internvla/discrete_action``, or ``/internvla/stop`` publishers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from internvla_ros2.identity import CHECKPOINT_REVISION, MODEL_REVISION
from internvla_ros2.observation import observation_digest
from internvla_ros2_msgs.action import Step
from internvla_ros2_msgs.msg import ObservationMetadata
from internvla_ros2_msgs.srv import Initialize, Reset
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


PROTOCOL_VERSION = 1
STATUS_OK = 0
FORBIDDEN_OUTPUT_TOPICS = (
    "/internvla/discrete_action",
    "/internvla/stop",
    "/cmd_vel",
)


def _assign_time(message: Any, nanoseconds: int) -> None:
    message.sec = int(nanoseconds // 1_000_000_000)
    message.nanosec = int(nanoseconds % 1_000_000_000)


def _wait_future(node: Any, future: Any, timeout_sec: float, label: str) -> Any:
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    if not future.done() or future.result() is None:
        raise TimeoutError(f"{label} timed out")
    return future.result()


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("latency sample is empty")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _publisher_counts(node: Any) -> dict[str, int]:
    return {topic: int(node.count_publishers(topic)) for topic in FORBIDDEN_OUTPUT_TOPICS}


def _wait_graph(node: Any, clients: list[Any], action: ActionClient, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        services_ready = all(client.wait_for_service(timeout_sec=0.1) for client in clients)
        action_ready = action.wait_for_server(timeout_sec=0.1)
        if services_ready and action_ready:
            return
    raise TimeoutError("typed InternVLA initialize/reset/step graph did not become ready")


def _wait_observation_subscribers(publishers: list[Any], timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if all(publisher.get_subscription_count() == 1 for publisher in publishers):
            return
        time.sleep(0.05)
    counts = [publisher.get_subscription_count() for publisher in publishers]
    raise TimeoutError(f"expected exactly one model subscriber per observation topic, got {counts}")


def _initialize(client: Any, node: Any, episode_id: str) -> tuple[str, int]:
    request = Initialize.Request()
    request.protocol_version = PROTOCOL_VERSION
    request.episode_id = episode_id
    request.model_revision = MODEL_REVISION
    request.checkpoint_revision = CHECKPOINT_REVISION
    response = _wait_future(node, client.call_async(request), 30.0, "initialize")
    if int(response.status_code) != STATUS_OK or not response.initialized:
        raise RuntimeError(f"initialize failed: {response.status_code} {response.status_message}")
    if response.episode_id != episode_id or int(response.reset_generation) != 0:
        raise RuntimeError("initialize returned unexpected episode/reset identity")
    return response.episode_id, int(response.reset_generation)


def _reset(
    client: Any,
    node: Any,
    episode_id: str,
    expected_generation: int,
    barrier_sequence: int,
) -> tuple[str, int]:
    request = Reset.Request()
    request.protocol_version = PROTOCOL_VERSION
    request.next_episode_id = episode_id
    request.expected_reset_generation = expected_generation
    request.reset_barrier_sequence_id = barrier_sequence
    response = _wait_future(node, client.call_async(request), 30.0, "reset")
    if int(response.status_code) != STATUS_OK:
        raise RuntimeError(f"reset failed: {response.status_code} {response.status_message}")
    if response.episode_id != episode_id or int(response.reset_generation) != expected_generation + 1:
        raise RuntimeError("reset returned unexpected episode/reset identity")
    # Reset.Response reports the barrier of the *new* generation.  The model
    # has just cleared last_sequence_id to -1 and the uint64 service field is
    # therefore represented as zero.  The request barrier still protects the
    # completed generation, but must not be echoed back as if it belonged to
    # the new episode.
    if int(response.reset_barrier_sequence_id) != 0:
        raise RuntimeError("reset returned nonzero new-generation sequence barrier")
    return response.episode_id, int(response.reset_generation)


def _publish_observation(
    node: Any,
    publishers: tuple[Any, Any, Any],
    *,
    episode_id: str,
    reset_generation: int,
    sequence_id: int,
    deadline_sec: float,
) -> tuple[Step.Goal, str]:
    # A contract-valid deterministic observation keeps the comparison focused
    # on model load/coexistence rather than simulator or transport variance.
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:, :, 1] = 24
    depth = np.full((480, 640, 1), 0.5, dtype=np.float32)
    instruction = "walk forward toward the doorway"
    instruction_tokens = [116, 105, 110, 121]
    gps = [0.0, 0.0, 0.0]
    rotation = [1.0, 0.0, 0.0, 0.0]
    digest = observation_digest(rgb, depth, instruction, instruction_tokens, gps, rotation)

    stamp_ns = int(node.get_clock().now().nanoseconds)
    if stamp_ns <= 0:
        raise RuntimeError("ROS system clock is zero")
    deadline_ns = stamp_ns + int(deadline_sec * 1e9)
    valid_until_ns = deadline_ns + int(5.0 * 1e9)
    stamp = node.get_clock().now().to_msg()
    # Preserve the single timestamp contract exactly; do not sample the clock
    # again after constructing the observation identity.
    _assign_time(stamp, stamp_ns)
    request_id = f"{episode_id}:{reset_generation}:{sequence_id}:observation-only"

    rgb_message = Image()
    rgb_message.header.stamp = stamp
    rgb_message.header.frame_id = "internvla_camera"
    rgb_message.height = 480
    rgb_message.width = 640
    rgb_message.encoding = "rgb8"
    rgb_message.is_bigendian = False
    rgb_message.step = 640 * 3
    rgb_message.data = rgb.tobytes()

    depth_message = Image()
    depth_message.header.stamp = stamp
    depth_message.header.frame_id = "internvla_camera"
    depth_message.height = 480
    depth_message.width = 640
    depth_message.encoding = "32FC1"
    depth_message.is_bigendian = False
    depth_message.step = 640 * 4
    depth_message.data = depth.astype("<f4", copy=False).tobytes()

    metadata = ObservationMetadata()
    metadata.header.stamp = stamp
    metadata.header.frame_id = "internvla_camera"
    metadata.protocol_version = PROTOCOL_VERSION
    metadata.episode_id = episode_id
    metadata.reset_generation = reset_generation
    metadata.sequence_id = sequence_id
    metadata.request_id = request_id
    metadata.observation_digest = digest
    metadata.client_wall_monotonic_ns = time.monotonic_ns()
    metadata.sim_stamp = stamp
    _assign_time(metadata.deadline, deadline_ns)
    _assign_time(metadata.valid_until, valid_until_ns)
    metadata.instruction = instruction
    metadata.instruction_tokens = instruction_tokens
    metadata.global_gps = gps
    metadata.global_rotation = rotation

    goal = Step.Goal()
    goal.header.stamp = stamp
    goal.header.frame_id = "internvla_camera"
    goal.protocol_version = PROTOCOL_VERSION
    goal.episode_id = episode_id
    goal.reset_generation = reset_generation
    goal.sequence_id = sequence_id
    goal.request_id = request_id
    goal.observation_digest = digest
    goal.client_wall_monotonic_ns = metadata.client_wall_monotonic_ns
    goal.sim_stamp = stamp
    goal.observation_stamp = stamp
    _assign_time(goal.deadline, deadline_ns)
    _assign_time(goal.valid_until, valid_until_ns)

    publishers[0].publish(rgb_message)
    publishers[1].publish(depth_message)
    publishers[2].publish(metadata)
    return goal, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("internvla_only", "co_loaded"), required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--deadline-sec", type=float, default=300.0)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--expected-reset-generation", type=int, default=0)
    parser.add_argument("--reset-barrier-sequence", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be fresh")
    if args.initialize == args.reset:
        parser.error("select exactly one of --initialize or --reset")
    if args.count != 5:
        parser.error("the frozen coexistence probe count is exactly 5")
    if not 30.0 <= args.deadline_sec <= 600.0:
        parser.error("deadline must be in [30, 600] seconds")
    if not args.episode_id.startswith("b::coexistence-probe::"):
        parser.error("episode must use the Lane-B coexistence-probe prefix")

    rclpy.init()
    node = rclpy.create_node(f"internvla_t5_observation_only_probe_{os.getpid()}")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=8,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    rgb_publisher = node.create_publisher(Image, "/internvla/observation/rgb", qos)
    depth_publisher = node.create_publisher(Image, "/internvla/observation/depth", qos)
    metadata_publisher = node.create_publisher(
        ObservationMetadata, "/internvla/observation/metadata", qos
    )
    initialize_client = node.create_client(Initialize, "/internvla/initialize")
    reset_client = node.create_client(Reset, "/internvla/reset")
    action = ActionClient(node, Step, "/internvla/step")
    try:
        forbidden_before = _publisher_counts(node)
        if any(forbidden_before.values()):
            raise RuntimeError(f"forbidden motion publisher already exists: {forbidden_before}")
        _wait_graph(node, [initialize_client, reset_client], action, 30.0)
        _wait_observation_subscribers(
            [rgb_publisher, depth_publisher, metadata_publisher], 30.0
        )
        if args.initialize:
            episode_id, reset_generation = _initialize(
                initialize_client, node, args.episode_id
            )
        else:
            episode_id, reset_generation = _reset(
                reset_client,
                node,
                args.episode_id,
                args.expected_reset_generation,
                args.reset_barrier_sequence,
            )

        records: list[dict[str, Any]] = []
        for sequence_id in range(args.count):
            goal, digest = _publish_observation(
                node,
                (rgb_publisher, depth_publisher, metadata_publisher),
                episode_id=episode_id,
                reset_generation=reset_generation,
                sequence_id=sequence_id,
                deadline_sec=args.deadline_sec,
            )
            started = time.perf_counter()
            goal_handle = _wait_future(
                node,
                action.send_goal_async(goal),
                10.0,
                f"step {sequence_id} goal acceptance",
            )
            if not goal_handle.accepted:
                raise RuntimeError(f"step {sequence_id} goal was rejected")
            wrapped = _wait_future(
                node,
                goal_handle.get_result_async(),
                args.deadline_sec + 5.0,
                f"step {sequence_id} result",
            )
            wall_latency = float(time.perf_counter() - started)
            result = wrapped.result
            if int(result.status_code) != STATUS_OK:
                raise RuntimeError(
                    f"step {sequence_id} failed: {result.status_code} {result.status_message}"
                )
            expected = (episode_id, reset_generation, sequence_id, goal.request_id)
            observed = (
                result.episode_id,
                int(result.reset_generation),
                int(result.sequence_id),
                result.request_id,
            )
            if observed != expected or bool(result.replayed):
                raise RuntimeError(f"step {sequence_id} returned stale/replayed identity")
            records.append(
                {
                    "sequence_id": sequence_id,
                    "request_id": goal.request_id,
                    "observation_digest": digest,
                    "status_code": int(result.status_code),
                    "inference_latency_sec": float(result.inference_latency_sec),
                    "round_trip_latency_sec": wall_latency,
                    "response_consumed_only": True,
                }
            )

        forbidden_after = _publisher_counts(node)
        if forbidden_after != forbidden_before or any(forbidden_after.values()):
            raise RuntimeError(f"forbidden motion publishers appeared: {forbidden_after}")
        inference = [record["inference_latency_sec"] for record in records]
        round_trip = [record["round_trip_latency_sec"] for record in records]
        payload = {
            "schema_version": 1,
            "status": "PASS",
            "phase": args.phase,
            "probe_mode": "observation_only_no_navigation_publishers",
            "episode_id": episode_id,
            "reset_generation": reset_generation,
            "sample_count": len(records),
            "model_revision": MODEL_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "observation_source": "deterministic_contract_valid_probe",
            "forbidden_output_publisher_counts_before": forbidden_before,
            "forbidden_output_publisher_counts_after": forbidden_after,
            "inference_latency_sec": {
                "p50": _nearest_rank(inference, 0.50),
                "p95": _nearest_rank(inference, 0.95),
                "values": inference,
            },
            "round_trip_latency_sec": {
                "p50": _nearest_rank(round_trip, 0.50),
                "p95": _nearest_rank(round_trip, 0.95),
                "values": round_trip,
            },
            "records": records,
            "recorded_unix": time.time(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
