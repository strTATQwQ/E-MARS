#!/usr/bin/env python3
"""Advise one bounded primitive or classify arrival from Rev-C snapshots.

The node has no velocity or terminal-stop publisher.  It only binds a verified
same-render-tick Rev-C snapshot to SlowPlanner protocol v1.  The client-side
termination arbiter, not this node, owns any eventual Step3-assisted STOP.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from slow_planner.base import CandidateFrontier, OrderedImage, SlowPlannerRequest
from slow_planner.client import SlowPlannerClient
from scripts.probe_t5_revc_snapshot_smoke import (
    CAMERA_ORDER,
    SmokeContractError,
    _atomic_json,
    _load_regular_json,
    validate_capture,
    wait_for_matching_ack,
)


CAMERA_POSES = {
    "front_left": (0.170, 0.051962, 0.200, math.radians(60.0), math.radians(-10.0)),
    "front": (0.200, 0.0, 0.200, 0.0, math.radians(-10.0)),
    "front_right": (0.170, -0.051962, 0.200, math.radians(-60.0), math.radians(-10.0)),
    "rear": (0.080, 0.0, 0.200, math.pi, math.radians(-10.0)),
}
PRIMITIVES = {
    1: CandidateFrontier(1, (0.0, 0.25), 0.25, 0.0),
    2: CandidateFrontier(2, (-0.18, 0.18), 0.255, -45.0),
    3: CandidateFrontier(3, (0.18, 0.18), 0.255, 45.0),
}


class TimeoutAdvisorError(RuntimeError):
    pass


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(descriptor, line.encode("utf-8")) != len(line.encode("utf-8")):
            raise OSError("short JSONL append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _context(value: dict[str, Any]) -> dict[str, Any]:
    kind = value.get("kind")
    if (
        value.get("schema_version") != 1
        or kind
        not in {
            "motion_timeout_after_confirmed_safe_stop",
            "arrival_check_after_completed_motion",
        }
        or not str(value.get("episode_id", "")).startswith("a::")
        or isinstance(value.get("reset_generation"), bool)
        or not isinstance(value.get("reset_generation"), int)
        or isinstance(value.get("trigger_sequence_id"), bool)
        or not isinstance(value.get("trigger_sequence_id"), int)
        or isinstance(value.get("expected_sequence_id"), bool)
        or not isinstance(value.get("expected_sequence_id"), int)
        or value.get("expected_sequence_id") != value.get("trigger_sequence_id") + 1
        or value.get("camera_order") != list(CAMERA_ORDER)
        or not str(value.get("trigger_request_id", ""))
        or not str(value.get("instruction", "")).strip()
        or isinstance(value.get("advisor_round"), bool)
        or not isinstance(value.get("advisor_round"), int)
    ):
        raise TimeoutAdvisorError("invalid Lane-A Step3 context")
    if kind == "motion_timeout_after_confirmed_safe_stop":
        if (
            value.get("advisor_round") != 0
            or isinstance(value.get("excluded_action"), bool)
            or not isinstance(value.get("excluded_action"), int)
            or value.get("excluded_action") not in PRIMITIVES
        ):
            raise TimeoutAdvisorError("invalid Lane-A timeout context")
    else:
        if (
            value.get("advisor_round") not in {1, 2}
            or value.get("required_confirmations") != 2
            or isinstance(value.get("completed_action"), bool)
            or not isinstance(value.get("completed_action"), int)
            or value.get("completed_action") not in PRIMITIVES
            or isinstance(value.get("camera_sensor_stamp_ns"), bool)
            or not isinstance(value.get("camera_sensor_stamp_ns"), int)
            or value.get("camera_sensor_stamp_ns") <= 0
            or isinstance(value.get("minimum_snapshot_sim_stamp_ns"), bool)
            or not isinstance(value.get("minimum_snapshot_sim_stamp_ns"), int)
            or value.get("minimum_snapshot_sim_stamp_ns") < 0
        ):
            raise TimeoutAdvisorError("invalid Lane-A arrival context")
    return value


def _request_snapshot(
    *,
    context: dict[str, Any],
    request_path: Path,
    ack_path: Path,
    result_root: Path,
    contract_path: Path,
    deadline: float,
) -> dict[str, Any]:
    # Capture the observation that produced the timed-out action.  The next
    # sequence is reserved for the advised action identity; Isaac's active
    # render identity remains the trigger sequence until that action begins.
    sequence_id = int(context["trigger_sequence_id"])
    for attempt in range(2):
        request_id = (
            f"a::step3-timeout::{context['reset_generation']}::"
            f"{context['trigger_sequence_id']}::{attempt}::{time.time_ns()}"
        )
        request = {
            "schema_version": 1,
            "request_id": request_id,
            "episode_id": context["episode_id"],
            "reset_generation": context["reset_generation"],
            "sequence_id": sequence_id,
        }
        if request_path.exists() or request_path.is_symlink():
            raise TimeoutAdvisorError("Rev-C snapshot request path is occupied")
        _atomic_json(request_path, request)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutAdvisorError("snapshot deadline expired")
        ack = wait_for_matching_ack(ack_path, request_id, remaining)
        if ack.get("status") == "CAPTURED":
            sidecar_count = len(
                list((result_root / "revc_snapshots").glob("*/snapshot.json"))
            )
            return validate_capture(
                result_root,
                contract_path,
                request,
                ack,
                expected_sidecar_count=sidecar_count,
                profile="lane_a_step3_timeout_advisor",
                expected_lane="a",
            )
        if (
            attempt == 0
            and ack.get("status") == "IDENTITY_MISMATCH"
            and ack.get("mismatched_field") == "sequence_id"
            and ack.get("episode_id") == context["episode_id"]
            and ack.get("reset_generation") == context["reset_generation"]
            and isinstance(ack.get("sequence_id"), int)
            and not isinstance(ack.get("sequence_id"), bool)
            and ack["sequence_id"] >= sequence_id
        ):
            sequence_id = int(ack["sequence_id"])
            continue
        raise TimeoutAdvisorError(f"snapshot capture failed: {ack.get('status')}")
    raise TimeoutAdvisorError("snapshot identity did not converge")


def _planner_request(
    context: dict[str, Any], capture: dict[str, Any], result_root: Path
) -> SlowPlannerRequest:
    sidecar = _load_regular_json(result_root / capture["sidecar"], "snapshot sidecar")
    images_by_id = {
        str(row["identity"]): result_root / str(row["path"])
        for row in capture["images"]
    }
    ordered = tuple(
        OrderedImage(
            view_id=view_id,
            pose=CAMERA_POSES[view_id],
            jpeg=images_by_id[view_id].read_bytes(),
            width=640,
            height=480,
        )
        for view_id in CAMERA_ORDER
    )
    if context["kind"] == "motion_timeout_after_confirmed_safe_stop":
        candidates = tuple(
            primitive
            for action, primitive in PRIMITIVES.items()
            if action != int(context["excluded_action"])
        )
        instruction = (
            f"{context['instruction']} Previous bounded action "
            f"{context['excluded_action']} timed out after confirmed safe-stop. "
            "Choose one listed short primitive that changes the local motion and "
            "best escapes the visible blockage; do not stop and do not invent a goal."
        )
        history = (
            f"timed_out_action={context['excluded_action']}",
            f"snapshot_sim_stamp_ns={sidecar['sim_stamp_before_ns']}",
        )
    else:
        candidates = tuple(PRIMITIVES.values())
        instruction = (
            f"{context['instruction']} The robot is physically safe-stopped after "
            f"bounded action {context['completed_action']}. Decide only whether the "
            "instruction's destination is visibly reached now. Return target_found "
            "only when the current four-camera evidence clearly shows arrival; "
            "otherwise abstain. Do not select a movement primitive."
        )
        history = (
            f"arrival_confirmation_round={context['advisor_round']}",
            f"required_confirmations={context['required_confirmations']}",
            f"snapshot_sim_stamp_ns={sidecar['sim_stamp_before_ns']}",
        )
    snapshot_id = (
        f"a::{context['episode_id']}::{context['reset_generation']}::"
        f"{context['trigger_sequence_id']}"
    )
    return SlowPlannerRequest(
        episode_id=context["episode_id"],
        snapshot_id=snapshot_id,
        instruction=instruction,
        ordered_images=ordered,
        candidate_frontiers=candidates,
        agent_pose=(0.0, 0.0, 0.0, 0.0),
        compact_history=history,
        timestamp=time.time(),
    )


def _arrival_outcome(
    decision: Any, minimum_confidence: float
) -> tuple[str, str]:
    if (
        decision.decision == "target_found"
        and not decision.fallback_used
        and float(decision.confidence) >= minimum_confidence
    ):
        return "ARRIVED", "step3_visible_destination_reached"
    return "NOT_ARRIVED", "step3_arrival_not_confirmed"


class TimeoutAdvisorNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("t5_step3_timeout_advisor")
        self.args = args
        self._lock = threading.Lock()
        self._active = False
        self._active_identity: tuple[str, int, int, str, int] | None = None
        self._queued_arrival_confirmation: dict[str, Any] | None = None
        self.publisher = self.create_publisher(
            String, "/internvla/t5_step3_timeout_advice", 10
        )
        self.create_subscription(
            String, "/internvla/t5_step3_timeout_context", self._on_context, 10
        )

    def _publish(self, context: dict[str, Any], **fields: Any) -> None:
        value = {
            "schema_version": 1,
            "episode_id": context["episode_id"],
            "reset_generation": context["reset_generation"],
            "trigger_sequence_id": context["trigger_sequence_id"],
            "trigger_request_id": context["trigger_request_id"],
            "advisor_round": context["advisor_round"],
            **fields,
        }
        message = String()
        message.data = json.dumps(value, sort_keys=True, separators=(",", ":"))
        self.publisher.publish(message)
        _append_jsonl(self.args.output, {"wall_time_unix": time.time(), **value})

    def _on_context(self, message: String) -> None:
        try:
            context = _context(json.loads(str(message.data)))
        except (TypeError, ValueError, json.JSONDecodeError, TimeoutAdvisorError) as exc:
            self.get_logger().warning(f"discarding timeout context: {exc}")
            return
        identity = (
            context["episode_id"],
            context["reset_generation"],
            context["trigger_sequence_id"],
            context["trigger_request_id"],
            context["advisor_round"],
        )
        with self._lock:
            if self._active:
                if identity == self._active_identity:
                    return
                if (
                    self._active_identity is not None
                    and identity[:4] == self._active_identity[:4]
                    and self._active_identity[4] == 1
                    and identity[4] == 2
                    and self._queued_arrival_confirmation is None
                ):
                    self._queued_arrival_confirmation = context
                    return
                self._publish(
                    context,
                    status="FALLBACK",
                    confidence=0.0,
                    reason="advisor_request_already_active",
                )
                return
            self._active = True
            self._active_identity = identity
        threading.Thread(target=self._work, args=(context,), daemon=True).start()

    def _work(self, context: dict[str, Any]) -> None:
        started = time.monotonic()
        deadline = started + self.args.deadline_sec
        try:
            capture = _request_snapshot(
                context=context,
                request_path=self.args.request,
                ack_path=self.args.ack,
                result_root=self.args.result_root,
                contract_path=self.args.contract,
                deadline=deadline,
            )
            request = _planner_request(context, capture, self.args.result_root)
            sidecar = _load_regular_json(
                self.args.result_root / capture["sidecar"], "snapshot sidecar"
            )
            snapshot_sim_stamp_ns = int(sidecar["sim_stamp_before_ns"])
            if snapshot_sim_stamp_ns <= int(
                context.get("minimum_snapshot_sim_stamp_ns", 0)
            ):
                raise TimeoutAdvisorError(
                    "arrival snapshot did not advance in simulation time"
                )
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000.0))
            if remaining_ms <= 1:
                raise TimeoutAdvisorError("no wall budget remains for Step3")
            with SlowPlannerClient(self.args.endpoint, timeout_ms=remaining_ms) as client:
                decision, metrics = client.decide(request)
            if context["kind"] == "arrival_check_after_completed_motion":
                status, reason = _arrival_outcome(
                    decision, self.args.arrival_minimum_confidence
                )
                self._publish(
                    context,
                    status=status,
                    confidence=float(decision.confidence),
                    reason=reason,
                    snapshot_id=request.snapshot_id,
                    snapshot_sim_stamp_ns=snapshot_sim_stamp_ns,
                    camera_count=len(request.ordered_images),
                    service_wall_latency_sec=time.monotonic() - started,
                    metrics=metrics.to_mapping(),
                )
            elif (
                decision.decision != "select_frontier"
                or decision.frontier_id not in {
                    item.frontier_id for item in request.candidate_frontiers
                }
                or decision.confidence < self.args.minimum_confidence
                or decision.fallback_used
            ):
                self._publish(
                    context,
                    status="FALLBACK",
                    confidence=float(decision.confidence),
                    reason="step3_abstain_or_low_confidence",
                    snapshot_id=request.snapshot_id,
                    snapshot_sim_stamp_ns=snapshot_sim_stamp_ns,
                    camera_count=len(request.ordered_images),
                    service_wall_latency_sec=time.monotonic() - started,
                )
            else:
                self._publish(
                    context,
                    status="ADVISE",
                    advised_action=int(decision.frontier_id),
                    confidence=float(decision.confidence),
                    reason="bounded_visible_escape_primitive",
                    snapshot_id=request.snapshot_id,
                    snapshot_sim_stamp_ns=snapshot_sim_stamp_ns,
                    camera_count=len(request.ordered_images),
                    service_wall_latency_sec=time.monotonic() - started,
                    metrics=metrics.to_mapping(),
                )
        except BaseException as exc:
            self._publish(
                context,
                status="FALLBACK",
                confidence=0.0,
                reason=f"advisor_failure:{type(exc).__name__}"[:128],
                service_wall_latency_sec=time.monotonic() - started,
            )
            self.get_logger().warning(f"Step3 timeout advisor fallback: {exc}")
        finally:
            queued: dict[str, Any] | None
            with self._lock:
                self._active = False
                self._active_identity = None
                queued = self._queued_arrival_confirmation
                self._queued_arrival_confirmation = None
                if queued is not None:
                    self._active = True
                    self._active_identity = (
                        queued["episode_id"],
                        queued["reset_generation"],
                        queued["trigger_sequence_id"],
                        queued["trigger_request_id"],
                        queued["advisor_round"],
                    )
            if queued is not None:
                threading.Thread(
                    target=self._work, args=(queued,), daemon=True
                ).start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:8202")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--ack", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-sec", type=float, default=12.0)
    parser.add_argument("--minimum-confidence", type=float, default=0.35)
    parser.add_argument("--arrival-minimum-confidence", type=float, default=0.80)
    args = parser.parse_args()
    if not 1.0 <= args.deadline_sec <= 12.0:
        parser.error("deadline must be in [1, 12] seconds")
    if not 0.35 <= args.minimum_confidence <= 1.0:
        parser.error("minimum confidence must be in [0.35, 1]")
    if not 0.80 <= args.arrival_minimum_confidence <= 1.0:
        parser.error("arrival minimum confidence must be in [0.80, 1]")
    args.result_root = args.result_root.resolve()
    for path in (args.request, args.ack, args.output):
        if not path.resolve().is_relative_to(args.result_root):
            parser.error("runtime paths must stay under result root")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = TimeoutAdvisorNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
