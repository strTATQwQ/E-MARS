"""T4 adapter wrapper exposing cancellation only for the recovery owner."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from internvla_nav2_adapter.active_node import ActiveNav2Adapter, _semantic_age
from internvla_ros2.recovery_contract import (
    OP_CANCEL_AND_DISABLE,
    OP_REQUEST_REPLAN,
    STATUS_CONFLICT,
    STATUS_INTERNAL,
    STATUS_TIMEOUT,
    RecoveryContractError,
    goal_identity,
    initialize_response,
    reject_response,
    response_snapshot,
    restore_response,
    system2_replan_policy,
    system2_primitive_signature,
    t5_completion_sim_enabled,
    trajectory_signature,
    validate_request,
)
from internvla_ros2_msgs.srv import RecoveryControl, ResolveCommand
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger


class T4RecoveryAdapter(ActiveNav2Adapter):
    # A recovery latch must never hold the evaluator for the full episode when
    # the model keeps returning the excluded local action.  Three committed
    # identities are enough to prove that no immediately fresh path is queued;
    # the current command remains stopped and the next observation is allowed
    # to ask the model for a new action.
    RECOVERY_EXCLUDED_HOLD_LIMIT = 3
    RECOVERY_BOUNDED_FALLBACK_MESSAGE = (
        "recovery latch bounded fallback requires fresh trajectory"
    )

    def __init__(self) -> None:
        # ActiveNav2Adapter writes its READY summary from its constructor.
        # Guard our override until every T4-only field exists; otherwise the
        # parent constructor dispatches into a partially initialized subclass.
        self._t4_summary_ready = False
        super().__init__()
        self._recovery_uses_sim_time = t5_completion_sim_enabled()
        if self._recovery_uses_sim_time and not bool(
            self.get_parameter("use_sim_time").value
        ):
            raise RuntimeError("T5 completion_sim recovery requires use_sim_time=true")
        self._last_recovery_contract_sim_ns = 0
        self.declare_parameter("recovery_safety_freshness_timeout_sec", 1.0)
        self.recovery_safety_freshness_timeout = float(
            self.get_parameter("recovery_safety_freshness_timeout_sec").value
        )
        if not 0.05 <= self.recovery_safety_freshness_timeout <= 2.0:
            raise RuntimeError("invalid recovery safety freshness timeout")
        self.allow_system2_recovery_replan = (
            os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
            and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
        )
        self.system2_replan_policy = system2_replan_policy()
        factor_defaults = {
            "system_mode": "full_system1_system2",
            "trajectory_mode": "full_trajectory",
            "termination_mode": "model_stop",
            "history_mode": "on",
            "recovery_mode": "off",
            "view_mode": "go2_view",
        }
        factor_environment = {
            "system_mode": "INTERNVLA_T4_SYSTEM_MODE",
            "trajectory_mode": "INTERNVLA_T4_TRAJECTORY_MODE",
            "termination_mode": "INTERNVLA_T4_TERMINATION_MODE",
            "history_mode": "INTERNVLA_T4_HISTORY_MODE",
            "recovery_mode": "INTERNVLA_T4_RECOVERY_MODE",
            "view_mode": "INTERNVLA_T4_VIEW_MODE",
        }
        self.factors: dict[str, str] = {}
        for name, default in factor_defaults.items():
            self.declare_parameter(name, os.environ.get(factor_environment[name], default))
            self.factors[name] = str(self.get_parameter(name).value)
        allowed = {
            "system_mode": {
                "full_system1_system2",
                "oracle_high_level_system1",
                "system2_oracle_local_path",
            },
            "trajectory_mode": {"full_trajectory", "endpoint", "straight_line"},
            "termination_mode": {"model_stop", "oracle_termination"},
            "history_mode": {"on", "off"},
            "recovery_mode": {"on", "off"},
            "view_mode": {"go2_view", "h1_view"},
        }
        for name, levels in allowed.items():
            if self.factors[name] not in levels:
                raise RuntimeError(f"unsupported {name}: {self.factors[name]}")
        self.declare_parameter(
            "ablation_variant_id",
            os.environ.get("INTERNVLA_T4_VARIANT_ID", "none"),
        )
        self.declare_parameter(
            "ablation_config_sha256",
            os.environ.get("INTERNVLA_T4_VARIANT_CONFIG_SHA256", "none"),
        )
        self.declare_parameter(
            "ablation_dataset_file",
            os.environ.get("INTERNVLA_T4_ABLATION_DATASET_FILE", ""),
        )
        self.ablation_variant_id = str(
            self.get_parameter("ablation_variant_id").value
        )
        self.ablation_config_sha256 = str(
            self.get_parameter("ablation_config_sha256").value
        )
        if self.ablation_variant_id != "none" and (
            len(self.ablation_config_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.ablation_config_sha256
            )
        ):
            raise RuntimeError("ablation_config_sha256 must identify the generated config")
        dataset_value = str(self.get_parameter("ablation_dataset_file").value)
        self.ablation_episodes: list[dict[str, object]] = []
        oracle_required = (
            self.factors["system_mode"] != "full_system1_system2"
            or self.factors["termination_mode"] == "oracle_termination"
        )
        if oracle_required:
            dataset = Path(dataset_value).resolve()
            if not dataset.is_file():
                raise FileNotFoundError("ablation_dataset_file is required")
            with gzip.open(dataset, "rt", encoding="utf-8") as stream:
                self.ablation_episodes = list(json.load(stream)["episodes"])
            if not self.ablation_episodes:
                raise RuntimeError("ablation dataset has no episodes")
        self.latest_system2_action: int | None = None
        self.latest_non_stop_system2_action: int | None = None
        self.activation_counts = {
            "system": 0,
            "trajectory": 0,
            "termination": 0,
        }
        self.recovery_cancel_count = 0
        self.recovery_latched = False
        self.recovery_latched_id = ""
        self.recovery_latched_goal_id = ""
        self.recovery_excluded_absolute: set[str] = set()
        self.recovery_excluded_shape: set[str] = set()
        self.recovery_replan_armed = False
        self.recovery_replan_deadline_ns = 0
        self.recovery_excluded_hold_count = 0
        self._typed_recovery_results: dict[
            str, tuple[tuple[object, ...], dict[str, object]]
        ] = {}
        self.create_service(
            RecoveryControl,
            "/internvla/t4_recovery_adapter",
            self._on_typed_recovery,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/internvla/t4_cancel_nav2",
            self._on_recovery_cancel,
            callback_group=self.callback_group,
        )
        self._t4_summary_ready = True
        self._write_summary("READY")

    def _recovery_contract_now_ns(self) -> int:
        if not self._recovery_uses_sim_time:
            return time.time_ns()
        value = int(self.get_clock().now().nanoseconds)
        if value <= 0 or value < self._last_recovery_contract_sim_ns:
            raise RecoveryContractError(
                STATUS_TIMEOUT,
                "recovery simulation clock is unavailable or regressed",
            )
        self._last_recovery_contract_sim_ns = value
        return value

    @staticmethod
    def _yaw_from_xyzw(x: float, y: float, z: float, w: float) -> float:
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _episode_reference(
        self, episode_id: str, generation: int
    ) -> list[tuple[float, float]]:
        if generation < 0 or generation >= len(self.ablation_episodes):
            raise TimeoutError(
                "ablation generation is outside the frozen episode order"
            )
        episode = self.ablation_episodes[generation]
        if str(episode.get("episode_id", "")) != str(episode_id):
            raise TimeoutError("ablation episode identity does not match reset generation")
        start = [float(value) for value in episode["start_position"]]
        rotation = [float(value) for value in episode["start_rotation"]]
        habitat_yaw = 2.0 * math.atan2(rotation[1], rotation[3])
        start_yaw = math.atan2(
            math.sin(math.pi / 2.0 - habitat_yaw),
            math.cos(math.pi / 2.0 - habitat_yaw),
        )
        start_xy = (start[0], -start[2])
        cosine, sine = math.cos(start_yaw), math.sin(start_yaw)
        output: list[tuple[float, float]] = []
        for point in episode["reference_path"]:
            dx = float(point[0]) - start_xy[0]
            dy = -float(point[2]) - start_xy[1]
            output.append((cosine * dx + sine * dy, -sine * dx + cosine * dy))
        return output

    def _oracle_local_path(
        self, episode_id: str, generation: int
    ) -> tuple[list[tuple[float, float]], float]:
        with self.odom_lock:
            odom = self.latest_odom
        if odom is None:
            raise TimeoutError
        current = (odom[0], odom[1])
        reference = self._episode_reference(episode_id, generation)
        nearest = min(
            range(len(reference)),
            key=lambda index: math.hypot(
                reference[index][0] - current[0], reference[index][1] - current[1]
            ),
        )
        route = [current, *reference[nearest:]]
        selected = [route[0]]
        distance = 0.0
        for first, second in zip(route, route[1:]):
            segment = math.hypot(second[0] - first[0], second[1] - first[1])
            if segment <= 1e-8:
                continue
            if distance + segment <= 1.0:
                selected.append(second)
                distance += segment
                continue
            ratio = (1.0 - distance) / segment
            selected.append(
                (
                    first[0] + ratio * (second[0] - first[0]),
                    first[1] + ratio * (second[1] - first[1]),
                )
            )
            break
        if len(selected) == 1:
            selected.append(reference[-1])
        cosine, sine = math.cos(odom[2]), math.sin(odom[2])
        local = [
            (
                cosine * (point[0] - current[0]) + sine * (point[1] - current[1]),
                -sine * (point[0] - current[0]) + cosine * (point[1] - current[1]),
            )
            for point in selected
        ]
        goal_distance = math.hypot(
            reference[-1][0] - current[0], reference[-1][1] - current[1]
        )
        return local, goal_distance

    @staticmethod
    def _set_local_path(command: object, points: list[tuple[float, float]]) -> None:
        command.local_path.poses.clear()
        command.local_path.header.frame_id = "base_link"
        for x, y in points:
            pose = PoseStamped()
            pose.header = command.local_path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            command.local_path.poses.append(pose)
        command.trajectory_valid = bool(points)
        command.trajectory_source = 1 if points else 0

    @staticmethod
    def _first_action(points: list[tuple[float, float]]) -> int:
        target = next((point for point in points if math.hypot(*point) > 0.15), (0.0, 0.0))
        angle = math.atan2(target[1], target[0])
        if angle > 0.35:
            return 2
        if angle < -0.35:
            return 3
        return 1

    def _apply_ablation(self, request: ResolveCommand.Request) -> None:
        command = request.command
        episode_id = str(command.episode_id)
        generation = int(command.reset_generation)
        original = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in command.local_path.poses
        ]
        original_digest = hashlib.sha256(
            json.dumps(original, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        original_stop = bool(command.stop) or int(command.discrete_action) == 0
        original_action = int(command.discrete_action)
        original_action_source = int(command.action_source)
        if int(command.action_source) == 1:
            self.latest_system2_action = int(command.discrete_action)
            if int(command.discrete_action) != 0:
                self.latest_non_stop_system2_action = int(command.discrete_action)

        oracle_distance: float | None = None
        no_real_system2_fallback = False
        if self.factors["system_mode"] == "oracle_high_level_system1":
            oracle, oracle_distance = self._oracle_local_path(episode_id, generation)
            if not bool(command.trajectory_valid):
                command.discrete_action = (
                    0 if oracle_distance <= 2.5 else self._first_action(oracle)
                )
                command.stop = oracle_distance <= 2.5
            self.activation_counts["system"] += 1
        elif self.factors["system_mode"] == "system2_oracle_local_path":
            oracle, oracle_distance = self._oracle_local_path(episode_id, generation)
            if self.latest_system2_action is None:
                # Never synthesize the historical default-forward action before
                # a real System2 output has been observed.
                command.discrete_action = 0
                command.stop = True
                self._set_local_path(command, [])
                no_real_system2_fallback = True
            else:
                command.discrete_action = self.latest_system2_action
                command.stop = self.latest_system2_action == 0 or oracle_distance <= 2.5
                self._set_local_path(command, [] if command.stop else oracle)
            self.activation_counts["system"] += 1

        if self.factors["trajectory_mode"] == "endpoint" and original:
            self._set_local_path(command, [(0.0, 0.0), original[-1]])
            self.activation_counts["trajectory"] += 1
        elif self.factors["trajectory_mode"] == "straight_line" and original:
            endpoint = original[-1]
            self._set_local_path(
                command,
                [
                    (endpoint[0] * index / 8.0, endpoint[1] * index / 8.0)
                    for index in range(9)
                ],
            )
            self.activation_counts["trajectory"] += 1

        oracle_continuation_unavailable = False
        if self.factors["termination_mode"] == "oracle_termination":
            _, oracle_distance = self._oracle_local_path(episode_id, generation)
            command.stop = oracle_distance <= 2.5
            if command.stop:
                command.discrete_action = 0
                self._set_local_path(command, [])
            elif int(command.discrete_action) == 0:
                if self.latest_non_stop_system2_action is None:
                    # No pre-System2 default-forward action: remain safely
                    # stopped and expose that the oracle arm could not continue.
                    command.stop = True
                    oracle_continuation_unavailable = True
                else:
                    command.discrete_action = self.latest_non_stop_system2_action
            self.activation_counts["termination"] += 1

        if self.ablation_variant_id != "none":
            self._append(
                {
                    "schema_version": 1,
                    "event": "ablation_transform",
                    "variant_id": self.ablation_variant_id,
                    "config_sha256": self.ablation_config_sha256,
                    "factors": dict(self.factors),
                    "episode_id": episode_id,
                    "reset_generation": generation,
                    "sequence_id": int(command.sequence_id),
                    "original_path_sha256": original_digest,
                    "original_point_count": len(original),
                    "transformed_point_count": len(command.local_path.poses),
                    "original_model_stop": original_stop,
                    "original_discrete_action": original_action,
                    "original_action_source": original_action_source,
                    "oracle_threshold_m": 2.5 if oracle_distance is not None else None,
                    "oracle_threshold_source": (
                        "frozen_evaluator_success_radius"
                        if oracle_distance is not None
                        else None
                    ),
                    "oracle_distance_source": (
                        "evaluator_ground_truth" if oracle_distance is not None else None
                    ),
                    "oracle_distance_m": oracle_distance,
                    "no_real_system2_fallback": no_real_system2_fallback,
                    "oracle_continuation_unavailable": oracle_continuation_unavailable,
                    "activation_counts": dict(self.activation_counts),
                }
            )

    def _resolve(
        self, request: ResolveCommand.Request, response: ResolveCommand.Response
    ) -> ResolveCommand.Response:
        command = request.command
        with self.operation_lock:
            if self.recovery_latched:
                identity_changed = (
                    str(command.episode_id) != self.active_episode
                    or int(command.reset_generation) != self.active_generation
                )
                if identity_changed:
                    self._append(
                        {
                            "schema_version": 1,
                            "event": "recovery_latch_reset_boundary",
                            "episode_id": str(command.episode_id),
                            "reset_generation": int(command.reset_generation),
                            "sequence_id": int(command.sequence_id),
                            "recovery_id": self.recovery_latched_id,
                        }
                    )
                    self.recovery_latched = False
                    self.recovery_latched_id = ""
                    self.recovery_latched_goal_id = ""
                    self.recovery_replan_armed = False
                    self.recovery_replan_deadline_ns = 0
                    self.recovery_excluded_hold_count = 0
                    self.recovery_excluded_absolute.clear()
                    self.recovery_excluded_shape.clear()
                    self._publish_motion(False)
                else:
                    return self._resolve_recovery_latched(request, response)
        self._apply_ablation(request)
        return super()._resolve(request, response)

    def _resolve_recovery_latched(
        self,
        request: ResolveCommand.Request,
        response: ResolveCommand.Response,
    ) -> ResolveCommand.Response:
        command = request.command
        # Caller holds operation_lock, so the latch cannot be armed or cleared
        # between fingerprint validation and the decision below.
        if (
            self.recovery_replan_armed
            and self.recovery_replan_deadline_ns > 0
            and self._recovery_contract_now_ns() >= self.recovery_replan_deadline_ns
        ):
            # Consume this exact command identity without executing it.  Both
            # recovery consumers received the same semantic deadline, so an
            # expired request cannot race into Nav2 after the client timed out.
            self._check_identity(command)
            response.status_code = 0
            response.status_message = self.RECOVERY_BOUNDED_FALLBACK_MESSAGE
            response.episode_id = str(command.episode_id)
            response.reset_generation = int(command.reset_generation)
            response.sequence_id = int(command.sequence_id)
            response.request_id = str(command.request_id)
            response.discrete_action = -1
            response.stop = False
            response.nav2_goal_sent = False
            response.nav2_plan_valid = False
            response.resolution_latency_sec = 0.0
            self._append(
                {
                    "schema_version": 1,
                    "event": "recovery_latch_deadline_bounded_fallback",
                    "episode_id": str(command.episode_id),
                    "reset_generation": int(command.reset_generation),
                    "sequence_id": int(command.sequence_id),
                    "recovery_id": self.recovery_latched_id,
                    "deadline_ns": self.recovery_replan_deadline_ns,
                    "motion_disabled": True,
                }
            )
            self.recovery_latched = False
            self.recovery_latched_id = ""
            self.recovery_latched_goal_id = ""
            self.recovery_replan_armed = False
            self.recovery_replan_deadline_ns = 0
            self.recovery_excluded_hold_count = 0
            self.recovery_excluded_absolute.clear()
            self.recovery_excluded_shape.clear()
            self._publish_motion(False)
            return response
        points = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in command.local_path.poses
        ]
        try:
            signature = trajectory_signature(points)
        except (TypeError, ValueError):
            signature = None
        signature_source = "local_path"
        fresh_system2_action = bool(
            self.allow_system2_recovery_replan
            and not bool(command.stop)
            and int(command.action_source) == 1
            and int(command.discrete_action) in {1, 2, 3}
        )
        if signature is None and fresh_system2_action:
            try:
                generated_path = self._system2_path(
                    command, int(command.discrete_action)
                )
                signature = trajectory_signature(
                    [
                        (
                            float(pose.pose.position.x),
                            float(pose.pose.position.y),
                        )
                        for pose in generated_path.poses
                    ]
                )
                signature_source = "system2_generated_path"
            except (TimeoutError, TypeError, ValueError):
                signature = None
        primitive_signature = None
        if fresh_system2_action:
            with self.odom_lock:
                odom = self.latest_odom
            if odom is not None:
                try:
                    primitive_signature = system2_primitive_signature(
                        action=int(command.discrete_action),
                        episode_id=str(command.episode_id),
                        reset_generation=int(command.reset_generation),
                        sequence_id=int(command.sequence_id),
                        observation_digest=str(command.observation_digest),
                        x=float(odom[0]),
                        y=float(odom[1]),
                        yaw_rad=float(odom[2]),
                    )
                except (IndexError, TypeError, ValueError):
                    primitive_signature = None
        policy = getattr(self, "system2_replan_policy", "observation_bound")
        # T5 turn primitives rotate about the footprint centre, so all of
        # their generated path samples intentionally share the same x/y.
        # observation_bound therefore requires the pose/action/observation
        # primitive and treats the generated path as supplementary.  strict
        # reproduces the original dual-signature gate.  raw_wire_warn is an
        # explicitly scoped completion_sim diagnostic: it records what the
        # signature gate would reject but lets a finite, identity-bound
        # System2 primitive reach the existing Nav2 adapter.
        required_signature = primitive_signature if fresh_system2_action else signature
        signatures = tuple(
            candidate
            for candidate in (
                (signature, primitive_signature)
                if fresh_system2_action
                else (signature,)
            )
            if candidate is not None
        )
        evidence_signature = signature or primitive_signature
        evidence_source = (
            signature_source if signature is not None else "system2_primitive"
        )
        signature_fresh = bool(
            self.recovery_replan_armed
            and required_signature is not None
            and (
                not fresh_system2_action
                or policy != "strict"
                or signature is not None
            )
            and all(
                candidate.absolute_sha256 not in self.recovery_excluded_absolute
                and candidate.shape_sha256 not in self.recovery_excluded_shape
                for candidate in signatures
            )
        )
        raw_wire_safe = bool(
            fresh_system2_action
            and self.recovery_replan_armed
            and self.latest_odom is not None
            and all(math.isfinite(float(value)) for value in self.latest_odom)
        )
        fresh = bool(
            raw_wire_safe if policy == "raw_wire_warn" else signature_fresh
        )
        if policy == "raw_wire_warn" and raw_wire_safe and not signature_fresh:
            self._append(
                {
                    "schema_version": 1,
                    "event": "raw_wire_system2_signature_bypass_warn",
                    "episode_id": str(command.episode_id),
                    "reset_generation": int(command.reset_generation),
                    "sequence_id": int(command.sequence_id),
                    "recovery_id": self.recovery_latched_id,
                    "absolute_sha256": (
                        evidence_signature.absolute_sha256
                        if evidence_signature is not None
                        else None
                    ),
                    "shape_sha256": (
                        evidence_signature.shape_sha256
                        if evidence_signature is not None
                        else None
                    ),
                    "trajectory_source": evidence_source,
                    "deviation": "signature_and_semantic_rejection_warn_only",
                }
            )
        if not fresh:
            excluded_after_arm = bool(
                self.recovery_replan_armed
                and any(
                    candidate is not None
                    and (
                        candidate.absolute_sha256
                        in self.recovery_excluded_absolute
                        or candidate.shape_sha256 in self.recovery_excluded_shape
                    )
                    for candidate in signatures
                )
            )
            if excluded_after_arm:
                self.recovery_excluded_hold_count += 1
            try:
                # Suppression is still a committed response for this command
                # identity.  Advance only the episode/sequence barrier so the
                # next fresh command is contiguous; no goal or motion is sent.
                self._check_identity(command)
            except ValueError as exc:
                response.status_code = 2
                response.status_message = f"recovery latch identity error: {exc}"
                response.episode_id = str(command.episode_id)
                response.reset_generation = int(command.reset_generation)
                response.sequence_id = int(command.sequence_id)
                response.request_id = str(command.request_id)
                response.discrete_action = -1
                response.stop = False
                response.nav2_goal_sent = False
                response.nav2_plan_valid = False
                response.resolution_latency_sec = 0.0
                return response
            response.status_code = 0
            response.status_message = (
                "recovery latch holding until replan gate is armed"
                if not self.recovery_replan_armed
                else "recovery latch suppressed excluded trajectory"
            )
            response.episode_id = str(command.episode_id)
            response.reset_generation = int(command.reset_generation)
            response.sequence_id = int(command.sequence_id)
            response.request_id = str(command.request_id)
            response.discrete_action = -1
            response.stop = False
            response.nav2_goal_sent = False
            response.nav2_plan_valid = False
            response.resolution_latency_sec = 0.0
            bounded_fallback = bool(
                excluded_after_arm
                and self.recovery_excluded_hold_count
                >= self.RECOVERY_EXCLUDED_HOLD_LIMIT
            )
            if bounded_fallback:
                response.status_message = self.RECOVERY_BOUNDED_FALLBACK_MESSAGE
            self._append(
                {
                    "schema_version": 1,
                    "event": "recovery_latch_suppressed_trajectory",
                    "episode_id": str(command.episode_id),
                    "reset_generation": int(command.reset_generation),
                    "sequence_id": int(command.sequence_id),
                    "recovery_id": self.recovery_latched_id,
                    "absolute_sha256": (
                        evidence_signature.absolute_sha256
                        if evidence_signature is not None
                        else None
                    ),
                    "shape_sha256": (
                        evidence_signature.shape_sha256
                        if evidence_signature is not None
                        else None
                    ),
                    "trajectory_source": evidence_source,
                    "excluded_hold_count": self.recovery_excluded_hold_count,
                }
            )
            if bounded_fallback:
                self._append(
                    {
                        "schema_version": 1,
                        "event": "recovery_latch_bounded_fallback",
                        "episode_id": str(command.episode_id),
                        "reset_generation": int(command.reset_generation),
                        "sequence_id": int(command.sequence_id),
                        "recovery_id": self.recovery_latched_id,
                        "excluded_hold_count": self.recovery_excluded_hold_count,
                        "trajectory_source": evidence_source,
                        "motion_disabled": True,
                    }
                )
                self.recovery_latched = False
                self.recovery_latched_id = ""
                self.recovery_latched_goal_id = ""
                self.recovery_replan_armed = False
                self.recovery_replan_deadline_ns = 0
                self.recovery_excluded_hold_count = 0
                self.recovery_excluded_absolute.clear()
                self.recovery_excluded_shape.clear()
                self._publish_motion(False)
            return response
        try:
            self._apply_ablation(request)
            resolved = super()._resolve(request, response)
            if not (
                int(resolved.status_code) == 0
                and bool(resolved.nav2_goal_sent)
                and bool(resolved.nav2_plan_valid)
            ):
                self._publish_motion(False)
                return resolved
            self._append(
                {
                    "schema_version": 1,
                    "event": "recovery_latch_fresh_trajectory_accepted",
                    "episode_id": str(command.episode_id),
                    "reset_generation": int(command.reset_generation),
                    "sequence_id": int(command.sequence_id),
                    "recovery_id": self.recovery_latched_id,
                    "absolute_sha256": (
                        evidence_signature.absolute_sha256
                        if evidence_signature is not None
                        else None
                    ),
                    "shape_sha256": (
                        evidence_signature.shape_sha256
                        if evidence_signature is not None
                        else None
                    ),
                    "trajectory_source": evidence_source,
                    "system2_replan_policy": policy,
                }
            )
        except BaseException:
            self._publish_motion(False)
            raise
        self.recovery_latched = False
        self.recovery_latched_id = ""
        self.recovery_latched_goal_id = ""
        self.recovery_replan_armed = False
        self.recovery_replan_deadline_ns = 0
        self.recovery_excluded_hold_count = 0
        self.recovery_excluded_absolute.clear()
        self.recovery_excluded_shape.clear()
        return resolved

    def _on_typed_recovery(
        self,
        request: RecoveryControl.Request,
        response: RecoveryControl.Response,
    ) -> RecoveryControl.Response:
        if int(request.operation) == OP_CANCEL_AND_DISABLE:
            return self._on_typed_recovery_cancel(request, response)
        if int(request.operation) == OP_REQUEST_REPLAN:
            return self._on_typed_replan_arm(request, response)
        initialize_response(response, request)
        response.status_code = 2
        response.status_message = "unsupported adapter recovery operation"
        return response

    def _on_recovery_cancel(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        try:
            with self.operation_lock:
                self._cancel_active(wait=True)
                self._publish_motion(False)
                self.recovery_cancel_count += 1
                self._append(
                    {
                        "schema_version": 1,
                        "event": "legacy_recovery_cancel",
                        "episode_id": self.active_episode,
                        "reset_generation": self.active_generation,
                        "last_sequence_id": self.last_sequence,
                        "recovery_cancel_count": self.recovery_cancel_count,
                    }
                )
            response.success = True
            response.message = "owned Nav2 goal canceled and motion disabled"
        except BaseException as exc:
            response.success = False
            response.message = f"Nav2 cancellation was not confirmed: {exc!r}"[:512]
        return response

    def _on_typed_recovery_cancel(
        self,
        request: RecoveryControl.Request,
        response: RecoveryControl.Response,
    ) -> RecoveryControl.Response:
        initialize_response(response, request)
        try:
            with self.operation_lock:
                identity = validate_request(
                    request,
                    expected_operation=OP_CANCEL_AND_DISABLE,
                    active_episode_id=self.active_episode,
                    active_reset_generation=self.active_generation,
                    active_goal_id=goal_identity(
                        self.active_episode,
                        self.active_generation,
                        self.last_sequence,
                    ),
                    now_ns=self._recovery_contract_now_ns(),
                )
                identity_key: tuple[object, ...] = (
                    identity.episode_id,
                    identity.reset_generation,
                    identity.goal_id,
                    identity.recovery_id,
                    identity.operation,
                )
                previous = self._typed_recovery_results.get(identity.operation_id)
                if previous is not None:
                    if previous[0] != identity_key:
                        response.status_code = STATUS_CONFLICT
                        response.status_message = "operation_id was already consumed"
                        return response
                    restore_response(response, previous[1])
                    return response
                self._cancel_active(wait=True)
                self._publish_motion(False)
                self.recovery_latched = True
                self.recovery_latched_id = identity.recovery_id
                self.recovery_latched_goal_id = identity.goal_id
                self.recovery_replan_armed = False
                self.recovery_replan_deadline_ns = 0
                self.recovery_excluded_hold_count = 0
                self.recovery_excluded_absolute.clear()
                self.recovery_excluded_shape.clear()
                self.recovery_cancel_count += 1
                with self.cmd_condition:
                    linear, angular = self.latest_cmd
                    cmd_age = _semantic_age(self, self.latest_cmd_monotonic)
                response.success = True
                response.status_code = 0
                response.status_message = "Nav2 goal canceled and motion disabled"
                response.motion_disabled = True
                response.measured_linear_speed_peak_mps = abs(float(linear))
                response.measured_angular_speed_peak_rps = abs(float(angular))
                response.safety_fresh = (
                    cmd_age is not None
                    and 0.0 <= cmd_age <= self.recovery_safety_freshness_timeout
                )
                self._append(
                    {
                        "schema_version": 1,
                        "event": "typed_recovery_cancel",
                        "episode_id": identity.episode_id,
                        "reset_generation": identity.reset_generation,
                        "goal_id": identity.goal_id,
                        "recovery_id": identity.recovery_id,
                        "operation_id": identity.operation_id,
                        "recovery_cancel_count": self.recovery_cancel_count,
                        "motion_disabled": response.motion_disabled,
                        "safety_fresh": response.safety_fresh,
                    }
                )
                if len(self._typed_recovery_results) >= 256:
                    self._typed_recovery_results.pop(
                        next(iter(self._typed_recovery_results))
                    )
                self._typed_recovery_results[identity.operation_id] = (
                    identity_key,
                    response_snapshot(response),
                )
        except RecoveryContractError as exc:
            reject_response(response, exc)
        except BaseException as exc:
            response.success = False
            response.status_code = STATUS_INTERNAL
            response.status_message = f"cancel failed: {exc!r}"[:512]
            self._publish_motion(False)
        return response

    def _on_typed_replan_arm(
        self,
        request: RecoveryControl.Request,
        response: RecoveryControl.Response,
    ) -> RecoveryControl.Response:
        initialize_response(response, request)
        try:
            with self.operation_lock:
                active_goal_id = goal_identity(
                    self.active_episode,
                    self.active_generation,
                    self.last_sequence,
                )
                if (
                    self.recovery_latched
                    and self.recovery_latched_id == str(request.recovery_id)
                ):
                    # Suppressed commands are committed identities so the next
                    # command remains contiguous, but they do not replace the
                    # goal identity frozen by this typed recovery transaction.
                    active_goal_id = self.recovery_latched_goal_id
                identity = validate_request(
                    request,
                    expected_operation=OP_REQUEST_REPLAN,
                    active_episode_id=self.active_episode,
                    active_reset_generation=self.active_generation,
                    active_goal_id=active_goal_id,
                    now_ns=self._recovery_contract_now_ns(),
                )
                identity_key: tuple[object, ...] = (
                    identity.episode_id,
                    identity.reset_generation,
                    identity.goal_id,
                    identity.recovery_id,
                    identity.operation,
                )
                previous = self._typed_recovery_results.get(identity.operation_id)
                if previous is not None:
                    if previous[0] != identity_key:
                        response.status_code = STATUS_CONFLICT
                        response.status_message = "operation_id was already consumed"
                        return response
                    restore_response(response, previous[1])
                    return response
                if (
                    not self.recovery_latched
                    or self.recovery_latched_id != identity.recovery_id
                ):
                    response.status_code = STATUS_CONFLICT
                    response.status_message = "recovery latch identity is not active"
                    return response
                absolute = set(request.excluded_absolute_sha256)
                shape = set(request.excluded_shape_sha256)
                if not absolute or not shape:
                    response.status_code = 2
                    response.status_message = "replan arm requires old trajectory signatures"
                    return response
                self.recovery_excluded_absolute = absolute
                self.recovery_excluded_shape = shape
                self.recovery_replan_armed = True
                self.recovery_replan_deadline_ns = identity.deadline_ns
                self.recovery_excluded_hold_count = 0
                response.success = True
                response.status_code = 0
                response.status_message = "fresh trajectory gate armed"
                response.motion_disabled = True
                self._append(
                    {
                        "schema_version": 1,
                        "event": "recovery_replan_gate_armed",
                        "episode_id": identity.episode_id,
                        "reset_generation": identity.reset_generation,
                        "goal_id": identity.goal_id,
                        "recovery_id": identity.recovery_id,
                        "operation_id": identity.operation_id,
                        "excluded_absolute_sha256": sorted(absolute),
                        "excluded_shape_sha256": sorted(shape),
                    }
                )
                if len(self._typed_recovery_results) >= 256:
                    self._typed_recovery_results.pop(
                        next(iter(self._typed_recovery_results))
                    )
                self._typed_recovery_results[identity.operation_id] = (
                    identity_key,
                    response_snapshot(response),
                )
        except RecoveryContractError as exc:
            with self.operation_lock:
                if (
                    int(exc.status_code) == STATUS_TIMEOUT
                    and str(exc) == "recovery operation deadline expired"
                    and self.recovery_latched
                    and self.recovery_latched_id == str(request.recovery_id)
                ):
                    # Validate every non-time identity field against the frozen
                    # transaction before releasing an arm request that arrived
                    # after the shared deadline.  It remains a timeout response,
                    # but cannot leave an unarmed latch holding the episode.
                    deadline_ns = (
                        int(request.deadline.sec) * 1_000_000_000
                        + int(request.deadline.nanosec)
                    )
                    try:
                        validate_request(
                            request,
                            expected_operation=OP_REQUEST_REPLAN,
                            active_episode_id=self.active_episode,
                            active_reset_generation=self.active_generation,
                            active_goal_id=self.recovery_latched_goal_id,
                            now_ns=max(1, deadline_ns - 1),
                        )
                    except RecoveryContractError:
                        pass
                    else:
                        self._append(
                            {
                                "schema_version": 1,
                                "event": (
                                    "recovery_latch_expired_before_arm_released"
                                ),
                                "episode_id": str(request.episode_id),
                                "reset_generation": int(request.reset_generation),
                                "recovery_id": str(request.recovery_id),
                                "deadline_ns": deadline_ns,
                                "motion_disabled": True,
                            }
                        )
                        self.recovery_latched = False
                        self.recovery_latched_id = ""
                        self.recovery_latched_goal_id = ""
                        self.recovery_replan_armed = False
                        self.recovery_replan_deadline_ns = 0
                        self.recovery_excluded_hold_count = 0
                        self.recovery_excluded_absolute.clear()
                        self.recovery_excluded_shape.clear()
                        self._publish_motion(False)
            reject_response(response, exc)
        except BaseException as exc:
            response.success = False
            response.status_code = STATUS_INTERNAL
            response.status_message = f"replan gate failed: {exc!r}"[:512]
        return response

    def _write_summary(self, status: str) -> None:
        if not self._t4_summary_ready:
            ActiveNav2Adapter._write_summary(self, status)
            return
        self._append(
            {
                "schema_version": 1,
                "event": "ablation_runtime_summary",
                "variant_id": self.ablation_variant_id,
                "config_sha256": self.ablation_config_sha256,
                "factors": dict(self.factors),
                "activation_counts": dict(self.activation_counts),
                "system2_replan_policy": self.system2_replan_policy,
            }
        )
        super()._write_summary(status)
        # Parent summary is intentionally authoritative. The event stream holds
        # each recovery cancellation without changing T3 adapter fields.


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = T4RecoveryAdapter()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._cancel_active()
            node._publish_motion(False)
        except BaseException:
            pass
        node._write_summary("FINISHED" if not node.failure_count else "FAIL")
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
