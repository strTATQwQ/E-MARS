"""Private T5 Lane-B coordinator for one bounded Step3 decision per episode."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from internnav_t5_lane_b_msgs.srv import CommitFrontier, PrepareFrontiers
from internvla_ros2.protocol import STATUS_OK
from internvla_ros2_msgs.msg import NavigationCommand
from internvla_ros2_msgs.srv import ResolveCommand
from slow_planner.base import CandidateFrontier, OrderedImage
from slow_planner.client import SlowPlannerClient
from slow_planner.lane_b import (
    LaneBDecisionSidecar,
    LaneBIntent,
    LaneBPlannerAdapter,
    LaneBPlannerMode,
    LaneBSnapshotSidecar,
    TimedRevCImage,
    build_lane_b_request,
    validate_rev_c_snapshot,
)


VIEW_ORDER = ("front_left", "front", "front_right", "rear")
MAX_IMAGE_BYTES = 256 * 1024
MAX_TOTAL_IMAGE_BYTES = 1024 * 1024


class DirectStep3Failure(RuntimeError):
    """A pure-direct decision that must terminate through coordinator safe-stop."""


def _future_result(future: Any, timeout_sec: float, description: str) -> Any:
    event = threading.Event()
    future.add_done_callback(lambda _: event.set())
    if not event.wait(timeout_sec):
        raise TimeoutError(f"{description} timeout")
    exception = future.exception()
    if exception is not None:
        raise exception
    return future.result()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_rev_c_contract(root: Path) -> tuple[str, dict[str, str], dict[str, tuple[float, ...]]]:
    path = root / "configs" / "internnav_t5" / "revc_four_camera_snapshot.json"
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if tuple(value.get("camera_order") or ()) != VIEW_ORDER:
        raise RuntimeError("Rev-C camera order drifted")
    frames = value["frames"]
    mast = frames["temporary_T_base_link_F_M"]
    translation = [float(item) for item in mast["translation_m"]]
    optics = value["optics"]
    extrinsics: dict[str, str] = {}
    poses: dict[str, tuple[float, ...]] = {}
    for camera in value["cameras"]:
        view_id = str(camera["identity"])
        position = [float(item) for item in camera["position_F_M_m"]]
        yaw = math.radians(float(camera["yaw_deg"]))
        pose = (
            translation[0] + position[0],
            translation[1] + position[1],
            translation[2] + position[2],
            yaw,
            math.radians(-float(optics["pitch_down_deg"])),
        )
        poses[view_id] = pose
        extrinsics[view_id] = _canonical_sha256(
            {
                "schema_version": 1,
                "base_frame": frames["base_frame"],
                "mast_frame": frames["mast_frame"],
                "temporary_T_base_link_F_M": mast,
                "camera": camera,
                "pitch_down_deg": optics["pitch_down_deg"],
            }
        )
    return hashlib.sha256(raw).hexdigest(), extrinsics, poses


class LaneBStep3LiveCoordinator:
    def __init__(self, node: Any) -> None:
        self.node = node
        control_root = Path(
            os.environ.get("INTERNNAV_T1_CONTROL_ROOT", "")
        ).resolve()
        if not control_root.is_dir():
            raise RuntimeError("Lane-B Step3 control root is unavailable")
        self.config_sha256, self.extrinsics, self.camera_poses = _load_rev_c_contract(
            control_root
        )
        result_root_text = os.environ.get("T5_LANE_B_RESULTS", "")
        result_root = (
            Path(result_root_text).resolve()
            if result_root_text
            else Path(node.result_dir).resolve().parent
        )
        self.result_root = result_root / "step3"
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.snapshot_writer = LaneBSnapshotSidecar(
            self.result_root / "snapshots.jsonl",
            camera_dir=self.result_root / "cameras",
        )
        self.decision_writer = LaneBDecisionSidecar(
            self.result_root / "frontend_decisions.jsonl"
        )
        self.health_path = self.result_root / "health.json"
        self.model_health_path = self.result_root / "model_health.json"
        self.endpoint = os.environ.get(
            "INTERNNAV_T5_STEP3_ENDPOINT", "tcp://127.0.0.1:8200"
        )
        if self.endpoint != "tcp://127.0.0.1:8200":
            raise RuntimeError("Lane-B Step3 endpoint drifted")
        self.timeout_sec = 12.0
        self.prepare_client = node.create_client(
            PrepareFrontiers, "step3/prepare_frontiers"
        )
        self.commit_client = node.create_client(
            CommitFrontier, "step3/commit_frontier"
        )
        direct_enabled = (
            os.environ.get("INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL", "0") == "1"
        )
        self.mode = (
            LaneBPlannerMode.DIRECT_HIGH_LEVEL
            if direct_enabled
            else LaneBPlannerMode.BOUNDED_ADVISOR
        )
        self.adapter = LaneBPlannerAdapter(self.mode)
        self._attempted_episodes: set[tuple[str, int]] = set()
        self._attempted_requests: set[tuple[str, int, int, str]] = set()
        self._write_health("READY", {})

    def _write_health(self, status: str, extra: Mapping[str, Any]) -> None:
        model_health: dict[str, Any] = {}
        try:
            loaded = json.loads(self.model_health_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                model_health = loaded
        except (OSError, ValueError):
            pass
        value = {
            **model_health,
            "schema_version": 1,
            "status": status,
            "ready": bool(model_health.get("ready", False)),
            "enabled": True,
            "endpoint": self.endpoint,
            "attempted_episode_count": len(self._attempted_episodes),
            "attempted_request_count": len(self._attempted_requests),
            "planner_mode": self.mode.value,
            "internvla_model_loaded": False if self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL else None,
            "internvla_fallback_allowed": self.mode is not LaneBPlannerMode.DIRECT_HIGH_LEVEL,
            "raw_text_exposed": False,
            "motion_authority": "bounded_prevalidated_frontier_only",
            "terminal_stop_authority": "none",
            "updated_wall_time_s": time.time(),
            **dict(extra),
        }
        temporary = self.health_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.health_path)

    @staticmethod
    def _planner_episode(runtime_episode_id: str) -> str:
        if not runtime_episode_id.startswith("b::"):
            raise ValueError("runtime episode is not Lane B")
        value = runtime_episode_id[3:]
        if not value or "::" in value:
            raise ValueError("runtime episode identity is invalid")
        return value

    def _decode_snapshot(
        self,
        value: Mapping[str, Any],
        *,
        command: NavigationCommand,
        request_sim_ns: int,
    ) -> Any:
        if value.get("schema_version") != 1:
            raise ValueError("advisor snapshot schema is invalid")
        planner_episode = self._planner_episode(command.episode_id)
        expected_snapshot_id = (
            f"b::{planner_episode}::{int(command.reset_generation)}::"
            f"{int(command.sequence_id)}"
        )
        if (
            value.get("runtime_episode_id") != command.episode_id
            or int(value.get("reset_generation", -1)) != int(command.reset_generation)
            or int(value.get("sequence_id", -1)) != int(command.sequence_id)
            or value.get("snapshot_id") != expected_snapshot_id
            or value.get("same_render_tick") is not True
            or value.get("config_sha256") != self.config_sha256
        ):
            raise ValueError("advisor snapshot identity or contract mismatch")
        sim_stamp_ns = value.get("sim_stamp_ns")
        if (
            isinstance(sim_stamp_ns, bool)
            or not isinstance(sim_stamp_ns, int)
            or sim_stamp_ns <= 0
            or sim_stamp_ns != int(request_sim_ns)
        ):
            raise ValueError("advisor snapshot simulation stamp is invalid")
        rows = value.get("images")
        if not isinstance(rows, list) or len(rows) != len(VIEW_ORDER):
            raise ValueError("advisor snapshot must contain four images")
        frames = []
        total_bytes = 0
        for expected_view, row in zip(VIEW_ORDER, rows):
            if not isinstance(row, Mapping) or row.get("view_id") != expected_view:
                raise ValueError("advisor snapshot image order mismatch")
            encoded = str(row.get("jpeg_base64") or "")
            try:
                jpeg = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError("advisor snapshot JPEG encoding is invalid") from exc
            total_bytes += len(jpeg)
            if not jpeg or len(jpeg) > MAX_IMAGE_BYTES:
                raise ValueError("advisor snapshot JPEG exceeds per-image bound")
            if hashlib.sha256(jpeg).hexdigest() != row.get("jpeg_sha256"):
                raise ValueError("advisor snapshot JPEG digest mismatch")
            if row.get("extrinsic_sha256") != self.extrinsics[expected_view]:
                raise ValueError("advisor snapshot extrinsic digest mismatch")
            if row.get("source_frame_id") == "":
                raise ValueError("advisor snapshot source frame is missing")
            frames.append(
                TimedRevCImage(
                    image=OrderedImage(
                        view_id=expected_view,
                        pose=self.camera_poses[expected_view],
                        jpeg=jpeg,
                        width=int(row.get("width", 0)),
                        height=int(row.get("height", 0)),
                    ),
                    sim_stamp_s=sim_stamp_ns / 1_000_000_000.0,
                    extrinsic_sha256=self.extrinsics[expected_view],
                    source_frame_id=str(row.get("source_frame_id")),
                )
            )
        if total_bytes > MAX_TOTAL_IMAGE_BYTES:
            raise ValueError("advisor snapshot JPEG aggregate exceeds bound")
        return validate_rev_c_snapshot(
            episode_id=planner_episode,
            snapshot_id=expected_snapshot_id,
            snapshot_sim_stamp_s=sim_stamp_ns / 1_000_000_000.0,
            frames=frames,
            config_sha256=self.config_sha256,
            expected_config_sha256=self.config_sha256,
            expected_extrinsic_sha256=self.extrinsics,
            max_frame_age_s=0.5,
            max_inter_camera_skew_s=0.2,
        )

    def _prepare(
        self, command: NavigationCommand, snapshot_id: str
    ) -> Any:
        if not self.prepare_client.wait_for_service(timeout_sec=2.0):
            raise TimeoutError("Step3 prepare service unavailable")
        request = PrepareFrontiers.Request()
        request.protocol_version = 1
        request.episode_id = command.episode_id
        request.reset_generation = int(command.reset_generation)
        request.sequence_id = int(command.sequence_id)
        request.request_id = command.request_id
        request.snapshot_id = snapshot_id
        request.observation_digest = command.observation_digest
        request.fallback_command = command
        return _future_result(
            self.prepare_client.call_async(request), 10.0, "Step3 frontier prepare"
        )

    def _commit(
        self,
        command: NavigationCommand,
        *,
        snapshot_id: str,
        token: str,
        frontier_id: int | None,
        path_sha256: str,
    ) -> Any:
        if not self.commit_client.wait_for_service(timeout_sec=2.0):
            raise TimeoutError("Step3 commit service unavailable")
        request = CommitFrontier.Request()
        request.protocol_version = 1
        request.episode_id = command.episode_id
        request.reset_generation = int(command.reset_generation)
        request.sequence_id = int(command.sequence_id)
        request.request_id = command.request_id
        request.snapshot_id = snapshot_id
        request.token = token
        request.select_frontier = frontier_id is not None
        request.frontier_id = int(frontier_id or 0)
        request.expected_path_sha256 = path_sha256 if frontier_id is not None else ""
        return _future_result(
            self.commit_client.call_async(request), 10.0, "Step3 frontier commit"
        )

    def _commit_resolution(self, command: NavigationCommand, commit: Any) -> Any:
        response = ResolveCommand.Response()
        response.status_code = int(commit.status_code)
        response.status_message = str(commit.status_message)
        response.episode_id = command.episode_id
        response.reset_generation = int(command.reset_generation)
        response.sequence_id = int(command.sequence_id)
        response.request_id = command.request_id
        response.discrete_action = int(commit.discrete_action)
        response.stop = False
        response.nav2_goal_sent = bool(commit.nav2_goal_sent)
        response.nav2_plan_valid = bool(commit.nav2_plan_valid)
        response.resolution_latency_sec = float(commit.resolution_latency_sec)
        return response

    def resolve(
        self,
        command: NavigationCommand,
        *,
        advisor_snapshot: Mapping[str, Any] | None,
        instruction: str,
        agent_pose: tuple[float, ...],
        request_sim_ns: int,
    ) -> Any:
        episode_key = (str(command.episode_id), int(command.reset_generation))
        request_key = (
            str(command.episode_id),
            int(command.reset_generation),
            int(command.sequence_id),
            str(command.request_id),
        )
        if advisor_snapshot is None:
            if self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL:
                return self.node._direct_safe_stop(
                    command, "current_rev_c_snapshot_unavailable"
                )
            return self.node._resolve_frozen_nav2(command)
        if self.mode is LaneBPlannerMode.BOUNDED_ADVISOR and episode_key in self._attempted_episodes:
            return self.node._resolve_frozen_nav2(command)
        if request_key in self._attempted_requests:
            if self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL:
                return self.node._direct_safe_stop(command, "duplicate_direct_request")
            return self.node._resolve_frozen_nav2(command)
        self._attempted_episodes.add(episode_key)
        self._attempted_requests.add(request_key)
        token = ""
        request = None
        decision_recorded = False
        started = time.perf_counter()
        try:
            snapshot = self._decode_snapshot(
                advisor_snapshot,
                command=command,
                request_sim_ns=request_sim_ns,
            )
            self.snapshot_writer.append(snapshot)
            prepare = self._prepare(command, snapshot.identity.snapshot_id)
            if int(prepare.status_code) != STATUS_OK or not prepare.token or not prepare.frontiers:
                if self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL:
                    raise DirectStep3Failure("no_current_legal_frontiers")
                self._write_health(
                    "FALLBACK",
                    {"fallback_reason": "no_current_legal_frontiers"},
                )
                return self.node._resolve_frozen_nav2(command)
            token = str(prepare.token)
            frontiers = tuple(
                CandidateFrontier(
                    frontier_id=int(value.frontier_id),
                    relative_xz=(float(value.relative_x), float(value.relative_z)),
                    distance_m=float(value.distance_m),
                    bearing_deg=float(value.bearing_deg),
                )
                for value in prepare.frontiers
            )
            request = build_lane_b_request(
                snapshot,
                instruction=instruction,
                candidate_frontiers=frontiers,
                agent_pose=agent_pose,
            )
            sim_before_ns = int(self.node.get_clock().now().nanoseconds)
            if sim_before_ns != int(request_sim_ns):
                raise RuntimeError("simulation time changed before Step3 inference")
            with SlowPlannerClient(
                self.endpoint, timeout_ms=int(self.timeout_sec * 1000)
            ) as client:
                decision, metrics = client.decide(request)
            sim_after_ns = int(self.node.get_clock().now().nanoseconds)
            if sim_before_ns <= 0 or sim_after_ns != sim_before_ns:
                raise RuntimeError("simulation time changed during Step3 inference")
            outcome = self.adapter.resolve(request, decision)
            selected_id = (
                int(outcome.frontier_id)
                if outcome.intent
                in {
                    LaneBIntent.FRONTIER_ADVICE,
                    LaneBIntent.FRONTIER_GOAL_CANDIDATE,
                }
                and outcome.frontier_id is not None
                else None
            )
            digest_by_id = {
                int(value.frontier_id): str(value.path_sha256)
                for value in prepare.frontiers
            }
            commit = self._commit(
                command,
                snapshot_id=snapshot.identity.snapshot_id,
                token=token,
                frontier_id=selected_id,
                path_sha256=digest_by_id.get(selected_id, ""),
            )
            token = ""
            if selected_id is not None and (
                int(commit.status_code) != STATUS_OK
                or not commit.accepted
                or not commit.executed
                or commit.fallback_required
                or str(commit.path_sha256) != digest_by_id[selected_id]
            ):
                if self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL:
                    outcome = self.adapter.resolve_failure(request, "commit_rejected")
                else:
                    outcome = replace(
                        outcome,
                        intent=LaneBIntent.INTERNVLA_FALLBACK_REQUIRED,
                        frontier_id=None,
                        recommended_frontier=None,
                        abstain=True,
                        fallback_used=True,
                        fallback_reason="commit_rejected",
                        requires_internvla_fallback=True,
                    )
            if (
                selected_id is not None
                and not outcome.requires_internvla_fallback
                and not outcome.requires_safe_stop
            ):
                # The adapter has already accepted the one-shot path.  Logging
                # must never turn that success into a second fallback command.
                try:
                    self.decision_writer.append(outcome, metrics)
                    decision_recorded = True
                    self._write_health(
                        "COMMITTED",
                        {
                            "snapshot_id": snapshot.identity.snapshot_id,
                            "recommended_frontier": selected_id,
                            "step3_latency_ms": metrics.end_to_end_ms,
                            "total_advisor_latency_ms": (time.perf_counter() - started) * 1000.0,
                        },
                    )
                except BaseException:
                    pass
                return self._commit_resolution(command, commit)
            self.decision_writer.append(outcome, metrics)
            decision_recorded = True
            self._write_health(
                "SAFE_STOP" if self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL else "FALLBACK",
                {
                    "snapshot_id": snapshot.identity.snapshot_id,
                    "fallback_reason": outcome.fallback_reason,
                    "safe_stop_reason": outcome.safe_stop_reason,
                    "semantic_abstain": outcome.source_decision == "abstain",
                    "step3_latency_ms": metrics.end_to_end_ms,
                },
            )
            if self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL:
                raise DirectStep3Failure(
                    outcome.safe_stop_reason or "direct_decision_rejected"
                )
            return self.node._resolve_frozen_nav2(command)
        except BaseException as exc:
            if token and request is not None:
                try:
                    self._commit(
                        command,
                        snapshot_id=request.snapshot_id,
                        token=token,
                        frontier_id=None,
                        path_sha256="",
                    )
                except BaseException:
                    pass
            if (
                self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL
                and request is not None
                and not decision_recorded
            ):
                try:
                    failure = self.adapter.resolve_failure(
                        request,
                        str(exc)
                        if isinstance(exc, DirectStep3Failure)
                        else "direct_runtime_failure",
                    )
                    self.decision_writer.append(
                        failure,
                        {"end_to_end_ms": (time.perf_counter() - started) * 1000.0},
                    )
                except BaseException:
                    pass
            self._write_health(
                "SAFE_STOP" if self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL else "FALLBACK",
                {
                    "fallback_reason": (
                        "" if self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL
                        else "advisor_runtime_failure"
                    ),
                    "safe_stop_reason": (
                        str(exc)
                        if isinstance(exc, DirectStep3Failure)
                        else "direct_runtime_failure"
                        if self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL
                        else ""
                    ),
                    "error_class": type(exc).__name__,
                },
            )
            if self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL:
                reason = (
                    str(exc)
                    if isinstance(exc, DirectStep3Failure)
                    else "direct_runtime_failure"
                )
                return self.node._direct_safe_stop(command, reason)
            return self.node._resolve_frozen_nav2(command)
