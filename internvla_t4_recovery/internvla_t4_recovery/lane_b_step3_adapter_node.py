"""Lane-B-only Step3 frontier transaction overlay for the frozen T4 adapter."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import rclpy
from internnav_t5_lane_b_msgs.srv import CommitFrontier, PrepareFrontiers
from internvla_nav2_adapter.active_node import (
    ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, STATUS_INTERNAL_ERROR,
    STATUS_INVALID_REQUEST, STATUS_OK, STATUS_STALE, STATUS_TIMEOUT,
    _future_result, _time_ns,
)
from internvla_nav2_adapter.lane_b_frontier_transaction import (
    FrontierTransactionError, FrontierTransactionStore, PreparedFrontier,
)
from internvla_ros2_msgs.srv import ResolveCommand
from internvla_t4_recovery.adapter_node import T4RecoveryAdapter
from nav2_msgs.srv import IsPathValid
from nav_msgs.msg import Path as NavPath
from rclpy.executors import MultiThreadedExecutor


class LaneBStep3RecoveryAdapter(T4RecoveryAdapter):
    """Add one-shot bounded frontiers without changing the frozen Lane-A class."""

    def __init__(self) -> None:
        advisor = os.environ.get("INTERNNAV_T5_STEP3_LIVE_ADVISOR", "0") == "1"
        direct = os.environ.get("INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL", "0") == "1"
        if not (
            advisor != direct
            and os.environ.get("INTERNNAV_T5_LANE", "") == "b"
            and os.environ.get("INTERNNAV_T5_LANE_NAMESPACE", "") == "/t5/lane_b"
        ):
            raise RuntimeError("exactly one Lane-B Step3 adapter mode is required")
        self.step3_direct_high_level = direct
        super().__init__()
        if not self._t5_sim_time_semantics:
            raise RuntimeError("Step3 live adapter requires frozen T5 sim-time semantics")
        self.step3_live_advisor_enabled = True
        self.step3_frontier_token_ttl = 15.0
        self.step3_advisor_records_path = self.result_dir / "step3_advisor_records.jsonl"
        if self.step3_advisor_records_path.exists():
            raise RuntimeError("refusing to append existing Step3 adapter record")
        self.step3_frontier_store = FrontierTransactionStore(ttl_sec=15.0)
        self.step3_commit_cleanup_blocked = False
        self.step3_path_valid_client = self.create_client(
            IsPathValid, "is_path_valid", callback_group=self.callback_group
        )
        self.create_service(PrepareFrontiers, "step3/prepare_frontiers",
            self._prepare_step3_frontiers, callback_group=self.callback_group)
        self.create_service(CommitFrontier, "step3/commit_frontier",
            self._commit_step3_frontier, callback_group=self.callback_group)
        self._write_summary("READY")

    def _resolve(self, request: ResolveCommand.Request,
                 response: ResolveCommand.Response) -> ResolveCommand.Response:
        with self.operation_lock:
            store = getattr(self, "step3_frontier_store", None)
            if store is not None:
                store.invalidate()
            if getattr(self, "step3_commit_cleanup_blocked", False):
                self._publish_motion(False)
                self._cancel_active(wait=True)
                self.step3_commit_cleanup_blocked = False
            return super()._resolve(request, response)

    @staticmethod
    def _step3_path_sha256(path: NavPath) -> str:
            payload = {
                "frame_id": str(path.header.frame_id),
                "poses": [
                    [
                        float(pose.pose.position.x),
                        float(pose.pose.position.y),
                        float(pose.pose.position.z),
                        float(pose.pose.orientation.x),
                        float(pose.pose.orientation.y),
                        float(pose.pose.orientation.z),
                        float(pose.pose.orientation.w),
                    ]
                    for pose in path.poses
                ],
            }
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _step3_identity(
            episode_id: str,
            reset_generation: int,
            sequence_id: int,
            request_id: str,
            snapshot_id: str,
        ) -> tuple[str, int, int, str, str]:
            return (
                str(episode_id),
                int(reset_generation),
                int(sequence_id),
                str(request_id),
                str(snapshot_id),
            )

    @staticmethod
    def _expected_step3_snapshot_id(
            episode_id: str, reset_generation: int, sequence_id: int
        ) -> str:
            prefix = "b::"
            if not str(episode_id).startswith(prefix):
                raise ValueError("Lane-B episode identity is missing")
            raw_episode = str(episode_id)[len(prefix) :]
            if not raw_episode or "::" in raw_episode:
                raise ValueError("Lane-B episode identity is invalid")
            return f"b::{raw_episode}::{int(reset_generation)}::{int(sequence_id)}"

    def _validate_step3_next_identity(
            self, episode_id: str, reset_generation: int, sequence_id: int
        ) -> None:
            generation = int(reset_generation)
            sequence = int(sequence_id)
            if self.active_generation < 0:
                if sequence != 0:
                    raise ValueError("first sequence is not zero")
            elif generation == self.active_generation:
                if episode_id != self.active_episode or sequence != self.last_sequence + 1:
                    raise ValueError("non-contiguous command identity")
            elif generation > self.active_generation:
                if sequence != 0:
                    raise ValueError("invalid reset barrier")
            else:
                raise ValueError("invalid reset generation")

    def _commit_step3_identity(
            self, episode_id: str, reset_generation: int, sequence_id: int
        ) -> None:
            self._validate_step3_next_identity(
                episode_id, reset_generation, sequence_id
            )
            if int(reset_generation) > self.active_generation:
                self._clear_frozen_system1_target()
                with self.plan_condition:
                    self.latest_plan = []
                    self.latest_plan_monotonic = 0.0
                self.reset_settle_until = 0.0
            self.active_generation = int(reset_generation)
            self.active_episode = str(episode_id)
            self.last_sequence = int(sequence_id)

    def _step3_path_is_valid(self, path: NavPath) -> bool:
            client = self.step3_path_valid_client
            if client is None or not client.wait_for_service(timeout_sec=self.server_timeout):
                raise TimeoutError
            request = IsPathValid.Request()
            request.path = path
            response = _future_result(
                client.call_async(request), self.server_timeout
            )
            return bool(response.is_valid) and not list(response.invalid_pose_indices)

    def _append_step3_advisor(self, record: dict[str, Any]) -> None:
            value = dict(record)
            value.setdefault("schema_version", 1)
            value.setdefault("recorded_wall_time", time.time())
            line = json.dumps(
                value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ) + "\n"
            with self.step3_advisor_records_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def _prepare_step3_frontiers(self, request: Any, response: Any) -> Any:
            with self.operation_lock:
                return self._prepare_step3_frontiers_locked(request, response)

    def _prepare_step3_frontiers_locked(self, request: Any, response: Any) -> Any:
            started = time.monotonic()
            command = request.fallback_command
            response.episode_id = request.episode_id
            response.reset_generation = request.reset_generation
            response.sequence_id = request.sequence_id
            response.request_id = request.request_id
            response.snapshot_id = request.snapshot_id
            record: dict[str, Any] = {
                "phase": "prepare",
                "episode_id": str(request.episode_id),
                "reset_generation": int(request.reset_generation),
                "sequence_id": int(request.sequence_id),
                "request_id": str(request.request_id),
                "snapshot_id": str(request.snapshot_id),
                "frontiers": [],
            }
            try:
                if int(request.protocol_version) != 1:
                    raise ValueError("unsupported Step3 frontier protocol")
                if (
                    command.episode_id != request.episode_id
                    or int(command.reset_generation) != int(request.reset_generation)
                    or int(command.sequence_id) != int(request.sequence_id)
                    or command.request_id != request.request_id
                    or command.observation_digest != request.observation_digest
                ):
                    raise ValueError("fallback command identity mismatch")
                expected_snapshot = self._expected_step3_snapshot_id(
                    request.episode_id,
                    int(request.reset_generation),
                    int(request.sequence_id),
                )
                if request.snapshot_id != expected_snapshot:
                    raise ValueError("snapshot identity mismatch")
                self._validate_step3_next_identity(
                    request.episode_id,
                    int(request.reset_generation),
                    int(request.sequence_id),
                )
                sim_now_ns = int(self.get_clock().now().nanoseconds)
                if sim_now_ns <= 0 or sim_now_ns > _time_ns(command.valid_until):
                    raise ValueError("stale command")
                self._cancel_active(wait=True)
                self.step3_commit_cleanup_blocked = False
                self._publish_motion(False)
                geometries = {
                    ACTION_LEFT: (0, 0.0, 0.0, 0.0, -15.0),
                    ACTION_FORWARD: (1, 0.0, self.flash_forward_step, self.flash_forward_step, 0.0),
                    ACTION_RIGHT: (2, 0.0, 0.0, 0.0, 15.0),
                }
                prepared: list[PreparedFrontier] = []
                for action in (ACTION_LEFT, ACTION_FORWARD, ACTION_RIGHT):
                    path = self._system2_path(command, action)
                    if not self._step3_path_is_valid(path):
                        continue
                    frontier_id, relative_x, relative_z, distance_m, bearing_deg = geometries[action]
                    prepared.append(
                        PreparedFrontier(
                            frontier_id=frontier_id,
                            discrete_action=action,
                            relative_x=relative_x,
                            relative_z=relative_z,
                            distance_m=distance_m,
                            bearing_deg=bearing_deg,
                            path_sha256=self._step3_path_sha256(path),
                            path=path,
                        )
                    )
                if not prepared:
                    response.status_code = STATUS_OK
                    response.status_message = "no current legal Nav2 frontiers"
                    record["status"] = "NO_LEGAL_FRONTIERS"
                    return response
                assert self.step3_frontier_store is not None
                identity = self._step3_identity(
                    request.episode_id,
                    int(request.reset_generation),
                    int(request.sequence_id),
                    request.request_id,
                    request.snapshot_id,
                )
                transaction = self.step3_frontier_store.prepare(
                    identity, prepared, prepared_sim_ns=sim_now_ns
                )
                from internnav_t5_lane_b_msgs.msg import CandidateFrontier

                response.token = transaction.token
                response.prepared_sim_time.sec = sim_now_ns // 1_000_000_000
                response.prepared_sim_time.nanosec = sim_now_ns % 1_000_000_000
                for value in prepared:
                    message = CandidateFrontier()
                    message.frontier_id = value.frontier_id
                    message.discrete_action = value.discrete_action
                    message.relative_x = value.relative_x
                    message.relative_z = value.relative_z
                    message.distance_m = value.distance_m
                    message.bearing_deg = value.bearing_deg
                    message.path_sha256 = value.path_sha256
                    response.frontiers.append(message)
                    record["frontiers"].append(
                        {
                            "frontier_id": value.frontier_id,
                            "discrete_action": value.discrete_action,
                            "relative_xz": [value.relative_x, value.relative_z],
                            "distance_m": value.distance_m,
                            "bearing_deg": value.bearing_deg,
                            "path_sha256": value.path_sha256,
                        }
                    )
                response.status_code = STATUS_OK
                response.status_message = "prepared"
                record["status"] = "PREPARED"
                record["prepared_sim_ns"] = sim_now_ns
            except ValueError as exc:
                response.status_code = (
                    STATUS_STALE if str(exc) == "stale command" else STATUS_INVALID_REQUEST
                )
                response.status_message = str(exc)
                record["status"] = "REJECTED"
                record["error"] = str(exc)
            except TimeoutError:
                response.status_code = STATUS_TIMEOUT
                response.status_message = "Nav2 path validation timeout"
                record["status"] = "TIMEOUT"
            except BaseException as exc:
                response.status_code = STATUS_INTERNAL_ERROR
                response.status_message = f"Step3 frontier prepare error: {exc!r}"[:512]
                record["status"] = "ERROR"
                record["error"] = response.status_message
            finally:
                record["latency_sec"] = float(time.monotonic() - started)
                self._append_step3_advisor(record)
            return response

    def _commit_step3_frontier(self, request: Any, response: Any) -> Any:
            with self.operation_lock:
                return self._commit_step3_frontier_locked(request, response)

    def _commit_step3_frontier_locked(self, request: Any, response: Any) -> Any:
            started = time.monotonic()
            response.episode_id = request.episode_id
            response.reset_generation = request.reset_generation
            response.sequence_id = request.sequence_id
            response.request_id = request.request_id
            response.snapshot_id = request.snapshot_id
            response.fallback_required = True
            record: dict[str, Any] = {
                "phase": "commit",
                "episode_id": str(request.episode_id),
                "reset_generation": int(request.reset_generation),
                "sequence_id": int(request.sequence_id),
                "request_id": str(request.request_id),
                "snapshot_id": str(request.snapshot_id),
                "select_frontier": bool(request.select_frontier),
                "frontier_id": int(request.frontier_id),
            }
            try:
                if int(request.protocol_version) != 1:
                    raise FrontierTransactionError("unsupported Step3 frontier protocol")
                identity = self._step3_identity(
                    request.episode_id,
                    int(request.reset_generation),
                    int(request.sequence_id),
                    request.request_id,
                    request.snapshot_id,
                )
                assert self.step3_frontier_store is not None
                selected, prepared_sim_ns = self.step3_frontier_store.consume(
                    identity=identity,
                    token=request.token,
                    select_frontier=bool(request.select_frontier),
                    frontier_id=(int(request.frontier_id) if request.select_frontier else None),
                    expected_path_sha256=str(request.expected_path_sha256),
                )
                response.accepted = True
                if selected is None:
                    response.status_code = STATUS_OK
                    if self.step3_direct_high_level:
                        response.status_message = "released for coordinator safe-stop"
                        record["status"] = "RELEASED_FOR_SAFE_STOP"
                    else:
                        response.status_message = "released for frozen InternVLA fallback"
                        record["status"] = "RELEASED_FOR_FALLBACK"
                    return response
                sim_now_ns = int(self.get_clock().now().nanoseconds)
                if sim_now_ns != prepared_sim_ns:
                    raise FrontierTransactionError("simulation time advanced before commit")
                if self._step3_path_sha256(selected.path) != selected.path_sha256:
                    raise FrontierTransactionError("stored frontier path digest changed")
                if not self._step3_path_is_valid(selected.path):
                    raise FrontierTransactionError("frontier path is no longer valid")
                self._validate_step3_next_identity(
                    request.episode_id,
                    int(request.reset_generation),
                    int(request.sequence_id),
                )
                self._send_path(selected.path)
                # The evaluator calls this service synchronously while Isaac's
                # /clock is frozen at the prepared snapshot.  A sim-time Nav2
                # controller cannot publish a post-goal cmd_vel until this call
                # returns and the evaluator advances physics again.  FollowPath
                # acceptance plus IsPathValid is therefore the atomic commit;
                # continuous control remains at the existing safe zero until
                # Nav2 publishes asynchronously after /clock resumes.
                if int(self.get_clock().now().nanoseconds) != prepared_sim_ns:
                    self._publish_motion(False)
                    self._cancel_active(wait=True)
                    raise FrontierTransactionError(
                        "simulation time advanced during frontier commit"
                    )
                action = int(selected.discrete_action)
                self._commit_step3_identity(
                    request.episode_id,
                    int(request.reset_generation),
                    int(request.sequence_id),
                )
                self._publish_motion(True)
                self.active_path_publisher.publish(selected.path)
                self.goal_count += 1
                self.nav2_control_count += 1
                self.quantization_count += 1
                response.status_code = STATUS_OK
                response.status_message = "committed"
                response.executed = True
                response.fallback_required = False
                response.discrete_action = action
                response.nav2_goal_sent = True
                response.nav2_plan_valid = True
                response.path_sha256 = selected.path_sha256
                record.update(
                    {
                        "status": "COMMITTED",
                        "selected_action": action,
                        "path_sha256": selected.path_sha256,
                        "prepared_sim_ns": prepared_sim_ns,
                        "commit_sim_ns": sim_now_ns,
                        "fresh_cmd_required_during_commit": False,
                        "initial_control_before_clock_resume": "safe_zero",
                    }
                )
            except FrontierTransactionError as exc:
                response.status_code = STATUS_STALE
                response.status_message = str(exc)
                record["status"] = "REJECTED"
                record["error"] = str(exc)
            except TimeoutError:
                self._publish_motion(False)
                try:
                    self._cancel_active(wait=True)
                    self.step3_commit_cleanup_blocked = False
                except BaseException:
                    self.step3_commit_cleanup_blocked = True
                response.status_code = STATUS_TIMEOUT
                response.status_message = "Nav2 frontier commit timeout"
                record["status"] = "TIMEOUT"
            except BaseException as exc:
                self._publish_motion(False)
                try:
                    self._cancel_active(wait=True)
                    self.step3_commit_cleanup_blocked = False
                except BaseException:
                    self.step3_commit_cleanup_blocked = True
                response.status_code = STATUS_INTERNAL_ERROR
                response.status_message = f"Step3 frontier commit error: {exc!r}"[:512]
                record["status"] = "ERROR"
                record["error"] = response.status_message
            finally:
                response.resolution_latency_sec = float(time.monotonic() - started)
                record["latency_sec"] = response.resolution_latency_sec
                self._append_step3_advisor(record)
            return response

def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LaneBStep3RecoveryAdapter()
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
