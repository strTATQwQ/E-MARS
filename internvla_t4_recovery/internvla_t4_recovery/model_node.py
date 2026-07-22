"""T4 model wrapper with in-generation history invalidation."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from internvla_ros2.fault_injection import FaultControlReader, fault_profile_enabled
from internvla_ros2.model_node import InternVLAModelNode
from internvla_ros2.protocol import RequestIdentity
from internvla_ros2.recovery_contract import (
    OP_CLEAR_MODEL_CACHE,
    STATUS_CONFLICT,
    STATUS_INTERNAL,
    STATUS_TIMEOUT,
    RecoveryContractError,
    goal_identity,
    initialize_response,
    reject_response,
    response_snapshot,
    restore_response,
    t5_completion_sim_enabled,
    validate_request,
)
from internvla_ros2_msgs.srv import RecoveryControl
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger


class T4RecoveryModelNode(InternVLAModelNode):
    def __init__(self) -> None:
        super().__init__()
        self._recovery_uses_sim_time = t5_completion_sim_enabled()
        if self._recovery_uses_sim_time and not bool(
            self.get_parameter("use_sim_time").value
        ):
            raise RuntimeError("T5 completion_sim recovery requires use_sim_time=true")
        self._last_recovery_contract_sim_ns = 0
        self.declare_parameter(
            "ablation_variant_id",
            os.environ.get("INTERNVLA_T4_VARIANT_ID", "none"),
        )
        self.declare_parameter(
            "ablation_config_sha256",
            os.environ.get("INTERNVLA_T4_VARIANT_CONFIG_SHA256", "none"),
        )
        self.declare_parameter(
            "history_mode", os.environ.get("INTERNVLA_T4_HISTORY_MODE", "on")
        )
        self.ablation_variant_id = str(
            self.get_parameter("ablation_variant_id").value
        )
        self.ablation_config_sha256 = str(
            self.get_parameter("ablation_config_sha256").value
        )
        self.history_mode = str(self.get_parameter("history_mode").value)
        if self.history_mode not in {"on", "off"}:
            raise RuntimeError("history_mode must be on or off")
        if self.ablation_variant_id != "none" and (
            len(self.ablation_config_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.ablation_config_sha256
            )
        ):
            raise RuntimeError(
                "ablation_config_sha256 must identify the generated config"
            )
        self.declare_parameter(
            "recovery_audit_path",
            os.environ.get("INTERNVLA_T4_RECOVERY_MODEL_AUDIT", ""),
        )
        value = str(self.get_parameter("recovery_audit_path").value)
        self.recovery_audit_path = Path(value).resolve() if value else None
        if self.ablation_variant_id != "none" and self.recovery_audit_path is None:
            raise RuntimeError("an ablation model requires recovery_audit_path")
        if self.recovery_audit_path is not None:
            self.recovery_audit_path.parent.mkdir(parents=True, exist_ok=True)
            if self.recovery_audit_path.exists():
                raise FileExistsError(self.recovery_audit_path)
        self.recovery_clear_count = 0
        self.cache_epoch = 0
        self._typed_recovery_results: dict[
            str, tuple[tuple[object, ...], dict[str, object]]
        ] = {}
        self.history_off_reset_count = 0
        self.history_frame_counts: dict[tuple[str, int], int] = {}
        self.history_reset_counts: dict[str, int] = {}
        self._audit_lock = threading.Lock()
        self._fault_control = (
            FaultControlReader("dgx_model") if fault_profile_enabled() else None
        )
        self._write_lifecycle_manifest()
        self.create_service(
            RecoveryControl,
            "/internvla/t4_recovery_model",
            self._on_typed_clear_history,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger,
            "/internvla/t4_clear_history",
            self._on_clear_history,
            callback_group=self._callback_group,
        )

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

    def _write_lifecycle_manifest(self) -> None:
        if self.ablation_variant_id == "none":
            return
        if self.recovery_audit_path is None:
            raise RuntimeError("an ablation model requires recovery_audit_path")
        path = self.recovery_audit_path.parent / "model_lifecycle_manifest.json"
        payload = {
            "schema_version": 1,
            "status": "STARTED",
            "model_host": "dgx_spark",
            "variant_id": self.ablation_variant_id,
            "config_sha256": self.ablation_config_sha256,
            "history_mode": self.history_mode,
            "pid": os.getpid(),
            "wall_time_unix": time.time(),
        }
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        self._write_audit(
            {
                "schema_version": 1,
                "event": "ablation_model_lifecycle_started",
                "variant_id": self.ablation_variant_id,
                "config_sha256": self.ablation_config_sha256,
                "history_mode": self.history_mode,
                "pid": os.getpid(),
                "wall_time_unix": time.time(),
            }
        )

    def _on_initialize(self, request: Any, response: Any) -> Any:
        before = (
            self._barrier.initialized,
            self._barrier.episode_id,
            self._barrier.reset_generation,
        )
        result = super()._on_initialize(request, response)
        after = (
            self._barrier.initialized,
            self._barrier.episode_id,
            self._barrier.reset_generation,
        )
        if after != before and after[0]:
            key = (str(after[1]), int(after[2]))
            self.history_frame_counts.setdefault(key, 0)
            self.history_reset_counts.setdefault(str(after[1]), 0)
            self._write_audit(
                {
                    "schema_version": 1,
                    "event": "ablation_history_episode_initialized",
                    "variant_id": self.ablation_variant_id,
                    "episode_id": key[0],
                    "reset_generation": key[1],
                    "history_mode": self.history_mode,
                    "history_frame_count": 0,
                    "history_reset_count": self.history_reset_counts[key[0]],
                    "wall_time_unix": time.time(),
                }
            )
        return result

    def _on_reset(self, request: Any, response: Any) -> Any:
        before = (self._barrier.episode_id, self._barrier.reset_generation)
        result = super()._on_reset(request, response)
        after = (self._barrier.episode_id, self._barrier.reset_generation)
        if after != before and self._barrier.initialized:
            episode_id = str(after[0])
            generation = int(after[1])
            self.history_reset_counts[episode_id] = (
                self.history_reset_counts.get(episode_id, 0) + 1
            )
            self.history_frame_counts.setdefault((episode_id, generation), 0)
            self._write_audit(
                {
                    "schema_version": 1,
                    "event": "ablation_history_generation_reset",
                    "variant_id": self.ablation_variant_id,
                    "episode_id": episode_id,
                    "reset_generation": generation,
                    "history_mode": self.history_mode,
                    "history_frame_count": 0,
                    "history_reset_count": self.history_reset_counts[episode_id],
                    "wall_time_unix": time.time(),
                }
            )
        return result

    def _execute_step(self, goal_handle: Any) -> Any:
        request = goal_handle.request
        fault_control = getattr(self, "_fault_control", None)
        if fault_control is not None:
            event_id = fault_control.consume_once("model_request_timeout")
            if event_id is not None:
                identity = RequestIdentity(
                    str(request.episode_id),
                    int(request.reset_generation),
                    int(request.sequence_id),
                    str(request.request_id),
                )
                message = "injected T5 completion_sim model request timeout"
                self._set_state(self._lifecycle, STATUS_TIMEOUT, message, True)
                goal_handle.abort()
                fault_control.record(
                    event_id,
                    "model_request_timeout",
                    "injected_timeout",
                    max(0, int(self.get_clock().now().nanoseconds)),
                )
                return self._error_result(identity, STATUS_TIMEOUT, message)
        with self._state_lock:
            fresh = (
                self._barrier.initialized
                and str(request.episode_id) == self._barrier.episode_id
                and int(request.reset_generation) == self._barrier.reset_generation
                and int(request.sequence_id) == self._barrier.last_sequence_id + 1
            )
        if self.history_mode == "off" and fresh:
            with self._inference_lock:
                if self._agent is not None:
                    self._agent.reset([0])
                    self._system1_queue_remaining = 0
                    self.history_off_reset_count += 1
                    self._write_audit(
                        {
                            "schema_version": 1,
                            "event": "history_disabled_step_reset",
                            "episode_id": self._barrier.episode_id,
                            "reset_generation": self._barrier.reset_generation,
                            "sequence_id": int(request.sequence_id),
                            "history_off_reset_count": self.history_off_reset_count,
                            "wall_time_unix": time.time(),
                        }
                    )
        result = super()._execute_step(goal_handle)
        if fresh:
            key = (str(request.episode_id), int(request.reset_generation))
            self.history_frame_counts[key] = self.history_frame_counts.get(key, 0) + 1
            self._write_audit(
                {
                    "schema_version": 1,
                    "event": "ablation_history_frame_consumed",
                    "variant_id": self.ablation_variant_id,
                    "episode_id": key[0],
                    "reset_generation": key[1],
                    "sequence_id": int(request.sequence_id),
                    "history_mode": self.history_mode,
                    "history_frame_count": self.history_frame_counts[key],
                    "history_reset_count": self.history_reset_counts.get(key[0], 0),
                    "wall_time_unix": time.time(),
                }
            )
        return result

    def _write_audit(self, record: dict[str, Any]) -> None:
        if self.recovery_audit_path is None:
            return
        with self._audit_lock, self.recovery_audit_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _on_clear_history(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        acquired = self._inference_lock.acquire(timeout=2.0)
        if not acquired:
            response.success = False
            response.message = "inference lock timeout; history unchanged"
            return response
        try:
            if self._agent is None or not self._barrier.initialized:
                response.success = False
                response.message = "model is not initialized"
                return response
            self._clear_history_locked("legacy_trigger", "legacy")
            response.success = True
            response.message = "history cleared; next call requires fresh System 1"
            return response
        finally:
            self._inference_lock.release()

    def _clear_history_locked(
        self,
        recovery_id: str,
        operation_id: str,
        *,
        preserve_observation_tuple: bool = False,
    ) -> int:
        self._agent.reset([0])
        self._system1_queue_remaining = 0
        self._results.clear()
        if not preserve_observation_tuple:
            self._clear_observations()
        self.recovery_clear_count += 1
        self.cache_epoch += 1
        self._write_audit(
            {
                "schema_version": 1,
                "event": "history_cleared",
                "episode_id": self._barrier.episode_id,
                "reset_generation": self._barrier.reset_generation,
                "last_sequence_id": self._barrier.last_sequence_id,
                "recovery_id": recovery_id,
                "operation_id": operation_id,
                "clear_count": self.recovery_clear_count,
                "cache_epoch": self.cache_epoch,
                "observation_tuple_preserved": preserve_observation_tuple,
                "episode_identity_changed": False,
                "wall_time_unix": time.time(),
            }
        )
        return self.cache_epoch

    def _on_typed_clear_history(
        self,
        request: RecoveryControl.Request,
        response: RecoveryControl.Response,
    ) -> RecoveryControl.Response:
        initialize_response(response, request)
        acquired = self._inference_lock.acquire(timeout=2.0)
        if not acquired:
            response.status_code = STATUS_INTERNAL
            response.status_message = "inference lock timeout; cache unchanged"
            return response
        try:
            if self._agent is None or not self._barrier.initialized:
                raise RecoveryContractError(STATUS_INTERNAL, "model is not initialized")
            identity = validate_request(
                request,
                expected_operation=OP_CLEAR_MODEL_CACHE,
                active_episode_id=self._barrier.episode_id,
                active_reset_generation=self._barrier.reset_generation,
                active_goal_id=goal_identity(
                    self._barrier.episode_id,
                    self._barrier.reset_generation,
                    self._barrier.last_sequence_id,
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
            response.cache_epoch = self._clear_history_locked(
                identity.recovery_id,
                identity.operation_id,
                preserve_observation_tuple=self._recovery_uses_sim_time,
            )
            response.success = True
            response.status_code = 0
            response.status_message = "model cache cleared for active identity"
            response.motion_disabled = False
            response.safety_fresh = False
            if len(self._typed_recovery_results) >= 256:
                self._typed_recovery_results.pop(next(iter(self._typed_recovery_results)))
            self._typed_recovery_results[identity.operation_id] = (
                identity_key,
                response_snapshot(response),
            )
        except RecoveryContractError as exc:
            reject_response(response, exc)
        except BaseException as exc:
            response.success = False
            response.status_code = STATUS_INTERNAL
            response.status_message = f"cache clear failed: {exc!r}"[:512]
        finally:
            self._inference_lock.release()
        return response


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = T4RecoveryModelNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
