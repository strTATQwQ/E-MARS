"""Strict real-Go2 Step3-first natural-language mission gateway.

This node has no motion, Nav2-goal, or terminal-STOP publisher.  It validates a
high-level operator mission, sends the raw multilingual instruction only to
Step3, and publishes a canonical English mission for the existing InternVLA
client/navigation stack to consume after its own safety gates pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from slow_planner.client import SlowPlannerClient
from slow_planner.mission import (
    MISSION_PROTOCOL_VERSION,
    MISSION_ROUTE,
    CanonicalMission,
    MissionNormalizationRequest,
)


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MissionGatewayError(RuntimeError):
    pass


class MissionNormalizer(Protocol):
    def normalize_instruction(
        self, request: MissionNormalizationRequest
    ) -> tuple[CanonicalMission, Any]: ...


def _identifier(value: Any, name: str) -> str:
    result = str(value or "")
    if not _IDENTIFIER_RE.fullmatch(result):
        raise MissionGatewayError(f"{name} is not a bounded identifier")
    return result


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise MissionGatewayError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise MissionGatewayError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise MissionGatewayError(f"{name} must be a non-negative integer")
    return result


@dataclass(frozen=True)
class MissionEnvelope:
    mission_id: str
    episode_id: str
    reset_generation: int
    sequence_id: int
    identity: str
    raw_instruction: str
    source_instruction_sha256: str
    config_sha256: str
    created_wall_time_s: float
    deadline_wall_monotonic_ns: int
    route: str
    internvla_raw_instruction_allowed: bool
    schema_version: int = MISSION_PROTOCOL_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MissionEnvelope":
        mission_id = _identifier(value.get("mission_id"), "mission_id")
        episode_id = _identifier(value.get("episode_id"), "episode_id")
        reset_generation = _nonnegative(
            value.get("reset_generation"), "reset_generation"
        )
        sequence_id = _nonnegative(value.get("sequence_id"), "sequence_id")
        identity = str(value.get("identity") or "")
        expected_identity = (
            f"real::{mission_id}::{reset_generation}::{sequence_id}"
        )
        if identity != expected_identity:
            raise MissionGatewayError("mission identity is inconsistent")
        instruction = " ".join(str(value.get("raw_instruction") or "").split())
        if not instruction or len(instruction) > 480:
            raise MissionGatewayError("raw instruction is empty or too long")
        digest = str(value.get("source_instruction_sha256") or "")
        if digest != hashlib.sha256(instruction.encode("utf-8")).hexdigest():
            raise MissionGatewayError("raw instruction digest mismatch")
        config_sha256 = str(value.get("config_sha256") or "")
        if not _SHA256_RE.fullmatch(config_sha256):
            raise MissionGatewayError("config_sha256 is invalid")
        route = str(value.get("route") or "")
        if route != MISSION_ROUTE:
            raise MissionGatewayError("mission did not select the Step3-first route")
        if value.get("internvla_raw_instruction_allowed") is not False:
            raise MissionGatewayError("raw InternVLA instruction path must be disabled")
        schema_version = int(value.get("schema_version", -1))
        if schema_version != MISSION_PROTOCOL_VERSION:
            raise MissionGatewayError("mission protocol version mismatch")
        created_wall_time_s = float(value.get("created_wall_time_s") or 0.0)
        deadline_ns = _nonnegative(
            value.get("deadline_wall_monotonic_ns"),
            "deadline_wall_monotonic_ns",
        )
        if created_wall_time_s <= 0 or deadline_ns <= 0:
            raise MissionGatewayError("mission clocks are required")
        return cls(
            mission_id=mission_id,
            episode_id=episode_id,
            reset_generation=reset_generation,
            sequence_id=sequence_id,
            identity=identity,
            raw_instruction=instruction,
            source_instruction_sha256=digest,
            config_sha256=config_sha256,
            created_wall_time_s=created_wall_time_s,
            deadline_wall_monotonic_ns=deadline_ns,
            route=route,
            internvla_raw_instruction_allowed=False,
            schema_version=schema_version,
        )


class MissionGatewayCore:
    def __init__(
        self,
        normalizer: MissionNormalizer,
        *,
        expected_config_sha256: str,
        monotonic_ns=time.monotonic_ns,
    ) -> None:
        if not _SHA256_RE.fullmatch(expected_config_sha256):
            raise ValueError("expected_config_sha256 must be a digest")
        self.normalizer = normalizer
        self.expected_config_sha256 = expected_config_sha256
        self.monotonic_ns = monotonic_ns
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[str, dict[str, Any]]] = {}

    def process(self, value: Mapping[str, Any]) -> dict[str, Any]:
        envelope = MissionEnvelope.from_mapping(value)
        if envelope.config_sha256 != self.expected_config_sha256:
            raise MissionGatewayError("mission config SHA does not match this gateway")
        if self.monotonic_ns() > envelope.deadline_wall_monotonic_ns:
            raise MissionGatewayError("mission expired before Step3 normalization")
        with self._lock:
            cached = self._cache.get(envelope.identity)
            if cached is not None:
                digest, result = cached
                if digest != envelope.source_instruction_sha256:
                    raise MissionGatewayError(
                        "mission identity was reused with different text"
                    )
                return dict(result)
            request = MissionNormalizationRequest(
                mission_id=envelope.mission_id,
                episode_id=envelope.episode_id,
                reset_generation=envelope.reset_generation,
                sequence_id=envelope.sequence_id,
                instruction=envelope.raw_instruction,
                config_sha256=envelope.config_sha256,
                timestamp=time.time(),
            )
            mission, metrics = self.normalizer.normalize_instruction(request)
            if self.monotonic_ns() > envelope.deadline_wall_monotonic_ns:
                raise MissionGatewayError("mission expired while Step3 was responding")
            if mission.abstain:
                raise MissionGatewayError("Step3 abstained; mission remains in safe hold")
            result = {
                "schema_version": MISSION_PROTOCOL_VERSION,
                "status": "CANONICAL_READY",
                "identity": envelope.identity,
                "mission": mission.to_mapping(),
                "step3": {
                    "route": MISSION_ROUTE,
                    "model_variant": getattr(metrics, "model_variant", ""),
                    "end_to_end_ms": float(getattr(metrics, "end_to_end_ms", 0.0)),
                    "raw_text_exposed": False,
                },
                "internvla": {
                    "instruction_source": "canonical_instruction",
                    "raw_instruction_allowed": False,
                    "canonical_instruction": mission.canonical_instruction,
                },
            }
            self._cache[envelope.identity] = (
                envelope.source_instruction_sha256,
                dict(result),
            )
            return result


class CanonicalMissionLatch:
    """Hold only a validated Step3 result for the active real-Go2 episode."""

    def __init__(self, *, expected_config_sha256: str) -> None:
        if not _SHA256_RE.fullmatch(expected_config_sha256):
            raise ValueError("expected_config_sha256 must be a digest")
        self.expected_config_sha256 = expected_config_sha256
        self._lock = threading.RLock()
        self._mission: CanonicalMission | None = None

    def update(self, value: Mapping[str, Any]) -> CanonicalMission:
        if value.get("schema_version") != MISSION_PROTOCOL_VERSION:
            raise MissionGatewayError("canonical mission protocol version mismatch")
        if value.get("status") != "CANONICAL_READY":
            raise MissionGatewayError("canonical mission is not ready")
        internvla = value.get("internvla")
        if not isinstance(internvla, Mapping):
            raise MissionGatewayError("canonical mission InternVLA projection is missing")
        if (
            internvla.get("instruction_source") != "canonical_instruction"
            or internvla.get("raw_instruction_allowed") is not False
        ):
            raise MissionGatewayError("canonical mission permits an unsafe raw path")
        raw_mission = value.get("mission")
        if not isinstance(raw_mission, Mapping):
            raise MissionGatewayError("canonical mission payload is missing")
        mission = CanonicalMission.from_mapping(raw_mission)
        if mission.config_sha256 != self.expected_config_sha256:
            raise MissionGatewayError("canonical mission config SHA mismatch")
        if mission.abstain:
            raise MissionGatewayError("canonical mission abstained")
        if internvla.get("canonical_instruction") != mission.canonical_instruction:
            raise MissionGatewayError("canonical instruction projection mismatch")
        with self._lock:
            current = self._mission
            if current is not None and current.identity == mission.identity:
                if current != mission:
                    raise MissionGatewayError(
                        "canonical mission identity was reused with different content"
                    )
                return current
            self._mission = mission
        return mission

    def instruction_for(self, episode_id: str, reset_generation: int) -> str:
        with self._lock:
            mission = self._mission
        if mission is None:
            raise MissionGatewayError("no Step3 canonical mission is active")
        if (
            mission.episode_id != str(episode_id)
            or mission.reset_generation != int(reset_generation)
        ):
            raise MissionGatewayError("canonical mission episode/reset mismatch")
        return mission.canonical_instruction


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    endpoint = os.environ.get(
        "INTERNVLA_REAL_MISSION_STEP3_ENDPOINT", "tcp://127.0.0.1:8200"
    )
    expected_sha = os.environ.get("INTERNVLA_REAL_MISSION_CONFIG_SHA256", "")
    state_path = Path(
        os.environ.get(
            "INTERNVLA_REAL_MISSION_STATE_PATH",
            "/tmp/internvla_real_go2_mission_state.json",
        )
    )
    input_topic = os.environ.get(
        "INTERNVLA_REAL_MISSION_INPUT_TOPIC", "/user_instruction"
    )
    output_topic = os.environ.get(
        "INTERNVLA_REAL_MISSION_CANONICAL_TOPIC",
        "/internvla/mission/canonical",
    )
    status_topic = os.environ.get(
        "INTERNVLA_REAL_MISSION_STATUS_TOPIC", "/internvla/mission/status"
    )

    client = SlowPlannerClient(endpoint, timeout_ms=12_000)
    core = MissionGatewayCore(client, expected_config_sha256=expected_sha)

    class MissionGatewayNode(Node):
        def __init__(self) -> None:
            super().__init__("strict_real_go2_mission_gateway")
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.canonical = self.create_publisher(String, output_topic, qos)
            self.status = self.create_publisher(String, status_topic, qos)
            self.create_subscription(String, input_topic, self._on_mission, qos)
            initial = {
                "schema_version": 1,
                "status": "SAFE_HOLD_WAITING_FOR_STEP3_FIRST_MISSION",
                "motion_authority": "none",
                "internvla_raw_instruction_allowed": False,
            }
            _atomic_json(state_path, initial)
            self._publish(self.status, initial)

        @staticmethod
        def _publish(publisher: Any, value: Mapping[str, Any]) -> None:
            message = String()
            message.data = json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            publisher.publish(message)

        def _on_mission(self, message: Any) -> None:
            try:
                value = json.loads(message.data)
                if not isinstance(value, Mapping):
                    raise MissionGatewayError("mission message must be an object")
                result = core.process(value)
                _atomic_json(state_path, result)
                self._publish(self.canonical, result)
                self._publish(self.status, result)
            except Exception as exc:
                failure = {
                    "schema_version": 1,
                    "status": "SAFE_HOLD_MISSION_REJECTED",
                    "error": f"{type(exc).__name__}:{exc}"[:512],
                    "motion_authority": "none",
                    "internvla_raw_instruction_allowed": False,
                }
                _atomic_json(state_path, failure)
                self._publish(self.status, failure)

    rclpy.init()
    node = MissionGatewayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        client.close()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
