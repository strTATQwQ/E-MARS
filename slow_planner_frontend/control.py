"""Fail-closed high-level mission staging for strict real-Go2 deployments.

The panel never publishes velocity, terminal STOP, or a Nav2 goal.  It writes
bounded, identity-bound mission envelopes for an integration-owned ROS ingress
process.  Raw operator text is explicitly routed to Step3 normalization before
InternVLA may observe a canonical instruction.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MISSION_PROTOCOL_VERSION = 1
MISSION_ROUTE = "step3_instruction_normalization_v1"
_SAFE_INSTRUCTION = re.compile(r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]{1,480}$")
_MISSION_ID = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$")


class ControlPlaneError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)


def _expand_path(value: Any) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value or "")))
    if not expanded or "$" in expanded or "%" in expanded:
        raise ValueError(f"unresolved control-plane path: {value!r}")
    return Path(expanded).resolve()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class ControlPlaneConfig:
    enabled: bool
    navigation_dispatch_enabled: bool
    human_arm_allowed: bool
    motion_bridge_enabled: bool
    command_dir: Path
    state_path: Path
    navigation_state_path: Path
    mission_gateway_state_path: Path
    audit_path: Path
    instruction_topic: str
    canonical_instruction_topic: str
    cancel_topic: str
    estop_topic: str
    mission_deadline_s: float
    operator_arm_phrase: str
    required_components: tuple[str, ...]
    config_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ControlPlaneConfig":
        frontend = value.get("frontend", value)
        if not isinstance(frontend, Mapping):
            raise ValueError("frontend config must be an object")
        raw = frontend.get("control_plane") or {}
        if not isinstance(raw, Mapping):
            raise ValueError("frontend.control_plane must be an object")
        required_paths = (
            "command_dir",
            "state_path",
            "navigation_state_path",
            "mission_gateway_state_path",
            "audit_path",
        )
        missing = [name for name in required_paths if not str(raw.get(name) or "")]
        if missing:
            raise ValueError(f"control-plane paths missing: {missing}")
        topics = {
            name: str(raw.get(name) or "")
            for name in (
                "instruction_topic",
                "canonical_instruction_topic",
                "cancel_topic",
                "estop_topic",
            )
        }
        if any(not topic.startswith("/") for topic in topics.values()):
            raise ValueError("control-plane topics must be absolute")
        deadline = float(raw.get("mission_deadline_s", 30.0))
        if not 5.0 <= deadline <= 300.0:
            raise ValueError("mission_deadline_s must be in [5,300]")
        arm_phrase = str(raw.get("operator_arm_phrase") or "")
        if len(arm_phrase) < 12 or len(arm_phrase) > 96:
            raise ValueError("operator_arm_phrase must be explicit and bounded")
        required_components = tuple(
            str(item) for item in (raw.get("required_components") or [])
        )
        if bool(raw.get("navigation_dispatch_enabled", False)) and not required_components:
            raise ValueError("dispatch requires an explicit component readiness list")
        if len(set(required_components)) != len(required_components):
            raise ValueError("required_components must not contain duplicates")
        canonical = {
            key: raw.get(key)
            for key in sorted(raw)
            if key not in {"operator_arm_phrase"}
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            navigation_dispatch_enabled=bool(
                raw.get("navigation_dispatch_enabled", False)
            ),
            human_arm_allowed=bool(raw.get("human_arm_allowed", False)),
            motion_bridge_enabled=bool(raw.get("motion_bridge_enabled", False)),
            command_dir=_expand_path(raw["command_dir"]),
            state_path=_expand_path(raw["state_path"]),
            navigation_state_path=_expand_path(raw["navigation_state_path"]),
            mission_gateway_state_path=_expand_path(
                raw["mission_gateway_state_path"]
            ),
            audit_path=_expand_path(raw["audit_path"]),
            instruction_topic=topics["instruction_topic"],
            canonical_instruction_topic=topics["canonical_instruction_topic"],
            cancel_topic=topics["cancel_topic"],
            estop_topic=topics["estop_topic"],
            mission_deadline_s=deadline,
            operator_arm_phrase=arm_phrase,
            required_components=required_components,
            config_sha256=digest,
        )


class MissionControlStore:
    def __init__(self, config: ControlPlaneConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._armed = False
        self._estop_latched = True
        self._generation = 0
        self._missions: dict[str, dict[str, Any]] = {}
        self._publish_state("startup_fail_closed")

    def _audit(self, event: str, **fields: Any) -> None:
        value = {
            "schema_version": 1,
            "event": event,
            "wall_time_s": time.time(),
            "wall_monotonic_ns": time.monotonic_ns(),
            "armed": self._armed,
            "estop_latched": self._estop_latched,
            **fields,
        }
        _append_jsonl(self.config.audit_path, value)

    def _state(self, reason: str) -> dict[str, Any]:
        latest = next(reversed(self._missions.values()), None) if self._missions else None
        return {
            "schema_version": 1,
            "mode": "strict_real_go2",
            "route": MISSION_ROUTE,
            "armed": self._armed,
            "estop_latched": self._estop_latched,
            "generation": self._generation,
            "navigation_dispatch_enabled": self.config.navigation_dispatch_enabled,
            "human_arm_allowed": self.config.human_arm_allowed,
            "motion_bridge_enabled": self.config.motion_bridge_enabled,
            "internvla_raw_instruction_allowed": False,
            "reason": reason,
            "latest_mission": self._public_mission(latest) if latest else None,
            "gateway": self._gateway_state(),
            "updated_wall_time_s": time.time(),
        }

    def _gateway_state(self) -> dict[str, Any]:
        try:
            if self.config.mission_gateway_state_path.stat().st_size > 32_768:
                raise ValueError("gateway state is oversized")
            value = json.loads(
                self.config.mission_gateway_state_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return {"status": "NOT_RUNNING", "canonical_ready": False}
        if not isinstance(value, Mapping) or value.get("schema_version") != 1:
            return {"status": "INVALID_STATE", "canonical_ready": False}
        status = str(value.get("status") or "UNKNOWN")[:96]
        public: dict[str, Any] = {
            "status": status,
            "canonical_ready": status == "CANONICAL_READY",
            "identity": str(value.get("identity") or "")[:256],
        }
        mission = value.get("mission")
        internvla = value.get("internvla")
        if status == "CANONICAL_READY" and isinstance(mission, Mapping):
            if (
                mission.get("config_sha256") != self.config.config_sha256
                or mission.get("internvla_raw_instruction_allowed") is not False
                or not isinstance(internvla, Mapping)
                or internvla.get("raw_instruction_allowed") is not False
            ):
                return {"status": "INVALID_BINDING", "canonical_ready": False}
            public["mission"] = {
                key: mission.get(key)
                for key in (
                    "mission_id",
                    "episode_id",
                    "reset_generation",
                    "sequence_id",
                    "source_language",
                    "canonical_instruction",
                    "target_description",
                    "constraints",
                    "confidence",
                )
            }
            step3 = value.get("step3")
            if isinstance(step3, Mapping):
                public["step3"] = {
                    "route": str(step3.get("route") or "")[:96],
                    "model_variant": str(step3.get("model_variant") or "")[:96],
                    "end_to_end_ms": step3.get("end_to_end_ms"),
                    "raw_text_exposed": False,
                }
        return public

    def _publish_state(self, reason: str) -> dict[str, Any]:
        value = self._state(reason)
        _atomic_json(self.config.state_path, value)
        return value

    @staticmethod
    def _public_mission(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: item
            for key, item in value.items()
            if key not in {"raw_instruction", "operator_arm_phrase"}
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state("query")

    def _navigation_blockers(self) -> list[str]:
        try:
            value = json.loads(
                self.config.navigation_state_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return ["navigation_state_unavailable"]
        if not isinstance(value, Mapping):
            return ["navigation_state_invalid"]
        blockers = []
        for field in (
            "ready",
            "ros_fresh",
            "sensors_fresh",
            "tf_fresh",
            "watchdog_healthy",
            "control_mux_healthy",
        ):
            if value.get(field) is not True:
                blockers.append(f"{field}_false")
        if value.get("estop_latched") is not False:
            blockers.append("navigation_estop_latched")
        components = value.get("components") or {}
        if not isinstance(components, Mapping):
            blockers.append("navigation_components_invalid")
        else:
            blockers.extend(
                f"component_{name}_not_ready"
                for name in self.config.required_components
                if components.get(name) is not True
            )
        return blockers

    def arm(self, phrase: str) -> dict[str, Any]:
        with self._lock:
            if not self.config.enabled or not self.config.human_arm_allowed:
                raise ControlPlaneError(
                    423, "ARM_DISABLED", "strict real-Go2 arm is disabled by configuration"
                )
            if phrase != self.config.operator_arm_phrase:
                self._audit("arm_rejected", reason="phrase_mismatch")
                raise ControlPlaneError(403, "ARM_PHRASE_MISMATCH", "arm phrase rejected")
            if self._estop_latched:
                raise ControlPlaneError(
                    409, "ESTOP_LATCHED", "clear the E-stop latch before arming"
                )
            self._armed = True
            self._generation += 1
            self._audit("armed", generation=self._generation)
            return self._publish_state("operator_armed")

    def estop(self, *, action: str, phrase: str = "") -> dict[str, Any]:
        with self._lock:
            if action == "latch":
                self._estop_latched = True
                self._armed = False
                self._generation += 1
                self._audit("estop_latched", generation=self._generation)
                if self.config.enabled:
                    _atomic_json(
                        self.config.command_dir
                        / f"estop-{time.monotonic_ns()}-latch.json",
                        {
                            "schema_version": 1,
                            "action": "latch",
                            "estop_latched": True,
                            "estop_topic": self.config.estop_topic,
                            "wall_monotonic_ns": time.monotonic_ns(),
                        },
                    )
                return self._publish_state("estop_latched")
            if action != "clear":
                raise ControlPlaneError(422, "INVALID_ESTOP_ACTION", "action must be latch or clear")
            if phrase != self.config.operator_arm_phrase:
                self._audit("estop_clear_rejected", reason="phrase_mismatch")
                raise ControlPlaneError(403, "ARM_PHRASE_MISMATCH", "E-stop clear phrase rejected")
            if not self.config.enabled or not self.config.human_arm_allowed:
                raise ControlPlaneError(
                    423, "ESTOP_CLEAR_DISABLED", "E-stop clear is disabled by configuration"
                )
            self._estop_latched = False
            self._armed = False
            self._generation += 1
            self._audit("estop_cleared_disarmed", generation=self._generation)
            _atomic_json(
                self.config.command_dir
                / f"estop-{time.monotonic_ns()}-clear.json",
                {
                    "schema_version": 1,
                    "action": "clear",
                    "estop_latched": False,
                    "estop_topic": self.config.estop_topic,
                    "wall_monotonic_ns": time.monotonic_ns(),
                },
            )
            return self._publish_state("estop_cleared_disarmed")

    def stage_mission(self, instruction: str, *, dispatch: bool = False) -> dict[str, Any]:
        with self._lock:
            normalized = " ".join(str(instruction).split())
            if not _SAFE_INSTRUCTION.fullmatch(normalized):
                raise ControlPlaneError(
                    422,
                    "INVALID_INSTRUCTION",
                    "instruction is empty, too long, or contains control characters",
                )
            mission_id = f"m-{secrets.token_hex(8)}"
            if not _MISSION_ID.fullmatch(mission_id):
                raise AssertionError("generated mission id is invalid")
            now_wall = time.time()
            now_monotonic_ns = time.monotonic_ns()
            record = {
                "schema_version": MISSION_PROTOCOL_VERSION,
                "route": MISSION_ROUTE,
                "mission_id": mission_id,
                "episode_id": f"real-{mission_id}",
                "reset_generation": self._generation,
                "sequence_id": 0,
                "identity": f"real::{mission_id}::{self._generation}::0",
                "raw_instruction": normalized,
                "source_instruction_sha256": hashlib.sha256(
                    normalized.encode("utf-8")
                ).hexdigest(),
                "config_sha256": self.config.config_sha256,
                "created_wall_time_s": now_wall,
                "created_wall_monotonic_ns": now_monotonic_ns,
                "deadline_wall_time_s": now_wall + self.config.mission_deadline_s,
                "deadline_wall_monotonic_ns": now_monotonic_ns
                + int(self.config.mission_deadline_s * 1e9),
                "instruction_topic": self.config.instruction_topic,
                "canonical_instruction_topic": self.config.canonical_instruction_topic,
                "internvla_raw_instruction_allowed": False,
                "requested_dispatch": bool(dispatch),
                "status": "STAGED",
            }
            self._missions[mission_id] = record
            if dispatch:
                blockers = []
                if not self.config.enabled:
                    blockers.append("control_plane_disabled")
                if not self.config.navigation_dispatch_enabled:
                    blockers.append("navigation_dispatch_disabled")
                if not self._armed:
                    blockers.append("human_arm_inactive")
                if self._estop_latched:
                    blockers.append("estop_latched")
                blockers.extend(self._navigation_blockers())
                if blockers:
                    record["status"] = "STAGED_DISPATCH_BLOCKED"
                    record["dispatch_blockers"] = blockers
                else:
                    record["status"] = "QUEUED_FOR_STEP3"
                    _atomic_json(
                        self.config.command_dir / f"mission-{mission_id}.json",
                        record,
                    )
            self._audit(
                "mission_staged",
                mission_id=mission_id,
                source_instruction_sha256=record["source_instruction_sha256"],
                status=record["status"],
            )
            self._publish_state("mission_staged")
            return self._public_mission(record)

    def cancel(self, mission_id: str) -> dict[str, Any]:
        with self._lock:
            if not _MISSION_ID.fullmatch(str(mission_id)):
                raise ControlPlaneError(404, "MISSION_NOT_FOUND", "mission not found")
            mission = self._missions.get(mission_id)
            if mission is None:
                raise ControlPlaneError(404, "MISSION_NOT_FOUND", "mission not found")
            if mission["status"] == "CANCELED":
                return self._public_mission(mission)
            mission["status"] = "CANCELED"
            cancel = {
                "schema_version": 1,
                "mission_id": mission_id,
                "identity": mission["identity"],
                "cancel_topic": self.config.cancel_topic,
                "wall_time_s": time.time(),
            }
            if self.config.enabled:
                _atomic_json(
                    self.config.command_dir / f"cancel-{mission_id}.json", cancel
                )
            self._audit("mission_canceled", mission_id=mission_id)
            self._publish_state("mission_canceled")
            return self._public_mission(mission)


class MissionCommandOutbox:
    """Publish each validated high-level command at most once across restarts."""

    def __init__(
        self,
        config: ControlPlaneConfig,
        *,
        publish_mission,
        publish_cancel,
        publish_estop,
        monotonic_ns=time.monotonic_ns,
    ) -> None:
        self.config = config
        self.publish_mission = publish_mission
        self.publish_cancel = publish_cancel
        self.publish_estop = publish_estop
        self.monotonic_ns = monotonic_ns
        self.ledger_path = config.command_dir / ".published_commands.jsonl"
        self._published: set[str] = set()
        if self.ledger_path.is_file():
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, Mapping) and str(value.get("filename") or ""):
                    self._published.add(str(value["filename"]))

    def _ack(self, path: Path, kind: str, *, reason: str = "") -> None:
        _append_jsonl(
            self.ledger_path,
            {
                "schema_version": 1,
                "filename": path.name,
                "kind": kind,
                "published_wall_time_s": time.time(),
                "published_wall_monotonic_ns": self.monotonic_ns(),
                "reason": reason,
            },
        )
        self._published.add(path.name)

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("command must be an object")
        return value

    def drain(self) -> list[dict[str, Any]]:
        if not self.config.enabled or not self.config.command_dir.is_dir():
            return []
        published: list[dict[str, Any]] = []
        for path in sorted(self.config.command_dir.glob("*.json")):
            if path.name in self._published:
                continue
            if not path.name.startswith(("mission-", "cancel-", "estop-")):
                continue
            try:
                value = self._read(path)
                if path.name.startswith("mission-"):
                    if (
                        value.get("route") != MISSION_ROUTE
                        or value.get("internvla_raw_instruction_allowed") is not False
                        or value.get("config_sha256") != self.config.config_sha256
                        or value.get("status") != "QUEUED_FOR_STEP3"
                        or int(value.get("deadline_wall_monotonic_ns") or 0)
                        < self.monotonic_ns()
                    ):
                        raise ValueError(
                            "mission command failed Step3-first validation"
                        )
                    kind = "mission"
                elif path.name.startswith("cancel-"):
                    if not _MISSION_ID.fullmatch(str(value.get("mission_id") or "")):
                        raise ValueError("cancel command mission identity is invalid")
                    kind = "cancel"
                else:
                    if value.get("action") not in {"latch", "clear"} or not isinstance(
                        value.get("estop_latched"), bool
                    ):
                        raise ValueError("E-stop command is invalid")
                    kind = "estop"
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                reason = type(exc).__name__
                self._ack(path, "rejected", reason=reason)
                published.append(
                    {"filename": path.name, "kind": "rejected", "reason": reason}
                )
                continue
            if kind == "mission":
                self.publish_mission(value)
            elif kind == "cancel":
                self.publish_cancel(value)
            else:
                self.publish_estop(bool(value["estop_latched"]))
            self._ack(path, kind)
            published.append({"filename": path.name, "kind": kind})
        return published
