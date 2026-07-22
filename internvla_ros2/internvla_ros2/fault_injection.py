"""T5 completion_sim-only fault schedule and runtime control helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from .identity import CHECKPOINT_REVISION, MODEL_REVISION


FAULT_PROFILE = "completion_sim_minimal_v1"
FAULT_KINDS = (
    "model_request_timeout",
    "model_service_restart",
    "dgx_ros_node_restart",
    "episode_reset",
    "lidar_depth_short_outage",
    "network_short_outage_and_recovery",
)
FAULT_SPECS = (
    ("fi-01-model-request-timeout", FAULT_KINDS[0], 60.0, 0.0, "dgx_model"),
    ("fi-02-model-service-restart", FAULT_KINDS[1], 140.0, 0.0, "dgx_model_supervisor"),
    ("fi-03-dgx-ros-node-restart", FAULT_KINDS[2], 240.0, 0.0, "dgx_onboard_supervisor"),
    ("fi-04-episode-reset", FAULT_KINDS[3], 340.0, 0.0, "x86_model_agent"),
    ("fi-05-lidar-depth-short-outage", FAULT_KINDS[4], 420.0, 8.0, "x86_isaac_controller"),
    ("fi-06-network-short-outage", FAULT_KINDS[5], 510.0, 8.0, "x86_cross_host_data_plane"),
)
ONE_SHOT_FAULTS = frozenset(FAULT_KINDS[:4])
BOUNDED_OUTAGE_FAULTS = frozenset(FAULT_KINDS[4:])
FAULT_RESTART_ACTIONS = frozenset(
    {"model_service_restart", "dgx_ros_node_restart"}
)


def build_fault_restart_session(
    health: Mapping[str, Any],
    *,
    lane: str,
    event_id: str,
    action: str,
    captured_unix: float | None = None,
) -> dict[str, Any]:
    """Bind one initialized model generation across a bounded restart."""

    value = {
        "schema_version": 1,
        "profile": FAULT_PROFILE,
        "status": "PASS",
        "lane": lane,
        "event_id": event_id,
        "action": action,
        "status_code": health.get("status_code"),
        "status_message": str(health.get("status_message", "")),
        "initialized": health.get("initialized"),
        "lifecycle_state": health.get("lifecycle_state"),
        "episode_id": str(health.get("episode_id", "")),
        "reset_generation": health.get("reset_generation"),
        "last_sequence_id": health.get("last_sequence_id"),
        "model_revision": str(health.get("model_revision", "")),
        "checkpoint_revision": str(health.get("checkpoint_revision", "")),
        "captured_unix": time.time() if captured_unix is None else captured_unix,
    }
    return validate_fault_restart_session(
        value, expected_lane=lane, expected_event_id=event_id, expected_action=action
    )


def validate_fault_restart_session(
    value: Any,
    *,
    expected_lane: str,
    expected_event_id: str | None = None,
    expected_action: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless ``value`` is one exact, live T5 model session."""

    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("fault restart session schema_version must be 1")
    lane = value.get("lane")
    event_id = value.get("event_id")
    action = value.get("action")
    episode_id = value.get("episode_id")
    reset_generation = value.get("reset_generation")
    last_sequence_id = value.get("last_sequence_id")
    captured_unix = value.get("captured_unix")
    if (
        value.get("profile") != FAULT_PROFILE
        or value.get("status") != "PASS"
        or expected_lane not in {"a", "b"}
        or lane != expected_lane
        or not isinstance(event_id, str)
        or not event_id.startswith("fi-")
        or len(event_id) > 96
        or action not in FAULT_RESTART_ACTIONS
        or (expected_event_id is not None and event_id != expected_event_id)
        or (expected_action is not None and action != expected_action)
        or value.get("status_code") != 0
        or value.get("initialized") is not True
        or value.get("lifecycle_state") != 2
        or not isinstance(episode_id, str)
        or not episode_id.startswith(f"{lane}::")
        or len(episode_id.encode("utf-8")) > 256
        or isinstance(reset_generation, bool)
        or not isinstance(reset_generation, int)
        or reset_generation < 0
        or isinstance(last_sequence_id, bool)
        or not isinstance(last_sequence_id, int)
        or last_sequence_id < -1
        or value.get("model_revision") != MODEL_REVISION
        or value.get("checkpoint_revision") != CHECKPOINT_REVISION
        or not _is_number(captured_unix)
        or float(captured_unix) <= 0.0
    ):
        raise ValueError("fault restart session identity mismatch")
    return dict(value)


def load_fault_restart_session(
    path: Path,
    *,
    expected_lane: str,
    expected_event_id: str | None = None,
    expected_action: str | None = None,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ValueError("fault restart session must be a regular file")
    return validate_fault_restart_session(
        json.loads(resolved.read_text(encoding="utf-8")),
        expected_lane=expected_lane,
        expected_event_id=expected_event_id,
        expected_action=expected_action,
    )


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_fault_plan(value: Any) -> dict[str, Any]:
    """Validate and return the exact minimal 600-sim-second fault plan."""

    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("fault plan schema_version must be 1")
    if value.get("profile") != FAULT_PROFILE:
        raise ValueError("unsupported fault injection profile")
    if value.get("soak_duration_sim_sec") != 600:
        raise ValueError("fault injection profile requires a 600 sim-s soak")
    if value.get("schedule_timebase") != "sim_time":
        raise ValueError("fault schedule must use simulation time")
    if value.get("wall_time_scope") != "actuator_and_process_liveness_only":
        raise ValueError("wall time may be used only for liveness")
    events = value.get("events")
    if (
        not isinstance(events, list)
        or len(events) != len(FAULT_KINDS)
        or len(events) != len(FAULT_SPECS)
    ):
        raise ValueError("fault plan must contain exactly the six required events")
    seen_ids: set[str] = set()
    previous_end = 0.0
    observed_kinds: list[str] = []
    # The WSL coordination host uses Python 3.8, which predates zip(strict=).
    # Exact cardinality is enforced above before pairing the frozen schedule.
    for event, expected in zip(events, FAULT_SPECS):
        if not isinstance(event, dict):
            raise ValueError("fault event must be an object")
        event_id = event.get("event_id")
        kind = event.get("kind")
        start = event.get("start_sim_offset_sec")
        duration = event.get("duration_sim_sec")
        target = event.get("target")
        if (
            not isinstance(event_id, str)
            or not event_id.startswith("fi-")
            or len(event_id) > 96
            or event_id in seen_ids
        ):
            raise ValueError("fault event_id is invalid or duplicated")
        if kind not in FAULT_KINDS:
            raise ValueError("fault kind is outside the minimal profile")
        if not _is_number(start) or not _is_number(duration):
            raise ValueError("fault offsets and durations must be finite numbers")
        start_value = float(start)
        duration_value = float(duration)
        if start_value < previous_end + 20.0 or start_value >= 570.0:
            raise ValueError("fault events require a bounded recovery gap")
        if kind in ONE_SHOT_FAULTS and duration_value != 0.0:
            raise ValueError("one-shot fault duration must be zero")
        if kind in BOUNDED_OUTAGE_FAULTS and not 1.0 <= duration_value <= 15.0:
            raise ValueError("short outage duration must be 1-15 sim seconds")
        if not isinstance(target, str) or not target:
            raise ValueError("fault event target is required")
        observed_spec = (
            event_id,
            kind,
            start_value,
            duration_value,
            target,
        )
        if observed_spec != expected:
            raise ValueError("fault event differs from the frozen minimal schedule")
        previous_end = start_value + duration_value
        seen_ids.add(event_id)
        observed_kinds.append(str(kind))
    if tuple(observed_kinds) != FAULT_KINDS:
        raise ValueError("fault event order must match the frozen minimal profile")
    invariants = value.get("required_invariants")
    expected_invariants = {
        "safe_stop_before_destructive_restart": True,
        "sensor_and_action_freshness_timebase": "sim_time",
        "stale_response_discard": True,
        "wall_time_used_only_for_liveness": True,
        "final_residual_count": 0,
        "real_go2_eligible": False,
    }
    if invariants != expected_invariants:
        raise ValueError("fault plan safety invariants differ from the frozen contract")
    return value


def load_fault_plan(path: Path) -> dict[str, Any]:
    return validate_fault_plan(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class FaultActivation:
    event_id: str
    kind: str


@dataclass(frozen=True)
class FaultControlSnapshot:
    lane: str
    revision: int
    observed_sim_ns: int
    active: tuple[FaultActivation, ...]

    def event_for(self, kind: str) -> str | None:
        for activation in self.active:
            if activation.kind == kind:
                return activation.event_id
        return None


def validate_control_snapshot(value: Any, *, expected_lane: str) -> FaultControlSnapshot:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("fault control schema_version must be 1")
    if value.get("profile") != FAULT_PROFILE:
        raise ValueError("fault control profile mismatch")
    if expected_lane not in {"a", "b"} or value.get("lane") != expected_lane:
        raise ValueError("fault control lane mismatch")
    revision = value.get("revision")
    observed_sim_ns = value.get("observed_sim_ns")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("fault control revision is invalid")
    if (
        isinstance(observed_sim_ns, bool)
        or not isinstance(observed_sim_ns, int)
        or observed_sim_ns < 0
    ):
        raise ValueError("fault control sim stamp is invalid")
    active_value = value.get("active")
    if not isinstance(active_value, list):
        raise ValueError("fault control active set must be a list")
    active: list[FaultActivation] = []
    seen_ids: set[str] = set()
    seen_kinds: set[str] = set()
    for item in active_value:
        if not isinstance(item, dict):
            raise ValueError("fault activation must be an object")
        event_id = item.get("event_id")
        kind = item.get("kind")
        if (
            not isinstance(event_id, str)
            or not event_id.startswith("fi-")
            or event_id in seen_ids
            or kind not in FAULT_KINDS
            or kind in seen_kinds
        ):
            raise ValueError("fault activation identity is invalid or duplicated")
        seen_ids.add(event_id)
        seen_kinds.add(str(kind))
        active.append(FaultActivation(event_id, str(kind)))
    return FaultControlSnapshot(
        expected_lane, revision, observed_sim_ns, tuple(active)
    )


def fault_profile_enabled() -> bool:
    profile = os.environ.get("INTERNNAV_T5_FAULT_INJECTION_PROFILE", "off")
    if profile == "off":
        return False
    if profile != FAULT_PROFILE:
        raise RuntimeError("unsupported T5 fault injection profile")
    if (
        os.environ.get("INTERNNAV_RUNTIME_POLICY") != "completion_sim"
        or os.environ.get("INTERNNAV_SIMULATION_TARGET") != "isaac"
        or os.environ.get("INTERNNAV_T5_LANE") not in {"a", "b"}
    ):
        raise RuntimeError("fault injection is restricted to an isolated T5 completion_sim lane")
    return True


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


class FaultControlReader:
    """Read an atomic control snapshot and record component-local transitions."""

    def __init__(self, component: str):
        if not fault_profile_enabled():
            raise RuntimeError("fault reader requires the enabled minimal profile")
        lane = os.environ["INTERNNAV_T5_LANE"]
        control = Path(os.environ.get("INTERNNAV_T5_FAULT_CONTROL_PATH", ""))
        events = Path(os.environ.get("INTERNNAV_T5_FAULT_EVENT_PATH", ""))
        if not control.is_absolute() or not events.is_absolute():
            raise RuntimeError("fault control and event paths must be absolute")
        self.lane = lane
        self.component = component
        self.control_path = control
        self.event_path = events
        self.last_revision = -1
        self.last_active = self._restore_active_transitions()
        self.consumed: set[str] = set()

    def _restore_active_transitions(self) -> dict[str, str]:
        """Continue an active/recovered transition across a component restart."""

        if not self.event_path.is_file():
            return {}
        active: dict[str, str] = {}
        records = read_event_records(
            self.event_path.read_text(encoding="utf-8").splitlines()
        )
        for record in records:
            if (
                record.get("profile") != FAULT_PROFILE
                or record.get("lane") != self.lane
                or record.get("component") != self.component
            ):
                continue
            kind = record.get("kind")
            event_id = record.get("event_id")
            if not isinstance(kind, str) or not isinstance(event_id, str):
                raise ValueError("fault event transition identity is invalid")
            if record.get("phase") == "observed_active":
                active[kind] = event_id
            elif (
                record.get("phase") == "observed_recovered"
                and active.get(kind) == event_id
            ):
                active.pop(kind)
        return active

    def read(self) -> FaultControlSnapshot:
        if not self.control_path.exists():
            return FaultControlSnapshot(self.lane, 0, 0, ())
        value = json.loads(self.control_path.read_text(encoding="utf-8"))
        snapshot = validate_control_snapshot(value, expected_lane=self.lane)
        if snapshot.revision < self.last_revision:
            raise RuntimeError("fault control revision regressed")
        current = {item.kind: item.event_id for item in snapshot.active}
        for kind, event_id in current.items():
            if self.last_active.get(kind) != event_id:
                self.record(event_id, kind, "observed_active", snapshot.observed_sim_ns)
        for kind, event_id in self.last_active.items():
            if current.get(kind) != event_id:
                self.record(event_id, kind, "observed_recovered", snapshot.observed_sim_ns)
        self.last_revision = snapshot.revision
        self.last_active = current
        return snapshot

    def consume_once(self, kind: str) -> str | None:
        snapshot = self.read()
        event_id = snapshot.event_for(kind)
        if event_id is None or event_id in self.consumed:
            return None
        self.consumed.add(event_id)
        self.record(event_id, kind, "consumed", snapshot.observed_sim_ns)
        return event_id

    def record(
        self, event_id: str, kind: str, phase: str, observed_sim_ns: int
    ) -> None:
        _append_jsonl(
            self.event_path,
            {
                "schema_version": 1,
                "profile": FAULT_PROFILE,
                "lane": self.lane,
                "component": self.component,
                "event_id": event_id,
                "kind": kind,
                "phase": phase,
                "observed_sim_ns": int(observed_sim_ns),
                "wall_unix": time.time(),
            },
        )


def read_event_records(lines: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("fault event record is invalid")
        records.append(value)
    return records
