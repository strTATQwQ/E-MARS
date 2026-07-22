"""Named runtime policies separating functional simulation from strict evidence.

The relaxed policy is selected only by the explicit ``completion_sim`` session
profile.  Hardware and all legacy entrypoints therefore remain strict by
default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    name: str
    recorder_mode: str
    bridge_timeout_sec: float
    downstream_watchdog_sec: float
    downstream_growth_timeout_sec: float
    transform_tolerance_sec: float
    source_timeout_sec: float
    costmap_current_timeout_sec: float
    observation_persistence_sec: float
    collision_monitor_mode: str
    nvblox_mode: str
    global_map_mode: str
    render_stamp_deviation_tolerance_ns: int
    exact_atomic_evidence_required: bool
    sigterm_143_is_normal_with_zero_residuals: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sensor_wire_send_timeout_sec(self) -> float:
        """Keep strict transport timing frozen; bound completion backpressure."""

        return 5.0 if self.name == "completion_sim" else 0.2

    @property
    def render_resync_limit(self) -> int:
        """Bound render-only catch-up; strict still performs exactly one render."""

        return 4 if self.name == "completion_sim" else 1


POLICIES = MappingProxyType(
    {
        "strict_evidence": RuntimePolicy(
            name="strict_evidence",
            recorder_mode="fatal_exact_batch",
            bridge_timeout_sec=0.35,
            downstream_watchdog_sec=0.35,
            downstream_growth_timeout_sec=0.35,
            transform_tolerance_sec=0.0,
            source_timeout_sec=0.35,
            costmap_current_timeout_sec=0.35,
            observation_persistence_sec=0.0,
            collision_monitor_mode="enforce",
            nvblox_mode="active_required",
            global_map_mode="nvblox",
            render_stamp_deviation_tolerance_ns=0,
            exact_atomic_evidence_required=True,
            sigterm_143_is_normal_with_zero_residuals=False,
        ),
        "completion_sim": RuntimePolicy(
            name="completion_sim",
            recorder_mode="nonfatal_consumer_shadow",
            bridge_timeout_sec=5.0,
            downstream_watchdog_sec=5.0,
            downstream_growth_timeout_sec=5.0,
            transform_tolerance_sec=2.5,
            source_timeout_sec=5.0,
            costmap_current_timeout_sec=5.0,
            observation_persistence_sec=5.0,
            collision_monitor_mode="warn_only",
            nvblox_mode="shadow",
            global_map_mode="static_with_lidar_local",
            render_stamp_deviation_tolerance_ns=1_000_000,
            exact_atomic_evidence_required=False,
            sigterm_143_is_normal_with_zero_residuals=True,
        ),
    }
)


def policy_for_session_profile(profile_name: str) -> RuntimePolicy:
    return POLICIES[
        "completion_sim"
        if profile_name in {"completion_sim", "completion_sim_map"}
        else "strict_evidence"
    ]


def require_policy(name: str) -> RuntimePolicy:
    try:
        return POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown runtime policy: {name}") from exc
