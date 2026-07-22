"""Fail-closed configuration for the T4.5 completion-sim recovery core."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


HARD_MAX_LINEAR_SPEED_MPS = 0.05
HARD_MAX_ANGULAR_SPEED_RPS = 0.35
HARD_MAX_RETREAT_DISTANCE_M = 0.15
HARD_MAX_SCAN_YAW_RAD = math.pi / 3.0
HARD_MAX_RECOVERIES_PER_EPISODE = 3
HARD_MAX_ACTION_ATTEMPTS = 3
HARD_MAX_REPLAN_REQUESTS_PER_STAGE = 3
HARD_MAX_PLAN_REJECTIONS_PER_STAGE = 3
HARD_MAX_COMMANDS_PER_RECOVERY = 24
HARD_MAX_RECOVERY_DURATION_SEC = 60.0
HARD_MAX_EVENT_HISTORY = 4096


def _object(value: object, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ValueError(f"{name} keys mismatch; missing={missing}, extra={extra}")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output) or not minimum <= output <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return output


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    """Validated parameters; safety booleans are frozen, not tuning knobs."""

    profile_id: str
    runtime_policy: str
    simulation_only: bool
    real_go2_allowed: bool
    hardware_motion_allowed: bool
    simulation_estop_required: bool
    bounded_velocity_required: bool
    strict_evidence_unchanged: bool
    no_progress_window_sec: float
    minimum_window_coverage: float
    minimum_sample_count: int
    maximum_observation_gap_sec: float
    minimum_displacement_m: float
    minimum_goal_progress_m: float
    loop_window_sec: float
    loop_cell_size_m: float
    loop_revisit_count: int
    loop_min_travel_m: float
    loop_max_net_displacement_m: float
    plan_cycle_repetitions: int
    trajectory_resolution_m: float
    trajectory_max_points: int
    taboo_capacity: int
    action_timeout_sec: float
    plan_wait_timeout_sec: float
    retry_backoff_sec: float
    cooldown_sec: float
    maximum_action_attempts: int
    maximum_replan_requests_per_stage: int
    maximum_plan_rejections_per_stage: int
    maximum_recoveries_per_episode: int
    maximum_commands_per_recovery: int
    maximum_recovery_duration_sec: float
    scan_yaw_rad: float
    scan_angular_speed_rps: float
    safety_freshness_timeout_sec: float
    retreat_enabled: bool
    retreat_distance_m: float
    retreat_linear_speed_mps: float
    retreat_minimum_rear_clearance_m: float
    event_history_capacity: int

    def __post_init__(self) -> None:
        _validate_config_instance(self)

    def validate(self) -> None:
        """Revalidate at the machine boundary as defense in depth."""

        _validate_config_instance(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RecoveryConfig":
        root = _object(
            value,
            "root",
            {
                "schema_version",
                "profile_id",
                "runtime_guard",
                "detector",
                "bounds",
                "motion",
                "events",
            },
        )
        if type(root["schema_version"]) is not int or root["schema_version"] != 1:
            raise ValueError("schema_version must be 1")
        profile_id = root["profile_id"]
        if not isinstance(profile_id, str) or profile_id not in {"A", "B"}:
            raise ValueError("profile_id must be A or B")

        guard = _object(
            root["runtime_guard"],
            "runtime_guard",
            {
                "runtime_policy",
                "simulation_only",
                "real_go2_allowed",
                "hardware_motion_allowed",
                "simulation_estop_required",
                "bounded_velocity_required",
                "strict_evidence_unchanged",
            },
        )
        if guard["runtime_policy"] != "completion_sim":
            raise ValueError("recovery profiles are completion_sim only")
        frozen_guard = {
            "simulation_only": True,
            "real_go2_allowed": False,
            "hardware_motion_allowed": False,
            "simulation_estop_required": True,
            "bounded_velocity_required": True,
            "strict_evidence_unchanged": True,
        }
        for key, required in frozen_guard.items():
            if _boolean(guard[key], f"runtime_guard.{key}") is not required:
                raise ValueError(f"runtime_guard.{key} must remain {required}")

        detector = _object(
            root["detector"],
            "detector",
            {
                "no_progress_window_sec",
                "minimum_window_coverage",
                "minimum_sample_count",
                "maximum_observation_gap_sec",
                "minimum_displacement_m",
                "minimum_goal_progress_m",
                "loop_window_sec",
                "loop_cell_size_m",
                "loop_revisit_count",
                "loop_min_travel_m",
                "loop_max_net_displacement_m",
                "plan_cycle_repetitions",
                "trajectory_resolution_m",
                "trajectory_max_points",
                "taboo_capacity",
            },
        )
        bounds = _object(
            root["bounds"],
            "bounds",
            {
                "action_timeout_sec",
                "plan_wait_timeout_sec",
                "retry_backoff_sec",
                "cooldown_sec",
                "maximum_action_attempts",
                "maximum_replan_requests_per_stage",
                "maximum_plan_rejections_per_stage",
                "maximum_recoveries_per_episode",
                "maximum_commands_per_recovery",
                "maximum_recovery_duration_sec",
            },
        )
        motion = _object(
            root["motion"],
            "motion",
            {
                "scan_yaw_rad",
                "scan_angular_speed_rps",
                "safety_freshness_timeout_sec",
                "retreat_enabled",
                "retreat_distance_m",
                "retreat_linear_speed_mps",
                "retreat_minimum_rear_clearance_m",
            },
        )
        events = _object(root["events"], "events", {"history_capacity"})

        output = cls(
            profile_id=profile_id,
            runtime_policy=str(guard["runtime_policy"]),
            simulation_only=bool(guard["simulation_only"]),
            real_go2_allowed=bool(guard["real_go2_allowed"]),
            hardware_motion_allowed=bool(guard["hardware_motion_allowed"]),
            simulation_estop_required=bool(guard["simulation_estop_required"]),
            bounded_velocity_required=bool(guard["bounded_velocity_required"]),
            strict_evidence_unchanged=bool(guard["strict_evidence_unchanged"]),
            no_progress_window_sec=_number(
                detector["no_progress_window_sec"],
                "detector.no_progress_window_sec",
                2.0,
                10.0,
            ),
            minimum_window_coverage=_number(
                detector["minimum_window_coverage"],
                "detector.minimum_window_coverage",
                0.7,
                1.0,
            ),
            minimum_sample_count=_integer(
                detector["minimum_sample_count"],
                "detector.minimum_sample_count",
                4,
                100,
            ),
            maximum_observation_gap_sec=_number(
                detector["maximum_observation_gap_sec"],
                "detector.maximum_observation_gap_sec",
                0.05,
                2.0,
            ),
            minimum_displacement_m=_number(
                detector["minimum_displacement_m"],
                "detector.minimum_displacement_m",
                0.02,
                0.30,
            ),
            minimum_goal_progress_m=_number(
                detector["minimum_goal_progress_m"],
                "detector.minimum_goal_progress_m",
                0.02,
                0.30,
            ),
            loop_window_sec=_number(
                detector["loop_window_sec"], "detector.loop_window_sec", 2.0, 20.0
            ),
            loop_cell_size_m=_number(
                detector["loop_cell_size_m"], "detector.loop_cell_size_m", 0.05, 0.5
            ),
            loop_revisit_count=_integer(
                detector["loop_revisit_count"],
                "detector.loop_revisit_count",
                2,
                10,
            ),
            loop_min_travel_m=_number(
                detector["loop_min_travel_m"],
                "detector.loop_min_travel_m",
                0.10,
                3.0,
            ),
            loop_max_net_displacement_m=_number(
                detector["loop_max_net_displacement_m"],
                "detector.loop_max_net_displacement_m",
                0.02,
                0.50,
            ),
            plan_cycle_repetitions=_integer(
                detector["plan_cycle_repetitions"],
                "detector.plan_cycle_repetitions",
                2,
                4,
            ),
            trajectory_resolution_m=_number(
                detector["trajectory_resolution_m"],
                "detector.trajectory_resolution_m",
                0.01,
                0.20,
            ),
            trajectory_max_points=_integer(
                detector["trajectory_max_points"],
                "detector.trajectory_max_points",
                2,
                4096,
            ),
            taboo_capacity=_integer(
                detector["taboo_capacity"], "detector.taboo_capacity", 32, 256
            ),
            action_timeout_sec=_number(
                bounds["action_timeout_sec"], "bounds.action_timeout_sec", 0.1, 15.0
            ),
            plan_wait_timeout_sec=_number(
                bounds["plan_wait_timeout_sec"],
                "bounds.plan_wait_timeout_sec",
                0.1,
                15.0,
            ),
            retry_backoff_sec=_number(
                bounds["retry_backoff_sec"], "bounds.retry_backoff_sec", 0.0, 2.0
            ),
            cooldown_sec=_number(
                bounds["cooldown_sec"], "bounds.cooldown_sec", 1.0, 10.0
            ),
            maximum_action_attempts=_integer(
                bounds["maximum_action_attempts"],
                "bounds.maximum_action_attempts",
                1,
                HARD_MAX_ACTION_ATTEMPTS,
            ),
            maximum_replan_requests_per_stage=_integer(
                bounds["maximum_replan_requests_per_stage"],
                "bounds.maximum_replan_requests_per_stage",
                1,
                HARD_MAX_REPLAN_REQUESTS_PER_STAGE,
            ),
            maximum_plan_rejections_per_stage=_integer(
                bounds["maximum_plan_rejections_per_stage"],
                "bounds.maximum_plan_rejections_per_stage",
                1,
                HARD_MAX_PLAN_REJECTIONS_PER_STAGE,
            ),
            maximum_recoveries_per_episode=_integer(
                bounds["maximum_recoveries_per_episode"],
                "bounds.maximum_recoveries_per_episode",
                1,
                HARD_MAX_RECOVERIES_PER_EPISODE,
            ),
            maximum_commands_per_recovery=_integer(
                bounds["maximum_commands_per_recovery"],
                "bounds.maximum_commands_per_recovery",
                5,
                HARD_MAX_COMMANDS_PER_RECOVERY,
            ),
            maximum_recovery_duration_sec=_number(
                bounds["maximum_recovery_duration_sec"],
                "bounds.maximum_recovery_duration_sec",
                5.0,
                HARD_MAX_RECOVERY_DURATION_SEC,
            ),
            scan_yaw_rad=_number(
                motion["scan_yaw_rad"],
                "motion.scan_yaw_rad",
                0.1,
                HARD_MAX_SCAN_YAW_RAD,
            ),
            scan_angular_speed_rps=_number(
                motion["scan_angular_speed_rps"],
                "motion.scan_angular_speed_rps",
                0.01,
                HARD_MAX_ANGULAR_SPEED_RPS,
            ),
            safety_freshness_timeout_sec=_number(
                motion["safety_freshness_timeout_sec"],
                "motion.safety_freshness_timeout_sec",
                0.05,
                2.0,
            ),
            retreat_enabled=_boolean(
                motion["retreat_enabled"], "motion.retreat_enabled"
            ),
            retreat_distance_m=_number(
                motion["retreat_distance_m"],
                "motion.retreat_distance_m",
                0.01,
                HARD_MAX_RETREAT_DISTANCE_M,
            ),
            retreat_linear_speed_mps=_number(
                motion["retreat_linear_speed_mps"],
                "motion.retreat_linear_speed_mps",
                0.01,
                HARD_MAX_LINEAR_SPEED_MPS,
            ),
            retreat_minimum_rear_clearance_m=_number(
                motion["retreat_minimum_rear_clearance_m"],
                "motion.retreat_minimum_rear_clearance_m",
                HARD_MAX_RETREAT_DISTANCE_M,
                2.0,
            ),
            event_history_capacity=_integer(
                events["history_capacity"],
                "events.history_capacity",
                64,
                HARD_MAX_EVENT_HISTORY,
            ),
        )
        minimum_motion_timeout = max(
            output.scan_yaw_rad / output.scan_angular_speed_rps,
            (
                output.retreat_distance_m / output.retreat_linear_speed_mps
                if output.retreat_enabled
                else 0.0
            ),
        )
        if output.action_timeout_sec < minimum_motion_timeout:
            raise ValueError(
                "bounds.action_timeout_sec is shorter than a bounded motion action"
            )
        if output.maximum_recovery_duration_sec < output.action_timeout_sec:
            raise ValueError(
                "maximum_recovery_duration_sec must cover at least one action timeout"
            )
        if output.retreat_minimum_rear_clearance_m < output.retreat_distance_m:
            raise ValueError("rear clearance must cover the full retreat distance")
        return output


def _validate_config_instance(config: RecoveryConfig) -> None:
    if not isinstance(config.profile_id, str) or config.profile_id not in {"A", "B"}:
        raise ValueError("profile_id must be A or B")
    if config.runtime_policy != "completion_sim":
        raise ValueError("runtime_policy must be completion_sim")
    frozen_guards = {
        "simulation_only": True,
        "real_go2_allowed": False,
        "hardware_motion_allowed": False,
        "simulation_estop_required": True,
        "bounded_velocity_required": True,
        "strict_evidence_unchanged": True,
    }
    for name, expected in frozen_guards.items():
        if _boolean(getattr(config, name), name) is not expected:
            raise ValueError(f"{name} must remain {expected}")

    number_bounds = {
        "no_progress_window_sec": (2.0, 10.0),
        "minimum_window_coverage": (0.7, 1.0),
        "maximum_observation_gap_sec": (0.05, 2.0),
        "minimum_displacement_m": (0.02, 0.30),
        "minimum_goal_progress_m": (0.02, 0.30),
        "loop_window_sec": (2.0, 20.0),
        "loop_cell_size_m": (0.05, 0.5),
        "loop_min_travel_m": (0.10, 3.0),
        "loop_max_net_displacement_m": (0.02, 0.50),
        "trajectory_resolution_m": (0.01, 0.20),
        "action_timeout_sec": (0.1, 15.0),
        "plan_wait_timeout_sec": (0.1, 15.0),
        "retry_backoff_sec": (0.0, 2.0),
        "cooldown_sec": (1.0, 10.0),
        "maximum_recovery_duration_sec": (
            5.0,
            HARD_MAX_RECOVERY_DURATION_SEC,
        ),
        "scan_yaw_rad": (0.1, HARD_MAX_SCAN_YAW_RAD),
        "scan_angular_speed_rps": (0.01, HARD_MAX_ANGULAR_SPEED_RPS),
        "safety_freshness_timeout_sec": (0.05, 2.0),
        "retreat_distance_m": (0.01, HARD_MAX_RETREAT_DISTANCE_M),
        "retreat_linear_speed_mps": (0.01, HARD_MAX_LINEAR_SPEED_MPS),
        "retreat_minimum_rear_clearance_m": (
            HARD_MAX_RETREAT_DISTANCE_M,
            2.0,
        ),
    }
    for name, (minimum, maximum) in number_bounds.items():
        _number(getattr(config, name), name, minimum, maximum)

    integer_bounds = {
        "minimum_sample_count": (4, 100),
        "loop_revisit_count": (2, 10),
        "plan_cycle_repetitions": (2, 4),
        "trajectory_max_points": (2, 4096),
        "taboo_capacity": (32, 256),
        "maximum_action_attempts": (1, HARD_MAX_ACTION_ATTEMPTS),
        "maximum_replan_requests_per_stage": (
            1,
            HARD_MAX_REPLAN_REQUESTS_PER_STAGE,
        ),
        "maximum_plan_rejections_per_stage": (
            1,
            HARD_MAX_PLAN_REJECTIONS_PER_STAGE,
        ),
        "maximum_recoveries_per_episode": (
            1,
            HARD_MAX_RECOVERIES_PER_EPISODE,
        ),
        "maximum_commands_per_recovery": (5, HARD_MAX_COMMANDS_PER_RECOVERY),
        "event_history_capacity": (64, HARD_MAX_EVENT_HISTORY),
    }
    for name, (minimum, maximum) in integer_bounds.items():
        _integer(getattr(config, name), name, minimum, maximum)
    _boolean(config.retreat_enabled, "retreat_enabled")

    minimum_motion_timeout = max(
        config.scan_yaw_rad / config.scan_angular_speed_rps,
        (
            config.retreat_distance_m / config.retreat_linear_speed_mps
            if config.retreat_enabled
            else 0.0
        ),
    )
    if config.action_timeout_sec < minimum_motion_timeout:
        raise ValueError("action_timeout_sec is shorter than a bounded motion action")
    if config.maximum_recovery_duration_sec < config.action_timeout_sec:
        raise ValueError("maximum recovery duration must cover an action timeout")
    if config.retreat_minimum_rear_clearance_m < config.retreat_distance_m:
        raise ValueError("rear clearance must cover the full retreat distance")


def load_recovery_config(path: str | Path) -> RecoveryConfig:
    """Load one immutable JSON candidate; no environment overrides are accepted."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("recovery config root must be an object")
    return RecoveryConfig.from_mapping(value)
