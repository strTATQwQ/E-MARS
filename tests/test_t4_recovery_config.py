from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from t4_completion.recovery.config import (
    HARD_MAX_ANGULAR_SPEED_RPS,
    HARD_MAX_LINEAR_SPEED_MPS,
    HARD_MAX_RETREAT_DISTANCE_M,
    RecoveryConfig,
    load_recovery_config,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_A = ROOT / "configs/completion_sim/recovery/profile_a.json"
PROFILE_B = ROOT / "configs/completion_sim/recovery/profile_b.json"


def _mapping(path: Path = PROFILE_A) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_b_profiles_load_and_share_frozen_safety_contract() -> None:
    first = load_recovery_config(PROFILE_A)
    second = load_recovery_config(PROFILE_B)
    assert (first.profile_id, second.profile_id) == ("A", "B")
    for profile in (first, second):
        assert profile.scan_angular_speed_rps <= HARD_MAX_ANGULAR_SPEED_RPS
        assert profile.retreat_linear_speed_mps <= HARD_MAX_LINEAR_SPEED_MPS
        assert profile.retreat_distance_m <= HARD_MAX_RETREAT_DISTANCE_M
        assert profile.maximum_commands_per_recovery >= 5
    for mapping in (_mapping(PROFILE_A), _mapping(PROFILE_B)):
        assert mapping["runtime_guard"] == {
            "runtime_policy": "completion_sim",
            "simulation_only": True,
            "real_go2_allowed": False,
            "hardware_motion_allowed": False,
            "simulation_estop_required": True,
            "bounded_velocity_required": True,
            "strict_evidence_unchanged": True,
        }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("runtime_guard", "runtime_policy"), "strict_evidence"),
        (("runtime_guard", "simulation_only"), False),
        (("runtime_guard", "real_go2_allowed"), True),
        (("runtime_guard", "hardware_motion_allowed"), True),
        (("runtime_guard", "simulation_estop_required"), False),
        (("runtime_guard", "strict_evidence_unchanged"), False),
        (("motion", "scan_angular_speed_rps"), 0.350001),
        (("motion", "retreat_linear_speed_mps"), 0.050001),
        (("motion", "retreat_distance_m"), 0.150001),
        (("bounds", "maximum_recoveries_per_episode"), 4),
        (("bounds", "maximum_action_attempts"), 4),
        (("bounds", "maximum_commands_per_recovery"), 25),
        (("bounds", "cooldown_sec"), 0.0),
    ],
)
def test_unsafe_or_unbounded_configuration_is_rejected(
    path: tuple[str, str], value: object
) -> None:
    mapping = _mapping()
    section = mapping[path[0]]
    assert isinstance(section, dict)
    section[path[1]] = value
    with pytest.raises(ValueError):
        RecoveryConfig.from_mapping(mapping)


def test_unknown_keys_and_boolean_integer_confusion_fail_closed() -> None:
    mapping = _mapping()
    mapping["unexpected"] = True
    with pytest.raises(ValueError, match="keys mismatch"):
        RecoveryConfig.from_mapping(mapping)

    mapping = _mapping()
    detector = mapping["detector"]
    assert isinstance(detector, dict)
    detector["minimum_sample_count"] = True
    with pytest.raises(ValueError, match="integer"):
        RecoveryConfig.from_mapping(mapping)

    mapping = _mapping()
    mapping["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        RecoveryConfig.from_mapping(mapping)

    mapping = _mapping()
    mapping["profile_id"] = ["A"]
    with pytest.raises(ValueError, match="profile_id"):
        RecoveryConfig.from_mapping(mapping)


def test_source_mapping_is_not_mutated() -> None:
    mapping = _mapping()
    original = deepcopy(mapping)
    RecoveryConfig.from_mapping(mapping)
    assert mapping == original


def test_frozen_dataclass_revalidates_replace_and_direct_machine_inputs() -> None:
    profile = load_recovery_config(PROFILE_A)
    with pytest.raises(ValueError, match="scan_angular_speed_rps"):
        replace(profile, scan_angular_speed_rps=99.0)
    with pytest.raises(ValueError, match="real_go2_allowed"):
        replace(profile, real_go2_allowed=True)


def test_loader_requires_a_real_json_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_recovery_config(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        load_recovery_config(invalid)
