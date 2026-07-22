from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "internvla_t4_sensors"))

from internvla_t4_sensors.real_go2_mission_gateway import (
    CanonicalMissionLatch,
    MissionGatewayCore,
    MissionGatewayError,
)
from slow_planner.base import PlannerMetrics
from slow_planner.mission import CanonicalMission, MISSION_ROUTE


CONFIG_SHA = "b" * 64


def envelope(
    instruction: str = "穿过门口，在红色椅子旁边停下",
    *,
    now_ns: int = 1_000_000_000,
) -> dict:
    mission_id = "m-0123456789abcdef"
    return {
        "schema_version": 1,
        "route": MISSION_ROUTE,
        "mission_id": mission_id,
        "episode_id": f"real-{mission_id}",
        "reset_generation": 3,
        "sequence_id": 0,
        "identity": f"real::{mission_id}::3::0",
        "raw_instruction": instruction,
        "source_instruction_sha256": hashlib.sha256(
            instruction.encode("utf-8")
        ).hexdigest(),
        "config_sha256": CONFIG_SHA,
        "created_wall_time_s": time.time(),
        "deadline_wall_monotonic_ns": now_ns + 30_000_000_000,
        "internvla_raw_instruction_allowed": False,
    }


class FakeNormalizer:
    def __init__(self, *, abstain: bool = False) -> None:
        self.abstain = abstain
        self.requests = []

    def normalize_instruction(self, request):
        self.requests.append(request)
        return (
            CanonicalMission(
                mission_id=request.mission_id,
                episode_id=request.episode_id,
                reset_generation=request.reset_generation,
                sequence_id=request.sequence_id,
                source_language="zh",
                canonical_instruction=(
                    "" if self.abstain else "Go through the doorway and stop beside the red chair."
                ),
                target_description=(
                    "" if self.abstain else "the red chair beyond the doorway"
                ),
                constraints=() if self.abstain else ("Stop beside the chair",),
                confidence=0.0 if self.abstain else 0.9,
                abstain=self.abstain,
                source_instruction_sha256=hashlib.sha256(
                    request.instruction.encode("utf-8")
                ).hexdigest(),
                config_sha256=request.config_sha256,
            ),
            PlannerMetrics(
                model_variant="step3_vl_10b_bf16", end_to_end_ms=8123.0
            ),
        )


def test_gateway_sends_chinese_only_to_step3_and_releases_canonical_english() -> None:
    normalizer = FakeNormalizer()
    core = MissionGatewayCore(
        normalizer, expected_config_sha256=CONFIG_SHA, monotonic_ns=lambda: 1_000_000_000
    )
    result = core.process(envelope())
    assert normalizer.requests[0].instruction.startswith("穿过门口")
    assert result["status"] == "CANONICAL_READY"
    assert result["internvla"]["canonical_instruction"].startswith("Go through")
    assert result["internvla"]["raw_instruction_allowed"] is False
    assert "穿过门口" not in str(result)
    assert result["step3"]["raw_text_exposed"] is False


def test_gateway_rejects_expiry_config_mismatch_and_direct_internvla_route() -> None:
    normalizer = FakeNormalizer()
    expired = MissionGatewayCore(
        normalizer,
        expected_config_sha256=CONFIG_SHA,
        monotonic_ns=lambda: 100_000_000_000,
    )
    with pytest.raises(MissionGatewayError, match="expired"):
        expired.process(envelope())
    assert normalizer.requests == []

    current = MissionGatewayCore(
        normalizer,
        expected_config_sha256=CONFIG_SHA,
        monotonic_ns=lambda: 1_000_000_000,
    )
    wrong_config = envelope()
    wrong_config["config_sha256"] = "c" * 64
    with pytest.raises(MissionGatewayError, match="config SHA"):
        current.process(wrong_config)
    direct = envelope()
    direct["internvla_raw_instruction_allowed"] = True
    with pytest.raises(MissionGatewayError, match="raw InternVLA"):
        current.process(direct)


def test_step3_abstain_is_safe_hold_not_raw_instruction_fallback() -> None:
    core = MissionGatewayCore(
        FakeNormalizer(abstain=True),
        expected_config_sha256=CONFIG_SHA,
        monotonic_ns=lambda: 1_000_000_000,
    )
    with pytest.raises(MissionGatewayError, match="safe hold"):
        core.process(envelope())


def test_identity_is_idempotent_and_cannot_be_reused_with_different_text() -> None:
    normalizer = FakeNormalizer()
    core = MissionGatewayCore(
        normalizer, expected_config_sha256=CONFIG_SHA, monotonic_ns=lambda: 1_000_000_000
    )
    first = core.process(envelope())
    second = core.process(envelope())
    assert first == second
    assert len(normalizer.requests) == 1
    changed = envelope("走到蓝色门口")
    with pytest.raises(MissionGatewayError, match="reused"):
        core.process(changed)


def test_gateway_source_has_no_motion_or_terminal_stop_publisher() -> None:
    import internvla_t4_sensors.real_go2_mission_gateway as gateway

    source = inspect.getsource(gateway)
    assert "cmd_vel" not in source
    assert "NavigationCommand" not in source
    assert "NavigateToPose" not in source
    assert "ACTION_STOP" not in source


def test_canonical_latch_releases_only_matching_step3_episode_and_reset() -> None:
    result = MissionGatewayCore(
        FakeNormalizer(),
        expected_config_sha256=CONFIG_SHA,
        monotonic_ns=lambda: 1_000_000_000,
    ).process(envelope())
    latch = CanonicalMissionLatch(expected_config_sha256=CONFIG_SHA)
    mission = latch.update(result)
    assert mission.source_language == "zh"
    assert latch.instruction_for(mission.episode_id, 3).startswith("Go through")
    with pytest.raises(MissionGatewayError, match="episode/reset"):
        latch.instruction_for(mission.episode_id, 4)
    with pytest.raises(MissionGatewayError, match="episode/reset"):
        latch.instruction_for("real-another-mission", 3)


def test_canonical_latch_rejects_projection_or_config_tampering() -> None:
    result = MissionGatewayCore(
        FakeNormalizer(),
        expected_config_sha256=CONFIG_SHA,
        monotonic_ns=lambda: 1_000_000_000,
    ).process(envelope())
    latch = CanonicalMissionLatch(expected_config_sha256=CONFIG_SHA)
    tampered = dict(result)
    tampered["internvla"] = dict(result["internvla"])
    tampered["internvla"]["raw_instruction_allowed"] = True
    with pytest.raises(MissionGatewayError, match="unsafe raw path"):
        latch.update(tampered)

    wrong_sha = dict(result)
    wrong_sha["mission"] = dict(result["mission"])
    wrong_sha["mission"]["config_sha256"] = "c" * 64
    with pytest.raises(MissionGatewayError, match="config SHA"):
        latch.update(wrong_sha)


def test_strict_real_client_uses_latched_canonical_instruction_only() -> None:
    source = (
        ROOT
        / "internvla_t4_sensors"
        / "internvla_t4_sensors"
        / "client_node.py"
    ).read_text(encoding="utf-8")
    assert "INTERNVLA_STRICT_REAL_MISSION_REQUIRED" in source
    assert "strict real mission mode requires use_sim_time=false" in source
    assert "strict real mission mode forbids evaluator/GT pose publication" in source
    assert 'kwargs["instruction"] = canonical' in source
    assert 'kwargs["instruction_tokens"] = []' in source


def test_strict_real_ingress_config_and_launcher_are_fail_closed() -> None:
    import json

    config = json.loads(
        (ROOT / "configs/internnav_t5/strict_real_go2_mission_ingress.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["use_sim_time"] is False
    assert config["completion_sim_inheritance_allowed"] is False
    assert config["natural_language_route"]["required_normalizer"] == "Step3-VL-10B"
    assert config["natural_language_route"]["internvla_raw_instruction_allowed"] is False
    assert config["control_boundary"]["human_arm"] == "DISABLED"
    assert config["control_boundary"]["estop"] == "LATCHED"
    assert config["control_boundary"]["nonzero_motion_test_allowed"] is False
    launcher = (ROOT / "scripts/run_t5_strict_real_mission_gateway.sh").read_text(
        encoding="utf-8"
    )
    assert "completion_sim/Isaac overlays" in launcher
    assert "cmd_vel" not in launcher
    assert "navigate_to_pose" not in launcher
    unit = (
        ROOT / "deploy/systemd/internvla-real-mission-gateway.service"
    ).read_text(encoding="utf-8")
    assert "run_t5_strict_real_mission_gateway.sh" in unit
    assert "EnvironmentFile=%h/.config/internnav/strict-real-go2.env" in unit
