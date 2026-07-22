from __future__ import annotations

import hashlib
import json
import time

import pytest

from slow_planner.base import PlannerMetrics, SlowPlannerProtocolError
from slow_planner.client import SlowPlannerClient
from slow_planner.mission import (
    MISSION_NORMALIZATION_TYPE,
    MISSION_ROUTE,
    CanonicalMission,
    MissionNormalizationRequest,
    contains_complete_normalization_json,
    normalization_prompt,
    parse_canonical_mission,
)


CONFIG_SHA = "a" * 64


def request(instruction: str = "穿过门口，在红色椅子旁边停下") -> MissionNormalizationRequest:
    return MissionNormalizationRequest(
        mission_id="mission-1",
        episode_id="real-mission-1",
        reset_generation=2,
        sequence_id=0,
        instruction=instruction,
        config_sha256=CONFIG_SHA,
        timestamp=time.time(),
    )


def normalized_json(**changes: object) -> str:
    value: dict[str, object] = {
        "source_language": "zh",
        "canonical_instruction": "Go through the doorway and stop beside the red chair.",
        "target_description": "the red chair beyond the doorway",
        "constraints": ["Stop beside the chair"],
        "confidence": 0.92,
        "abstain": False,
    }
    value.update(changes)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def test_chinese_instruction_is_only_released_as_canonical_english() -> None:
    raw = request()
    mission = parse_canonical_mission(raw, normalized_json())
    assert mission.source_language == "zh"
    assert mission.canonical_instruction.startswith("Go through")
    assert "穿" not in mission.canonical_instruction
    assert mission.source_instruction_sha256 == hashlib.sha256(
        raw.instruction.encode("utf-8")
    ).hexdigest()
    assert mission.to_mapping()["internvla_raw_instruction_allowed"] is False
    assert raw.route == MISSION_ROUTE


def test_normalizer_rejects_prose_duplicate_keys_and_non_ascii_release() -> None:
    raw = request()
    with pytest.raises(SlowPlannerProtocolError):
        parse_canonical_mission(raw, "Reasoning: " + normalized_json())
    with pytest.raises(SlowPlannerProtocolError):
        parse_canonical_mission(
            raw,
            normalized_json(
                canonical_instruction="前往红色椅子",
                target_description="红色椅子",
            ),
        )
    duplicate = normalized_json()[:-1] + ',"abstain":false}'
    with pytest.raises(SlowPlannerProtocolError):
        parse_canonical_mission(raw, duplicate)


def test_abstain_cannot_release_any_downstream_instruction() -> None:
    raw = request("随便走走")
    mission = parse_canonical_mission(
        raw,
        normalized_json(
            canonical_instruction="",
            target_description="",
            constraints=[],
            confidence=0.0,
            abstain=True,
        ),
    )
    assert mission.abstain is True
    assert mission.canonical_instruction == ""
    with pytest.raises(SlowPlannerProtocolError):
        parse_canonical_mission(
            raw,
            normalized_json(abstain=True),
        )


def test_prompt_contains_chinese_but_forbids_reasoning_and_invention() -> None:
    prompt = normalization_prompt(request())
    assert "穿过门口" in prompt
    assert "No reasoning or prose" in prompt
    assert "Do not add objects or goals" in prompt
    assert contains_complete_normalization_json(normalized_json()) is True


class FakeSocket:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.sent: dict[str, object] | None = None

    def send_json(self, value: dict[str, object]) -> None:
        self.sent = value

    def recv_json(self) -> dict[str, object]:
        return self.response


def test_client_checks_step3_identity_before_releasing_canonical_text() -> None:
    raw = request()
    mission = parse_canonical_mission(raw, normalized_json())
    metrics = PlannerMetrics(model_variant="step3_vl_10b_bf16")
    client = SlowPlannerClient.__new__(SlowPlannerClient)
    client.socket = FakeSocket(
        {
            "ok": True,
            "normalization": mission.to_mapping(),
            "metrics": metrics.to_mapping(),
            "server_total_ms": 1.0,
        }
    )
    result, _ = client.normalize_instruction(raw)
    assert result == mission
    assert client.socket.sent is not None
    assert client.socket.sent["type"] == MISSION_NORMALIZATION_TYPE

    mismatched = CanonicalMission(
        **{
            **mission.__dict__,
            "sequence_id": mission.sequence_id + 1,
        }
    )
    client.socket = FakeSocket(
        {
            "ok": True,
            "normalization": mismatched.to_mapping(),
            "metrics": metrics.to_mapping(),
            "server_total_ms": 1.0,
        }
    )
    with pytest.raises(SlowPlannerProtocolError, match="mismatched"):
        client.normalize_instruction(raw)
