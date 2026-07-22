from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from slow_planner.base import PlannerMetrics
from slow_planner.mission import MissionNormalizationRequest, parse_canonical_mission
from slow_planner.sim_mission import (
    SimulationInstructionNormalizer,
    SimulationNormalizationConfig,
)
from slow_planner.step37_flash import Step37FlashMissionNormalizer
from slow_planner.serve import _build_planner


CONFIG_SHA = "b" * 64
ROOT = Path(__file__).resolve().parents[1]


def _request() -> MissionNormalizationRequest:
    return MissionNormalizationRequest(
        mission_id="mission-1",
        episode_id="episode-1",
        reset_generation=0,
        sequence_id=0,
        instruction="穿过门口，在红色椅子旁停下",
        config_sha256=CONFIG_SHA,
        timestamp=time.time(),
    )


def _canonical_json() -> str:
    return json.dumps(
        {
            "source_language": "zh",
            "canonical_instruction": "Go through the doorway and stop beside the red chair.",
            "target_description": "the red chair beyond the doorway",
            "constraints": ["Stop beside the chair"],
            "confidence": 0.94,
            "abstain": False,
        },
        separators=(",", ":"),
    )


class _HTTPResponse:
    def __init__(self, value: dict[str, object]) -> None:
        self.wire = json.dumps(value).encode("utf-8")

    def __enter__(self) -> "_HTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self.wire


def test_step37_flash_uses_openai_compatible_json_mode(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _HTTPResponse(
            {
                "choices": [{"message": {"content": _canonical_json()}}],
                "usage": {"prompt_tokens": 23, "completion_tokens": 41},
            }
        )

    monkeypatch.setenv("STEPFUN_API_KEY", "test-only-key")
    monkeypatch.setattr("slow_planner.step37_flash.urllib.request.urlopen", fake_urlopen)
    planner = Step37FlashMissionNormalizer(
        base_url="https://api.stepfun.com/v1", timeout_s=12.0
    )
    mission, metrics = planner.normalize_instruction(_request())
    payload = json.loads(captured["request"].data.decode("utf-8"))
    assert payload["model"] == "step-3.7-flash"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0.0
    assert mission.canonical_instruction.startswith("Go through")
    assert metrics.input_token_count == 23
    assert metrics.output_token_count == 41


def test_step37_flash_requires_environment_credential(monkeypatch) -> None:
    monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
    planner = Step37FlashMissionNormalizer(base_url="https://api.stepfun.com/v1")
    assert planner.health()["ready"] is False
    with pytest.raises(RuntimeError, match="missing Step API credential"):
        planner.normalize_instruction(_request())


class _FakeClient:
    mission = parse_canonical_mission(_request(), _canonical_json())

    def __init__(self, endpoint: str, *, timeout_ms: int) -> None:
        assert endpoint == "tcp://127.0.0.1:8210"
        assert timeout_ms == 12000

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def health(self):
        return {"ready": True, "model_variant": "step_3_7_flash_normalizer"}

    def normalize_instruction(self, _request):
        return self.mission, PlannerMetrics(model_variant="step_3_7_flash_normalizer")


def test_sim_layer_is_disabled_by_default_and_can_select_flash() -> None:
    disabled = SimulationInstructionNormalizer(
        SimulationNormalizationConfig.from_mapping({}),
        config_sha256=CONFIG_SHA,
        client_factory=lambda *_args, **_kwargs: pytest.fail("client must stay disabled"),
    )
    passthrough = disabled.normalize(
        " Move forward ",
        mission_id="mission-1",
        episode_id="episode-1",
        reset_generation=0,
        sequence_id=0,
    )
    assert passthrough.instruction == "Move forward"
    assert passthrough.normalized is False
    enabled = SimulationInstructionNormalizer(
        SimulationNormalizationConfig.from_mapping(
            {
                "enabled": True,
                "provider": "step_3_7_flash",
                "endpoint": "tcp://127.0.0.1:8210",
            }
        ),
        config_sha256=CONFIG_SHA,
        client_factory=_FakeClient,
    )
    result = enabled.normalize(
        _request().instruction,
        mission_id="mission-1",
        episode_id="episode-1",
        reset_generation=0,
        sequence_id=0,
    )
    assert result.normalized is True
    assert result.to_mapping()["requires_retokenization"] is True
    assert result.provider == "step_3_7_flash"
    assert result.instruction == _FakeClient.mission.canonical_instruction


def test_sim_passthrough_fallback_is_explicit() -> None:
    class FailingClient(_FakeClient):
        def health(self):
            raise TimeoutError("offline")

    layer = SimulationInstructionNormalizer(
        SimulationNormalizationConfig.from_mapping(
            {
                "enabled": True,
                "provider": "step_3_7_flash",
                "endpoint": "tcp://127.0.0.1:8210",
                "failure_policy": "passthrough",
            }
        ),
        config_sha256=CONFIG_SHA,
        client_factory=FailingClient,
    )
    result = layer.normalize(
        "keep the frozen instruction",
        mission_id="mission-1",
        episode_id="episode-1",
        reset_generation=0,
        sequence_id=0,
    )
    assert result.normalized is False
    assert result.instruction == "keep the frozen instruction"
    assert result.fallback_reason == "TimeoutError"


def test_checked_in_profiles_are_mutually_exclusive_and_flash_builds() -> None:
    profile_dir = ROOT / "configs" / "completion_sim"
    expected = {
        "mission_normalization_disabled.yaml": (False, "step3_vl_10b"),
        "mission_normalization_step3_vl.yaml": (True, "step3_vl_10b"),
        "mission_normalization_step37_flash.yaml": (True, "step_3_7_flash"),
    }
    for name, pair in expected.items():
        value = yaml.safe_load((profile_dir / name).read_text(encoding="utf-8"))
        config = SimulationNormalizationConfig.from_mapping(
            value["mission_normalization"]
        )
        assert (config.enabled, config.provider) == pair
    flash_config = yaml.safe_load(
        (ROOT / "configs" / "slow_models" / "step_3_7_flash_normalizer.yaml").read_text(
            encoding="utf-8"
        )
    )
    planner = _build_planner(flash_config)
    assert planner.model_variant == "step_3_7_flash_normalizer"
    assert planner.health()["normalization_only"] is True
