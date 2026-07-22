"""Runtime switch for optional completion-simulation instruction normalization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from .client import SlowPlannerClient
from .mission import MissionNormalizationRequest, instruction_sha256, new_request


PROVIDERS = frozenset({"step3_vl_10b", "step_3_7_flash"})
FAILURE_POLICIES = frozenset({"passthrough", "reject"})
EXPECTED_VARIANTS = {
    "step3_vl_10b": "step3_vl_10b_bf16",
    "step_3_7_flash": "step_3_7_flash_normalizer",
}


@dataclass(frozen=True)
class SimulationNormalizationConfig:
    enabled: bool
    provider: str
    endpoint: str
    timeout_ms: int = 12_000
    failure_policy: str = "passthrough"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SimulationNormalizationConfig":
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("mission normalization enabled must be boolean")
        provider = str(value.get("provider") or "step3_vl_10b")
        if provider not in PROVIDERS:
            raise ValueError(f"unsupported mission normalization provider={provider!r}")
        endpoint = str(value.get("endpoint") or "tcp://127.0.0.1:8210")
        if not endpoint.startswith("tcp://"):
            raise ValueError("mission normalizer endpoint must use tcp://")
        timeout_ms = int(value.get("timeout_ms", 12_000))
        if not 1_000 <= timeout_ms <= 30_000:
            raise ValueError("mission normalizer timeout_ms must be in [1000,30000]")
        failure_policy = str(value.get("failure_policy") or "passthrough")
        if failure_policy not in FAILURE_POLICIES:
            raise ValueError("mission normalizer failure_policy is invalid")
        return cls(
            enabled=enabled,
            provider=provider,
            endpoint=endpoint,
            timeout_ms=timeout_ms,
            failure_policy=failure_policy,
        )


@dataclass(frozen=True)
class SimulationInstructionResolution:
    instruction: str
    normalized: bool
    provider: str
    source_instruction_sha256: str
    fallback_reason: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "normalized": self.normalized,
            "requires_retokenization": self.normalized,
            "provider": self.provider,
            "source_instruction_sha256": self.source_instruction_sha256,
            "fallback_reason": self.fallback_reason,
        }


class SimulationInstructionNormalizer:
    """Normalize once at mission ingress, or preserve the frozen sim baseline."""

    def __init__(
        self,
        config: SimulationNormalizationConfig,
        *,
        config_sha256: str,
        client_factory: Any = SlowPlannerClient,
    ) -> None:
        if len(config_sha256) != 64 or any(c not in "0123456789abcdef" for c in config_sha256):
            raise ValueError("config_sha256 must be a lowercase SHA-256 digest")
        self.config = config
        self.config_sha256 = config_sha256
        self.client_factory = client_factory

    def normalize(
        self,
        instruction: str,
        *,
        mission_id: str,
        episode_id: str,
        reset_generation: int,
        sequence_id: int,
    ) -> SimulationInstructionResolution:
        source = " ".join(str(instruction).split())
        source_sha = instruction_sha256(source)
        if not self.config.enabled:
            return SimulationInstructionResolution(
                instruction=source,
                normalized=False,
                provider="disabled",
                source_instruction_sha256=source_sha,
            )
        request: MissionNormalizationRequest = new_request(
            mission_id=mission_id,
            episode_id=episode_id,
            reset_generation=reset_generation,
            sequence_id=sequence_id,
            instruction=source,
            config_sha256=self.config_sha256,
        )
        try:
            with self.client_factory(
                self.config.endpoint, timeout_ms=self.config.timeout_ms
            ) as client:
                health = client.health()
                expected = EXPECTED_VARIANTS[self.config.provider]
                if health.get("ready") is not True or health.get("model_variant") != expected:
                    raise RuntimeError("mission normalizer health/provider mismatch")
                mission, _metrics = client.normalize_instruction(request)
            if mission.abstain:
                raise RuntimeError("mission normalizer abstained")
            return SimulationInstructionResolution(
                instruction=mission.canonical_instruction,
                normalized=True,
                provider=self.config.provider,
                source_instruction_sha256=source_sha,
            )
        except Exception as exc:
            if self.config.failure_policy == "reject":
                raise
            return SimulationInstructionResolution(
                instruction=source,
                normalized=False,
                provider=self.config.provider,
                source_instruction_sha256=source_sha,
                fallback_reason=type(exc).__name__,
            )


def config_sha256(raw_config: bytes) -> str:
    return hashlib.sha256(raw_config).hexdigest()
