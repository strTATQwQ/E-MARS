"""OpenAI-compatible Step-3.7-Flash mission normalizer.

The adapter is deliberately normalization-only. It never receives camera
frames, candidate frontiers, or motion authority, and it reads its API key
from an environment variable rather than checked-in configuration.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping

from .base import PlannerMetrics, SlowPlannerProtocolError
from .mission import (
    CanonicalMission,
    MissionNormalizationRequest,
    normalization_prompt,
    parse_canonical_mission,
)


class Step37FlashMissionNormalizer:
    model_variant = "step_3_7_flash_normalizer"
    precision_mode = "remote_api"

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "step-3.7-flash",
        api_key_env: str = "STEPFUN_API_KEY",
        timeout_s: float = 12.0,
        max_tokens: int = 256,
    ) -> None:
        normalized_url = str(base_url).rstrip("/")
        parsed = urllib.parse.urlparse(normalized_url)
        local_http = parsed.scheme == "http" and parsed.hostname in {
            "127.0.0.1",
            "localhost",
            "::1",
        }
        if parsed.scheme != "https" and not local_http:
            raise ValueError("Step-3.7-Flash base_url must use HTTPS or loopback HTTP")
        if not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("Step-3.7-Flash base_url is invalid")
        if model != "step-3.7-flash":
            raise ValueError("the simulation normalizer supports model=step-3.7-flash only")
        if not api_key_env or not api_key_env.replace("_", "A").isalnum():
            raise ValueError("api_key_env must be an environment variable name")
        if not 1.0 <= float(timeout_s) <= 30.0:
            raise ValueError("Step-3.7-Flash timeout_s must be in [1,30]")
        if not 64 <= int(max_tokens) <= 512:
            raise ValueError("Step-3.7-Flash max_tokens must be in [64,512]")
        self.base_url = normalized_url
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Step37FlashMissionNormalizer":
        if not bool(config.get("normalization_only", False)):
            raise ValueError("Step-3.7-Flash must be configured as normalization_only")
        if int(config.get("max_retries", 0)) != 0:
            raise ValueError("Step-3.7-Flash normalization forbids hidden retries")
        if not bool(config.get("redact_raw_text", True)):
            raise ValueError("Step-3.7-Flash normalization requires raw-text redaction")
        return cls(
            base_url=str(config.get("api_base_url", "https://api.stepfun.com/v1")),
            model=str(config.get("api_model", "step-3.7-flash")),
            api_key_env=str(config.get("api_key_env", "STEPFUN_API_KEY")),
            timeout_s=float(config.get("generation_wall_budget_s", 12.0)),
            max_tokens=int(config.get("max_new_tokens", 256)),
        )

    def health(self) -> dict[str, Any]:
        return {
            "ready": bool(os.environ.get(self.api_key_env)),
            "model_variant": self.model_variant,
            "precision_mode": self.precision_mode,
            "protocol_version": 1,
            "api_model": self.model,
            "api_key_env": self.api_key_env,
            "normalization_only": True,
        }

    def generate_raw(self, *_args: Any, **_kwargs: Any) -> tuple[str, PlannerMetrics]:
        raise SlowPlannerProtocolError(
            "Step-3.7-Flash is normalization-only and cannot make navigation decisions"
        )

    def normalize_instruction(
        self, request: MissionNormalizationRequest
    ) -> tuple[CanonicalMission, PlannerMetrics]:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"missing Step API credential in {self.api_key_env}")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Normalize robot-navigation instructions. Return only the "
                        "requested JSON object. Do not expose reasoning."
                    ),
                },
                {"role": "user", "content": normalization_prompt(request)},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        http_request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=wire,
            method="POST",
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                response_wire = response.read(1_048_577)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Step-3.7-Flash HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Step-3.7-Flash transport failed") from exc
        if len(response_wire) > 1_048_576:
            raise RuntimeError("Step-3.7-Flash response exceeds size bound")
        try:
            response_value = json.loads(response_wire.decode("utf-8"))
            choices = response_value["choices"]
            content = choices[0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("content is not text")
            usage = response_value.get("usage") or {}
        except (KeyError, IndexError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Step-3.7-Flash returned an invalid response envelope") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        mission = parse_canonical_mission(request, content)
        metrics = PlannerMetrics(
            input_token_count=int(usage.get("prompt_tokens") or 0),
            output_token_count=int(usage.get("completion_tokens") or 0),
            model_generate_ms=elapsed_ms,
            end_to_end_ms=elapsed_ms,
            model_variant=self.model_variant,
            precision_mode=self.precision_mode,
        )
        return mission, metrics

    def close(self) -> None:
        return None
