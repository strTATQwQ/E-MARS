from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .base import PlannerDecision, PlannerMetrics, PlannerRunner, SlowPlannerRequest
from .hf_base import GenerationThreadTimeout
from .mission import (
    MISSION_NORMALIZATION_TYPE,
    MissionNormalizationRequest,
    normalization_response,
)


_PRIVATE_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:hf_|sk-)[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)\b(?:OPENAI_API_KEY|HF_TOKEN)\s*[:=]\s*\S+"),
)


def _private_trace_text(value: str) -> tuple[str, int]:
    """Redact credential-shaped text without inspecting process state."""

    redacted = str(value)
    count = 0
    for pattern in _PRIVATE_SECRET_PATTERNS:
        redacted, replacements = pattern.subn("[REDACTED_CREDENTIAL]", redacted)
        count += replacements
    return redacted, count


def _explicit_model_reasoning(raw_text: str) -> str | None:
    """Return only rationale that the model explicitly emitted, if present."""

    match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    if match is not None and match.group(1).strip():
        return match.group(1).strip()
    try:
        value = json.loads(raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(value, dict) and isinstance(value.get("reasoning"), str):
        reasoning = value["reasoning"].strip()
        return reasoning or None
    return None


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("slow planner config must be an object")
    return value


def _expand(value: Any) -> str:
    return os.path.expanduser(os.path.expandvars(str(value)))


def _build_planner(config: dict[str, Any]):
    variant = str(config["model_variant"])
    common = {
        "device": str(config.get("device", "cuda")),
        "max_new_tokens": int(config.get("max_new_tokens", 48)),
        "processor_use_fast": bool(config.get("processor_use_fast", False)),
    }
    if variant == "qwen25_baseline_bf16":
        from .qwen25_baseline import Qwen25BaselineSlowPlanner

        return Qwen25BaselineSlowPlanner.from_pretrained(
            repo=_expand(config["repo"]),
            model_path=_expand(config["model_path"]),
            helper_path=_expand(config["helper_path"]),
            attn_implementation=str(config.get("attn_implementation", "flash_attention_2")),
            **common,
        )
    if variant == "cosmos_reason2_32b_bf16":
        from .cosmos_reason2_32b import CosmosReason2SlowPlanner

        return CosmosReason2SlowPlanner.from_pretrained(
            _expand(config["model_path"]),
            attn_implementation=str(config.get("attn_implementation", "sdpa")),
            **common,
        )
    if variant == "step3_vl_10b_bf16":
        planner_mode = str(config.get("planner_mode", "bounded_advisor_v1"))
        if planner_mode == "task_state_v1":
            from .step3_task_state import Step3TaskStateSlowPlanner

            planner_class = Step3TaskStateSlowPlanner
        elif planner_mode == "bounded_advisor_v1":
            from .step3_vl_10b import Step3VLSlowPlanner

            planner_class = Step3VLSlowPlanner
        else:
            raise ValueError(f"unsupported Step3 planner_mode={planner_mode!r}")

        return planner_class.from_pretrained(
            _expand(config["model_path"]),
            fix_mistral_regex=bool(config.get("fix_mistral_regex", True)),
            expected_transformers_version=str(config.get("transformers_version", "4.57.6")),
            checkpoint_key_mapping=str(
                config.get("checkpoint_key_mapping", "step3_flat_checkpoint_to_nested_v1")
            ),
            require_clean_checkpoint_load=bool(config.get("require_clean_checkpoint_load", True)),
            require_all_parameters_bf16=bool(config.get("require_all_parameters_bf16", True)),
            skip_private_reasoning=bool(config.get("skip_private_reasoning", True)),
            stop_on_complete_json=bool(config.get("stop_on_complete_json", True)),
            generation_wall_budget_s=float(config.get("generation_wall_budget_s", 10.5)),
            generation_join_grace_s=float(config.get("generation_join_grace_s", 0.5)),
            **common,
        )
    if variant == "step_3_7_flash_normalizer":
        from .step37_flash import Step37FlashMissionNormalizer

        return Step37FlashMissionNormalizer.from_config(config)
    raise ValueError(f"unsupported model_variant={variant!r}")


class SlowPlannerServer:
    def __init__(self, planner: Any, config: dict[str, Any]) -> None:
        import zmq

        self.planner = planner
        self.runner = PlannerRunner(planner, max_retries=int(config.get("max_retries", 1)))
        self.max_request_age_s = float(config.get("max_request_age_s", 5.0))
        self.redact_raw_text = bool(config.get("redact_raw_text", False))
        self.log_path = Path(_expand(config.get("log_path", "results/slow_planner_service.jsonl")))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        private_trace_text = os.environ.get("STEP3_PRIVATE_TRACE_PATH", "").strip()
        self.private_trace_path: Path | None = None
        if private_trace_text:
            private_trace_path = Path(private_trace_text).expanduser()
            if not private_trace_path.is_absolute():
                raise ValueError("STEP3_PRIVATE_TRACE_PATH must be absolute")
            private_trace_path.parent.mkdir(parents=True, exist_ok=False, mode=0o700)
            os.chmod(private_trace_path.parent, 0o700)
            self.private_trace_path = private_trace_path
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(_expand(config.get("bind", "tcp://0.0.0.0:8200")))
        configured_precision = str(config.get("precision_mode", planner.precision_mode))
        if configured_precision != planner.precision_mode:
            raise ValueError(
                f"configured precision_mode={configured_precision!r} differs from planner={planner.precision_mode!r}"
            )
        if not bool(config.get("deterministic_decoding", True)):
            raise ValueError("formal slow planner requires deterministic_decoding=true")
        if int(config.get("batch_size", 1)) != 1:
            raise ValueError("formal slow planner requires batch_size=1")
        if bool(config.get("pacore", False)):
            raise ValueError("formal slow planner forbids PaCoRe")
        if planner.model_variant == "step3_vl_10b_bf16":
            required = {
                "max_new_tokens": planner.max_new_tokens == 96,
                "max_retries": int(config.get("max_retries", 1)) == 0,
                "skip_private_reasoning": planner.skip_private_reasoning is True,
                "complete_json_stop": planner.stop_on_complete_json is True,
                "generation_wall_budget": planner.generation_wall_budget_s == 10.5,
                "generation_join_grace": planner.generation_join_grace_s == 0.5,
                "redact_raw_text": self.redact_raw_text is True,
                "planner_mode": str(
                    config.get("planner_mode", "bounded_advisor_v1")
                )
                == str(
                    getattr(planner, "planner_mode", "bounded_advisor_v1")
                ),
            }
            if not all(required.values()):
                raise ValueError(f"Step3 deadline/redaction contract mismatch: {required}")
        if planner.model_variant == "step_3_7_flash_normalizer":
            required = {
                "normalization_only": bool(config.get("normalization_only", False)),
                "max_retries": int(config.get("max_retries", 0)) == 0,
                "redact_raw_text": self.redact_raw_text is True,
            }
            if not all(required.values()):
                raise ValueError(
                    f"Step-3.7-Flash normalization contract mismatch: {required}"
                )
        self.runtime_health = {
            "service_config_sha256": str(config.get("_config_sha256", "")),
            "revision": str(config.get("revision", "")),
            "max_retries": int(config.get("max_retries", 1)),
            "max_request_age_s": self.max_request_age_s,
            "batch_size": int(config.get("batch_size", 1)),
            "redact_raw_text": self.redact_raw_text,
            "normalization_only": bool(config.get("normalization_only", False)),
            "private_trace_enabled": self.private_trace_path is not None,
            "private_reasoning_capture": "model_emitted_only",
        }

    def _log(self, value: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _decision_mapping(self, decision: PlannerDecision) -> dict[str, Any]:
        value = decision.to_mapping()
        if self.redact_raw_text:
            value.pop("raw_text", None)
        return value

    def _write_private_trace(
        self,
        *,
        request: SlowPlannerRequest,
        prompt: str | None,
        decision: PlannerDecision,
        metrics: PlannerMetrics,
        received_unix_ns: int,
        completed_unix_ns: int,
        received_monotonic_ns: int,
        completed_monotonic_ns: int,
    ) -> None:
        path = getattr(self, "private_trace_path", None)
        if path is None:
            return
        request_metadata = request.metadata()
        image_metadata = []
        for image, metadata in zip(request.ordered_images, request_metadata["ordered_images"]):
            image_metadata.append(
                {
                    **metadata,
                    "sha256": hashlib.sha256(image.jpeg).hexdigest(),
                    "bytes": len(image.jpeg),
                }
            )
        request_metadata["ordered_images"] = image_metadata
        raw_text, raw_redactions = _private_trace_text(decision.raw_text)
        sanitized_prompt, prompt_redactions = _private_trace_text(prompt or "")
        reasoning = _explicit_model_reasoning(raw_text)
        if reasoning is not None:
            reasoning, reasoning_redactions = _private_trace_text(reasoning)
        else:
            reasoning_redactions = 0
        parsed = decision.to_mapping()
        parsed.pop("raw_text", None)
        payload = {
            "schema_version": 1,
            "classification": "PRIVATE_MODEL_TRACE",
            "episode_id": request.episode_id,
            "snapshot_id": request.snapshot_id,
            "request": request_metadata,
            "step3_user_prompt": sanitized_prompt,
            "step3_raw_response": raw_text,
            "parsed_decision": parsed,
            "metrics": metrics.to_mapping(),
            "reasoning_present": reasoning is not None,
            "reasoning": reasoning,
            "reasoning_capture_status": (
                "model_emitted_reasoning_captured"
                if reasoning is not None
                else "not_available_by_design_skip_private_reasoning"
            ),
            "credential_redaction_count": (
                prompt_redactions + raw_redactions + reasoning_redactions
            ),
            "received_wall_time_unix_ns": received_unix_ns,
            "completed_wall_time_unix_ns": completed_unix_ns,
            "received_wall_monotonic_ns": received_monotonic_ns,
            "completed_wall_monotonic_ns": completed_monotonic_ns,
            "wall_response_duration_ms": (
                completed_monotonic_ns - received_monotonic_ns
            ) / 1_000_000.0,
        }
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(path, 0o600)

    @staticmethod
    def _stale_decision(request: SlowPlannerRequest, reason: str) -> PlannerDecision:
        return PlannerDecision(
            episode_id=request.episode_id,
            snapshot_id=request.snapshot_id,
            decision="abstain",
            frontier_id=None,
            target_relative_xz=None,
            confidence=0.0,
            fallback_used=True,
            fallback_reason=reason,
        )

    def run(self) -> None:
        while True:
            parts = self.socket.recv_multipart()
            received_unix_ns = time.time_ns()
            received_monotonic_ns = time.monotonic_ns()
            started = time.perf_counter()
            try:
                envelope = json.loads(parts[0].decode("utf-8"))
                if envelope.get("type") == "health":
                    self.socket.send_json({"ok": True, **self.planner.health(), **self.runtime_health})
                    continue
                if envelope.get("type") == MISSION_NORMALIZATION_TYPE:
                    if len(parts) != 1:
                        raise ValueError("mission normalization is text-only")
                    request = MissionNormalizationRequest.from_wire(
                        envelope.get("request") or {}
                    )
                    age_s = time.time() - request.timestamp
                    if abs(age_s) > self.max_request_age_s:
                        raise ValueError(
                            "stale mission request"
                            if age_s > 0
                            else "mission request clock skew"
                        )
                    normalizer = getattr(self.planner, "normalize_instruction", None)
                    if normalizer is None:
                        raise ValueError(
                            "loaded planner does not support Step3 mission normalization"
                        )
                    mission, metrics = normalizer(request)
                    server_ms = (time.perf_counter() - started) * 1000.0
                    metrics = replace(
                        metrics,
                        end_to_end_ms=max(metrics.end_to_end_ms, server_ms),
                    )
                    response = normalization_response(
                        mission, metrics, server_total_ms=server_ms
                    )
                    request_audit = request.metadata()
                    request_audit.pop("instruction", None)
                    self._log(
                        {
                            "timestamp": time.time(),
                            "request_type": MISSION_NORMALIZATION_TYPE,
                            "request": request_audit,
                            "normalization": mission.to_mapping(),
                            "metrics": metrics.to_mapping(),
                            "server_total_ms": server_ms,
                        }
                    )
                    self.socket.send(
                        json.dumps(
                            response,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    continue
                if envelope.get("type") != "decide":
                    raise ValueError("unknown request type")
                request = SlowPlannerRequest.from_wire(envelope.get("request") or {}, parts[1:])
                format_prompt = getattr(self.planner, "format_prompt", None)
                trace_prompt = None
                if callable(format_prompt):
                    try:
                        trace_prompt = str(format_prompt(request))
                    except Exception:
                        # Private evidence must never change the control result.
                        trace_prompt = None
                age_s = time.time() - request.timestamp
                if abs(age_s) > self.max_request_age_s:
                    reason = "stale_request" if age_s > 0 else "clock_skew"
                    decision = self._stale_decision(request, reason)
                    metrics = PlannerMetrics(model_variant=self.planner.model_variant, precision_mode=self.planner.precision_mode)
                else:
                    decision, metrics = self.runner.decide(request)
                server_ms = (time.perf_counter() - started) * 1000.0
                metrics = replace(metrics, end_to_end_ms=max(metrics.end_to_end_ms, server_ms))
                completed_unix_ns = time.time_ns()
                completed_monotonic_ns = time.monotonic_ns()
                self._write_private_trace(
                    request=request,
                    prompt=trace_prompt,
                    decision=decision,
                    metrics=metrics,
                    received_unix_ns=received_unix_ns,
                    completed_unix_ns=completed_unix_ns,
                    received_monotonic_ns=received_monotonic_ns,
                    completed_monotonic_ns=completed_monotonic_ns,
                )
                decision_mapping = self._decision_mapping(decision)
                response = {
                    "ok": True,
                    "decision": decision_mapping,
                    "metrics": metrics.to_mapping(),
                    "server_total_ms": server_ms,
                }
                self._log(
                    {
                        "timestamp": time.time(),
                        "request": request.metadata(),
                        "decision": decision_mapping,
                        "metrics": metrics.to_mapping(),
                        "server_total_ms": server_ms,
                    }
                )
                self.socket.send(json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            except GenerationThreadTimeout:
                raise
            except Exception as exc:
                self.socket.send_json({"ok": False, "error": f"{type(exc).__name__}:{exc}"})

    def close(self) -> None:
        self.socket.close(linger=0)
        self.planner.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve an independent OmniNav slow planner over ZeroMQ.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    config["_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    planner = _build_planner(config)
    server = SlowPlannerServer(planner, config)
    try:
        server.run()
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
