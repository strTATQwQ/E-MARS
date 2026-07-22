from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .base import PlannerDecision, PlannerMetrics, PlannerRunner, SlowPlannerRequest
from .hf_base import GenerationThreadTimeout


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
        from .step3_vl_10b import Step3VLSlowPlanner

        return Step3VLSlowPlanner.from_pretrained(
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
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(str(config.get("bind", "tcp://0.0.0.0:8200")))
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
            }
            if not all(required.values()):
                raise ValueError(f"Step3 deadline/redaction contract mismatch: {required}")
        self.runtime_health = {
            "service_config_sha256": str(config.get("_config_sha256", "")),
            "revision": str(config.get("revision", "")),
            "max_retries": int(config.get("max_retries", 1)),
            "max_request_age_s": self.max_request_age_s,
            "batch_size": int(config.get("batch_size", 1)),
            "redact_raw_text": self.redact_raw_text,
        }

    def _log(self, value: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _decision_mapping(self, decision: PlannerDecision) -> dict[str, Any]:
        value = decision.to_mapping()
        if self.redact_raw_text:
            value.pop("raw_text", None)
        return value

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
            started = time.perf_counter()
            try:
                envelope = json.loads(parts[0].decode("utf-8"))
                if envelope.get("type") == "health":
                    self.socket.send_json({"ok": True, **self.planner.health(), **self.runtime_health})
                    continue
                if envelope.get("type") != "decide":
                    raise ValueError("unknown request type")
                request = SlowPlannerRequest.from_wire(envelope.get("request") or {}, parts[1:])
                age_s = time.time() - request.timestamp
                if abs(age_s) > self.max_request_age_s:
                    reason = "stale_request" if age_s > 0 else "clock_skew"
                    decision = self._stale_decision(request, reason)
                    metrics = PlannerMetrics(model_variant=self.planner.model_variant, precision_mode=self.planner.precision_mode)
                else:
                    decision, metrics = self.runner.decide(request)
                server_ms = (time.perf_counter() - started) * 1000.0
                metrics = replace(metrics, end_to_end_ms=max(metrics.end_to_end_ms, server_ms))
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
