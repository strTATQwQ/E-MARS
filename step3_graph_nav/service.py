from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .arrival_verifier import ArrivalRequest, ArrivalVerifier, parse_arrival
from .io import append_jsonl, load_yaml
from .model import Step3SplitModel
from .navigation_policy import NavigationPolicy, parse_navigation_action
from .protocol import GraphNavProtocolError, GraphNavRequest, InvalidCandidateId


def _expand(value: Any) -> str:
    return os.path.expanduser(os.path.expandvars(str(value)))


def _validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    required = {
        "precision": "bf16",
        "temperature": 0.0,
        "do_sample": False,
        "batch_size": 1,
        "pacore": False,
        "retries": 0,
        "fallback": "none",
        "reasoning_mode": "bounded_private_then_schema_score",
        "split_policy": True,
    }
    mismatches = {
        key: {"required": value, "actual": model.get(key)}
        for key, value in required.items()
        if model.get(key) != value
    }
    if mismatches:
        raise ValueError(f"split-policy generation contract mismatch: {mismatches}")
    if "action_stop_logit_bias" in model or "stop_logit_bias" in model:
        raise ValueError("global STOP logit bias is forbidden")
    reasoning_tokens = int(model["reasoning_tokens"])
    if not 16 <= reasoning_tokens <= 32:
        raise ValueError("reasoning_tokens must be within 16..32")


class Step3SplitPolicyServer:
    def __init__(self, model: Step3SplitModel, config: dict[str, Any], config_sha256: str) -> None:
        import zmq

        self.model = model
        self.navigation = NavigationPolicy(model)
        self.arrival = ArrivalVerifier(model)
        self.config = config
        service_cfg = config["service"]
        self.max_request_age_s = float(service_cfg.get("max_request_age_s", 120.0))
        self.log_path = Path(_expand(service_cfg["log_path"])).resolve()
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(str(service_cfg["bind"]))
        self.arrival_threshold = float(config["arrival"]["initial_threshold"])
        self.arrival_threshold_locked = False
        model_cfg = config["model"]
        self.runtime_health = {
            "ok": True,
            "ready": True,
            "service": "step3_split_policy",
            "split_protocol_version": 2,
            "service_config_sha256": config_sha256,
            "revision": str(model_cfg["revision"]),
            "temperature": float(model_cfg["temperature"]),
            "do_sample": bool(model_cfg["do_sample"]),
            "reasoning_tokens": int(model_cfg["reasoning_tokens"]),
            "batch_size": int(model_cfg["batch_size"]),
            "pacore": bool(model_cfg["pacore"]),
            "retries": int(model_cfg["retries"]),
            "fallback": str(model_cfg["fallback"]),
            "reasoning_mode": str(model_cfg["reasoning_mode"]),
            "navigation_schema": "move_only_candidate_id",
            "arrival_schema": "arrived_boolean_only",
            "global_stop_logit_bias": False,
        }

    def health(self) -> dict[str, Any]:
        return {
            **self.model.health(),
            **self.runtime_health,
            "arrival_threshold": self.arrival_threshold,
            "arrival_threshold_locked": self.arrival_threshold_locked,
        }

    def _check_age(self, timestamp: float) -> None:
        age_s = time.time() - float(timestamp)
        if abs(age_s) > self.max_request_age_s:
            raise GraphNavProtocolError(f"request age {age_s:.3f}s exceeds limit")

    def _set_threshold(self, envelope: dict[str, Any]) -> dict[str, Any]:
        if self.arrival_threshold_locked:
            raise GraphNavProtocolError("arrival threshold is already frozen for validation/canary")
        value = envelope.get("threshold")
        if isinstance(value, bool):
            raise GraphNavProtocolError("arrival threshold must be numeric")
        threshold = float(value)
        if not math.isfinite(threshold):
            raise GraphNavProtocolError("arrival threshold must be finite")
        self.arrival_threshold = threshold
        self.arrival_threshold_locked = True
        append_jsonl(
            self.log_path,
            {
                "timestamp": time.time(),
                "type": "arrival_threshold_frozen",
                "threshold": threshold,
            },
        )
        return {"ok": True, "arrival_threshold": threshold, "locked": True}

    def run(self) -> None:
        while True:
            parts = self.socket.recv_multipart()
            started = time.perf_counter()
            try:
                envelope = json.loads(parts[0].decode("utf-8"))
                request_type = envelope.get("type")
                if request_type == "health":
                    self.socket.send_json(self.health())
                    continue
                if request_type == "set_arrival_threshold":
                    self.socket.send_json(self._set_threshold(envelope))
                    continue
                if request_type == "navigate":
                    request = GraphNavRequest.from_wire(envelope.get("request") or {}, parts[1:])
                    self._check_age(request.timestamp)
                    raw_text, metrics = self.navigation.decide(request)
                    action = None
                    parse_ok = False
                    candidate_id_valid = False
                    parse_error = ""
                    error_type = ""
                    try:
                        parsed = parse_navigation_action(raw_text, request.candidate_ids)
                        action = parsed.to_mapping()
                        parse_ok = True
                        candidate_id_valid = True
                    except (GraphNavProtocolError, InvalidCandidateId) as exc:
                        parse_error = str(exc)
                        error_type = type(exc).__name__
                        candidate_id_valid = not isinstance(exc, InvalidCandidateId)
                    metrics["server_total_ms"] = (time.perf_counter() - started) * 1000.0
                    response = {
                        "ok": True,
                        "parse_ok": parse_ok,
                        "candidate_id_valid": candidate_id_valid,
                        "action": action,
                        "raw_output": raw_text,
                        "parse_error": parse_error,
                        "error_type": error_type,
                        "metrics": metrics,
                    }
                    append_jsonl(
                        self.log_path,
                        {
                            "timestamp": time.time(),
                            "type": "navigate",
                            "request": request.metadata(),
                            "raw_output": raw_text,
                            "parse_ok": parse_ok,
                            "candidate_id_valid": candidate_id_valid,
                            "action": action,
                            "parse_error": parse_error,
                            "error_type": error_type,
                            "metrics": metrics,
                        },
                    )
                    self.socket.send_json(response)
                    continue
                if request_type == "verify_arrival":
                    request = ArrivalRequest.from_wire(envelope.get("request") or {}, parts[1:])
                    self._check_age(request.timestamp)
                    raw_text, metrics = self.arrival.verify(
                        request, threshold=self.arrival_threshold
                    )
                    parse_error = ""
                    parse_ok = False
                    arrived = None
                    try:
                        arrived = parse_arrival(raw_text)
                        parse_ok = True
                    except GraphNavProtocolError as exc:
                        parse_error = str(exc)
                    metrics["server_total_ms"] = (time.perf_counter() - started) * 1000.0
                    response = {
                        "ok": True,
                        "parse_ok": parse_ok,
                        "arrived": arrived,
                        "raw_output": raw_text,
                        "parse_error": parse_error,
                        "metrics": metrics,
                    }
                    append_jsonl(
                        self.log_path,
                        {
                            "timestamp": time.time(),
                            "type": "verify_arrival",
                            "request": request.metadata(),
                            "raw_output": raw_text,
                            "parse_ok": parse_ok,
                            "arrived": arrived,
                            "parse_error": parse_error,
                            "metrics": metrics,
                        },
                    )
                    self.socket.send_json(response)
                    continue
                raise GraphNavProtocolError("unknown split-policy request type")
            except Exception as exc:
                self.socket.send_json({"ok": False, "error": f"{type(exc).__name__}:{exc}"})

    def close(self) -> None:
        self.socket.close(linger=0)
        self.model.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve split Step3 navigation and arrival policies.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    _validate_config(config)
    model = Step3SplitModel.from_config(config)
    server = Step3SplitPolicyServer(
        model, config, hashlib.sha256(config_path.read_bytes()).hexdigest()
    )
    try:
        server.run()
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

