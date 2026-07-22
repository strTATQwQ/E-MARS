from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .perception_planning_suite import percentile
from .v12_step_value_utils import ROUTE_MODE, ROUTE_STOP_MODE, STEP_ONLY_MODE, STOP_MODE, load_jsonl


STEP_MODES = (ROUTE_MODE, STOP_MODE, ROUTE_STOP_MODE, STEP_ONLY_MODE)


def _details(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("details")
    return value if isinstance(value, dict) else event


def multimodal_http_evidence(run: Path) -> dict[str, Any]:
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    episode_modes = {
        str(row.get("episode_id") or ""): str(row.get("mode") or "")
        for row in metrics.get("episodes", [])
    }
    rows: list[dict[str, Any]] = []
    for event in load_jsonl(run / "events.jsonl"):
        details = _details(event)
        if str(details.get("event_type") or event.get("event") or "") != "step_http_response":
            continue
        episode_id = str(event.get("episode_id") or details.get("episode_id") or "")
        mode = episode_modes.get(episode_id, "")
        if mode not in STEP_MODES:
            continue
        snapshot = details.get("image_snapshot") if isinstance(details.get("image_snapshot"), dict) else {}
        try:
            age_sec = float(snapshot.get("age_sec"))
        except (TypeError, ValueError):
            age_sec = None
        try:
            frame_seq = int(snapshot.get("frame_seq"))
        except (TypeError, ValueError):
            frame_seq = -1
        result = str(details.get("result") or "")
        model = str(details.get("model") or "")
        multimodal = bool(details.get("multimodal"))
        fresh = bool(multimodal and frame_seq >= 0 and age_sec is not None and age_sec <= 0.75)
        rows.append(
            {
                "episode_id": episode_id,
                "mode": mode,
                "request_id": str(details.get("request_id") or ""),
                "role": str(details.get("role") or ""),
                "result": result,
                "model": model,
                "multimodal": multimodal,
                "fresh_image": fresh,
                "frame_seq": frame_seq,
                "image_age_sec": age_sec,
                "latency_sec": details.get("latency_s"),
                "oracle_context_leakage": int(bool(details.get("oracle_context_leakage", False))),
            }
        )
    accepted = [row for row in rows if row["result"] == "accepted" and row["model"] == "step_http"]
    latencies = [float(row["latency_sec"]) for row in accepted if row["latency_sec"] is not None]
    return {
        "calls": rows,
        "accepted": len(accepted),
        "accepted_multimodal": sum(row["multimodal"] for row in accepted),
        "fresh_multimodal": sum(row["fresh_image"] for row in accepted),
        "oracle_context_leakage": sum(row["oracle_context_leakage"] for row in rows),
        "fallback_or_mock": sum("mock" in row["model"].lower() or "fallback" in row["result"].lower() for row in rows),
        "step_latency_p95_sec": percentile(latencies, 0.95),
    }


def evaluate_multimodal_value_run(run: Path, *, existing_gate: dict[str, Any]) -> dict[str, Any]:
    evidence = multimodal_http_evidence(run)
    failures: list[str] = []
    if not bool(existing_gate.get("pass")):
        failures.append("underlying paired value gate failed")
    if evidence["accepted"] <= 0:
        failures.append("no accepted real Step role calls")
    if evidence["accepted_multimodal"] != evidence["accepted"]:
        failures.append("not every accepted Step role call was multimodal")
    if evidence["fresh_multimodal"] != evidence["accepted"]:
        failures.append("not every accepted multimodal call used a fresh image")
    if evidence["oracle_context_leakage"] != 0:
        failures.append("oracle context leakage detected")
    if evidence["fallback_or_mock"] != 0:
        failures.append("fallback or mock Step evidence detected")
    latency = evidence["step_latency_p95_sec"]
    if latency is None or latency > 5.0:
        failures.append("Step request p95 latency exceeds 5 seconds or is missing")
    return {
        "schema_version": 1,
        "pass": not failures,
        "underlying_value_gate_pass": bool(existing_gate.get("pass")),
        "multimodal_evidence": evidence,
        "failures": failures,
        "qualification_evidence": False,
        "locomotion_fidelity": "ideal_kinematic",
        "sim2real_gate": "NOT READY FOR REAL ROBOT AUTONOMY",
        "value_claim": (
            "true multimodal OmniNav+Step value gate passed"
            if not failures
            else "true multimodal OmniNav+Step value remains unproven"
        ),
    }
