from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any

from .base import (
    PlannerDecision,
    PlannerMetrics,
    SlowPlannerProtocolError,
    SlowPlannerRequest,
    StructuredPlannerDecision,
)


class SlowPlannerClient:
    def __init__(self, endpoint: str, *, timeout_ms: int = 180_000) -> None:
        import zmq

        self.endpoint = endpoint
        self.timeout_ms = int(timeout_ms)
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(endpoint)

    def health(self) -> dict[str, Any]:
        self.socket.send_json({"type": "health"})
        return dict(self.socket.recv_json())

    def decide(self, request: SlowPlannerRequest) -> tuple[PlannerDecision, PlannerMetrics]:
        metadata = {"type": "decide", "request": request.metadata()}
        parts = [json.dumps(metadata, separators=(",", ":")).encode("utf-8")]
        parts.extend(image.jpeg for image in request.ordered_images)
        started = time.perf_counter()
        self.socket.send_multipart(parts)
        response = json.loads(self.socket.recv().decode("utf-8"))
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "slow planner request failed"))
        row = dict(response["decision"])
        target = row.get("target_relative_xz")
        decision_class = (
            StructuredPlannerDecision
            if "scene_summary" in row
            else PlannerDecision
        )
        structured = (
            {
                "scene_summary": str(row.get("scene_summary") or ""),
                "target_evidence": tuple(
                    str(value) for value in (row.get("target_evidence") or [])
                ),
                "blocked_directions": tuple(
                    str(value) for value in (row.get("blocked_directions") or [])
                ),
                "recommended_frontier": row.get("recommended_frontier"),
                "target_found": row.get("target_found"),
                "abstain": row.get("abstain"),
            }
            if decision_class is StructuredPlannerDecision
            else {}
        )
        decision = decision_class(
            episode_id=str(row["episode_id"]),
            snapshot_id=str(row["snapshot_id"]),
            decision=str(row["decision"]),
            frontier_id=row.get("frontier_id"),
            target_relative_xz=(float(target[0]), float(target[1])) if target is not None else None,
            confidence=float(row["confidence"]),
            raw_text=str(row.get("raw_text") or ""),
            parse_attempts=int(row.get("parse_attempts", 1)),
            fallback_used=bool(row.get("fallback_used", False)),
            fallback_reason=str(row.get("fallback_reason") or ""),
            protocol_version=int(row.get("protocol_version", 1)),
            **structured,
        )
        if decision.episode_id != request.episode_id or decision.snapshot_id != request.snapshot_id:
            raise SlowPlannerProtocolError("stale or mismatched slow-planner response")
        metrics_row = dict(response["metrics"])
        metrics_row["raw_image_resolutions"] = tuple(
            (int(value[0]), int(value[1])) for value in metrics_row.get("raw_image_resolutions", [])
        )
        metrics = PlannerMetrics(**metrics_row)
        server_ms = float(response.get("server_total_ms") or metrics.end_to_end_ms)
        return decision, replace(metrics, network_ms=max(0.0, roundtrip_ms - server_ms))

    def close(self) -> None:
        self.socket.close(linger=0)

    def __enter__(self) -> "SlowPlannerClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
