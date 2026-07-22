from __future__ import annotations

import json
import time
from typing import Any

from .arrival_verifier import ArrivalRequest
from .protocol import GraphNavRequest


class Step3GraphNavClient:
    def __init__(self, endpoint: str, *, timeout_ms: int = 60_000) -> None:
        import zmq

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self.socket.connect(endpoint)

    def health(self) -> dict[str, Any]:
        self.socket.send_json({"type": "health"})
        return dict(self.socket.recv_json())

    def _request(self, parts: list[bytes]) -> dict[str, Any]:
        started = time.perf_counter()
        self.socket.send_multipart(parts)
        response = dict(self.socket.recv_json())
        client_roundtrip_ms = (time.perf_counter() - started) * 1000.0
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "Step3 split-policy request failed"))
        metrics = dict(response.get("metrics") or {})
        if metrics:
            metrics["client_roundtrip_ms"] = client_roundtrip_ms
            metrics["network_ms"] = max(
                0.0, client_roundtrip_ms - float(metrics.get("server_total_ms") or 0.0)
            )
            response["metrics"] = metrics
        return response

    def navigate(self, request: GraphNavRequest) -> dict[str, Any]:
        metadata = {"type": "navigate", "request": request.metadata()}
        parts = [json.dumps(metadata, separators=(",", ":")).encode("utf-8")]
        parts.extend(item.jpeg for item in request.candidates)
        return self._request(parts)

    def decide(self, request: GraphNavRequest) -> dict[str, Any]:
        """Compatibility alias; the split service always dispatches NavigationPolicy."""
        return self.navigate(request)

    def verify_arrival(self, request: ArrivalRequest) -> dict[str, Any]:
        metadata = {"type": "verify_arrival", "request": request.metadata()}
        parts = [json.dumps(metadata, separators=(",", ":")).encode("utf-8")]
        parts.extend(item.jpeg for item in request.current_views)
        parts.extend(item.jpeg for item in request.history_frames)
        return self._request(parts)

    def set_arrival_threshold(self, threshold: float) -> dict[str, Any]:
        envelope = {"type": "set_arrival_threshold", "threshold": float(threshold)}
        return self._request([json.dumps(envelope, separators=(",", ":")).encode("utf-8")])

    def close(self) -> None:
        self.socket.close(linger=0)

    def __enter__(self) -> "Step3GraphNavClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
