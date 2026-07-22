from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..contracts import NavigationOutput, ProtocolError
from ..service import NavigationInferenceService
from .wire import decode_request, encode_response, parse_control


class ZmqNavigationServer:
    def __init__(
        self,
        service: NavigationInferenceService,
        *,
        bind: str,
        log_path: str | Path,
    ) -> None:
        import zmq

        self.zmq = zmq
        self.service = service
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(bind)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        while True:
            parts = self.socket.recv_multipart()
            received_at = time.time()
            if len(parts) == 1:
                response = self._control(parts[0])
                self.socket.send_json(response)
                continue
            try:
                request = decode_request(parts)
                output = self.service.process(request)
                output = replace(output, server_total_latency_ms=(time.time() - received_at) * 1000.0)
                response = encode_response(output)
                self._log(
                    {
                        "event": "navigation_response",
                        "received_at": received_at,
                        "sent_at": time.time(),
                        "request": request.metadata(),
                        "response": output.to_mapping(),
                    }
                )
            except Exception as exc:
                episode_id, frame_id, timestamp = self._best_effort_request_identity(parts)
                output = NavigationOutput.safe_stop(
                    episode_id=episode_id,
                    frame_id=frame_id,
                    reason=f"protocol_error:{type(exc).__name__}",
                    request_timestamp=timestamp,
                    model_variant=self.service.adapter.model_variant,
                    precision_mode=self.service.adapter.precision_mode,
                )
                response = encode_response(output)
                self._log(
                    {
                        "event": "protocol_error",
                        "received_at": received_at,
                        "sent_at": time.time(),
                        "error": repr(exc),
                        "response": output.to_mapping(),
                    }
                )
            self.socket.send(response)

    def _control(self, value: bytes) -> dict[str, Any]:
        try:
            payload = parse_control(value)
            kind = payload.get("type")
            if kind == "health":
                return {"ok": True, "health": self.service.adapter.health(), "time": time.time()}
            return {"ok": False, "error": f"unsupported control type {kind!r}"}
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}

    def _log(self, payload: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    @staticmethod
    def _best_effort_request_identity(parts: list[bytes]) -> tuple[str, int, float]:
        try:
            payload = json.loads(parts[0].decode("utf-8"))
            episode_id = str(payload.get("episode_id") or "protocol-error")
            frame_id = max(0, int(payload.get("frame_id", 0)))
            timestamp = float(payload.get("timestamp", 0.0))
            return episode_id, frame_id, timestamp
        except Exception:
            return "protocol-error", 0, 0.0

    def close(self) -> None:
        self.socket.close(linger=0)
