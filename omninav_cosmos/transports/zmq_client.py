from __future__ import annotations

from typing import Any

from ..contracts import NavigationOutput, NavigationRequest
from ..service import LatestResponseGate
from .wire import control_message, decode_response, encode_request


class ZmqNavigationClient:
    def __init__(self, endpoint: str, *, timeout_ms: int = 800) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        import zmq

        self.zmq = zmq
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.context = zmq.Context.instance()
        self.socket = self._new_socket()
        self.response_gate = LatestResponseGate()
        self._episode_id = ""

    def _new_socket(self):
        socket = self.context.socket(self.zmq.REQ)
        socket.setsockopt(self.zmq.LINGER, 0)
        socket.setsockopt(self.zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(self.zmq.RCVTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        return socket

    def reset_episode(self, episode_id: str) -> None:
        self._episode_id = episode_id
        self.response_gate.reset(episode_id)

    def infer(self, request: NavigationRequest) -> NavigationOutput:
        if request.episode_id != self._episode_id:
            self.reset_episode(request.episode_id)
        try:
            self.socket.send_multipart(encode_request(request))
            if not self.socket.poll(self.timeout_ms, self.zmq.POLLIN):
                raise TimeoutError("model request timeout")
            output = decode_response(self.socket.recv())
        except Exception as exc:
            self._reconnect()
            return NavigationOutput.safe_stop(
                episode_id=request.episode_id,
                frame_id=request.frame_id,
                reason=f"network_error:{type(exc).__name__}",
                request_timestamp=request.timestamp,
            )
        if not self.response_gate.accept(output):
            return NavigationOutput.safe_stop(
                episode_id=request.episode_id,
                frame_id=request.frame_id,
                reason="stale_or_mismatched_response",
                request_timestamp=request.timestamp,
                model_variant=output.model_variant,
                precision_mode=output.precision_mode,
            )
        return output

    def health(self) -> dict[str, Any]:
        """Return service health using the same timeout/reconnect policy as inference."""
        try:
            self.socket.send(control_message("health"))
            if not self.socket.poll(self.timeout_ms, self.zmq.POLLIN):
                raise TimeoutError("model health request timeout")
            payload = self.socket.recv_json()
        except Exception:
            self._reconnect()
            raise
        if not isinstance(payload, dict) or not payload.get("ok"):
            raise RuntimeError(f"model health check failed: {payload!r}")
        health = payload.get("health")
        if not isinstance(health, dict):
            raise RuntimeError("model health response omitted health object")
        return dict(health)

    def _reconnect(self) -> None:
        self.socket.close(linger=0)
        self.socket = self._new_socket()

    def close(self) -> None:
        self.socket.close(linger=0)
