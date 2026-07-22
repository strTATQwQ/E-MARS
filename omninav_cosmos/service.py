from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Callable

from .backbones.base import BackboneAdapter
from .contracts import NavigationOutput, NavigationRequest


class NavigationInferenceService:
    """Synchronous inference core with episode, frame, and deadline safety gates."""

    def __init__(
        self,
        adapter: BackboneAdapter,
        *,
        max_request_age_s: float = 1.0,
        clock: Callable[[], float] = time.time,
        retired_episode_limit: int = 128,
    ) -> None:
        if max_request_age_s <= 0:
            raise ValueError("max_request_age_s must be positive")
        self.adapter = adapter
        self.max_request_age_s = float(max_request_age_s)
        self.clock = clock
        self._active_episode = ""
        self._last_frame_id = -1
        self._retired_episodes: deque[str] = deque(maxlen=retired_episode_limit)
        self._lock = Lock()

    @property
    def active_episode(self) -> str:
        return self._active_episode

    def process(self, request: NavigationRequest) -> NavigationOutput:
        with self._lock:
            gate_reason = self._gate(request)
            if gate_reason:
                return self._safe_stop(request, gate_reason)
            try:
                output = self.adapter.infer(request)
            except Exception as exc:
                return self._safe_stop(request, f"inference_error:{type(exc).__name__}")
            if output.episode_id != request.episode_id or output.frame_id != request.frame_id:
                return self._safe_stop(request, "adapter_response_mismatch")
            self._last_frame_id = request.frame_id
            return output

    def _gate(self, request: NavigationRequest) -> str:
        age_s = self.clock() - request.timestamp
        if age_s > self.max_request_age_s:
            return "stale_request"
        if age_s < -self.max_request_age_s:
            return "clock_skew"
        if request.episode_id in self._retired_episodes:
            return "retired_episode"
        if not self._active_episode:
            self._start_episode(request.episode_id)
        elif request.episode_id != self._active_episode:
            if not request.reset_episode:
                return "episode_reset_required"
            self._retired_episodes.append(self._active_episode)
            self._start_episode(request.episode_id)
        elif request.reset_episode:
            self._start_episode(request.episode_id)
        if request.frame_id <= self._last_frame_id:
            return "stale_or_duplicate_frame"
        return ""

    def _start_episode(self, episode_id: str) -> None:
        self.adapter.reset_episode(episode_id)
        self._active_episode = episode_id
        self._last_frame_id = -1

    def _safe_stop(self, request: NavigationRequest, reason: str) -> NavigationOutput:
        return NavigationOutput.safe_stop(
            episode_id=request.episode_id,
            frame_id=request.frame_id,
            reason=reason,
            request_timestamp=request.timestamp,
            model_variant=self.adapter.model_variant,
            precision_mode=self.adapter.precision_mode,
        )


class LatestResponseGate:
    """Isaac-side guard preventing old model results from replacing newer actions."""

    def __init__(self) -> None:
        self._episode_id = ""
        self._last_frame_id = -1

    def reset(self, episode_id: str) -> None:
        self._episode_id = episode_id
        self._last_frame_id = -1

    def accept(self, output: NavigationOutput) -> bool:
        if output.episode_id != self._episode_id or output.frame_id <= self._last_frame_id:
            return False
        self._last_frame_id = output.frame_id
        return True

