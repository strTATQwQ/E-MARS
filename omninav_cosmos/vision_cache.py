from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class VisionCacheKey:
    episode_id: str
    frame_id: int
    view: str
    width: int
    height: int
    model_fingerprint: str


class EpisodeVisionCache:
    """Thread-safe, episode-isolated ring cache for encoded visual features."""

    def __init__(self, capacity: int = 20, *, enabled: bool = True) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.enabled = enabled
        self._active_episode = ""
        self._entries: OrderedDict[VisionCacheKey, Any] = OrderedDict()
        self._lock = RLock()

    @property
    def active_episode(self) -> str:
        return self._active_episode

    def begin_episode(self, episode_id: str) -> None:
        episode_id = str(episode_id).strip()
        if not episode_id:
            raise ValueError("episode_id is required")
        with self._lock:
            if episode_id != self._active_episode:
                self._entries.clear()
                self._active_episode = episode_id

    def reset(self, episode_id: str | None = None) -> None:
        with self._lock:
            if episode_id is None or episode_id == self._active_episode:
                self._entries.clear()
                if episode_id is None:
                    self._active_episode = ""

    def get(self, key: VisionCacheKey) -> Any | None:
        if not self.enabled:
            return None
        with self._lock:
            self._validate_key(key)
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key: VisionCacheKey, feature: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._validate_key(key)
            self._invalidate_incompatible(key)
            self._entries[key] = feature
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)

    def keys(self) -> tuple[VisionCacheKey, ...]:
        with self._lock:
            return tuple(self._entries)

    def _validate_key(self, key: VisionCacheKey) -> None:
        if not self._active_episode:
            raise RuntimeError("begin_episode must be called before cache access")
        if key.episode_id != self._active_episode:
            raise RuntimeError("cross-episode cache access rejected")
        if key.frame_id < 0 or key.width <= 0 or key.height <= 0:
            raise ValueError("invalid cache key dimensions or frame_id")
        if not key.view or not key.model_fingerprint:
            raise ValueError("cache key requires view and model_fingerprint")

    def _invalidate_incompatible(self, new_key: VisionCacheKey) -> None:
        incompatible = [
            key
            for key in self._entries
            if key.view == new_key.view
            and (
                key.width != new_key.width
                or key.height != new_key.height
                or key.model_fingerprint != new_key.model_fingerprint
            )
        ]
        for key in incompatible:
            del self._entries[key]

