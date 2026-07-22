from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..contracts import NavigationOutput, NavigationRequest


class BackboneAdapter(ABC):
    model_variant: str
    precision_mode: str

    @abstractmethod
    def infer(self, request: NavigationRequest) -> NavigationOutput:
        raise NotImplementedError

    @abstractmethod
    def reset_episode(self, episode_id: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def health(self) -> dict[str, Any]:
        return {
            "model_variant": self.model_variant,
            "precision_mode": self.precision_mode,
            "ready": True,
        }

