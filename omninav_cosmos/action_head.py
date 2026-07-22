from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ActionHeadConfig:
    hidden_size: int
    waypoint_count: int = 5
    attention_heads: int = 4
    action_former_layers: int = 1
    arrive_count: int = 5
    predict_heading: bool = True
    predict_confidence: bool = True
    normalize_heading: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.waypoint_count <= 0:
            raise ValueError("hidden_size and waypoint_count must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.action_former_layers <= 0:
            raise ValueError("action_former_layers must be positive")
        if self.arrive_count not in (1, self.waypoint_count):
            raise ValueError("arrive_count must be 1 or waypoint_count")


@dataclass
class ActionHeadOutput:
    waypoints: Tensor
    heading_sin_cos: Tensor
    arrive_logits: Tensor
    confidence: Tensor
    action_feature: Tensor


class _HeadBlock(nn.Module):
    def __init__(self, hidden_size: int, attention_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, attention_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, query: Tensor, hidden_states: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        attended, _ = self.attn(
            query,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = self.norm1(query + attended)
        return self.norm2(query + self.mlp(query))


class NavigationActionHead(nn.Module):
    """Backbone-neutral OmniNav action query and prediction heads.

    With one action-former layer and ``normalize_heading=False``, tensor shapes
    and waypoint accumulation match the released Qwen2.5-VL implementation.
    """

    def __init__(self, config: ActionHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.query_action = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        if config.action_former_layers == 1:
            self.query_multihead_attn = nn.MultiheadAttention(
                config.hidden_size,
                config.attention_heads,
                batch_first=True,
            )
            self.action_former = None
        else:
            self.query_multihead_attn = None
            self.action_former = nn.ModuleList(
                [_HeadBlock(config.hidden_size, config.attention_heads) for _ in range(config.action_former_layers)]
            )
        self.wp_predictor = nn.Linear(config.hidden_size, config.waypoint_count * 2)
        self.wp_predictor_angle = (
            nn.Linear(config.hidden_size, config.waypoint_count * 2) if config.predict_heading else None
        )
        self.arrive_predictor = nn.Linear(config.hidden_size, config.arrive_count)
        self.confidence_predictor = nn.Linear(config.hidden_size, 1) if config.predict_confidence else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_action, mean=0.0, std=0.02)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> ActionHeadOutput:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"hidden_states must be [batch, sequence, {self.config.hidden_size}], got {tuple(hidden_states.shape)}"
            )
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError("attention_mask must match hidden_states batch and sequence dimensions")
            key_padding_mask = ~attention_mask.to(dtype=torch.bool, device=hidden_states.device)
        query = self.query_action.expand(hidden_states.shape[0], -1, -1)
        if self.query_multihead_attn is not None:
            query, _ = self.query_multihead_attn(
                query,
                hidden_states,
                hidden_states,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        else:
            assert self.action_former is not None
            for block in self.action_former:
                query = block(query, hidden_states, key_padding_mask)
        action_feature = query.squeeze(1)
        waypoint_deltas = self.wp_predictor(action_feature).view(-1, self.config.waypoint_count, 2)
        waypoints = torch.cumsum(waypoint_deltas, dim=1)
        if self.wp_predictor_angle is None:
            heading = torch.zeros_like(waypoints)
            heading[..., 1] = 1.0
        else:
            heading = torch.tanh(self.wp_predictor_angle(action_feature)).view(
                -1, self.config.waypoint_count, 2
            )
            if self.config.normalize_heading:
                heading = torch.nn.functional.normalize(heading, dim=-1, eps=1e-6)
        arrive_logits = self.arrive_predictor(action_feature)
        if self.confidence_predictor is None:
            arrive_prob = torch.sigmoid(arrive_logits)
            confidence = torch.maximum(arrive_prob, 1.0 - arrive_prob).mean(dim=-1)
        else:
            confidence = torch.sigmoid(self.confidence_predictor(action_feature)).squeeze(-1)
        return ActionHeadOutput(waypoints, heading, arrive_logits, confidence, action_feature)

    def load_legacy_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> tuple[list[str], list[str]]:
        """Load released OmniNav head keys without modifying the legacy model."""
        prefixes = ("", "model.", "module.")
        target = self.state_dict()
        mapped: dict[str, Any] = {}
        for target_key in target:
            if target_key.startswith("confidence_predictor"):
                continue
            candidates = [target_key]
            if target_key.startswith("action_former."):
                candidates.append(target_key.replace("action_former.", "query_multihead_multi_attn.blocks.", 1))
            for prefix in prefixes:
                found = next((state[prefix + candidate] for candidate in candidates if prefix + candidate in state), None)
                if found is not None:
                    mapped[target_key] = found
                    break
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        required_missing = [key for key in missing if not key.startswith("confidence_predictor")]
        if strict and required_missing:
            raise KeyError(f"missing legacy action-head keys: {required_missing}")
        return list(missing), list(unexpected)

