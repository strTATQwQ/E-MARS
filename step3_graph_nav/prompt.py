"""Compatibility aliases for the move-only NavigationPolicy prompt."""

from .navigation_policy import NAVIGATION_SYSTEM_PROMPT as SYSTEM_PROMPT
from .navigation_policy import build_navigation_prompt as build_prompt

__all__ = ["SYSTEM_PROMPT", "build_prompt"]
