"""Offline-safe composer for the T4 completion map fast path.

The package intentionally has no ROS imports at module import time.  Config
validation, launch composition, and the deterministic smoke model therefore
run on a developer machine without Isaac, a ROS graph, or shared resources.
"""

from .composer import ComposeRequest, compose_bundle, compose_plan
from .contract import ContractError, LoadedConfigs, load_and_validate

__all__ = [
    "ComposeRequest",
    "ContractError",
    "LoadedConfigs",
    "compose_bundle",
    "compose_plan",
    "load_and_validate",
]
