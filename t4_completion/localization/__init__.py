"""Completion-only T4.3 localization selector.

The package deliberately has no ROS or Isaac import at module import time.  The
core can therefore be tested offline while the ROS adapter remains an optional
runtime boundary.
"""

from .config import LocalizationConfig, load_config
from .contracts import CanonicalOutput, Pose, PoseSample, SelectionDecision
from .selector import LocalizationSelector

__all__ = [
    "CanonicalOutput",
    "LocalizationConfig",
    "LocalizationSelector",
    "Pose",
    "PoseSample",
    "SelectionDecision",
    "load_config",
]
