"""Non-invasive capture of InternVLA's metric local trajectory boundary."""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CapturedTrajectoryCandidates:
    """Derived candidates plus the exact upstream diffusion tensor shape."""

    raw_shape: tuple[int, ...]
    trajectories: Any


class MetricTrajectoryCapture:
    """Patch the policy symbol while preserving the original discrete call.

    The original function is called once on the real tensor to produce actions.
    A detached clone is passed through its existing continuous branch to expose
    the cumulative-and-averaged metric XY polyline.  The clone prevents the
    continuous branch's in-place unnormalization from changing model outputs.
    """

    def __init__(self, *, capture_candidates: bool = False) -> None:
        self._local = threading.local()
        self._installed = False
        self._original: Any = None
        self._capture_candidates = bool(capture_candidates)

    def install(self) -> None:
        if self._installed:
            return
        from internnav.model.utils import vln_utils
        from internnav.model.basemodel.internvla_n1 import internvla_n1_policy

        original = vln_utils.traj_to_actions
        self._original = original

        @functools.wraps(original)
        def traced(dp_actions: Any, use_discrate_action: bool = True) -> Any:
            if not use_discrate_action:
                return original(dp_actions, use_discrate_action=False)
            continuous_input = dp_actions.detach().clone()
            raw_shape = tuple(int(value) for value in continuous_input.shape)
            trajectory = original(continuous_input, use_discrate_action=False)
            if self._capture_candidates:
                # The upstream continuous branch has unnormalized XY deltas in
                # place.  Preserve every diffusion sample before the real
                # discrete branch mutates its input and averages all 32.
                import numpy as np

                raw = np.asarray(
                    continuous_input.float().cpu().numpy(), dtype=np.float32
                )
                if raw_shape == (32, 32, 3) and tuple(raw.shape) == raw_shape:
                    deltas = raw[:, :, :2]
                    starts = np.zeros((deltas.shape[0], 1, 2), dtype=np.float32)
                    candidates = np.concatenate(
                        (starts, np.cumsum(deltas, axis=1, dtype=np.float32)),
                        axis=1,
                    )
                else:
                    candidates = raw
                self._local.candidates = np.ascontiguousarray(
                    candidates, dtype=np.float32
                )
                self._local.candidate_raw_shape = raw_shape
            actions = original(dp_actions, use_discrate_action=True)
            self._local.trajectory = trajectory.copy()
            return actions

        # Patch both the defining module and the symbol imported by the policy.
        vln_utils.traj_to_actions = traced
        internvla_n1_policy.traj_to_actions = traced
        self._installed = True

    def begin_step(self) -> None:
        self._local.trajectory = None
        if self._capture_candidates:
            self._local.candidates = None
            self._local.candidate_raw_shape = None

    def take(self) -> Any | None:
        trajectory = getattr(self._local, "trajectory", None)
        self._local.trajectory = None
        if trajectory is None:
            return None
        import numpy as np

        value = np.asarray(trajectory, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != 2:
            raise RuntimeError(f"metric trajectory must be [N,2], observed {value.shape}")
        if not np.isfinite(value).all():
            raise RuntimeError("metric trajectory contains NaN/Inf")
        return value

    def take_candidates(self) -> Any | None:
        if not self._capture_candidates:
            return None
        candidates = getattr(self._local, "candidates", None)
        raw_shape = getattr(self._local, "candidate_raw_shape", None)
        self._local.candidates = None
        self._local.candidate_raw_shape = None
        if candidates is None:
            return None
        import numpy as np

        value = np.asarray(candidates, dtype=np.float32)
        if value.ndim != 3 or value.shape[2] != 2:
            # Shape/count validation and deterministic fallback belong to the
            # reranker, so retain malformed upstream shapes for its audit.
            trajectories = value
        else:
            trajectories = np.ascontiguousarray(value, dtype=np.float32)
        return CapturedTrajectoryCandidates(
            raw_shape=tuple(int(item) for item in raw_shape or ()),
            trajectories=trajectories,
        )
