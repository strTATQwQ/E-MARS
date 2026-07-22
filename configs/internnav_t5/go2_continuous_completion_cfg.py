"""T5-only navigation completion overlay for the simplified Go2 base."""

from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path


control_root = Path(os.environ["INTERNNAV_T1_CONTROL_ROOT"]).resolve()
phase = os.environ.get("INTERNVLA_T3_PHASE", "").strip()
if phase == "continuous_oracle":
    source = control_root / "configs/internnav_t3/go2_continuous_oracle_cfg.py"
elif phase in {"canary", "pilot"}:
    source = control_root / "configs/internnav_t3/go2_continuous_active_cfg.py"
else:
    raise RuntimeError(f"unsupported T5 completion_sim phase: {phase!r}")

spec = importlib.util.spec_from_file_location("internnav_t5_continuous_source", source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import the T5 continuous source config: {source}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = deepcopy(module.eval_cfg)

# Any explicitly materialized inherited rates are verified before applying this
# T5-only navigation deviation.  The active/oracle overlay may omit these keys
# until InternNav merges its defaults; the final merged values are checked by
# preflight before Isaac starts. The already bounded root twist is integrated
# into a yaw-only SE(2) root pose at 20 Hz with held joints and live footprint,
# contact and LiDAR geometry. The default retains real simulated
# RGB-D every fourth tick at 5 Hz; an explicit diagnostic may use every eighth
# tick at 2.5 Hz.  Frozen T4 and real-Go2 configurations never import this
# overlay.
base_physics_dt = eval_cfg.env.env_settings.get("physics_dt")
if base_physics_dt is not None and abs(float(base_physics_dt) - 0.005) > 1e-12:
    raise RuntimeError("T5 planar completion requires the frozen 0.005 s base")
base_rendering_interval = eval_cfg.env.env_settings.get("rendering_interval")
if base_rendering_interval is not None and int(base_rendering_interval) != 5:
    raise RuntimeError("T5 planar completion requires the frozen interval-5 base")
eval_cfg.env.env_settings["physics_dt"] = 0.05
rendering_interval = int(
    os.environ.get("INTERNVLA_GO2_EXPECT_RENDERING_INTERVAL", "4")
)
if rendering_interval not in {4, 8}:
    raise RuntimeError("T5 completion rendering interval must be 4 or 8")
eval_cfg.env.env_settings["rendering_interval"] = rendering_interval
