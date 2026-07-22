"""Go2 active Nav2 overlay over the frozen official T0 episode config."""

from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path


phase = os.environ.get("INTERNNAV_T0_PHASE", "").strip()
if phase not in {"canary", "stress", "pilot"}:
    raise RuntimeError("Go2 active phase must be canary, stress, or pilot")
t0_control_root = Path(os.environ["INTERNNAV_T0_CONTROL_ROOT"]).resolve()
source = t0_control_root / "configs/internnav_t0/official_agent_server_cfg.py"
spec = importlib.util.spec_from_file_location("internnav_go2_active_source", source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import frozen T0 overlay: {source}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = deepcopy(module.eval_cfg)
eval_cfg.task.task_name = os.environ["INTERNVLA_GO2_ACTIVE_TASK_NAME"]
eval_cfg.task.robot_usd_path = os.environ["INTERNVLA_GO2_WRAPPER_USD"]
eval_cfg.task.camera_prim_path = "base/internvla_camera"
eval_cfg.agent.model_settings["vis_debug_path"] = f"logs/internnav_t2/go2_nav2_{phase}_agent_debug"
