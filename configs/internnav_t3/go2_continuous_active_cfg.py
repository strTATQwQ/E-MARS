"""T3 InternVLA + continuous Go2 overlay over the frozen T2 episode sets."""

from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path


control_root = Path(os.environ["INTERNNAV_T1_CONTROL_ROOT"]).resolve()
source = control_root / "configs/internnav_t1_t2/go2_nav2_active_cfg.py"
spec = importlib.util.spec_from_file_location("internnav_t3_active_source", source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import frozen T2 active overlay: {source}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = deepcopy(module.eval_cfg)
eval_cfg.task.task_name = os.environ["INTERNVLA_T3_TASK_NAME"]
eval_cfg.task.robot_flash = False
eval_cfg.task.flash_collision = False
eval_cfg.task.one_step_stand_still = True
eval_cfg.task.task_settings["max_step"] = int(
    os.environ.get("INTERNVLA_T3_MAX_STEP", "8000")
)
eval_cfg.eval_settings["vis_output"] = False
eval_cfg.agent.model_settings["vis_debug"] = False
eval_cfg.agent.model_settings["vis_debug_path"] = (
    f"logs/internnav_t3/{eval_cfg.task.task_name}"
)
