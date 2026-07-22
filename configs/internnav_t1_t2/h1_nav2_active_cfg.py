"""H1 Nav2-active output overlay over the frozen official T0 config.

The imported T0 overlay already pins the upstream evaluator, dataset subset,
model, prompt, controller, action space, and metric semantics.  This file only
assigns a distinct task/debug namespace for the ROS 2 + Nav2 active run.
"""

from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path


phase = os.environ.get("INTERNNAV_T0_PHASE", "").strip()
if phase not in {"canary", "pilot"}:
    raise RuntimeError("H1 Nav2 active phase must be canary or pilot")

t0_control_root = Path(os.environ["INTERNNAV_T0_CONTROL_ROOT"]).resolve()
source = t0_control_root / "configs" / "internnav_t0" / "official_agent_server_cfg.py"
spec = importlib.util.spec_from_file_location("internnav_h1_nav2_active_source", source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import frozen T0 overlay: {source}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = deepcopy(module.eval_cfg)

# Output namespaces are the only differences from the frozen official overlay.
eval_cfg.task.task_name = os.environ["INTERNVLA_H1_ACTIVE_TASK_NAME"]
eval_cfg.agent.model_settings["vis_debug_path"] = os.environ.get(
    "INTERNVLA_H1_ACTIVE_DEBUG_PATH",
    f"logs/internnav_t1/h1_nav2_{phase}_agent_debug",
)
