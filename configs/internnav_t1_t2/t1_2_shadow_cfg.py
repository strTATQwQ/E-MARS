"""Output-only T1.2 overlay over the frozen T0 canary config."""

from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path


control_root = Path(os.environ["INTERNNAV_T0_CONTROL_ROOT"]).resolve()
source = control_root / "configs" / "internnav_t0" / "official_agent_server_cfg.py"
spec = importlib.util.spec_from_file_location("internnav_t1_shadow_source", source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import frozen T0 overlay: {source}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = deepcopy(module.eval_cfg)

# Only output namespaces differ from the frozen official canary.
eval_cfg.task.task_name = "internnav_t1_shadow"
eval_cfg.agent.model_settings["vis_debug_path"] = "logs/internnav_t1/shadow_agent_debug"
