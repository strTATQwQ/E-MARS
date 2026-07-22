"""Output-only overlay for the frozen T1.0 legacy replay capture."""

from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path


control_root = Path(os.environ["INTERNNAV_T0_CONTROL_ROOT"]).resolve()
source = control_root / "configs" / "internnav_t0" / "official_agent_server_cfg.py"
spec = importlib.util.spec_from_file_location("internnav_t1_t0_source", source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import frozen T0 overlay: {source}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = deepcopy(module.eval_cfg)

# Output isolation only. Evaluation semantics remain the frozen canary values.
eval_cfg.task.task_name = "internnav_t1_replay_capture"
eval_cfg.agent.model_settings["vis_debug_path"] = "logs/internnav_t1/replay_capture_agent_debug"
