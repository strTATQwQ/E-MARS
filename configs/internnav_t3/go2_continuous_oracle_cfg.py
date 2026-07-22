"""T3 continuous Go2 Oracle overlay over the frozen ten-episode T2 set."""

from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path


control_root = Path(os.environ["INTERNNAV_T1_CONTROL_ROOT"]).resolve()
source = control_root / "configs/internnav_t1_t2/go2_nav2_oracle_cfg.py"
requested_dataset_root = Path(os.environ["INTERNVLA_ORACLE_DATASET_ROOT"]).resolve()
base_oracle_root = Path(
    os.environ.get(
        "INTERNVLA_T3_BASE_ORACLE_DATASET_ROOT",
        str(control_root / "episodes/h1_nav2_oracle"),
    )
).resolve()
spec = importlib.util.spec_from_file_location("internnav_t3_oracle_source", source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import frozen T2 Oracle overlay: {source}")
module = importlib.util.module_from_spec(spec)
# The frozen T2 overlay intentionally asserts exactly ten episodes.  Validate
# inheritance against that immutable source, then restore the T3 phase dataset
# (five diagnostics or ten obstacle/oracle episodes) below.
previous_dataset_root = os.environ["INTERNVLA_ORACLE_DATASET_ROOT"]
os.environ["INTERNVLA_ORACLE_DATASET_ROOT"] = str(base_oracle_root)
try:
    spec.loader.exec_module(module)
finally:
    os.environ["INTERNVLA_ORACLE_DATASET_ROOT"] = previous_dataset_root
eval_cfg = deepcopy(module.eval_cfg)
eval_cfg.dataset.dataset_settings["base_data_dir"] = str(requested_dataset_root)
eval_cfg.eval_settings["output_path"] = str(
    Path(os.environ["INTERNVLA_ORACLE_RESULT_DIR"]).resolve()
)
eval_cfg.task.task_name = os.environ["INTERNVLA_T3_TASK_NAME"]
eval_cfg.task.robot_flash = False
eval_cfg.task.flash_collision = False
eval_cfg.task.one_step_stand_still = True
phase = os.environ.get("INTERNVLA_T3_PHASE", "continuous_oracle")
default_max_step = 2500 if phase == "diagnostics" else 12000
eval_cfg.task.task_settings["max_step"] = int(
    os.environ.get("INTERNVLA_T3_MAX_STEP", str(default_max_step))
)
eval_cfg.eval_settings["vis_output"] = False
eval_cfg.agent.model_settings["vis_debug"] = False
eval_cfg.agent.model_settings["vis_debug_path"] = (
    f"logs/internnav_t3/{eval_cfg.task.task_name}"
)
