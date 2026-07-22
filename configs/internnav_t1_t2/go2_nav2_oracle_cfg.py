"""Ten-episode Go2 Nav2 oracle overlay over the official task semantics."""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
from copy import deepcopy
from pathlib import Path


internnav_root = Path(os.environ["INTERNNAV_ROOT"]).resolve()
upstream = internnav_root / "scripts/eval/configs/h1_internvla_n1_async_cfg.py"
spec = importlib.util.spec_from_file_location("internnav_go2_nav2_oracle_upstream", upstream)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load official config: {upstream}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = deepcopy(module.eval_cfg)

dataset_root = Path(os.environ["INTERNVLA_ORACLE_DATASET_ROOT"]).resolve()
episode_file = dataset_root / "val_unseen/val_unseen.json.gz"
with gzip.open(episode_file, "rt", encoding="utf-8") as stream:
    episode_count = len(json.load(stream)["episodes"])
if episode_count != 10:
    raise RuntimeError(f"oracle overlay must contain 10 episodes, got {episode_count}")

eval_cfg.eval_settings["use_agent_server"] = True
eval_cfg.eval_settings["output_path"] = str(Path(os.environ["INTERNVLA_ORACLE_RESULT_DIR"]).resolve())
eval_cfg.dataset.dataset_settings["base_data_dir"] = str(dataset_root)
eval_cfg.task.task_name = os.environ["INTERNVLA_ORACLE_TASK_NAME"]
eval_cfg.task.robot_usd_path = os.environ["INTERNVLA_GO2_WRAPPER_USD"]
eval_cfg.task.camera_prim_path = "base/internvla_camera"
eval_cfg.agent.model_settings["vis_debug"] = False
eval_cfg.agent.model_settings["vis_debug_path"] = f"logs/internnav_t2/{eval_cfg.task.task_name}"
