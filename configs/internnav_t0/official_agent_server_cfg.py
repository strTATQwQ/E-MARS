"""Fail-closed T0 overlay for the frozen official InternNav Isaac config.

This module imports the upstream config instead of copying its semantic values.
Only the Agent Server endpoint, the immutable episode overlay, and isolated
output names are changed.  Model, prompt, action space, controller, success
criterion, sensor, and simulator settings remain upstream values.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
from copy import deepcopy
from pathlib import Path


EXPECTED_EPISODES = {"gate1": 1, "canary": 5, "stress": 10, "pilot": 20}
phase = os.environ.get("INTERNNAV_T0_PHASE", "").strip()
if phase not in EXPECTED_EPISODES:
    raise RuntimeError(
        "INTERNNAV_T0_PHASE must be one of gate1, canary, stress, or pilot; "
        "refusing an unbounded evaluation"
    )

expected_episode_count = EXPECTED_EPISODES[phase]
t4_expected_count = os.environ.get("INTERNVLA_T4_EXPECTED_COUNT", "").strip()
if t4_expected_count:
    if not (
        phase == "pilot"
        and os.environ.get("INTERNNAV_RUNTIME_POLICY") == "completion_sim"
        and os.environ.get("INTERNNAV_SIMULATION_TARGET") == "isaac"
        and os.environ.get("INTERNNAV_T4_RESOURCE_LEASE_ACK") == "dgx+isaac"
    ):
        raise RuntimeError(
            "a bounded T4 pilot episode-count override requires "
            "completion_sim, Isaac, and the joint resource lease"
        )
    try:
        expected_episode_count = int(t4_expected_count)
    except ValueError as exc:
        raise RuntimeError("T4 pilot episode count must be an integer") from exc
    if not 1 <= expected_episode_count <= EXPECTED_EPISODES["pilot"]:
        raise RuntimeError("T4 pilot episode count must be within [1, 20]")

internnav_root = Path(os.environ.get("INTERNNAV_ROOT", ".")).resolve()
upstream_path = (
    internnav_root
    / "scripts"
    / "eval"
    / "configs"
    / "h1_internvla_n1_async_cfg.py"
)
if not upstream_path.is_file():
    raise FileNotFoundError(f"frozen upstream config is missing: {upstream_path}")

spec = importlib.util.spec_from_file_location("internnav_t0_upstream_cfg", upstream_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import upstream config: {upstream_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = deepcopy(module.eval_cfg)

dataset_root = Path(os.environ["INTERNNAV_T0_DATASET_ROOT"]).resolve()
result_dir = Path(os.environ["INTERNNAV_T0_RESULT_DIR"]).resolve()
episode_file = dataset_root / "val_unseen" / "val_unseen.json.gz"
if not episode_file.is_file():
    raise FileNotFoundError(f"immutable {phase} episode overlay is missing: {episode_file}")
with gzip.open(episode_file, "rt", encoding="utf-8") as stream:
    episode_count = len(json.load(stream)["episodes"])
if episode_count != expected_episode_count:
    raise RuntimeError(
        f"{phase} overlay has {episode_count} raw episodes; "
        f"expected exactly {expected_episode_count}"
    )

# Required Dual System process separation.
eval_cfg.agent.server_host = os.environ["INTERNNAV_SERVER_HOST"]
eval_cfg.eval_settings["use_agent_server"] = True

# The overlay is a byte-for-byte subset of the pinned official val_unseen file.
# No loader, filtering, prompt, policy, action, controller, or metric is changed.
eval_cfg.dataset.dataset_settings["base_data_dir"] = str(dataset_root)

# The upstream evaluator already exposes output_path.  Relocate it only; all
# metric definitions and episode completion semantics remain upstream values.
eval_cfg.eval_settings["output_path"] = str(result_dir)

# task_name is the upstream evaluator's output selector.  vis_debug_path is
# relocated with it so all debug frames remain within the isolated run root.
eval_cfg.task.task_name = f"internnav_t0_{phase}"
eval_cfg.agent.model_settings["vis_debug_path"] = os.environ.get(
    "INTERNNAV_T0_AGENT_DEBUG_PATH",
    f"logs/internnav_t0/{phase}_agent_vis_debug",
)
