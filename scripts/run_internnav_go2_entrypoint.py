#!/usr/bin/env python3
"""Run the official evaluator with the Go2 runtime and selected typed client."""

from __future__ import annotations

import json
import os
import runpy
import sys
import traceback
from functools import wraps
from pathlib import Path
from typing import Any

from run_internnav_isaac6_entrypoint import (
    _install_legacy_core_namespace,
    _install_simulation_manager_bridge,
    _patch_runner_to_reuse_app,
)


def _temporary_method(instance: Any, name: str, replacement: Any) -> tuple[bool, Any]:
    """Install an instance wrapper while preserving descriptor restoration."""

    namespace = getattr(instance, "__dict__", {})
    had_instance_value = name in namespace
    previous = namespace.get(name)
    setattr(instance, name, replacement)
    return had_instance_value, previous


def _restore_temporary_method(
    instance: Any, name: str, state: tuple[bool, Any]
) -> None:
    had_instance_value, previous = state
    if had_instance_value:
        setattr(instance, name, previous)
    else:
        delattr(instance, name)


def _install_t5_pre_warmup_reset_order(evaluator_class: type[Any]) -> None:
    """Move T5 identity reset ahead of the next episode's warm-up.

    Upstream InternNav resets the environment, performs its stand-still
    warm-up, and only then resets the agent.  The distributed T5 client arms
    a fresh-odometry barrier during that late reset, so the first model call
    cannot observe another physics sample and safely times out.  This T5-only
    wrapper resets the agent immediately after ``env.reset`` returns and
    consumes the upstream duplicate after warm-up.  No stale odometry is
    reused and no freshness threshold is relaxed.
    """

    original = evaluator_class.terminate_ops
    if getattr(original, "_internnav_t5_pre_warmup_reset", False):
        return

    @wraps(original)
    def terminate_ops(self: Any, obs_ls: Any, reset_infos: Any, terminated_ls: Any) -> Any:
        pending = set(getattr(self, "_internnav_t5_pre_warmup_reset_ids", set()))
        real_env_reset = self.env.reset
        real_agent_reset = self.agent.reset

        def agent_reset_once(indices: Any) -> Any:
            reset_ids = [int(value) for value in indices]
            skipped = [value for value in reset_ids if value in pending]
            forwarded = [value for value in reset_ids if value not in pending]
            pending.difference_update(skipped)
            if skipped:
                print(
                    "INTERNVLA_T5_POST_WARMUP_RESET_SKIPPED "
                    + json.dumps(
                        {"schema_version": 1, "env_ids": skipped},
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            if forwarded:
                return real_agent_reset(forwarded)
            return None

        def env_reset_then_agent(indices: Any) -> Any:
            reset_ids = [int(value) for value in indices]
            if reset_ids != [0]:
                raise RuntimeError(
                    "T5 pre-warmup reset supports exactly one isolated evaluator env"
                )
            observations, new_infos = real_env_reset(reset_ids)
            if len(new_infos) not in {0, len(reset_ids)}:
                raise RuntimeError("T5 env.reset returned an ambiguous reset-info set")
            live_ids = (
                []
                if len(new_infos) == 0
                else [
                    env_id
                    for env_id, info in zip(reset_ids, new_infos)
                    if info is not None
                ]
            )
            if live_ids:
                if pending.intersection(live_ids):
                    raise RuntimeError("T5 pre-warmup reset identity was not consumed")
                real_agent_reset(live_ids)
                pending.update(live_ids)
                print(
                    "INTERNVLA_T5_PRE_WARMUP_RESET "
                    + json.dumps(
                        {"schema_version": 1, "env_ids": live_ids},
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            return observations, new_infos

        agent_state = _temporary_method(self.agent, "reset", agent_reset_once)
        env_state = _temporary_method(self.env, "reset", env_reset_then_agent)
        try:
            return original(self, obs_ls, reset_infos, terminated_ls)
        finally:
            self._internnav_t5_pre_warmup_reset_ids = pending
            _restore_temporary_method(self.env, "reset", env_state)
            _restore_temporary_method(self.agent, "reset", agent_state)

    terminate_ops._internnav_t5_pre_warmup_reset = True
    evaluator_class.terminate_ops = terminate_ops


def _registered_vln_distributed_evaluator(evaluator_base: Any) -> type[Any]:
    """Return the registered class; InternNav's register decorator returns None."""

    evaluator_class = getattr(evaluator_base, "evaluators", {}).get(
        "vln_distributed"
    )
    if not isinstance(evaluator_class, type):
        raise RuntimeError("registered vln_distributed evaluator class is unavailable")
    return evaluator_class


def main() -> None:
    from isaacsim import SimulationApp

    launcher_config = {
        "headless": True,
        "anti_aliasing": 0,
        "hide_ui": False,
        "multi_gpu": False,
    }
    # CUDA_VISIBLE_DEVICES constrains CUDA, but it does not select the Vulkan
    # device used by Kit's RTX renderer.  T5 dual-lane runs therefore provide
    # the physical renderer index explicitly while physics remains logical
    # device zero inside the lane's CUDA visibility mask.  Absent variables
    # preserve the frozen T4 launcher behavior exactly.
    render_gpu = os.environ.get("INTERNVLA_ISAAC_RENDER_GPU")
    physics_gpu = os.environ.get("INTERNVLA_ISAAC_PHYSICS_GPU")
    if render_gpu is not None:
        if not render_gpu.isdecimal():
            raise RuntimeError("INTERNVLA_ISAAC_RENDER_GPU must be an integer")
        launcher_config["active_gpu"] = int(render_gpu)
    if physics_gpu is not None:
        if not physics_gpu.isdecimal():
            raise RuntimeError("INTERNVLA_ISAAC_PHYSICS_GPU must be an integer")
        launcher_config["physics_gpu"] = int(physics_gpu)
    simulation_app = SimulationApp(launcher_config)
    if os.environ.get("INTERNVLA_GO2_EXECUTION_MODE", "flash") == "continuous":
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.sensors.experimental.physics")
        simulation_app.update()
    _install_legacy_core_namespace()
    _install_simulation_manager_bridge()
    _patch_runner_to_reuse_app(simulation_app)
    import internnav_go2_runtime as go2_runtime

    print(f"INTERNVLA_GO2_RUNTIME_MODULE={go2_runtime.__file__}", flush=True)
    go2_runtime.install_go2_runtime()
    mode = os.environ.get("INTERNVLA_GO2_CLIENT_MODE", "")
    if mode == "oracle":
        from internvla_nav2_oracle_agent_client import Nav2OracleAgentClient as Client
    elif mode == "model":
        if (
            os.environ.get("INTERNNAV_T5_STEP3_LIVE_ADVISOR", "0") == "1"
            or os.environ.get("INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL", "0") == "1"
        ):
            from internnav_t5_lane_b_step3_agent_client import (
                LaneBStep3ROS2IPCAgentClient as Client,
            )
        else:
            from internvla_ipc_agent_client import ROS2IPCAgentClient as Client
    elif mode == "continuous_oracle":
        from internvla_go2_continuous_agent_client import (
            ContinuousNav2OracleAgentClient as Client,
        )
    elif mode == "continuous_model":
        if (
            os.environ.get("INTERNNAV_T5_STEP3_LIVE_ADVISOR", "0") == "1"
            or os.environ.get("INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL", "0") == "1"
        ):
            from internnav_t5_lane_b_step3_agent_client import (
                LaneBStep3ContinuousROS2IPCAgentClient as Client,
            )
        else:
            from internvla_go2_continuous_agent_client import (
                ContinuousROS2IPCAgentClient as Client,
            )
    else:
        raise RuntimeError(
            "INTERNVLA_GO2_CLIENT_MODE must be oracle, model, "
            "continuous_oracle, or continuous_model"
        )
    import internnav.utils

    internnav.utils.AgentClient = Client
    if mode in {"continuous_model", "continuous_oracle"} and (
        os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
        and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
        and os.environ.get("INTERNNAV_T5_LANE", "") in {"a", "b"}
    ):
        from internnav.evaluator import Evaluator

        _install_t5_pre_warmup_reset_order(
            _registered_vln_distributed_evaluator(Evaluator)
        )
    eval_script = Path(os.environ["INTERNNAV_ROOT"]).resolve() / "scripts/eval/eval.py"
    sys.argv = [str(eval_script), *sys.argv[1:]]
    try:
        runpy.run_path(str(eval_script), run_name="__main__")
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    if simulation_app.is_running():
        simulation_app.close()


if __name__ == "__main__":
    main()
