#!/usr/bin/env python3
"""Run the official evaluator with the fixed Nav2 oracle AgentClient."""

from __future__ import annotations

import os
import runpy
import sys
import traceback
from pathlib import Path

from internvla_nav2_oracle_agent_client import Nav2OracleAgentClient
from run_internnav_isaac6_entrypoint import (
    _install_legacy_core_namespace,
    _install_simulation_manager_bridge,
    _patch_runner_to_reuse_app,
)


def main() -> None:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {"headless": True, "anti_aliasing": 0, "hide_ui": False, "multi_gpu": False}
    )
    _install_legacy_core_namespace()
    _install_simulation_manager_bridge()
    _patch_runner_to_reuse_app(simulation_app)
    import internnav.utils

    internnav.utils.AgentClient = Nav2OracleAgentClient
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
