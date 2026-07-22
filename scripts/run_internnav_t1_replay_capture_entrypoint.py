#!/usr/bin/env python3
"""Run the frozen legacy evaluator while recording a bounded T1 replay."""

from __future__ import annotations

import os
import runpy
import sys
import traceback
from pathlib import Path

from capture_internnav_t1_replay import make_recording_client
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

    from internnav.utils.comm_utils.client import AgentClient as LegacyAgentClient
    import internnav.utils

    internnav.utils.AgentClient = make_recording_client(LegacyAgentClient)
    internnav_root = Path(os.environ["INTERNNAV_ROOT"]).resolve()
    eval_script = internnav_root / "scripts" / "eval" / "eval.py"
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
