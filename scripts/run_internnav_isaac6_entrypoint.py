#!/usr/bin/env python3
"""Run the pinned InternNav evaluator on the GB10-compatible Isaac 6 runtime.

InternUtopia 2.2.0 imports Isaac Sim 4.x modules from ``omni.isaac.core`` and
creates ``SimulationApp`` only after it registers every extension.  Isaac Sim 6
requires the application to be running before those registrations, and moved
the same deprecated Core API to ``isaacsim.core``.  This entrypoint supplies
only those two runtime bridges.  It then executes InternNav's unmodified
``scripts/eval/eval.py`` with its original runner, task, controller, actions,
metrics, and Agent Server protocol.
"""

from __future__ import annotations

import importlib
import os
import runpy
import sys
import traceback
import types
from pathlib import Path
from typing import Any


_MODULE_ALIASES = {
    "omni.isaac.core": "isaacsim.core.api",
    "omni.isaac.core.simulation_context": "isaacsim.core.api.simulation_context",
    "omni.isaac.core.loggers": "isaacsim.core.api.loggers",
    "omni.isaac.core.scenes": "isaacsim.core.api.scenes",
    "omni.isaac.core.objects": "isaacsim.core.api.objects",
    "omni.isaac.core.objects.cuboid": "isaacsim.core.api.objects.cuboid",
    "omni.isaac.core.prims": "isaacsim.core.prims",
    "omni.isaac.core.prims.xform_prim": "isaacsim.core.prims.impl.xform_prim",
    "omni.isaac.core.utils": "isaacsim.core.utils",
    "omni.isaac.core.utils.extensions": "isaacsim.core.utils.extensions",
    "omni.isaac.core.utils.numpy": "isaacsim.core.utils.numpy",
    "omni.isaac.core.utils.numpy.rotations": "isaacsim.core.utils.numpy.rotations",
    "omni.isaac.core.utils.prims": "isaacsim.core.utils.prims",
    "omni.isaac.core.utils.rotations": "isaacsim.core.utils.rotations",
    "omni.isaac.core.utils.stage": "isaacsim.core.utils.stage",
}


def _register_module_alias(old_name: str, new_name: str) -> None:
    module = importlib.import_module(new_name)
    sys.modules[old_name] = module
    parent_name, attribute = old_name.rsplit(".", 1)
    parent = sys.modules[parent_name]
    setattr(parent, attribute, module)


def _install_legacy_core_namespace() -> None:
    """Expose Isaac 6's deprecated Core API under InternUtopia's 4.x names."""

    import omni

    if "omni.isaac" not in sys.modules:
        package = types.ModuleType("omni.isaac")
        package.__path__ = []  # type: ignore[attr-defined]
        sys.modules["omni.isaac"] = package
        setattr(omni, "isaac", package)

    # Parents must precede their children so normal ``from ... import`` works.
    for old_name, new_name in _MODULE_ALIASES.items():
        _register_module_alias(old_name, new_name)

    # Isaac 4.x used the singular names below for one prim.  Isaac 6 keeps the
    # same API as explicit ``Single*`` wrappers and reuses the short names for
    # batched views.  Present a proxy module so Isaac 6 internals retain their
    # native batched classes while InternUtopia receives the old constructors.
    native_prims = importlib.import_module("isaacsim.core.prims")
    legacy_prims = types.ModuleType("omni.isaac.core.prims")
    legacy_prims.__dict__.update(native_prims.__dict__)
    legacy_prims.__name__ = "omni.isaac.core.prims"
    legacy_prims.__package__ = "omni.isaac.core"
    legacy_prims.__path__ = []  # type: ignore[attr-defined]
    legacy_prims.RigidPrim = native_prims.SingleRigidPrim
    legacy_prims.GeometryPrim = native_prims.SingleGeometryPrim
    legacy_prims.XFormPrim = native_prims.SingleXFormPrim
    sys.modules["omni.isaac.core.prims"] = legacy_prims
    sys.modules["omni.isaac.core"].prims = legacy_prims

    native_xform = importlib.import_module("isaacsim.core.prims.impl.xform_prim")
    legacy_xform = types.ModuleType("omni.isaac.core.prims.xform_prim")
    legacy_xform.__dict__.update(native_xform.__dict__)
    legacy_xform.__name__ = "omni.isaac.core.prims.xform_prim"
    legacy_xform.__package__ = "omni.isaac.core.prims"
    legacy_xform.XFormPrim = native_prims.SingleXFormPrim
    sys.modules["omni.isaac.core.prims.xform_prim"] = legacy_xform
    legacy_prims.xform_prim = legacy_xform


def _patch_runner_to_reuse_app(simulation_app: Any) -> None:
    """Keep the official runner but give it the already-started application."""

    from internutopia.core.runner import SimulatorRunner

    def setup_isaacsim(self: Any) -> None:
        simulator = self.config.simulator
        unsupported = {
            "headless": getattr(simulator, "headless", None) is not True,
            "native": bool(getattr(simulator, "native", False)),
            "webrtc": bool(getattr(simulator, "webrtc", False)),
            "multi_gpu": bool(getattr(simulator, "multi_gpu", False)),
        }
        if any(unsupported.values()):
            raise RuntimeError(
                "Isaac 6 compatibility bootstrap only permits the frozen "
                f"headless single-GPU config; observed deviations: {unsupported}"
            )

        self._simulation_app = simulation_app
        self._simulation_app._carb_settings.set(  # noqa: SLF001
            "/physics/cooking/ujitsoCollisionCooking", False
        )

    SimulatorRunner.setup_isaacsim = setup_isaacsim


def _install_simulation_manager_bridge() -> None:
    """Map the removed Isaac 5 reset hook to Isaac 6's public equivalent."""

    from isaacsim.core.simulation_manager import SimulationManager

    if not hasattr(SimulationManager, "_create_simulation_view"):

        def create_simulation_view(_event: Any) -> None:
            SimulationManager.initialize_physics()

        SimulationManager._create_simulation_view = staticmethod(  # type: ignore[attr-defined]
            create_simulation_view
        )


def main() -> None:
    from isaacsim import SimulationApp

    # These are the values used by InternUtopia 2.2.0 for the frozen upstream
    # headless, non-streaming, single-GPU simulator config.
    simulation_app = SimulationApp(
        {
            "headless": True,
            "anti_aliasing": 0,
            "hide_ui": False,
            "multi_gpu": False,
        }
    )

    _install_legacy_core_namespace()
    _install_simulation_manager_bridge()
    _patch_runner_to_reuse_app(simulation_app)

    internnav_root = Path(os.environ["INTERNNAV_ROOT"]).resolve()
    eval_script = internnav_root / "scripts" / "eval" / "eval.py"
    if not eval_script.is_file():
        raise FileNotFoundError(f"pinned upstream evaluator is missing: {eval_script}")

    sys.argv = [str(eval_script), *sys.argv[1:]]
    try:
        runpy.run_path(str(eval_script), run_name="__main__")
    except BaseException:
        # SimulationApp shutdown can otherwise consume the last buffered stderr
        # lines.  Preserve the original evaluator failure before closing Kit.
        traceback.print_exc()
        sys.stderr.flush()
        # Kit's graceful close replaces an in-flight Python exception with exit
        # code 0.  On failure, let the OS tear down this isolated process so the
        # launcher receives the evaluator's failure unambiguously.
        os._exit(1)
    if simulation_app.is_running():
        simulation_app.close()


if __name__ == "__main__":
    main()
