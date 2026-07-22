"""Frozen, path-independent Isaac Sim 6 startup policy for Worker 01R."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


ISAAC_EXPERIENCE_FILENAME = "internnav_model_free_sensor.kit"
ISAAC_STARTUP_READY_FILENAME = "isaac_startup_ready.json"
FROZEN_EXPERIENCE_SHA256 = (
    "136665df0c7724ba1a863070ec25b81590973a3def62d6de7af06d67a55d7a9a"
)
FROZEN_DEPENDENCIES = (
    "isaacsim.core.api",
    "isaacsim.sensors.camera",
    "isaacsim.sensors.experimental.physics",
    "isaacsim.simulation_app",
    "omni.kit.loop-isaac",
)
FROZEN_EXTENSION_FOLDERS = ("extsDeprecated", "extscache")
FROZEN_ENABLED_EXTENSIONS = (
    "isaacsim.sensors.camera",
    "isaacsim.sensors.experimental.physics",
)
FROZEN_PACKAGE = {
    "title": "InternNav Model-Free Sensor",
    "description": "Minimal headless physics and camera experience for 01R",
    "version": "1.0.0",
}
FROZEN_SETTINGS: dict[str, Any] = {
    "app.name": "InternNav Model-Free Sensor",
    "app.version": "1.0.0",
    "app.vulkan": True,
    "app.fastShutdown": True,
    "app.enableDeveloperWarnings": False,
    "app.content.emptyStageOnStart": True,
    "app.file.ignoreUnsavedStage": True,
    "app.gatherRenderResults": True,
    "app.asyncRendering": False,
    "app.useFabricSceneDelegate": True,
    "app.hydraEngine.waitIdle": True,
    "app.updateOrder.checkForHydraRenderComplete": 1000,
    "app.settings.persistent": False,
    "app.settings.fabricDefaultStageFrameHistoryCount": 3,
    "app.runLoops.main.manualModeEnabled": True,
    "app.runLoops.main.rateLimitEnabled": False,
    "app.renderer.resolution.width": 640,
    "app.renderer.resolution.height": 480,
    "app.renderer.skipWhileMinimized": False,
    "app.renderer.sleepMsOnFocus": 0,
    "app.renderer.sleepMsOutOfFocus": 0,
    "renderer.asyncInit": True,
    "renderer.gpuEnumeration.glInterop.enabled": False,
    "omni.replicator.asyncRendering": False,
    "persistent.omni.replicator.captureOnPlay": True,
    "rtx.hydra.supportMultiTickRate": True,
    "rtx.rendering.perSensorTickTlas": True,
}
FROZEN_LAUNCH_BASE: dict[str, Any] = {
    "headless": True,
    "anti_aliasing": 0,
    "hide_ui": True,
    "multi_gpu": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(prefix: str, value: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(_flatten(name, item))
        else:
            flattened[name] = item
    return flattened


def experience_path() -> Path:
    return Path(__file__).with_name(ISAAC_EXPERIENCE_FILENAME)


def frozen_policy_evidence() -> dict[str, Any]:
    """Return the exact path-independent startup policy embedded in evidence."""

    return {
        "schema_version": 1,
        "experience": {
            "repo_relative_path": f"sensor_runtime/{ISAAC_EXPERIENCE_FILENAME}",
            "sha256": FROZEN_EXPERIENCE_SHA256,
            "package": dict(FROZEN_PACKAGE),
            "dependencies": list(FROZEN_DEPENDENCIES),
            "settings": dict(FROZEN_SETTINGS),
        },
        "extension_folders": [
            {
                "name": name,
                "is_directory": True,
                "resolved_within_isaacsim_package": True,
            }
            for name in FROZEN_EXTENSION_FOLDERS
        ],
        "enabled_extensions": list(FROZEN_ENABLED_EXTENSIONS),
        "launch_config": {
            **FROZEN_LAUNCH_BASE,
            "extra_args": [
                token
                for name in FROZEN_EXTENSION_FOLDERS
                for token in ("--ext-folder", name)
            ],
        },
    }


def validate_experience(path: Path | None = None) -> dict[str, Any]:
    # The outer host/source-provenance process only validates already-written
    # startup evidence and may run Python 3.10.  TOML parsing is needed solely
    # inside the Isaac 6 worker, whose frozen runtime is Python 3.12.
    try:
        import tomllib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "frozen Isaac experience parsing requires the Isaac Python runtime"
        ) from exc
    selected = experience_path() if path is None else path
    if not selected.is_file():
        raise RuntimeError(f"frozen Isaac experience is missing: {selected}")
    observed_hash = _sha256(selected)
    if observed_hash != FROZEN_EXPERIENCE_SHA256:
        raise RuntimeError("frozen Isaac experience SHA-256 drifted")
    try:
        payload = tomllib.loads(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError("frozen Isaac experience is unreadable") from exc
    if set(payload) != {"package", "dependencies", "settings"}:
        raise RuntimeError("frozen Isaac experience top-level schema drifted")
    if payload.get("package") != FROZEN_PACKAGE:
        raise RuntimeError("frozen Isaac experience package metadata drifted")
    dependencies = payload.get("dependencies")
    if (
        not isinstance(dependencies, dict)
        or tuple(dependencies) != FROZEN_DEPENDENCIES
        or any(value != {} for value in dependencies.values())
    ):
        raise RuntimeError("frozen Isaac experience dependency set drifted")
    settings = payload.get("settings")
    if not isinstance(settings, dict) or _flatten("", settings) != FROZEN_SETTINGS:
        raise RuntimeError("frozen Isaac experience settings drifted")
    return frozen_policy_evidence()["experience"]


def resolve_extension_folders(package_root: Path) -> list[Path]:
    try:
        root = package_root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("Isaac package root is missing") from exc
    if not root.is_dir():
        raise RuntimeError("Isaac package root is not a directory")
    resolved: list[Path] = []
    for name in FROZEN_EXTENSION_FOLDERS:
        candidate = root / name
        try:
            target = candidate.resolve(strict=True)
            target.relative_to(root)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Isaac extension folder {name} is missing or escapes the package root"
            ) from exc
        if not target.is_dir():
            raise RuntimeError(f"Isaac extension folder {name} is not a directory")
        resolved.append(target)
    return resolved


def require_frozen_launch_config(
    config: Mapping[str, Any], package_root: Path
) -> dict[str, Any]:
    folders = resolve_extension_folders(package_root)
    expected: dict[str, Any] = {
        **FROZEN_LAUNCH_BASE,
        "extra_args": [
            token
            for folder in folders
            for token in ("--ext-folder", str(folder))
        ],
    }
    if dict(config) != expected:
        raise RuntimeError("Isaac SimulationApp launch config drifted")
    return frozen_policy_evidence()["launch_config"]


def prepare_frozen_experience(
    isaacsim_module: ModuleType,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    selected = experience_path().resolve(strict=True)
    validate_experience(selected)
    module_file = getattr(isaacsim_module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError("isaacsim package does not expose a filesystem root")
    package_root = Path(module_file).resolve(strict=True).parent
    folders = resolve_extension_folders(package_root)
    config: dict[str, Any] = {
        **FROZEN_LAUNCH_BASE,
        "extra_args": [
            token
            for folder in folders
            for token in ("--ext-folder", str(folder))
        ],
    }
    require_frozen_launch_config(config, package_root)
    return selected, config, frozen_policy_evidence()


def build_startup_ready(
    *,
    policy: Mapping[str, Any],
    simulation_app_elapsed_sec: float,
    extensions_ready_elapsed_sec: float,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": "READY",
        "simulation_app_elapsed_sec": float(simulation_app_elapsed_sec),
        "extensions_ready_elapsed_sec": float(extensions_ready_elapsed_sec),
        "policy": dict(policy),
    }
    return require_frozen_startup_ready(payload)


def require_frozen_startup_ready(payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != {
        "schema_version",
        "status",
        "simulation_app_elapsed_sec",
        "extensions_ready_elapsed_sec",
        "policy",
    }:
        raise RuntimeError("Isaac startup-ready artifact schema drifted")
    if payload.get("schema_version") != 1 or payload.get("status") != "READY":
        raise RuntimeError("Isaac startup-ready status is invalid")
    try:
        app_elapsed = float(payload["simulation_app_elapsed_sec"])
        extensions_elapsed = float(payload["extensions_ready_elapsed_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Isaac startup-ready timing is invalid") from exc
    if (
        not math.isfinite(app_elapsed)
        or not math.isfinite(extensions_elapsed)
        or app_elapsed < 0.0
        or extensions_elapsed < app_elapsed
        or extensions_elapsed >= 20.0
    ):
        raise RuntimeError("Isaac startup-ready timing missed the frozen 20 second bound")
    if payload.get("policy") != frozen_policy_evidence():
        raise RuntimeError("Isaac startup-ready policy evidence drifted")
    return {
        "schema_version": 1,
        "status": "READY",
        "simulation_app_elapsed_sec": app_elapsed,
        "extensions_ready_elapsed_sec": extensions_elapsed,
        "policy": frozen_policy_evidence(),
    }
