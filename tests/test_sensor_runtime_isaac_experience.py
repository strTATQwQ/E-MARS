from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

import sensor_runtime.isaac_experience as isaac_experience_module
from sensor_runtime.isaac_experience import (
    FROZEN_DEPENDENCIES,
    FROZEN_EXPERIENCE_SHA256,
    FROZEN_EXTENSION_FOLDERS,
    FROZEN_SETTINGS,
    build_startup_ready,
    experience_path,
    frozen_policy_evidence,
    prepare_frozen_experience,
    require_frozen_launch_config,
    require_frozen_startup_ready,
    resolve_extension_folders,
    validate_experience,
)


def _fake_isaacsim(tmp_path: Path) -> tuple[ModuleType, Path]:
    root = tmp_path / "isaacsim"
    root.mkdir()
    module_file = root / "__init__.py"
    module_file.write_text("\n", encoding="utf-8")
    for name in FROZEN_EXTENSION_FOLDERS:
        (root / name).mkdir()
    module = ModuleType("isaacsim")
    module.__file__ = str(module_file)
    return module, root


def test_experience_hash_dependencies_and_settings_are_exact() -> None:
    assert "tomllib" not in vars(isaac_experience_module)
    evidence = validate_experience()
    assert evidence["sha256"] == FROZEN_EXPERIENCE_SHA256
    assert evidence["dependencies"] == list(FROZEN_DEPENDENCIES)
    assert evidence["settings"] == FROZEN_SETTINGS


def test_experience_hash_drift_is_rejected(tmp_path: Path) -> None:
    mutated = tmp_path / "mutated.kit"
    mutated.write_bytes(experience_path().read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="SHA-256 drifted"):
        validate_experience(mutated)


def test_package_extension_folders_and_launch_config_are_fail_closed(
    tmp_path: Path,
) -> None:
    module, root = _fake_isaacsim(tmp_path)
    selected, config, evidence = prepare_frozen_experience(module)
    assert selected == experience_path().resolve()
    assert resolve_extension_folders(root) == [
        (root / name).resolve() for name in FROZEN_EXTENSION_FOLDERS
    ]
    assert require_frozen_launch_config(config, root) == evidence["launch_config"]
    assert str(root.resolve()) not in str(evidence)

    mutated = dict(config)
    mutated["hide_ui"] = False
    with pytest.raises(RuntimeError, match="launch config drifted"):
        require_frozen_launch_config(mutated, root)

    (root / FROZEN_EXTENSION_FOLDERS[-1]).rmdir()
    with pytest.raises(RuntimeError, match="missing or escapes"):
        resolve_extension_folders(root)


def test_extension_symlink_escape_is_rejected_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "isaacsim"
    root.mkdir()
    (root / FROZEN_EXTENSION_FOLDERS[0]).mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / FROZEN_EXTENSION_FOLDERS[1]).symlink_to(
            outside, target_is_directory=True
        )
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(RuntimeError, match="missing or escapes"):
        resolve_extension_folders(root)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "status", "config", "app_time", "extension_time"],
)
def test_startup_ready_artifact_rejects_every_policy_or_timing_drift(
    mutation: str,
) -> None:
    payload = build_startup_ready(
        policy=frozen_policy_evidence(),
        simulation_app_elapsed_sec=11.0,
        extensions_ready_elapsed_sec=11.2,
    )
    changed = deepcopy(payload)
    if mutation == "missing":
        changed.pop("policy")
    elif mutation == "status":
        changed["status"] = "STALE"
    elif mutation == "config":
        changed["policy"]["launch_config"]["multi_gpu"] = True
    elif mutation == "app_time":
        changed["simulation_app_elapsed_sec"] = 12.0
        changed["extensions_ready_elapsed_sec"] = 11.2
    else:
        changed["extensions_ready_elapsed_sec"] = 20.0
    with pytest.raises(RuntimeError):
        require_frozen_startup_ready(changed)
