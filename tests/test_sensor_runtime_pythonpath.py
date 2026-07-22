from __future__ import annotations

import os
from pathlib import Path

import pytest

from sensor_runtime.pythonpath_policy import (
    FROZEN_SETUP_PYTHONPATH,
    FROZEN_SETUP_PYTHONPATH_SHA256,
    ROS_JAZZY_SITE,
    controlled_pythonpath_from_setup,
    main,
    require_frozen_setup_sha256,
    validate_child_pythonpath,
)


def _joined(entries: tuple[str, ...] | list[str]) -> str:
    return os.pathsep.join(entries)


def test_frozen_setup_selects_only_verified_source_and_standard_ros(
    tmp_path: Path,
) -> None:
    controlled = controlled_pythonpath_from_setup(
        tmp_path, _joined(FROZEN_SETUP_PYTHONPATH)
    )
    assert controlled == [str(tmp_path.resolve()), str(ROS_JAZZY_SITE)]
    assert all("/workspaces/isaac/build/" not in item for item in controlled)
    assert all("/workspaces/isaac/install/" not in item for item in controlled)
    assert validate_child_pythonpath(tmp_path, _joined(controlled)) == controlled
    assert len(FROZEN_SETUP_PYTHONPATH_SHA256) == 64
    assert (
        require_frozen_setup_sha256(FROZEN_SETUP_PYTHONPATH_SHA256)
        == FROZEN_SETUP_PYTHONPATH_SHA256
    )


@pytest.mark.parametrize("value", [None, "", "0" * 64, "A" * 64])
def test_setup_digest_mismatch_is_rejected(value: object) -> None:
    with pytest.raises(RuntimeError, match="frozen release"):
        require_frozen_setup_sha256(value)


@pytest.mark.parametrize(
    "entries",
    [
        FROZEN_SETUP_PYTHONPATH[:-1],
        tuple(reversed(FROZEN_SETUP_PYTHONPATH)),
        FROZEN_SETUP_PYTHONPATH
        + ("/workspaces/isaac/build/unreviewed_package",),
        FROZEN_SETUP_PYTHONPATH
        + ("/workspaces/isaac/install/unreviewed/lib/python3.12/site-packages",),
        FROZEN_SETUP_PYTHONPATH + (str(ROS_JAZZY_SITE),),
    ],
)
def test_setup_drift_is_rejected(entries: tuple[str, ...], tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        controlled_pythonpath_from_setup(tmp_path, _joined(entries))


@pytest.mark.parametrize(
    "extra",
    [
        "/workspaces/isaac/build/go2_sensor_bridge",
        "/workspaces/isaac/install/go2_sensor_bridge/lib/python3.12/site-packages",
        "/opt/ros/jazzy/unreviewed",
    ],
)
def test_child_pythonpath_rejects_every_extra_entry(
    extra: str, tmp_path: Path
) -> None:
    value = _joined([str(tmp_path), str(ROS_JAZZY_SITE), extra])
    with pytest.raises(RuntimeError, match="must be exactly"):
        validate_child_pythonpath(tmp_path, value)


def test_policy_cli_emits_controlled_value(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert (
        main(
            [
                "--control-root",
                str(tmp_path),
                "--setup-pythonpath",
                _joined(FROZEN_SETUP_PYTHONPATH),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == _joined(
        [str(tmp_path.resolve()), str(ROS_JAZZY_SITE)]
    )


def test_policy_cli_fails_closed_on_setup_drift(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert (
        main(
            [
                "--control-root",
                str(tmp_path),
                "--setup-pythonpath",
                _joined(FROZEN_SETUP_PYTHONPATH[:-1]),
            ]
        )
        == 2
    )
    assert "differs from the frozen" in capsys.readouterr().err


def test_ros_container_uses_isolated_policy_and_absolute_system_python() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "sensor_runtime/run_ros_container.sh"
    ).read_text(encoding="utf-8")
    assert "/usr/bin/python3 -I" in script
    assert "exec /usr/bin/setsid --wait /usr/bin/python3" in script
    assert "/workspaces/isaac/install/*" not in script
    assert "/workspaces/isaac/build/*" not in script
    for name in (
        "INTERNNAV_SESSION_PROFILE",
        "INTERNNAV_T4_MAP_COMPANION_MODULE",
        "INTERNNAV_T4_MAP_CONFIG_DIR",
        "INTERNNAV_T4_MAP_NVBLOX_MODE",
        "INTERNNAV_T4_MAP_NVBLOX_HEALTH",
        "INTERNNAV_SIMULATION_TARGET",
    ):
        assert f'-e "{name}=${{{name}:-}}"' in script
