"""Fail-closed ROS container PYTHONPATH construction and validation."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path, PurePosixPath


ROS_JAZZY_SITE = PurePosixPath("/opt/ros/jazzy/lib/python3.12/site-packages")

# This is the exact PYTHONPATH emitted by the frozen Isaac ROS workspace setup
# on 2026-07-17.  Project build/install entries are recognized only so that a
# change in the setup environment fails closed; none is propagated to 01R.
# The model-free bridge and recorder use the verified control-root source plus
# standard Jazzy messages, as proven by the coordinator import diagnostic.
FROZEN_SETUP_PYTHONPATH = (
    "/workspaces/isaac/build/internvla_t4_sensors",
    "/workspaces/isaac/install/internvla_t4_sensors/lib/python3.12/site-packages",
    "/workspaces/isaac/build/internvla_t4_recovery",
    "/workspaces/isaac/install/internvla_t4_recovery/lib/python3.12/site-packages",
    "/workspaces/isaac/build/internvla_ros2",
    "/workspaces/isaac/install/internvla_ros2/lib/python3.12/site-packages",
    "/workspaces/isaac/build/internvla_nav2_adapter",
    "/workspaces/isaac/install/internvla_nav2_adapter/lib/python3.12/site-packages",
    "/workspaces/isaac/build/internvla_go2_controller",
    "/workspaces/isaac/install/internvla_go2_controller/lib/python3.12/site-packages",
    "/workspaces/isaac/install/internvla_ros2_msgs/lib/python3.12/site-packages",
    "/workspaces/isaac/build/go2_sensor_bridge",
    "/workspaces/isaac/install/go2_sensor_bridge/lib/python3.12/site-packages",
    str(ROS_JAZZY_SITE),
)
FROZEN_SETUP_PYTHONPATH_SHA256 = hashlib.sha256(
    ":".join(FROZEN_SETUP_PYTHONPATH).encode("utf-8")
).hexdigest()


def _split_exact(value: str, *, label: str) -> list[str]:
    entries = value.split(os.pathsep)
    if (
        not entries
        or any(not item for item in entries)
        or len(entries) != len(set(entries))
    ):
        raise RuntimeError(f"{label} contains empty or duplicate entries")
    return entries


def controlled_pythonpath_from_setup(control_root: Path, setup_value: str) -> list[str]:
    """Verify the complete setup output and select only source + standard ROS."""

    entries = _split_exact(setup_value, label="setup PYTHONPATH")
    if tuple(entries) != FROZEN_SETUP_PYTHONPATH:
        raise RuntimeError("setup PYTHONPATH differs from the frozen Isaac ROS release")
    return [str(control_root.resolve()), str(ROS_JAZZY_SITE)]


def require_frozen_setup_sha256(value: object) -> str:
    if value != FROZEN_SETUP_PYTHONPATH_SHA256:
        raise RuntimeError("setup PYTHONPATH SHA-256 does not match the frozen release")
    return FROZEN_SETUP_PYTHONPATH_SHA256


def validate_child_pythonpath(control_root: Path, value: str) -> list[str]:
    entries = _split_exact(value, label="child PYTHONPATH")
    expected = [str(control_root.resolve()), str(ROS_JAZZY_SITE)]
    if (
        Path(entries[0]).resolve(strict=False) != control_root.resolve()
        or entries[1:] != expected[1:]
    ):
        raise RuntimeError(
            "child PYTHONPATH must be exactly the verified control root and frozen ROS Jazzy site"
        )
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--setup-pythonpath", required=True)
    args = parser.parse_args(argv)
    try:
        controlled = controlled_pythonpath_from_setup(
            args.control_root, args.setup_pythonpath
        )
    except RuntimeError as exc:
        print(f"ROS PYTHONPATH policy rejected setup: {exc}", file=sys.stderr)
        return 2
    print(os.pathsep.join(controlled))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
