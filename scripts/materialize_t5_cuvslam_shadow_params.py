#!/usr/bin/env python3
"""Compose the reviewed T5 cuVSLAM bridge overlay onto Lane-A Nav2 params."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


BRIDGE_KEY = "internvla_go2_controller_bridge"
OVERLAY_BRIDGE_KEY = "/**/internvla_go2_controller_bridge"
EXPECTED_OVERLAY = {
    "enable_stereo_feed": True,
    "pose_source": "ground_truth",
    "use_sim_time": True,
}


def _mapping(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("{} must contain a YAML mapping".format(path))
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(base_path: Path, overlay_path: Path, output_path: Path) -> Dict[str, Any]:
    base_path = base_path.resolve()
    overlay_path = overlay_path.resolve()
    output_path = output_path.resolve()
    base = _mapping(base_path)
    overlay = _mapping(overlay_path)
    overlay_bridge = overlay.get(OVERLAY_BRIDGE_KEY)
    if not isinstance(overlay_bridge, dict):
        raise ValueError("cuVSLAM overlay bridge block is missing")
    overlay_params = overlay_bridge.get("ros__parameters")
    if overlay_params != EXPECTED_OVERLAY:
        raise ValueError("cuVSLAM overlay bridge parameters drifted")
    base_bridge = base.get(BRIDGE_KEY)
    if not isinstance(base_bridge, dict):
        raise ValueError("base Nav2 bridge block is missing")
    base_params = base_bridge.get("ros__parameters")
    if not isinstance(base_params, dict):
        raise ValueError("base Nav2 bridge parameters are missing")
    # Keep the offline/legacy block aligned.  The T5 namespaced bridge receives
    # enable_stereo_feed through an explicit controller argv override in
    # run_t4_dgx_onboard.sh.  A wildcard selector cannot live in this shared
    # file because Nav2's RewrittenYaml prefixes every selector with the Lane
    # namespace and would turn /**/ into an invalid repeated-slash name.
    base_params.update(EXPECTED_OVERLAY)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite cuVSLAM parameter evidence")
    output_path.write_text(
        yaml.safe_dump(base, sort_keys=False), encoding="utf-8"
    )
    payload = {
        "schema_version": 1,
        "status": "READY",
        "mode": "cuvslam_shadow_gt_authority",
        "base": {"path": str(base_path), "sha256": _sha(base_path)},
        "overlay": {"path": str(overlay_path), "sha256": _sha(overlay_path)},
        "output": {"path": str(output_path), "sha256": _sha(output_path)},
        "effective_bridge_parameters": dict(EXPECTED_OVERLAY),
        "namespaced_bridge_parameter_delivery": "onboard_explicit_argv",
        "navigation_pose_authority": "ground_truth",
        "cuvslam_has_navigation_authority": False,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = materialize(args.base, args.overlay, args.output)
    args.receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
