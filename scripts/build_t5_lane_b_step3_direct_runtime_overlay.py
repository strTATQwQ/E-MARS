#!/usr/bin/env python3
"""Add the Lane-B-only Step3 direct observation capture hook to the R3 overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected one Step3 direct runtime token, found {count}: {old!r}"
        )
    return text.replace(old, new)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if not (
        os.environ.get("INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL") == "1"
        and os.environ.get("INTERNNAV_T5_LANE") == "b"
        and os.environ.get("INTERNNAV_T5_LANE_NAMESPACE") == "/t5/lane_b"
        and os.environ.get("INTERNNAV_T5_ID_PREFIX") == "b::"
    ):
        raise RuntimeError("Step3 direct runtime overlay is restricted to T5 Lane B")

    builder = Path(__file__).with_name("build_t4_r3_sensor_runtime_overlay.py")
    with tempfile.TemporaryDirectory(prefix="t5_step3_direct_runtime_") as temporary:
        base_output = Path(temporary) / "r3_overlay.py"
        base_manifest = Path(temporary) / "r3_manifest.json"
        subprocess.run(
            [
                sys.executable,
                str(builder),
                "--source",
                str(args.source.resolve()),
                "--output",
                str(base_output),
                "--manifest",
                str(base_manifest),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        text = base_output.read_text(encoding="utf-8")
        inherited = json.loads(base_manifest.read_text(encoding="utf-8"))
        inherited_sha256 = _sha256(base_output)

    text = _replace_once(
        text,
        "    original_get_rgb_depth = VLNEvalTask.get_rgb_depth\n\n"
        "    def get_rgb_depth_with_source_metadata(self: Any) -> dict[str, Any]:\n"
        "        observation = original_get_rgb_depth(self)\n"
        "        return _attach_t5_camera_source_metadata(observation)\n",
        "    original_get_rgb_depth = VLNEvalTask.get_rgb_depth\n"
        "    _revc_observation_capture_hook: dict[str, Any] = {\"callable\": None}\n\n"
        "    def get_rgb_depth_with_source_metadata(self: Any) -> dict[str, Any]:\n"
        "        observation = _attach_t5_camera_source_metadata(\n"
        "            original_get_rgb_depth(self)\n"
        "        )\n"
        "        capture = _revc_observation_capture_hook.get(\"callable\")\n"
        "        if capture is None:\n"
        "            raise RuntimeError(\"Step3 direct Rev-C capture hook is unavailable\")\n"
        "        from internvla_go2_controller.runtime import get_execution_identity\n\n"
        "        identity = get_execution_identity()\n"
        "        decision_sequence = identity.sequence_id + (0 if identity.stop else 1)\n"
        "        result = capture(\n"
        "            episode_id=identity.episode_id,\n"
        "            reset_generation=identity.reset_generation,\n"
        "            sequence_id=decision_sequence,\n"
        "            sim_stamp_ns=globals().get(\"_T5_SIM_CLOCK_NS\"),\n"
        "            state_only=False,\n"
        "        )\n"
        "        if result.get(\"revc_snapshot_status\") != \"CAPTURED\":\n"
        "            raise RuntimeError(\n"
        "                \"Step3 direct current-observation Rev-C capture failed: \"\n"
        "                + str(result.get(\"revc_snapshot_status\", \"UNKNOWN\"))\n"
        "            )\n"
        "        observation[\"step3_direct_revc_capture\"] = result\n"
        "        return observation\n",
    )
    text = _replace_once(
        text,
        "            self.obstacle_targets: list[list[float]] = []\n"
        "            self.bootstrap_generation = -1\n"
        "            self._imu_previous_normal_sample = None\n"
        "            self._revc_last_request_id = \"\"\n"
        "            self._revc_last_preview_monotonic = -math.inf\n",
        "            self.obstacle_targets: list[list[float]] = []\n"
        "            self.bootstrap_generation = -1\n"
        "            self._imu_previous_normal_sample = None\n"
        "            self._revc_last_request_id = \"\"\n"
        "            self._revc_last_preview_monotonic = -math.inf\n"
        "            _revc_observation_capture_hook[\"callable\"] = (\n"
        "                self._sample_revc_snapshot\n"
        "            )\n",
    )
    text = _replace_once(
        text,
        "            request.update(\n"
        "                self._sample_revc_snapshot(\n"
        "                    episode_id=episode_id,\n"
        "                    reset_generation=reset_generation,\n"
        "                    sequence_id=sequence_id,\n"
        "                    sim_stamp_ns=t5_revc_sim_stamp_ns,\n"
        "                    state_only=state_only,\n"
        "                )\n"
        "            )\n",
        "            # Direct capture already ran at the observation sampling boundary.\n",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="\n")
    compile(text, str(args.output), "exec")
    manifest = {
        **inherited,
        "schema_version": max(3, int(inherited.get("schema_version", 1))),
        "status": "PASS",
        "output_sha256": _sha256(args.output),
        "step3_direct_high_level": {
            "enabled": True,
            "lane": "b",
            "identity_prefix": "b::",
            "capture_phase": "observation_sampling_before_agent_step",
            "inherited_r3_overlay_sha256": inherited_sha256,
            "internvla_loaded": False,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
