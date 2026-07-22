#!/usr/bin/env python3
"""Generate an instrumented T4 runtime from the frozen T3 source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MARKER = """            return {
                \"depth_height\": int(sample.shape[0]),"""

INSTRUMENTATION = """            camera_audit_path = os.environ.get(\"INTERNVLA_T4_CAMERA_AUDIT\", \"\")
            if camera_audit_path:
                audit_valid = (sample > 0.1) & (sample <= 6.0)
                audit_near = audit_valid & (sample <= 0.45)
                base_roll, base_pitch = _tilt(base_rotation)
                audit_record = {
                    \"schema_version\": 1,
                    \"wall_time_unix\": time.time(),
                    \"control_update_index\": self.control_update_index,
                    \"valid_sample_count\": int(np.count_nonzero(audit_valid)),
                    \"near_field_sample_count\": int(np.count_nonzero(audit_near)),
                    \"near_field_fraction\": (
                        float(np.count_nonzero(audit_near))
                        / float(np.count_nonzero(audit_valid))
                        if np.count_nonzero(audit_valid)
                        else 0.0
                    ),
                    \"near_field_definition\": \"valid rendered depth <= 0.45 m; conservative body-occlusion proxy\",
                    \"base_roll_rad\": float(base_roll),
                    \"base_pitch_rad\": float(base_pitch),
                    \"camera_height_m\": float(os.environ.get(\"INTERNVLA_T4_CAMERA_HEIGHT_M\", \"0.62\")),
                    \"camera_pitch_down_deg\": float(os.environ.get(\"INTERNVLA_T4_CAMERA_PITCH_DOWN_DEG\", \"30\")),
                    \"camera_hfov_deg\": float(os.environ.get(\"INTERNVLA_T4_CAMERA_HFOV_DEG\", \"90\")),
                    \"camera_vfov_deg\": (
                        float(os.environ[\"INTERNVLA_T4_CAMERA_VFOV_DEG\"])
                        if os.environ.get(\"INTERNVLA_T4_CAMERA_VFOV_DEG\")
                        else None
                    ),
                    \"camera_model\": os.environ.get(\"INTERNVLA_T4_CAMERA_MODEL\", \"generic_rgbd\"),
                }
                camera_audit_target = Path(camera_audit_path)
                camera_audit_target.parent.mkdir(parents=True, exist_ok=True)
                with camera_audit_target.open(\"a\", encoding=\"utf-8\", newline=\"\\n\") as camera_audit_stream:
                    camera_audit_stream.write(json.dumps(audit_record, sort_keys=True) + \"\\n\")
            return {
                \"depth_height\": int(sample.shape[0]),"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    text = source.read_text(encoding="utf-8")
    if text.count(MARKER) != 1:
        raise RuntimeError("frozen T3 runtime instrumentation marker is not unique")
    generated = text.replace(MARKER, INSTRUMENTATION)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated, encoding="utf-8", newline="\n")
    payload = {
        "schema_version": 1,
        "source": "scripts/internnav_go2_runtime.py",
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_sha256": hashlib.sha256(generated.encode("utf-8")).hexdigest(),
        "patch": "T4 camera near-field and base-tilt audit only",
        "control_semantics_changed": False,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
