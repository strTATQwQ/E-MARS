#!/usr/bin/env python3
"""Parameterize the frozen T3 runtime for the calibrated T4 RGB-D stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one runtime token, found {count}: {old!r}")
    return text.replace(old, new)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    text = replace_once(
        text,
        "import math\nimport json",
        "import base64\nimport math\nimport json",
    )
    text = replace_once(
        text,
        "import time\nfrom pathlib import Path",
        "import time\nimport zlib\nfrom pathlib import Path",
    )
    text = replace_once(
        text,
        "            self.joint_subset = ArticulationSubset(robot.articulation, JOINT_NAMES)\n"
        "            self.ipc = ControllerIPCClient(\n"
        "                os.environ.get(\n"
        "                    \"INTERNVLA_GO2_CONTROLLER_SOCKET\",\n"
        "                    \"/tmp/internvla_go2_controller.sock\",\n"
        "                ),\n"
        "                timeout_sec=0.25,\n"
        "            )",
        "            self.joint_subset = ArticulationSubset(robot.articulation, JOINT_NAMES)\n"
        "            controller_ipc_timeout_sec = float(\n"
        "                os.environ.get(\"INTERNVLA_T4_CONTROLLER_IPC_TIMEOUT_SEC\", \"5.0\")\n"
        "            )\n"
        "            if not 0.25 <= controller_ipc_timeout_sec <= 5.0:\n"
        "                raise RuntimeError(\"invalid bounded T4 controller IPC timeout\")\n"
        "            self.ipc = ControllerIPCClient(\n"
        "                os.environ.get(\n"
        "                    \"INTERNVLA_GO2_CONTROLLER_ENDPOINT\",\n"
        "                    os.environ.get(\n"
        "                        \"INTERNVLA_GO2_CONTROLLER_SOCKET\",\n"
        "                        \"/tmp/internvla_go2_controller.sock\",\n"
        "                    ),\n"
        "                ),\n"
        "                timeout_sec=controller_ipc_timeout_sec,\n"
        "            )",
    )
    text = replace_once(
        text,
        '            sensor = self.robot.sensors.get("pano_camera_0")',
        '            sensor = self.robot.sensors.get("t4_d435i_depth")',
    )
    text = replace_once(
        text,
        "    from internnav.configs.evaluator import ControllerCfg\n",
        "    from internnav.configs.evaluator import ControllerCfg, SensorCfg\n"
        "    from internnav.env.utils.internutopia_extension.configs.sensors import VLNCameraCfg\n",
    )
    text = replace_once(
        text,
        "            row_stride = 10\n            column_stride = 10",
        "            row_stride = int(os.environ.get(\"INTERNVLA_T4_DEPTH_STRIDE\", \"4\"))\n"
        "            column_stride = row_stride\n"
        "            if row_stride not in (1, 2, 4, 5, 8, 10):\n"
        "                raise RuntimeError(\"unsupported T4 metric-depth stride\")",
    )
    text = replace_once(
        text,
        '            return {\n                "depth_height": int(sample.shape[0]),',
        '            depth_uint16 = np.rint(\n'
        '                np.clip(sample, 0.0, 65.535) * 1000.0\n'
        '            ).astype("<u2", copy=False)\n'
        '            depth_raw = depth_uint16.tobytes(order="C")\n'
        '            depth_compressed = zlib.compress(depth_raw, level=1)\n'
        '            depth_encoded = base64.b64encode(depth_compressed).decode("ascii")\n'
        '            if len(depth_encoded) > 480 * 1024:\n'
        '                raise RuntimeError("compressed T4 depth payload exceeds bound")\n'
        '            return {\n'
        '                "depth_height": int(sample.shape[0]),',
    )
    text = replace_once(
        text,
        '                "depth_values": sample.reshape(-1).astype(float).tolist(),',
        '                "depth_encoding": "uint16_mm_zlib_b64_v1",\n'
        '                "depth_zlib_b64": depth_encoded,\n'
        '                "depth_uncompressed_bytes": len(depth_raw),\n'
        '                "depth_compressed_bytes": len(depth_compressed),\n'
        '                "depth_capture_wall_time_unix": time.time(),',
    )
    text = replace_once(
        text,
        "                                - 0.2\n                            )",
        "                                - float(os.environ.get(\"INTERNVLA_T4_DEPTH_FORWARD_M\", \"0.20\"))\n"
        "                            )",
    )
    text = replace_once(
        text,
        "                            relative_z = base_z + sz * half_sizes[2] - 0.2\n"
        "                            camera_x = -relative_y\n"
        "                            camera_y = 0.5 * relative_x + 0.8660254037844386 * relative_z\n"
        "                            view_depth = (\n"
        "                                0.8660254037844386 * relative_x - 0.5 * relative_z\n"
        "                            )\n"
        "                            if (\n"
        "                                0.1 < view_depth <= 6.0\n"
        "                                and abs(camera_x / view_depth) <= 320.0 / 585.0\n"
        "                                and abs(camera_y / view_depth) <= 240.0 / 585.0\n"
        "                            ):",
        "                            relative_z = (\n"
        "                                base_z + sz * half_sizes[2]\n"
        "                                - (float(os.environ.get(\"INTERNVLA_T4_DEPTH_HEIGHT_M\", \"0.62\")) - 0.42)\n"
        "                            )\n"
        "                            camera_pitch = math.radians(float(os.environ.get(\"INTERNVLA_T4_DEPTH_PITCH_DOWN_DEG\", \"20\")))\n"
        "                            camera_sine = math.sin(camera_pitch)\n"
        "                            camera_cosine = math.cos(camera_pitch)\n"
        "                            camera_x = -relative_y\n"
        "                            camera_y = camera_sine * relative_x + camera_cosine * relative_z\n"
        "                            view_depth = camera_cosine * relative_x - camera_sine * relative_z\n"
        "                            horizontal_limit = math.tan(math.radians(float(os.environ.get(\"INTERNVLA_T4_DEPTH_HFOV_DEG\", \"87\"))) / 2.0)\n"
        "                            vertical_limit = math.tan(math.radians(float(os.environ.get(\"INTERNVLA_T4_DEPTH_VFOV_DEG\", \"58\"))) / 2.0)\n"
        "                            if (\n"
        "                                0.1 < view_depth <= 6.0\n"
        "                                and abs(camera_x / view_depth) <= horizontal_limit\n"
        "                                and abs(camera_y / view_depth) <= vertical_limit\n"
        "                            ):",
    )
    text = replace_once(
        text,
        '                    "camera_translation_from_base": [0.2, 0.0, 0.2],',
        '                    "camera_translation_from_base": [\n'
        '                        float(os.environ.get("INTERNVLA_T4_DEPTH_FORWARD_M", "0.20")),\n'
        '                        0.0,\n'
        '                        float(os.environ.get("INTERNVLA_T4_DEPTH_HEIGHT_M", "0.62")) - 0.42,\n'
        '                    ],',
    )
    text = replace_once(
        text,
        '                    "nominal_camera_world_height": float(position[2]) + 0.2,',
        '                    "nominal_camera_world_height": (\n'
        '                        float(position[2])\n'
        '                        + float(os.environ.get("INTERNVLA_T4_DEPTH_HEIGHT_M", "0.62"))\n'
        '                        - 0.42\n'
        '                    ),',
    )
    text = replace_once(
        text,
        "        def _update_command(self, *, state_only: bool = False) -> None:\n",
        "        def _sample_stereo(self) -> dict[str, Any]:\n"
        "            if os.environ.get(\"INTERNVLA_T4_ENABLE_STEREO_ODOMETRY\", \"0\") != \"1\":\n"
        "                return {}\n"
        "            if self.control_update_index % self.depth_control_interval:\n"
        "                return {}\n"
        "            payload: dict[str, Any] = {\"stereo_width\": 320, \"stereo_height\": 240}\n"
        "            for side in (\"left\", \"right\"):\n"
        "                sensor = self.robot.sensors.get(f\"t4_stereo_{side}\")\n"
        "                if sensor is None:\n"
        "                    if not getattr(self, \"_t4_stereo_diagnostic_emitted\", False):\n"
        "                        self._t4_stereo_diagnostic_emitted = True\n"
        "                        print(\"INTERNNAV_T4_STEREO_DIAGNOSTIC \" + json.dumps({\n"
        "                            \"reason\": \"sensor_missing\",\n"
        "                            \"side\": side,\n"
        "                            \"sensor_keys\": sorted(str(key) for key in self.robot.sensors.keys()),\n"
        "                        }, sort_keys=True), flush=True)\n"
        "                    return {}\n"
        "                sensor_data = sensor.get_data()\n"
        "                rgba = np.asarray(sensor_data.get(\"rgba\"))\n"
        "                if (\n"
        "                    rgba.ndim != 3\n"
        "                    or tuple(rgba.shape[:2]) != (240, 320)\n"
        "                    or rgba.shape[2] < 3\n"
        "                ):\n"
        "                    if not getattr(self, \"_t4_stereo_diagnostic_emitted\", False):\n"
        "                        self._t4_stereo_diagnostic_emitted = True\n"
        "                        print(\"INTERNNAV_T4_STEREO_DIAGNOSTIC \" + json.dumps({\n"
        "                            \"reason\": \"invalid_rgba\",\n"
        "                            \"side\": side,\n"
        "                            \"data_keys\": sorted(str(key) for key in sensor_data.keys()),\n"
        "                            \"shape\": list(rgba.shape),\n"
        "                            \"dtype\": str(rgba.dtype),\n"
        "                        }, sort_keys=True), flush=True)\n"
        "                    return {}\n"
        "                rgb = np.ascontiguousarray(rgba[:, :, :3].astype(np.uint8))\n"
        "                gray = np.clip(\n"
        "                    0.299 * rgb[:, :, 0]\n"
        "                    + 0.587 * rgb[:, :, 1]\n"
        "                    + 0.114 * rgb[:, :, 2],\n"
        "                    0,\n"
        "                    255,\n"
        "                ).astype(np.uint8)\n"
        "                compressed = zlib.compress(gray.tobytes(order=\"C\"), level=1)\n"
        "                payload[f\"stereo_{side}_zlib_b64\"] = base64.b64encode(compressed).decode(\"ascii\")\n"
        "            return payload\n"
        "\n"
        "        def _update_command(self, *, state_only: bool = False) -> None:\n",
    )
    text = replace_once(
        text,
        "            request.update(self._sample_depth())\n            try:\n",
        "            if not state_only:\n"
        "                request.update(self._sample_depth())\n"
        "                request.update(self._sample_stereo())\n"
        "            try:\n",
    )
    text = replace_once(
        text,
        "        robot.sensors = [\n"
        "            sensor\n"
        "            for sensor in robot.sensors\n"
        "            if sensor.sensor_settings.get(\"name\") != \"tp_pointcloud\"\n"
        "        ]\n",
        "        robot.sensors = [\n"
        "            sensor\n"
        "            for sensor in robot.sensors\n"
        "            if sensor.sensor_settings.get(\"name\") != \"tp_pointcloud\"\n"
        "        ]\n"
        "        robot.sensors.append(\n"
        "            SensorCfg(\n"
        "                sensor_settings=VLNCameraCfg(\n"
        "                    name=\"t4_d435i_depth\",\n"
        "                    prim_path=\"base/t4_d435i_depth\",\n"
        "                    enable=True,\n"
        "                    resolution=[640, 480],\n"
        "                ).model_dump(),\n"
        "            )\n"
        "        )\n"
        "        if os.environ.get(\"INTERNVLA_T4_ENABLE_STEREO_ODOMETRY\", \"0\") == \"1\":\n"
        "            for side in (\"left\", \"right\"):\n"
        "                robot.sensors.append(\n"
        "                    SensorCfg(\n"
        "                        sensor_settings=VLNCameraCfg(\n"
        "                            name=f\"t4_stereo_{side}\",\n"
        "                            prim_path=f\"base/t4_stereo_{side}\",\n"
        "                            enable=True,\n"
        "                            resolution=[320, 240],\n"
        "                        ).model_dump(),\n"
        "                    )\n"
        "                )\n",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "source_sha256": sha256(source),
        "output_sha256": sha256(output),
        "changes": [
            "parameterized_depth_stride",
            "dedicated_d435i_depth_sensor",
            "parameterized_depth_extrinsics",
            "parameterized_d435i_depth_frustum",
            "parameterized_runtime_audit",
            "optional_calibrated_stereo_payload",
            "bounded_stereo_sensor_diagnostic_and_rgb_rgba_acceptance",
            "bounded_uint16_depth_ipc",
            "bounded_completion_controller_ipc_timeout",
            "state_only_sensor_fast_path",
            "dgx_onboard_tcp_controller_endpoint",
        ],
        "motion_control_semantics_changed": False,
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
