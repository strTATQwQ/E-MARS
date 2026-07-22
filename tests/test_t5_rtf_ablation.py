from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ISAAC = ROOT / "scripts/run_t5_distributed_isaac.sh"
FAST = ROOT / "coordination/run_t5_fast_lane_online.sh"


def test_navigation_fast_is_promoted_and_other_rtf_profiles_are_canary_only() -> None:
    isaac = ISAAC.read_text(encoding="utf-8")
    fast = FAST.read_text(encoding="utf-8")
    for profile in (
        "navigation_fast",
        "off",
        "baseline",
        "lidar_off_probe",
        "lidar_720",
        "depth_stride8",
        "sensor_2p5hz",
    ):
        assert profile in isaac
        assert profile in fast
    assert 'test "$execution_profile" = engineering_canary' in isaac
    assert 'test "$engineering_canary_sec" = 60' in isaac
    assert 'test "$profile" = canary60 || usage' in fast
    assert '${INTERNNAV_T5_RTF_ABLATION_PROFILE:-navigation_fast}' in isaac
    assert '${INTERNNAV_T5_RTF_ABLATION_PROFILE:-navigation_fast}' in fast
    assert 'navigation_fast|off) ;;' in fast
    assert "rtf_lidar_ray_count=720" in isaac
    assert 'not in {"off", "navigation_fast"}' in isaac
    assert 'INTERNNAV_T5_RTF_ABLATION_PROFILE="$rtf_ablation_profile"' in fast
    assert 'export INTERNVLA_T4_R3_LIDAR_RAY_COUNT="$rtf_lidar_ray_count"' in isaac
    assert 'export INTERNVLA_T4_DEPTH_STRIDE_OVERRIDE="$rtf_depth_stride"' in isaac
    assert 'export INTERNVLA_GO2_SENSOR_HZ="$rtf_sensor_hz"' in isaac
    assert '--output "$result_dir/rtf_ablation_summary.json"' in fast


def test_lidar_720_is_frozen_into_generated_runtime(tmp_path: Path) -> None:
    source = ROOT / "scripts/internnav_go2_runtime.py"
    output = tmp_path / "runtime.py"
    manifest = tmp_path / "manifest.json"
    env = os.environ.copy()
    env.update(
        INTERNVLA_T4_R3_ENABLE_LIDAR="1",
        INTERNVLA_T4_R3_LIDAR_RAY_COUNT="720",
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_t4_r3_sensor_runtime_overlay.py"),
            "--source",
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["frozen_geometry_sources"]["lidar_ray_count"] == 720
    assert payload["frozen_geometry_sources"]["lidar_width"] == 90
    assert "            width = 90" in output.read_text(encoding="utf-8")


def test_rtf_summary_reports_host_limit_and_spin_metrics(tmp_path: Path) -> None:
    (tmp_path / "health").mkdir()
    (tmp_path / "isaac_contract.json").write_text(
        json.dumps({"rtf_ablation": {"profile": "lidar_720"}}), encoding="utf-8"
    )
    telemetry = {
        "sampled_unix": 15.0,
        "cpu": {"aggregate_percent": 30.0, "per_core_percent": {"cpu0": 95.0}},
        "memory": {"ram_used_percent": 50.0, "ram_used_bytes": 100, "swap_used_bytes": 0},
        "gpu": {"utilization_gpu_percent": 40.0, "memory_used_mib": 1000.0,
                "memory_total_mib": 16000.0},
        "errors": [],
    }
    (tmp_path / "host_telemetry.jsonl").write_text(
        json.dumps(telemetry) + "\n", encoding="utf-8"
    )
    samples = [
        {"sampled_unix": 10.0, "clock": {"last_clock_ns": 1_000_000_000}},
        {"sampled_unix": 20.0, "clock": {"last_clock_ns": 4_000_000_000}},
    ]
    (tmp_path / "engineering_canary.json").write_text(
        json.dumps({"started_unix": 10.0, "finished_unix": 20.0}),
        encoding="utf-8",
    )
    (tmp_path / "health/engineering_canary_samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8"
    )
    (tmp_path / "kit_gpu_audit.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    (tmp_path / "health/runtime_gpu_evidence.json").write_text(
        json.dumps({"status": "PASS", "physics_resolved_gpu_uuid": "GPU-x",
                    "expected_gpu_uuid": "GPU-x",
                    "runtime_compute_process_uuid_observed": True,
                    "physx_runtime_gpu_execution_confirmed": True}),
        encoding="utf-8"
    )
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/evaluator_outer.log").write_text("RTX GPU active\n", encoding="utf-8")
    output = tmp_path / "rtf_ablation_summary.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_t5_rtf_ablation.py"),
         "--result-root", str(tmp_path), "--output", str(output)],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["engineering_window"]["rtf"] == 0.3
    assert payload["classification"] == "ISAAC_HOST_THROUGHPUT_LIMITED"
    assert payload["bottleneck"] == "CPU_PHYSICS_OR_SENSOR_GENERATION"
    assert payload["host"]["raw_sample_count"] == 1
    assert payload["host"]["sample_count"] == 1
    assert payload["status"] == "PASS"


def test_telemetry_sampler_is_owned_and_stopped_by_run() -> None:
    isaac = ISAAC.read_text(encoding="utf-8")
    assert 'telemetry_pid=""' in isaac
    assert 'record_pid_event host_telemetry "$telemetry_pid" started' in isaac
    assert 'stop_host_group host_telemetry "$telemetry_pid"' in isaac
    assert "sample_t5_host_telemetry.py" in isaac
    assert "rtf_ablation_summary.json" in isaac
    assert 'if test "$rtf_diagnostic_enabled" = 1; then' in isaac
