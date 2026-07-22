from __future__ import annotations

import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "internnav_t5" / "go2_continuous_completion_cfg.py"
RUNNER = ROOT / "scripts" / "run_t5_distributed_isaac.sh"
PREFLIGHT = ROOT / "scripts" / "preflight_go2_continuous.py"


def _write_source(root: Path, name: str, marker: str) -> None:
    path = root / "configs" / "internnav_t3" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "from types import SimpleNamespace\n"
        f"eval_cfg = SimpleNamespace(marker={marker!r}, "
        "env=SimpleNamespace(env_settings={}))\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("phase", "source_name", "marker"),
    [
        ("canary", "go2_continuous_active_cfg.py", "active"),
        ("pilot", "go2_continuous_active_cfg.py", "active"),
        ("continuous_oracle", "go2_continuous_oracle_cfg.py", "oracle"),
    ],
)
def test_t5_config_applies_navigation_only_rate_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    source_name: str,
    marker: str,
) -> None:
    _write_source(tmp_path, source_name, marker)
    monkeypatch.setenv("INTERNNAV_T1_CONTROL_ROOT", str(tmp_path))
    monkeypatch.setenv("INTERNVLA_T3_PHASE", phase)

    config = runpy.run_path(str(CONFIG))["eval_cfg"]

    assert config.marker == marker
    assert config.env.env_settings == {
        "physics_dt": 0.05,
        "rendering_interval": 4,
    }


def test_t5_runner_binds_planar_profile_but_t4_preflight_keeps_defaults() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    preflight = PREFLIGHT.read_text(encoding="utf-8")

    assert (
        'export INTERNVLA_T3_CONFIG="$root/configs/internnav_t5/'
        'go2_continuous_completion_cfg.py"'
    ) in runner
    assert "export INTERNVLA_GO2_MOTION_PROFILE=t5_completion_planar_root_velocity" in runner
    assert "export INTERNVLA_GO2_PHYSICS_HZ=20" in runner
    assert "export INTERNVLA_GO2_CONTROL_HZ=20" in runner
    assert 'export INTERNVLA_GO2_SENSOR_HZ="$rtf_sensor_hz"' in runner
    assert "export INTERNVLA_GO2_EXPECT_PHYSICS_DT=0.05" in runner
    assert 'export INTERNVLA_GO2_EXPECT_RENDERING_INTERVAL="$rtf_rendering_interval"' in runner
    assert 'export INTERNVLA_T4_DEPTH_STRIDE_OVERRIDE="$rtf_depth_stride"' in runner
    assert "rtf_sensor_hz=5" in runner
    assert "rtf_rendering_interval=4" in runner
    assert "rtf_depth_stride=4" in runner
    assert '"high_fidelity_go2_dynamics_claimed": False' in runner
    assert '"real_go2_configuration_inherits_deviation": False' in runner
    assert (
        'os.environ.get(\n        "INTERNVLA_GO2_EXPECT_RENDERING_INTERVAL", "5"\n    )'
        in preflight
    )
    assert 'expected_rendering_interval not in {4, 5, 8}' in preflight
    assert '"INTERNVLA_GO2_EXPECT_PHYSICS_DT", "0.005"' in preflight
    assert '"INTERNVLA_GO2_MOTION_PROFILE", "physics_root_velocity"' in preflight
    assert (
        'payload["rendering_interval"] == expected_rendering_interval' in preflight
    )


def test_t5_sensor_2p5hz_profile_uses_render_stride_eight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_source(tmp_path, "go2_continuous_active_cfg.py", "active")
    monkeypatch.setenv("INTERNNAV_T1_CONTROL_ROOT", str(tmp_path))
    monkeypatch.setenv("INTERNVLA_T3_PHASE", "canary")
    monkeypatch.setenv("INTERNVLA_GO2_EXPECT_RENDERING_INTERVAL", "8")

    config = runpy.run_path(str(CONFIG))["eval_cfg"]

    assert config.env.env_settings["physics_dt"] == 0.05
    assert config.env.env_settings["rendering_interval"] == 8


def test_sensor_rate_defaults_to_t4_ten_hz_and_is_integer_divisor() -> None:
    runtime = (ROOT / "scripts" / "internnav_go2_runtime.py").read_text(
        encoding="utf-8"
    )

    assert 'os.environ.get("INTERNVLA_GO2_SENSOR_HZ", "10")' in runtime
    assert "int(round(self.control_hz / self.sensor_hz))" in runtime
    assert "effective_sensor_hz - self.sensor_hz" in runtime
