import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "run_sim2real_low_speed_sessions.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_sim2real_low_speed_sessions", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_isaac_flags_freeze_low_speed_and_leave_real_robot_disabled():
    module = _module()
    flags = module.ISAAC_FLAGS

    assert flags[flags.index("--linear_speed") + 1] == "0.20"
    assert flags[flags.index("--max_yaw_rate") + 1] == "0.30"
    assert flags[flags.index("--low_speed_gait_command") + 1] == "0.45"
    assert "--low_speed_motion_guard" in flags
    assert flags[flags.index("--low_speed_actual_hard_limit") + 1] == "0.160"
    assert flags[flags.index("--low_speed_total_hard_limit") + 1] == "0.165"
    assert flags[flags.index("--low_speed_velocity_filter_alpha") + 1] == "0.20"
    assert "--low_speed_gait_servo" in flags
    assert "--low_speed_yaw_servo" not in flags
    assert module.ISAAC_RESTART_ATTEMPTS == 2
    assert module.ISAAC_EMPTY_LOG_HUNG_SEC == 60.0


def test_checkpoint_override_is_explicit_and_does_not_mutate_frozen_flags():
    module = _module()
    original = list(module.ISAAC_FLAGS)
    path = "/tmp/policy with spaces.pt"

    flags = module.build_isaac_flags(path)

    assert flags[-2:] == ["--checkpoint", path]
    assert module.ISAAC_FLAGS == original


def test_low_speed_mode_extends_phase_timeouts_without_relaxing_speed_caps():
    import yaml

    module = _module()
    modes = yaml.safe_load((module.ROOT / "configs" / "ablation_modes.yaml").read_text(encoding="utf-8"))["modes"]
    mode = modes["omninav_step_route_stop_sim2real_low_speed"]

    assert mode["branch_advance_timeout_sec"] == 150
    assert mode["branch_forward_speed_mps"] == 0.20
    assert mode["branch_max_yaw_rate_radps"] == 0.30
    assert mode["controller_decision_source"] == "step"
