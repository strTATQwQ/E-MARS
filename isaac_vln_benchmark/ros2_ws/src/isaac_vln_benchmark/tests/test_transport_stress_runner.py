import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "run_sim2real_transport_stress.py"


def module():
    spec = importlib.util.spec_from_file_location("run_sim2real_transport_stress", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def test_stress_profiles_cover_latency_loss_and_reset_with_real_robot_disabled():
    value = module()

    assert [row["name"] for row in value.PROFILES] == ["latency", "packet_loss", "reset"]
    assert value.PROFILES[1]["drop"] == 1.0
    assert value.PROFILES[2]["reset"] is True
    assert all(value.config_for(row)["policy"]["real_robot"] == "disabled" for row in value.PROFILES)
