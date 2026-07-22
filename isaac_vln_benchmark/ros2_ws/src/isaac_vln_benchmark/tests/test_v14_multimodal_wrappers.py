import json
import sys
from pathlib import Path

import pytest
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))


def test_v14_screen_requires_completed_perception_suite(tmp_path):
    from run_v14_multimodal_value_screening import require_perception_suite

    (tmp_path / "gate.json").write_text(
        json.dumps({"pass": True, "screening_allowed": True, "qualification_evidence": False}),
        encoding="utf-8",
    )
    assert require_perception_suite(tmp_path)["pass"] is True
    (tmp_path / "gate.json").write_text(json.dumps({"pass": False}), encoding="utf-8")
    with pytest.raises(SystemExit):
        require_perception_suite(tmp_path)


def test_v14_confirmation_requires_fresh_image_screen(tmp_path):
    from run_v14_multimodal_paired_confirmation import require_multimodal_screen

    (tmp_path / "multimodal_value_gate.json").write_text(json.dumps({"pass": False}), encoding="utf-8")
    with pytest.raises(SystemExit):
        require_multimodal_screen(tmp_path)


def test_v14_screen_forwards_multimodal_scheduler(tmp_path, monkeypatch):
    import run_v14_multimodal_value_screening as wrapper

    (tmp_path / "gate.json").write_text(
        json.dumps({"pass": True, "screening_allowed": True, "qualification_evidence": False}),
        encoding="utf-8",
    )
    captured = {}

    def fake_screen(command):
        captured["command"] = command
        return 0

    monkeypatch.setattr(wrapper, "run_screening", fake_screen)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_v14_multimodal_value_screening.py", "--perception-suite", str(tmp_path), "--dry-run"],
    )
    assert wrapper.main() == 0
    index = captured["command"].index("--scheduler-config")
    assert captured["command"][index + 1].endswith("scheduler_isaac_v14_multimodal.yaml")


def test_v12_launcher_keeps_explicit_scheduler_config():
    from run_v12_step_value_screening import with_scheduler_config

    command = with_scheduler_config(["python", "run_live_success_benchmark.py"], "frozen.yaml")
    assert command[-2:] == ["--scheduler-config", "frozen.yaml"]


def test_v16_camera_topics_require_actual_viewport_for_both_models():
    config_path = Path(__file__).resolve().parents[4] / ".." / "ros2_ws" / "src" / "omninav_step_scheduler" / "config" / "scheduler_isaac_v14_multimodal.yaml"
    config = yaml.safe_load(config_path.resolve().read_text(encoding="utf-8"))

    step = config["model_clients"]["step_http"]
    omninav = config["model_clients"]["omninav"]
    assert step["front_image_topic"] == "/camera/front/isaac_image"
    assert (step["horizontal_flip"], step["vertical_flip"]) == (True, True)
    assert omninav["front_image_topic"] == "/camera/front/isaac_image"
    assert omninav["synthetic_fallback"] is False
    assert (omninav["horizontal_flip"], omninav["vertical_flip"]) == (True, True)
