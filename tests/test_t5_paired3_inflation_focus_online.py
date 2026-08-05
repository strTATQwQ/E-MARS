import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/internnav_t5/paired3_inflation_focus_manifest.json"
RUNNER = ROOT / "coordination/run_t5_paired3_inflation_focus_online.sh"


def test_paired3_manifest_freezes_focus_episodes_and_arms() -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert value["status"] == "FROZEN_FOR_EXECUTION"
    assert value["source"] == {
        "pair_set": "paired10_a",
        "source_lane": "a",
        "episode_keys": ["5840_1474", "2613_625", "6623_1657"],
    }
    assert value["arms"]["lane_a"]["evaluation_arm"] == "internvla_only"
    assert value["arms"]["lane_b"] == {
        "evaluation_arm": "internvla_step3",
        "step3_timeout_advisor": True,
        "step3_task_state_control": True,
    }


def test_paired3_runner_uses_same_source_episode_and_full_capture() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "export INTERNNAV_T5_PILOT_SOURCE_LANE=a" in text
    assert "export INTERNVLA_T5_FULL_RGB_CAPTURE=1" in text
    assert "export INTERNVLA_T5_D435_5HZ_CAPTURE=1" in text
    assert "run_lane a 0 0" in text
    assert "run_lane b 1 1" in text
    assert "deadline=$((SECONDS + 60))" in text
    assert "pilot-screen1" in text
