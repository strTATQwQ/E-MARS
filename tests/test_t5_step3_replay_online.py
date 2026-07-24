from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_step3_replay_is_lane_b_only_bounded_and_exact() -> None:
    text = (ROOT / "coordination/run_t5_step3_replay_online.sh").read_text(
        encoding="utf-8"
    )
    assert 'with_resource_lease.sh" dgx-b' in text
    assert "10.100.120.122" in text
    assert "10.100.100.128" not in text
    assert "T5_DEPLOYMENT_REF" in text
    assert "--mode bounded_advisor" in text
    assert "--request-class warmup" in text
    assert "--request-class production" in text
    assert "warmup.json" in text
    assert "tcp://127.0.0.1:8200" in text
    assert 'PYTHONPATH="$deployment${PYTHONPATH:+:$PYTHONPATH}"' in text
    assert "deadline_met" in text
    assert "motion_authority" in text
    assert "raw_text" in text
    assert "cmd_vel" not in text
    assert "terminal STOP" not in text


def test_step3_replay_does_not_accept_credentials() -> None:
    text = (ROOT / "coordination/run_t5_step3_replay_online.sh").read_text(
        encoding="utf-8"
    )
    for forbidden in ("HF_TOKEN", "OPENAI_API_KEY", "sshpass", "password"):
        assert forbidden not in text
