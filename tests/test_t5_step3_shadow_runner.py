from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_step3_shadow_runner_is_local_readonly_and_fail_closed() -> None:
    text = (ROOT / "scripts/run_t5_step3_shadow.sh").read_text(encoding="utf-8")
    assert "tcp://127.0.0.1:8200" in text
    assert "http://127.0.0.1:8300/api/v1/state" in text
    assert '"motion_authority": "none"' in text
    assert '"terminal_stop_authority": "none"' in text
    assert 'health.get("checkpoint_load_clean") is True' in text
    assert 'health.get("parameter_count") == 10_171_750_144' in text
    assert 'set(health.get("parameter_dtype_counts") or {}) == {"torch.bfloat16"}' in text
    assert 'health.get("max_new_tokens") == 96' in text
    assert 'health.get("max_retries") == 0' in text
    assert 'health.get("skip_private_reasoning") is True' in text
    assert 'health.get("stop_on_complete_json") is True' in text
    assert 'health.get("generation_wall_budget_s") == 10.5' in text
    assert 'health.get("generation_join_grace_s") == 0.5' in text
    assert 'health.get("redact_raw_text") is True' in text
    assert "INTERNNAV_T5_STEP3_READY_TIMEOUT_SEC:-900" in text
    assert "stop_group" in text
    assert "kill -TERM" in text
    assert "kill -KILL" in text
    assert '"raw_text":' not in text
    assert "cmd_vel" not in text


def test_step3_shadow_runner_requires_the_matching_owned_dgx_lease() -> None:
    text = (ROOT / "scripts/run_t5_step3_shadow.sh").read_text(encoding="utf-8")
    assert 'INTERNNAV_T5_RESOURCE_LEASE_ACK:-}' in text
    assert "dgx-a|lane-a" in text
    assert "dgx-b|lane-b" in text
    assert "expected_user=railgun" in text
    assert "expected_user=rail" in text
    assert 'test "$(id -un)" = "$expected_user"' in text
    assert "CUDA_VISIBLE_DEVICES=0" in text
