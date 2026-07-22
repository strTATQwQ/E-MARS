from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_step3_online_prepare_uses_only_dgx_b_and_exact_archive() -> None:
    text = (ROOT / "coordination/prepare_t5_step3_runtime_online.sh").read_text(
        encoding="utf-8"
    )
    assert 'with_resource_lease.sh" dgx-b' in text
    assert "with_resource_lease.sh\" dgx-a" not in text
    assert "10.100.120.116" in text
    assert "10.100.100.128" not in text
    assert 'git_command=(git -C "$root")' in text
    assert 'git_command=(git.exe -C "$root_windows")' in text
    assert '"${git_command[@]}" archive --format=tar "$code_sha"' in text
    assert "T5_DEPLOYMENT_REF" in text
    assert "deployment_archive_sha256" in text
    assert "step3-vl-10b-tf4.57.6" in text
    assert "health-only" in text
    assert '"$target:$remote_result/setup"' in text
    assert '"$target:$remote_result/health"' in text
    assert '"$target:$remote_result/."' not in text
    assert "clean_bf16_model_ready" in text
    assert "frontend_readonly" in text
    assert "owned_cleanup" in text


def test_step3_online_prepare_has_no_secret_or_other_lane_surface() -> None:
    text = (ROOT / "coordination/prepare_t5_step3_runtime_online.sh").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "sshpass",
        "password",
        "10.100.120.123",
        "isaac_gpu",
    ):
        assert forbidden not in text
