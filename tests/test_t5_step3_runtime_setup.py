from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_step3_runtime_setup_is_pinned_and_symmetric_across_dgx_hosts() -> None:
    script = (ROOT / "scripts/setup_t5_step3_runtime.sh").read_text(
        encoding="utf-8"
    )
    requirements = (
        ROOT / "configs/internnav_t5/step3_runtime_requirements.txt"
    ).read_text(encoding="utf-8")
    assert "expected_user=railgun" in script
    assert "expected_user=rail" in script
    assert "expected_home=/home/railgun" in script
    assert "expected_home=/home/rail" in script
    assert 'test "$(id -un)" = "$expected_user"' in script
    assert "INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" in script
    assert "dgx-a|lane-a" in script
    assert "dgx-b|lane-b" in script
    assert "torch==2.11.0" in requirements
    assert "torchvision==0.26.0" in requirements
    assert "transformers==4.57.6" in requirements
    assert "fastapi==0.139.2" in requirements
    assert "uvicorn[standard]==0.51.0" in requirements
    assert "torch.cuda.is_available()" in script
    assert '"torch": "2.11.0+cu130"' in script
    assert '"torchvision": "0.26.0+cu130"' in script
    assert "T5_STEP3_RUNTIME_READY.json" in script
    assert "rm -rf" not in script
    assert "sudo " not in script


def test_step3_runtime_setup_does_not_accept_credentials() -> None:
    script = (ROOT / "scripts/setup_t5_step3_runtime.sh").read_text(
        encoding="utf-8"
    )
    for forbidden in ("HF_TOKEN", "HUGGING_FACE", "OPENAI_API_KEY", "password"):
        assert forbidden not in script


def test_step3_health_telemetry_accepts_unified_memory_na() -> None:
    script = (ROOT / "scripts/run_t5_step3_shadow.sh").read_text(
        encoding="utf-8"
    )
    assert '"[N/A]"' in script
    assert "return None" in script
