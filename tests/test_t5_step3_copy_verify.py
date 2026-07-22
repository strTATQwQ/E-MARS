from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_step3_copy_verifier_holds_both_dgx_locks_and_hashes_in_parallel() -> None:
    text = (ROOT / "coordination/verify_t5_step3_lan_copy.sh").read_text(
        encoding="utf-8"
    )
    assert text.index('with_resource_lease.sh" dgx-a') < text.index(
        'with_resource_lease.sh" dgx-b'
    )
    assert "railgun@10.100.100.128" in text
    assert "rail@10.100.120.116" in text
    assert text.count("sha256sum") == 2
    assert 'source_pid=$!' in text
    assert 'destination_pid=$!' in text
    assert 'wait "$source_pid"' in text
    assert 'wait "$destination_pid"' in text
    assert '"exact_file_and_content_match": source == destination' in text
    assert "COPY_VERIFIED" in text


def test_step3_copy_verifier_has_no_secret_inputs() -> None:
    text = (ROOT / "coordination/verify_t5_step3_lan_copy.sh").read_text(
        encoding="utf-8"
    )
    for forbidden in ("HF_TOKEN", "OPENAI_API_KEY", "sshpass", "password"):
        assert forbidden not in text
