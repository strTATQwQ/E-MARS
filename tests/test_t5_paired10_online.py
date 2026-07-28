from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/internnav_t5/paired10_retrospective_manifest.json"
RUNNER = ROOT / "coordination/run_t5_paired10_online.sh"
ARCHIVER = ROOT / "coordination/finalize_t5_paired10_replay.sh"


def test_paired10_manifest_freezes_two_disjoint_sets_and_balanced_arms() -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sets = value["episode_sets"]
    keys_a = sets["paired10_a"]["episode_keys"]
    keys_b = sets["paired10_b"]["episode_keys"]
    assert len(keys_a) == len(set(keys_a)) == 10
    assert len(keys_b) == len(set(keys_b)) == 10
    assert set(keys_a).isdisjoint(keys_b)
    assert value["evidence_classification"]["held_out"] is False
    assert value["evidence_classification"]["execution_count"] == 40
    rounds = value["rounds"]
    assert [rounds[0]["lane_a"]["evaluation_arm"], rounds[0]["lane_b"]["evaluation_arm"]] == [
        "internvla_only",
        "internvla_step3",
    ]
    assert [rounds[1]["lane_a"]["evaluation_arm"], rounds[1]["lane_b"]["evaluation_arm"]] == [
        "internvla_step3",
        "internvla_only",
    ]
    assert value["capture_contract"]["d435_rgb"]["sim_hz"] == 5
    assert value["latency_contract"]["step3"]["required_fields"][-1] == (
        "wall_response_duration_ms"
    )


def test_paired10_runner_launches_lanes_in_parallel_then_requires_release() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "run_lane a" in text and "run_lane b" in text
    assert text.index('run_lane a "$advisor_a"') < text.index('wait "$a_pid"')
    assert text.index('run_lane b "$advisor_b"') < text.index('wait "$a_pid"')
    assert "verify_clean_release" in text
    assert text.index("run_round round1 0 1") < text.index("run_round round2 1 0")
    assert "INTERNVLA_T5_FULL_RGB_CAPTURE=1" in text
    assert "INTERNVLA_T5_D435_5HZ_CAPTURE=1" in text
    assert "INTERNVLA_T5_TERMINATION_MODE=oracle_termination" in text


def test_serial_postprocess_uses_existing_shared_io_lock_and_builds_timeline() -> None:
    text = ARCHIVER.read_text(encoding="utf-8")
    assert "/tmp/internnav_t5_isaac_shared_assets.lock" in text
    assert "internnav_t5_isaac_a internnav_t5_isaac_b" in text
    assert "/home/song/opt/keyshot-network-2026.1/keyshot_network/bin/ffmpeg" in text
    assert 'scripts/finalize_t5_d435_capture.py' in text
    assert 'scripts/build_t5_unified_replay.py' in text
    assert 'scripts/summarize_t5_paired10.py' in text
    assert "streamed_uncompressed_tar" in text
    assert "remote_originals_retained" in text
