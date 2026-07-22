from probe_go2_policy_velocity_response import percentile, summarize


def test_percentile_uses_nearest_rank():
    assert percentile([0.1, 0.2, 0.3, 0.4], 0.95) == 0.4


def test_summary_marks_fall_and_ignores_warmup_motion():
    rows = [
        {"elapsed_sec": 1.0, "x": 0.0, "y": 0.0, "z": 0.4, "vx": 1.0, "vy": 0.0, "wz": 1.0},
        {"elapsed_sec": 3.0, "x": 1.0, "y": 0.0, "z": 0.3, "vx": 0.1, "vy": 0.0, "wz": 0.1},
        {"elapsed_sec": 4.0, "x": 1.2, "y": 0.0, "z": 0.17, "vx": 0.2, "vy": 0.0, "wz": 0.2},
    ]

    result = summarize(rows, warmup_sec=2.0)

    assert result["progress_m"] == 0.2
    assert result["speed_max_mps"] == 0.2
    assert result["fell"] is True
