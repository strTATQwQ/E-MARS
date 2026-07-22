import json

from isaac_vln_benchmark.multimodal_value_evidence import evaluate_multimodal_value_run
from isaac_vln_benchmark.v12_step_value_utils import ROUTE_STOP_MODE


def _write_run(path, *, multimodal=True, age_sec=0.1):
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_id": "episode_a", "mode": ROUTE_STOP_MODE},
                ]
            }
        ),
        encoding="utf-8",
    )
    event = {
        "episode_id": "episode_a",
        "details": {
            "event_type": "step_http_response",
            "request_id": "request_a",
            "model": "step_http",
            "result": "accepted",
            "role": "route_choice",
            "multimodal": multimodal,
            "image_snapshot": {"frame_seq": 11, "age_sec": age_sec},
            "latency_s": 4.8,
        },
    }
    (path / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")


def test_multimodal_value_gate_requires_fresh_real_image(tmp_path):
    _write_run(tmp_path)
    result = evaluate_multimodal_value_run(tmp_path, existing_gate={"pass": True})
    assert result["pass"] is True
    assert result["multimodal_evidence"]["fresh_multimodal"] == 1
    assert result["qualification_evidence"] is False


def test_multimodal_value_gate_rejects_stale_or_nonvisual_call(tmp_path):
    _write_run(tmp_path, multimodal=False, age_sec=1.0)
    result = evaluate_multimodal_value_run(tmp_path, existing_gate={"pass": True})
    assert result["pass"] is False
    assert "not every accepted Step role call was multimodal" in result["failures"]
