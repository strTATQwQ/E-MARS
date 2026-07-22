import json

from isaac_vln_benchmark.v12_step_value_utils import (
    ROUTE_STOP_MODE,
    SCREEN_MODES,
    evaluate_v12_screening,
)


def _episodes():
    tasks = []
    for task_type, count in (("simple_navigation", 6), ("turn_choice", 6), ("semantic_target", 3)):
        for index in range(count):
            tasks.append((f"{task_type}_{index}", task_type))
    rows = []
    for mode in SCREEN_MODES:
        for index, (task_id, task_type) in enumerate(tasks):
            success = index < 6
            clean = index < 4
            if mode == ROUTE_STOP_MODE:
                success = index < 6 or index in {6, 12}
                clean = index < 6
            rows.append(
                {
                    "episode_id": f"{mode}_{task_id}",
                    "mode": mode,
                    "task_id": task_id,
                    "task_type": task_type,
                    "success": success,
                    "clean_success": clean,
                    "mission_time_sec": 10.0,
                    "path_length_m": 2.0,
                    "num_collisions": 0,
                }
            )
    return rows


def _step_events(episodes):
    events = []
    roles = {
        "omninav_step_route_only_v12_screen": {"turn_choice": "route_choice"},
        "omninav_step_stop_only_v12_screen": {"semantic_target": "semantic_stop"},
        "omninav_step_route_stop_v12_screen": {"turn_choice": "route_choice", "semantic_target": "semantic_stop"},
        "step_only_v12_sanity": {
            "simple_navigation": "semantic_stop",
            "turn_choice": "route_choice",
            "semantic_target": "semantic_stop",
        },
    }
    for episode in episodes:
        role = roles.get(episode["mode"], {}).get(episode["task_type"])
        if role:
            events.append(
                {
                    "episode_id": episode["episode_id"],
                    "event": "metrics_event_jsonl",
                    "details": {
                        "event_type": "step_http_response",
                        "model": "step_http",
                        "result": "accepted",
                        "role": role,
                        "latency_s": 1.2,
                    },
                }
            )
    return events


def test_v12_screening_requires_real_step_and_paired_gain(tmp_path):
    episodes = _episodes()
    (tmp_path / "metrics.json").write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
    events = _step_events(episodes)
    events.append(
        {
            "episode_id": next(row["episode_id"] for row in episodes if row["mode"] == ROUTE_STOP_MODE),
            "details": {"event_type": "omninav_model_fallback", "model": "omninav_model"},
        }
    )
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    result = evaluate_v12_screening(tmp_path)

    assert result["pass"] is True
    assert result["route_stop_success_delta"] == 2
    assert result["route_stop_clean_delta"] == 2
    assert result["effect_counts"][ROUTE_STOP_MODE]["helped"] == 2
    assert result["real_step_http"][ROUTE_STOP_MODE]["accepted"] == 9
    assert result["confirmation"] == "ALLOWED"


def test_v12_screening_rejects_mock_or_fallback_step(tmp_path):
    episodes = _episodes()
    (tmp_path / "metrics.json").write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
    step_episode = next(row for row in episodes if row["mode"] == ROUTE_STOP_MODE and row["task_type"] == "turn_choice")
    events = _step_events(episodes)
    events.append(
        {
            "episode_id": step_episode["episode_id"],
            "details": {"event_type": "step_fallback", "model": "step_http"},
        }
    )
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    result = evaluate_v12_screening(tmp_path)

    assert result["pass"] is False
    assert f"{ROUTE_STOP_MODE} contains mock/fallback Step evidence" in result["failures"]
