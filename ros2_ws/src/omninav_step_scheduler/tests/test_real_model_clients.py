from omninav_step_scheduler.omninav_model_client_node import (
    OmniNavModelClientNode,
    action_config_for_mode,
    image_orientation_for_mode,
    orient_rgb_array,
    waypoint_tensor_to_action,
)
import threading
from omninav_step_scheduler.schemas import parse_step_plan
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from omninav_step_scheduler.step_http_client_node import (
    confirmed_sensor_track_for_step,
    assistant_prefill_for_request,
    build_chat_payload,
    attach_image_to_messages,
    coerce_step_response,
    first_json_object,
    enforce_visual_attribute_gate,
    named_color_pixel_fraction,
    parse_semantic_executive_response,
    strict_json_object,
    strip_assistant_prefill,
    ros_image_to_data_url,
    persist_data_url_snapshot,
    step_image_contract_for_mode,
)


def test_semantic_response_defaults_only_an_explicit_empty_evidence_string():
    raw = json.dumps(
        {
            "subgoal_type": "approach",
            "target": "red marker",
            "relation": "",
            "constraints": [],
            "completion_evidence": "",
            "recovery": "scan",
            "confidence": 0.9,
        }
    )
    plan, defaulted = parse_semantic_executive_response(raw)
    assert defaulted is True
    assert plan["completion_evidence"] == "pending observable confirmation"
    assert plan["completion_evidence_source"] == "client_safe_default"

    invalid = json.loads(raw)
    invalid.pop("completion_evidence")
    with pytest.raises(Exception):
        parse_semantic_executive_response(json.dumps(invalid))


def test_step_confirmation_fuses_only_confirmed_actual_sensor_track():
    value = {
        "episode_id": "episode_a",
        "target_id": "fire extinguisher",
        "confirmed": True,
        "fresh": True,
        "visible": True,
        "distance_m": 1.8,
        "frame_seq": 9,
        "source": "actual_sensor_spatial_track",
    }
    fused = confirmed_sensor_track_for_step(
        value,
        episode_id="episode_a",
        model_visible=True,
        model_confidence=0.9,
    )
    assert fused["confirmed"] is True
    assert fused["fusion_source"] == "two_frame_grounded_sam_plus_step_confirmation"
    assert confirmed_sensor_track_for_step(
        value | {"source": "oracle"},
        episode_id="episode_a",
        model_visible=True,
        model_confidence=0.9,
    ) is None


def test_first_json_object_extracts_json_from_model_text():
    parsed = first_json_object('Here is the plan: {"subgoal": "red door", "confidence": 0.8}')
    assert parsed == {"subgoal": "red door", "confidence": 0.8}


def test_step_response_coercion_accepts_omninav_instruction_alias():
    req = {
        "request_id": "step_req_1",
        "timestamp_request": 10.0,
        "pose_at_request": [1.0, 2.0, 3.0],
        "pending_mode": "stop",
    }
    plan_payload = coerce_step_response(
        req,
        """
        {
          "omninav_instruction": "go to the red exit sign",
          "subgoal": "red exit sign",
          "success_condition": "arrived",
          "recommended_pending_mode": "move_slow",
          "confidence": 0.81
        }
        """,
    )
    plan = parse_step_plan(plan_payload)
    assert plan.navila_or_omninav_instruction == "go to the red exit sign"
    assert plan.recommended_pending_mode == "move_slow"
    assert plan.confidence == 0.81


def test_step_response_coercion_falls_back_to_safe_stop_plan():
    plan_payload = coerce_step_response(
        {
            "request_id": "step_req_2",
            "timestamp_request": 20.0,
            "pose_at_request": [0.0, 0.0, 0.0],
            "prompt": {
                "messages": [
                    {
                        "role": "user",
                        "content": '{"Mission": "Move to the end of the hallway.", "Current subgoal": ""}',
                    }
                ]
            },
        },
        "not json",
        error="timeout",
    )
    plan = parse_step_plan(plan_payload)
    assert plan.recommended_pending_mode == "stop"
    assert plan.confidence == 0.25
    assert plan.navila_or_omninav_instruction == "Move to the end of the hallway."
    assert "client_error:timeout" in plan.raw_json["raw_json"]["repair_notes"]


def test_real_step_payload_disables_reasoning_for_latency_gate():
    _, payload, _, stream = build_chat_payload(
        {"prompt": {"messages": [{"role": "user", "content": "Return JSON"}]}},
        {
            "model_clients": {
                "step_http": {
                    "enable_thinking": False,
                    "reasoning_budget": 0,
                    "assistant_prefill": "<think>\n\n</think>\n",
                }
            }
        },
    )
    body = json.loads(payload)

    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["reasoning_budget"] == 0
    assert body["messages"][-1] == {"role": "assistant", "content": "<think>\n\n</think>\n"}
    assert stream is True


def test_multimodal_step_payload_contains_real_image_url_part():
    messages = attach_image_to_messages(
        [{"role": "user", "content": "inspect target"}],
        "data:image/jpeg;base64,abc",
    )
    _, payload, _, _ = build_chat_payload(
        {
            "multimodal": True,
            "image_data_url": "data:image/jpeg;base64,abc",
            "prompt": {"messages": [{"role": "user", "content": "inspect target"}]},
        },
        {},
    )
    body = json.loads(payload)

    assert messages[0]["content"][1]["type"] == "image_url"
    assert body["messages"][0]["content"][1]["image_url"]["url"].endswith("abc")


def test_step_image_contract_defaults_to_primary_and_allows_controlled_micro_mode():
    config = {"model_clients": {"step_http": {"horizontal_flip": True, "vertical_flip": True}}}

    assert step_image_contract_for_mode(config, {}) == {
        "source": "primary",
        "horizontal_flip": True,
        "vertical_flip": True,
        "snapshot_output_dir": "",
    }
    assert step_image_contract_for_mode(
        config,
        {
            "mode_config": {
                "step_image_source": "controlled",
                "step_horizontal_flip": False,
                "step_vertical_flip": False,
            }
        },
    ) == {
        "source": "controlled",
        "horizontal_flip": False,
        "vertical_flip": False,
        "snapshot_output_dir": "",
    }


def test_step_snapshot_persists_exact_encoded_request_image(tmp_path):
    data_url = "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode("ascii")

    path = persist_data_url_snapshot(
        data_url,
        str(tmp_path),
        episode_id="ep/unsafe name",
        role="semantic_stop",
        frame_seq=17,
    )

    assert Path(path).name == "ep_unsafe_name_semantic_stop_frame000017.jpg"
    assert Path(path).read_bytes() == b"jpeg-bytes"


def test_ros_rgb_image_encodes_as_jpeg_data_url():
    msg = SimpleNamespace(
        width=2,
        height=1,
        step=6,
        encoding="rgb8",
        data=bytes([255, 0, 0, 0, 255, 0]),
    )

    result = ros_image_to_data_url(msg, max_width=2, jpeg_quality=80)

    assert result.startswith("data:image/jpeg;base64,")
    assert len(result) > 40


def test_ros_image_horizontal_flip_reverses_pixel_order():
    import base64
    import io

    from PIL import Image

    msg = SimpleNamespace(
        width=4,
        height=1,
        step=12,
        encoding="rgb8",
        data=bytes([255, 0, 0] * 2 + [0, 0, 255] * 2),
    )

    result = ros_image_to_data_url(msg, max_width=4, jpeg_quality=95, horizontal_flip=True)
    decoded = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[1]))).convert("RGB")

    assert decoded.getpixel((0, 0))[2] > decoded.getpixel((0, 0))[0]
    assert decoded.getpixel((3, 0))[0] > decoded.getpixel((3, 0))[2]


def test_named_color_pixel_fraction_reads_ros_pixels_without_oracle_metadata():
    msg = SimpleNamespace(
        width=4,
        height=1,
        step=12,
        encoding="rgb8",
        data=bytes([255, 0, 0] * 2 + [0, 0, 255] * 2),
    )

    assert named_color_pixel_fraction(msg, "red") == 0.5
    assert named_color_pixel_fraction(msg, "blue") == 0.5
    assert named_color_pixel_fraction(msg, "yellow") == 0.0


def test_visual_attribute_gate_can_only_reject_a_missing_required_color():
    plan = {
        "stop": True,
        "target_visible": True,
        "estimated_distance_ok": True,
        "confidence": 0.99,
        "reason": "model positive",
    }

    gated, rejected = enforce_visual_attribute_gate(
        plan,
        {"required_color": "yellow", "present": False, "source": "ros_image_pixels"},
    )
    unchanged, accepted = enforce_visual_attribute_gate(
        plan,
        {"required_color": "yellow", "present": True, "source": "ros_image_pixels"},
    )

    assert rejected is True
    assert gated["stop"] is False
    assert gated["target_visible"] is False
    assert gated["estimated_distance_ok"] is True
    assert unchanged == plan
    assert accepted is False


def test_step_role_acceptance_requires_strict_json_after_prefill():
    config = {"model_clients": {"step_http": {"assistant_prefill": "<think>\n\n</think>\n"}}}
    raw = '<think>\n\n</think>\n{"route_choice":"left"}'

    assert strict_json_object(strip_assistant_prefill(raw, config)) == {"route_choice": "left"}
    try:
        strict_json_object('```json\n{"route_choice":"left"}\n```')
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("markdown-wrapped JSON must not count as strict Step evidence")


def test_role_prefill_starts_required_json_key_and_remains_parseable():
    config = {"model_clients": {"step_http": {"assistant_prefill": "<think>\n\n</think>\n"}}}
    prefill = assistant_prefill_for_request({"role": "semantic_stop"}, config)
    raw = prefill + 'true,"target_visible":true,"estimated_distance_ok":true,"confidence":1,"reason":"in range"}'

    assert prefill.endswith('{"stop":')
    assert strict_json_object(strip_assistant_prefill(raw, config))["stop"] is True

    semantic_prefill = assistant_prefill_for_request({"role": "semantic_executive"}, config)
    semantic_raw = semantic_prefill + 'find","target":"red marker","relation":"","constraints":[],' \
        '"completion_evidence":"visible","recovery":"scan","confidence":0.9}'
    assert semantic_prefill.endswith('{"subgoal_type":"')
    assert strict_json_object(strip_assistant_prefill(semantic_raw, config))["subgoal_type"] == "find"


def test_waypoint_to_action_moves_forward_for_low_yaw_forward_waypoint():
    action = waypoint_tensor_to_action([[[0.30, 0.03], [0.50, 0.04]]], [[0.05]], config={"turn_threshold_deg": 12.0})
    assert action["primitive"] == "move_forward"
    assert action["distance_m"] > 0.25
    assert action["yaw_deg"] > 0.0
    assert action["confidence"] > 0.5


def test_waypoint_to_action_turns_for_large_lateral_waypoint():
    action = waypoint_tensor_to_action([[[0.05, -0.30]]], [[0.02]], config={"turn_threshold_deg": 12.0})
    assert action["primitive"] == "turn_right"
    assert action["distance_m"] == 0.0


def test_waypoint_to_action_stops_when_arrive_probability_is_high():
    action = waypoint_tensor_to_action([[[0.30, 0.00]]], [[0.95]], config={"stop_arrive_threshold": 0.75})
    assert action["primitive"] == "stop"
    assert action["confidence"] >= 0.9


def test_omninav_image_orientation_is_configurable_and_non_mutating():
    np = pytest.importorskip("numpy")

    frame = np.array(
        [
            [[1, 0, 0], [2, 0, 0]],
            [[3, 0, 0], [4, 0, 0]],
        ],
        dtype=np.uint8,
    )

    oriented = orient_rgb_array(frame, horizontal_flip=True, vertical_flip=True)

    assert oriented[..., 0].tolist() == [[4, 3], [2, 1]]
    assert frame[..., 0].tolist() == [[1, 2], [3, 4]]


def test_omninav_mode_can_override_image_orientation():
    config = {"model_clients": {"omninav": {"horizontal_flip": False, "vertical_flip": True}}}

    assert image_orientation_for_mode(config, {}) == (False, True)
    assert image_orientation_for_mode(
        config,
        {"mode_config": {"omninav_horizontal_flip": True, "omninav_vertical_flip": False}},
    ) == (True, False)


def test_omninav_mode_can_probe_short_forward_waypoint_without_changing_base_config():
    config = {
        "model_clients": {
            "omninav": {
                "min_waypoint_norm_m": 0.03,
                "stop_distance_m": 0.04,
                "forward_min_m": 0.05,
                "waypoint_forward_axis": "y",
            }
        }
    }
    mode = {
        "mode_config": {
            "omninav_min_waypoint_norm_m": 0.01,
            "omninav_stop_distance_m": 0.01,
            "omninav_forward_min_m": 0.01,
        }
    }

    base_action = waypoint_tensor_to_action(
        [[[0.001, 0.018]]], [[0.01]], config=action_config_for_mode(config, {})
    )
    probe_action = waypoint_tensor_to_action(
        [[[0.001, 0.018]]], [[0.01]], config=action_config_for_mode(config, mode)
    )

    assert base_action["primitive"] == "stop"
    assert probe_action["primitive"] == "move_forward"
    assert config["model_clients"]["omninav"]["stop_distance_m"] == 0.04


def test_omninav_busy_model_queues_latest_request_without_fallback_action():
    node = object.__new__(OmniNavModelClientNode)
    node.model_ready = True
    node.model_error = None
    node.model_lock = threading.Lock()
    node.pending_request_lock = threading.Lock()
    node.pending_request = None
    metrics = []
    fallbacks = []
    node.publish_metric = lambda event, **fields: metrics.append((event, fields))
    node.publish_fallback_action = lambda *args, **kwargs: fallbacks.append((args, kwargs))
    node.model_lock.acquire()

    node.on_request(SimpleNamespace(data=json.dumps({"request_id": "req-latest", "episode_id": "ep"})))

    assert node.pending_request["request_id"] == "req-latest"
    assert fallbacks == []
    assert metrics[-1][0] == "omninav_model_request_queued"
    node.model_lock.release()
