import pytest

from omninav_cosmos.contracts import NavigationOutput, NavigationRequest, ProtocolError


def request_payload(**updates):
    payload = {
        "episode_id": "episode-1",
        "frame_id": 7,
        "timestamp": 12.5,
        "instruction": "go around the box and stop by the red chair",
        "rgb_front": b"jpeg",
        "agent_pose": [1.0, 2.0, 0.1],
        "last_action": {"linear": 0.1},
        "collision_state": {"collision": False},
    }
    payload.update(updates)
    return payload


def test_request_contract_keeps_images_out_of_metadata():
    request = NavigationRequest.from_mapping(request_payload(rgb_left=b"left", optional_depth=b"depth"))
    metadata = request.metadata()
    assert metadata["has_left"] is True
    assert metadata["has_depth"] is True
    assert "rgb_front" not in metadata
    assert request.agent_pose == (1.0, 2.0, 0.1)


@pytest.mark.parametrize(
    "updates",
    [
        {"episode_id": ""},
        {"frame_id": -1},
        {"timestamp": float("nan")},
        {"instruction": ""},
        {"rgb_front": b""},
        {"protocol_version": 999},
    ],
)
def test_request_rejects_invalid_payload(updates):
    with pytest.raises(ProtocolError):
        NavigationRequest.from_mapping(request_payload(**updates))


def test_safe_stop_is_explicit_and_serializable():
    output = NavigationOutput.safe_stop(
        episode_id="episode-1",
        frame_id=9,
        reason="request_timeout",
        request_timestamp=3.0,
    )
    payload = output.to_mapping()
    assert payload["arrive_or_stop"] is True
    assert payload["safe_stop_reason"] == "request_timeout"
    assert payload["waypoints"] == []


def test_output_requires_one_heading_per_waypoint():
    with pytest.raises(ProtocolError):
        NavigationOutput(
            episode_id="episode-1",
            frame_id=1,
            waypoints=((1.0, 0.0),),
            heading_sin_cos=(),
            arrive_or_stop=False,
            confidence=0.5,
            model_latency_ms=10.0,
        )

