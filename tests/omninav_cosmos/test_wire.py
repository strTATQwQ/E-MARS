from omninav_cosmos.contracts import NavigationOutput, NavigationRequest
from omninav_cosmos.transports.wire import decode_request, decode_response, encode_request, encode_response


def test_request_multipart_round_trip_preserves_optional_views():
    request = NavigationRequest(
        episode_id="ep-a",
        frame_id=3,
        timestamp=12.0,
        instruction="go through the door",
        rgb_front=b"front",
        rgb_left=b"left",
        optional_depth=b"depth",
        agent_pose=(1.0, 2.0, 0.3),
    )
    decoded = decode_request(encode_request(request))
    assert decoded == request


def test_response_round_trip_keeps_safety_and_latency_fields():
    output = NavigationOutput(
        episode_id="ep-a",
        frame_id=3,
        waypoints=((0.2, 0.0),),
        heading_sin_cos=((0.0, 1.0),),
        arrive_or_stop=False,
        confidence=0.7,
        model_latency_ms=123.0,
        vision_latency_ms=45.0,
        model_variant="cosmos",
        precision_mode="bf16",
        cache_hit=True,
        action_head_trained=True,
        peak_memory_mib=1234.5,
    )
    assert decode_response(encode_response(output)) == output
