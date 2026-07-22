from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..contracts import NavigationOutput, NavigationRequest, ProtocolError


def encode_request(request: NavigationRequest) -> list[bytes]:
    metadata = request.metadata()
    metadata["type"] = "navigation_request"
    frames = [
        request.rgb_front,
        request.rgb_left or b"",
        request.rgb_right or b"",
        request.optional_depth or b"",
    ]
    metadata["binary_layout"] = ["rgb_front", "rgb_left", "rgb_right", "optional_depth"]
    return [json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), *frames]


def decode_request(parts: Sequence[bytes]) -> NavigationRequest:
    if len(parts) != 5:
        raise ProtocolError(f"request multipart must contain 5 frames, got {len(parts)}")
    try:
        metadata = json.loads(parts[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid request metadata JSON") from exc
    if metadata.get("type") != "navigation_request":
        raise ProtocolError("unsupported request message type")
    if metadata.get("binary_layout") != ["rgb_front", "rgb_left", "rgb_right", "optional_depth"]:
        raise ProtocolError("unsupported binary layout")
    payload = dict(metadata)
    payload.update(
        rgb_front=parts[1],
        rgb_left=parts[2] or None,
        rgb_right=parts[3] or None,
        optional_depth=parts[4] or None,
    )
    return NavigationRequest.from_mapping(payload)


def encode_response(output: NavigationOutput) -> bytes:
    payload = output.to_mapping()
    payload["type"] = "navigation_response"
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_response(value: bytes) -> NavigationOutput:
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid response JSON") from exc
    if payload.get("type") != "navigation_response":
        raise ProtocolError("unsupported response message type")
    waypoints = tuple((float(item[0]), float(item[1])) for item in payload.get("waypoints", []))
    headings = tuple((float(item[0]), float(item[1])) for item in payload.get("heading_sin_cos", []))
    return NavigationOutput(
        episode_id=str(payload.get("episode_id") or ""),
        frame_id=int(payload.get("frame_id")),
        waypoints=waypoints,
        heading_sin_cos=headings,
        arrive_or_stop=bool(payload.get("arrive_or_stop")),
        confidence=float(payload.get("confidence")),
        model_latency_ms=float(payload.get("model_latency_ms")),
        vision_latency_ms=float(payload.get("vision_latency_ms", 0.0)),
        server_total_latency_ms=float(payload.get("server_total_latency_ms", 0.0)),
        request_timestamp=float(payload.get("request_timestamp", 0.0)),
        model_variant=str(payload.get("model_variant") or ""),
        precision_mode=str(payload.get("precision_mode") or ""),
        arrive_logits=tuple(float(value) for value in payload.get("arrive_logits", [])),
        cache_hit=bool(payload.get("cache_hit", False)),
        safe_stop_reason=str(payload.get("safe_stop_reason") or ""),
        action_head_trained=bool(payload.get("action_head_trained", False)),
        peak_memory_mib=float(payload.get("peak_memory_mib", 0.0)),
        coordinate_frame=str(payload.get("coordinate_frame") or "base_link_x_forward_y_left"),
        protocol_version=int(payload.get("protocol_version", 1)),
    )


def control_message(kind: str, **payload: Any) -> bytes:
    return json.dumps({"type": kind, **payload}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def parse_control(value: bytes) -> Mapping[str, Any]:
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid control JSON") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("control message must be an object")
    return payload
