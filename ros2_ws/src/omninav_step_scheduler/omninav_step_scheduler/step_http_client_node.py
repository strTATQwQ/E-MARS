from __future__ import annotations

import base64
import colorsys
import io
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .schemas import (
    PENDING_MODES,
    attach_timebase,
    clock_domain_from_node,
    deep_get,
    load_yaml_file,
    make_metric,
    new_id,
    node_ros_now_sec,
    now,
    parse_step_plan,
)
from .semantic_executive import parse_semantic_subgoal_json, safe_semantic_fallback
from .step_roles import (
    coerce_route_choice_response,
    coerce_semantic_stop_response,
    parse_route_choice_json,
    parse_semantic_stop_json,
    TargetTrackState,
    target_visual_attributes,
)


def step_image_contract_for_mode(config: dict[str, Any], mode_payload: dict[str, Any]) -> dict[str, Any]:
    mode = mode_payload.get("mode_config") if isinstance(mode_payload.get("mode_config"), dict) else {}
    source = str(mode.get("step_image_source") or "primary").lower()
    if source not in {"primary", "controlled"}:
        source = "primary"
    return {
        "source": source,
        "horizontal_flip": bool(
            mode.get("step_horizontal_flip", deep_get(config, "model_clients.step_http.horizontal_flip", False))
        ),
        "vertical_flip": bool(
            mode.get("step_vertical_flip", deep_get(config, "model_clients.step_http.vertical_flip", False))
        ),
        "snapshot_output_dir": str(
            mode.get("step_snapshot_output_dir", deep_get(config, "model_clients.step_http.snapshot_output_dir", ""))
            or ""
        ),
    }


def persist_data_url_snapshot(
    data_url: str,
    output_dir: str,
    *,
    episode_id: str,
    role: str,
    frame_seq: int,
) -> str:
    if not output_dir:
        return ""
    encoded = str(data_url).split(",", 1)
    if len(encoded) != 2:
        raise ValueError("snapshot data URL has no payload")
    safe_episode = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(episode_id or "episode"))[:120]
    safe_role = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(role or "step"))[:40]
    path = Path(output_dir).expanduser() / f"{safe_episode}_{safe_role}_frame{int(frame_seq):06d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(encoded[1]))
    return str(path)

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except Exception:  # pragma: no cover - exercised on ROS hosts
    rclpy = None
    ExternalShutdownException = RuntimeError
    Node = object
    Image = None
    String = None


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def first_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        loaded = json.loads(stripped)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        pass
    match = JSON_BLOCK_RE.search(stripped)
    if not match:
        return None
    try:
        loaded = json.loads(match.group(0))
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        return None


def strict_json_object(text: str) -> dict[str, Any]:
    value = json.loads(str(text or "").strip())
    if not isinstance(value, dict):
        raise ValueError("Step role response must be one JSON object")
    return value


def parse_semantic_executive_response(text: str) -> tuple[dict[str, Any], bool]:
    payload = strict_json_object(text)
    evidence_defaulted = (
        "completion_evidence" in payload
        and not str(payload.get("completion_evidence") or "").strip()
    )
    if evidence_defaulted:
        payload["completion_evidence"] = "pending observable confirmation"
        payload["completion_evidence_source"] = "client_safe_default"
    parsed = parse_semantic_subgoal_json(payload)
    return parsed.to_dict(include_metadata=True), evidence_defaulted


def strip_assistant_prefill(text: str, config: dict[str, Any]) -> str:
    prefill = str(deep_get(config, "model_clients.step_http.assistant_prefill", "") or "")
    value = str(text or "")
    return value[len(prefill) :].lstrip() if prefill and value.startswith(prefill) else value.strip()


def assistant_prefill_for_request(req: dict[str, Any], config: dict[str, Any]) -> str:
    prefill = str(deep_get(config, "model_clients.step_http.assistant_prefill", "") or "")
    role = str(req.get("role") or "")
    if role == "semantic_stop":
        return prefill + '{"stop":'
    if role == "route_choice":
        return prefill + '{"route_choice":"'
    if role == "semantic_executive":
        return prefill + '{"subgoal_type":"'
    return prefill


def coerce_step_response(
    req: dict[str, Any],
    model_text: str,
    *,
    error: str | None = None,
    response_timestamp: float | None = None,
    node: Any | None = None,
) -> dict[str, Any]:
    """Return a schema-valid StepPlan payload from model text or a conservative fallback."""

    parsed = first_json_object(model_text) or {}
    notes: list[str] = []
    if not parsed:
        notes.append("model_json_missing")
    if error:
        notes.append(f"client_error:{error}")

    mission_hint = _mission_hint(req)
    pending_mode = str(parsed.get("recommended_pending_mode") or req.get("pending_mode") or "stop")
    if pending_mode not in PENDING_MODES:
        notes.append(f"invalid_pending_mode:{pending_mode}")
        pending_mode = "stop"

    constraints = parsed.get("constraints") if isinstance(parsed.get("constraints"), dict) else {}
    ts_response = float(response_timestamp if response_timestamp is not None else node_ros_now_sec(node))
    plan = {
        "request_id": str(req.get("request_id") or new_id("step")),
        "timestamp_request": float(req.get("timestamp_request", now())),
        "timestamp_response": ts_response,
        "multimodal": bool(req.get("multimodal", False)),
        "pose_at_request": _pose(req.get("pose_at_request")),
        "navila_or_omninav_instruction": str(
            parsed.get("navila_or_omninav_instruction")
            or parsed.get("omninav_instruction")
            or parsed.get("navila_instruction")
            or mission_hint
        ).strip(),
        "subgoal": str(parsed.get("subgoal") or mission_hint).strip(),
        "success_condition": str(parsed.get("success_condition") or "Stop when the requested visual goal is reached.").strip(),
        "constraints": {
            "max_speed_mps": _float(constraints.get("max_speed_mps"), 0.2),
            "avoid_people": bool(constraints.get("avoid_people", True)),
            "stop_if_uncertain": bool(constraints.get("stop_if_uncertain", True)),
            "forbidden_zones": list(constraints.get("forbidden_zones") or []),
        },
        "replan_triggers": list(parsed.get("replan_triggers") or ["low_confidence", "blocked_path", "target_not_visible"]),
        "recommended_pending_mode": pending_mode,
        "confidence": clamp01(parsed.get("confidence"), 0.55 if parsed else 0.25),
        "raw_json": {
            "parsed": parsed,
            "raw_text": model_text,
            "repair_notes": notes,
        },
    }
    if not plan["navila_or_omninav_instruction"]:
        plan["navila_or_omninav_instruction"] = "Move cautiously toward the mission target and stop if uncertain."
    if not plan["subgoal"]:
        plan["subgoal"] = plan["navila_or_omninav_instruction"]

    # Keep failures local to the client; the mission manager should receive a valid plan or a safe stop plan.
    parse_step_plan(plan)
    return attach_timebase(
        plan,
        node=node,
        episode_id=str(req.get("episode_id") or ""),
        mission_id=str(req.get("mission_id") or ""),
        request_id=str(plan["request_id"]),
        clock_domain=str(req.get("clock_domain") or clock_domain_from_node(node)),
        source_stamp=ts_response,
        created_ros_time=ts_response,
    )


def build_chat_payload(req: dict[str, Any], config: dict[str, Any]) -> tuple[str, bytes, float, bool]:
    endpoint = str(req.get("endpoint") or deep_get(config, "model_clients.step_http.endpoint") or deep_get(config, "step.endpoint", "http://127.0.0.1:8080/v1/chat/completions"))
    model = str(req.get("model") or deep_get(config, "model_clients.step_http.model") or deep_get(config, "step.model", "Step-3.7-Flash"))
    timeout = float(deep_get(config, "model_clients.step_http.timeout_sec", req.get("timeout_sec") or deep_get(config, "step.timeout_sec", 30.0)))
    stream = bool(deep_get(config, "model_clients.step_http.use_streaming", deep_get(config, "step.use_streaming", True)))
    prompt = req.get("prompt") if isinstance(req.get("prompt"), dict) else {}
    messages = list(prompt.get("messages")) if isinstance(prompt.get("messages"), list) else []
    if not messages:
        messages = [
            {"role": "system", "content": "You are a semantic supervisor for a quadruped robot. Return JSON only."},
            {"role": "user", "content": json.dumps({"request": req}, ensure_ascii=False)},
        ]
    image_data_url = str(req.get("image_data_url") or "")
    if bool(req.get("multimodal", False)):
        if not image_data_url:
            raise ValueError("multimodal Step request requires a fresh image")
        messages = attach_image_to_messages(messages, image_data_url)
    assistant_prefill = assistant_prefill_for_request(req, config)
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(req.get("max_tokens") or deep_get(config, "model_clients.step_http.max_tokens", deep_get(config, "step.max_tokens", 128))),
        "temperature": float(req.get("temperature") if req.get("temperature") is not None else deep_get(config, "model_clients.step_http.temperature", deep_get(config, "step.temperature", 0.0))),
        "stream": stream,
        "chat_template_kwargs": {
            "enable_thinking": bool(deep_get(config, "model_clients.step_http.enable_thinking", False)),
        },
        "reasoning_budget": int(deep_get(config, "model_clients.step_http.reasoning_budget", 0)),
    }
    return endpoint, json.dumps(payload, ensure_ascii=False).encode("utf-8"), timeout, stream


def attach_image_to_messages(messages: list[dict[str, Any]], image_data_url: str) -> list[dict[str, Any]]:
    result = [dict(message) for message in messages]
    for index in range(len(result) - 1, -1, -1):
        if str(result[index].get("role") or "") != "user":
            continue
        content = result[index].get("content", "")
        parts = list(content) if isinstance(content, list) else [{"type": "text", "text": str(content)}]
        parts.append({"type": "image_url", "image_url": {"url": image_data_url}})
        result[index]["content"] = parts
        return result
    raise ValueError("multimodal prompt has no user message")


def ros_image_to_pil(msg: Any):
    from PIL import Image as PilImage

    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    encoding = str(msg.encoding or "rgb8").lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(encoding)
    if width <= 0 or height <= 0 or channels is None or step < width * channels:
        raise ValueError(f"unsupported ROS image layout: {width}x{height} step={step} encoding={encoding}")
    mode, raw_mode = {
        "rgb8": ("RGB", "RGB"),
        "bgr8": ("RGB", "BGR"),
        "rgba8": ("RGBA", "RGBA"),
        "bgra8": ("RGBA", "BGRA"),
        "mono8": ("L", "L"),
    }[encoding]
    pil_image = PilImage.frombytes(mode, (width, height), bytes(msg.data), "raw", raw_mode, step, 1)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image


def ros_image_to_data_url(
    msg: Any,
    *,
    max_width: int = 640,
    jpeg_quality: int = 85,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
) -> str:
    from PIL import Image as PilImage

    pil_image = ros_image_to_pil(msg)
    if horizontal_flip:
        transpose_api = getattr(PilImage, "Transpose", PilImage)
        pil_image = pil_image.transpose(transpose_api.FLIP_LEFT_RIGHT)
    if vertical_flip:
        transpose_api = getattr(PilImage, "Transpose", PilImage)
        pil_image = pil_image.transpose(transpose_api.FLIP_TOP_BOTTOM)
    if max_width > 0 and pil_image.width > max_width:
        target_height = max(1, round(pil_image.height * max_width / pil_image.width))
        resampling_api = getattr(PilImage, "Resampling", PilImage)
        pil_image = pil_image.resize((max_width, target_height), resampling_api.BILINEAR)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=max(40, min(95, int(jpeg_quality))), optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def named_color_pixel_fraction(msg: Any, color: str) -> float:
    color = str(color or "").lower()
    image = ros_image_to_pil(msg)
    total = max(1, image.width * image.height)
    pixels = image.get_flattened_data() if hasattr(image, "get_flattened_data") else image.getdata()
    matches = sum(_pixel_matches_named_color(pixel, color) for pixel in pixels)
    return matches / total


def _pixel_matches_named_color(pixel: tuple[int, int, int], color: str) -> bool:
    red, green, blue = (channel / 255.0 for channel in pixel[:3])
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    if color == "red":
        return saturation >= 0.45 and value >= 0.25 and (hue <= 0.04 or hue >= 0.96)
    if color == "orange":
        return saturation >= 0.45 and value >= 0.25 and 0.04 < hue <= 0.105
    if color == "yellow":
        return saturation >= 0.40 and value >= 0.30 and 0.105 < hue <= 0.20
    if color == "green":
        return saturation >= 0.35 and value >= 0.20 and 0.20 < hue <= 0.46
    if color == "blue":
        return saturation >= 0.40 and value >= 0.20 and 0.52 < hue <= 0.72
    if color == "purple":
        return saturation >= 0.35 and value >= 0.20 and 0.72 < hue <= 0.92
    if color == "white":
        return saturation <= 0.15 and value >= 0.75
    if color == "black":
        return value <= 0.16
    if color in {"gray", "grey"}:
        return saturation <= 0.18 and 0.18 < value < 0.75
    return False


def enforce_visual_attribute_gate(
    plan: dict[str, Any], evidence: dict[str, Any] | None
) -> tuple[dict[str, Any], bool]:
    if not evidence or not evidence.get("required_color") or evidence.get("present") is not False:
        return plan, False
    gated = dict(plan)
    gated.update(
        {
            "stop": False,
            "target_visible": False,
            "confidence": 0.0,
            "reason": f"{evidence['required_color']} absent",
            "visual_attribute_gate": "rejected",
        }
    )
    return gated, True


def confirmed_sensor_track_for_step(
    value: Any,
    *,
    episode_id: str,
    model_visible: bool,
    model_confidence: float,
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not model_visible:
        return None
    if str(value.get("source") or "") != "actual_sensor_spatial_track":
        return None
    if not (bool(value.get("confirmed")) and bool(value.get("fresh")) and bool(value.get("visible"))):
        return None
    track_episode = str(value.get("episode_id") or "")
    if episode_id and track_episode and track_episode != episode_id:
        return None
    try:
        distance = float(value.get("distance_m"))
        frame_seq = int(value.get("frame_seq"))
    except (TypeError, ValueError):
        return None
    if distance <= 0.0:
        return None
    return dict(value) | {
        "confirmed": True,
        "visible": True,
        "fresh": True,
        "distance_m": distance,
        "frame_seq": frame_seq,
        "model_confirmation": True,
        "model_confidence": max(0.0, min(1.0, float(model_confidence))),
        "fusion_source": "two_frame_grounded_sam_plus_step_confirmation",
    }


def call_openai_compatible_chat(endpoint: str, payload: bytes, timeout: float, stream: bool) -> str:
    request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if stream:
            chunks: list[str] = []
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content is None:
                    content = choice.get("text")
                if content:
                    chunks.append(str(content))
            return "".join(chunks)
        loaded = json.loads(response.read().decode("utf-8", errors="replace"))
    if "choices" in loaded and loaded["choices"]:
        choice = loaded["choices"][0]
        message = choice.get("message") or {}
        if "content" in message:
            return str(message["content"])
        if "text" in choice:
            return str(choice["text"])
    return json.dumps(loaded, ensure_ascii=False)


class StepHttpClientNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run StepHttpClientNode")
        super().__init__("step_http_client")
        self.declare_parameter("config_file", "")
        self.declare_parameter("response_topic", "")
        self.declare_parameter("route_choice_topic", "")
        self.declare_parameter("semantic_stop_topic", "")
        self.declare_parameter("semantic_subgoal_topic", "")
        self.config = load_yaml_file(self.get_parameter("config_file").value)
        self.busy = threading.Lock()
        self.latest_images: dict[str, dict[str, Any]] = {}
        self.image_contract = step_image_contract_for_mode(self.config, {})
        track_cfg = deep_get(self.config, "model_clients.step_http.target_track", {}) or {}
        self.target_track = TargetTrackState(
            required_hits=int(track_cfg.get("required_hits", 2)),
            high_confidence_single_hit=float(track_cfg.get("high_confidence_single_hit", 0.85)),
            min_confidence=float(track_cfg.get("min_confidence", 0.60)),
            max_misses=int(track_cfg.get("max_misses", 2)),
            max_age_sec=float(track_cfg.get("max_age_sec", 3.0)),
        )
        response_topic = str(
            self.get_parameter("response_topic").value
            or deep_get(self.config, "model_clients.step_http.response_topic", "/step/response_json")
        )
        route_choice_topic = str(
            self.get_parameter("route_choice_topic").value
            or deep_get(self.config, "model_clients.step_http.route_choice_topic", "/step/route_choice_json")
        )
        semantic_stop_topic = str(
            self.get_parameter("semantic_stop_topic").value
            or deep_get(self.config, "model_clients.step_http.semantic_stop_topic", "/step/semantic_stop_json")
        )
        semantic_subgoal_topic = str(
            self.get_parameter("semantic_subgoal_topic").value
            or deep_get(self.config, "model_clients.step_http.semantic_subgoal_topic", "/step/semantic_subgoal_json")
        )
        self.response_pub = self.create_publisher(String, response_topic, 10)
        self.route_choice_pub = self.create_publisher(String, route_choice_topic, 10)
        self.semantic_stop_pub = self.create_publisher(String, semantic_stop_topic, 10)
        self.semantic_subgoal_pub = self.create_publisher(String, semantic_subgoal_topic, 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/step/request_json", self.on_request, 10)
        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
        image_topic = str(deep_get(self.config, "model_clients.step_http.front_image_topic", "/camera/front/image"))
        if Image is not None and image_topic:
            self.create_subscription(Image, image_topic, lambda msg: self.on_image("primary", msg), 2)
        controlled_topic = str(deep_get(self.config, "model_clients.step_http.controlled_image_topic", ""))
        if Image is not None and controlled_topic and controlled_topic != image_topic:
            self.create_subscription(Image, controlled_topic, lambda msg: self.on_image("controlled", msg), 2)

    def on_benchmark_mode(self, msg):
        self.image_contract = step_image_contract_for_mode(self.config, _safe_json(msg.data))
        self.publish_metric("step_image_contract", result="configured", **self.image_contract)

    def on_image(self, source: str, msg):
        previous = self.latest_images.get(source, {})
        self.latest_images[source] = {
            "image": msg,
            "received": time.monotonic(),
            "frame_seq": int(previous.get("frame_seq", 0)) + 1,
        }

    def on_request(self, msg):
        req = _safe_json(msg.data)
        if not self.busy.acquire(blocking=False):
            self.publish_metric("step_http_busy", request_id=req.get("request_id"), result="dropped")
            return
        threading.Thread(target=self.handle_request, args=(req,), daemon=True).start()

    def handle_request(self, req: dict[str, Any]):
        req = dict(req)
        t0 = time.perf_counter()
        endpoint = ""
        try:
            if bool(req.get("multimodal", False)):
                image_source = str(self.image_contract["source"])
                snapshot = self.latest_images.get(image_source)
                if not snapshot:
                    raise ValueError(f"no {image_source} front image available for multimodal Step request")
                image = snapshot["image"]
                image_age = max(0.0, time.monotonic() - float(snapshot["received"]))
                max_age = float(deep_get(self.config, "model_clients.step_http.max_image_age_sec", 0.75))
                if image_age > max_age:
                    raise ValueError(f"{image_source} front image stale: {image_age:.3f}s > {max_age:.3f}s")
                req["image_data_url"] = ros_image_to_data_url(
                    image,
                    max_width=int(deep_get(self.config, "model_clients.step_http.image_max_width", 640)),
                    jpeg_quality=int(deep_get(self.config, "model_clients.step_http.image_jpeg_quality", 85)),
                    horizontal_flip=bool(self.image_contract["horizontal_flip"]),
                    vertical_flip=bool(self.image_contract["vertical_flip"]),
                )
                snapshot_artifact = persist_data_url_snapshot(
                    req["image_data_url"],
                    str(self.image_contract.get("snapshot_output_dir") or ""),
                    episode_id=str(req.get("episode_id") or ""),
                    role=str(req.get("role") or "step"),
                    frame_seq=int(snapshot["frame_seq"]),
                )
                req["image_snapshot"] = {
                    "source": image_source,
                    "frame_seq": int(snapshot["frame_seq"]),
                    "age_sec": round(image_age, 6),
                    "width": int(image.width),
                    "height": int(image.height),
                    "encoding": str(image.encoding),
                    "request_width": min(
                        int(image.width),
                        int(deep_get(self.config, "model_clients.step_http.image_max_width", 640)),
                    ),
                    "request_height": round(
                        int(image.height)
                        * min(
                            int(image.width),
                            int(deep_get(self.config, "model_clients.step_http.image_max_width", 640)),
                        )
                        / max(1, int(image.width))
                    ),
                    "horizontal_flip": bool(self.image_contract["horizontal_flip"]),
                    "vertical_flip": bool(self.image_contract["vertical_flip"]),
                    "artifact_path": snapshot_artifact,
                }
                attribute_gate_cfg = deep_get(
                    self.config, "model_clients.step_http.visual_attribute_gate", {}
                ) or {}
                if bool(attribute_gate_cfg.get("enabled", False)):
                    target = str(req.get("target") or req.get("active_subgoal") or "")
                    required_color = target_visual_attributes(target).get("color", "")
                    if required_color:
                        fraction = named_color_pixel_fraction(image, required_color)
                        threshold = float(attribute_gate_cfg.get("min_color_pixel_fraction", 0.0002))
                        req["visual_attribute_evidence"] = {
                            "source": "ros_image_pixels",
                            "required_color": required_color,
                            "pixel_fraction": round(fraction, 8),
                            "min_pixel_fraction": threshold,
                            "present": bool(fraction >= threshold),
                        }
            endpoint, payload, timeout, stream = build_chat_payload(req, self.config)
            self.publish_metric("step_http_request", request_id=req.get("request_id"), endpoint=endpoint, stream=stream)
            text = call_openai_compatible_chat(endpoint, payload, timeout, stream)
            text = strip_assistant_prefill(text, self.config)
            role = str(req.get("role") or "")
            if role == "route_choice":
                try:
                    plan = parse_route_choice_json(strict_json_object(text))
                    result = "accepted"
                    error = None
                except Exception as exc:
                    error = f"route_schema:{exc!r}"
                    plan = coerce_route_choice_response(req, text, error=error)
                    result = "fallback"
            elif role == "semantic_stop":
                try:
                    plan = parse_semantic_stop_json(strict_json_object(text))
                    result = "accepted"
                    error = None
                except Exception as exc:
                    error = f"semantic_stop_schema:{exc!r}"
                    plan = coerce_semantic_stop_response(req, text, error=error)
                    result = "fallback"
            elif role == "semantic_executive":
                try:
                    plan, evidence_defaulted = parse_semantic_executive_response(text)
                    result = "accepted_evidence_defaulted" if evidence_defaulted else "accepted"
                    error = None
                except Exception as exc:
                    error = f"semantic_executive_schema:{exc!r}"
                    plan = safe_semantic_fallback(reason=error)
                    result = "fallback"
            else:
                plan = coerce_step_response(req, text, response_timestamp=node_ros_now_sec(self), node=self)
                result = "accepted"
                error = None
        except (urllib.error.URLError, TimeoutError, OSError, Exception) as exc:
            error = repr(exc)
            text = ""
            role = str(req.get("role") or "")
            if role == "route_choice":
                plan = coerce_route_choice_response(req, text, error=error)
            elif role == "semantic_stop":
                plan = coerce_semantic_stop_response(req, text, error=error)
            elif role == "semantic_executive":
                plan = safe_semantic_fallback(reason=error)
            else:
                plan = coerce_step_response(req, text, error=error, response_timestamp=node_ros_now_sec(self), node=self)
            result = "fallback"
        finally:
            self.busy.release()

        response_stamp = node_ros_now_sec(self)
        client_latency_sec = time.perf_counter() - t0
        role = str(req.get("role") or "")
        if role == "semantic_stop":
            plan, attribute_rejected = enforce_visual_attribute_gate(
                plan,
                req.get("visual_attribute_evidence")
                if isinstance(req.get("visual_attribute_evidence"), dict)
                else None,
            )
            if attribute_rejected:
                result = f"{result}_attribute_rejected"
        if role in {"route_choice", "semantic_stop"}:
            target = str(req.get("target") or req.get("active_subgoal") or req.get("instruction") or "")
            if role == "route_choice":
                view = str(plan.get("visible_in_view") or "none")
                visible = view != "none"
            else:
                view = "front" if bool(plan.get("target_visible", False)) else "none"
                visible = bool(plan.get("target_visible", False))
            frame_seq = (req.get("image_snapshot") or {}).get("frame_seq") if isinstance(req.get("image_snapshot"), dict) else None
            track = self.target_track.update(
                timestamp=response_stamp,
                episode_id=str(req.get("episode_id") or ""),
                target=target,
                visible=visible,
                confidence=float(plan.get("confidence") or 0.0),
                visible_in_view=view,
                frame_seq=int(frame_seq) if frame_seq is not None else None,
            )
            fused_track = confirmed_sensor_track_for_step(
                req.get("sensor_track"),
                episode_id=str(req.get("episode_id") or ""),
                model_visible=visible,
                model_confidence=float(plan.get("confidence") or 0.0),
            )
            if fused_track is not None:
                track = fused_track
            plan = dict(plan) | {
                "target": target,
                "track": track,
                "image_snapshot": req.get("image_snapshot", {}),
                "visual_attribute_evidence": req.get("visual_attribute_evidence", {}),
                "multimodal": bool(req.get("multimodal", False)),
                "step_latency_sec": round(client_latency_sec, 6),
            }
        if str(req.get("role") or "") in {"route_choice", "semantic_stop", "semantic_executive"}:
            plan = attach_timebase(
                dict(plan) | {"request_id": str(req.get("request_id") or plan.get("request_id") or new_id("step_role"))},
                node=self,
                episode_id=str(req.get("episode_id") or ""),
                mission_id=str(req.get("mission_id") or ""),
                request_id=str(req.get("request_id") or plan.get("request_id") or ""),
                clock_domain=str(req.get("clock_domain") or clock_domain_from_node(self)),
                source_stamp=response_stamp,
                created_ros_time=response_stamp,
            )
        if str(req.get("role") or "") == "route_choice":
            self.publish_json(self.route_choice_pub, plan)
        elif str(req.get("role") or "") == "semantic_stop":
            self.publish_json(self.semantic_stop_pub, plan)
        elif str(req.get("role") or "") == "semantic_executive":
            self.publish_json(self.semantic_subgoal_pub, plan)
        else:
            self.publish_json(self.response_pub, plan)
        self.publish_metric(
            "step_http_response",
            request_id=req.get("request_id") or plan.get("request_id"),
            endpoint=endpoint,
            latency_s=client_latency_sec,
            result=result,
            role=str(req.get("role") or "generic"),
            multimodal=bool(req.get("multimodal", False)),
            image_snapshot=req.get("image_snapshot", {}),
            output=plan,
            confidence=plan["confidence"],
            error=error,
        )

    def publish_metric(self, event_type: str, **kwargs):
        self.publish_json(self.metric_pub, make_metric(event_type, model="step_http", **kwargs))

    @staticmethod
    def publish_json(pub, payload):
        msg = String()
        msg.data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def _pose(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [_float(value[0], 0.0), _float(value[1], 0.0), _float(value[2], 0.0)]
    return [0.0, 0.0, 0.0]


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mission_hint(req: dict[str, Any]) -> str:
    prompt = req.get("prompt") if isinstance(req.get("prompt"), dict) else {}
    messages = prompt.get("messages") if isinstance(prompt.get("messages"), list) else []
    for message in reversed(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        event = payload.get("Event reason") if isinstance(payload.get("Event reason"), dict) else {}
        mission = str(payload.get("Mission") or event.get("instruction") or event.get("mission") or "").strip()
        subgoal = str(payload.get("Current subgoal") or event.get("current_subgoal") or event.get("subgoal") or "").strip()
        if mission:
            return mission
        if subgoal:
            return subgoal
    reason = str(req.get("reason") or "mission target")
    return f"Continue toward the mission target; current Step trigger is {reason}."


def main(args=None):
    rclpy.init(args=args)
    node = StepHttpClientNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
