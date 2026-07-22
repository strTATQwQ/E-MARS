import base64
import math
import zlib
from array import array

import pytest

from isaac_vln_benchmark.sensor_perception import (
    build_target_observation,
    camera_intrinsics_from_physical,
    frame_stamps_synchronized,
    free_space_sectors_m,
    masked_depth_median_m,
    ppm_bytes,
    result_matches_active_context,
)


def depth_bytes(width: int, height: int, value_mm: int) -> bytes:
    values = array("H", [value_mm] * (width * height))
    return values.tobytes()


def encoded_mask(mask: bytes) -> str:
    return base64.b64encode(zlib.compress(mask)).decode("ascii")


def test_physical_camera_intrinsics_match_isaac_pinhole_config():
    value = camera_intrinsics_from_physical(256, 144)
    assert value["fx"] == pytest.approx(256 * 24.0 / 20.955)
    assert value["fy"] == value["fx"]
    assert value["horizontal_fov_deg"] == pytest.approx(47.18, abs=0.1)


def test_masked_depth_and_free_space_use_actual_depth_pixels():
    width, height = 6, 3
    depth = depth_bytes(width, height, 2500)
    mask = bytes([0, 1, 1, 0, 0, 0] * height)
    assert masked_depth_median_m(depth, mask, width=width, height=height) == 2.5
    sectors = free_space_sectors_m(depth, width=width, height=height, vertical_start=0.0, vertical_end=1.0)
    assert sectors == {"left": 2.5, "front": 2.5, "right": 2.5}


def test_target_observation_projects_right_image_target_to_negative_bearing():
    width, height = 8, 4
    intrinsics = camera_intrinsics_from_physical(width, height)
    mask = bytearray(width * height)
    for y in (1, 2):
        for x in (5, 6):
            mask[y * width + x] = 1
    response = {
        "detections": [
            {
                "label": "fire extinguisher",
                "score": 0.91,
                "bbox_xyxy_norm": [0.6, 0.2, 0.9, 0.8],
                "mask_zlib_base64": encoded_mask(bytes(mask)),
            }
        ],
        "model": {"detector": "grounding-dino-base", "segmenter": "sam2.1-hiera-large"},
    }
    observation, decoded = build_target_observation(
        episode_id="episode_a",
        target_id="fire extinguisher",
        frame_seq=7,
        source_stamp_sec=100.0,
        width=width,
        height=height,
        intrinsics=intrinsics,
        depth=depth_bytes(width, height, 2000),
        response=response,
        model_latency_sec=1.2,
    )
    assert decoded == bytes(mask)
    assert observation["visible"] is True
    assert observation["distance_m"] == 2.0
    assert observation["bearing_rad"] < 0.0
    assert observation["point_base_xy_m"][0] == 2.0
    assert len(observation["candidates"]) == 1
    assert observation["candidate_index"] == 0
    assert "target_pose" not in observation


def test_target_observation_retains_multiple_mask_depth_candidates():
    width, height = 8, 4
    left = bytearray(width * height)
    right = bytearray(width * height)
    for y in (1, 2):
        for x in (1, 2):
            left[y * width + x] = 1
        for x in (5, 6):
            right[y * width + x] = 1
    response = {
        "detections": [
            {
                "label": "red sign distractor",
                "score": 0.95,
                "bbox_xyxy_norm": [0.6, 0.2, 0.9, 0.8],
                "mask_zlib_base64": encoded_mask(bytes(right)),
            },
            {
                "label": "red sign",
                "score": 0.70,
                "bbox_xyxy_norm": [0.1, 0.2, 0.4, 0.8],
                "mask_zlib_base64": encoded_mask(bytes(left)),
            },
        ]
    }
    observation, _ = build_target_observation(
        episode_id="episode_a",
        target_id="red sign",
        frame_seq=1,
        source_stamp_sec=1.0,
        width=width,
        height=height,
        intrinsics=camera_intrinsics_from_physical(width, height),
        depth=depth_bytes(width, height, 2500),
        response=response,
        model_latency_sec=0.2,
    )
    assert len(observation["candidates"]) == 2
    assert observation["candidate_index"] == 0
    assert {row["candidate_index"] for row in observation["candidates"]} == {0, 1}
    assert all(row["candidate_source"] == "groundingdino_sam2_image" for row in observation["candidates"])


def test_temporal_candidate_source_and_propagation_audit_are_preserved():
    width, height = 4, 2
    mask = bytes([1, 1, 0, 0] * height)
    observation, _ = build_target_observation(
        episode_id="episode_a",
        target_id="red sign",
        frame_seq=2,
        source_stamp_sec=2.0,
        width=width,
        height=height,
        intrinsics=camera_intrinsics_from_physical(width, height),
        depth=depth_bytes(width, height, 3000),
        response={
            "detections": [
                {
                    "label": "red sign",
                    "score": 0.99,
                    "candidate_source": "sam2_video_propagation",
                    "bbox_xyxy_norm": [0.0, 0.0, 0.5, 1.0],
                    "mask_zlib_base64": encoded_mask(mask),
                }
            ],
            "temporal_propagation": {"accepted": True, "latency_sec": 0.4},
        },
        model_latency_sec=0.5,
    )
    assert observation["candidate_source"] == "sam2_video_propagation"
    assert observation["temporal_propagation"]["accepted"] is True


def test_no_detection_produces_visible_false_but_keeps_free_space():
    width, height = 6, 3
    observation, mask = build_target_observation(
        episode_id="episode_a",
        target_id="blue box",
        frame_seq=1,
        source_stamp_sec=10.0,
        width=width,
        height=height,
        intrinsics=camera_intrinsics_from_physical(width, height),
        depth=depth_bytes(width, height, 3000),
        response={"detections": [], "model": {}},
        model_latency_sec=0.5,
    )
    assert observation["visible"] is False
    assert observation["free_space_m"]["front"] == 3.0
    assert not any(mask)


def test_missing_required_visual_attribute_rejects_geometric_detection():
    width, height = 4, 2
    mask = bytes([1, 1, 0, 0] * height)
    observation, _ = build_target_observation(
        episode_id="episode_a",
        target_id="fire extinguisher",
        frame_seq=1,
        source_stamp_sec=1.0,
        width=width,
        height=height,
        intrinsics=camera_intrinsics_from_physical(width, height),
        depth=depth_bytes(width, height, 2000),
        response={
            "detections": [
                {
                    "score": 0.9,
                    "label": "fire extinguisher",
                    "bbox_xyxy_norm": [0.0, 0.0, 0.5, 1.0],
                    "mask_zlib_base64": encoded_mask(mask),
                    "visual_attribute_evidence": {
                        "required_colors": ["red"],
                        "pixel_ratios": {"red": 0.0},
                        "exists": False,
                    },
                }
            ]
        },
        model_latency_sec=0.1,
    )
    assert observation["visible"] is False
    assert observation["candidates"][0]["reason"] == "required_visual_attribute_absent"


def test_horizontal_flip_normalizes_bearing_and_depth_sectors_to_step_view():
    width, height = 6, 2
    depth = array("H", [1000, 1000, 2000, 2000, 3000, 3000] * height).tobytes()
    mask = bytes([1, 1, 0, 0, 0, 0] * height)
    response = {
        "detections": [
            {
                "score": 0.9,
                "label": "target",
                "bbox_xyxy_norm": [0.0, 0.0, 0.34, 1.0],
                "mask_zlib_base64": encoded_mask(mask),
            }
        ]
    }
    intrinsics = {"fx": 6.0, "fy": 6.0, "cx": 2.5, "cy": 0.5}
    kwargs = {
        "episode_id": "ep",
        "target_id": "target",
        "frame_seq": 1,
        "source_stamp_sec": 1.0,
        "width": width,
        "height": height,
        "intrinsics": intrinsics,
        "depth": depth,
        "response": response,
        "model_latency_sec": 0.1,
    }
    normal, _ = build_target_observation(**kwargs)
    flipped, _ = build_target_observation(**kwargs, horizontal_flip=True)
    assert normal["bearing_rad"] == pytest.approx(-flipped["bearing_rad"], abs=1e-6)
    assert flipped["free_space_m"]["left"] == normal["free_space_m"]["right"]
    assert flipped["bearing_horizontal_flip_applied"] is True


def test_ppm_and_timestamp_contract():
    raw = bytes([10, 20, 30] * 4)
    value = ppm_bytes(raw, 2, 2)
    assert value.startswith(b"P6\n2 2\n255\n")
    assert value.endswith(raw)
    assert frame_stamps_synchronized(10.0, 10.0019)
    assert not frame_stamps_synchronized(10.0, 10.003)


def test_async_result_must_match_active_episode_and_target():
    result = {"episode_id": "episode_a", "target_id": "Red Sign"}
    assert result_matches_active_context(result, episode_id="episode_a", target_id="red sign")
    assert not result_matches_active_context(result, episode_id="episode_b", target_id="red sign")
    assert not result_matches_active_context(result, episode_id="episode_a", target_id="blue box")
