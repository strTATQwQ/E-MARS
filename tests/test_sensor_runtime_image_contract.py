from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from sensor_runtime.image_contract import (
    downsample_rgb_2x2,
    require_rgb_content,
    rgb_content_evidence,
)
from sensor_runtime.isaac_backend import IsaacModelFreeBackend
from sensor_runtime.workload import CaptureNotReady


@pytest.mark.parametrize("value", [0, 37, 255])
def test_rgb_content_rejects_black_and_constant_placeholders(value: int) -> None:
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    evidence = rgb_content_evidence(image)
    assert evidence["valid"] is False
    with pytest.raises(ValueError, match="black or constant"):
        require_rgb_content(image, evidence, "fixture")


def test_rgb_content_requires_exact_pixel_evidence_and_dynamic_range() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[2:8, 3:10] = [10, 50, 200]
    evidence = rgb_content_evidence(image)
    assert evidence["valid"] is True and evidence["dynamic_range"] == 200
    assert require_rgb_content(image, evidence, "fixture") == evidence
    with pytest.raises(ValueError, match="does not match"):
        require_rgb_content(image, {**evidence, "nonzero_count": 1}, "fixture")


def test_rgb_conversion_rejects_nan_and_infinity_before_sanitizing() -> None:
    for invalid in (np.nan, np.inf, -np.inf):
        image = np.zeros((4, 5, 4), dtype=np.float32)
        image[1, 2, 0] = invalid
        with pytest.raises(CaptureNotReady, match="NaN or infinity"):
            IsaacModelFreeBackend._rgb8(image, (4, 5))


def test_front_area_downsample_preserves_half_scale_pixel_center_model() -> None:
    source = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    observed = downsample_rgb_2x2(source)
    expected = np.rint(
        source.astype(np.float32).reshape(2, 2, 3, 2, 3).mean(axis=(1, 3))
    ).astype(np.uint8)
    np.testing.assert_array_equal(observed, expected)
    # Standard area resampling maps (159.5 + 0.5) / 2 - 0.5 to 79.5.
    assert (159.5 + 0.5) / 2.0 - 0.5 == 79.5
    assert (119.5 + 0.5) / 2.0 - 0.5 == 59.5
