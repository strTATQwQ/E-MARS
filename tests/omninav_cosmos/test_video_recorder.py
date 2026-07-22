from __future__ import annotations

from dataclasses import dataclass

from scripts.record_ros_video import image_message_to_rgb


@dataclass
class FakeImage:
    width: int
    height: int
    step: int
    encoding: str
    data: bytes


def test_bgr_image_with_row_padding_converts_to_packed_rgb():
    message = FakeImage(
        width=2,
        height=1,
        step=8,
        encoding="bgr8",
        data=bytes([30, 20, 10, 60, 50, 40, 0, 0]),
    )
    assert image_message_to_rgb(message) == bytes([10, 20, 30, 40, 50, 60])
