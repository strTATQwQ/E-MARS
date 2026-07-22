from PIL import Image

from omninav_cosmos.backbones.qwen25_legacy import Qwen25LegacyAdapter


def test_slowfast_preprocess_matches_frozen_seven_view_geometry():
    images = [Image.new("RGB", (486, 420)) for _ in range(7)]
    resized = Qwen25LegacyAdapter._slowfast_preprocess(images)
    assert [image.size for image in resized] == [
        (168, 140),
        (168, 140),
        (168, 140),
        (168, 140),
        (644, 560),
        (168, 140),
        (644, 560),
    ]
