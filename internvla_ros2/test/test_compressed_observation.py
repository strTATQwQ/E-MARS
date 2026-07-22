import unittest

import numpy as np

from internvla_ros2.compressed_observation import (
    ObservationCodecError,
    decode_depth_png,
    decode_rgb_jpeg,
    encode_depth_png,
    encode_rgb_jpeg,
)
from internvla_ros2.observation import compressed_observation_digest


class CompressedObservationTests(unittest.TestCase):
    def setUp(self):
        rows = np.arange(480, dtype=np.uint16)[:, None]
        cols = np.arange(640, dtype=np.uint16)[None, :]
        self.rgb = np.stack(
            (
                ((rows + cols) % 256).astype(np.uint8),
                ((2 * rows + cols) % 256).astype(np.uint8),
                ((rows + 2 * cols) % 256).astype(np.uint8),
            ),
            axis=2,
        )
        self.depth = (((rows + cols) % 1024) / 1023.0).astype(np.float32)[:, :, None]

    def test_round_trip_is_bounded_and_normalized(self):
        rgb_payload = encode_rgb_jpeg(self.rgb)
        depth_payload = encode_depth_png(self.depth)
        decoded_rgb = decode_rgb_jpeg(rgb_payload)
        decoded_depth = decode_depth_png(depth_payload)
        self.assertEqual(decoded_rgb.shape, self.rgb.shape)
        self.assertEqual(decoded_depth.shape, self.depth.shape)
        self.assertEqual(decoded_rgb.dtype, np.uint8)
        self.assertEqual(decoded_depth.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(decoded_depth - self.depth))), 1.0 / 65535.0)
        self.assertLess(len(rgb_payload), self.rgb.nbytes)
        self.assertLess(len(depth_payload), self.depth.nbytes)

    def test_digest_covers_exact_payload(self):
        rgb_payload = encode_rgb_jpeg(self.rgb)
        depth_payload = encode_depth_png(self.depth)
        common = ("go", [1, 2], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        first = compressed_observation_digest(rgb_payload, depth_payload, *common)
        second = compressed_observation_digest(rgb_payload, depth_payload, *common)
        changed = compressed_observation_digest(rgb_payload[:-1] + b"x", depth_payload, *common)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_rejects_wrong_format_corruption_and_size(self):
        with self.assertRaises(ObservationCodecError):
            decode_rgb_jpeg(b"not-jpeg")
        with self.assertRaises(ObservationCodecError):
            decode_depth_png(encode_rgb_jpeg(self.rgb))
        with self.assertRaises(ObservationCodecError):
            encode_rgb_jpeg(self.rgb, maximum_bytes=16)
        corrupt = bytearray(encode_depth_png(self.depth))
        corrupt[len(corrupt) // 2] ^= 0xFF
        with self.assertRaises(ObservationCodecError):
            decode_depth_png(bytes(corrupt))


if __name__ == "__main__":
    unittest.main()
