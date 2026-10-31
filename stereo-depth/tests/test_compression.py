import shutil
import unittest

import numpy as np

from pc.validate.compression import (
    CompressionSetting,
    benchmark_compression,
    h264_roundtrip,
    jpeg_roundtrip,
)
from shared.config import SgbmConfig
from tests.helpers import make_fake_calib, make_scene


class TestCompression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = make_scene(width=320, height=240, fx=265.0)
        cls.frames = [cls.scene.render(seed=50 + i) for i in range(4)]
        cls.left = [f.left for f in cls.frames]
        cls.right = [f.right for f in cls.frames]
        cls.calib = make_fake_calib(320, 240, fx=265.0)

    def test_jpeg_roundtrip(self):
        decoded, size = jpeg_roundtrip(self.left, 70)
        self.assertEqual(len(decoded), 4)
        self.assertEqual(decoded[0].shape, self.left[0].shape)
        self.assertGreater(size, 0)
        self.assertLess(size, sum(f.nbytes for f in self.left))

    @unittest.skipUnless(shutil.which("ffmpeg"), "缺少 ffmpeg")
    def test_h264_roundtrip(self):
        decoded, size = h264_roundtrip(self.left, fps=30, bitrate_kbps=800)
        self.assertEqual(len(decoded), 4)
        self.assertEqual(decoded[0].shape, self.left[0].shape)
        self.assertGreater(size, 0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "缺少 ffmpeg")
    def test_benchmark(self):
        masks = [
            {"name": o["name"], "mask": (self.frames[0].instances == o["index"])}
            for o in self.frames[0].objects
        ]
        result = benchmark_compression(
            self.left,
            self.right,
            self.calib,
            SgbmConfig(num_disparities=128),
            fps=30,
            object_masks=masks,
            settings=[
                CompressionSetting("raw", "raw"),
                CompressionSetting("jpeg_q80", "jpeg", quality=80),
            ],
        )
        self.assertEqual(len(result["settings"]), 2)
        self.assertAlmostEqual(result["settings"][0]["valid_ratio_vs_ref"], 1.0, places=3)
        self.assertIn("object_depth_shift_m", result["settings"][1])
        self.assertGreater(result["reference"]["valid_px_mean"], 0)


if __name__ == "__main__":
    unittest.main()
