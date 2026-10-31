import unittest
from types import SimpleNamespace

import numpy as np

from pc.depth.pointcloud import colored_pointcloud, points_in_bbox, voxel_downsample
from pc.fuse.router import FusedDisparity


class VoxelDownsampleTest(unittest.TestCase):
    def test_reduces_points_and_preserves_centroid(self):
        rng = np.random.default_rng(0)
        # 三个 0.1 m 的簇，体素 0.05 m 应合并到少量点
        pts = np.vstack([rng.normal([0, 0, 1], 0.01, (200, 3)),
                         rng.normal([0.5, 0, 2], 0.01, (200, 3))]).astype(np.float32)
        cols = rng.integers(0, 256, (len(pts), 3), dtype=np.uint8)
        out, out_cols = voxel_downsample(pts, cols, 0.05)
        self.assertLess(len(out), len(pts))
        self.assertGreater(len(out), 0)
        self.assertEqual(out.shape[1], 3)
        self.assertEqual(out_cols.shape, out.shape)
        np.testing.assert_allclose(out.mean(axis=0), pts.mean(axis=0), atol=0.02)

    def test_zero_voxel_is_noop(self):
        pts = np.zeros((4, 3), np.float32)
        cols = np.zeros((4, 3), np.uint8)
        out, out_cols = voxel_downsample(pts, cols, 0.0)
        self.assertIs(out, pts)
        self.assertIs(out_cols, cols)

    def test_empty(self):
        out, cols = voxel_downsample(np.zeros((0, 3), np.float32),
                                     np.zeros((0, 3), np.uint8), 0.03)
        self.assertEqual(len(out), 0)
        self.assertEqual(cols.shape, (0, 3))

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            voxel_downsample(np.zeros((3, 3), np.float32), np.zeros((2, 3), np.uint8), 0.03)


class CloudProjectionTest(unittest.TestCase):
    def setUp(self):
        self.calib = SimpleNamespace(focal_px=500.0, baseline_m=0.08, cx=320.0, cy=240.0)
        self.disp = np.full((480, 640), 40.0, np.float32)
        self.fused = FusedDisparity(
            disp=self.disp, valid=np.ones_like(self.disp, bool),
            source=np.ones(self.disp.shape, np.uint8), stats={},
        )

    def test_center_pixel_maps_to_optical_axis(self):
        img = np.zeros((480, 640, 3), np.uint8)
        pts, _ = colored_pointcloud(img, self.fused, self.calib, stride=1, max_depth_m=10)
        # 中心像素 (320,240) 应对应 X≈0, Y≈0
        z = 500.0 * 0.08 / 40.0
        center = pts[(np.abs(pts[:, 0]) < 1e-3) & (np.abs(pts[:, 1]) < 1e-3)]
        self.assertGreater(len(center), 0)
        np.testing.assert_allclose(center[0, 2], z, rtol=1e-5)

    def test_points_in_bbox_offsets_back_to_full_image(self):
        img = np.zeros((480, 640, 3), np.uint8)
        pts, _ = points_in_bbox(img, self.fused, self.calib, (100, 200, 140, 260), stride=2)
        z = 500.0 * 0.08 / 40.0
        # 框内全局 u ∈ [100,140)，X = (u-cx)*Z/f
        self.assertAlmostEqual(pts[:, 0].min(), (100 - 320) * z / 500.0, places=3)
        self.assertAlmostEqual(pts[:, 1].min(), (200 - 240) * z / 500.0, places=3)


if __name__ == "__main__":
    unittest.main()
