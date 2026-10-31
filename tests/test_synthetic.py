import unittest

import cv2
import numpy as np

from pc.calib.calibrate import detect_corners
from shared.synthetic import default_scene, generate_calibration_views


class TestScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = default_scene()
        cls.render = cls.scene.render(seed=2)

    def test_shapes_and_dtypes(self):
        r = self.render
        self.assertEqual(r.left.shape, (480, 640, 3))
        self.assertEqual(r.right.shape, (480, 640, 3))
        self.assertEqual(r.left.dtype, np.uint8)
        self.assertEqual(r.depth.shape, (480, 640))
        self.assertTrue(np.isfinite(r.depth[r.valid]).all())

    def test_disparity_matches_geometry(self):
        r = self.render
        expected = 530.0 * 0.083 / r.depth
        diff = np.abs(r.disp - expected)[r.valid]
        self.assertLess(float(np.median(diff)), 1e-3)

    def test_depth_to_disparity_sign(self):
        r = self.render
        # 近处视差必须大于远处（正视差约定）
        near = r.disp[r.instances == 0]
        far = r.disp[r.instances == 2]
        self.assertGreater(float(np.median(near)), float(np.median(far)))

    def test_occlusion_marked_invalid(self):
        r = self.render
        # 物体边界左侧（被前景遮挡）应存在无效像素
        self.assertGreater(int((~r.valid).sum()), 0)
        self.assertLess(float(r.valid.mean()), 0.9)

    def test_object_ground_truth(self):
        r = self.render
        self.assertEqual(len(r.objects), 3)
        for obj in r.objects:
            center = obj["gt_surface_center_m"]
            self.assertAlmostEqual(center[2], obj["gt_front_depth_m"], places=5)
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            self.assertGreater(x2, x1)
            self.assertGreater(y2, y1)

    def test_camera_height_ground_plane(self):
        r = self.render
        # 地面像素反投影后 Y 应≈相机高度（真值自洽性）
        ys, xs = np.mgrid[0:480, 0:640]
        Z = r.depth
        Y = (ys - 239.5) * Z / 530.0
        sel = (r.instances == -1) & r.valid
        self.assertLess(float(np.median(np.abs(Y[sel] - 0.25))), 1e-3)


class TestCheckerboard(unittest.TestCase):
    def test_rendered_board_detectable_and_corners_align(self):
        v = generate_calibration_views(2, seed=21, D_left=np.zeros(5), D_right=np.zeros(5))
        img = v["left"][0]
        corners = detect_corners(img, (9, 6))
        self.assertIsNotNone(corners)
        self.assertEqual(corners.shape, (54, 1, 2))
        gt = v["gt_left_corners"][0].reshape(-1, 2)
        got = corners.reshape(-1, 2)
        # SB 的角点顺序可能与经典行优先相差 180° 旋转（等价于棋盘转 180°）
        err_fwd = np.median(np.linalg.norm(got - gt, axis=1))
        err_rev = np.median(np.linalg.norm(got - gt[::-1], axis=1))
        self.assertLess(float(min(err_fwd, err_rev)), 0.6)

    def test_distortion_changes_pixels(self):
        from shared.synthetic import distort_image

        v = generate_calibration_views(1, seed=7, D_left=np.zeros(5), D_right=np.zeros(5))
        img = v["left"][0]
        K = v["K_left"]
        D = np.array([-0.12, 0.02, 0.0, 0.0, 0.0])
        out = distort_image(img, K, D)
        self.assertEqual(out.shape, img.shape)
        self.assertGreater(float(np.mean(np.abs(out.astype(float) - img.astype(float)))), 0.5)


if __name__ == "__main__":
    unittest.main()
