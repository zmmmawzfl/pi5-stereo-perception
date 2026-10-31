import unittest

import cv2
import numpy as np

from pc.stereo.sgbm import SGBMMatcher
from pc.stereo.subpixel import refine_disparity_subpixel
from shared.config import SgbmConfig
from shared.synthetic import default_scene


class TestSGBM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.render = default_scene().render(seed=1)
        cls.matcher = SGBMMatcher(SgbmConfig())
        cls.result = cls.matcher.compute(cls.render.left, cls.render.right)

    def test_shapes_and_validity(self):
        r = self.result
        self.assertEqual(r.disp.shape, (480, 640))
        self.assertEqual(r.disp.dtype, np.float32)
        self.assertEqual(r.valid.dtype, bool)
        self.assertGreater(float(r.coverage), 0.3)
        self.assertTrue(np.all(r.disp[~r.valid] == 0))

    def test_disparity_accuracy_within_reference(self):
        r = self.result
        sel = r.valid & self.render.valid
        self.assertGreater(int(sel.sum()), 10000)
        err = np.abs(r.disp[sel] - self.render.disp[sel])
        self.assertLess(float(np.median(err)), 0.5)
        self.assertLess(float(np.percentile(err, 90)), 1.5)

    def test_right_reference_disparity_available(self):
        r = self.result
        self.assertIsNotNone(r.right_disp)
        self.assertIsNotNone(r.right_valid)
        self.assertGreater(float(r.right_valid.mean()), 0.1)
        self.assertTrue(np.all(r.right_disp[r.right_valid] > 0))

    def test_depth_formula(self):
        f, b = 530.0, 0.083
        sel = self.result.valid & self.render.valid
        z = f * b / self.result.disp[sel]
        self.assertLess(float(np.median(np.abs(z - self.render.depth[sel]))), 0.02)

    def test_subpixel_refinement_is_bounded(self):
        gl = cv2.cvtColor(self.render.left, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(self.render.right, cv2.COLOR_BGR2GRAY)
        sub = refine_disparity_subpixel(gl, gr, self.result.disp, self.result.valid)
        self.assertEqual(sub.shape, self.result.disp.shape)
        delta = np.abs(sub[self.result.valid] - self.result.disp[self.result.valid])
        self.assertLess(float(np.max(delta)), 0.51)
        sel = self.result.valid & self.render.valid
        self.assertLess(float(np.median(np.abs(sub[sel] - self.render.disp[sel]))), 1.0)

    def test_wls_optional(self):
        if not hasattr(cv2, "ximgproc"):
            self.skipTest("opencv-contrib 不可用")
        m = SGBMMatcher(SgbmConfig(), use_wls=True)
        res = m.compute(self.render.left, self.render.right)
        self.assertTrue(res.meta.get("used_wls"))
        self.assertGreater(float(res.coverage), 0.3)


if __name__ == "__main__":
    unittest.main()
