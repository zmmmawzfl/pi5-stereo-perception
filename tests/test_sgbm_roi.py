import unittest

import numpy as np

from pc.calib.calibrate import CalibrationResult
from pc.stereo.sgbm import SGBMMatcher
from shared.config import SgbmConfig, load_config
from shared.synthetic import default_scene


class SgbmScaleRoiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calib = CalibrationResult.load("runs/demo/calib/stereo_calib.npz")
        cls.images = default_scene().render(seed=100)

    def _run(self, **overrides):
        cfg = SgbmConfig(**{"num_disparities": 96, "subpixel": False, **overrides})
        return SGBMMatcher(cfg).compute(self.images.left, self.images.right)

    def test_half_scale_returns_full_size_and_close_depth(self):
        full = self._run(scale=1.0)
        half = self._run(scale=0.5)
        self.assertEqual(half.disp.shape, full.disp.shape)
        # 取近处物体前表面（GT 0.55–0.66 m），比较两种尺度的中位深度
        mask = (self.images.depth > 0.55) & (self.images.depth < 0.66)
        z_full = self.calib.focal_px * self.calib.baseline_m / np.median(full.disp[full.valid & mask])
        z_half = self.calib.focal_px * self.calib.baseline_m / np.median(half.disp[half.valid & mask])
        self.assertAlmostEqual(float(z_half), float(z_full), delta=0.05)
        self.assertAlmostEqual(float(z_half), 0.62, delta=0.05)
        self.assertGreater(half.valid.sum(), 1000)

    def test_roi_masks_outside(self):
        out = self._run(roi_xywh="0.0,0.5,1.0,0.5")
        self.assertFalse(out.valid[:240, :].any())
        self.assertTrue(out.valid[240:, :].any())

    def test_invalid_configs(self):
        with self.assertRaises(ValueError):
            load_config(overrides={"sgbm.scale": 0.0})
        with self.assertRaises(ValueError):
            load_config(overrides={"sgbm.num_disparities": 100})
        with self.assertRaises(ValueError):
            load_config(overrides={"sgbm.scale": 0.5, "sgbm.min_disparity": 2})
        with self.assertRaises(ValueError):
            load_config(overrides={"sgbm.roi_xywh": "0,0,2,1"})


if __name__ == "__main__":
    unittest.main()
