import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pc.calib.calibrate import CalibrationResult, calibrate_stereo
from shared.synthetic import generate_calibration_views


class TestStereoCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = np.array([-0.06, 0.01, 0.0, 0.0, 0.0])
        cls.views = generate_calibration_views(
            n_views=14, seed=3, D_left=cls.D, D_right=cls.D, baseline=0.083
        )
        cls.calib = calibrate_stereo(
            cls.views["left"], cls.views["right"], pattern=(9, 6), square_m=0.03, min_views=10
        )

    def test_baseline_recovered(self):
        self.assertAlmostEqual(self.calib.baseline_m, 0.083, delta=0.083 * 0.015)

    def test_reprojection_and_epipolar(self):
        self.assertLess(self.calib.rms_stereo, 0.5)
        self.assertLess(self.calib.rms_mono_left, 0.3)
        epi = self.calib.meta["epipolar_dy"]["median_abs_px"]
        self.assertLess(epi, 0.2)

    def test_focal_consistent_with_depth(self):
        self.assertGreater(self.calib.focal_px, 400)
        z = self.calib.disparity_to_depth(np.array([self.calib.focal_px * self.calib.baseline_m], np.float32))
        self.assertAlmostEqual(float(z[0]), 1.0, places=4)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "calib.npz"
            self.calib.save(p)
            c2 = CalibrationResult.load(p)
            self.assertAlmostEqual(c2.baseline_m, self.calib.baseline_m, places=9)
            self.assertAlmostEqual(c2.focal_px, self.calib.focal_px, places=9)
            np.testing.assert_allclose(c2.K1, self.calib.K1, rtol=1e-12)

    def test_wrong_view_count_rejected(self):
        from pc.calib.calibrate import CalibrationError

        with self.assertRaises(CalibrationError):
            calibrate_stereo(self.views["left"][:3], self.views["right"][:3], min_views=8)


if __name__ == "__main__":
    unittest.main()
