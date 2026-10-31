import unittest

import numpy as np

from pc.detect.base import Detection
from pc.fuse.router import SOURCE_SGBM, FusedDisparity
from pc.objects.aggregator import ObjectAggregator, fit_ground_plane
from shared.config import ObjectConfig
from shared.synthetic import default_scene
from tests.helpers import make_fake_calib


class TestGroundPlane(unittest.TestCase):
    def test_recovers_plane(self):
        rng = np.random.default_rng(0)
        n = 3000
        X = rng.uniform(-1, 1, n)
        Z = rng.uniform(0.6, 3.0, n)
        Y = 0.25 + rng.normal(0, 0.002, n)
        plane = fit_ground_plane(np.column_stack([X, Y, Z]), iters=200, thresh_m=0.01)
        self.assertIsNotNone(plane)
        self.assertAlmostEqual(plane.r, 0.25, delta=0.003)
        self.assertAlmostEqual(plane.p, 0.0, delta=0.01)
        self.assertAlmostEqual(plane.q, 0.0, delta=0.01)
        self.assertGreater(plane.inlier_ratio, 0.9)
        self.assertTrue(plane.is_plausible())

    def test_rejects_tilted_plane(self):
        from pc.objects.aggregator import GroundPlane

        self.assertFalse(GroundPlane(p=2.0, q=0.0, r=0.25, inlier_ratio=0.9, rms_m=0.001, n_points=100).is_plausible())
        self.assertFalse(GroundPlane(p=0.0, q=0.0, r=-0.2, inlier_ratio=0.9, rms_m=0.001, n_points=100).is_plausible())

    def test_too_few_points(self):
        self.assertIsNone(fit_ground_plane(np.zeros((10, 3))))


class TestAggregator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calib = make_fake_calib()
        cls.render = default_scene().render(seed=4)
        fused = FusedDisparity(
            disp=cls.render.disp.copy(),
            valid=cls.render.valid.copy(),
            source=np.full(cls.render.disp.shape, SOURCE_SGBM, np.uint8),
            stats={},
        )
        cls.fused = fused
        cls.dets = [
            Detection("object", tuple(o["bbox_xyxy"]), 0.9) for o in cls.render.objects
        ]
        cls.result = ObjectAggregator(cls.calib, ObjectConfig()).aggregate(cls.dets, fused)

    def test_all_objects_aggregated(self):
        self.assertEqual(len(self.result.objects), 3)
        self.assertIsNotNone(self.result.ground_plane)
        self.assertTrue(self.result.ground_plane.is_plausible())

    def test_depth_and_center(self):
        for obj, gt in zip(self.result.objects, self.render.objects):
            center = np.array(obj.center_3d)
            gt_c = np.array(gt["gt_surface_center_m"])
            self.assertLess(abs(center[2] - gt_c[2]), 0.01)
            self.assertLess(np.linalg.norm(center - gt_c), 0.03)
            self.assertTrue(obj.ground_applied)
            self.assertEqual(obj.depth_source, "sgbm")

    def test_unconstrained_kept(self):
        for obj in self.result.objects:
            self.assertEqual(len(obj.unconstrained_center_3d), 3)
            self.assertNotEqual(obj.center_3d, obj.unconstrained_center_3d)

    def test_object_json_contract(self):
        d = self.result.objects[0].to_dict()
        for key in ("class", "center_3d", "bbox_2d", "confidence", "depth_source"):
            self.assertIn(key, d)
        self.assertEqual(len(d["center_3d"]), 3)

    def test_skips_object_without_valid_pixels(self):
        fused = FusedDisparity(
            disp=self.render.disp.copy(),
            valid=self.render.valid.copy(),
            source=np.full(self.render.disp.shape, SOURCE_SGBM, np.uint8),
            stats={},
        )
        x1, y1, x2, y2 = self.render.objects[0]["bbox_xyxy"]
        fused.valid[y1 : y2 + 1, x1 : x2 + 1] = False
        res = ObjectAggregator(self.calib, ObjectConfig()).aggregate(self.dets[:1], fused)
        self.assertEqual(len(res.objects), 0)
        self.assertEqual(res.diagnostics["skipped"][0]["reason"], "insufficient_valid_pixels")


if __name__ == "__main__":
    unittest.main()
