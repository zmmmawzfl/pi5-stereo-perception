import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pc.depth.pointcloud import colored_pointcloud, write_ply
from pc.pipeline import StereoPipeline
from pc.validate.metrics import coverage_report, object_error_rows
from shared.config import load_config
from tests.helpers import get_cached_calib, make_scene


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calib = get_cached_calib()
        cls.cfg = load_config()
        cls.scene = make_scene()
        cls.pipeline = StereoPipeline(cls.cfg, cls.calib)
        cls.render = cls.scene.render(seed=8)
        cls.result = cls.pipeline.process(cls.render.left, cls.render.right)

    def test_objects_and_contract(self):
        res = self.result
        self.assertEqual(len(res.objects), 3)
        self.assertEqual(res.fused.stats["policy"], "sgbm_priority_no_average")
        self.assertIn("sgbm", res.timings_ms)
        self.assertIn("total", res.timings_ms)
        payload = res.object_json()
        self.assertEqual(len(payload["objects"]), 3)
        for obj in payload["objects"]:
            self.assertEqual(len(obj["center_3d"]), 3)
            self.assertIn(obj["depth_source"], ("sgbm", "learned", "mixed", "unknown"))

    def test_depth_accuracy(self):
        rows = object_error_rows(self.result.objects, self.render.objects)
        self.assertEqual(len(rows), 3)
        dz = [abs(r["dz_m"]) for r in rows]
        self.assertLess(float(np.median(dz)), 0.02)
        self.assertLess(float(np.max(dz)), 0.05)

    def test_coverage(self):
        cov = coverage_report(self.result.fused, self.render.valid)
        self.assertGreater(cov.coverage, 0.1)
        self.assertGreaterEqual(cov.sgbm_px, 1)
        self.assertEqual(cov.learned_px, 0)
        self.assertIsNotNone(cov.coverage_on_reference_valid)

    def test_pointcloud_and_ply(self):
        pts, colors = colored_pointcloud(self.render.left, self.result.fused, self.calib, stride=4)
        self.assertEqual(pts.shape[1], 3)
        self.assertEqual(colors.shape, pts.shape)
        self.assertGreater(len(pts), 100)
        with tempfile.TemporaryDirectory() as tmp:
            p = write_ply(Path(tmp) / "cloud.ply", pts, colors)
            data = p.read_bytes()
            self.assertTrue(data.startswith(b"ply"))
            self.assertIn(b"element vertex", data)

    def test_epipolar_check(self):
        epi = self.pipeline.epipolar_check(self.render.left, self.render.right)
        self.assertIn("n_matches", epi)
        if epi["n_matches"]:
            # ORB 关键点坐标是整数，垂直残差天然量化到 1 px；这里只做粗检
            self.assertLessEqual(epi["median_abs_dy_px"], 2.0)


if __name__ == "__main__":
    unittest.main()
