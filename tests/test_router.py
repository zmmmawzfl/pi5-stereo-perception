import unittest

import numpy as np

from pc.fuse.router import SOURCE_INVALID, SOURCE_LEARNED, SOURCE_SGBM, FuseRouter
from pc.stereo.base import DisparityResult


def make_inputs():
    h, w = 60, 100
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 255, (h, w), dtype=np.uint8)
    gray[:, 60:] = 100  # 右侧弱纹理

    d_l = np.full((h, w), 20.0, np.float32)
    valid = np.ones((h, w), bool)
    d_l[:, 0:10] = 0.0
    valid[:, 0:10] = False  # 空洞，交给学习型

    # 遮挡/不一致区：给出 30，但右目参考仍为 20
    d_l[:, 10:20] = 30.0
    right = np.full((h, w), 20.0, np.float32)
    right_valid = np.ones((h, w), bool)

    sgbm = DisparityResult(d_l, valid, "sgbm", right_disp=right, right_valid=right_valid)
    learned = DisparityResult(
        np.full((h, w), 24.0, np.float32), np.ones((h, w), bool), "learned"
    )
    return gray, sgbm, learned


class TestRouter(unittest.TestCase):
    def test_sgbm_priority_and_no_averaging(self):
        gray, sgbm, learned = make_inputs()
        fused = FuseRouter(lr_max_diff=1.0, texture_threshold=10.0).fuse(sgbm, learned, gray)
        textured = FuseRouter.texture_score(gray) >= 10.0

        sgbm_px = fused.source == SOURCE_SGBM
        learned_px = fused.source == SOURCE_LEARNED
        self.assertTrue(np.any(sgbm_px))
        self.assertTrue(np.any(learned_px))
        # 接受 SGBM 的像素只能取 SGBM 值，绝不能是 22（加权平均）
        np.testing.assert_allclose(fused.disp[sgbm_px], 20.0)
        np.testing.assert_allclose(fused.disp[learned_px], 24.0)
        self.assertFalse(np.any(fused.valid & ~(sgbm_px | learned_px)))

        # 弱纹理区不允许 SGBM 通过
        self.assertFalse(np.any(sgbm_px & ~textured))
        # 空洞被学习型填上
        self.assertTrue(np.all(learned_px[:, 0:10]))

    def test_lr_rejection_uses_learned(self):
        gray, sgbm, learned = make_inputs()
        fused = FuseRouter(lr_max_diff=1.0, texture_threshold=10.0).fuse(sgbm, learned, gray)
        self.assertGreater(fused.stats["lr_rejected_px"], 0)
        # 第 10~20 列不一致，且纹理正常 → 只能由学习型补
        self.assertTrue(np.all(fused.source[:, 10:20] == SOURCE_LEARNED))

    def test_without_learned_leaves_holes(self):
        gray, sgbm, _ = make_inputs()
        fused = FuseRouter(lr_max_diff=1.0, texture_threshold=10.0).fuse(sgbm, None, gray)
        self.assertFalse(np.any(fused.source == SOURCE_LEARNED))
        self.assertTrue(np.any(fused.source == SOURCE_INVALID))
        self.assertAlmostEqual(fused.coverage, fused.stats["sgbm_share"], places=6)

    def test_missing_right_disparity_falls_back(self):
        gray, sgbm, _ = make_inputs()
        sgbm.right_disp = None
        sgbm.right_valid = None
        fused = FuseRouter().fuse(sgbm, None, gray)
        self.assertFalse(fused.stats["lr_available"])
        self.assertTrue(fused.stats["sgbm_accepted_px"] > 0)

    def test_stats_are_auditable(self):
        gray, sgbm, learned = make_inputs()
        fused = FuseRouter().fuse(sgbm, learned, gray)
        self.assertEqual(fused.stats["policy"], "sgbm_priority_no_average")
        self.assertEqual(fused.stats["learned_used_px"], int((fused.source == SOURCE_LEARNED).sum()))
        self.assertGreaterEqual(fused.stats["learned_overlap_px"], 0)


if __name__ == "__main__":
    unittest.main()
