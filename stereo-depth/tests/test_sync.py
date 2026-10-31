import unittest

from pi.sync.p_controller import PhaseLockController, simulate_sync


class TestController(unittest.TestCase):
    def test_sign_and_clamp(self):
        c = PhaseLockController(kp=0.05, base_period_us=33333, max_step_us=300)
        # 从动时间戳偏大 10ms → 周期应缩短，且不超过 max_step
        p = c.update(0, 10_000_000)
        self.assertAlmostEqual(p, 33333 - 300)
        # 偏小 10ms → 周期应拉长
        p2 = PhaseLockController(kp=0.05, base_period_us=33333, max_step_us=300).update(10_000_000, 0)
        self.assertAlmostEqual(p2, 33333 + 300)

    def test_stays_in_bounds(self):
        c = PhaseLockController(kp=0.5, min_frame_us=30000, max_frame_us=34000,
                                max_step_us=5000, base_period_us=32000)
        for _ in range(100):
            p = c.update(0, 10_000_000)
            self.assertGreaterEqual(p, 30000)
            self.assertLessEqual(p, 34000)

    def test_target_offset(self):
        c = PhaseLockController(kp=0.05, target_offset_ns=2_000_000, base_period_us=33333,
                                max_step_us=300)
        p = c.update(0, 2_000_000)  # 正好等于目标 → 不前馈修正
        self.assertEqual(p, 33333)

    def test_invalid_kp(self):
        with self.assertRaises(ValueError):
            PhaseLockController(kp=0)
        with self.assertRaises(ValueError):
            PhaseLockController(kp=1.0)


class TestSimulation(unittest.TestCase):
    def test_control_reduces_phase_error(self):
        r = simulate_sync(n_frames=400, kp=0.05, control_from=50, settle_frac=0.25)
        before = abs(r.stats["before"]["median"])
        after = abs(r.stats["after"]["median"])
        self.assertGreater(before, 5.0, "未加控制时应看到明显漂移")
        self.assertLess(after, 0.05, f"控制后应接近 0，实际 {after} ms")
        self.assertLess(r.stats["after_iqr_ms"], 0.1)

    def test_no_control_drifts(self):
        r = simulate_sync(n_frames=200, kp=0.05, control_from=10_000, settle_frac=0.5)
        self.assertEqual(r.stats["after"]["n"], 0)  # 从未进入稳态窗口
        spread = max(r.before_offset_ms) - min(r.before_offset_ms)
        self.assertGreater(spread, 5.0)


if __name__ == "__main__":
    unittest.main()
