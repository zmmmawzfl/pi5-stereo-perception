import time
import unittest

from pc.validate.runtime import ResourceMonitor


class RuntimeMonitorTest(unittest.TestCase):
    def test_snapshot_has_resources(self):
        m = ResourceMonitor()
        time.sleep(0.02)
        s = m.snapshot().as_dict()
        self.assertGreaterEqual(s["wall_s"], 0.02)
        self.assertGreaterEqual(s["cpu_time_s"], 0.0)
        self.assertGreaterEqual(s["cpu_percent"], 0.0)
        if s["rss_mb"] is not None:
            self.assertGreater(s["rss_mb"], 1.0)


if __name__ == "__main__":
    unittest.main()
