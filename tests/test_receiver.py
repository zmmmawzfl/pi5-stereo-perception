import time
import unittest
import uuid

import numpy as np

from pi.capture.base import CapturedPair
from pi.stream.publisher import ZmqPublisher
from pc.receive.subscriber import ZmqPairSource

try:
    import zmq  # noqa: F401
    HAS_ZMQ = True
except Exception:
    HAS_ZMQ = False


@unittest.skipUnless(HAS_ZMQ, "未安装 pyzmq")
class TestZeroMqPairing(unittest.TestCase):
    def test_publish_subscribe_pairs_by_id(self):
        endpoint = f"inproc://sd_test_{uuid.uuid4().hex}"
        pub = ZmqPublisher(endpoint, jpeg_quality=85)
        src = ZmqPairSource(endpoint, stale_s=0.5, poll_ms=20)
        time.sleep(0.2)  # 等订阅关系建立

        base = 1_000_000_000
        for i in range(3):
            left = np.full((48, 64, 3), 20 + i, np.uint8)
            right = np.full((48, 64, 3), 30 + i, np.uint8)
            pub.publish(CapturedPair(left, right, base + i * 33_333_333,
                                     base + i * 33_333_333 + 1_000_000, seq=i))

        got = []
        try:
            for rp in src.pairs(max_pairs=3, idle_timeout_s=5.0):
                got.append(rp)
        finally:
            pub.close()

        self.assertEqual(len(got), 3)
        self.assertEqual([g.pair_id for g in got], [0, 1, 2])
        self.assertEqual(got[0].left.shape, (48, 64, 3))
        self.assertAlmostEqual(got[1].ts_diff_ns, 1_000_000)
        self.assertEqual(src.stats.emitted, 3)
        self.assertEqual(src.stats.orphans, 0)


if __name__ == "__main__":
    unittest.main()
