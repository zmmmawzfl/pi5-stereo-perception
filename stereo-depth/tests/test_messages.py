import unittest

import numpy as np

from shared.messages import (
    FramePacket,
    PairAssembler,
    ProtocolError,
    StereoPair,
    make_packet,
)


def pkt(pair_id, side, seq=0, ts=None, payload=b"abc"):
    return make_packet(
        pair_id=pair_id, seq=seq, side=side,
        sensor_ts_ns=ts if ts is not None else 1_000_000_000 + pair_id * 33_333_333,
        width=4, height=3, encoding="jpeg", payload=payload, capture_wall_ns=123,
    )


class TestWireFormat(unittest.TestCase):
    def test_bytes_roundtrip(self):
        p = pkt(7, "left", payload=b"\x00\x01\x02")
        q = FramePacket.decode(p.encode())
        self.assertEqual(q.pair_id, 7)
        self.assertEqual(q.side, "left")
        self.assertEqual(q.payload, b"\x00\x01\x02")
        self.assertEqual(q.header()["bytes"], 3)

    def test_parts_roundtrip(self):
        p = pkt(9, "right")
        q = FramePacket.from_parts(p.to_parts())
        self.assertEqual(q.pair_id, 9)

    def test_bad_magic(self):
        blob = bytearray(pkt(1, "left").encode())
        blob[0:4] = b"XXXX"
        with self.assertRaises(ProtocolError):
            FramePacket.decode(bytes(blob))

    def test_truncated(self):
        with self.assertRaises(ProtocolError):
            FramePacket.decode(b"SDP1")

    def test_payload_length_mismatch(self):
        p = pkt(1, "left")
        parts = p.to_parts()
        parts[1] = b"short"
        with self.assertRaises(ProtocolError):
            FramePacket.from_parts(parts)

    def test_empty_payload_rejected(self):
        with self.assertRaises(ProtocolError):
            pkt(1, "left", payload=b"")

    def test_bad_side(self):
        with self.assertRaises(ProtocolError):
            pkt(1, "middle")


class TestPairAssembler(unittest.TestCase):
    def test_out_of_order_arrival_pairs_by_id(self):
        a = PairAssembler()
        a.add(pkt(3, "right"), now_ns=0)
        out = a.add(pkt(3, "left"), now_ns=0)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], StereoPair)
        self.assertEqual(out[0].pair_id, 3)
        self.assertEqual(out[0].ts_diff_ns, 0)

    def test_never_pairs_across_ids(self):
        a = PairAssembler()
        a.add(pkt(1, "left"), now_ns=0)
        self.assertEqual(a.add(pkt(2, "right"), now_ns=0), [])
        self.assertEqual(a.open_pairs, 2)
        a.flush_stale(now_ns=int(2e9))
        self.assertEqual(a.stats.orphans, 2)

    def test_duplicate_side_ignored(self):
        a = PairAssembler()
        a.add(pkt(1, "left"), now_ns=0)
        a.add(pkt(1, "left"), now_ns=0)
        self.assertEqual(a.stats.duplicates, 1)
        out = a.add(pkt(1, "right"), now_ns=0)
        self.assertEqual(len(out), 1)

    def test_late_pair_dropped_and_counted(self):
        a = PairAssembler()
        a.add(pkt(5, "left"), now_ns=0)
        a.add(pkt(5, "right"), now_ns=0)
        a.add(pkt(3, "left"), now_ns=0)
        a.add(pkt(3, "right"), now_ns=0)
        self.assertEqual(a.stats.out_of_order, 2)
        self.assertEqual(a.stats.emitted, 1)

    def test_stale_orphan_evicted(self):
        a = PairAssembler(stale_s=1.0)
        a.add(pkt(1, "left"), now_ns=0)
        dead = a.flush_stale(now_ns=int(1.5e9))
        self.assertEqual(dead, [1])
        self.assertEqual(a.stats.orphans, 1)
        self.assertEqual(a.open_pairs, 0)

    def test_ts_diff_stats_recorded(self):
        a = PairAssembler()
        left = pkt(1, "left", ts=1_000_000_000)
        right = pkt(1, "right", ts=1_002_000_000)
        a.add(left, now_ns=0)
        a.add(right, now_ns=0)
        self.assertAlmostEqual(a.stats.ts_diff_ms[0], 2.0, places=6)

    def test_max_open_evicts(self):
        a = PairAssembler(max_open=3)
        for i in range(6):
            a.add(pkt(i, "left"), now_ns=i * 1000)
        self.assertLessEqual(a.open_pairs, 3)
        self.assertGreater(a.stats.orphans, 0)


if __name__ == "__main__":
    unittest.main()
