"""PC 端接收：ZMQ 订阅 + 按 pair_id 重组 + 解码。

配对完全由 PairAssembler 负责：乱序、丢帧、重复都被显式计数，不靠
"先左后右"的到达顺序。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from shared.messages import FramePacket, PairAssembler, PairingStats, StereoPair


def decode_packet(packet: FramePacket) -> np.ndarray:
    if packet.encoding == "jpeg":
        img = cv2.imdecode(np.frombuffer(packet.payload, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"JPEG 解码失败 pair={packet.pair_id} side={packet.side}")
        return img
    if packet.encoding == "raw_bgr8":
        expected = packet.width * packet.height * 3
        if len(packet.payload) != expected:
            raise ValueError(
                f"raw 载荷长度不符: {len(packet.payload)} != {expected} (pair={packet.pair_id})"
            )
        return np.frombuffer(packet.payload, np.uint8).reshape(packet.height, packet.width, 3).copy()
    raise ValueError(f"未知编码: {packet.encoding}")


@dataclass
class ReceivedPair:
    pair: StereoPair
    left: np.ndarray
    right: np.ndarray

    @property
    def pair_id(self) -> int:
        return self.pair.pair_id

    @property
    def ts_diff_ns(self) -> int:
        return self.pair.ts_diff_ns


class ZmqPairSource:
    def __init__(self, endpoint: str, stale_s: float = 1.0, poll_ms: int = 100) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("缺少 pyzmq：pip install pyzmq（或 python3-zmq）") from exc
        self.zmq = zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.RCVHWM, 8)
        self.sock.connect(endpoint)
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)
        self.assembler = PairAssembler(stale_s=stale_s)
        self.poll_ms = int(poll_ms)
        self.endpoint = endpoint

    @property
    def stats(self) -> PairingStats:
        return self.assembler.stats

    def pairs(
        self, max_pairs: Optional[int] = None, idle_timeout_s: Optional[float] = None
    ) -> Iterator[ReceivedPair]:
        emitted = 0
        import time

        deadline = None if idle_timeout_s is None else time.monotonic() + float(idle_timeout_s)
        try:
            while max_pairs is None or emitted < max_pairs:
                events = dict(self.poller.poll(self.poll_ms))
                if self.sock in events:
                    parts = self.sock.recv_multipart()
                    packet = FramePacket.from_parts(parts)
                    for pair in self.assembler.add(packet):
                        yield ReceivedPair(pair, decode_packet(pair.left), decode_packet(pair.right))
                        emitted += 1
                        deadline = (
                            None if idle_timeout_s is None else time.monotonic() + float(idle_timeout_s)
                        )
                self.assembler.flush_stale()
                if deadline is not None and time.monotonic() > deadline:
                    break
                time.sleep(0)  # 让出 GIL，避免空转占满核心
        finally:
            self.close()

    def close(self) -> None:
        self.sock.close(0)


def recording_pair_summary(stats: PairingStats) -> dict:
    out = stats.as_dict()
    out["open_pairs"] = None
    return out
