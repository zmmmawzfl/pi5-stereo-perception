"""推流 + 时间戳旁路。

本模块发的是"帧 + 元数据"的 ZMQ PUB 通道：
  * 视频帧用 JPEG（可调质量），低延迟、实现简单；
  * 每帧 header 里带 SensorTimestamp 与 pair_id，配对不依赖到达顺序；
  * 另开一个 meta PUB，只发 ``{pair_id, ts_L, ts_R, skew_ms}``，便于 PC 端
    单独监控同步质量，也方便将来换成 rpi-webrtc-streamer 时把时间戳旁路
    接到 WebRTC 数据通道。

生产环境若要 H.264 硬/软编码 + WebRTC，用 rpi-webrtc-streamer 推视频，
时间戳仍走这里的 meta 通道；ZMQ 的 JPEG 通道作为调试/备用链路。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

from shared.config import AppConfig, StreamConfig
from shared.messages import FramePacket, make_packet
from shared.recording import PairRecorder

from pi.capture.base import CapturedPair


@dataclass
class PublishStats:
    published: int = 0
    dropped: int = 0
    bytes_sent: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "published": self.published,
            "dropped": self.dropped,
            "bytes_sent": self.bytes_sent,
            "avg_kb": round(self.bytes_sent / max(self.published, 1) / 1024.0, 1),
        }


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


class NullPublisher:
    def __init__(self) -> None:
        self.stats = PublishStats()

    def publish(self, pair: CapturedPair) -> None:
        self.stats.published += 1

    def close(self) -> None:
        pass


class ZmqPublisher:
    def __init__(
        self,
        endpoint: str,
        jpeg_quality: int = 90,
        hwm: int = 4,
        meta_endpoint: Optional[str] = None,
    ) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("缺少 pyzmq：pip install pyzmq（或 python3-zmq）") from exc
        self.zmq = zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, int(hwm))
        self.sock.bind(endpoint)
        self.meta_sock = None
        if meta_endpoint:
            self.meta_sock = self.ctx.socket(zmq.PUB)
            self.meta_sock.setsockopt(zmq.SNDHWM, 1000)
            self.meta_sock.bind(meta_endpoint)
        self.endpoint = endpoint
        self.jpeg_quality = int(jpeg_quality)
        self.stats = PublishStats()
        self._wall_offset_ns = time.time_ns() - time.monotonic_ns()

    def publish(self, pair: CapturedPair) -> None:
        h, w = pair.left.shape[:2]
        for side, img, ts in (
            ("left", pair.left, pair.ts_left_ns),
            ("right", pair.right, pair.ts_right_ns),
        ):
            payload = encode_jpeg(img, self.jpeg_quality)
            packet = make_packet(
                pair_id=pair.pair_id,
                seq=pair.seq,
                side=side,
                sensor_ts_ns=ts,
                width=w,
                height=h,
                encoding="jpeg",
                payload=payload,
            )
            try:
                self.sock.send_multipart(packet.to_parts(), flags=self.zmq.NOBLOCK)
                self.stats.published += 1
                self.stats.bytes_sent += len(payload)
            except self.zmq.Again:
                self.stats.dropped += 1

        if self.meta_sock is not None:
            meta = {
                "pair_id": pair.pair_id,
                "seq": pair.seq,
                "ts_L": pair.ts_left_ns,
                "ts_R": pair.ts_right_ns,
                "skew_ms": pair.ts_diff_ns / 1e6,
                "wall_ns": time.time_ns(),
                "encoding": "jpeg",
            }
            try:
                self.meta_sock.send_string(json.dumps(meta), flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                self.stats.dropped += 1

    def close(self) -> None:
        self.sock.close(0)
        if self.meta_sock is not None:
            self.meta_sock.close(0)


class FilePublisher:
    """无损落盘（PNG），用于压缩伪影量化与离线复算。"""

    def __init__(self, root: str, image_format: str = "png", jpeg_quality: int = 95) -> None:
        self.recorder = PairRecorder(root, image_format=image_format, jpeg_quality=jpeg_quality)
        self.stats = PublishStats()

    def publish(self, pair: CapturedPair) -> None:
        self.recorder.add(pair.left, pair.right, pair.ts_left_ns, pair.ts_right_ns, pair.pair_id)
        self.stats.published += 1

    def close(self) -> None:
        self.recorder.close()


class TeePublisher:
    def __init__(self, *publishers) -> None:
        self.publishers = [p for p in publishers if p is not None]
        self.stats = PublishStats()

    def publish(self, pair: CapturedPair) -> None:
        for p in self.publishers:
            p.publish(pair)
        self.stats.published += 1

    def close(self) -> None:
        for p in self.publishers:
            p.close()


def build_publisher(cfg: StreamConfig, record_root: Optional[str] = None):
    """按配置构建发布器；record_root 非空时额外无损落盘。"""
    pub = None
    transport = (cfg.transport or "none").lower()
    if transport == "zmq":
        meta = None
        if cfg.endpoint.startswith("tcp://") and cfg.endpoint.rsplit(":", 1)[-1] == "5555":
            meta = cfg.endpoint.rsplit(":", 1)[0] + ":5556"
        pub = ZmqPublisher(cfg.endpoint, cfg.jpeg_quality, meta_endpoint=meta)
    elif transport in ("none", ""):
        pub = NullPublisher()
    else:
        raise ValueError(f"未知 stream.transport: {cfg.transport}")

    if record_root:
        return TeePublisher(pub, FilePublisher(record_root))
    return pub
