"""跨 Pi / PC 的帧消息格式与配对逻辑。

设计要点（阶段一前必须定死）：
  * 每帧自带 ``pair_id``：配对只认 id，绝不依赖"先收到左再收到右"。
  * 每帧自带 ``sensor_ts_ns``（Pi 端 SensorTimestamp，单调、同源）与
    ``capture_wall_ns``（墙钟，仅用于跨机诊断）。
  * 传输通道与配对逻辑解耦：ZMQ 多段消息、文件、内存队列共用同一 header。

线格式（bytes / ZMQ 多段）::

    MAGIC(4) | header_len(uint32 LE) | header(JSON utf-8) | payload

ZMQ 优先使用多段 ``[header_json, payload]``，避免一次大拷贝。
"""
from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

MAGIC = b"SDP1"
PROTOCOL_VERSION = 1
HEADER_PREFIX = struct.Struct("<4sI")

# 编码标识：raw_bgr8 为无损原始帧，jpeg / h264 为压缩通道
ENCODINGS = ("raw_bgr8", "jpeg", "h264")


class ProtocolError(ValueError):
    """线格式非法。"""


@dataclass(frozen=True)
class FramePacket:
    """单目一帧 + 其元数据。"""

    pair_id: int
    seq: int
    side: str  # "left" | "right"
    sensor_ts_ns: int
    capture_wall_ns: int
    width: int
    height: int
    encoding: str
    payload: bytes
    version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ProtocolError(f"side 必须是 left/right，收到 {self.side!r}")
        if self.encoding not in ENCODINGS:
            raise ProtocolError(f"未知编码 {self.encoding!r}")
        if self.width <= 0 or self.height <= 0:
            raise ProtocolError("width/height 必须为正")
        if not self.payload:
            raise ProtocolError("payload 为空")

    # ---- 头部 ----
    def header(self) -> Dict[str, Any]:
        return {
            "v": self.version,
            "pair_id": int(self.pair_id),
            "seq": int(self.seq),
            "side": self.side,
            "sensor_ts_ns": int(self.sensor_ts_ns),
            "capture_wall_ns": int(self.capture_wall_ns),
            "w": int(self.width),
            "h": int(self.height),
            "enc": self.encoding,
            "bytes": len(self.payload),
        }

    def header_json(self) -> bytes:
        return json.dumps(self.header(), separators=(",", ":"), ensure_ascii=True).encode()

    @classmethod
    def from_header(cls, header: Dict[str, Any], payload: bytes) -> "FramePacket":
        if int(header.get("v", -1)) != PROTOCOL_VERSION:
            raise ProtocolError(f"协议版本不支持: {header.get('v')!r}")
        if int(header.get("bytes", -1)) != len(payload):
            raise ProtocolError(
                f"payload 长度不符: header={header.get('bytes')} actual={len(payload)}"
            )
        return cls(
            pair_id=int(header["pair_id"]),
            seq=int(header["seq"]),
            side=str(header["side"]),
            sensor_ts_ns=int(header["sensor_ts_ns"]),
            capture_wall_ns=int(header["capture_wall_ns"]),
            width=int(header["w"]),
            height=int(header["h"]),
            encoding=str(header["enc"]),
            payload=payload,
        )

    # ---- 编解码 ----
    def encode(self) -> bytes:
        header = self.header_json()
        return HEADER_PREFIX.pack(MAGIC, len(header)) + header + self.payload

    @classmethod
    def decode(cls, blob: bytes) -> "FramePacket":
        if len(blob) < HEADER_PREFIX.size:
            raise ProtocolError("数据短于帧头")
        magic, hlen = HEADER_PREFIX.unpack_from(blob, 0)
        if magic != MAGIC:
            raise ProtocolError(f"magic 不匹配: {magic!r}")
        start = HEADER_PREFIX.size
        end = start + hlen
        if end > len(blob):
            raise ProtocolError("header 长度越界")
        try:
            header = json.loads(blob[start:end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"header 解析失败: {exc}") from exc
        return cls.from_header(header, blob[end:])

    def to_parts(self) -> List[bytes]:
        """ZMQ 多段：[header, payload]。"""
        return [self.header_json(), self.payload]

    @classmethod
    def from_parts(cls, parts: Iterable[bytes]) -> "FramePacket":
        parts = list(parts)
        if len(parts) != 2:
            raise ProtocolError(f"需要 2 段消息，收到 {len(parts)} 段")
        try:
            header = json.loads(parts[0].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"header 解析失败: {exc}") from exc
        return cls.from_header(header, parts[1])

    # ---- 便捷 ----
    @property
    def ts_ms(self) -> float:
        return self.sensor_ts_ns / 1e6


@dataclass(frozen=True)
class StereoPair:
    """一对已配好的左右帧。"""

    pair_id: int
    seq: int
    left: FramePacket
    right: FramePacket

    def __post_init__(self) -> None:
        if self.left.pair_id != self.pair_id or self.right.pair_id != self.pair_id:
            raise ProtocolError("pair_id 不一致")
        if self.left.side != "left" or self.right.side != "right":
            raise ProtocolError("左右帧顺序错误")

    @property
    def ts_diff_ns(self) -> int:
        """右 - 左。中位数应接近 0；若设备存在固定偏移，由服务端补偿。"""
        return self.right.sensor_ts_ns - self.left.sensor_ts_ns

    @property
    def avg_wall_ns(self) -> int:
        return (self.left.capture_wall_ns + self.right.capture_wall_ns) // 2


@dataclass
class PairingStats:
    """配对质量计数器。"""

    received: int = 0
    emitted: int = 0
    duplicates: int = 0
    orphans: int = 0
    out_of_order: int = 0
    ts_diff_ms: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        from .stats import summarize

        return {
            "received": self.received,
            "emitted": self.emitted,
            "duplicates": self.duplicates,
            "orphans": self.orphans,
            "out_of_order": self.out_of_order,
            "ts_diff_ms": summarize(self.ts_diff_ms),
        }


class PairAssembler:
    """按 pair_id 重组左右帧，与传输方式无关。

    - 乱序到达：只要 pair_id 相同即可配；
    - 重复到达：计数并丢弃，不覆盖已收集的一侧；
    - 单边丢失：超时后计入 orphan 并淘汰，绝不跨 pair 拼接。
    """

    def __init__(self, stale_s: float = 1.0, max_open: int = 64) -> None:
        self.stale_s = float(stale_s)
        self.max_open = int(max_open)
        self._open: Dict[int, Dict[str, FramePacket]] = {}
        self._first_seen: Dict[int, float] = {}
        self._last_emitted = -1
        self.stats = PairingStats()

    @property
    def open_pairs(self) -> int:
        return len(self._open)

    def add(self, packet: FramePacket, now_ns: Optional[int] = None) -> List[StereoPair]:
        """送入一帧，返回本次凑齐的所有 StereoPair（通常 0 或 1 个）。"""
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        self.stats.received += 1

        if packet.pair_id < self._last_emitted:
            # 已经越过这个 pair：迟到帧直接丢弃，但单独计数（不是重复）
            self.stats.out_of_order += 1
            return []
        if packet.pair_id == self._last_emitted:
            self.stats.duplicates += 1
            return []

        slot = self._open.setdefault(packet.pair_id, {})
        self._first_seen.setdefault(packet.pair_id, now)

        if packet.side in slot:
            self.stats.duplicates += 1
            return []

        slot[packet.side] = packet
        self._evict_overflow(now)

        out: List[StereoPair] = []
        if "left" in slot and "right" in slot:
            pair = StereoPair(
                pair_id=packet.pair_id,
                seq=packet.seq,
                left=slot["left"],
                right=slot["right"],
            )
            del self._open[packet.pair_id]
            self._first_seen.pop(packet.pair_id, None)

            self._last_emitted = max(self._last_emitted, packet.pair_id)

            self.stats.emitted += 1
            self.stats.ts_diff_ms.append(pair.ts_diff_ns / 1e6)
            out.append(pair)
        return out

    def flush_stale(self, now_ns: Optional[int] = None, force: bool = False) -> List[int]:
        """淘汰超时或强制的未配对槽位，返回被淘汰的 pair_id 列表。"""
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        cutoff = self.stale_s * 1e9
        dead = [
            pid
            for pid, ts in self._first_seen.items()
            if force or (now - ts) > cutoff
        ]
        for pid in dead:
            self._open.pop(pid, None)
            self._first_seen.pop(pid, None)
            self.stats.orphans += 1
        return dead

    def _evict_overflow(self, now_ns: int) -> None:
        if len(self._open) <= self.max_open:
            return
        # 淘汰最老的槽位，防止内存无界增长
        oldest = sorted(self._first_seen.items(), key=lambda kv: kv[1])
        for pid, _ in oldest[: len(self._open) - self.max_open]:
            self._open.pop(pid, None)
            self._first_seen.pop(pid, None)
            self.stats.orphans += 1


def make_packet(
    pair_id: int,
    seq: int,
    side: str,
    sensor_ts_ns: int,
    width: int,
    height: int,
    encoding: str,
    payload: bytes,
    capture_wall_ns: Optional[int] = None,
) -> FramePacket:
    """构造 FramePacket；墙钟默认取当前时间。"""
    return FramePacket(
        pair_id=int(pair_id),
        seq=int(seq),
        side=side,
        sensor_ts_ns=int(sensor_ts_ns),
        capture_wall_ns=int(capture_wall_ns if capture_wall_ns is not None else time.time_ns()),
        width=int(width),
        height=int(height),
        encoding=encoding,
        payload=payload,
    )


def with_payload(packet: FramePacket, payload: bytes, encoding: Optional[str] = None) -> FramePacket:
    """替换载荷（例如编码后重新打包）。"""
    return replace(
        packet,
        payload=payload,
        encoding=encoding or packet.encoding,
        capture_wall_ns=packet.capture_wall_ns,
    )
