"""采集后端统一接口。

实机路径（Picamera2）、无相机路径（SyntheticBackend）、回放路径
（ReplayBackend）产出同一个 CapturedPair：左/右 BGR、各自的 SensorTimestamp、
递增的 pair_id。上层服务不关心帧从哪来。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class CapturedPair:
    left: np.ndarray
    right: np.ndarray
    ts_left_ns: int
    ts_right_ns: int
    seq: int

    @property
    def pair_id(self) -> int:
        return self.seq

    @property
    def ts_diff_ns(self) -> int:
        return self.ts_right_ns - self.ts_left_ns


class CameraBackend(Protocol):
    width: int
    height: int
    fps: int

    def start(self) -> None: ...
    def read(self) -> CapturedPair: ...
    def close(self) -> None: ...
    def set_slave_frame_duration(self, period_us: int) -> None: ...
