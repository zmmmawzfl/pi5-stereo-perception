"""无相机时的采集后端：合成帧 + 可仿真的双相机时间戳。

用途：
  * 摄像头未到货时把 Pi 服务、配对、推流、落盘整条链路跑通；
  * 同步环回归测试：从动相机周期有偏差，不开控制时相位误差线性漂移，
    开控制后被压回目标附近。

图像默认只渲染一次并缓存——同步基线只关心时间戳，不需要每帧新图。
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from shared.synthetic import SyntheticScene, default_scene

from .base import CapturedPair


class SyntheticBackend:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        master_period_us: Optional[float] = None,
        slave_period_us: Optional[float] = None,
        jitter_us: float = 0.10,
        initial_offset_ms: float = 6.0,
        seed: int = 0,
        scene: Optional[SyntheticScene] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        base = 1e6 / float(fps)
        self.master_period_us = float(master_period_us if master_period_us is not None else base)
        self.slave_period_us = float(slave_period_us if slave_period_us is not None else base * 1.008)
        self.jitter_us = float(jitter_us)
        self.initial_offset_ms = float(initial_offset_ms)
        self._scene = scene or default_scene(width=width, height=height)
        self._rng = np.random.default_rng(seed)
        self._render: Optional[object] = None
        self._t_master = 1_000_000_000.0
        self._t_slave = self._t_master + self.initial_offset_ms * 1e6
        self._seq = 0
        self._started = False

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._render is None:
            self._render = self._scene.render(seed=1)
        self._started = True

    def read(self) -> CapturedPair:
        if not self._started:
            self.start()
        assert self._render is not None
        # 每帧独立抖动：真实 SensorTimestamp 也有 ~0.1ms 量级的抖动
        self._t_master += self.master_period_us * 1000.0 + self._rng.normal(
            0.0, self.jitter_us * 1000.0
        )
        self._t_slave += self.slave_period_us * 1000.0 + self._rng.normal(
            0.0, self.jitter_us * 1000.0
        )
        pair = CapturedPair(
            left=self._render.left.copy(),
            right=self._render.right.copy(),
            ts_left_ns=int(self._t_master),
            ts_right_ns=int(self._t_slave),
            seq=self._seq,
        )
        self._seq += 1
        return pair

    def set_slave_frame_duration(self, period_us: int) -> None:
        self.slave_period_us = float(period_us)

    def close(self) -> None:
        self._started = False


class ReplayBackend:
    """把录制目录当作相机回放；时间戳原样取出。"""

    def __init__(self, root: str, loop: bool = False) -> None:
        from shared.recording import PairReplay

        self._replay = PairReplay(root)
        self.width = 0
        self.height = 0
        self.fps = 30
        self.loop = loop
        self._iter = None
        self._seq = 0

    def start(self) -> None:
        self._iter = iter(self._replay)

    def read(self) -> CapturedPair:
        if self._iter is None:
            self.start()
        assert self._iter is not None
        try:
            rec, left, right = next(self._iter)
        except StopIteration:
            if not self.loop:
                raise
            self.start()
            assert self._iter is not None
            rec, left, right = next(self._iter)
        self.height, self.width = left.shape[:2]
        return CapturedPair(left, right, rec.ts_left_ns, rec.ts_right_ns, self._seq)

    def set_slave_frame_duration(self, period_us: int) -> None:
        # 回放模式无法改周期
        pass

    def close(self) -> None:
        self._iter = None
