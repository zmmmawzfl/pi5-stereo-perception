"""P 控制同步环：微调从动相机的 FrameDurationLimits。

设计（这是在仿真里把误差压下去的那一版）：
  * 周期指令 = 标称周期 - Kp * 相位误差，而不是在上一周期上累加。
    累加式写法等价于对周期做积分，闭环极点会跑到单位圆外（仿真里会
    直接发散，别问怎么知道的）。
  * 标称周期来自 fps，是前馈项：误差为 0 时从动相机回到与主动相同的
    周期，不会留下稳态偏差。
  * 输出限幅 ±max_step_us，并限制在 [min_frame_us, max_frame_us]，避免
    一步把相机推到不可用区间。

符号约定：error = ts_slave - ts_master - target。从动相机时间戳偏大，
说明它的帧来得更晚，需要**缩短**周期追上去，因此 step = -Kp * error。
输出单位 µs，写入 FrameDurationLimits=(v, v)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from shared.stats import summarize


@dataclass
class PhaseLockState:
    period_us: float
    last_error_us: Optional[float] = None
    steps: int = 0


class PhaseLockController:
    def __init__(
        self,
        kp: float = 0.05,
        target_offset_ns: int = 0,
        min_frame_us: int = 30_000,
        max_frame_us: int = 40_000,
        max_step_us: int = 300,
        base_period_us: Optional[float] = None,
    ) -> None:
        if not (0.0 < kp < 1.0):
            raise ValueError("kp 必须在 (0,1)，推荐从 0.05 起")
        if min_frame_us >= max_frame_us:
            raise ValueError("min_frame_us 必须小于 max_frame_us")
        self.kp = float(kp)
        self.target_offset_ns = int(target_offset_ns)
        self.min_frame_us = int(min_frame_us)
        self.max_frame_us = int(max_frame_us)
        self.max_step_us = int(max_step_us)
        self.base_period_us = float(
            base_period_us if base_period_us is not None else (min_frame_us + max_frame_us) / 2
        )
        self.state = PhaseLockState(period_us=self.base_period_us)

    def update(self, ts_master_ns: int, ts_slave_ns: int) -> int:
        """返回从动相机新的帧周期（µs），应写入 FrameDurationLimits=(v, v)。"""
        error_us = ((int(ts_slave_ns) - int(ts_master_ns)) - self.target_offset_ns) / 1000.0
        step = float(np.clip(-self.kp * error_us, -self.max_step_us, self.max_step_us))
        period = float(
            np.clip(self.base_period_us + step, self.min_frame_us, self.max_frame_us)
        )
        self.state.period_us = period
        self.state.last_error_us = error_us
        self.state.steps += 1
        return int(round(period))

    @property
    def period_us(self) -> float:
        return self.state.period_us


# --------------------------------------------------------------------------- #
# 无硬件时的同步仿真：证明控制环方向正确、能把相位误差压下去
# --------------------------------------------------------------------------- #
@dataclass
class SyncSimulationResult:
    before_offset_ms: List[float]
    after_offset_ms: List[float]
    periods_us: List[float]
    stats: Dict[str, object]


def simulate_sync(
    n_frames: int = 300,
    kp: float = 0.05,
    master_period_us: float = 33333.0,
    slave_period_us: float = 33600.0,
    jitter_us: float = 0.10,
    initial_offset_ms: float = 6.0,
    control_from: int = 50,
    settle_frac: float = 0.25,
    seed: int = 0,
) -> SyncSimulationResult:
    """两路自由运行相机的时间戳仿真。

    control_from 之前不加控制（复现"出厂读数"）；之后启用 P 环。
    after 只取最后 settle_frac 的稳态窗口，避免把收敛瞬态当成控制效果。
    """
    rng = np.random.default_rng(seed)
    ctl = PhaseLockController(
        kp=kp,
        min_frame_us=30_000,
        max_frame_us=40_000,
        max_step_us=300,
        base_period_us=master_period_us,
    )
    # 从动相机实际周期从 slave_period_us 起步（体现"出厂"偏差）
    ctl.state.period_us = float(slave_period_us)
    t_master = 1_000_000_000.0
    t_slave = t_master + initial_offset_ms * 1e6
    offsets: List[float] = []
    periods: List[float] = []
    before: List[float] = []
    after: List[float] = []
    for i in range(n_frames):
        t_master += master_period_us * 1000.0 + rng.normal(0, jitter_us * 1000.0)
        t_slave += ctl.period_us * 1000.0 + rng.normal(0, jitter_us * 1000.0)
        diff_ms = (t_slave - t_master) / 1e6
        offsets.append(diff_ms)
        periods.append(ctl.period_us)
        if i < control_from:
            before.append(diff_ms)
        elif i >= max(control_from, int(n_frames * (1.0 - settle_frac))):
            # 只统计控制开启后、收敛完成的稳态窗口
            after.append(diff_ms)
        if i >= control_from:
            ctl.update(int(t_master), int(t_slave))
    before_stats = summarize(before)
    after_stats = summarize(after)
    return SyncSimulationResult(
        before_offset_ms=before,
        after_offset_ms=after,
        periods_us=periods,
        stats={
            "before": before_stats,
            "after": after_stats,
            "before_iqr_ms": before_stats["p75"] - before_stats["p25"] if before_stats["n"] else None,
            "after_iqr_ms": after_stats["p75"] - after_stats["p25"] if after_stats["n"] else None,
            "final_period_us": ctl.period_us,
            "kp": kp,
        },
    )
