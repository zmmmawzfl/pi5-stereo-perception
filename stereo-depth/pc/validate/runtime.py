"""运行时资源观测：CPU、RSS、峰值 RSS（不引入 psutil 依赖）。

实时链路的验收不只看 ms/帧，还要看 CPU 是否被吃满、内存是否持续上涨。
这里用 /proc 与 resource 取进程级数据，Linux/Pi 5 上零依赖。
"""
from __future__ import annotations

import resource
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional


def _rss_mb() -> Optional[float]:
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def _peak_rss_mb() -> Optional[float]:
    try:
        # Linux ru_maxrss 单位是 KiB；macOS 是字节
        val = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return val / (1024.0 * 1024.0) if sys.platform == "darwin" else val / 1024.0
    except (ValueError, OSError):
        return None


@dataclass
class ResourceSnapshot:
    wall_s: float
    cpu_time_s: float
    cpu_percent: float          # 进程累计 CPU / 墙钟；多线程可 >100
    rss_mb: Optional[float]
    peak_rss_mb: Optional[float]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class ResourceMonitor:
    """记录从 start 到 snapshot 的进程资源消耗。"""

    def __init__(self) -> None:
        self._wall0 = time.perf_counter()
        self._cpu0 = time.process_time()
        self._rss0 = _rss_mb()

    def snapshot(self) -> ResourceSnapshot:
        wall = max(time.perf_counter() - self._wall0, 1e-9)
        cpu = max(time.process_time() - self._cpu0, 0.0)
        rss = _rss_mb()
        peak = _peak_rss_mb()
        return ResourceSnapshot(
            wall_s=round(wall, 3),
            cpu_time_s=round(cpu, 3),
            cpu_percent=round(cpu / wall * 100.0, 1),
            rss_mb=None if rss is None else round(rss, 2),
            peak_rss_mb=None if peak is None else round(peak, 2),
        )
