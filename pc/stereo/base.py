"""立体匹配器统一接口。

约定：
  * ``DisparityResult.disp`` 为 float32 视差（像素），无效像素填 0；
  * ``valid`` 为 bool 掩码，二者必须同时使用；
  * SGBM 与学习型匹配器输出同一结构，路由层只认 valid/source，不认来源实现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol

import numpy as np


@dataclass
class DisparityResult:
    disp: np.ndarray
    valid: np.ndarray
    source: str = "sgbm"
    latency_ms: float = 0.0
    # 右目参考视差（正值），用于左右一致性检查；不可用为 None
    right_disp: Optional[np.ndarray] = None
    right_valid: Optional[np.ndarray] = None
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.disp.shape != self.valid.shape:
            raise ValueError("disp 与 valid 形状不一致")
        if self.right_disp is not None and self.right_valid is None:
            raise ValueError("提供 right_disp 时必须同时提供 right_valid")

    @property
    def coverage(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0


class StereoMatcher(Protocol):
    name: str

    def compute(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> DisparityResult: ...
