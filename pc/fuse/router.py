"""SGBM 优先的融合路由。

规则（阶段三，不可协商）：
  1. SGBM 有效、且通过左右一致性检查、且局部纹理达标 → 直接用 SGBM；
  2. 只有不满足 1 的像素，才允许采用学习型匹配器输出；
  3. **不做加权平均**，每个像素的深度来源唯一且可追溯；
  4. 学习型输出也要通过自身 valid 掩码，不用它覆盖可信的 SGBM。

输出 source 图（每像素来源）会被物体聚合与报告使用，因此"某个物体深度
来自 SGBM 还是补全"是可审计的，而不是一句口头结论。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

from pc.stereo.base import DisparityResult

SOURCE_INVALID = 0
SOURCE_SGBM = 1
SOURCE_LEARNED = 2


@dataclass
class FusedDisparity:
    disp: np.ndarray
    valid: np.ndarray
    source: np.ndarray  # uint8 每像素来源
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return float(self.valid.mean())

    def source_coverage(self, source_id: int) -> float:
        return float((self.source == source_id).mean())


class FuseRouter:
    def __init__(self, lr_max_diff: float = 1.0, texture_threshold: float = 8.0) -> None:
        self.lr_max_diff = float(lr_max_diff)
        self.texture_threshold = float(texture_threshold)

    # ------------------------------------------------------------------ #
    @staticmethod
    def texture_score(gray: np.ndarray, window: int = 5) -> np.ndarray:
        """局部标准差，作为"纹理是否足够支持 SGBM"的判据。"""
        g = gray.astype(np.float32)
        mean = cv2.boxFilter(g, -1, (window, window), normalize=True)
        sq = cv2.boxFilter(g * g, -1, (window, window), normalize=True)
        var = np.maximum(sq - mean * mean, 0.0)
        return np.sqrt(var)

    def lr_consistent(
        self,
        left_disp: np.ndarray,
        left_valid: np.ndarray,
        right_disp: Optional[np.ndarray],
        right_valid: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """左右一致性：|d_L(x) - d_R(x - d_L)| <= lr_max_diff。

        返回 (ok, evaluated, available)。right_disp 为正值视差。
        """
        ok = np.zeros_like(left_valid, dtype=bool)
        evaluated = np.zeros_like(left_valid, dtype=bool)
        if right_disp is None or right_valid is None:
            return left_valid.copy(), left_valid.copy(), False

        h, w = left_disp.shape
        ys, xs = np.mgrid[0:h, 0:w]
        xr = np.rint(xs - left_disp).astype(np.int32)
        inb = left_valid & (xr >= 0) & (xr < w)
        xrc = np.clip(xr, 0, w - 1)
        d_r = right_disp[ys, xrc]
        v_r = right_valid[ys, xrc]
        evaluated = inb
        ok = inb & v_r & (np.abs(left_disp - d_r) <= self.lr_max_diff)
        return ok, evaluated, True

    # ------------------------------------------------------------------ #
    def fuse(
        self,
        sgbm: DisparityResult,
        learned: Optional[DisparityResult],
        left_bgr: np.ndarray,
    ) -> FusedDisparity:
        gray = left_bgr if left_bgr.ndim == 2 else cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        texture = self.texture_score(gray)

        lr_ok, lr_eval, lr_available = self.lr_consistent(
            sgbm.disp, sgbm.valid, sgbm.right_disp, sgbm.right_valid
        )
        textured = texture >= self.texture_threshold
        sgbm_ok = sgbm.valid & lr_ok & textured

        learned_valid = learned.valid if learned is not None else np.zeros_like(sgbm.valid)
        fill = (~sgbm_ok) & learned_valid

        disp = np.zeros_like(sgbm.disp, dtype=np.float32)
        source = np.zeros(sgbm.disp.shape, dtype=np.uint8)
        if learned is not None:
            disp[fill] = learned.disp[fill]
            source[fill] = SOURCE_LEARNED
        disp[sgbm_ok] = sgbm.disp[sgbm_ok]
        source[sgbm_ok] = SOURCE_SGBM
        valid = sgbm_ok | fill

        overlap = sgbm_ok & learned_valid
        total = float(sgbm.valid.size)
        stats: Dict[str, object] = {
            "lr_available": lr_available,
            "lr_evaluated_px": int(lr_eval.sum()),
            "lr_rejected_px": int((sgbm.valid & ~lr_ok).sum()),
            "low_texture_rejected_px": int((sgbm.valid & lr_ok & ~textured).sum()),
            "sgbm_accepted_px": int(sgbm_ok.sum()),
            "learned_present": learned is not None,
            "learned_valid_px": int(learned_valid.sum()),
            "learned_used_px": int(fill.sum()),
            "learned_overlap_px": int(overlap.sum()),
            "sgbm_share": float(sgbm_ok.sum() / total),
            "learned_share": float(fill.sum() / total),
            "coverage": float(valid.sum() / total),
            "policy": "sgbm_priority_no_average",
        }
        if learned is not None:
            stats["learned_source"] = learned.source
            stats["learned_latency_ms"] = learned.latency_ms
        return FusedDisparity(disp=disp, valid=valid, source=source, stats=stats)


def fuse_default(sgbm: DisparityResult, learned: Optional[DisparityResult], left_bgr: np.ndarray) -> FusedDisparity:
    return FuseRouter().fuse(sgbm, learned, left_bgr)
