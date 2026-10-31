"""SGBM：全项目的度量锚点（阶段三之前唯一的深度来源）。

实现要点：
  * 视差原始值为 int16 * 16，必须除以 16 再进深度公式；
  * 同时计算"右目为参考"的视差，供左右一致性检查使用；
  * 可选 WLS 滤波（需要 opencv-contrib），默认关闭，因为 Pi 5 上代价明显；
  * 参数全部来自配置，报告里会原样记录，保证实验可复现。
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from shared.config import SgbmConfig

from .base import DisparityResult

try:  # opencv-contrib 才有；缺失时退化为不做 LR 检查
    _HAS_XIMGPROC = hasattr(cv2, "ximgproc")
except Exception:  # pragma: no cover
    _HAS_XIMGPROC = False


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class SGBMMatcher:
    name = "sgbm"

    def __init__(self, cfg: SgbmConfig, compute_right: bool = True, use_wls: bool = False) -> None:
        self.cfg = cfg
        self.compute_right = compute_right
        self.use_wls = bool(use_wls and _HAS_XIMGPROC)
        bs = cfg.block_size
        p1 = cfg.p1 if cfg.p1 > 0 else 8 * bs * bs
        p2 = cfg.p2 if cfg.p2 > 0 else 32 * bs * bs
        mode = {
            "sgbm": cv2.STEREO_SGBM_MODE_SGBM,
            "hh": cv2.STEREO_SGBM_MODE_HH,
            "sgbm_3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        }.get(cfg.mode, cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=cfg.min_disparity,
            numDisparities=cfg.num_disparities,
            blockSize=bs,
            P1=p1,
            P2=p2,
            disp12MaxDiff=cfg.disp12_max_diff,
            uniquenessRatio=cfg.uniqueness_ratio,
            speckleWindowSize=cfg.speckle_window_size,
            speckleRange=cfg.speckle_range,
            preFilterCap=63,
            mode=mode,
        )
        self.right_matcher = None
        if compute_right and _HAS_XIMGPROC:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.matcher)

    # ------------------------------------------------------------------ #
    def compute(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> DisparityResult:
        t0 = time.perf_counter()
        gl = _to_gray(left_bgr)
        gr = _to_gray(right_bgr)

        raw_l = self.matcher.compute(gl, gr)
        disp = raw_l.astype(np.float32) / 16.0
        valid = self._valid_from_raw(raw_l, disp)
        disp[~valid] = 0.0  # 统一契约：无效像素视差为 0

        right_disp: Optional[np.ndarray] = None
        right_valid: Optional[np.ndarray] = None
        meta = {"raw_scale": 16.0, "used_wls": False}

        if self.right_matcher is not None:
            raw_r = self.right_matcher.compute(gr, gl)
            # 右目参考视差为负值（匹配点在右侧），取绝对值方便比较
            rd = -raw_r.astype(np.float32) / 16.0
            # 右目参考视差为负值；无效哨兵 = -numDisparities*16
            right_valid = (raw_r < 0) & (rd > 0) & (rd < self.cfg.num_disparities)
            right_disp = np.where(right_valid, rd, 0.0).astype(np.float32)

        if self.use_wls and self.right_matcher is not None:
            try:
                wls = cv2.ximgproc.createDisparityWLSFilter(self.matcher)
                wls.setLambda(8000)
                wls.setSigmaColor(1.5)
                filtered = wls.filter(raw_l, gl, None, self.right_matcher.compute(gr, gl))
                if filtered is not None:
                    f = filtered.astype(np.float32)
                    # WLS 输出与输入同尺度（1/16 px 单位），跟随 OpenCV 版本差异做归一
                    if np.nanmax(f) > disp.max() * 4 + 1:
                        f = f / 16.0
                    f[~np.isfinite(f)] = 0.0
                    valid = f > 0
                    f[~valid] = 0.0
                    disp = f
                    meta["used_wls"] = True
            except cv2.error as exc:  # pragma: no cover - 依赖 contrib 行为
                meta["wls_error"] = str(exc)

        latency = (time.perf_counter() - t0) * 1000.0
        meta["num_disparities"] = self.cfg.num_disparities
        return DisparityResult(
            disp=disp.astype(np.float32),
            valid=valid,
            source="sgbm",
            latency_ms=latency,
            right_disp=right_disp,
            right_valid=right_valid,
            meta=meta,
        )

    def _valid_from_raw(self, raw: np.ndarray, disp: np.ndarray) -> np.ndarray:
        # SGBM 用 -16 表示无效；同时排除非有限值
        valid = np.isfinite(disp) & (disp > 0.0)
        if self.cfg.disp12_max_diff >= 0:
            # disp12MaxDiff 已在匹配器内部生效，这里只做基本过滤
            pass
        return valid


def build_sgbm(cfg: SgbmConfig, use_wls: bool = False) -> SGBMMatcher:
    return SGBMMatcher(cfg, compute_right=True, use_wls=use_wls)
