"""SGBM：全项目的度量锚点（阶段三之前唯一的深度来源）。

实现要点：
  * 视差原始值为 int16 * 16，必须除以 16 再进深度公式；
  * 同时计算"右目为参考"的视差，供左右一致性检查使用；
  * 先按配置做 ROI 裁剪 / 降分辨率匹配，再把视差上采样回原图（scale 换算）；
  * 可选 WLS 滤波（需要 opencv-contrib），默认关闭，因为 Pi 5 上代价明显；
  * 参数全部来自配置，报告里会原样记录，保证实验可复现。

降分辨率的尺度关系：缩到 scale 倍后，视差也变成 d*scale；上采样时除以
scale 才回到原图视差。若 min_disparity != 0，SGBM 内部的 minDisparity 不会
自动缩放，因此配置层禁止 scale<1 与 min_disparity!=0 同时出现。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

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


@dataclass(frozen=True)
class _Geometry:
    """一次 ROI/缩放计算在原图中的位置与比例。"""

    x0: int
    y0: int
    cw: int
    ch: int
    scale: float
    full_w: int
    full_h: int


class SGBMMatcher:
    name = "sgbm"

    def __init__(self, cfg: SgbmConfig, compute_right: bool = True, use_wls: bool = False) -> None:
        self.cfg = cfg
        self.compute_right = compute_right
        self.use_wls = bool(use_wls and _HAS_XIMGPROC)
        self.scale = float(cfg.scale)
        self.roi = self._parse_roi(cfg.roi_xywh)
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

    @staticmethod
    def _parse_roi(spec: str) -> Optional[Tuple[float, float, float, float]]:
        if not spec:
            return None
        x, y, w, h = (float(v) for v in str(spec).split(","))
        return (x, y, w, h)

    # ------------------------------------------------------------------ #
    def _geometry(self, h: int, w: int) -> _Geometry:
        if self.roi is None:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            rx, ry, rw, rh = self.roi
            x0 = int(round(rx * w))
            y0 = int(round(ry * h))
            x1 = int(round((rx + rw) * w))
            y1 = int(round((ry + rh) * h))
            x0, y0 = max(0, min(x0, w - 1)), max(0, min(y0, h - 1))
            x1, y1 = max(x0 + 1, min(x1, w)), max(y0 + 1, min(y1, h))
        cw, ch = x1 - x0, y1 - y0
        return _Geometry(x0, y0, cw, ch, self.scale, w, h)

    def _prepare(self, gl: np.ndarray, gr: np.ndarray, geo: _Geometry) -> Tuple[np.ndarray, np.ndarray]:
        wl = gl[geo.y0 : geo.y0 + geo.ch, geo.x0 : geo.x0 + geo.cw]
        wr = gr[geo.y0 : geo.y0 + geo.ch, geo.x0 : geo.x0 + geo.cw]
        if self.scale != 1.0:
            ww = max(16, int(round(geo.cw * self.scale)))
            wh = max(16, int(round(geo.ch * self.scale)))
            wl = cv2.resize(wl, (ww, wh), interpolation=cv2.INTER_AREA)
            wr = cv2.resize(wr, (ww, wh), interpolation=cv2.INTER_AREA)
        return wl, wr

    def _restore(
        self, small: np.ndarray, small_valid: np.ndarray, geo: _Geometry
    ) -> Tuple[np.ndarray, np.ndarray]:
        """把工作分辨率视差还原到原图坐标；无效像素统一为 0。"""
        if small.shape[::-1] != (geo.cw, geo.ch):
            disp = cv2.resize(small.astype(np.float32), (geo.cw, geo.ch), interpolation=cv2.INTER_LINEAR)
            disp /= geo.scale
            valid = cv2.resize(
                small_valid.astype(np.uint8), (geo.cw, geo.ch), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            disp = small.astype(np.float32)
            valid = small_valid.astype(bool)
        valid &= np.isfinite(disp) & (disp > 0.0)
        disp[~valid] = 0.0

        full_disp = np.zeros((geo.full_h, geo.full_w), dtype=np.float32)
        full_valid = np.zeros((geo.full_h, geo.full_w), dtype=bool)
        full_disp[geo.y0 : geo.y0 + geo.ch, geo.x0 : geo.x0 + geo.cw] = disp
        full_valid[geo.y0 : geo.y0 + geo.ch, geo.x0 : geo.x0 + geo.cw] = valid
        return full_disp, full_valid

    # ------------------------------------------------------------------ #
    def compute(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> DisparityResult:
        t0 = time.perf_counter()
        gl = _to_gray(left_bgr)
        gr = _to_gray(right_bgr)
        if gl.shape != gr.shape:
            raise ValueError(f"左右图尺寸不一致: {gl.shape} vs {gr.shape}")
        if self.scale != 1.0 and self.cfg.min_disparity != 0:
            raise ValueError("scale<1 时 min_disparity 必须为 0")

        geo = self._geometry(*gl.shape)
        work_l, work_r = self._prepare(gl, gr, geo)
        raw_l = self.matcher.compute(work_l, work_r)
        disp_small = raw_l.astype(np.float32) / 16.0
        valid_small = np.isfinite(disp_small) & (disp_small > 0.0)
        meta = {"raw_scale": 16.0, "used_wls": False, "scale": self.scale,
                "roi": self.cfg.roi_xywh, "work_size": [int(work_l.shape[1]), int(work_l.shape[0])]}

        raw_r = None
        if self.right_matcher is not None:
            raw_r = self.right_matcher.compute(work_r, work_l)

        if self.use_wls and raw_r is not None:
            try:
                wls = cv2.ximgproc.createDisparityWLSFilter(self.matcher)
                wls.setLambda(8000)
                wls.setSigmaColor(1.5)
                filtered = wls.filter(raw_l, work_l, None, raw_r)
                if filtered is not None:
                    f = filtered.astype(np.float32)
                    # WLS 输出与输入可能同为 1/16 单位，也可能已是像素；按量级归一
                    cur_max = float(np.nanmax(disp_small)) if disp_small.size else 0.0
                    if np.nanmax(f) > cur_max * 4 + 1:
                        f = f / 16.0
                    f[~np.isfinite(f)] = 0.0
                    valid_small = f > 0.0
                    f[~valid_small] = 0.0
                    disp_small = f
                    meta["used_wls"] = True
            except cv2.error as exc:  # pragma: no cover - 依赖 contrib 行为
                meta["wls_error"] = str(exc)

        disp, valid = self._restore(disp_small, valid_small, geo)

        right_disp: Optional[np.ndarray] = None
        right_valid: Optional[np.ndarray] = None
        if raw_r is not None:
            # 右目参考视差为负值（匹配点在右侧），取绝对值方便比较
            rd = -raw_r.astype(np.float32) / 16.0
            rv = (raw_r < 0) & (rd > 0) & (rd < self.cfg.num_disparities)
            right_disp, right_valid = self._restore(rd, rv, geo)

        latency = (time.perf_counter() - t0) * 1000.0
        meta["num_disparities"] = self.cfg.num_disparities
        return DisparityResult(
            disp=disp,
            valid=valid,
            source="sgbm",
            latency_ms=latency,
            right_disp=right_disp,
            right_valid=right_valid,
            meta=meta,
        )


def build_sgbm(cfg: SgbmConfig, use_wls: bool = False) -> SGBMMatcher:
    return SGBMMatcher(cfg, compute_right=True, use_wls=use_wls)
