"""极线校正：去畸变 + 行对齐，并给出可在线检查的极线残差。

校正后的两幅图满足：同名点在同一行，视差 d = u_L - u_R > 0。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from pc.calib.calibrate import CalibrationResult
from shared.stats import summarize


@dataclass
class Rectifier:
    calib: CalibrationResult

    def __post_init__(self) -> None:
        w, h = self.calib.image_size
        self.map_lx, self.map_ly = cv2.initUndistortRectifyMap(
            self.calib.K1,
            self.calib.D1,
            self.calib.R1,
            self.calib.P1,
            (w, h),
            cv2.CV_32FC1,
        )
        self.map_rx, self.map_ry = cv2.initUndistortRectifyMap(
            self.calib.K2,
            self.calib.D2,
            self.calib.R2,
            self.calib.P2,
            (w, h),
            cv2.CV_32FC1,
        )

    @property
    def image_size(self) -> Tuple[int, int]:
        return self.calib.image_size

    def rectify_left(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map_lx, self.map_ly, cv2.INTER_LINEAR)

    def rectify_right(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map_rx, self.map_ry, cv2.INTER_LINEAR)

    def rectify(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.rectify_left(left), self.rectify_right(right)

    def reproject_disparity(self, disp: np.ndarray) -> np.ndarray:
        """视差 -> 3D（校正坐标系），返回 HxWx3，无效点为 nan。"""
        f = self.calib.focal_px
        cx, cy = self.calib.cx, self.calib.cy
        h, w = disp.shape
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        Z = self.calib.disparity_to_depth(disp)
        X = (xs - cx) * Z / f
        Y = (ys - cy) * Z / f
        return np.stack([X, Y, Z], axis=-1)


def measure_epipolar_residual(
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    max_features: int = 1200,
    min_dx: float = 2.0,
) -> Dict[str, object]:
    """用 ORB 稀疏匹配检查校正质量：同名点的 |dy| 应接近 0。

    只统计水平视差 > min_dx 的匹配（排除地面/背景中重复纹理造成的误配）。
    这是实机上每次启动都可以跑的一次性自检。
    """
    gray_l = rect_left if rect_left.ndim == 2 else cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_r = rect_right if rect_right.ndim == 2 else cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=12)
    kp_l, des_l = orb.detectAndCompute(gray_l, None)
    kp_r, des_r = orb.detectAndCompute(gray_r, None)
    if des_l is None or des_r is None or len(kp_l) < 10 or len(kp_r) < 10:
        return {"n_matches": 0, "median_abs_dy_px": None, "p90_abs_dy_px": None}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_l, des_r)
    dys = []
    dxs = []
    for m in matches:
        pl = kp_l[m.queryIdx].pt
        pr = kp_r[m.trainIdx].pt
        dx = pl[0] - pr[0]
        if dx < min_dx:
            continue
        dys.append(abs(pl[1] - pr[1]))
        dxs.append(dx)
    if not dys:
        return {"n_matches": 0, "median_abs_dy_px": None, "p90_abs_dy_px": None}
    stats = summarize(dys)
    return {
        "n_matches": len(dys),
        "median_abs_dy_px": stats["median"],
        "p90_abs_dy_px": float(np.percentile(dys, 90)),
        "median_dx_px": float(np.median(dxs)),
    }
