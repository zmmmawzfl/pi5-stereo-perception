"""视差亚像素细化（抛物线拟合匹配代价）。

SGBM 输出整数视差；在 1.5m 处 1 px 对应约 5 cm 深度，物体质心误差会被
量化误差主导。这里用 d-1 / d / d+1 的窗口 SAD 代价做抛物线插值，把质心
的视差精度提到亚像素。只在需要时（物体区域）调用，成本可控。
"""
from __future__ import annotations

import cv2
import numpy as np


def _shift_right(
    right_gray: np.ndarray, disp: np.ndarray, delta: float
) -> np.ndarray:
    """在右图上按 x - (d + delta) 采样，返回与左图对齐的图像。"""
    h, w = right_gray.shape
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    mapx = np.broadcast_to(xs - disp - delta, (h, w)).astype(np.float32)
    mapy = np.broadcast_to(ys, (h, w)).astype(np.float32)
    return cv2.remap(
        right_gray,
        mapx,
        mapy,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def refine_disparity_subpixel(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    disp: np.ndarray,
    valid: np.ndarray,
    window: int = 5,
    max_offset: float = 0.5,
) -> np.ndarray:
    """对 valid 像素做 ±1 px 代价抛物线拟合，返回细化后的 float32 视差。"""
    if left_gray.ndim == 3:
        left_gray = cv2.cvtColor(left_gray, cv2.COLOR_BGR2GRAY)
    if right_gray.ndim == 3:
        right_gray = cv2.cvtColor(right_gray, cv2.COLOR_BGR2GRAY)

    left_f = left_gray.astype(np.float32)
    d = np.where(valid, disp, 0.0).astype(np.float32)

    costs = []
    for delta in (-1.0, 0.0, 1.0):
        warped = _shift_right(right_gray.astype(np.float32), d, delta)
        sad = np.abs(left_f - warped)
        costs.append(cv2.boxFilter(sad, -1, (window, window), normalize=True))

    c_m, c_0, c_p = costs
    denom = c_m - 2.0 * c_0 + c_p
    # 抛物线顶点偏移：0.5 * (c_-1 - c_+1) / (c_-1 - 2c_0 + c_+1)
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = 0.5 * (c_m - c_p) / denom
    offset = np.clip(np.nan_to_num(offset, nan=0.0, posinf=0.0, neginf=0.0), -max_offset, max_offset)
    # 代价曲线不呈凸形（denom<=0）时保持整数视差
    offset[denom <= 1e-6] = 0.0

    out = d + offset
    out[~valid] = 0.0
    return out.astype(np.float32)
