"""彩色连通域检测器：YOLO 权重未就位时的链路占位检测器。

按色相分桶做连通域，避免相邻的不同颜色物体被形态学闭运算粘成一个框。
它的职责不是"替代 YOLO 做识别"，而是让接收→校正→匹配→聚合→输出的
数据通路在没有权重的情况下也能被端到端验证。报告里会明确标注
``detector=color``，不能把它的精度当作检测指标。
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .base import Detection

# OpenCV 色相范围 0..179；红色跨 0 边界，单独处理
_HUE_BINS: Tuple[Tuple[int, int], ...] = (
    (0, 12),
    (12, 28),
    (28, 48),
    (48, 85),
    (85, 110),
    (110, 140),
    (140, 160),
    (160, 180),
)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(boxes: List[Tuple[float, float, float, float, float]], iou_thr: float = 0.45):
    """boxes: (x1,y1,x2,y2,score)。"""
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep: List[Tuple[float, float, float, float, float]] = []
    for b in boxes:
        if all(_iou(b[:4], k[:4]) < iou_thr for k in keep):
            keep.append(b)
    return keep


class ColorBlobDetector:
    name = "color"

    def __init__(
        self,
        min_area_ratio: float = 0.0015,
        sat_min: int = 80,
        val_min: int = 45,
        class_name: str = "object",
    ) -> None:
        self.min_area_ratio = float(min_area_ratio)
        self.sat_min = int(sat_min)
        self.val_min = int(val_min)
        self.class_name = class_name

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat_ok = (hsv[:, :, 1] >= self.sat_min) & (hsv[:, :, 2] >= self.val_min)
        kernel = np.ones((3, 3), np.uint8)
        min_area = self.min_area_ratio * w * h

        candidates: List[Tuple[float, float, float, float, float]] = []
        for lo, hi in _HUE_BINS:
            hue_ok = (hsv[:, :, 0] >= lo) & (hsv[:, :, 0] < hi)
            mask = (sat_ok & hue_ok).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if area < min_area or bw < 8 or bh < 8:
                    continue
                conf = float(np.clip(0.35 + 0.6 * min(1.0, area / (0.03 * w * h)), 0.0, 0.99))
                candidates.append((float(x), float(y), float(x + bw - 1), float(y + bh - 1), conf))

        kept = _nms(candidates)
        out = [
            Detection(self.class_name, (x1, y1, x2, y2), conf)
            for x1, y1, x2, y2, conf in kept
        ]
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out


class NullDetector:
    name = "none"

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        return []
