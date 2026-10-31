"""可视化辅助：物体框 + 3D 中心回投、视差来源图。"""
from __future__ import annotations

import cv2
import numpy as np

from pc.calib.calibrate import CalibrationResult
from pc.fuse.router import SOURCE_LEARNED, SOURCE_SGBM


def draw_objects(
    image: np.ndarray,
    objects,
    calib: CalibrationResult,
    show_source: bool = True,
) -> np.ndarray:
    """在（校正后的）左图上画框、质心与深度来源。"""
    img = image.copy()
    for obj in objects:
        x1, y1, x2, y2 = (int(round(v)) for v in obj.bbox_2d)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        X, Y, Z = obj.center_3d
        if Z > 1e-6:
            u = int(round(calib.P1[0, 0] * X / Z + calib.cx))
            v = int(round(calib.P1[1, 1] * Y / Z + calib.cy))
            cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
        label = f"{obj.class_name} Z={Z:.2f}m"
        if show_source:
            label += f" {obj.depth_source}"
        cv2.putText(
            img, label, (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
        )
    return img


def source_map(fused) -> np.ndarray:
    vis = np.zeros((*fused.disp.shape, 3), np.uint8)
    vis[fused.source == SOURCE_SGBM] = (0, 200, 0)
    vis[fused.source == SOURCE_LEARNED] = (255, 128, 0)
    return vis
