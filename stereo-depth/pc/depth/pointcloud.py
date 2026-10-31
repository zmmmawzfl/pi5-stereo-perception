"""深度图 / 点云输出（不依赖 Open3D）。"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from pc.calib.calibrate import CalibrationResult
from pc.fuse.router import FusedDisparity


def depth_from_fused(fused: FusedDisparity, calib: CalibrationResult) -> np.ndarray:
    depth = np.full(fused.disp.shape, np.nan, dtype=np.float32)
    valid = fused.valid & (fused.disp > 0)
    depth[valid] = calib.focal_px * calib.baseline_m / fused.disp[valid]
    return depth


def colored_pointcloud(
    left_bgr: np.ndarray,
    fused: FusedDisparity,
    calib: CalibrationResult,
    stride: int = 2,
    max_depth_m: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (N,3) 相机坐标系点与 (N,3) BGR 颜色。"""
    h, w = fused.disp.shape
    f = calib.focal_px
    cx, cy = calib.cx, calib.cy
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    valid = fused.valid[::stride, ::stride] & (fused.disp[::stride, ::stride] > 0)
    d = fused.disp[::stride, ::stride][valid]
    Z = f * calib.baseline_m / d
    X = (xs[valid] - cx) * Z / f
    Y = (ys[valid] - cy) * Z / f
    keep = Z <= max_depth_m
    points = np.column_stack([X[keep], Y[keep], Z[keep]]).astype(np.float32)
    colors = left_bgr[::stride, ::stride][valid][keep].reshape(-1, 3)
    return points, colors


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> Path:
    """二进制小端 PLY，Open3D / MeshLab / CloudCompare 都能直接打开。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    assert colors.shape == (n, 3)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    data = np.empty(
        n,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    # OpenCV 是 BGR
    data["red"], data["green"], data["blue"] = colors[:, 2], colors[:, 1], colors[:, 0]
    with open(p, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())
    return p


def depth_visualization(depth: np.ndarray, max_depth_m: float = 3.0) -> np.ndarray:
    """深度伪彩色图（近=暖色），无效点为黑色。"""
    valid = np.isfinite(depth) & (depth > 0)
    norm = np.zeros_like(depth, dtype=np.uint8)
    norm[valid] = np.clip(depth[valid] / max_depth_m * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color
