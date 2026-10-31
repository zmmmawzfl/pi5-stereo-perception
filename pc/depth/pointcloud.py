"""深度图 / 点云输出（不依赖 Open3D）。"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:  # 只为类型注解，避免运行期循环导入
    from pc.pipeline import FrameResult

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
    voxel_size_m: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (N,3) 相机坐标系点与 (N,3) BGR 颜色。

    voxel_size_m > 0 时在反投影后做体素降采样：同一 3D 体素内的点/颜色取均值。
    0.03 m 是机器人避障的常用起点，能把 640×480 的点数压到万级。
    """
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
    if voxel_size_m and voxel_size_m > 0:
        points, colors = voxel_downsample(points, colors, voxel_size_m)
    return points, colors


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel_size_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    """体素栅格降采样：每个体素输出一个质心点与平均颜色。

    纯 NumPy 实现（不依赖 Open3D），按整数体素索引做 unique + 累加。
    点数 2 万级时耗时 < 10 ms；空点云或 voxel<=0 时原样返回。
    """
    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.uint8)
    if voxel_size_m <= 0 or len(pts) == 0:
        return pts, cols
    if pts.ndim != 2 or pts.shape[1] != 3 or cols.shape != pts.shape:
        raise ValueError(f"points/colors 形状不匹配: {pts.shape} vs {cols.shape}")

    keys = np.floor(pts / float(voxel_size_m)).astype(np.int64)
    # lexsort + reduceat 比 np.unique(axis=0) 快约 2–3 倍（30k 点：~17 ms vs ~43 ms）
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    new_group = np.empty(len(sorted_keys), dtype=bool)
    new_group[0] = True
    new_group[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    starts = np.flatnonzero(new_group)
    counts = np.diff(np.append(starts, len(sorted_keys))).astype(np.float64)

    psum = np.add.reduceat(pts[order], starts, axis=0).astype(np.float64)
    csum = np.add.reduceat(cols[order].astype(np.float64), starts, axis=0)

    out_points = (psum / counts[:, None]).astype(np.float32)
    out_colors = np.clip(np.rint(csum / counts[:, None]), 0, 255).astype(np.uint8)
    return out_points, out_colors


def points_in_bbox(
    left_bgr: np.ndarray,
    fused: FusedDisparity,
    calib: CalibrationResult,
    bbox_xyxy: Tuple[int, int, int, int],
    stride: int = 2,
    max_depth_m: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """提取一个 2D 框内的彩色点云，供抓取初筛/目标点云使用。"""
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    h, w = fused.disp.shape
    x1, x2 = max(0, min(x1, x2)), min(w, x2)
    y1, y2 = max(0, min(y1, y2)), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    sub = FusedDisparity(
        disp=fused.disp[y1:y2, x1:x2],
        valid=fused.valid[y1:y2, x1:x2],
        source=fused.source[y1:y2, x1:x2]
        if fused.source is not None
        else np.zeros((y2 - y1, x2 - x1), dtype=np.uint8),
        stats={},
    )
    pts, cols = colored_pointcloud(
        left_bgr[y1:y2, x1:x2], sub, calib, stride=stride, max_depth_m=max_depth_m
    )
    # colored_pointcloud 以子图左上角为光心，需要补回主点偏移
    if len(pts):
        f = calib.focal_px
        pts[:, 0] += x1 * pts[:, 2] / f
        pts[:, 1] += y1 * pts[:, 2] / f
    return pts, cols


def cloud_from_result(
    result: "FrameResult",
    calib: CalibrationResult,
    stride: int = 2,
    max_depth_m: float = 6.0,
    voxel_size_m: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    return colored_pointcloud(
        result.rectified_left,
        result.fused,
        calib,
        stride=stride,
        max_depth_m=max_depth_m,
        voxel_size_m=voxel_size_m,
    )

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
