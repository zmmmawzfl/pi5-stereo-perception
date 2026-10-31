"""离线 TSDF 融合（延伸项，需要 Open3D）。

先决条件（阶段四开始前必须确认）：
  * 位姿来源：转台/机械臂（可信）还是手持 VO（漂移不可控）；
  * 深度图已按同一标定换算成米，且与彩色图对齐。

本模块不生成位姿，也不猜位姿。没有位姿就不要融合——用错误的位姿做 TSDF
只会得到一团看似平滑的垃圾。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np


class TSDFUnavailable(RuntimeError):
    pass


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise TSDFUnavailable(
            "缺少 open3d：pip install open3d（PC 端，Python 3.10 环境）"
        ) from exc
    return o3d


def fuse_sequence(
    color_paths: Sequence[str],
    depth_paths: Sequence[str],
    poses_c2w: np.ndarray,
    calib,
    out_mesh: str,
    voxel_m: float = 0.01,
    sdf_trunc_m: float = 0.04,
    depth_scale: float = 1000.0,
    depth_trunc_m: float = 3.0,
) -> Dict[str, Any]:
    """融合彩色图 + 16 位深度图序列，输出三角网格。

    poses_c2w: (N,4,4) 相机到世界变换；depth 图为 uint16 毫米（或按 depth_scale）。
    """
    o3d = _require_open3d()
    if len(color_paths) != len(depth_paths) or len(color_paths) != len(poses_c2w):
        raise ValueError("彩色图/深度图/位姿数量必须一致")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(calib.image_size[0]), int(calib.image_size[1]),
        float(calib.P1[0, 0]), float(calib.P1[1, 1]),
        float(calib.P1[0, 2]), float(calib.P1[1, 2]),
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_m),
        sdf_trunc=float(sdf_trunc_m),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    t0 = time.perf_counter()
    for cpath, dpath, pose in zip(color_paths, depth_paths, poses_c2w):
        color = o3d.io.read_image(str(cpath))
        depth = o3d.io.read_image(str(dpath))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=float(depth_scale), depth_trunc=float(depth_trunc_m),
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(np.asarray(pose, float)))
    integrate_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    mesh = volume.extract_triangle_mesh()
    extract_s = time.perf_counter() - t1
    mesh.compute_vertex_normals()
    out = Path(out_mesh)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out), mesh)

    stats = {
        "frames": len(color_paths),
        "integrate_s": integrate_s,
        "integrate_fps": len(color_paths) / integrate_s if integrate_s > 0 else None,
        "extract_s": extract_s,
        "extract_fps_equivalent": 1.0 / extract_s if extract_s > 0 else None,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "voxel_m": voxel_m,
        "sdf_trunc_m": sdf_trunc_m,
        "out": str(out),
    }
    out.with_suffix(".json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def load_poses(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        poses = np.load(p)
    elif p.suffix == ".json":
        poses = np.asarray(json.loads(p.read_text(encoding="utf-8")), dtype=float)
    else:
        raise ValueError("位姿文件只支持 .npy / .json（N,4,4）")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"位姿形状应为 (N,4,4)，得到 {poses.shape}")
    return poses
