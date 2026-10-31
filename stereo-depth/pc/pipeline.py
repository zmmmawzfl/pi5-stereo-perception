"""PC 端处理管线：校正 → 匹配 → 融合 → 检测 → 物体级 3D 感知。

主线目标是实时双目场景感知：恢复视差/深度、生成点云、输出对象级几何与
导航用感知结果。不是完整 3D 建模；不要求可编辑 CAD / 高保真网格。
一次 process() 的产物包含每个阶段的耗时、每像素来源、物体列表与深度图，
报告与 JSON 都从这里取数，避免"报告数字"和"代码行为"各说各话。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from pc.calib.calibrate import CalibrationResult
from pc.depth.pointcloud import depth_from_fused
from pc.detect.base import Detection
from pc.detect.factory import build_detector
from pc.fuse.router import FusedDisparity, FuseRouter
from pc.objects.aggregator import AggregateResult, ObjectAggregator
from pc.rectify.rectifier import Rectifier, measure_epipolar_residual
from pc.stereo.base import DisparityResult
from pc.stereo.learned import build_learned_matcher
from pc.stereo.sgbm import SGBMMatcher
from pc.stereo.subpixel import refine_disparity_subpixel
from shared.config import AppConfig


@dataclass
class FrameResult:
    frame_id: int
    timings_ms: Dict[str, float]
    detections: List[Detection]
    aggregate: AggregateResult
    fused: FusedDisparity
    depth: np.ndarray
    rectified_left: np.ndarray
    sgbm: DisparityResult
    learned: Optional[DisparityResult] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def objects(self):
        return self.aggregate.objects

    def object_json(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timings_ms": {k: round(float(v), 3) for k, v in self.timings_ms.items()},
            "objects": [o.to_dict() for o in self.objects],
            "disparity": self.fused.stats,
            "ground": self.aggregate.diagnostics.get("ground_plane"),
            "skipped": self.aggregate.diagnostics.get("skipped", []),
            "learned": None
            if self.learned is None
            else {"source": self.learned.source, "latency_ms": round(self.learned.latency_ms, 3)},
        }


class StereoPipeline:
    def __init__(self, cfg: AppConfig, calib: CalibrationResult) -> None:
        self.cfg = cfg
        self.calib = calib
        if tuple(calib.image_size) != (cfg.camera.width, cfg.camera.height):
            print(
                f"[pipeline] 注意：标定分辨率 {calib.image_size} 与配置 "
                f"{(cfg.camera.width, cfg.camera.height)} 不一致，将按标定尺寸做校正"
            )
        self.rectifier = Rectifier(calib)
        self.sgbm = SGBMMatcher(cfg.sgbm, compute_right=True)
        self.router = FuseRouter(
            lr_max_diff=cfg.sgbm.lr_max_diff,
            texture_threshold=cfg.sgbm.texture_threshold,
        )
        self.learned = build_learned_matcher(cfg.sgbm)
        self.detector = build_detector(cfg.detect)
        self.aggregator = ObjectAggregator(calib, cfg.objects)
        self._frame_id = 0

    # ------------------------------------------------------------------ #
    def process(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> FrameResult:
        timings: Dict[str, float] = {}

        t = time.perf_counter()
        rect_l, rect_r = self.rectifier.rectify(left_bgr, right_bgr)
        timings["rectify"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        sgbm = self.sgbm.compute(rect_l, rect_r)
        timings["sgbm"] = (time.perf_counter() - t) * 1000.0

        learned: Optional[DisparityResult] = None
        if self.learned is not None:
            t = time.perf_counter()
            learned = self.learned.compute(rect_l, rect_r)  # type: ignore[attr-defined]
            timings["learned"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        fused = self.router.fuse(sgbm, learned, rect_l)
        timings["fuse"] = (time.perf_counter() - t) * 1000.0

        if self.cfg.sgbm.subpixel and np.any(fused.valid):
            t = time.perf_counter()
            gray_l = rect_l if rect_l.ndim == 2 else _gray(rect_l)
            gray_r = rect_r if rect_r.ndim == 2 else _gray(rect_r)
            sub = refine_disparity_subpixel(gray_l, gray_r, fused.disp, fused.valid)
            fused.disp = np.where(fused.valid, sub, 0.0).astype(np.float32)
            timings["subpixel"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        detections = self.detector.detect(rect_l)
        timings["detect"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        aggregate = self.aggregator.aggregate(detections, fused, rect_l)
        timings["aggregate"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        depth = depth_from_fused(fused, self.calib)
        timings["depth"] = (time.perf_counter() - t) * 1000.0

        timings["total"] = sum(v for k, v in timings.items() if k != "total")
        result = FrameResult(
            frame_id=self._frame_id,
            timings_ms=timings,
            detections=detections,
            aggregate=aggregate,
            fused=fused,
            depth=depth,
            rectified_left=rect_l,
            sgbm=sgbm,
            learned=learned,
            meta={"detector": self.detector.name},
        )
        self._frame_id += 1
        return result

    def epipolar_check(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> Dict[str, object]:
        rect_l, rect_r = self.rectifier.rectify(left_bgr, right_bgr)
        return measure_epipolar_residual(rect_l, rect_r)


def _gray(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def build_pipeline(cfg: AppConfig, calib: CalibrationResult) -> StereoPipeline:
    return StereoPipeline(cfg, calib)
