"""物体级 3D 聚合：框内视差中位数 → 反投影 → 地面约束。

输出契约（全项目共用）::

    {"class", "center_3d", "bbox_2d", "confidence", "depth_source", ...}

地面约束只在"该类物体通常放在支撑平面上"时生效；无约束与有约束两组结果
都会被保留（``center_3d`` 与 ``unconstrained_center_3d``），报告里分列对比，
不允许只报好看的那一组。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pc.calib.calibrate import CalibrationResult
from pc.detect.base import Detection
from pc.fuse.router import SOURCE_LEARNED, SOURCE_SGBM, FusedDisparity
from shared.config import ObjectConfig


# --------------------------------------------------------------------------- #
# 支撑平面
# --------------------------------------------------------------------------- #
@dataclass
class GroundPlane:
    """Y = p*X + q*Z + r 形式（相机坐标系，Y 向下）。"""

    p: float
    q: float
    r: float
    inlier_ratio: float
    rms_m: float
    n_points: int

    def y_at(self, x: float, z: float) -> float:
        return self.p * x + self.q * z + self.r

    def is_plausible(self) -> bool:
        # 地面法向接近 -Y（相机 Y 向下），且平面在相机下方
        return (
            abs(self.p) < 0.5
            and abs(self.q) < 0.5
            and self.r > 0.02
            and self.inlier_ratio >= 0.25
            and self.rms_m < 0.08
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "q": self.q,
            "r": self.r,
            "inlier_ratio": self.inlier_ratio,
            "rms_m": self.rms_m,
            "n_points": self.n_points,
        }


def fit_ground_plane(
    points: np.ndarray,
    iters: int = 250,
    thresh_m: float = 0.02,
    seed: int = 0,
    max_points: int = 4000,
) -> Optional[GroundPlane]:
    """RANSAC 拟合 Y = pX + qZ + r；点数不足或平面不可信返回 None。"""
    pts = np.asarray(points, np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 50:
        return None
    rng = np.random.default_rng(seed)
    if len(pts) > max_points:
        idx = rng.choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    best_inliers: Optional[np.ndarray] = None
    best_count = 0
    for _ in range(int(iters)):
        i = rng.choice(len(pts), 3, replace=False)
        sample = pts[i]
        A = np.column_stack([sample[:, 0], sample[:, 2], np.ones(3)])
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        try:
            coef = np.linalg.solve(A, sample[:, 1])
        except np.linalg.LinAlgError:
            continue
        pred = coef[0] * pts[:, 0] + coef[1] * pts[:, 2] + coef[2]
        inliers = np.abs(pts[:, 1] - pred) <= thresh_m
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < 50:
        return None

    # 最小二乘精修
    inl = pts[best_inliers]
    A = np.column_stack([inl[:, 0], inl[:, 2], np.ones(len(inl))])
    coef, *_ = np.linalg.lstsq(A, inl[:, 1], rcond=None)
    pred = A @ coef
    residual = inl[:, 1] - pred
    rms = float(np.sqrt(np.mean(residual**2)))
    inlier_ratio = float(best_count / len(pts))
    return GroundPlane(
        p=float(coef[0]),
        q=float(coef[1]),
        r=float(coef[2]),
        inlier_ratio=inlier_ratio,
        rms_m=rms,
        n_points=best_count,
    )


# --------------------------------------------------------------------------- #
# 输出结构
# --------------------------------------------------------------------------- #
@dataclass
class ObjectEstimate:
    class_name: str
    confidence: float
    bbox_2d: Tuple[float, float, float, float]
    center_3d: Tuple[float, float, float]
    unconstrained_center_3d: Tuple[float, float, float]
    depth_m: float
    depth_source: str
    valid_ratio: float
    n_pixels: int
    height_m: Optional[float] = None
    ground_applied: bool = False
    contact_3d: Optional[Tuple[float, float, float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox_2d": [round(float(v), 2) for v in self.bbox_2d],
            "center_3d": [round(float(v), 4) for v in self.center_3d],
            "unconstrained_center_3d": [round(float(v), 4) for v in self.unconstrained_center_3d],
            "depth_m": round(float(self.depth_m), 4),
            "depth_source": self.depth_source,
            "valid_ratio": round(float(self.valid_ratio), 4),
            "n_pixels": int(self.n_pixels),
            "height_m": None if self.height_m is None else round(float(self.height_m), 4),
            "ground_applied": bool(self.ground_applied),
            "contact_3d": None
            if self.contact_3d is None
            else [round(float(v), 4) for v in self.contact_3d],
            "extra": self.extra,
        }


@dataclass
class AggregateResult:
    objects: List[ObjectEstimate]
    ground_plane: Optional[GroundPlane]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 聚合器
# --------------------------------------------------------------------------- #
class ObjectAggregator:
    def __init__(self, calib: CalibrationResult, cfg: ObjectConfig) -> None:
        self.calib = calib
        self.cfg = cfg
        self._ground_classes = {c.strip() for c in cfg.ground_classes.split(",") if c.strip()}

    # ---- 地面 -------------------------------------------------------- #
    def fit_ground(self, fused: FusedDisparity) -> Optional[GroundPlane]:
        h, w = fused.disp.shape
        ys, xs = np.mgrid[0:h, 0:w]
        band_top = int((1.0 - self.cfg.ground_band) * h)
        sel = fused.valid & (ys >= band_top)
        if int(sel.sum()) < self.cfg.min_ground_points:
            return None
        d = fused.disp[sel]
        Z = self.calib.focal_px * self.calib.baseline_m / np.maximum(d, 1e-6)
        X = (xs[sel] - self.calib.cx) * Z / self.calib.focal_px
        Y = (ys[sel] - self.calib.cy) * Z / self.calib.focal_px
        pts = np.column_stack([X, Y, Z])
        plane = fit_ground_plane(
            pts,
            iters=self.cfg.ransac_iters,
            thresh_m=self.cfg.ransac_thresh_m,
        )
        if plane is None or not plane.is_plausible():
            return None
        return plane

    # ---- 主流程 ------------------------------------------------------ #
    def aggregate(
        self,
        detections: Sequence[Detection],
        fused: FusedDisparity,
        left_bgr: Optional[np.ndarray] = None,
    ) -> AggregateResult:
        h, w = fused.disp.shape
        f = self.calib.focal_px
        cx, cy = self.calib.cx, self.calib.cy
        depth = self.calib.disparity_to_depth(fused.disp)
        plane = self.fit_ground(fused)

        objects: List[ObjectEstimate] = []
        skipped: List[Dict[str, Any]] = []
        for det in detections:
            d = det.clipped(w, h)
            x1, y1, x2, y2 = d.bbox_xyxy
            bw, bh = x2 - x1, y2 - y1
            if bw < 6 or bh < 6:
                skipped.append({"class": d.class_name, "reason": "bbox_too_small"})
                continue
            # 收缩框，避开物体边界处的前景/背景混合像素
            sx = self.cfg.bbox_shrink * bw / 2.0
            sy = self.cfg.bbox_shrink * bh / 2.0
            ix1, iy1 = int(np.floor(x1 + sx)), int(np.floor(y1 + sy))
            ix2, iy2 = int(np.ceil(x2 - sx)), int(np.ceil(y2 - sy))
            ix1, iy1 = max(0, ix1), max(0, iy1)
            ix2, iy2 = min(w - 1, ix2), min(h - 1, iy2)
            if ix2 <= ix1 or iy2 <= iy1:
                skipped.append({"class": d.class_name, "reason": "empty_roi"})
                continue

            roi_valid = fused.valid[iy1 : iy2 + 1, ix1 : ix2 + 1]
            n_valid = int(roi_valid.sum())
            area = float((ix2 - ix1 + 1) * (iy2 - iy1 + 1))
            ratio = n_valid / area if area > 0 else 0.0
            if n_valid < self.cfg.min_pixels or ratio < self.cfg.min_valid_ratio:
                skipped.append(
                    {
                        "class": d.class_name,
                        "reason": "insufficient_valid_pixels",
                        "valid_ratio": round(ratio, 3),
                        "n_pixels": n_valid,
                    }
                )
                continue

            roi_disp = fused.disp[iy1 : iy2 + 1, ix1 : ix2 + 1][roi_valid]
            d_med = float(np.median(roi_disp))
            if d_med <= 0:
                skipped.append({"class": d.class_name, "reason": "bad_disparity"})
                continue
            Z = f * self.calib.baseline_m / d_med
            u_c, v_c = d.center
            X = (u_c - cx) * Z / f
            Y = (v_c - cy) * Z / f
            unconstrained = (float(X), float(Y), float(Z))

            src_roi = fused.source[iy1 : iy2 + 1, ix1 : ix2 + 1][roi_valid]
            n_sgbm = int((src_roi == SOURCE_SGBM).sum())
            n_learned = int((src_roi == SOURCE_LEARNED).sum())
            if n_sgbm and n_learned:
                source = "sgbm" if n_sgbm >= n_learned else "learned"
            elif n_sgbm:
                source = "sgbm"
            elif n_learned:
                source = "learned"
            else:
                source = "unknown"

            center = unconstrained
            height_m: Optional[float] = None
            contact: Optional[Tuple[float, float, float]] = None
            applied = False

            if self._should_constrain(d.class_name) and plane is not None:
                constrained = self._ground_center(d, fused, plane, Z)
                if constrained is not None:
                    center, height_m, contact = constrained
                    applied = True

            objects.append(
                ObjectEstimate(
                    class_name=d.class_name,
                    confidence=d.confidence,
                    bbox_2d=(float(x1), float(y1), float(x2), float(y2)),
                    center_3d=center,
                    unconstrained_center_3d=unconstrained,
                    depth_m=float(Z),
                    depth_source=source,
                    valid_ratio=float(ratio),
                    n_pixels=n_valid,
                    height_m=height_m,
                    ground_applied=applied,
                    contact_3d=contact,
                )
            )

        diagnostics = {
            "detections": len(detections),
            "accepted": len(objects),
            "skipped": skipped,
            "ground_plane": None if plane is None else plane.to_dict(),
            "ground_valid": plane is not None,
        }
        return AggregateResult(objects=objects, ground_plane=plane, diagnostics=diagnostics)

    # ---- 辅助 -------------------------------------------------------- #
    def _should_constrain(self, class_name: str) -> bool:
        return "*" in self._ground_classes or class_name in self._ground_classes

    def _ground_center(
        self,
        det: Detection,
        fused: FusedDisparity,
        plane: GroundPlane,
        z_hint: float,
        band: float = 0.15,
    ) -> Optional[Tuple[Tuple[float, float, float], float, Tuple[float, float, float]]]:
        h, w = fused.disp.shape
        f = self.calib.focal_px
        x1, y1, x2, y2 = det.bbox_xyxy
        bh = y2 - y1
        bw = x2 - x1

        # 底部条带：物体与支撑面的接触处
        by1 = int(np.floor(y2 - band * bh))
        by2 = int(np.ceil(y2))
        bx1 = int(np.floor(x1 + 0.2 * bw))
        bx2 = int(np.ceil(x2 - 0.2 * bw))
        by1, bx1 = max(0, by1), max(0, bx1)
        by2, bx2 = min(h - 1, by2), min(w - 1, bx2)
        if by2 <= by1 or bx2 <= bx1:
            return None
        bottom = fused.disp[by1 : by2 + 1, bx1 : bx2 + 1]
        bvalid = fused.valid[by1 : by2 + 1, bx1 : bx2 + 1]
        if int(bvalid.sum()) < 20:
            return None
        d_b = float(np.median(bottom[bvalid]))
        if d_b <= 0:
            return None
        z_b = f * self.calib.baseline_m / d_b
        u_b = (bx1 + bx2) / 2.0
        x_b = (u_b - self.calib.cx) * z_b / f
        ground_y = plane.y_at(x_b, z_b)

        # 顶部条带：顶面是水平面 Y = H - h，其上任意点的 Y 都等于物体顶高。
        # 逐像素反投影后取低分位，避免条带下缘混入前表面上部的点。
        ty1 = int(np.floor(y1))
        ty2 = int(np.ceil(y1 + max(2.0, band * bh)))
        tx1, tx2 = max(0, bx1), min(w - 1, bx2)
        ty1, ty2 = max(0, ty1), min(h - 1, ty2)
        top = fused.disp[ty1 : ty2 + 1, tx1 : tx2 + 1]
        tvalid = fused.valid[ty1 : ty2 + 1, tx1 : tx2 + 1]
        if int(tvalid.sum()) < 10:
            return None
        rows = np.arange(ty1, ty2 + 1, dtype=np.float32)[:, None]
        z_top = f * self.calib.baseline_m / np.maximum(top, 1e-6)
        y_map = (rows - self.calib.cy) * z_top / f
        y_top = float(np.percentile(y_map[tvalid], 25))
        z_t = float(np.median(z_top[tvalid]))

        height = ground_y - y_top
        # 物理合理性：高度必须为正、不能超过量程的一半、不能离谱
        if not (0.02 < height < 2.0) or height > 0.8 * min(z_b, z_t):
            return None
        if abs(height - (bh * z_b / f)) > 0.5 * height + 0.05:
            # 顶部条带很可能压到了背景/遮挡物，放弃约束而不是给错误修正
            return None

        # X 用框中心对应的 X（与 z_hint 一致），避免接触点噪声污染水平分量
        x_center = ((x1 + x2) / 2.0 - self.calib.cx) * z_hint / f
        center = (float(x_center), float(ground_y - height / 2.0), float(z_hint))
        contact = (float(x_b), float(ground_y), float(z_b))
        return center, float(height), contact
