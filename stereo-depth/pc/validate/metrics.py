"""验证指标：物体级误差、覆盖率、延迟统计、GT 匹配。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from pc.depth.pointcloud import depth_from_fused
from pc.fuse.router import FusedDisparity
from pc.objects.aggregator import ObjectEstimate
from shared.stats import abs_error_stats, signed_error_stats


# --------------------------------------------------------------------------- #
def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / ua) if ua > 0 else 0.0


def match_objects_to_gt(
    objects: Sequence[ObjectEstimate],
    gt_objects: Sequence[Dict[str, Any]],
    iou_thresh: float = 0.2,
) -> List[Dict[str, Any]]:
    """按 2D 框 IoU 一对一匹配（贪心，IoU 大者优先）。"""
    pairs: List[tuple[float, int, int]] = []
    for i, est in enumerate(objects):
        for j, gt in enumerate(gt_objects):
            iou = bbox_iou(est.bbox_2d, gt["bbox_xyxy"])
            if iou >= iou_thresh:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_i, used_j = set(), set()
    out: List[Dict[str, Any]] = []
    for iou, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append({"est_index": i, "gt_index": j, "iou": iou})
    return out


def object_error_rows(
    objects: Sequence[ObjectEstimate],
    gt_objects: Sequence[Dict[str, Any]],
    use_constraint: bool = True,
    iou_thresh: float = 0.2,
) -> List[Dict[str, Any]]:
    """逐物体误差行（含 X/Y/Z/3D 与来源、是否地面约束）。"""
    matches = match_objects_to_gt(objects, gt_objects, iou_thresh)
    rows: List[Dict[str, Any]] = []
    for m in matches:
        est = objects[m["est_index"]]
        gt = gt_objects[m["gt_index"]]
        gt_c = np.asarray(gt["gt_surface_center_m"], np.float64)
        est_c = np.asarray(est.center_3d if use_constraint else est.unconstrained_center_3d, np.float64)
        err = est_c - gt_c
        rows.append(
            {
                "name": gt["name"],
                "class": est.class_name,
                "class_match": est.class_name == gt["class_name"],
                "bbox_iou": round(float(m["iou"]), 3),
                "gt_depth_m": float(gt_c[2]),
                "est_depth_m": float(est_c[2]),
                "dx_m": float(err[0]),
                "dy_m": float(err[1]),
                "dz_m": float(err[2]),
                "depth_abs_err_m": float(abs(err[2])),
                "depth_rel_err": float(abs(err[2]) / max(abs(gt_c[2]), 1e-6)),
                "err_3d_m": float(np.linalg.norm(err)),
                "depth_source": est.depth_source,
                "ground_applied": bool(est.ground_applied),
                "valid_ratio": float(est.valid_ratio),
                "n_pixels": int(est.n_pixels),
                "confidence": float(est.confidence),
            }
        )
    return rows


def summarize_rows(rows: Sequence[Dict[str, Any]], use_constraint: bool = True) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    dz = [r["dz_m"] for r in rows]
    rel = [r["depth_rel_err"] for r in rows]
    e3d = [r["err_3d_m"] for r in rows]
    return {
        "n": len(rows),
        "dz": signed_error_stats(dz),
        "depth_abs": abs_error_stats([abs(v) for v in dz]),
        "depth_rel": abs_error_stats(rel),
        "err_3d": abs_error_stats(e3d),
        "ground_applied": sum(1 for r in rows if r["ground_applied"]),
    }


# --------------------------------------------------------------------------- #
@dataclass
class CoverageReport:
    total_px: int
    valid_px: int
    coverage: float
    sgbm_px: int
    learned_px: int
    coverage_on_reference_valid: Optional[float]
    valid_on_object_px: Optional[float]


def coverage_report(fused: FusedDisparity, reference_valid: Optional[np.ndarray] = None) -> CoverageReport:
    from pc.fuse.router import SOURCE_LEARNED, SOURCE_SGBM

    total = int(fused.disp.size)
    valid = int(fused.valid.sum())
    cov_ref = None
    if reference_valid is not None and reference_valid.size:
        n_ref = int(reference_valid.sum())
        cov_ref = float((fused.valid & reference_valid).sum() / n_ref) if n_ref else None
    return CoverageReport(
        total_px=total,
        valid_px=valid,
        coverage=valid / total if total else 0.0,
        sgbm_px=int((fused.source == SOURCE_SGBM).sum()),
        learned_px=int((fused.source == SOURCE_LEARNED).sum()),
        coverage_on_reference_valid=cov_ref,
        valid_on_object_px=None,
    )


def depth_error_on_reference(
    fused: FusedDisparity, gt_disp: np.ndarray, reference_valid: np.ndarray
) -> Dict[str, Any]:
    """在真值有效区域计算视差误差，用于压缩伪影等离线对照。"""
    v = fused.valid & reference_valid
    if not np.any(v):
        return {"n": 0}
    err = fused.disp[v].astype(np.float64) - gt_disp[v].astype(np.float64)
    return {
        "n": int(v.sum()),
        "median_abs_px": float(np.median(np.abs(err))),
        "p90_abs_px": float(np.percentile(np.abs(err), 90)),
        "bias_px": float(np.median(err)),
    }


def depth_shift_on_mask(
    fused: FusedDisparity, mask: np.ndarray, calib, reference_fused: FusedDisparity
) -> Optional[float]:
    """同一物体区域内，两组视差换算出的深度中位数之差（米）。"""
    m = mask & fused.valid & reference_fused.valid
    if int(m.sum()) < 20:
        return None
    z1 = calib.focal_px * calib.baseline_m / fused.disp[m]
    z2 = calib.focal_px * calib.baseline_m / reference_fused.disp[m]
    return float(np.median(z1) - np.median(z2))
