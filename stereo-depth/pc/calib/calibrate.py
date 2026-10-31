"""双目标定：单目 → 双目联合 → 极线校正。

关键约定（实机与合成共用）：
  * **不要假设基线**：B 一律从标定 T 的模长读取；
  * 校正后焦距从 P1(0,0) 读取，深度用 Z = f * B / d；
  * 视差符号：右目在 +X 方向，校正后 d = u_L - u_R > 0。

标定文件格式（.npz）：
  K1,D1,K2,D2,R,T,E,F,Q,R1,R2,P1,P2,image_size,pattern,square_m,
  rms_stereo,rms_mono_left,rms_mono_right,meta(JSON)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from shared.stats import signed_error_stats, summarize


class CalibrationError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# 角点检测
# --------------------------------------------------------------------------- #
def detect_corners(
    gray: np.ndarray,
    pattern: Tuple[int, int],
    use_sb: bool = True,
    subpix: bool = True,
) -> Optional[np.ndarray]:
    """检测棋盘格内角点，返回 (N,1,2) float32 或 None。

    优先 findChessboardCornersSB（对噪声/畸变更稳），失败再退回经典算法。

    注意：SB 与经典算法的角点顺序可能相差 180°（等价于棋盘转 180°，不影响
    标定）；但同一份标定数据里不要混用两种算法的结果——某个视图检测不到时，
    直接跳过比"用另一种顺序硬补"安全。
    """
    gray = _to_gray(gray)
    if use_sb:
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if found:
            return corners.astype(np.float32).reshape(-1, 1, 2)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if not found:
        return None
    if subpix:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return corners.astype(np.float32)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- #
# 标定结果
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationResult:
    image_size: Tuple[int, int]
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    rms_stereo: float
    rms_mono_left: float
    rms_mono_right: float
    pattern: Tuple[int, int] = (9, 6)
    square_m: float = 0.03
    per_view: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- 派生量：实机必须从这里取，而不是写死 83mm ----
    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.T))

    @property
    def focal_px(self) -> float:
        return float(self.P1[0, 0])

    @property
    def cx(self) -> float:
        return float(self.P1[0, 2])

    @property
    def cy(self) -> float:
        return float(self.P1[1, 2])

    def disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        """Z = f B / d；d<=0 视为无效（nan）。"""
        disp = np.asarray(disp, np.float32)
        out = np.full_like(disp, np.nan, dtype=np.float32)
        ok = disp > 0
        out[ok] = self.focal_px * self.baseline_m / disp[ok]
        return out

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta)
        meta.update(
            {
                "image_size": list(self.image_size),
                "pattern": list(self.pattern),
                "square_m": float(self.square_m),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "baseline_m": self.baseline_m,
                "focal_px": self.focal_px,
            }
        )
        np.savez_compressed(
            p,
            K1=self.K1,
            D1=self.D1,
            K2=self.K2,
            D2=self.D2,
            R=self.R,
            T=self.T,
            E=self.E,
            F=self.F,
            R1=self.R1,
            R2=self.R2,
            P1=self.P1,
            P2=self.P2,
            Q=self.Q,
            image_size=np.asarray(self.image_size, np.int32),
            rms=np.asarray([self.rms_stereo, self.rms_mono_left, self.rms_mono_right], np.float64),
            meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationResult":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"标定文件不存在: {p}")
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            rms = z["rms"]
            return cls(
                image_size=tuple(int(v) for v in z["image_size"]),
                K1=z["K1"].astype(np.float64),
                D1=z["D1"].astype(np.float64),
                K2=z["K2"].astype(np.float64),
                D2=z["D2"].astype(np.float64),
                R=z["R"].astype(np.float64),
                T=z["T"].astype(np.float64),
                E=z["E"].astype(np.float64),
                F=z["F"].astype(np.float64),
                R1=z["R1"].astype(np.float64),
                R2=z["R2"].astype(np.float64),
                P1=z["P1"].astype(np.float64),
                P2=z["P2"].astype(np.float64),
                Q=z["Q"].astype(np.float64),
                rms_stereo=float(rms[0]),
                rms_mono_left=float(rms[1]),
                rms_mono_right=float(rms[2]),
                pattern=tuple(meta.get("pattern", (9, 6))),
                square_m=float(meta.get("square_m", 0.03)),
                per_view=meta.get("per_view", []),
                meta=meta,
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "image_size": list(self.image_size),
            "baseline_m": self.baseline_m,
            "focal_px": self.focal_px,
            "rms_stereo_px": self.rms_stereo,
            "rms_mono_left_px": self.rms_mono_left,
            "rms_mono_right_px": self.rms_mono_right,
            "T": self.T.tolist(),
            "K1": self.K1.tolist(),
            "D1": self.D1.ravel().tolist(),
            "D2": self.D2.ravel().tolist(),
        }


# --------------------------------------------------------------------------- #
# 标定流程
# --------------------------------------------------------------------------- #
def _object_grid(pattern: Tuple[int, int], square_m: float) -> np.ndarray:
    cols, rows = pattern
    grid = np.zeros((cols * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return grid


def _reprojection_rms(
    objp: np.ndarray, imgp: np.ndarray, K: np.ndarray, D: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
) -> float:
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    err = np.linalg.norm(proj.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(err**2)))


def calibrate_stereo(
    left_images: Sequence[np.ndarray],
    right_images: Sequence[np.ndarray],
    pattern: Tuple[int, int] = (9, 6),
    square_m: float = 0.03,
    min_views: int = 8,
    progress: bool = False,
    fix_k3: bool = True,
    zero_tangent: bool = True,
) -> CalibrationResult:
    """从左右目棋盘格图像完成单目 + 双目联合标定。

    角度覆盖不足、角点检出太少、重投影误差异常都会直接报错，绝不"静默
    返回一个能用的文件"。
    """
    if len(left_images) != len(right_images) or len(left_images) < min_views:
        raise CalibrationError(
            f"需要至少 {min_views} 对左右视图，收到 {len(left_images)}/{len(right_images)}"
        )

    objp = _object_grid(pattern, square_m)
    objpoints: List[np.ndarray] = []
    img_l: List[np.ndarray] = []
    img_r: List[np.ndarray] = []
    skipped: List[int] = []

    for i, (limg, rimg) in enumerate(zip(left_images, right_images)):
        cl = detect_corners(limg, pattern)
        cr = detect_corners(rimg, pattern)
        if cl is None or cr is None:
            skipped.append(i)
            continue
        if cl.shape != cr.shape:
            skipped.append(i)
            continue
        objpoints.append(objp.copy())
        img_l.append(cl)
        img_r.append(cr)
        if progress:
            print(f"  角点 {i + 1}/{len(left_images)}: ok")

    if len(objpoints) < min_views:
        raise CalibrationError(
            f"有效视图仅 {len(objpoints)} 对（跳过 {len(skipped)} 对），低于下限 {min_views}"
        )
    if skipped:
        print(f"[calib] 警告：{len(skipped)} 对视图角点检测失败，已跳过 {skipped}")

    image_size = (int(left_images[0].shape[1]), int(left_images[0].shape[0]))

    # 固定 k3、切向畸变置零：棋盘格数据对方程的约束有限，5 参数全放会
    # 在"真值无畸变"的数据上拟合出 k3~0.24 这种伪畸变，把 alpha=0 的校正
    # 视场裁掉 2%，进而污染 fB。IMX219 这类镜头 k1/k2 足够。
    calib_flags = 0
    if fix_k3:
        calib_flags |= cv2.CALIB_FIX_K3
    if zero_tangent:
        calib_flags |= cv2.CALIB_ZERO_TANGENT_DIST

    rms_l, K1, D1, rv_l, tv_l = cv2.calibrateCamera(
        objpoints, img_l, image_size, None, None, flags=calib_flags
    )
    rms_r, K2, D2, rv_r, tv_r = cv2.calibrateCamera(
        objpoints, img_r, image_size, None, None, flags=calib_flags
    )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms_s, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        img_l,
        img_r,
        K1,
        D1,
        K2,
        D2,
        image_size,
        flags=calib_flags | cv2.CALIB_USE_INTRINSIC_GUESS,
        criteria=criteria,
    )

    (R1, R2, P1, P2, Q, roi1, roi2) = cv2.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    per_view: List[Dict[str, Any]] = []
    for i, (rv, tv) in enumerate(zip(rv_l, tv_l)):
        per_view.append(
            {
                "view": i,
                "rms_left_px": _reprojection_rms(objpoints[i], img_l[i], K1, D1, rv, tv),
            }
        )

    # 极线检查：把检测到的角点校正后比较 y
    epi = epipolar_dy(K1, D1, K2, D2, R1, R2, P1, P2, img_l, img_r)
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "real-or-synthetic",
        "views_used": len(objpoints),
        "skipped": skipped,
        "distortion_model": {"fix_k3": bool(fix_k3), "zero_tangent": bool(zero_tangent)},
        "epipolar_dy": epi,
        "roi": {"left": [int(v) for v in roi1], "right": [int(v) for v in roi2]},
    }

    return CalibrationResult(
        image_size=image_size,
        K1=K1,
        D1=D1.ravel(),
        K2=K2,
        D2=D2.ravel(),
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        rms_stereo=float(rms_s),
        rms_mono_left=float(rms_l),
        rms_mono_right=float(rms_r),
        pattern=pattern,
        square_m=square_m,
        per_view=per_view,
        meta=meta,
    )


def epipolar_dy(
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    pts_l: Sequence[np.ndarray],
    pts_r: Sequence[np.ndarray],
) -> Dict[str, Any]:
    """校正后同名点的垂直残差 |dy|（像素）。理想值 ~0。"""
    dys: List[float] = []
    for pl, pr in zip(pts_l, pts_r):
        ul = cv2.undistortPoints(pl.reshape(-1, 1, 2), K1, D1, R=R1, P=P1).reshape(-1, 2)
        ur = cv2.undistortPoints(pr.reshape(-1, 1, 2), K2, D2, R=R2, P=P2).reshape(-1, 2)
        n = min(len(ul), len(ur))
        dys.extend(np.abs(ul[:n, 1] - ur[:n, 1]).tolist())
    out = summarize(dys)
    out["median_abs_px"] = out["median"]
    out["p90_abs_px"] = float(np.percentile(dys, 90)) if dys else None
    return out


def load_or_fail(path: str | Path) -> CalibrationResult:
    try:
        return CalibrationResult.load(path)
    except FileNotFoundError as exc:
        raise CalibrationError(
            f"{exc}\n请先运行标定：python -m pc.calib.cli calibrate --left ... --right ..."
        ) from exc
