"""合成双目场景与棋盘格标定数据（硬件未到货时的真值来源）。

为什么需要它：
  * 摄像头没到，但整条链路的**接口、单位、符号约定**现在就必须锁死；
  * 合成场景能给出逐像素真值深度、真值视差、物体真值中心，从而在没有
    实机的情况下验证 SGBM、融合路由、物体聚合、地面约束是否真的算对；
  * 它只用于验证与回归测试，不参与实机路径。

坐标约定（全项目统一）：
  * 相机坐标系：X 向右，Y 向下，Z 向前（右手系，与 OpenCV 一致）；
  * 左相机光心为原点，右相机光心在 (+B, 0, 0)；
  * 校正后视差 d = u_L - u_R = f * B / Z > 0；
  * 地面为平面 Y = camera_height（相机离地高度），物体坐在地面上。

场景渲染用矢量光线投射：每个像素对地面 + 若干轴对齐盒子求交，天然带
遮挡与正确深度，不依赖任何渲染库。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class SceneObject:
    """一个坐在地面上的轴对齐盒子。"""

    name: str
    x0: float
    x1: float
    z0: float
    z1: float
    height: float
    color_bgr: Tuple[int, int, int]
    class_name: str = "object"

    def bounds(self, ground_y: float) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array([min(self.x0, self.x1), ground_y - self.height, min(self.z0, self.z1)], float)
        hi = np.array([max(self.x0, self.x1), ground_y, max(self.z0, self.z1)], float)
        return lo, hi


@dataclass
class SceneConfig:
    width: int = 640
    height: int = 480
    fx: float = 530.0
    fy: float = 530.0
    cx: Optional[float] = None
    cy: Optional[float] = None
    baseline: float = 0.083
    camera_height: float = 0.25
    noise_sigma: float = 1.5
    max_depth: float = 6.0
    exposure_gain_right: float = 1.02

    def __post_init__(self) -> None:
        if self.cx is None:
            self.cx = (self.width - 1) / 2.0
        if self.cy is None:
            self.cy = (self.height - 1) / 2.0


@dataclass
class StereoRender:
    """一帧合成双目数据 + 真值。"""

    left: np.ndarray          # HxWx3 uint8 BGR
    right: np.ndarray         # HxWx3 uint8 BGR
    depth: np.ndarray         # HxW float32，左目 Z（米），无效为 nan
    disp: np.ndarray          # HxW float32，真值视差（像素）
    valid: np.ndarray         # HxW bool，未被遮挡/出界
    instances: np.ndarray     # HxW int16，-1 地面，-2 背景，>=0 物体下标
    objects: List[Dict[str, object]] = field(default_factory=list)
    config: Optional[SceneConfig] = None

    def object_mask(self, index: int) -> np.ndarray:
        return self.instances == index


# --------------------------------------------------------------------------- #
# 默认场景：三个不同距离的物体 + 强纹理地面
# --------------------------------------------------------------------------- #
def default_objects() -> List[SceneObject]:
    """深度约 0.65 / 1.00 / 1.50 m 的三个盒子，颜色可被 HSV 分割。"""
    return [
        SceneObject("near_red", -0.24, -0.06, 0.62, 0.74, 0.16, (40, 40, 210)),
        SceneObject("mid_green", 0.04, 0.21, 0.95, 1.08, 0.20, (50, 175, 60)),
        SceneObject("far_blue", -0.10, 0.06, 1.45, 1.58, 0.14, (205, 90, 40)),
    ]


def default_scene(**kwargs) -> "SyntheticScene":
    return SyntheticScene(objects=default_objects(), **kwargs)


class SyntheticScene:
    def __init__(
        self,
        objects: Optional[Sequence[SceneObject]] = None,
        config: Optional[SceneConfig] = None,
        **config_kwargs,
    ) -> None:
        self.cfg = config or SceneConfig(**config_kwargs)
        self.objects = list(objects if objects is not None else default_objects())

    # ---------------- 渲染 ----------------
    def render(self, seed: int = 0) -> StereoRender:
        left, depth_l, inst_l = self._render_view(0.0)
        right, depth_r, _ = self._render_view(self.cfg.baseline)

        # 真值视差 / 遮挡有效性
        h, w = depth_l.shape
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            disp = (self.cfg.fx * self.cfg.baseline / depth_l).astype(np.float32)
        disp[~np.isfinite(disp)] = 0.0

        u_r = np.rint(xs - disp).astype(np.int32)
        inb = (u_r >= 0) & (u_r < w) & np.isfinite(depth_l) & (disp > 0)
        valid = inb.copy()
        # 右目同一投影像素若明显更近，说明该点被遮挡
        rr = np.zeros_like(depth_l)
        rr[inb] = depth_r[ys[inb].astype(int), u_r[inb]]
        with np.errstate(invalid="ignore"):
            occluded = inb & np.isfinite(rr) & (rr < depth_l - 0.02 * depth_l - 0.01)
        valid &= ~occluded

        # 加传感器噪声（左右独立）
        rng = np.random.default_rng(seed)
        left = self._add_noise(left, rng, 1.0)
        right = self._add_noise(right, rng, self.cfg.exposure_gain_right)

        return StereoRender(
            left=left,
            right=right,
            depth=depth_l.astype(np.float32),
            disp=disp,
            valid=valid.astype(bool),
            instances=inst_l,
            objects=self._object_gt(inst_l, depth_l),
            config=self.cfg,
        )

    def _camera_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.cfg.fx, 0.0, self.cfg.cx],
                [0.0, self.cfg.fy, self.cfg.cy],
                [0.0, 0.0, 1.0],
            ],
            float,
        )

    def _render_view(self, offset_x: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = self.cfg.height, self.cfg.width
        u = np.arange(w, dtype=np.float64)
        v = np.arange(h, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        dx = (uu - self.cfg.cx) / self.cfg.fx
        dy = (vv - self.cfg.cy) / self.cfg.fy
        # 光心平移等价于对 X 加偏移
        ox, oy, oz = offset_x, 0.0, 0.0

        depth = np.full((h, w), np.inf)
        inst = np.full((h, w), -2, dtype=np.int16)
        img = self._background(uu, vv)

        # --- 地面 ---
        ground_y = self.cfg.camera_height
        with np.errstate(divide="ignore", invalid="ignore"):
            t_ground = (ground_y - oy) / dy
        g_hit = (dy > 1e-9) & (t_ground > 0) & (t_ground < self.cfg.max_depth)
        if np.any(g_hit):
            X = ox + dx[g_hit] * t_ground[g_hit]
            Z = oz + t_ground[g_hit]
            tex = self._ground_texture(X, Z)
            img[g_hit] = tex
            depth[g_hit] = t_ground[g_hit]
            inst[g_hit] = -1

        # --- 盒子（近到远逐个做 z-buffer） ---
        for idx, obj in enumerate(self.objects):
            lo, hi = obj.bounds(ground_y)
            tmin, tmax, axis = _ray_box(dx, dy, ox, oy, oz, lo, hi)
            hit = np.isfinite(tmin) & (tmin > 0) & (tmax >= tmin)
            closer = hit & (tmin < depth)
            if not np.any(closer):
                continue
            t = tmin[closer]
            X = ox + dx[closer] * t
            Y = oy + dy[closer] * t
            Z = oz + t
            ax = axis[closer]
            uu_b, vv_b = _face_uv(X, Y, Z, lo, hi, ax)
            tex = self._box_texture(obj, uu_b, vv_b, ax)
            img[closer] = tex
            depth[closer] = t
            inst[closer] = idx

        return np.clip(img, 0, 255).astype(np.uint8), depth, inst

    def _background(self, uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        h, w = uu.shape
        base = 150.0 - 45.0 * (vv / max(h - 1, 1))
        img = np.stack([base * 0.98, base * 0.99, base], axis=-1)
        # 远处墙面纹理，避免背景完全无纹理
        img += (10.0 * np.sin(uu / 23.0) * np.cos(vv / 31.0))[..., None]
        return img

    @staticmethod
    def _ground_texture(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        # 带限纹理：最低频 12 cm、最高频 4.5 cm，避免远距离混叠成白噪声
        s = (
            0.9 * np.sin(X * 2 * np.pi / 0.120) * np.sin(Z * 2 * np.pi / 0.145)
            + 0.8 * np.sin((X + Z) * 2 * np.pi / 0.075)
            + 0.6 * np.sin((X - 0.7 * Z) * 2 * np.pi / 0.045)
        )
        val = 118.0 + 42.0 * s
        return np.stack([val * 0.97, val * 0.99, val], axis=-1)

    @staticmethod
    def _box_texture(obj: SceneObject, u: np.ndarray, v: np.ndarray, axis: np.ndarray) -> np.ndarray:
        base = np.array(obj.color_bgr, float)[None, :]
        checker = ((np.floor(u * 10) + np.floor(v * 8)) % 2).astype(float)
        stripes = 0.90 + 0.10 * np.sin((u * 4.0 + v * 2.0) * 2 * np.pi)
        shade = np.where(axis == 1, 1.15, np.where(axis == 0, 0.72, 1.0))
        mod = (0.80 + 0.30 * checker) * stripes * shade
        return base * mod[:, None] + 8.0

    def _add_noise(self, img: np.ndarray, rng: np.random.Generator, gain: float) -> np.ndarray:
        out = img.astype(np.float32) * gain
        if self.cfg.noise_sigma > 0:
            out += rng.normal(0.0, self.cfg.noise_sigma, out.shape).astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ---------------- 真值 ----------------
    def _object_gt(self, inst: np.ndarray, depth: np.ndarray) -> List[Dict[str, object]]:
        h, w = inst.shape
        ys, xs = np.mgrid[0:h, 0:w]
        out: List[Dict[str, object]] = []
        for idx, obj in enumerate(self.objects):
            mask = inst == idx
            if not np.any(mask):
                continue
            ys_m, xs_m = ys[mask], xs[mask]
            bbox = [int(xs_m.min()), int(ys_m.min()), int(xs_m.max()), int(ys_m.max())]
            zs = depth[mask]
            # 真值：可见表面（前表面）中心。深度相机只能看到表面，不是盒子几何中心。
            gt = {
                "index": idx,
                "name": obj.name,
                "class_name": obj.class_name,
                "bbox_xyxy": bbox,
                "gt_surface_center_m": [
                    float((obj.x0 + obj.x1) / 2.0),
                    float(self.cfg.camera_height - obj.height / 2.0),
                    float(min(obj.z0, obj.z1)),
                ],
                "gt_box_center_m": [
                    float((obj.x0 + obj.x1) / 2.0),
                    float(self.cfg.camera_height - obj.height / 2.0),
                    float((obj.z0 + obj.z1) / 2.0),
                ],
                "gt_depth_median_m": float(np.median(zs[np.isfinite(zs)])),
                "gt_front_depth_m": float(min(obj.z0, obj.z1)),
                "n_pixels": int(mask.sum()),
            }
            out.append(gt)
        return out


# --------------------------------------------------------------------------- #
# 几何工具
# --------------------------------------------------------------------------- #
def _ray_box(
    dx: np.ndarray,
    dy: np.ndarray,
    ox: float,
    oy: float,
    oz: float,
    lo: np.ndarray,
    hi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """向量化 slab 求交，返回进入/离开距离与进入面轴（0=X,1=Y,2=Z）。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.stack([dx, dy, np.ones_like(dx)], axis=-1)
    o = np.array([ox, oy, oz], float)
    t1 = (lo[None, None, :] - o) * inv
    t2 = (hi[None, None, :] - o) * inv
    tlo = np.minimum(t1, t2)
    thi = np.maximum(t1, t2)
    tmin = np.max(tlo, axis=-1)
    tmax = np.min(thi, axis=-1)
    axis = np.argmax(tlo, axis=-1).astype(np.int16)
    return tmin, tmax, axis


def _face_uv(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, lo: np.ndarray, hi: np.ndarray, axis: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    span = np.maximum(hi - lo, 1e-9)
    u = np.where(
        axis == 0,
        (Z - lo[2]) / span[2],
        (X - lo[0]) / span[0],
    )
    v = np.where(
        axis == 0,
        (Y - lo[1]) / span[1],
        np.where(axis == 1, (Z - lo[2]) / span[2], (Y - lo[1]) / span[1]),
    )
    return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 棋盘格与标定视图
# --------------------------------------------------------------------------- #
def make_checker_texture(
    cols: int, rows: int, square_px: int = 40, margin_px: Optional[int] = None
) -> np.ndarray:
    """生成 (cols+1) x (rows+1) 个方格的棋盘格，四周留半格白边（静区）。

    方格数比内角点数多 1；否则画面里会多出可检测角点，导致
    findChessboardCorners 失败。白边是 OpenCV 角点检测要求的 quiet zone。
    """
    margin = square_px // 2 if margin_px is None else int(margin_px)
    nx, ny = cols + 1, rows + 1
    tex = np.full((ny * square_px + 2 * margin, nx * square_px + 2 * margin, 3), 255, np.uint8)
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                y0 = margin + j * square_px
                x0 = margin + i * square_px
                tex[y0 : y0 + square_px, x0 : x0 + square_px] = 30
    return tex


def _euler_to_rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """绕相机轴旋转：先绕 Y（yaw），再绕 X（pitch），最后绕 Z（roll）。"""
    ry = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]], float
    )
    rx = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]], float
    )
    rz = np.array(
        [[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]], float
    )
    return rz @ rx @ ry


def board_geometry_m(
    board_cols: int, board_rows: int, square_m: float, margin_squares: float = 0.5
) -> Tuple[float, float, np.ndarray]:
    """棋盘的物理宽高与中心（含白边），单位米。"""
    w = (board_cols + 1 + 2 * margin_squares) * square_m
    h = (board_rows + 1 + 2 * margin_squares) * square_m
    return w, h, np.array([w / 2.0, h / 2.0, 0.0])


def render_checkerboard(
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    board_cols: int,
    board_rows: int,
    square_m: float,
    image_size: Tuple[int, int],
    square_px: int = 40,
    background: int = 85,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """把棋盘格平面按针孔模型投影到图像；若有畸变 D，再对图像做畸变重采样。

    返回渲染图与真值角点（内角点，按 OpenCV 行优先顺序）。
    """
    w, h = image_size
    margin_sq = 0.5
    tex = make_checker_texture(board_cols, board_rows, square_px)
    tex_w_px, tex_h_px = tex.shape[1], tex.shape[0]
    margin_px = square_px // 2

    # 纹理像素 -> 棋盘米制坐标（原点在纹理左上角）
    scale = square_m / square_px
    S = np.diag([scale, scale, 1.0])
    _, _, center = board_geometry_m(board_cols, board_rows, square_m, margin_sq)

    H = K @ np.column_stack([R[:, 0], R[:, 1], t - R @ center]) @ S
    H /= H[2, 2]
    img = cv2.warpPerspective(
        tex, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(background, background, background),
    )

    # 内角点：纹理中 margin + (i+1) 个方格交界
    corners = []
    for j in range(board_rows):
        for i in range(board_cols):
            corners.append(
                [
                    (margin_px + (i + 1) * square_px) * scale,
                    (margin_px + (j + 1) * square_px) * scale,
                    0.0,
                ]
            )
    corners_3d = np.asarray(corners, np.float32)
    corners_cam = (R @ (corners_3d - center.astype(np.float32)).T).T + t
    proj = (K @ corners_cam.T).T
    proj = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)

    if D is not None and np.any(np.abs(D) > 0):
        img = distort_image(img, K, D, background)

    info = {"corners_3d": corners_3d, "corners_2d": proj, "H": H}
    return img, info


def distort_image(
    img: np.ndarray, K: np.ndarray, D: np.ndarray, background: int = 85
) -> np.ndarray:
    """对针孔图像施加镜头畸变（目标像素是畸变坐标，反查无畸变坐标采样）。"""
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    und = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D, P=K).reshape(h, w, 2)
    mapx = und[:, :, 0].astype(np.float32)
    mapy = und[:, :, 1].astype(np.float32)
    return cv2.remap(
        img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(background, background, background),
    )


def sample_board_poses(
    rng: np.random.Generator,
    n: int,
    K: np.ndarray,
    board_cols: int,
    board_rows: int,
    square_m: float,
    image_size: Tuple[int, int],
    min_depth: float = 0.40,
    max_depth: float = 0.78,
    max_tries: int = 500,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """采样 n 个"棋盘格完整落在画面内"的位姿 (R, t)。"""
    w, h = image_size
    poses: List[Tuple[np.ndarray, np.ndarray]] = []
    tries = 0
    while len(poses) < n and tries < max_tries:
        tries += 1
        d = rng.uniform(min_depth, max_depth)
        yaw = rng.uniform(-0.45, 0.45)
        pitch = rng.uniform(-0.35, 0.35)
        roll = rng.uniform(-0.30, 0.30)
        R = _euler_to_rot(yaw, pitch, roll)
        t = np.array(
            [
                rng.uniform(-0.09, 0.09) * d / 0.6,
                rng.uniform(-0.06, 0.06) * d / 0.6,
                d,
            ],
            float,
        )
        # 检查四个外角是否都在画面内并留边（含白边）
        board_w, board_h, center = board_geometry_m(board_cols, board_rows, square_m)
        corners = np.array(
            [
                [0, 0, 0],
                [board_w, 0, 0],
                [board_w, board_h, 0],
                [0, board_h, 0],
            ],
            float,
        )
        cam = (R @ (corners - center).T).T + t
        if np.any(cam[:, 2] <= 0.05):
            continue
        proj = (K @ cam.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        margin = 12
        if (
            proj[:, 0].min() < margin
            or proj[:, 0].max() > w - margin
            or proj[:, 1].min() < margin
            or proj[:, 1].max() > h - margin
        ):
            continue
        # 棋盘投影面积不能太小，保证角点可检测
        area = cv2.contourArea(proj.astype(np.float32))
        if area < 0.12 * w * h:
            continue
        poses.append((R, t))
    if len(poses) < n:
        raise RuntimeError(f"只采到 {len(poses)}/{n} 个可用标定位姿，请放宽约束")
    return poses


def generate_calibration_views(
    n_views: int = 15,
    seed: int = 3,
    image_size: Tuple[int, int] = (640, 480),
    K_left: Optional[np.ndarray] = None,
    K_right: Optional[np.ndarray] = None,
    D_left: Optional[np.ndarray] = None,
    D_right: Optional[np.ndarray] = None,
    baseline: float = 0.083,
    board_cols: int = 9,
    board_rows: int = 6,
    square_m: float = 0.03,
) -> Dict[str, object]:
    """生成一组左右目棋盘格视图（带真实畸变），用于离线标定回归测试。"""
    w, h = image_size
    K_left = K_left if K_left is not None else np.array(
        [[530.0, 0, (w - 1) / 2], [0, 530.0, (h - 1) / 2], [0, 0, 1]], float
    )
    K_right = K_right if K_right is not None else K_left.copy()
    D_left = np.zeros(5) if D_left is None else np.asarray(D_left, float)
    D_right = np.zeros(5) if D_right is None else np.asarray(D_right, float)

    rng = np.random.default_rng(seed)
    poses = sample_board_poses(rng, n_views, K_left, board_cols, board_rows, square_m, image_size)

    left_imgs, right_imgs, obj_pts = [], [], []
    gt_left_pts, gt_right_pts = [], []
    R_stereo = np.eye(3)
    for R, t in poses:
        img_l, info_l = render_checkerboard(
            K_left, D_left, R, t, board_cols, board_rows, square_m, (w, h)
        )
        img_r, info_r = render_checkerboard(
            K_right, D_right, R, t + np.array([-baseline, 0.0, 0.0]),
            board_cols, board_rows, square_m, (w, h),
        )
        left_imgs.append(img_l)
        right_imgs.append(img_r)
        obj_pts.append(info_l["corners_3d"].astype(np.float32))
        gt_left_pts.append(info_l["corners_2d"])
        gt_right_pts.append(info_r["corners_2d"])

    return {
        "left": left_imgs,
        "right": right_imgs,
        "object_points": obj_pts,
        "gt_left_corners": gt_left_pts,
        "gt_right_corners": gt_right_pts,
        "K_left": K_left,
        "K_right": K_right,
        "D_left": D_left,
        "D_right": D_right,
        "R": R_stereo,
        "T": np.array([-baseline, 0.0, 0.0]),
        "image_size": image_size,
        "pattern": (board_cols, board_rows),
    }
