"""测试共用夹具：合成标定、假标定、场景。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from pc.calib.calibrate import CalibrationResult, calibrate_stereo
from shared.synthetic import SceneConfig, SyntheticScene, default_objects, generate_calibration_views

_CALIB: Optional[CalibrationResult] = None


def make_fake_calib(width: int = 640, height: int = 480, fx: float = 530.0, baseline: float = 0.083) -> CalibrationResult:
    K = np.array([[fx, 0, (width - 1) / 2], [0, fx, (height - 1) / 2], [0, 0, 1]], float)
    return CalibrationResult(
        image_size=(width, height),
        K1=K.copy(), D1=np.zeros(5), K2=K.copy(), D2=np.zeros(5),
        R=np.eye(3), T=np.array([-baseline, 0.0, 0.0]),
        E=np.zeros((3, 3)), F=np.zeros((3, 3)),
        R1=np.eye(3), R2=np.eye(3), P1=K.copy(), P2=K.copy(), Q=np.eye(4),
        rms_stereo=0.0, rms_mono_left=0.0, rms_mono_right=0.0,
        pattern=(9, 6), square_m=0.03, per_view=[], meta={},
    )


def make_scene(width: int = 640, height: int = 480, fx: float = 530.0, baseline: float = 0.083) -> SyntheticScene:
    cfg = SceneConfig(width=width, height=height, fx=fx, fy=fx, baseline=baseline, camera_height=0.25)
    return SyntheticScene(objects=default_objects(), config=cfg)


def get_cached_calib() -> CalibrationResult:
    """生成一次（约 3 秒）并缓存到 runs/_test，供多个测试复用。"""
    global _CALIB
    if _CALIB is not None:
        return _CALIB
    path = Path("runs/_test/stereo_calib.npz")
    if path.exists():
        _CALIB = CalibrationResult.load(path)
        return _CALIB
    views = generate_calibration_views(
        n_views=14, seed=5, image_size=(640, 480),
        D_left=np.zeros(5), D_right=np.zeros(5), baseline=0.083,
    )
    calib = calibrate_stereo(
        views["left"], views["right"], pattern=(9, 6), square_m=0.03, min_views=10
    )
    calib.meta["source"] = "test"
    calib.save(path)
    _CALIB = calib
    return calib
