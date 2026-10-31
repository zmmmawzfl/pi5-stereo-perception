"""采集后端工厂。"""
from __future__ import annotations

from typing import Optional

from shared.config import AppConfig

from .base import CameraBackend
from .synthetic_backend import ReplayBackend, SyntheticBackend


def has_picamera2() -> bool:
    try:
        import picamera2  # noqa: F401

        return True
    except Exception:
        return False


def build_backend(cfg: AppConfig, replay_root: Optional[str] = None) -> CameraBackend:
    backend = (cfg.camera.backend or "auto").lower()
    if backend == "auto":
        backend = "picamera2" if has_picamera2() else "synthetic"
        print(f"[capture] backend=auto -> {backend}")

    if backend == "picamera2":
        from .picamera2_backend import Picamera2Backend

        return Picamera2Backend(
            width=cfg.camera.width,
            height=cfg.camera.height,
            fps=cfg.camera.fps,
            buffer_count=cfg.camera.buffer_count,
            left_index=cfg.camera.left_index,
            right_index=cfg.camera.right_index,
        )

    if backend == "synthetic":
        scene_kwargs = dict(cfg.scene or {})
        return SyntheticBackend(
            width=cfg.camera.width,
            height=cfg.camera.height,
            fps=cfg.camera.fps,
            **scene_kwargs,
        )

    if backend == "replay":
        if not replay_root:
            raise ValueError("replay 模式需要 --replay-root 指向录制目录")
        return ReplayBackend(replay_root)

    raise ValueError(f"未知 camera.backend: {cfg.camera.backend}")
