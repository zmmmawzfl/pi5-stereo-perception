"""Pi 端配置便捷入口（复用 shared.config，默认值针对 Pi 5 调整）。"""
from __future__ import annotations

from shared.config import AppConfig, load_config

PI_OVERRIDES = {
    "camera.backend": "auto",
    "stream.transport": "zmq",
    "stream.jpeg_quality": 90,
    "sync.kp": 0.05,
}


def load_pi_config(path: str | None = None, overrides: dict | None = None) -> AppConfig:
    merged = dict(PI_OVERRIDES)
    merged.update(overrides or {})
    return load_config(path, merged)


__all__ = ["AppConfig", "load_pi_config"]
