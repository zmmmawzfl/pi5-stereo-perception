"""检测器工厂：显式选择，错误可见。

auto 的选择规则：
  1. 配了 model_path 且文件存在 → YOLO；
  2. 否则 → ColorBlobDetector（报告里会写 detector=color）。
显式指定 yolo 而权重缺失时直接报错，不静默降级。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from shared.config import DetectConfig

from .base import Detection, Detector
from .color import ColorBlobDetector, NullDetector
from .yolo import YoloDetector


def build_detector(cfg: DetectConfig) -> Detector:
    backend = (cfg.backend or "auto").lower()
    if backend == "none":
        return NullDetector()
    if backend == "color":
        return ColorBlobDetector()
    if backend == "yolo":
        classes: Optional[List[int]] = None
        if cfg.classes:
            classes = [int(v) for v in cfg.classes.split(",") if v.strip()]
        return YoloDetector(cfg.model_path, cfg.conf, cfg.imgsz, classes, cfg.device)
    if backend == "auto":
        from pathlib import Path

        if cfg.model_path and Path(cfg.model_path).exists():
            classes = None
            if cfg.classes:
                classes = [int(v) for v in cfg.classes.split(",") if v.strip()]
            return YoloDetector(cfg.model_path, cfg.conf, cfg.imgsz, classes, cfg.device)
        print("[detect] 未找到 YOLO 权重，使用 ColorBlobDetector（仅用于链路验证）")
        return ColorBlobDetector()
    raise ValueError(f"未知 detect.backend: {cfg.backend}")
