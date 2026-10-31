"""YOLOv8 检测器（TensorRT / PyTorch），YOLO 权重不在仓库内。

用法：
  * 在 PC 上导出 FP16 引擎：``yolo export model=yolov8n.pt format=engine half=True``
  * 配置 ``detect.model_path`` 指向 .engine 或 .pt；
  * 缺权重时不会静默降级，而是抛 DetectorUnavailable，由上层决定是否改用
    ColorBlobDetector（并在报告里写明"未使用 YOLO"）。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import Detection


class DetectorUnavailable(RuntimeError):
    pass


class YoloDetector:
    name = "yolo"

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        imgsz: int = 640,
        classes: Optional[List[int]] = None,
        device: str = "0",
    ) -> None:
        if not model_path:
            raise DetectorUnavailable("未配置 detect.model_path（YOLO 权重/引擎路径）")
        if not Path(model_path).exists():
            raise DetectorUnavailable(f"YOLO 模型不存在: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailable(
                "缺少 ultralytics：pip install ultralytics；或用 --detect-backend color 先跑通链路"
            ) from exc
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes
        self.device = device
        self.names = self.model.names

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        out: List[Detection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            name = self.names.get(int(k), str(k)) if isinstance(self.names, dict) else str(self.names[int(k)])
            out.append(Detection(name, (float(x1), float(y1), float(x2), float(y2)), float(c)))
        return out
