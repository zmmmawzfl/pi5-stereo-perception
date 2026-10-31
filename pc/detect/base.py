"""检测器统一接口。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Tuple

import numpy as np


@dataclass
class Detection:
    class_name: str
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def clipped(self, width: int, height: int) -> "Detection":
        x1, y1, x2, y2 = self.bbox_xyxy
        return Detection(
            self.class_name,
            (
                float(np.clip(x1, 0, width - 1)),
                float(np.clip(y1, 0, height - 1)),
                float(np.clip(x2, 0, width - 1)),
                float(np.clip(y2, 0, height - 1)),
            ),
            self.confidence,
            self.extra,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "class": self.class_name,
            "bbox_xyxy": [round(float(v), 2) for v in self.bbox_xyxy],
            "confidence": round(float(self.confidence), 4),
        }


class Detector(Protocol):
    name: str

    def detect(self, bgr: np.ndarray) -> List[Detection]: ...
