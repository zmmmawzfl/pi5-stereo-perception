"""帧对落盘 / 回放：阶段一"正确的数据长什么样"的基线载体。

目录结构::

    <root>/
      pairs.jsonl          # 每行一对的元数据（pair_id, seq, ts_L, ts_R, 文件）
      frames/000000_left.png
      frames/000000_right.png

JSONL 逐行 flush，采集中途 Ctrl-C 也不会丢掉已写入的索引。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RecordedPair:
    pair_id: int
    seq: int
    ts_left_ns: int
    ts_right_ns: int
    left_path: str
    right_path: str

    @property
    def ts_diff_ns(self) -> int:
        return self.ts_right_ns - self.ts_left_ns


class PairRecorder:
    def __init__(
        self,
        root: str | Path,
        image_format: str = "png",
        jpeg_quality: int = 95,
    ) -> None:
        self.root = Path(root)
        self.frame_dir = self.root / "frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.image_format = image_format.lower()
        self.jpeg_quality = int(jpeg_quality)
        self.index_path = self.root / "pairs.jsonl"
        self._index = open(self.index_path, "a", encoding="utf-8")
        self.count = 0

    def add(
        self,
        left: np.ndarray,
        right: np.ndarray,
        ts_left_ns: int,
        ts_right_ns: int,
        pair_id: int,
        seq: Optional[int] = None,
    ) -> RecordedPair:
        seq = self.count if seq is None else int(seq)
        stem = f"{seq:06d}"
        left_rel = f"frames/{stem}_left.{self.image_format}"
        right_rel = f"frames/{stem}_right.{self.image_format}"
        params: List[int] = []
        if self.image_format in ("jpg", "jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        if not cv2.imwrite(str(self.root / left_rel), left, params):
            raise IOError(f"写入失败: {left_rel}")
        if not cv2.imwrite(str(self.root / right_rel), right, params):
            raise IOError(f"写入失败: {right_rel}")
        rec = RecordedPair(int(pair_id), seq, int(ts_left_ns), int(ts_right_ns), left_rel, right_rel)
        self._index.write(
            json.dumps(
                {
                    "pair_id": rec.pair_id,
                    "seq": rec.seq,
                    "ts_left_ns": rec.ts_left_ns,
                    "ts_right_ns": rec.ts_right_ns,
                    "ts_diff_ms": rec.ts_diff_ns / 1e6,
                    "left": rec.left_path,
                    "right": rec.right_path,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._index.flush()
        self.count += 1
        return rec

    def close(self) -> None:
        if not self._index.closed:
            self._index.close()

    def __enter__(self) -> "PairRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class PairReplay:
    """读取一个录制目录。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "pairs.jsonl"
        if not self.index_path.exists():
            raise FileNotFoundError(f"找不到录制索引: {self.index_path}")
        self.records: List[RecordedPair] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            self.records.append(
                RecordedPair(
                    pair_id=int(obj["pair_id"]),
                    seq=int(obj["seq"]),
                    ts_left_ns=int(obj["ts_left_ns"]),
                    ts_right_ns=int(obj["ts_right_ns"]),
                    left_path=obj["left"],
                    right_path=obj["right"],
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Tuple[RecordedPair, np.ndarray, np.ndarray]]:
        for rec in self.records:
            left = cv2.imread(str(self.root / rec.left_path), cv2.IMREAD_COLOR)
            right = cv2.imread(str(self.root / rec.right_path), cv2.IMREAD_COLOR)
            if left is None or right is None:
                raise FileNotFoundError(f"帧文件缺失: {rec.left_path} / {rec.right_path}")
            yield rec, left, right

    def load(self, rec: RecordedPair) -> Tuple[np.ndarray, np.ndarray]:
        left = cv2.imread(str(self.root / rec.left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(self.root / rec.right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise FileNotFoundError(f"帧文件缺失: {rec.left_path} / {rec.right_path}")
        return left, right


def write_json(path: str | Path, payload: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
