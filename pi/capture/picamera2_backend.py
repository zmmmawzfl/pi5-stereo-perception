"""Picamera2 双实例后端（实机路径）。

注意：
  * 两块 IMX219 必须都出现在 ``rpicam-hello --list-cameras`` 中，且 CSI 排线
    方向正确——"只能看到一个相机"九成是排线/overlay 问题，不是代码问题；
  * 每个相机独立 ``Picamera2(i)``，``buffer_count>=3`` 降低丢帧；
  * SensorTimestamp 与帧内容在同一 request 上取出，不能用"读图时刻"代替；
  * FrameDurationLimits 同时设置 min/max 才能锁定周期，单独设 min 无效。

本文件在没有 picamera2 的机器上无法导入执行（import 放在 start 内，模块本身
可被静态检查）。
"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from .base import CapturedPair


class CameraOpenError(RuntimeError):
    pass


class FrameSkewError(RuntimeError):
    pass


class Picamera2Backend:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_count: int = 3,
        left_index: int = 0,
        right_index: int = 1,
        max_skew_ms: Optional[float] = None,
        retries: int = 3,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_count = max(3, int(buffer_count))
        self.left_index = left_index
        self.right_index = right_index
        self.max_skew_ms = max_skew_ms if max_skew_ms is not None else 0.75 * 1000.0 / fps
        self.retries = int(retries)
        self._left: Any = None
        self._right: Any = None
        self._seq = 0
        self._frame_us = int(round(1e6 / fps))

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:  # pragma: no cover - 仅实机
            raise CameraOpenError(
                "缺少 picamera2：请用 Raspberry Pi OS 64-bit Bookworm 的系统 Python，"
                "并安装 python3-picamera2"
            ) from exc

        self._left = self._open(Picamera2, self.left_index, "left")
        self._right = self._open(Picamera2, self.right_index, "right")

    def _open(self, Picamera2, index: int, side: str):
        cam = Picamera2(index)
        controls = {
            # 锁定帧周期是同步的前提；曝光由 AE 在周期内调整
            "FrameDurationLimits": (self._frame_us, self._frame_us),
        }
        config = cam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"},
            buffer_count=self.buffer_count,
            controls=controls,
        )
        cam.configure(config)
        cam.start()
        # 等第一帧，避免 read 时 metadata 为空
        time.sleep(0.2)
        meta = cam.capture_metadata()
        if "SensorTimestamp" not in meta:
            raise CameraOpenError(f"{side} 相机 metadata 没有 SensorTimestamp")
        return cam

    # ------------------------------------------------------------------ #
    def read(self) -> CapturedPair:
        if self._left is None or self._right is None:
            self.start()
        last_skew = 0.0
        for _ in range(self.retries):
            pair = self._grab_once()
            last_skew = abs(pair.ts_diff_ns) / 1e6
            if last_skew <= self.max_skew_ms:
                pair.seq = self._seq
                self._seq += 1
                return pair
        raise FrameSkewError(
            f"左右帧时间戳差 {last_skew:.2f} ms 超过上限 {self.max_skew_ms:.2f} ms；"
            "请检查同步环、USB/CSI 带宽与散热降频"
        )

    def _grab_once(self) -> CapturedPair:
        assert self._left is not None and self._right is not None
        req_l = self._left.capture_request()
        try:
            left = req_l.make_array("main").copy()
            ts_l = int(req_l.get_metadata()["SensorTimestamp"])
        finally:
            req_l.release()

        req_r = self._right.capture_request()
        try:
            right = req_r.make_array("main").copy()
            ts_r = int(req_r.get_metadata()["SensorTimestamp"])
        finally:
            req_r.release()

        return CapturedPair(left=left, right=right, ts_left_ns=ts_l, ts_right_ns=ts_r, seq=-1)

    def set_slave_frame_duration(self, period_us: int) -> None:
        """同步环输出：只改从动相机（右目）的帧周期。"""
        if self._right is None:
            return
        base = 1e6 / float(self.fps)
        period_us = int(np.clip(period_us, int(0.8 * base), int(1.2 * base)))
        self._right.set_controls({"FrameDurationLimits": (period_us, period_us)})

    def close(self) -> None:
        for cam in (self._left, self._right):
            if cam is not None:
                try:
                    cam.stop()
                finally:
                    cam.close()
        self._left = self._right = None
