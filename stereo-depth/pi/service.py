"""Pi 端常驻服务：采集 → 同步环 → 推流/落盘。"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from pi.capture.base import CapturedPair
from pi.capture.factory import build_backend
from pi.stream.publisher import build_publisher
from pi.sync.p_controller import PhaseLockController
from shared.config import AppConfig, config_to_dict, load_config
from shared.recording import now_tag, write_json
from shared.stats import summarize


class PiService:
    def __init__(self, cfg: AppConfig, record_root: Optional[str] = None, replay_root: Optional[str] = None):
        self.cfg = cfg
        self.backend = build_backend(cfg, replay_root)
        self.publisher = build_publisher(cfg.stream, record_root)
        self.ctl = (
            PhaseLockController(
                kp=cfg.sync.kp,
                target_offset_ns=cfg.sync.target_offset_ns,
                min_frame_us=cfg.sync.min_frame_us,
                max_frame_us=cfg.sync.max_frame_us,
                max_step_us=cfg.sync.max_step_us,
                base_period_us=1e6 / float(cfg.camera.fps),
            )
            if cfg.sync.enabled
            else None
        )
        self.stats = {"frames": 0, "pairs": 0, "skew_ms": [], "periods_us": []}
        self._stop = False

    # ------------------------------------------------------------------ #
    def run(self, max_frames: Optional[int] = None, duration_s: Optional[float] = None) -> dict:
        self.backend.start()
        t_start = time.time()
        last_report = t_start
        try:
            while not self._stop:
                pair = self.backend.read()
                if pair is None:
                    break
                self.publisher.publish(pair)
                self.stats["pairs"] += 1
                self.stats["frames"] += 2
                self.stats["skew_ms"].append(pair.ts_diff_ns / 1e6)

                if self.ctl is not None:
                    period = self.ctl.update(pair.ts_left_ns, pair.ts_right_ns)
                    self.backend.set_slave_frame_duration(period)
                    self.stats["periods_us"].append(period)

                if max_frames is not None and self.stats["pairs"] >= max_frames:
                    break
                now = time.time()
                if duration_s is not None and now - t_start >= duration_s:
                    break
                if now - last_report >= 5.0:
                    last_report = now
                    elapsed = now - t_start
                    print(
                        f"[pi] {self.stats['pairs']} 对 / {elapsed:.1f}s "
                        f"({self.stats['pairs'] / max(elapsed, 1e-6):.1f} fps) "
                        f"skew_med={summarize(self.stats['skew_ms'])['median']:.3f}ms",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\n[pi] 收到中断，正在收尾", flush=True)
        finally:
            self.backend.close()
            self.publisher.close()

        report = {
            "pairs": self.stats["pairs"],
            "frames": self.stats["frames"],
            "skew_ms": summarize(self.stats["skew_ms"]),
            "period_us": summarize(self.stats["periods_us"]),
            "publisher": self.publisher.stats.as_dict() if hasattr(self.publisher, "stats") else {},
        }
        return report


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Pi 5 双目采集 + 同步 + 推流服务")
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", default=None, choices=["auto", "picamera2", "synthetic", "replay"])
    ap.add_argument("--replay-root", default=None)
    ap.add_argument("--record-root", default=None, help="无损落盘目录（PNG）")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--duration", type=float, default=None, help="运行秒数")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--out", default=None, help="服务统计 JSON 输出")
    args = ap.parse_args(argv)

    overrides = {}
    if args.backend:
        overrides["camera.backend"] = args.backend
    if args.no_sync:
        overrides["sync.enabled"] = False
    if args.no_stream:
        overrides["stream.transport"] = "none"
    cfg = load_config(args.config, overrides)

    svc = PiService(cfg, record_root=args.record_root, replay_root=args.replay_root)
    report = svc.run(max_frames=args.frames, duration_s=args.duration)
    print(report)

    if args.out:
        write_json(args.out, {"report": report, "config": config_to_dict(cfg)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
