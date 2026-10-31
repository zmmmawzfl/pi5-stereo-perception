"""第一天开工清单：双相机 SensorTimestamp 差的基线（不推流、不加控制）。

实机:
    python -m pi.tools.timestamp_baseline --frames 100
    python -m pi.tools.timestamp_baseline --frames 100 --with-control

无相机时自动用合成后端（会明确打印 backend=synthetic，数字仅为链路自检）：
    python -m pi.tools.timestamp_baseline --frames 100 --backend synthetic

输出：中位数 / P25 / P75 / IQR / min / max，并写 baseline.json。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from pi.capture.factory import build_backend
from pi.sync.p_controller import PhaseLockController
from shared.config import load_config
from shared.recording import write_json
from shared.stats import summarize


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SensorTimestamp 差基线（第一天）")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--backend", default="auto", choices=["auto", "picamera2", "synthetic"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--kp", type=float, default=0.05)
    ap.add_argument("--with-control", action="store_true", help="加 P 控制环（默认关闭）")
    ap.add_argument("--out", default="runs/day1_baseline.json")
    args = ap.parse_args(argv)

    cfg = load_config(None, {"camera.backend": args.backend, "camera.fps": args.fps})
    backend = build_backend(cfg)
    backend.start()

    ctl = (
        PhaseLockController(kp=args.kp, base_period_us=1e6 / args.fps)
        if args.with_control
        else None
    )
    diffs_ms = []
    periods = []
    try:
        for i in range(args.frames):
            pair = backend.read()
            diffs_ms.append(pair.ts_diff_ns / 1e6)
            if ctl is not None:
                period = ctl.update(pair.ts_left_ns, pair.ts_right_ns)
                backend.set_slave_frame_duration(period)
                periods.append(period)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{args.frames}", flush=True)
    finally:
        backend.close()

    stats = summarize(diffs_ms)
    iqr = (stats["p75"] - stats["p25"]) if stats["n"] >= 2 else 0.0
    print("\n=== SensorTimestamp 差（右 - 左，ms）===")
    print(f"  n={stats['n']}  median={stats['median']:.4f}  P25={stats['p25']:.4f}  "
          f"P75={stats['p75']:.4f}  IQR={iqr:.4f}")
    print(f"  min={stats['min']:.4f}  max={stats['max']:.4f}  "
          f"mean={stats['mean']:.4f}  std={stats['std']:.4f}")
    if ctl is not None:
        print(f"  控制结束周期={ctl.period_us:.1f} us  (初始 {1e6 / args.fps:.1f} us)")
    if args.backend != "picamera2":
        print("\n注意：当前不是真实相机，数字只用于验证采集/配对/统计链路。")

    write_json(
        Path(args.out),
        {
            "backend": args.backend,
            "frames": args.frames,
            "with_control": bool(ctl),
            "kp": args.kp if ctl else None,
            "ts_diff_ms": stats,
            "iqr_ms": iqr,
            "periods_us": periods[-10:],
            "per_frame_ms": diffs_ms,
        },
    )
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
