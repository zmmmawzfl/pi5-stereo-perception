"""对一段录制数据做压缩伪影量化，输出 JSON + Markdown 表。

    python -m tools.compression_report --replay-root data/recordings/day1 \
        --calib data/calib/stereo_calib.npz --out runs/compression --max-frames 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from pc.calib.calibrate import CalibrationResult
from pc.validate.compression import benchmark_compression
from pc.validate.report import compression_block, write_markdown
from shared.config import load_config
from shared.recording import PairReplay


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="压缩伪影量化（JPEG / H.264 vs 无损）")
    ap.add_argument("--replay-root", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="runs/compression")
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    calib = CalibrationResult.load(args.calib)
    replay = PairReplay(args.replay_root)
    left, right = [], []
    for i, (rec, l, r) in enumerate(replay):
        if i >= args.max_frames:
            break
        left.append(l)
        right.append(r)
    if len(left) < 2:
        print("至少需要 2 对帧", file=sys.stderr)
        return 2

    result = benchmark_compression(
        left, right, calib, cfg.sgbm, fps=cfg.camera.fps,
        texture_threshold=cfg.sgbm.texture_threshold,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compression.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(
        out / "compression.md",
        "压缩伪影量化",
        [("结果", compression_block(result)),
         ("说明", "无损帧为参考；弱纹理区按 5x5 局部标准差低于阈值的像素统计；"
                  "物体质心偏移需要真值/掩码，本命令未提供掩码时为空。\n")],
    )
    print(f"输出: {out / 'compression.json'}  {out / 'compression.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
