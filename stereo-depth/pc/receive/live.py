"""PC 端在线处理：订阅 Pi 推流（或回放录制目录）→ 物体级 3D 输出。

    python -m pc.receive.live --endpoint tcp://192.168.1.10:5555 \
        --calib data/calib/stereo_calib.npz --out runs/live

    python -m pc.receive.live --replay-root data/recordings/day1 \
        --calib data/calib/stereo_calib.npz --out runs/replay
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from pc.calib.calibrate import CalibrationResult
from pc.pipeline import build_pipeline
from pc.receive.subscriber import ZmqPairSource
from pc.validate.visualize import draw_objects, source_map
from shared.config import load_config
from shared.recording import PairReplay
from shared.stats import summarize


def _iter_zmq(endpoint: str, max_pairs: Optional[int], idle: float) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
    src = ZmqPairSource(endpoint)
    try:
        for rp in src.pairs(max_pairs=max_pairs, idle_timeout_s=idle):
            yield rp.pair_id, rp.ts_diff_ns, rp.left, rp.right
    finally:
        print(json.dumps({"pairing": src.stats.as_dict()}, ensure_ascii=False))


def _iter_replay(root: str) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
    replay = PairReplay(root)
    for rec, left, right in replay:
        yield rec.pair_id, rec.ts_diff_ns, left, right


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PC 端在线/回放物体级 3D")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    ap.add_argument("--replay-root", default=None)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="runs/live")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--idle-timeout", type=float, default=15.0)
    ap.add_argument("--save-every", type=int, default=30, help="每 N 对存一次可视化，0 关闭")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    calib = CalibrationResult.load(args.calib)
    pipeline = build_pipeline(cfg, calib)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "objects.jsonl").write_text("", encoding="utf-8")

    source = _iter_replay(args.replay_root) if args.replay_root else _iter_zmq(
        args.endpoint, args.max_pairs, args.idle_timeout
    )
    frames = 0
    timings = []
    t_start = time.time()
    try:
        for pair_id, skew_ns, left, right in source:
            result = pipeline.process(left, right)
            record = {
                "pair_id": pair_id,
                "skew_ms": skew_ns / 1e6,
                "wall_ns": time.time_ns(),
                "objects": [o.to_dict() for o in result.objects],
                "timings_ms": {k: round(v, 3) for k, v in result.timings_ms.items()},
                "coverage": float(result.fused.coverage),
            }
            with open(out / "objects.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            timings.append(result.timings_ms["total"])
            frames += 1
            if args.save_every and frames % args.save_every == 0:
                cv2.imwrite(str(out / f"overlay_{frames:05d}.png"),
                            draw_objects(result.rectified_left, result.objects, calib))
                cv2.imwrite(str(out / f"source_{frames:05d}.png"), source_map(result.fused))
            print(
                f"[live] pair={pair_id} objects={len(result.objects)} "
                f"total={result.timings_ms['total']:.0f}ms",
                flush=True,
            )
            if args.max_pairs and frames >= args.max_pairs:
                break
    except KeyboardInterrupt:
        print("\n[live] 中断", flush=True)

    elapsed = max(time.time() - t_start, 1e-6)
    summary = {
        "pairs": frames,
        "elapsed_s": elapsed,
        "fps": frames / elapsed,
        "timings_ms": summarize(timings),
        "calib": calib.summary(),
        "detector": pipeline.detector.name,
        "learned": pipeline.learned.name if pipeline.learned is not None else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if frames else 2


if __name__ == "__main__":
    sys.exit(main())
