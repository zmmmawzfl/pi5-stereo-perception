"""运动边界 / 同步残差实测：同一目标在静止与慢速运动下的深度抖动。

用法（先录两段：静止、匀速慢移）::

    python -m tools.motion_jitter --replay-root data/recordings/static \
        --calib data/calib/stereo_calib.npz --out runs/jitter_static.json
    python -m tools.motion_jitter --replay-root data/recordings/moving \
        --calib data/calib/stereo_calib.npz --out runs/jitter_moving.json \
        --baseline-json runs/jitter_static.json

输出：中位深度、深度标准差、MAD、帧间差、相对抖动，以及与静止基线的增量。
这个增量才是"卷帘快门 + 曝光不同步 + 匹配噪声"对几何的真实作用量。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pc.calib.calibrate import CalibrationResult
from pc.pipeline import build_pipeline
from shared.config import load_config
from shared.recording import PairReplay
from shared.stats import summarize


def _parse_roi(text: Optional[str], shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    if not text:
        return None
    vals = [int(v) for v in text.split(",")]
    if len(vals) != 4:
        raise ValueError("--roi 需要 x1,y1,x2,y2")
    h, w = shape
    x1, y1, x2, y2 = vals
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)


def _auto_roi(result) -> Tuple[int, int, int, int]:
    h, w = result.fused.disp.shape
    if result.objects:
        best = max(result.objects, key=lambda o: (o.bbox_2d[2] - o.bbox_2d[0]) * (o.bbox_2d[3] - o.bbox_2d[1]))
        x1, y1, x2, y2 = (int(round(v)) for v in best.bbox_2d)
        return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
    return int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)


def _series_stats(depths: Sequence[float], disps: Sequence[float], skews: Sequence[float]) -> Dict[str, object]:
    arr = np.asarray(depths, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    mad = float(np.median(np.abs(arr - np.median(arr))))
    diffs = np.abs(np.diff(arr))
    return {
        "n": int(arr.size),
        "depth_median_m": float(np.median(arr)),
        "depth_mean_m": float(arr.mean()),
        "depth_std_mm": float(arr.std(ddof=1) * 1000) if arr.size > 1 else 0.0,
        "depth_mad_mm": mad * 1000,
        "depth_p5_p95_mm": [float(np.percentile(arr, 5) * 1000), float(np.percentile(arr, 95) * 1000)],
        "frame_to_frame_median_abs_mm": float(np.median(diffs) * 1000) if diffs.size else 0.0,
        "relative_jitter_pct": float(arr.std(ddof=1) / arr.mean() * 100) if arr.size > 1 else 0.0,
        "disparity_std_px": float(np.std(disps, ddof=1)) if len(disps) > 1 else 0.0,
        "skew_ms": summarize(skews),
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="物体级深度抖动 / 运动边界")
    ap.add_argument("--replay-root", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="runs/jitter.json")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--roi", default=None, help="固定 ROI: x1,y1,x2,y2")
    ap.add_argument("--baseline-json", default=None, help="静止基线结果，用于算运动带来的增量")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    calib = CalibrationResult.load(args.calib)
    pipeline = build_pipeline(cfg, calib)
    replay = PairReplay(args.replay_root)

    depths: List[float] = []
    disps: List[float] = []
    skews: List[float] = []
    per_frame: List[Dict[str, object]] = []
    roi_fixed = None

    for i, (rec, left, right) in enumerate(replay):
        if i >= args.max_frames:
            break
        result = pipeline.process(left, right)
        if roi_fixed is None:
            roi_fixed = _parse_roi(args.roi, result.fused.disp.shape) or _auto_roi(result)
            print(f"[jitter] ROI={roi_fixed}")
        x1, y1, x2, y2 = roi_fixed
        d_roi = result.fused.disp[y1 : y2 + 1, x1 : x2 + 1]
        v_roi = result.fused.valid[y1 : y2 + 1, x1 : x2 + 1]
        z_roi = result.depth[y1 : y2 + 1, x1 : x2 + 1]
        if int(v_roi.sum()) < 20:
            continue
        d_med = float(np.median(d_roi[v_roi]))
        z_med = float(np.median(z_roi[np.isfinite(z_roi)]))
        depths.append(z_med)
        disps.append(d_med)
        skews.append(rec.ts_diff_ns / 1e6)
        per_frame.append({"pair_id": rec.pair_id, "depth_m": z_med, "disp_px": d_med,
                          "valid_ratio": float(v_roi.mean()), "skew_ms": rec.ts_diff_ns / 1e6})

    stats = _series_stats(depths, disps, skews)
    payload: Dict[str, object] = {
        "replay_root": str(Path(args.replay_root)),
        "roi_xyxy": roi_fixed,
        "stats": stats,
        "per_frame": per_frame,
    }

    if args.baseline_json and Path(args.baseline_json).exists():
        base = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        base_std = float(base.get("stats", {}).get("depth_std_mm") or 0.0)
        cur_std = float(stats.get("depth_std_mm") or 0.0)
        payload["baseline"] = {
            "path": args.baseline_json,
            "static_depth_std_mm": base_std,
            "motion_depth_std_mm": cur_std,
            "delta_std_mm": cur_std - base_std,
            "ratio": (cur_std / base_std) if base_std > 0 else None,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if "baseline" in payload:
        b = payload["baseline"]
        print(f"\n相对静止基线：抖动 {b['static_depth_std_mm']:.2f} -> {b['motion_depth_std_mm']:.2f} mm "
              f"（增量 {b['delta_std_mm']:+.2f} mm）")
    print(f"已写入 {out}")
    return 0 if stats.get("n") else 2


if __name__ == "__main__":
    sys.exit(main())
