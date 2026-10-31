"""标定命令行：实拍图像 / 合成数据 / 查看标定结果。

实拍（左右目录）:
    python -m pc.calib.cli calibrate --left data/calib/left --right data/calib/right \
        --pattern 9x6 --square-mm 30 --out data/calib/stereo_calib.npz
实拍（直接用 Pi 录制目录，自动跳过没有棋盘的帧）:
    python -m pc.calib.cli calibrate --replay-root data/recordings/calib_board \
        --pattern 9x6 --square-mm 30 --out data/calib/stereo_calib.npz
合成（无相机时验证流程与精度）:
    python -m pc.calib.cli synthetic --out data/calib/stereo_calib.npz --views 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from pc.calib.calibrate import CalibrationResult, calibrate_stereo
from shared.recording import PairReplay
from shared.synthetic import generate_calibration_views


def _load_images(folder: str) -> List[np.ndarray]:
    p = Path(folder)
    files = sorted([f for f in p.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")])
    if not files:
        raise FileNotFoundError(f"{folder} 下没有图像")
    imgs = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"读取失败: {f}")
        imgs.append(img)
    return imgs


def _load_replay(root: str, max_frames: int | None = None):
    """从标准录制目录（pairs.jsonl + frames/）读取左右图像。"""
    left, right = [], []
    for i, (_rec, l, r) in enumerate(PairReplay(root)):
        if max_frames and i >= max_frames:
            break
        left.append(l)
        right.append(r)
    return left, right


def cmd_calibrate(args) -> int:
    if args.replay_root:
        left, right = _load_replay(args.replay_root, args.max_frames)
        print(f"从录制目录读取 {len(left)} 对帧: {args.replay_root}")
    else:
        if not args.left or not args.right:
            print("必须提供 --left/--right，或用 --replay-root 指向录制目录", file=sys.stderr)
            return 2
        left = _load_images(args.left)
        right = _load_images(args.right)
    if len(left) != len(right):
        print(f"左右图像数量不一致: {len(left)} vs {len(right)}", file=sys.stderr)
        return 2
    if not left:
        print("没有可用图像", file=sys.stderr)
        return 2
    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    calib = calibrate_stereo(
        left, right, pattern=(cols, rows), square_m=args.square_mm / 1000.0,
        min_views=args.min_views, progress=True,
        fix_k3=not args.free_distortion, zero_tangent=not args.free_distortion,
    )
    calib.meta["source"] = "real-images"
    calib.save(args.out)
    print(f"标定完成: {args.out}")
    print(f"  baseline={calib.baseline_m * 1000:.3f} mm  f={calib.focal_px:.3f} px  "
          f"stereo RMS={calib.rms_stereo:.4f} px")
    return 0


def cmd_synthetic(args) -> int:
    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    views = generate_calibration_views(
        n_views=args.views, seed=args.seed, image_size=(args.width, args.height),
        board_cols=cols, board_rows=rows, square_m=args.square_mm / 1000.0,
        baseline=args.baseline_mm / 1000.0,
    )
    out_dir = Path(args.out).parent
    if args.save_images:
        (out_dir / "left").mkdir(parents=True, exist_ok=True)
        (out_dir / "right").mkdir(parents=True, exist_ok=True)
        for i, (l, r) in enumerate(zip(views["left"], views["right"])):
            cv2.imwrite(str(out_dir / "left" / f"left_{i:03d}.png"), l)
            cv2.imwrite(str(out_dir / "right" / f"right_{i:03d}.png"), r)
    calib = calibrate_stereo(
        views["left"], views["right"], pattern=(cols, rows),
        square_m=args.square_mm / 1000.0, min_views=min(8, args.views), progress=True,
        fix_k3=not args.free_distortion, zero_tangent=not args.free_distortion,
    )
    calib.meta["source"] = "synthetic"
    calib.meta["truth"] = {
        "fx": 530.0, "baseline_m": args.baseline_mm / 1000.0,
        "distortion": [-0.06, 0.01, 0.0, 0.0, 0.0],
    }
    calib.save(args.out)
    truth_b = args.baseline_mm / 1000.0
    print(f"标定完成: {args.out}")
    print(f"  baseline={calib.baseline_m * 1000:.3f} mm (真值 {truth_b * 1000:.1f}) "
          f"误差 {(calib.baseline_m - truth_b) / truth_b * 100:+.3f}%")
    print(f"  f={calib.focal_px:.3f} px  stereo RMS={calib.rms_stereo:.4f} px  "
          f"epi |dy| med={calib.meta.get('epipolar_dy', {}).get('median_abs_px')}")
    return 0


def cmd_inspect(args) -> int:
    c = CalibrationResult.load(args.calib)
    print(f"image_size={c.image_size} pattern={c.pattern} square={c.square_m * 1000:.1f}mm")
    print(f"B={c.baseline_m * 1000:.4f} mm  f={c.focal_px:.4f} px  cx={c.cx:.3f} cy={c.cy:.3f}")
    print(f"RMS stereo={c.rms_stereo:.4f} mono=({c.rms_mono_left:.4f},{c.rms_mono_right:.4f})")
    print(f"D1={c.D1.ravel().round(5).tolist()}")
    print(f"T={c.T.ravel().round(6).tolist()}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="双目相机标定")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("calibrate", help="用实拍左右图像标定")
    p.add_argument("--left", default=None, help="左图目录")
    p.add_argument("--right", default=None, help="右图目录")
    p.add_argument("--replay-root", default=None,
                   help="标准录制目录（pairs.jsonl + frames/），与 --left/--right 二选一")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--pattern", default="9x6", help="内角点数，如 9x6")
    p.add_argument("--square-mm", type=float, default=30.0)
    p.add_argument("--min-views", type=int, default=10)
    p.add_argument("--free-distortion", action="store_true",
                   help="放开 k3/切向畸变（默认固定 k3、切向置零，更稳）")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("synthetic", help="用合成棋盘格标定（无相机验证）")
    p.add_argument("--out", required=True)
    p.add_argument("--views", type=int, default=15)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--pattern", default="9x6")
    p.add_argument("--square-mm", type=float, default=30.0)
    p.add_argument("--baseline-mm", type=float, default=83.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--free-distortion", action="store_true")
    p.set_defaults(func=cmd_synthetic)

    p = sub.add_parser("inspect", help="查看标定文件")
    p.add_argument("--calib", required=True)
    p.set_defaults(func=cmd_inspect)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
