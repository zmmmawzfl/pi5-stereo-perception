"""压缩伪影量化：无损 vs JPEG vs H.264，对 SGBM 结果的影响。

验收口径（阶段一）：
  * 有效视差像素数变化；
  * 弱纹理区视差误差；
  * 物体质心深度偏移（有真值掩码时）；
  * 深度偏移 < 基线 1~2% 才认为压缩流可用于测距。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pc.calib.calibrate import CalibrationResult
from pc.fuse.router import FuseRouter
from pc.stereo.sgbm import SGBMMatcher
from shared.config import SgbmConfig
from shared.stats import abs_error_stats, summarize


@dataclass
class CompressionSetting:
    name: str
    kind: str  # "raw" | "jpeg" | "h264"
    quality: int = 90
    bitrate_kbps: int = 2500


def jpeg_roundtrip(frames: Sequence[np.ndarray], quality: int) -> Tuple[List[np.ndarray], int]:
    out: List[np.ndarray] = []
    total = 0
    for img in frames:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        total += len(buf)
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        out.append(dec)
    return out, total


def h264_roundtrip(
    frames: Sequence[np.ndarray],
    fps: int,
    bitrate_kbps: int,
    keyint: int = 30,
) -> Tuple[List[np.ndarray], int]:
    """用 ffmpeg libx264 编码再解码，返回帧与码流字节数。"""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未找到 ffmpeg，无法做 H.264 量化（apt install ffmpeg）")
    if len(frames) < 2:
        raise ValueError("H.264 对照至少需要 2 帧")
    h, w = frames[0].shape[:2]
    with tempfile.TemporaryDirectory(prefix="sd_h264_") as tmp:
        tmp_path = Path(tmp)
        in_raw = tmp_path / "in.raw"
        mp4 = tmp_path / "out.mp4"
        out_raw = tmp_path / "out.raw"
        with open(in_raw, "wb") as fh:
            for img in frames:
                fh.write(np.ascontiguousarray(img).tobytes())
        cmd_enc = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
            "-r", str(fps), "-i", str(in_raw),
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
            "-b:v", f"{int(bitrate_kbps)}k", "-g", str(keyint), "-bf", "0",
            "-pix_fmt", "yuv420p", str(mp4),
        ]
        subprocess.run(cmd_enc, check=True)
        cmd_dec = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
            "-f", "rawvideo", "-pix_fmt", "bgr24", str(out_raw),
        ]
        subprocess.run(cmd_dec, check=True)
        arr = np.fromfile(out_raw, dtype=np.uint8)
        n = len(frames)
        if arr.size != n * h * w * 3:
            raise RuntimeError(f"解码帧数不符: {arr.size} != {n * h * w * 3}")
        decoded = [arr[i * h * w * 3 : (i + 1) * h * w * 3].reshape(h, w, 3) for i in range(n)]
        size = mp4.stat().st_size
    return decoded, size


def _stack_frames(frames: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack(frames, axis=0)


def benchmark_compression(
    left_frames: Sequence[np.ndarray],
    right_frames: Sequence[np.ndarray],
    calib: CalibrationResult,
    sgbm_cfg: SgbmConfig,
    fps: int = 30,
    object_masks: Optional[Sequence[Dict[str, Any]]] = None,
    settings: Optional[Sequence[CompressionSetting]] = None,
    texture_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    if len(left_frames) != len(right_frames) or len(left_frames) < 2:
        raise ValueError("左右帧数量必须一致且不少于 2")
    settings = list(
        settings
        or [
            CompressionSetting("raw", "raw"),
            CompressionSetting("jpeg_q95", "jpeg", quality=95),
            CompressionSetting("jpeg_q80", "jpeg", quality=80),
            CompressionSetting("h264_2500k", "h264", bitrate_kbps=2500),
            CompressionSetting("h264_800k", "h264", bitrate_kbps=800),
        ]
    )
    thr = float(texture_threshold if texture_threshold is not None else sgbm_cfg.texture_threshold)
    router = FuseRouter(lr_max_diff=sgbm_cfg.lr_max_diff, texture_threshold=thr)
    matcher = SGBMMatcher(sgbm_cfg)
    n = len(left_frames)

    # 无损参考
    ref_fused = []
    ref_sgbm = []
    ref_texture = []
    for i in range(n):
        res = matcher.compute(left_frames[i], right_frames[i])
        ref_sgbm.append(res)
        fused = router.fuse(res, None, left_frames[i])
        ref_fused.append(fused)
        gray = cv2.cvtColor(left_frames[i], cv2.COLOR_BGR2GRAY)
        ref_texture.append(router.texture_score(gray))
    ref_valid_px = float(np.mean([f.valid.sum() for f in ref_fused]))
    ref_coverage = float(np.mean([f.coverage for f in ref_fused]))

    # 每个物体的参考深度中位数（用于把"深度偏移"换算成相对误差）
    ref_object_depth: Dict[str, float] = {}
    if object_masks:
        for m in object_masks:
            mask = np.asarray(m["mask"], bool)
            zs = []
            for f in ref_fused:
                sel = mask & f.valid
                if int(sel.sum()) >= 20:
                    zs.append(float(np.median(calib.focal_px * calib.baseline_m / f.disp[sel])))
            if zs:
                ref_object_depth[str(m["name"])] = float(np.median(zs))

    results: List[Dict[str, Any]] = []
    for setting in settings:
        if setting.kind == "raw":
            left_dec, right_dec = list(left_frames), list(right_frames)
            bytes_total = sum(int(img.nbytes) for img in left_frames) + sum(
                int(img.nbytes) for img in right_frames
            )
        elif setting.kind == "jpeg":
            left_dec, sz_l = jpeg_roundtrip(left_frames, setting.quality)
            right_dec, sz_r = jpeg_roundtrip(right_frames, setting.quality)
            bytes_total = sz_l + sz_r
        elif setting.kind == "h264":
            left_dec, sz_l = h264_roundtrip(left_frames, fps, setting.bitrate_kbps)
            right_dec, sz_r = h264_roundtrip(right_frames, fps, setting.bitrate_kbps)
            bytes_total = sz_l + sz_r
        else:
            raise ValueError(f"未知压缩类型: {setting.kind}")

        disp_errs: List[np.ndarray] = []
        weak_errs: List[np.ndarray] = []
        valid_counts: List[int] = []
        object_shifts: Dict[str, List[float]] = {}
        if object_masks:
            for m in object_masks:
                object_shifts[str(m["name"])] = []

        for i in range(n):
            res = matcher.compute(left_dec[i], right_dec[i])
            fused = router.fuse(res, None, left_dec[i])
            valid_counts.append(int(fused.valid.sum()))
            # 视差误差用原始 SGBM 有效区（不经纹理门限），否则弱纹理区永远没有样本
            common = res.valid & ref_sgbm[i].valid
            if np.any(common):
                err = np.abs(res.disp[common] - ref_sgbm[i].disp[common])
                disp_errs.append(err)
                weak = common & (ref_texture[i] < thr)
                if np.any(weak):
                    weak_errs.append(np.abs(res.disp[weak] - ref_sgbm[i].disp[weak]))
            if object_masks:
                for m in object_masks:
                    mask = np.asarray(m["mask"], bool)
                    both = mask & fused.valid & ref_fused[i].valid
                    if int(both.sum()) >= 20:
                        z1 = calib.focal_px * calib.baseline_m / fused.disp[both]
                        z2 = calib.focal_px * calib.baseline_m / ref_fused[i].disp[both]
                        object_shifts[str(m["name"])].append(float(np.median(z1) - np.median(z2)))

        all_err = np.concatenate(disp_errs) if disp_errs else np.array([])
        all_weak = np.concatenate(weak_errs) if weak_errs else np.array([])
        entry: Dict[str, Any] = {
            "name": setting.name,
            "kind": setting.kind,
            "param": setting.quality if setting.kind == "jpeg" else setting.bitrate_kbps,
            "frames": n,
            "bytes_per_pair": int(bytes_total / n),
            "bitrate_mbps": float(bytes_total * 8 / n * fps / 1e6),
            "valid_px_mean": float(np.mean(valid_counts)),
            "valid_ratio_vs_ref": float(np.mean(valid_counts) / max(ref_valid_px, 1e-6)),
            "coverage_mean": float(np.mean(valid_counts) / ref_fused[0].disp.size),
            "disp_err": abs_error_stats(all_err.tolist()) if all_err.size else {"n": 0},
            "weak_texture_err": abs_error_stats(all_weak.tolist()) if all_weak.size else {"n": 0},
            "object_depth_shift_m": {
                k: (summarize(v) if v else {"n": 0}) for k, v in object_shifts.items()
            },
            "object_depth_shift_rel": {
                k: (
                    abs(summarize(v)["median"]) / max(ref_object_depth.get(k, 1e-9), 1e-9)
                    if v
                    else None
                )
                for k, v in object_shifts.items()
            },
            "ref_object_depth_m": ref_object_depth,
        }
        results.append(entry)
        print(
            f"[compression] {setting.name}: valid={entry['valid_ratio_vs_ref']:.3f} "
            f"disp_med={entry['disp_err'].get('median')} px "
            f"({entry['bitrate_mbps']:.2f} Mbps)"
        )

    return {
        "reference": {
            "valid_px_mean": ref_valid_px,
            "coverage_mean": ref_coverage,
            "texture_threshold": thr,
        },
        "settings": results,
        "frames": n,
        "fps": fps,
        "object_masks": [str(m["name"]) for m in object_masks] if object_masks else [],
    }


def write_compression_json(path: str | Path, result: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
