"""无相机端到端演示：合成场景 → 标定 → 校正 → SGBM → 融合 → 检测 → 物体 3D → 报告。

它验证的是"接口、符号、单位、误差量级"是否自洽，不代表实机精度。
报告会显式标注运行硬件（本机可能是 Pi 5，不是目标 PC）与检测器类型。

    python -m tools.run_demo --out runs/demo --frames 8
    python -m tools.run_demo --out runs/demo --frames 8 --compression
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from pc.calib.calibrate import CalibrationResult, calibrate_stereo
from pc.depth.pointcloud import colored_pointcloud, depth_visualization, voxel_downsample, write_ply
from pc.fuse.router import SOURCE_LEARNED, SOURCE_SGBM
from pc.pipeline import StereoPipeline
from pc.validate import report as rpt
from pc.validate.compression import benchmark_compression, write_compression_json
from pc.validate.visualize import draw_objects, source_map
from pc.validate.metrics import (
    coverage_report,
    object_error_rows,
    summarize_rows,
)
from pi.sync.p_controller import simulate_sync
from shared.config import config_to_dict, load_config
from shared.stats import fmt, markdown_table, summarize
from shared.synthetic import SceneConfig, SyntheticScene, default_objects, generate_calibration_views


def build_calibration(path: Path, width: int, height: int, baseline: float) -> CalibrationResult:
    if path.exists():
        c = CalibrationResult.load(path)
        print(f"[demo] 复用标定 {path}  B={c.baseline_m * 1000:.2f}mm f={c.focal_px:.2f}px")
        return c
    print("[demo] 生成合成棋盘格并标定（无相机路径）...")
    fx = 530.0 * width / 640.0
    K = np.array([[fx, 0, (width - 1) / 2], [0, fx, (height - 1) / 2], [0, 0, 1]], float)
    views = generate_calibration_views(
        n_views=15,
        seed=11,
        image_size=(width, height),
        K_left=K,
        K_right=K,
        D_left=np.zeros(5),
        D_right=np.zeros(5),
        baseline=baseline,
    )
    calib = calibrate_stereo(
        views["left"], views["right"], pattern=(9, 6), square_m=0.03, min_views=10
    )
    calib.meta["source"] = "synthetic-pinhole"
    calib.save(path)
    print(
        f"[demo] 标定完成 B={calib.baseline_m * 1000:.3f}mm "
        f"f={calib.focal_px:.3f}px RMS={calib.rms_stereo:.4f}px"
    )
    return calib


def make_scene(width: int, height: int, baseline: float) -> SyntheticScene:
    fx = 530.0 * width / 640.0
    cfg = SceneConfig(
        width=width,
        height=height,
        fx=fx,
        fy=fx,
        camera_height=0.25,
        baseline=baseline,
    )
    return SyntheticScene(objects=default_objects(), config=cfg)


def save_source_map(fused, path: Path) -> None:
    cv2.imwrite(str(path), source_map(fused))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="无相机端到端演示 + 报告")
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--config", default=None)
    ap.add_argument("--calib", default=None, help="标定文件；缺省在 out/calib 下生成")
    ap.add_argument("--compression", action="store_true", help="运行压缩伪影量化（较慢）")
    ap.add_argument("--compression-frames", type=int, default=16)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    width, height = cfg.camera.width, cfg.camera.height
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    calib_path = Path(args.calib) if args.calib else out / "calib" / "stereo_calib.npz"
    calib = build_calibration(calib_path, width, height, 0.083)

    scene = make_scene(width, height, calib.baseline_m)
    pipeline = StereoPipeline(cfg, calib)

    rows_c: List[Dict[str, Any]] = []
    rows_u: List[Dict[str, Any]] = []
    timings: List[Dict[str, float]] = []
    coverages_all: List[float] = []
    coverages_gt: List[float] = []
    obj_coverages: List[float] = []
    best = {"n": -1, "result": None, "render": None}

    print(f"[demo] 处理 {args.frames} 帧（{width}×{height}，backend=synthetic）")
    for i in range(args.frames):
        render = scene.render(seed=100 + i)
        t0 = time.perf_counter()
        result = pipeline.process(render.left, render.right)
        timings.append(dict(result.timings_ms))
        cov = coverage_report(result.fused, render.valid)
        coverages_all.append(float(cov.coverage))
        coverages_gt.append(float(cov.coverage_on_reference_valid or 0.0))
        obj_mask = render.instances >= 0
        if np.any(obj_mask):
            obj_coverages.append(float((result.fused.valid & obj_mask).sum() / obj_mask.sum()))
        rows_c.extend(object_error_rows(result.objects, render.objects, use_constraint=True))
        rows_u.extend(object_error_rows(result.objects, render.objects, use_constraint=False))
        if len(result.objects) > best["n"]:
            best = {"n": len(result.objects), "result": result, "render": render}
        print(
            f"  frame {i}: objects={len(result.objects)} "
            f"coverage={cov.coverage:.3f} total={result.timings_ms['total']:.0f}ms",
            flush=True,
        )

    result = best["result"]
    render = best["render"]
    assert result is not None and render is not None

    # ---------------- 产物 ----------------
    cv2.imwrite(str(out / "overlay.png"), draw_objects(result.rectified_left, result.objects, calib))
    cv2.imwrite(str(out / "depth.png"), depth_visualization(result.depth, max_depth_m=3.0))
    disp_max = float(np.percentile(result.fused.disp[result.fused.valid], 95)) if np.any(result.fused.valid) else 1.0
    disp_vis = np.clip(result.fused.disp / max(disp_max, 1e-6) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / "disparity.png"), cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET))
    save_source_map(result.fused, out / "disparity_source.png")

    points_raw, colors_raw = colored_pointcloud(
        render.left, result.fused, calib,
        stride=cfg.pointcloud.stride, max_depth_m=cfg.pointcloud.max_depth_m,
    )
    points, colors = voxel_downsample(points_raw, colors_raw, cfg.pointcloud.voxel_size_m)
    write_ply(out / "cloud.ply", points, colors)
    (out / "objects.json").write_text(
        json.dumps(result.object_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- 汇总 ----------------
    stage_stats = {
        k: summarize([t.get(k, 0.0) for t in timings]) for k in timings[0].keys()
    }
    fps = 1000.0 / stage_stats["total"]["median"] if stage_stats["total"]["median"] else 0.0
    agg_c = summarize_rows(rows_c, use_constraint=True)
    agg_u = summarize_rows(rows_u, use_constraint=False)
    bands = rpt.distance_band_table(rows_c, edges=[0.0, 0.8, 1.2, 2.0])

    sync = simulate_sync(n_frames=300, kp=cfg.sync.kp)
    sync_stats = {
        "before": sync.stats["before"],
        "after": sync.stats["after"],
        "before_iqr_ms": sync.stats["before_iqr_ms"],
        "after_iqr_ms": sync.stats["after_iqr_ms"],
    }

    compression = None
    if args.compression:
        print(f"[demo] 压缩伪影量化：{args.compression_frames} 帧 × 5 组设置 ...")
        comp_scene = make_scene(width, height, calib.baseline_m)
        left_frames, right_frames = [], []
        masks = None
        for i in range(args.compression_frames):
            r = comp_scene.render(seed=300 + i)
            left_frames.append(r.left)
            right_frames.append(r.right)
            if masks is None:
                masks = [
                    {"name": o["name"], "mask": (r.instances == o["index"])}
                    for o in r.objects
                ]
        compression = benchmark_compression(
            left_frames,
            right_frames,
            calib,
            cfg.sgbm,
            fps=cfg.camera.fps,
            object_masks=masks,
            texture_threshold=cfg.sgbm.texture_threshold,
        )
        write_compression_json(out / "compression.json", compression)

    # ---------------- 报告 ----------------
    per_object = rpt.aggregate_by_name(rows_c)
    per_object_table = markdown_table(
        per_object,
        ["物体", "帧数", "GT 深度 (m)", "|ΔZ| 中位数 (mm)", "|ΔZ| P90 (mm)",
         "相对深度误差中位数", "3D 误差中位数 (mm)"],
    )
    timing_rows = [
        {"阶段": k, "中位数 (ms)": fmt(v["median"], 1), "P90 (ms)": fmt(v.get("p90"), 1)}
        for k, v in stage_stats.items()
    ]
    timing_table = markdown_table(timing_rows, ["阶段", "中位数 (ms)", "P90 (ms)"])

    learned_state = (
        f"已配置 {cfg.sgbm.learned_backend}，但本机无模型/GPU，未产生任何学习型实测数字。"
        if cfg.sgbm.learned_backend != "none"
        else "未启用（learned_backend=none）。阶段三的模型对照尚未执行，报告不包含任何伪造的学习型数据。"
    )
    coverage_note = (
        f"有效视差覆盖率（全图）{np.mean(coverages_all) * 100:.1f}%；"
        f"真值几何区域覆盖率 {np.mean(coverages_gt) * 100:.1f}%（按 GT 有效像素计）；"
        f"物体像素覆盖率 {np.mean(obj_coverages) * 100:.1f}%。\n\n"
        "> 合成场景的背景是无限远墙面，没有几何真值；全图覆盖率因此天然偏低，"
        "验收看「真值几何区域」与「物体像素」两列。"
    )
    summary_lines = [
        f"- 标定：B={calib.baseline_m * 1000:.3f} mm，f={calib.focal_px:.3f} px，"
        f"双目 RMS={calib.rms_stereo:.4f} px（合成针孔数据）。",
        f"- 物体级深度误差：|ΔZ| 中位数 {agg_c.get('depth_abs', {}).get('median', float('nan')) * 1000:.1f} mm，"
        f"相对误差中位数 {agg_c.get('depth_rel', {}).get('median', float('nan')) * 100:.2f}%（{agg_c.get('n', 0)} 个样本）。",
        f"- 处理帧率：{fps:.2f} fps（本机 {stage_stats['total']['median']:.0f} ms/帧）。",
        f"- 硬件：{rpt.environment()['platform']}。**这不是目标 PC 的帧率**，仅验证正确性。",
        f"- 检测器：{result.meta.get('detector')}（YOLO 权重未随仓库提供时使用彩色连通域占位）。",
        f"- 学习型匹配器：{learned_state}",
    ]
    if compression:
        rels = [
            v
            for s in compression["settings"]
            for v in (s.get("object_depth_shift_rel") or {}).values()
            if v is not None
        ]
        summary_lines.append(
            f"- 压缩伪影：最大物体相对深度偏移 {max(rels) * 100:.2f}%（验收阈值 2%）。"
            if rels
            else "- 压缩伪影：已量化，但本场景无物体掩码可算质心偏移。"
        )

    sections = [
        ("1. 结论摘要", "\n".join(summary_lines) + "\n"),
        ("2. 标定与极线校正", rpt.calibration_block(calib)),
        ("3. 时间戳同步基线", rpt.synchronization_block(sync_stats)),
        ("4. 物体级深度误差", per_object_table + "\n" + bands),
        ("5. 地面约束前后对比", rpt.comparison_table(rows_u, rows_c)),
        ("6. 覆盖率与处理帧率", coverage_note + "\n\n" + timing_table),
        (
            "7. 点云与体素降采样",
            f"- 像素步长 stride={cfg.pointcloud.stride}，最大深度 {cfg.pointcloud.max_depth_m:.1f} m。\n"
            f"- 降采样前 {len(points_raw):,} 点 → 体素 {cfg.pointcloud.voxel_size_m:.3f} m 后 "
            f"{len(points):,} 点（保留 {len(points) / max(len(points_raw), 1) * 100:.1f}%）。\n"
            f"- 点云坐标：X 右 / Y 下 / Z 前，单位米；PLY 与 ROS 2 PointCloud2 使用同一坐标系。\n",
        ),
        ("8. 压缩伪影量化", rpt.compression_block(compression)),
        (
            "9. 学习型匹配器与阶段三决策",
            learned_state
            + "\n\n阶段三唯一要回答的问题（误差是否下降、覆盖率是否显著扩大、帧率是否达标）"
            "必须在目标 PC 上用 Fast-FoundationStereo/HITNet 实测；当前证据不足以做保留/回退决策，"
            "因此默认策略仍是纯 SGBM。\n",
        ),
        (
            "10. 复现",
            "```bash\n"
            "python -m tools.run_demo --out runs/demo --frames 8 --compression\n"
            "python -m pc.calib.cli synthetic --out runs/demo/calib/stereo_calib.npz --views 15\n"
            "```\n",
        ),
        (
            "11. 适用范围与限制",
            "- 本报告由合成场景生成：无真实镜头畸变、无卷帘快门、无运动模糊、无 H.264 解码链路"
            "（压缩项单独测）；用于验证坐标/符号/单位与误差量级，不能替代实机验收。\n"
            "- 本机是 Raspberry Pi 5，帧率为 Pi 上的处理耗时；目标 PC（RTX 3060+）需复测。\n"
            "- 检测器为彩色连通域占位，检测精度不代表 YOLOv8 表现。\n",
        ),
    ]

    summary = {
        "environment": rpt.environment(),
        "config": config_to_dict(cfg),
        "calibration": calib.summary(),
        "frames": args.frames,
        "timings_ms": stage_stats,
        "fps": fps,
        "coverage": {
            "fused_overall": float(np.mean(coverages_all)),
            "on_gt_valid": float(np.mean(coverages_gt)),
            "object_pixels": float(np.mean(obj_coverages)) if obj_coverages else None,
        },
        "objects_constrained": agg_c,
        "objects_unconstrained": agg_u,
        "rows_constrained": rows_c,
        "rows_unconstrained": rows_u,
        "sync_simulation": sync_stats,
        "compression": compression,
        "detector": result.meta.get("detector"),
        "pointcloud": {
            "voxel_size_m": cfg.pointcloud.voxel_size_m,
            "stride": cfg.pointcloud.stride,
            "points_raw": int(len(points_raw)),
            "points": int(len(points)),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "objects_all.json").write_text(json.dumps(rows_c, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_report:
        rpt.write_markdown(
            out / "report.md",
            "实时双目 3D 场景感知验证报告（合成场景）",
            sections,
            meta=rpt.environment(),
        )

    print("\n=== 摘要 ===")
    print(f"标定 B={calib.baseline_m * 1000:.3f}mm f={calib.focal_px:.3f}px RMS={calib.rms_stereo:.4f}px")
    print(f"物体 |ΔZ| 中位数 {agg_c.get('depth_abs', {}).get('median', 0) * 1000:.1f} mm，"
          f"3D 中位数 {agg_c.get('err_3d', {}).get('median', 0) * 1000:.1f} mm")
    print(f"帧率 {fps:.2f} fps（Pi 5 本机，非目标 PC）")
    print(f"输出目录: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
