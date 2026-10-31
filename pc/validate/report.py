"""验证报告生成（Markdown）。所有数字来自 benchmark / pipeline 的原始输出。"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from pc.calib.calibrate import CalibrationResult
from shared.stats import fmt, markdown_table, summarize


def environment() -> Dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def calibration_block(calib: CalibrationResult) -> str:
    lines = [
        f"- 图像尺寸：{calib.image_size[0]}×{calib.image_size[1]}",
        f"- 基线 B（由 ‖T‖ 读取，非假设值）：**{calib.baseline_m * 1000:.3f} mm**",
        f"- 校正后焦距 f = P1(0,0)：**{calib.focal_px:.3f} px**",
        f"- 双目联合重投影 RMS：**{calib.rms_stereo:.4f} px**（验收 < 0.5 px）",
        f"- 单目重投影 RMS：左 {calib.rms_mono_left:.4f} px / 右 {calib.rms_mono_right:.4f} px（验收 < 0.3 px）",
        f"- 有效视图数：{calib.meta.get('views_used', '—')}，角点失败跳过：{calib.meta.get('skipped', [])}",
    ]
    epi = calib.meta.get("epipolar_dy", {})
    if epi:
        lines.append(
            f"- 校正后同名点 |dy|：中位数 {fmt(epi.get('median_abs_px'), 4)} px，"
            f"P90 {fmt(epi.get('p90_abs_px'), 4)} px"
        )
    return "\n".join(lines) + "\n"


def synchronization_block(sim: Optional[Dict[str, Any]]) -> str:
    if not sim:
        return "_未运行同步仿真。_\n"
    before = sim.get("before", {})
    after = sim.get("after", {})
    rows = [
        {
            "阶段": "未加控制（出厂读数）",
            "中位数 (ms)": fmt(before.get("median"), 4),
            "IQR (ms)": fmt(sim.get("before_iqr_ms"), 4),
            "P25 (ms)": fmt(before.get("p25"), 4),
            "P75 (ms)": fmt(before.get("p75"), 4),
        },
        {
            "阶段": "P 环稳定后",
            "中位数 (ms)": fmt(after.get("median"), 4),
            "IQR (ms)": fmt(sim.get("after_iqr_ms"), 4),
            "P25 (ms)": fmt(after.get("p25"), 4),
            "P75 (ms)": fmt(after.get("p75"), 4),
        },
    ]
    note = (
        "> ⚠️ 上表是**离线同步仿真**（`pi.sync.simulate_sync`），用于验证控制环方向与收敛性，"
        "不是 Pi 5 实机测量。实机数字必须用 `python -m pi.tools.timestamp_baseline` 采集。\n\n"
    )
    return note + markdown_table(rows, ["阶段", "中位数 (ms)", "IQR (ms)", "P25 (ms)", "P75 (ms)"])


def object_table(rows: Sequence[Dict[str, Any]], title: str = "逐物体误差") -> str:
    if not rows:
        return f"### {title}\n\n_无匹配物体（检查检测器与真值框的 IoU 匹配）。_\n"
    table_rows = []
    for r in rows:
        table_rows.append(
            {
                "物体": r["name"],
                "GT 深度 (m)": fmt(r["gt_depth_m"], 3),
                "估计深度 (m)": fmt(r["est_depth_m"], 3),
                "ΔZ (mm)": fmt(r["dz_m"] * 1000, 1),
                "ΔX (mm)": fmt(r["dx_m"] * 1000, 1),
                "ΔY (mm)": fmt(r["dy_m"] * 1000, 1),
                "3D 误差 (mm)": fmt(r["err_3d_m"] * 1000, 1),
                "相对深度误差": f"{r['depth_rel_err'] * 100:.2f}%",
                "来源": r["depth_source"],
                "地面约束": "是" if r["ground_applied"] else "否",
                "有效比": f"{r['valid_ratio'] * 100:.0f}%",
            }
        )
    cols = [
        "物体", "GT 深度 (m)", "估计深度 (m)", "ΔZ (mm)", "ΔX (mm)", "ΔY (mm)",
        "3D 误差 (mm)", "相对深度误差", "来源", "地面约束", "有效比",
    ]
    return f"### {title}\n\n" + markdown_table(table_rows, cols)


def aggregate_by_name(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["name"], []).append(r)
    out = []
    for name, rs in groups.items():
        dz = [abs(r["dz_m"]) for r in rs]
        e3d = [r["err_3d_m"] for r in rs]
        rel = [r["depth_rel_err"] for r in rs]
        out.append(
            {
                "物体": name,
                "帧数": len(rs),
                "GT 深度 (m)": fmt(np.median([r["gt_depth_m"] for r in rs]), 3),
                "|ΔZ| 中位数 (mm)": fmt(np.median(dz) * 1000, 1),
                "|ΔZ| P90 (mm)": fmt(np.percentile(dz, 90) * 1000, 1),
                "相对深度误差中位数": f"{np.median(rel) * 100:.2f}%",
                "3D 误差中位数 (mm)": fmt(np.median(e3d) * 1000, 1),
            }
        )
    return out


def comparison_table(
    unconstrained: Sequence[Dict[str, Any]], constrained: Sequence[Dict[str, Any]]
) -> str:
    """无约束 / 有约束两组数字并排。"""
    def agg(rows):
        out = {}
        for r in rows:
            out.setdefault(r["name"], []).append(r)
        return out

    u, c = agg(unconstrained), agg(constrained)
    names = sorted(set(u) | set(c))
    table = []
    for name in names:
        def med_abs(rows, key):
            if not rows:
                return None
            return float(np.median([abs(r[key]) for r in rows]))

        ur = u.get(name, [])
        cr = c.get(name, [])
        table.append(
            {
                "物体": name,
                "无约束 |ΔZ| (mm)": fmt(med_abs(ur, "dz_m") * 1000 if ur else None, 1),
                "有约束 |ΔZ| (mm)": fmt(med_abs(cr, "dz_m") * 1000 if cr else None, 1),
                "无约束 |ΔY| (mm)": fmt(med_abs(ur, "dy_m") * 1000 if ur else None, 1),
                "有约束 |ΔY| (mm)": fmt(med_abs(cr, "dy_m") * 1000 if cr else None, 1),
                "无约束 3D (mm)": fmt(np.median([r["err_3d_m"] for r in ur]) * 1000 if ur else None, 1),
                "有约束 3D (mm)": fmt(np.median([r["err_3d_m"] for r in cr]) * 1000 if cr else None, 1),
                "地面约束生效帧": f"{sum(1 for r in cr if r['ground_applied'])}/{len(cr)}",
            }
        )
    cols = [
        "物体", "无约束 |ΔZ| (mm)", "有约束 |ΔZ| (mm)", "无约束 |ΔY| (mm)",
        "有约束 |ΔY| (mm)", "无约束 3D (mm)", "有约束 3D (mm)", "地面约束生效帧",
    ]
    return "下表同时保留无约束/有约束两组数字：\n\n" + markdown_table(table, cols)


def distance_band_table(rows: Sequence[Dict[str, Any]], edges: Sequence[float]) -> str:
    bands = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = [r for r in rows if lo <= r["gt_depth_m"] < hi]
        if not sel:
            continue
        dz = [abs(r["dz_m"]) for r in sel]
        rel = [r["depth_rel_err"] for r in sel]
        bands.append(
            {
                "距离段 (m)": f"{lo:.2f}–{hi:.2f}",
                "样本帧×物体": len(sel),
                "|ΔZ| 中位数 (mm)": fmt(np.median(dz) * 1000, 1),
                "|ΔZ| P90 (mm)": fmt(np.percentile(dz, 90) * 1000, 1),
                "相对误差中位数": f"{np.median(rel) * 100:.2f}%",
            }
        )
    if not bands:
        return "_无距离段样本。_\n"
    return markdown_table(
        bands,
        ["距离段 (m)", "样本帧×物体", "|ΔZ| 中位数 (mm)", "|ΔZ| P90 (mm)", "相对误差中位数"],
    )


def compression_block(result: Optional[Dict[str, Any]]) -> str:
    if not result:
        return "_未运行压缩伪影量化（加 `--compression`）。_\n"
    rows = []
    for s in result["settings"]:
        weak = s.get("weak_texture_err", {})
        rels = [v for v in (s.get("object_depth_shift_rel") or {}).values() if v is not None]
        rows.append(
            {
                "设置": s["name"],
                "码率 (Mbps)": fmt(s["bitrate_mbps"], 2),
                "每对字节": f"{s['bytes_per_pair'] / 1024:.1f} KiB",
                "有效像素/参考": f"{s['valid_ratio_vs_ref'] * 100:.1f}%",
                "视差误差中位数 (px)": fmt(s["disp_err"].get("median"), 3),
                "弱纹理视差误差中位数 (px)": fmt(weak.get("median"), 3),
                "最大物体相对深度偏移": f"{max(rels) * 100:.2f}%" if rels else "—",
                "判定": "参考" if s["kind"] == "raw" else ("可用" if (not rels or max(rels) < 0.02) else "需提高码率"),
            }
        )
    cols = [
        "设置", "码率 (Mbps)", "每对字节", "有效像素/参考", "视差误差中位数 (px)",
        "弱纹理视差误差中位数 (px)", "最大物体相对深度偏移", "判定",
    ]
    return markdown_table(rows, cols)


def write_markdown(
    path: str | Path,
    title: str,
    sections: Sequence[tuple[str, str]],
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title}\n"]
    if meta:
        parts.append("## 运行信息\n")
        parts.append(markdown_table([{"项": k, "值": str(v)} for k, v in meta.items()], ["项", "值"]))
        parts.append("")
    for heading, body in sections:
        parts.append(f"## {heading}\n")
        parts.append(body.rstrip() + "\n")
    p.write_text("\n".join(parts), encoding="utf-8")
    return p
