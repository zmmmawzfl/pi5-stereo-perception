"""小样本统计工具：不引入 scipy，报告里所有数字都由这里产出。"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def _arr(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    """n / mean / std / median / p25 / p75 / min / max，缺失值自动剔除。"""
    arr = _arr(values)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def iqr(values: Iterable[float]) -> float:
    arr = _arr(values)
    if arr.size < 2:
        return 0.0
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def abs_error_stats(errors: Sequence[float]) -> Dict[str, Any]:
    """误差统计：中位数 / 均值 / P90 / RMSE / 偏置。"""
    arr = np.asarray(list(errors), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "max": float(arr.max()),
    }


def signed_error_stats(errors: Sequence[float]) -> Dict[str, Any]:
    """带符号误差：额外给出偏置（均值）与标准差。"""
    out = abs_error_stats(np.abs(errors))
    arr = np.asarray(list(errors), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size:
        out["bias"] = float(arr.mean())
        out["std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return out


def fmt(value: Any, digits: int = 3) -> str:
    """报告表格里的数字格式化。"""
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—"
        if value != 0 and abs(value) < 10 ** (-digits):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    """把字典列表转成 Markdown 表。"""
    if not rows:
        return "_（无数据）_\n"

    def cell(value) -> str:
        # Markdown 表格里未转义的 "|" 会切断列，必须转义（例如 |ΔZ|）
        return str(value).replace("|", "\\|")

    head = "| " + " | ".join(cell(c) for c in columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for row in rows:
        body.append("| " + " | ".join(cell(row.get(c, "")) for c in columns) + " |")
    return "\n".join([head, sep, *body]) + "\n"
