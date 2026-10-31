"""配置：YAML/JSON -> 冻结 dataclass，带默认值与校验。

命令行里所有可调参数最终都落到 AppConfig，方便把一次实验的全部参数
写进报告（可复现）。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, TypeVar, get_type_hints

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# 各子系统配置
# --------------------------------------------------------------------------- #
@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_count: int = 3
    # "auto" 时按平台自动选：有 Picamera2 用真实相机，否则合成
    backend: str = "auto"
    left_index: int = 0
    right_index: int = 1


@dataclass
class SyncConfig:
    enabled: bool = True
    kp: float = 0.05
    target_offset_ns: int = 0
    max_step_us: int = 300
    min_frame_us: int = 30_000
    max_frame_us: int = 40_000


@dataclass
class StreamConfig:
    # "zmq" | "none" | "file"
    transport: str = "zmq"
    endpoint: str = "tcp://0.0.0.0:5555"
    jpeg_quality: int = 90
    # 无损帧旁路（用于压缩伪影量化）
    raw_max_pairs: int = 0
    h264_bitrate_kbps: int = 2500


@dataclass
class CalibConfig:
    pattern_cols: int = 9
    pattern_rows: int = 6
    square_mm: float = 30.0
    min_views: int = 10
    clahe: bool = False
    calib_path: str = "data/calib/stereo_calib.npz"


@dataclass
class SgbmConfig:
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    p1: int = 8 * 5 * 5
    p2: int = 32 * 5 * 5
    uniqueness_ratio: int = 10
    speckle_window_size: int = 150
    speckle_range: int = 2
    disp12_max_diff: int = 1
    mode: str = "sgbm_3way"
    lr_max_diff: float = 1.0
    texture_threshold: float = 6.0
    min_valid_ratio: float = 0.05
    subpixel: bool = True
    # 学习型匹配器（阶段三）
    learned_backend: str = "none"  # none | onnx | tensorrt
    learned_model: str = ""
    learned_input_mode: str = "batch"  # batch(2x3xHxW) | concat(1x6xHxW)
    learned_input_size: str = ""       # 如 "640x480"；空 = 原图尺寸


@dataclass
class DetectConfig:
    backend: str = "auto"  # auto | yolo | color | none
    model_path: str = ""
    conf: float = 0.25
    imgsz: int = 640
    classes: Optional[str] = None  # 逗号分隔
    device: str = "0"


@dataclass
class ObjectConfig:
    bbox_shrink: float = 0.15
    min_valid_ratio: float = 0.20
    min_pixels: int = 200
    # 需要地面约束的类别；"*" 表示全部
    ground_classes: str = "object,box,cup,bottle,chair"
    ground_band: float = 0.45       # 取图像底部多少比例拟合地面（越大越能覆盖远处地面）
    ransac_iters: int = 300
    ransac_thresh_m: float = 0.02
    min_ground_points: int = 800
    snap_bottom: bool = True


@dataclass
class ProofConfig:
    """验证用真值（仅离线合成/已知距离场景有）。"""
    enabled: bool = False
    target_depths_m: str = "0.65,1.00,1.50"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    sgbm: SgbmConfig = field(default_factory=SgbmConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    objects: ObjectConfig = field(default_factory=ObjectConfig)
    proof: ProofConfig = field(default_factory=ProofConfig)
    # 合成场景（无相机时使用）
    scene: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 反序列化：dict -> dataclass（只认已知字段，拼错立即报错）
# --------------------------------------------------------------------------- #
def _construct(cls: Type[T], data: Mapping[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} 不是 dataclass")
    known = {f.name: f for f in fields(cls)}
    hints = get_type_hints(cls)
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__} 未知配置项: {sorted(unknown)}")
    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        hint = hints.get(name, f.type)
        if is_dataclass(hint):
            kwargs[name] = _construct(hint, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[call-arg]


def _with_defaults(cls: Type[T], data: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并：默认值 + 用户覆盖。"""
    merged = asdict(cls())  # type: ignore[call-arg]
    for key, value in data.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def load_config(path: Optional[str | Path] = None, overrides: Optional[Mapping[str, Any]] = None) -> AppConfig:
    """读取 YAML/JSON 配置并与默认值合并。"""
    data: Dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"配置文件顶层必须是映射: {p}")
    if overrides:
        for key, value in overrides.items():
            _deep_update(data, key.split("."), value)
    merged = _with_defaults(AppConfig, data)
    return _construct(AppConfig, merged)


def _deep_update(target: Dict[str, Any], path: list[str], value: Any) -> None:
    cur = target
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def config_to_dict(cfg: AppConfig) -> Dict[str, Any]:
    return asdict(cfg)


def dump_config(cfg: AppConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        p.write_text(yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        p.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
