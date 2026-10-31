# 接口与数据契约

本文件是"阶段一前必须定死"的那份契约。改这里之前先改代码里的 dataclass 和测试。

## 1. 坐标、单位、符号

| 项 | 约定 |
|---|---|
| 相机坐标系 | X 右、Y 下、Z 前（右手系，OpenCV 一致） |
| 左相机 | 光心为原点 |
| 右相机 | 光心在 `(+B, 0, 0)` |
| 基线 B | `‖T‖`（米），从标定文件读取，**不假设 83 mm** |
| 校正后焦距 f | `P1[0,0]`（像素） |
| 视差 | `d = u_L - u_R > 0` |
| 深度 | `Z = f * B / d`，`d<=0` 记为无效 |
| 反投影 | `X=(u-cx)Z/f`，`Y=(v-cy)Z/f`，`cx=P1[0,2]`，`cy=P1[1,2]` |
| 时间戳 | 整数纳秒；`SensorTimestamp` 为同一 Pi 的单调时钟域；墙钟只用于诊断 |
| 姿态角 | 棋盘格位姿为相机→棋盘，`cv2.Rodrigues` 约定 |

地面在相机系中近似 `Y = pX + qZ + r`（Y 向下，所以地面 r>0）。物体中心高度用"支撑面 + 物体高度"修正。

## 2. 帧消息

```text
MAGIC(4)="SDP1" | header_len(uint32 LE) | header(JSON) | payload
```

ZMQ 使用多段 `[header_json, payload]`，避免大 payload 再拷一次。
header 字段：

```json
{"v":1,"pair_id":42,"seq":42,"side":"left","sensor_ts_ns":1000000000,
 "capture_wall_ns":1690000000000000000,"w":640,"h":480,"enc":"jpeg","bytes":12345}
```

- `pair_id` 由 Pi 端递增分配；配对**只认它**，不认到达顺序。
- `enc`：`jpeg`（推流）/ `raw_bgr8`（无损旁路）/ `h264`（预留）。
- `side` 只能是 `left` / `right`。
- 解码后必须是 BGR `uint8`，形状 `(h,w,3)`。

### 配对状态机（`PairAssembler`）

| 情形 | 行为 | 计数 |
|---|---|---|
| 左右都到（任意顺序） | 输出 `StereoPair` | `emitted` |
| 同一侧重复到达 | 丢弃，不覆盖 | `duplicates` |
| 只有一侧，超过 `stale_s` | 淘汰 | `orphans` |
| `pair_id` 小于已输出的最大 id | 丢弃 | `out_of_order` |
| 打开的 pair 超过 `max_open` | 淘汰最老的 | `orphans` |

**绝不跨 `pair_id` 拼接**——这是阶段一最容易犯、后果最隐蔽的错。

## 3. 模块输入输出

### Pi

```python
CapturedPair(left: np.ndarray, right: np.ndarray,
             ts_left_ns: int, ts_right_ns: int, seq: int)
```

后端协议：`start() / read() -> CapturedPair / close() / set_slave_frame_duration(us)`。
实现：`Picamera2Backend`（实机）、`SyntheticBackend`（无相机）、`ReplayBackend`（回放）。

同步环：`PhaseLockController.update(ts_master_ns, ts_slave_ns) -> period_us`。
输出写入从动相机 `FrameDurationLimits=(v, v)`；误差定义 `ts_slave - ts_master - target`。

### PC 接收

```python
ReceivedPair(pair: StereoPair, left: np.ndarray, right: np.ndarray)
```
`pair.ts_diff_ns = right.sensor_ts_ns - left.sensor_ts_ns`。

### 匹配

```python
DisparityResult(
    disp: float32 HxW,      # 无效像素填 0
    valid: bool HxW,
    source: str,            # "sgbm" | "learned"
    latency_ms: float,
    right_disp: float32|None,   # 右目参考视差，正值
    right_valid: bool|None,
    meta: dict,
)
```

### 融合

```python
FusedDisparity(disp, valid, source: uint8 HxW, stats: dict)
# source: 0=无效, 1=SGBM, 2=学习型
```
`stats` 必须包含 `policy="sgbm_priority_no_average"` 与各拒绝原因计数。

### 物体级输出（JSON）

```json
{
  "class": "cup",
  "confidence": 0.87,
  "bbox_2d": [120.5, 210.0, 260.2, 388.4],
  "center_3d": [-0.145, 0.163, 0.620],
  "unconstrained_center_3d": [-0.145, 0.166, 0.620],
  "depth_m": 0.620,
  "depth_source": "sgbm",
  "valid_ratio": 0.43,
  "n_pixels": 15220,
  "height_m": 0.161,
  "ground_applied": true,
  "contact_3d": [-0.146, 0.250, 0.618],
  "extra": {}
}
```

- `center_3d` 是最终值（地面约束生效时为修正值）；`unconstrained_center_3d` 始终保留。
- `depth_source` 取框内像素来源的多数；混合时按多数标注，原始计数在 `extra` 可扩展。
- 有效像素不足的检测不输出，写入 `aggregate.diagnostics.skipped` 并带原因。

## 4. 文件格式

### 录制目录

```text
<root>/pairs.jsonl
<root>/frames/000000_left.png
<root>/frames/000000_right.png
```

`pairs.jsonl` 每行：

```json
{"pair_id":0,"seq":0,"ts_left_ns":1000000000,"ts_right_ns":1001000000,
 "ts_diff_ms":1.0,"left":"frames/000000_left.png","right":"frames/000000_right.png"}
```

逐行 flush；中断也不丢已写索引。`PairReplay` 读回。

### 标定文件 `.npz`

数组：`K1,D1,K2,D2,R,T,E,F,Q,R1,R2,P1,P2,image_size,rms,meta`。
`meta` 是 JSON 字符串（含 `views_used`、`epipolar_dy`、`roi`）。
接口：`CalibrationResult.load()/save()/summary()`。

### 点云 `.ply`

二进制小端，`x y z red green blue`（float32 + uint8），BGR 输入会在写出时转 RGB。

## 5. 失败策略

- 标定角点不足 / 视图数不够：抛 `CalibrationError`，不生成"能用"的文件。
- 学习型模型缺失：抛 `LearnedMatcherUnavailable`，**不静默用 SGBM 冒充补全**。
- YOLO 权重缺失且显式指定 `backend: yolo`：抛 `DetectorUnavailable`；`auto` 才退到彩色占位并打印告警。
- 左右帧时间戳差超过 0.75 帧周期：`Picamera2Backend` 重试，仍失败抛 `FrameSkewError`，由服务丢弃该帧对。
- 地面平面不可信（倾斜/在相机上方/内点太少）：不施加约束，`ground_applied=false`。

## 6. 配置

`configs/*.yaml` 与 `shared/config.py` 的 dataclass 一一对应；未知字段直接报错（防拼写错误静默生效）。
命令行覆盖用点路径：`load_config(path, {"camera.backend": "synthetic"})`。
