# 项目状态交接（给下一个 AI / 工程师）

> 更新时间：2026-11-01
> 仓库根目录：`/home/chenai/桌面/深度`（**注意：不再是 `深度/stereo-depth/`，项目已上移一层**）
> 当前提交：`c38a34e Add unit tests for runtime monitoring, SGBM matcher, synchronization, and synthetic scene validation`
> 当前工作区：HEAD `c38a34e` + 本交接文档与两处文案修改（未提交）；75 个测试全部通过。

---

## 0. 先读这个（TL;DR）

**这是什么：**
一套面向机器人的**实时双目 3D 场景感知**原型。输入左右双目视频，输出：

1. 校正后的左右图；
2. 视差图 / 深度图（米）；
3. 彩色点云（PLY + ROS 2 `PointCloud2`，默认 0.03 m 体素降采样）；
4. 物体级 JSON（2D 框、3D 中心、深度、`depth_source`、覆盖率）；
5. 一套可复现的质量报告（标定、同步、压缩、误差、帧率）。

**它不是什么：**
不是 3D 建模/CAD 软件，不做高保真纹理网格，不导出 Blender/SolidWorks，不直接给 6-DoF 抓取位姿，也不是 SLAM（自身不提供位姿）。

**现在能做什么：**
没有相机也能用**合成场景 + 录制回放**跑完整链路：标定、SGBM、融合、物体 3D、点云、压缩量化、报告。Pi↔PC 的 ZMQ 链路已在本机回环验证。

**现在不能做什么：**
实机相机采集、真实镜头标定、真实同步、真实精度、目标 PC 帧率、ROS 2/Foxglove/RTAB-Map/Nav2 运行期，都还没有验证。学习型匹配器（阶段三）没有任何实测。

**下一步最重要的事：**
相机到货后先做“第一天基线”（时间戳差 + 运动抖动 + 真实标定），再谈精度；ROS 2 相关只在装了 ROS 2 的机器上验证。

---

## 1. 项目目标与边界

### 1.1 最终目标

从双相机实时视频中恢复**度量正确**的深度与 3D 结构，面向：

- 机器人导航避障（点云进代价地图/占据栅格）；
- 抓取初筛（2D 检测框 + 框内点云 + 物体 3D 中心）；
- 场景理解（物体级 3D 位置与深度来源）；
- 双目硬件/算法评估（同步、标定、压缩、SGBM、ROI/降分辨率、学习型对照）。

### 1.2 主线技术路线

```text
双目校正
  → SGBM（96 视差，可选 ROI / 半分辨率）
  → 左右一致性 + 纹理门限
  → SGBM 优先融合（学习型只补失效像素，不平均）
  → 检测
  → 物体级聚合（框内中位视差 + 地面约束）
  → 深度图 / 体素彩色点云
  → PLY / ROS 2 PointCloud2 / objects JSON
```

SGBM 是**度量锚点**。学习型匹配器只是阶段三的候选补全路径，必须同时满足“误差下降、覆盖率显著扩大、帧率达标”三项才保留，否则退回纯 SGBM。TSDF 只是可选扩展，不能成为主线依赖。

### 1.3 非目标（明确不做）

- 完整 3D 建模、可编辑 CAD、逆向工程级重建；
- 高保真纹理网格、Blender/SolidWorks 导出；
- 6-DoF 抓取位姿、手眼标定（当前只到“抓取初筛”）；
- SLAM / 视觉里程计（TSDF 需要外部给位姿）；
- 依赖学习型单目深度先验（相对尺度会污染绝对测距）。

### 1.4 目标演变（重要，避免下一个会话按旧目标改）

最初（2026-10-31 上午）的目标是“Pi 5 传感器前端 + PC 端立体匹配与物体级 3D 输出”，带完整 8 周阶段计划、学习型匹配器阶段三、TSDF 阶段四。

当日下午用户重新定位为：**“实时双目 3D 场景感知 / 机器人视觉感知”**，明确：

- 主线必须传统 SGBM + 体素降采样 + ROI/降分辨率，`numDisparities=64 或 96`；
- 不依赖学习型深度模型；
- ROS 2 `PointCloud2` + Foxglove 可视化；
- 对象级 2D 框 / 3D 中心 / 深度来源；
- 导航用占据栅格/OctoMap，不追高保真网格；
- 抓取先用 2D 分割 + 点云投影，6-DoF 后续；
- 非目标见 1.3。

**后续工作以“机器人感知”这版目标为准，不要按 CAD 建模方向改。**

---

## 2. 当前状态总表

| 层 | 状态 | 证据 / 说明 |
|---|---|---|
| PC 核心管线（校正→SGBM→融合→物体→点云） | ✅ 合成验证通过 | 75 测试；`runs/demo/report.md` |
| Pi 采集/同步/推流代码 | ✅ 代码完成；❌ 实机未测 | 相机未到；合成后端/回放可用 |
| Pi↔PC ZMQ 链路 | ✅ 本机回环通过 | 6 对帧 `emitted=6, orphans=0`；`runs/live_link` |
| 双目标定 | ✅ 合成 + 录制目录验证 | 15 视图 RMS 0.047 px；录制 14 对视图中位 |
| 物体级 3D + 地面约束 | ✅ 合成验证 | `|ΔZ|` 中位 1.2 mm；`|ΔY|` 近目标 7.1→0.7 mm |
| 点云 + 体素降采样 | ✅ 合成/回放验证 | 20,620 → 3,518 点（0.03 m） |
| ROS 2 发布节点 | ✅ 代码完成；❌ 运行期未验证 | 本机没有 ROS 2 |
| Foxglove / RTAB-Map / Nav2 配置 | ✅ 文档/参数完成；❌ 未跑 | `docs/ros2_foxglove.md`、`ros2/config/` |
| YOLO 检测 | ⚠️ 代码完成；❌ 权重/精度未验证 | 无 ultralytics、无引擎、无权重；默认退彩色连通域 |
| 学习型匹配器（阶段三） | ⚠️ ONNX 适配器骨架；❌ 未实测 | 无 GPU、无模型、无数字 |
| TSDF（阶段四） | ⚠️ 代码完成；❌ 未跑 | 缺 Open3D、缺位姿；不作为主线 |
| 实机真实精度/同步/CPU/延迟 | ❌ 全部待测 | 相机未到；目标 PC 未测试 |
| 运动边界 / 准静态速度上界 | ⚠️ 工具就绪；❌ 未测 | `tools/motion_jitter.py` |

---

## 3. 环境与硬件

### 3.1 当前机器（开发/验证环境）

| 项 | 值 |
|---|---|
| 主机 | Raspberry Pi 5，4 核，约 3.9 GiB 内存（`free -h` 显示 3.9Gi） |
| OS | Linux 6.18.50+rpt-rpi-v8，aarch64 |
| Python | 3.13.5 |
| OpenCV | 4.10.0（含 `ximgproc`、SGBM、`findChessboardCornersSB`） |
| NumPy | 2.2.4 |
| pyzmq | 27.2.0 |
| PyYAML | 6.0.2 |
| ffmpeg | 7.1.5（H.264 压缩量化） |
| pytest | 9.2.0 |
| 缺失 | ROS 2、Torch、ultralytics、Open3D、SciPy |
| 虚拟环境 | `.venv/`，`--system-site-packages`，复用系统 OpenCV |

**注意：** 本机是 Pi 5，不是目标 PC。`runs/demo` 的 3.75 fps 是 **Pi 上跑 PC 端 SGBM 的耗时**，不是 RTX 3060 的性能。

### 3.2 目标硬件（用户计划）

Pi 5 端：

- Pi 5 8GB 优先；官方 27W USB-C PD；
- 双 IMX219，刚性固定基线（常见 83 mm，以标定 `T` 模长为准）；
- 2 根 CSI 排线，注意 cam0/cam1 方向；
- 主动散热；≥32GB microSD 或 NVMe；
- 千兆网线直连 PC 或交换机，避免 WiFi。

PC 端：

- NVIDIA GPU，≥6GB 显存，RTX 3060 级别起；
- 千兆网口；
- Ubuntu 22.04 优先（TensorRT/Open3D/CUDA 生态）。

标定与验证：

- 棋盘格或 Charuco，A3 硬质基板；方格尺寸实测后用 mm；
- 卷尺、纹理板、可选小转台。

### 3.3 依赖文件

- `requirements.txt`：numpy / PyYAML / pyzmq；
- `requirements-pc.txt`：OpenCV、ultralytics、onnxruntime-gpu、open3d；
- `requirements-dev.txt`：pytest。

---

## 4. 架构与数据流

### 4.1 端到端

```text
Pi 5
┌─────────────────────────────────────────────────────────────┐
│ Picamera2(left)  ─┐                                        │
│ Picamera2(right) ─┴→ CapturedPair(left,right,tsL,tsR,seq)  │
│                              │                             │
│                    PhaseLockController                     │
│                    update(tsL,tsR) -> period_us            │
│                    set_slave_frame_duration(period)        │
│                              │                             │
│                    ZmqPublisher (JPEG + header)            │
└──────────────────────────────┼─────────────────────────────┘
                               │ tcp://PI:5555
PC                             ▼
┌─────────────────────────────────────────────────────────────┐
│ ZmqPairSource → PairAssembler（只按 pair_id 配对）          │
│       │                                                     │
│       ▼                                                     │
│ Rectifier（remap，去畸变 + 极线校正）                       │
│       │                                                     │
│       ▼                                                     │
│ SGBMMatcher（96 视差，可选 ROI/scale，右目视差做 LR）      │
│       │                    ┌── 学习型（可选，默认关闭）──┐  │
│       ▼                    │ 只补 SGBM 失效像素，不平均  │  │
│ FuseRouter ←───────────────┘                              │  │
│       │                                                     │
│       ├─→ 亚像素精化（SAD 抛物线）                          │
│       ├─→ 检测（YOLO 或彩色连通域占位）                     │
│       ├─→ ObjectAggregator（框内中位视差 + 地面 RANSAC）    │
│       ├─→ depth_from_fused（Z=fB/d）                        │
│       └─→ colored_pointcloud → voxel_downsample → PLY       │
│                                                             │
│ 输出：objects.json / overlay.png / depth.png / cloud.ply     │
│       ROS 2：/stereo/points、/stereo/depth、...              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 关键时序

1. Pi 每读一对帧：`tsL, tsR` 来自同一 request 的 `SensorTimestamp`；
2. `PhaseLockController.update(tsL, tsR)` 算从动相机新周期；
3. 下一帧前把周期写入从动相机 `FrameDurationLimits`；
4. `ZmqPublisher` 分别发左/右 JPEG，header 带 `pair_id/side/sensor_ts_ns`；
5. PC `PairAssembler` 只按 `pair_id` 重组，**绝不靠到达顺序**；
6. PC 管线按固定顺序处理，耗时写进 `timings_ms`；
7. 输出 JSON/PLY/ROS 话题。

---

## 5. 目录与模块职责

```text
/home/chenai/桌面/深度/
├── pi/                         # 树莓派端
│   ├── capture/
│   │   ├── base.py             # CapturedPair / CameraBackend 协议
│   │   ├── synthetic_backend.py# 合成双目（无相机/测试）
│   │   ├── picamera2_backend.py# 双 Picamera2，SensorTimestamp，buffer_count>=3
│   │   └── factory.py          # auto/picamera2/synthetic/replay
│   ├── sync/p_controller.py    # PhaseLockController + simulate_sync
│   ├── stream/publisher.py     # ZMQ PUB、JPEG、FilePublisher、Tee、Null
│   ├── service.py              # 常驻服务：采集→同步→推流/录制
│   └── tools/
│       ├── timestamp_baseline.py # 第一天时间戳基线
│       └── record_pairs.py       # 无损录制
├── pc/
│   ├── receive/
│   │   ├── subscriber.py       # ZMQ SUB + PairAssembler
│   │   ├── record_pairs.py      # PC 端最小接收器（落盘）
│   │   └── live.py              # 在线/回放管线，写 JSON/overlay/PLY
│   ├── calib/
│   │   ├── calibrate.py        # 角点、单目→双目、stereoRectify、极线
│   │   └── cli.py              # calibrate/synthetic/inspect
│   ├── rectify/rectifier.py    # initUndistortRectifyMap + remap
│   ├── stereo/
│   │   ├── sgbm.py             # SGBM + ROI/scale + 右目视差 + 可选 WLS
│   │   ├── subpixel.py         # SAD 抛物线亚像素
│   │   ├── learned.py          # ONNX/TensorRT 适配器骨架
│   │   └── base.py             # DisparityResult
│   ├── fuse/router.py          # LR + 纹理门限 + SGBM 优先路由
│   ├── detect/
│   │   ├── yolo.py             # ultralytics YOLO
│   │   ├── color.py            # HSV 彩色连通域占位
│   │   └── factory.py          # auto/yolo/color/none
│   ├── objects/aggregator.py   # 框内中位视差、地面平面、3D 中心
│   ├── depth/pointcloud.py     # Z=fB/d、彩色点云、体素、框内点云、PLY
│   ├── ros2/pointcloud_node.py # ROS 2 发布节点（独立）
│   ├── tsdf/fuse_offline.py    # 可选 TSDF 融合（需 Open3D + 位姿）
│   ├── validate/
│   │   ├── metrics.py          # 误差、覆盖率、物体匹配
│   │   ├── compression.py      # JPEG/H.264 往返量化
│   │   ├── report.py           # Markdown 报告
│   │   ├── runtime.py          # 进程级 CPU/内存
│   │   └── visualize.py        # overlay/source map
│   └── pipeline.py             # 端到端编排
├── shared/
│   ├── messages.py             # SDP1 协议、PairAssembler
│   ├── config.py               # dataclass 配置 + 校验
│   ├── stats.py                # 统计与表格
│   ├── synthetic.py            # 合成场景（地面 + 3 个盒子）
│   └── recording.py            # 录制目录 / PairReplay
├── tools/
│   ├── run_demo.py             # 无相机端到端 demo + 报告
│   ├── compression_report.py   # 压缩伪影量化
│   └── motion_jitter.py        # 深度抖动 / 运动边界
├── ros2/
│   ├── config/rtabmap.yaml     # RTAB-Map（MemoryThr=300）
│   ├── config/nav2_obstacle.yaml
│   └── launch/stereo_perception.launch.py
├── configs/                    # default / pi / pc
├── tests/                      # 75 个 unittest
├── docs/                       # 接口、第一天、阶段三、验收、runbook、本文件
├── data/recordings/            # 测试录制（gitignore）
└── runs/                       # 报告与产物（gitignore）
```

---

## 6. 数据契约

### 6.1 帧消息（Pi → PC）

线格式：

```text
MAGIC(4)="SDP1" | header_len(uint32 LE) | header(JSON utf-8) | payload
```

ZMQ 优先用多段 `[header_json, payload]`。header：

```json
{"v":1,"pair_id":42,"seq":42,"side":"left","sensor_ts_ns":1000000000,
 "capture_wall_ns":1690000000000000000,"w":640,"h":480,"enc":"jpeg","bytes":12345}
```

- `pair_id` 由 Pi 端 `CapturedPair.pair_id = seq` 递增；
- `enc`：`jpeg` / `raw_bgr8` / `h264`；
- `side`：`left` / `right`；
- 解码后必须是 BGR `uint8`，形状 `(h,w,3)`。

### 6.2 配对状态机

`PairAssembler`：

| 情形 | 行为 | 计数 |
|---|---|---|
| 左右都到（任意顺序） | 输出 `StereoPair` | `emitted` |
| 同一侧重复 | 丢弃，不覆盖 | `duplicates` |
| 只有一侧超时 | 淘汰 | `orphans` |
| `pair_id` 小于已输出最大 id | 丢弃 | `out_of_order` |
| 打开 pair 超过 `max_open` | 淘汰最老 | `orphans` |

**绝不跨 `pair_id` 拼接。**

### 6.3 核心 Python 结构

```python
CapturedPair(left, right, ts_left_ns, ts_right_ns, seq)
  .pair_id == seq
  .ts_diff_ns == ts_right_ns - ts_left_ns

StereoPair(pair_id, seq, left: FramePacket, right: FramePacket)
  .ts_diff_ns == right.sensor_ts_ns - left.sensor_ts_ns

DisparityResult(disp, valid, source, latency_ms, right_disp, right_valid, meta)
FusedDisparity(disp, valid, source, stats)
  source: 0=无效, 1=SGBM, 2=学习型

FrameResult(frame_id, timings_ms, detections, aggregate, fused, depth,
            rectified_left, sgbm, learned, meta)
  .objects -> aggregate.objects
  .object_json()

ObjectEstimate(class_name, confidence, bbox_2d, center_3d,
               unconstrained_center_3d, depth_m, depth_source,
               valid_ratio, n_pixels, height_m, ground_applied,
               contact_3d, extra)
```

### 6.4 输出文件

`runs/demo/`：

- `report.md`：完整报告；
- `summary.json`：原始指标；
- `calib/stereo_calib.npz`：标定参数；
- `objects.json` / `objects_all.json`：物体级 JSON；
- `cloud.ply`：二进制小端 `x y z red green blue`；
- `overlay.png` / `depth.png` / `disparity.png` / `disparity_source.png`。

录制目录：

```text
<root>/pairs.jsonl
<root>/frames/000000_left.png
<root>/frames/000000_right.png
```

`pairs.jsonl` 每行含 `pair_id/seq/ts_left_ns/ts_right_ns/ts_diff_ms/left/right`。

### 6.5 ROS 2 话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/stereo/points` | `sensor_msgs/PointCloud2` | `x,y,z` float32 + `r,g,b` uint8，`point_step=16`，已体素降采样 |
| `/stereo/depth` | `sensor_msgs/Image` | `16UC1`，mm，0=无效 |
| `/stereo/left/image_rect` | `sensor_msgs/Image` | `bgr8` |
| `/stereo/left/camera_info` | `sensor_msgs/CameraInfo` | 校正内参：`D=0`，`K=P1[:3,:3]`，`P=P1` |
| `/stereo/objects` | `std_msgs/String` | JSON：对象、覆盖率、耗时、skew |

坐标系：`camera_left`，X 右、Y 下、Z 前，米。

---

## 7. 关键算法与设计决策

### 7.1 坐标、单位、符号

| 项 | 约定 |
|---|---|
| 相机坐标系 | X 右、Y 下、Z 前（OpenCV 右手系） |
| 左相机 | 光心原点 |
| 右相机 | 光心在 `(+B,0,0)` |
| 基线 B | `||T||` 米，从标定文件读，**不假设 83 mm** |
| 校正焦距 f | `P1[0,0]` |
| 视差 | `d = u_L - u_R > 0` |
| 深度 | `Z = f*B/d`，`d<=0` 无效 |
| 反投影 | `X=(u-cx)Z/f`，`Y=(v-cy)Z/f`，`cx=P1[0,2]`，`cy=P1[1,2]` |
| 时间戳 | 整数纳秒；`SensorTimestamp` 同一 Pi 单调时钟域 |

### 7.2 同步 P 环

`pi/sync/p_controller.py` 的公式（当前实现）：

```python
error_us = ((ts_slave_ns - ts_master_ns) - target_offset_ns) / 1000.0
step = clip(-kp * error_us, -max_step_us, max_step_us)
period = clip(base_period_us + step, min_frame_us, max_frame_us)
```

- `kp` 从 0.05 起；
- 从动相机晚到（error>0）→ 缩短周期；
- **不要改成累加周期**：早期实现 `p += kp*error` 会发散；现在每步从 `base_period_us` 重新算；
- `simulate_sync()` 只统计 `control_from` 之后、最后 `settle_frac` 的稳态窗口。

### 7.3 标定

`pc/calib/calibrate.py`：

- 默认 9×6 内角点，方格 30 mm；
- `detect_corners` 优先 `findChessboardCornersSB`，失败回退经典法；
- `calibrate_stereo(fix_k3=True, zero_tangent=True)`：
  - `cv2.CALIB_FIX_K3`：棋盘格数据对 k3 约束不足，放开会拟合出伪畸变，污染 `alpha=0` 的校正视场和 `fB`；
  - `cv2.CALIB_ZERO_TANGENT_DIST`：装配良好的模组切向畸变接近 0；
- `stereoRectify(..., flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)`；
- `B=||T||`，`f=P1[0,0]`，`cx/cy=P1[0,2]/P1[1,2]`；
- 保存 `K1,D1,K2,D2,R,T,E,F,Q,R1,R2,P1,P2,image_size,rms,meta`；
- 极线检查 `epipolar_dy` 写进 `meta`。

### 7.4 SGBM

`pc/stereo/sgbm.py`：

- 默认 `num_disparities=96`（16 的倍数），`block_size=5`，`mode=sgbm_3way`；
- SGBM 原始输出是 `int16 * 16`，**必须除以 16** 再进深度公式；
- 右目参考视差来自 `cv2.ximgproc.createRightMatcher`；右目视差为负，取 `-raw/16` 后再比较；
- `valid = raw < 0 & rd > 0 & rd < num_disparities`（右目）；
- ROI：`roi_xywh` 归一化 `"x,y,w,h"`，只算关注区域，ROI 外无效；
- 降分辨率：`scale` 对 ROI 做 `INTER_AREA`，视差上采样后 **除以 `scale`** 还原到原图；`scale<1` 时强制 `min_disparity=0`；
- 可选 WLS（`use_wls`），默认关闭，Pi 上代价明显；
- `meta` 记录 `scale/roi/work_size/num_disparities`。

### 7.5 融合路由

`pc/fuse/router.py`：

```text
sgbm_ok = sgbm.valid
          & LR(|dL - dR(x - dL)| <= 1)
          & texture(local std 5x5) >= 6.0
learned 只填 (~sgbm_ok) & learned.valid
不做加权平均
```

- `stats["policy"] == "sgbm_priority_no_average"`，每帧可审计；
- 学习型输出只允许落在 SGBM 失效/低置信像素；
- 若学习型大面积覆盖物体主体，先查 LR 符号/纹理阈值，不是模型更好。

### 7.6 亚像素

`pc/stereo/subpixel.py`：SAD 代价抛物线拟合，在 `±0.5 px` 内精化；只在 `fused.valid` 上做。

### 7.7 物体级聚合与地面约束

`pc/objects/aggregator.py`：

1. 检测框收缩 `bbox_shrink=0.15`，避开边界混合像素；
2. 框内有效视差取中位数 `d_med`；
3. `Z=fB/d_med`，`(X,Y)` 用框中心反投影，得到 `unconstrained_center_3d`；
4. 地面：图像底部 `ground_band=0.45` 的像素反投影，RANSAC 拟合 `Y=pX+qZ+r`；
5. 平面可信条件：`|p|<0.5`、`|q|<0.5`、`r>0.02`、内点率≥0.25、RMS<0.08 m；
6. 对 `ground_classes`（默认 `object,box,cup,bottle,chair`）：
   - 底部条带（框底 15%）取接触点视差 → `z_b`，地面 `y=plane.y_at(x_b,z_b)`；
   - 顶部条带（框顶 15%）逐像素反投影 Y，取 P25 得顶高；
   - `height = ground_y - y_top`，合理性检查（0.02–2 m、不超过 0.8×深度、与框像素高度一致性）；
   - `center = (x_center, ground_y - height/2, z_hint)`，`contact_3d=(x_b, ground_y, z_b)`；
7. 有效像素 < `min_pixels` 或比例 < `min_valid_ratio` 的检测跳过，写 `diagnostics.skipped`；
8. 始终保留无约束结果，报告同时给有/无约束两组数字。

### 7.8 点云与体素

`pc/depth/pointcloud.py`：

- `depth_from_fused`：`Z=fB/d`，无效 `nan`；
- `colored_pointcloud`：默认 `stride=2`、`max_depth_m=6`，反投影后可选体素；
- `voxel_downsample`：整数体素索引 + `lexsort + reduceat`，同体素取质心/平均色；30k 点约 17 ms；
- `points_in_bbox`：取 2D 框内点云并补回主点偏移 `X += x1*Z/f`、`Y += y1*Z/f`，供抓取初筛；
- `write_ply`：二进制小端，`x y z red green blue`（BGR→RGB）。

### 7.9 检测

- `YoloDetector`：ultralytics，缺权重/包抛 `DetectorUnavailable`；
- `ColorBlobDetector`：HSV 8 色相桶 + 连通域 + NMS，仅链路占位；
- `auto`：有 `model_path` 文件才用 YOLO，否则彩色占位并打印告警；
- 显式 `backend: yolo` 但权重缺失 → 直接报错，不静默降级。

### 7.10 学习型与 TSDF（可选，不阻塞主线）

- `pc/stereo/learned.py`：ONNX 适配器 + TensorRT 显式报错入口；缺模型抛 `LearnedMatcherUnavailable`；
- 阶段三顺序：Fast-FoundationStereo → HITNet → 降匹配分辨率 → 纯 SGBM；
- 单目深度先验不进入本阶段（相对尺度污染绝对测距）；
- `pc/tsdf/fuse_offline.py`：需要 Open3D + 显式 `poses_c2w`；没有位姿不融合。

---

## 8. 怎么运行

以下命令均在仓库根目录 `/home/chenai/桌面/深度` 执行。

### 8.1 测试

```bash
.venv/bin/python -m unittest discover -s tests -t .
# Ran 75 tests ... OK
```

### 8.2 无相机端到端 demo

```bash
.venv/bin/python -m tools.run_demo --out runs/demo --frames 8 --compression
```

产物：`runs/demo/report.md`、`summary.json`、`objects.json`、`cloud.ply`、`overlay.png` 等。

### 8.3 合成标定

```bash
.venv/bin/python -m pc.calib.cli synthetic \
    --out runs/demo/calib/stereo_calib.npz --views 15
```

### 8.4 真实标定（相机到货后）

```bash
# 目录模式：左右目录文件名排序一一对应
.venv/bin/python -m pc.calib.cli calibrate \
    --left data/calib/left --right data/calib/right \
    --pattern 9x6 --square-mm 30 --out data/calib/stereo_calib.npz

# 录制目录模式（自动跳过没有棋盘的帧）
.venv/bin/python -m pc.calib.cli calibrate \
    --replay-root data/recordings/calib_board \
    --pattern 9x6 --square-mm 30 --out data/calib/stereo_calib.npz
```

### 8.5 回放 / 在线处理

```bash
# 回放
.venv/bin/python -m pc.receive.live \
    --replay-root data/recordings/synth6 \
    --calib runs/demo/calib/stereo_calib.npz \
    --out runs/live --save-cloud-every 10

# 在线（先启动 Pi 推流）
.venv/bin/python -m pc.receive.live \
    --endpoint tcp://<PI_IP>:5555 \
    --calib data/calib/stereo_calib.npz \
    --out runs/live --save-every 30
```

输出 `objects.jsonl`、`overlay_*.png`、`source_*.png`、`cloud_*.ply`、`summary.json`。

### 8.6 Pi 服务与 ZMQ

```bash
# Pi：采集 + 同步 + 推流（相机到货后）
python -m pi.service --config configs/pi.yaml --record-root data/recordings/run1

# PC：最小接收器（先启动它，再让 Pi 发）
python -m pc.receive.record_pairs \
    --endpoint tcp://<PI_IP>:5555 --out data/recordings/day1 --max-pairs 100

# 本机回环（无相机）
.venv/bin/python -m pi.service --backend synthetic --frames 20
.venv/bin/python -m pc.receive.record_pairs \
    --endpoint tcp://127.0.0.1:5555 --out runs/link --max-pairs 6
```

端口：帧 `5555`，时间戳旁路 meta `5556`。

### 8.7 ROS 2

```bash
source /opt/ros/humble/setup.bash
python -m pc.ros2.pointcloud_node \
    --calib data/calib/stereo_calib.npz \
    --endpoint tcp://<PI_IP>:5555 \
    --publish-hz 10

# 无相机回放
python -m pc.ros2.pointcloud_node \
    --calib data/calib/stereo_calib.npz \
    --replay-root data/recordings/day1 --max-pairs 300

# Foxglove
ros2 run foxglove_bridge foxglove_bridge
```

RTAB-Map：

```bash
ros2 launch rtabmap_ros rtabmap.launch.py \
    rgb_topic:=/stereo/left/image_rect \
    depth_topic:=/stereo/depth \
    camera_info_topic:=/stereo/left/camera_info \
    frame_id:=camera_left approx_sync:=true \
    params_file:=ros2/config/rtabmap.yaml
```

Nav2 obstacle layer 见 `ros2/config/nav2_obstacle.yaml`；完整说明见 `docs/ros2_foxglove.md`。

### 8.8 压缩与运动抖动

```bash
.venv/bin/python -m tools.compression_report \
    --replay-root data/recordings/run1 \
    --calib data/calib/stereo_calib.npz \
    --out runs/compression --max-frames 200

.venv/bin/python -m tools.motion_jitter \
    --replay-root data/recordings/static \
    --calib data/calib/stereo_calib.npz \
    --out runs/jitter_static.json
```

---

## 9. 已验证数字（合成场景，Pi 5 本机）

| 指标 | 结果 | 备注 |
|---|---|---|
| 标定基线 B | 83.025 mm（真值 83.0） | 15 视图，合成针孔 |
| 校正焦距 f | 531.125 px | `P1(0,0)` |
| 双目 RMS | 0.0471 px | 验收 <0.5 px |
| 单目 RMS | 左 0.0465 / 右 0.0461 px | 验收 <0.3 px |
| 极线 `\|dy\|` 中位 | 0.0102 px | P90 0.0277 px |
| 物体 `\|ΔZ\|` 中位 | 1.2 mm（0.10%） | 24 样本；单物体最差 6.3 mm |
| 物体 3D 误差中位 | 4.5 mm | 地面约束后 |
| 地面约束 `\|ΔY\|` | 近 7.1→0.7 mm；远 5.5→0.5 mm | 有/无约束都保留 |
| 真值几何区覆盖率 | 65.4% | 纯 SGBM + LR + 纹理门限 |
| 物体像素覆盖率 | 49.0% | |
| 点云 | 20,620 → 3,518 点（17.1%） | 0.03 m 体素 |
| 处理耗时 | SGBM 150 ms；总 266 ms；3.75 fps | **Pi 5**，不是 PC |
| 压缩 H.264 2.5 Mbps | 最大物体深度偏移 0.41% | 800 kbps 0.41%；阈值 2% |
| 同步仿真 | 12.8 ms → 0.0012 ms | 只验证控制方向/收敛 |

**必须强调：** 合成场景没有真实镜头畸变、卷帘快门、运动模糊、真实检测器。这些数字证明坐标/符号/单位/误差量级自洽，**不是实机验收**。

---

## 10. 未验证 / 已知风险

| 风险/缺口 | 现状 | 影响 | 下一步 |
|---|---|---|---|
| 相机未到 | 无实机图像 | 所有真实精度/同步结论不成立 | 到货后先跑 day1 基线 |
| Picamera2 双实例 | 代码完成，未跑 | 可能遇到 CSI 排线/相机枚举/`SensorTimestamp` 问题 | `rpicam-hello --list-cameras`，再跑 `timestamp_baseline` |
| 时间戳差 ≠ 曝光同步 | 只有仿真 | 时间戳好看但运动抖动大 | 用运动纹理板的 SGBM 质心抖动判定 |
| 真实标定质量 | 只有合成/录制 | 畸变、装配刚性、温度漂移未知 | 15–25 对视图，查 RMS/极线 |
| Pi CPU/散热 | 未测 | 编码+推流可能满载/降频 | 全开时看 `vcgencmd`/`top`，锁定 governor |
| 目标 PC 帧率 | 未测 | 3.75 fps 是 Pi 数字 | RTX 3060 上跑同一 demo |
| ROS 2 运行期 | 未装 ROS 2 | 话题/TF/QoS 可能有坑 | 在 Ubuntu 22.04 + Humble 跑 node + Foxglove |
| YOLO 精度 | 权重缺失，彩色占位 | 物体框与类别不可用 | 导出 `yolov8n.engine`，用真实数据评估 |
| 学习型匹配器 | 未实测 | 无法判断是否值得保留 | 按 `docs/stage3.md` 三项验收 |
| TSDF | 缺 Open3D/位姿 | 不能作为主线 | 先确定转台/机械臂位姿来源，再离线试 |
| 相机移动/卷帘 | 未测 | 运动时深度抖动/几何畸变 | `tools/motion_jitter.py` 量化速度边界 |

---

## 11. 下一步优先级（给下一个会话）

### P0：相机到货后的第一天

1. `rpicam-hello --list-cameras` 确认两个 IMX219；
2. 跑 `pi.tools.timestamp_baseline --frames 100`（不推流、不加控制），记录中位数/IQR；
3. 再跑 `--with-control`，比较 P 环前后；
4. 用慢速运动纹理板记录 SGBM 质心深度抖动（时间戳差好看不代表曝光同步）；
5. 拍 15–25 对棋盘格，跑真实标定，检查 RMS/极线；
6. 0.3/0.8/1.5 m 已知距离目标，输出无/有地面约束误差表；
7. 记录 Pi 全开 CPU/温度/是否降频；记录端到端延迟。

### P1：机器人接入

1. 在 Ubuntu 22.04 + ROS 2 Humble 安装 `foxglove_bridge`、`rtabmap_ros`、Nav2；
2. 跑 `pc.ros2.pointcloud_node`，确认 `/stereo/points`、`/stereo/depth`、`camera_info`、`/stereo/objects`；
3. Foxglove：RGB 上色、关 normals、图像 5–10 Hz、点云 decay 0；
4. RTAB-Map：`MemoryThr=300`、`DetectionRate=1.0`，确认图像/深度/camera_info 同步；
5. Nav2：ObstacleLayer 订阅 `/stereo/points`，确认代价地图可跑；
6. 目标 PC 上测真实帧率/CPU/内存，决定 `scale`/ROI/96 视差档位。

### P2：精度与扩展

1. 真实数据上重跑压缩量化，确认 H.264 码率；
2. 有 YOLO 权重后评估检测对物体 3D 的影响；
3. 阶段三学习型 A/B（严格三项验收，不达标就回退）；
4. 运动边界测试（速度上界）；
5. TSDF 仅在位姿来源确定后做离线扩展。

---

## 12. 关键文件索引

| 想改什么 | 先看 |
|---|---|
| 配置/参数校验 | `shared/config.py`、`configs/default.yaml`、`configs/pc.yaml` |
| 帧协议/配对 | `shared/messages.py` |
| 合成场景 | `shared/synthetic.py` |
| Pi 采集/同步/推流 | `pi/service.py`、`pi/capture/`、`pi/sync/`、`pi/stream/` |
| 标定 | `pc/calib/calibrate.py`、`pc/calib/cli.py` |
| 校正 | `pc/rectify/rectifier.py` |
| SGBM/ROI/scale | `pc/stereo/sgbm.py` |
| 融合规则 | `pc/fuse/router.py` |
| 物体/地面 | `pc/objects/aggregator.py` |
| 点云/体素 | `pc/depth/pointcloud.py` |
| 检测 | `pc/detect/` |
| 管线编排 | `pc/pipeline.py` |
| 在线/回放 | `pc/receive/live.py`、`pc/receive/subscriber.py` |
| ROS 2 | `pc/ros2/pointcloud_node.py`、`ros2/` |
| 报告/指标 | `pc/validate/`、`tools/run_demo.py` |
| 验收/风险 | `docs/acceptance.md` |
| 接口契约 | `docs/interfaces.md` |
| 联调排障 | `docs/runbook.md` |
| ROS/Foxglove | `docs/ros2_foxglove.md` |
| 阶段三 | `docs/stage3.md` |

---

## 13. 踩过的坑（不要重复）

1. **同步 P 环不能累加周期**：早期 `p += kp*error` 会发散；现在每步从 `base_period_us` + `step` 重算，`step=-kp*error`。
2. **棋盘格多画边**：最初 `border_squares=1` 导致角点多出来，检测全失败；现在 `(cols+1)×(rows+1)` 方格 + 半格白边 quiet zone，SB/经典都能检出 54 角点。
3. **k3 过拟合**：合成无畸变数据上放开 k3 会拟合出 `k3≈0.24`，让 `alpha=0` 裁掉视场并污染 `fB`；默认 `CALIB_FIX_K3` + `CALIB_ZERO_TANGENT_DIST`。
4. **`record_pairs` 属性名**：曾用 `rp.pair.ts_left_ns`，正确是 `rp.pair.left.sensor_ts_ns`；已在回环中修复。
5. **`scale<1` 与 `min_disparity!=0`**：SGBM 内部 minDisparity 不自动缩放，配置层直接禁止。
6. **体素性能**：`np.unique(axis=0)` 在 30k 点约 43 ms；改 `lexsort+reduceat` 后约 17 ms。
7. **PUB/SUB 慢加入**：ZMQ PUB 不保留历史，PC 订阅端必须**先启动**再让 Pi 发；否则前几帧丢失是正常的。
8. **路径已上移**：项目从 `深度/stereo-depth/` 上移到 `深度/`；旧会话里的 `stereo-depth/...` 路径已失效，除非是历史文档。
9. **不要把合成数字当实机验收**：`runs/demo` 报告已显式标注“合成、Pi 5、非目标 PC”。
10. **不要用学习型模型冒充 SGBM**：缺模型时 `LearnedMatcherUnavailable` 显式报错，不静默降级。

---

## 14. 给下一个会话的建议开场

> 先读 `docs/PROJECT_STATE.md`（本文件）和 `README.md`。
> 当前主线是机器人双目感知，不是 3D 建模；默认纯 SGBM，96 视差，0.03 m 体素，ROS 2 发布代码已写但本机没装 ROS 2。
> 相机未到，不要假设任何实机数字；先跑 75 个测试和 `tools.run_demo`，再从 `docs/acceptance.md` 的“尚未完成”清单继续。
> 改代码前先看 `docs/interfaces.md` 的数据契约；改融合规则前先看 `docs/stage3.md` 的“SGBM 优先、不平均”。

---

## 15. 一句话交接

**这是一个能跑、能测、能出报告的实时双目感知基线：合成/回放链路完整，机器人 ROS 2 接入代码就绪；实机相机、真实标定/同步、目标 PC 帧率、ROS 2 运行期和学习型匹配器仍未验证。**
