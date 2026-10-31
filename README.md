# stereo-depth

**实时双目 3D 重建 / 机器人视觉感知。**
本项目的核心目标是：从双目相机实时视频中恢复深度、生成彩色点云、输出物体级 3D 感知结果，供导航避障、抓取初筛和场景理解使用。它不是完整 3D 建模软件，不输出可编辑 CAD/高保真网格，也不追求 Blender/SolidWorks 级别的几何建模。

主线必须保持在“实时双目感知”上：双目校正 → SGBM（降分辨率/ROI/有限视差）→ 点云/深度 → 体素降采样 → 机器人感知消费。TSDF 仅作为可选扩展，不作为主线依赖。SGBM 是度量锚点；学习型匹配器只作为阶段三的候选补全路径，必须过"误差下降 / 覆盖率扩大 / 帧率达标"三重检验才保留。

## 项目定位 / 非目标

- 目标：实时双目 3D 场景重建、深度估计、点云输出、对象级 3D 中心与导航感知。
- 主用途：机器人视觉感知、避障、抓取初筛、场景理解。
- 非目标：完整 3D 建模、可编辑 CAD、纹理高保真网格、Blender/SolidWorks 导出、逆向工程级重建。
- 结论：点云是感知输出，不是建模产物；TSDF 是扩展，不能成为主线依赖。

> **当前状态（诚实版）**
> - 摄像头尚未到货：Picamera2 实机路径已写好但**未实测**；无相机路径（合成场景/回放）已完整跑通。
> - 阶段三（Fast-FoundationStereo / HITNet）**未执行**：仓库里没有 GPU、模型和任何学习型实测数字。
> - 已实现并测试：消息协议与配对、同步 P 环、标定/校正、SGBM + 左右一致性、SGBM 优先路由、物体聚合、地面约束、压缩伪影量化、体素降采样点云、报告生成。
> - 已实现但**未运行**：ROS 2 发布节点（PointCloud2/深度/CameraInfo/对象 JSON）、RTAB-Map/Nav2 参数模板；本机没有 ROS 2。
> - 本机是 Raspberry Pi 5（aarch64），所以报告里的帧率是 **Pi 上跑 SGBM 的耗时**，不是目标 PC 的指标。
> - Pi→PC 的 ZMQ 链路已在本机回环（`pi.service` → `pc.receive.*`）跑通；跨两台机器的网线/交换机环境、PC 上的 GPU 路径仍待实测。

---

## 快速开始（无需相机）

```bash
cd stereo-depth
python3 -m venv --system-site-packages .venv        # 复用系统 cv2
.venv/bin/pip install -r requirements.txt
# 若环境里没有 cv2：PC 上 pip install -r requirements-pc.txt；
# Pi 上 sudo apt install python3-opencv python3-picamera2

# 75 个测试（含合成标定、SGBM/ROI/降分辨率、体素、路由、地面约束、ZMQ 收发）
.venv/bin/python -m unittest discover -s tests -t . -v

# 端到端 demo + 压缩伪影量化 + 报告
.venv/bin/python -m tools.run_demo --out runs/demo --frames 8 --compression
```

产物在 `runs/demo/`：

| 文件 | 内容 |
|---|---|
| `report.md` | 完整验证报告（标定、误差、覆盖率、帧率、压缩、结论） |
| `summary.json` | 所有原始指标（可复算） |
| `calib/stereo_calib.npz` | 标定参数（K/D/R/T/P1/P2/Q） |
| `objects.json` | 单帧物体级 3D 输出（JSON） |
| `cloud.ply` | 彩色点云（0.03 m 体素降采样；Open3D/MeshLab 可开，不作为 CAD 建模输出） |
| `overlay.png` / `depth.png` / `disparity.png` | 可视化 |
| `disparity_source.png` | 每像素深度来源（绿=SGBM，蓝=学习型，黑=无效） |

---

## 相机到货后的第一天

**Pi 5：**

```bash
rpicam-hello --list-cameras          # 必须看到两个 imx219
python -m pi.tools.timestamp_baseline --frames 100 --out runs/day1_baseline.json
python -m pi.tools.timestamp_baseline --frames 100 --with-control   # 第二遍：开 P 环
```

`timestamp_baseline` 默认**不推流、不加控制**，输出 `SensorTimestamp` 差的中位数、P25/P75、IQR——这是项目的"出厂读数"。

**PC：**

```bash
python -m pc.receive.record_pairs --endpoint tcp://<PI_IP>:5555 \
    --out data/recordings/day1 --max-pairs 100
```

接收端按 `pair_id` 重组，**不依赖"先左后右"**；`summary.json` 会给出接收/配对/重复/孤儿/乱序计数。

细节与排障见 [docs/day1.md](docs/day1.md)。

---

## Pi ↔ PC 联调

完整的四步联调（最短链路 → 标定 → 在线处理 → 离线复算）见
[docs/runbook.md](docs/runbook.md)。最短版本：

```bash
# Pi
python -m pi.service --config configs/pi.yaml --record-root data/recordings/smoke --frames 300

# PC（先启动接收端，再让 Pi 发）
python -m pc.receive.record_pairs --endpoint tcp://<PI_IP>:5555 \
    --out data/recordings/smoke_pc --max-pairs 100

# PC：直接从录制目录标定（自动跳过没有棋盘的帧）
python -m pc.calib.cli calibrate --replay-root data/recordings/calib_board \
    --pattern 9x6 --square-mm 30 --out data/calib/stereo_calib.npz

# PC：在线物体级 3D
python -m pc.receive.live --endpoint tcp://<PI_IP>:5555 \
    --calib data/calib/stereo_calib.npz --out runs/live
```

## ROS 2 / Foxglove / 导航

把管线结果接到机器人消费链（完整说明见 [docs/ros2_foxglove.md](docs/ros2_foxglove.md)）：

```bash
source /opt/ros/humble/setup.bash
python -m pc.ros2.pointcloud_node --calib data/calib/stereo_calib.npz \
    --endpoint tcp://<PI_IP>:5555

# 回放（无相机）
python -m pc.ros2.pointcloud_node --calib data/calib/stereo_calib.npz \
    --replay-root data/recordings/day1 --max-pairs 300
```

| 话题 | 内容 |
|---|---|
| `/stereo/points` | PointCloud2，x/y/z + r/g/b，已体素降采样 |
| `/stereo/depth` | 16UC1 深度图（mm） |
| `/stereo/left/image_rect` | 校正左图 |
| `/stereo/left/camera_info` | 校正内参（RTAB-Map 需要） |
| `/stereo/objects` | JSON：2D 框、3D 中心、depth_source |

Foxglove：`ros2 run foxglove_bridge foxglove_bridge`，3D 面板按 RGB 上色，不显示 normals；
点云已降采样，图像面板限 5–10 Hz。RTAB-Map 用 `ros2/config/rtabmap.yaml`
（`MemoryThr=300`、`DetectionRate=1.0`），Nav2 用 `ros2/config/nav2_obstacle.yaml`。
**这些 ROS 2 路径尚未在装有 ROS 2 的机器上跑过。**

抓取初筛：YOLO/分割给出 2D 框后，用 `pc.depth.pointcloud.points_in_bbox(left, fused, calib, box)`
直接取该框内的彩色点云，中心用 `objects` 的 `center_3d`；6-DoF 抓取位姿是后续工作，
不在当前主线内。

视差档位：96 视差在 640×480、B=83 mm、f≈531 px 下对应最近约 0.46 m；
`sgbm.scale=0.5` 时搜索范围等效 192 全分辨率像素，最近约 0.23 m，但精度下降，按场景取舍。

## 实机推流

```bash
# Pi：采集 + 同步环 + JPEG/ZMQ 推流（可选无损落盘）
python -m pi.service --config configs/pi.yaml --record-root data/recordings/run1

# PC：标定（左右目录内文件名排序一一对应）
python -m pc.calib.cli calibrate --left data/calib/left --right data/calib/right \
    --pattern 9x6 --square-mm 30 --out data/calib/stereo_calib.npz

# PC：在线物体级 3D
python -m pc.receive.live --endpoint tcp://<PI_IP>:5555 \
    --calib data/calib/stereo_calib.npz --out runs/live
```

生产视频可换 `rpi-webrtc-streamer`，时间戳仍走 ZMQ 旁路（`pi/stream/publisher.py` 的 meta 通道）。
H.264 的实际影响用 `pc/validate/compression.py` 量化，不要凭感觉。

---

## 目录

```
stereo-depth/
├── pi/                     # 树莓派端
│   ├── capture/           # Picamera2 双实例 / 合成后端 / 回放
│   ├── sync/              # P 控制环 + 离线同步仿真
│   ├── stream/            # ZMQ PUB（JPEG + 时间戳旁路）、无损落盘
│   ├── config/            # Pi 默认配置入口
│   ├── service.py         # 常驻服务：采集→同步→推流/录制
│   └── tools/             # 第一天基线、录制
├── pc/                     # PC 端
│   ├── receive/           # ZMQ 订阅、pair_id 重组、最小接收器、在线管线
│   ├── calib/             # 单目→双目标定、校正、命令行
│   ├── rectify/           # remap、极线残差自检
│   ├── stereo/            # SGBM(+WLS)、亚像素、学习型适配器
│   ├── fuse/              # 左右一致性 + 纹理门限 + SGBM 优先路由
│   ├── detect/            # YOLOv8(TensorRT/PT) + 彩色连通域占位
│   ├── objects/           # 框内中位数、支撑平面 RANSAC、地面约束
│   ├── depth/             # Z=fB/d、点云 PLY、体素降采样、框内目标点云
│   ├── ros2/              # ROS 2 发布节点（PointCloud2/深度/CameraInfo/对象）
│   ├── tsdf/              # Open3D TSDF（可选扩展，需位姿，不作为主线依赖）
│   ├── validate/          # 指标、压缩量化、报告、可视化
│   └── pipeline.py        # 端到端编排：双目校正 → 视差/深度 → 点云 → 物体级感知
├── shared/                 # 消息、配置、统计、合成场景、录制格式
├── tools/run_demo.py       # 无相机端到端 demo
├── ros2/                   # RTAB-Map / Nav2 参数模板、launch、接入说明
├── tests/                  # 75 个 unittest
├── configs/                # default / pi / pc
└── docs/                   # 接口、第一天、阶段三、验收映射
```

---

## 数据契约（阶段一前定死）

| 模块 | 输出 |
|---|---|
| `pi.capture` | `CapturedPair(left, right, ts_left_ns, ts_right_ns, seq)` |
| `pi.sync` | 从动相机新的 `FrameDurationLimits`（µs） |
| `pi.stream` | ZMQ 多段消息 `[header_json, payload]`，header 含 `pair_id/side/sensor_ts_ns` |
| `pc.receive` | `ReceivedPair(pair, left, right)`，按 `pair_id` 配对 |
| `pc.stereo` | `DisparityResult(disp, valid, right_disp, right_valid, source)` |
| `pc.fuse` | `FusedDisparity(disp, valid, source_map, stats)`，SGBM 优先、不平均 |
| `pc.objects` | `[{class, center_3d, bbox_2d, confidence, depth_source, ...}]` |
| `pc.depth` | `(points Nx3, colors Nx3)`；体素降采样后同一坐标系 |
| `pc.ros2` | PointCloud2（`x,y,z,r,g,b`，point_step=16）、16UC1 深度、CameraInfo、对象 JSON |

坐标与单位：X 右、Y 下、Z 前；米；校正后 `d = u_L - u_R > 0`；深度 `Z = fB/d`，`B=‖T‖`、`f=P1(0,0)` 一律从标定文件读取，**不写死 83 mm**。
完整定义见 [docs/interfaces.md](docs/interfaces.md)。

---

## 已验证的数字（合成场景，Pi 5 本机）

| 项 | 结果 | 说明 |
|---|---|---|
| 标定基线 | 83.025 mm（真值 83.0） | 合成针孔，15 视图 |
| 双目 RMS | 0.047 px | 验收 < 0.5 px |
| 极线 \|dy\| 中位数 | 0.011 px | 验收接近 0 |
| 物体 \|ΔZ\| 中位数 | 1.2 mm（0.10%）；单物体最差 6.3 mm | 0.62 / 0.95 / 1.45 m 三个目标 |
| 物体 3D 误差中位数 | 4.5 mm | 地面约束后 |
| 地面约束 \|ΔY\| | 近目标 7.1 → 0.7 mm | 远目标 5.5 → 0.5 mm |
| 真值几何区覆盖率 | 65.4%（物体像素 49.0%） | SGBM 96 视差 + LR + 纹理门限 |
| 点云降采样 | 20,620 → 3,518 点（17.1%） | 0.03 m 体素 |
| 处理耗时 | ≈266 ms/帧（3.75 fps；SGBM 150 ms） | **Pi 5**；目标 PC 需复测 |
| 压缩（H.264 2.5 Mbps） | 最大物体相对深度偏移 0.41% | 800 kbps 时 0.31%；阈值 2% |
| 同步 P 环（仿真） | 12.8 ms → 0.001 ms | 仅验证控制方向与收敛；实机须重测 |

> 合成场景没有镜头畸变、卷帘快门、运动模糊，也没有真实检测器。这些数字证明"坐标/符号/单位/误差量级"自洽，**不能当作实机验收**。

---

## 阶段进度

| 阶段 | 代码 | 状态 |
|---|---|---|
| 零：环境与基线 | `docs/day1.md` | 待相机 |
| 一：采集同步 + 推流 + 压缩量化 | `pi/`、`pc/validate/compression.py` | 代码完成；实机数据待采 |
| 二：标定 + SGBM 物体级输出 | `pc/calib`、`pc/stereo/sgbm.py`、`pc/objects` | 完成（合成验证） |
| 三：学习型对照 + 路由 | `pc/stereo/learned.py`、`pc/fuse/router.py` | 路由完成；模型实测**未执行** |
| 四：边界测试 + TSDF | `pc/tsdf/fuse_offline.py` | TSDF 需位姿；运动测试待实机 |
| 机器人消费：点云/ROS 2 | `pc/ros2/`、`ros2/` | 代码完成；ROS 2 运行期未验证 |

风险与应对见 [docs/acceptance.md](docs/acceptance.md) 与 `docs/stage3.md`。

---

## 环境

- Python 3.10+（本机 3.13.5）
- OpenCV 4.10（SGBM；WLS 需 contrib，本机可用）
- NumPy 2.2、PyYAML、pyzmq 27
- 可选：ffmpeg（H.264 量化）、ultralytics（YOLO）、onnxruntime-gpu / TensorRT、Open3D

Pi 上不要用 pip 装 opencv（会和系统 `python3-opencv` 冲突），用 `--system-site-packages` 的 venv 或直接系统 Python。
PC 端完整依赖见 `requirements-pc.txt`（含 YOLO/ONNX/Open3D）。
