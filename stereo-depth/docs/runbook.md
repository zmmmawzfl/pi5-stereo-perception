# Pi ↔ PC 联调运行手册

按顺序做四件事：**最短链路 → 标定 → 在线处理 → 离线复算**。
每一步都有明确的"看到什么才算通"。

> 本手册的命令已在同一台机器上用 `pi.service`（synthetic）→ ZMQ → `pc.receive.record_pairs`
> / `pc.receive.live` 跑通；跨两台机器的真实网线/交换机环境尚未实测。

## 0. 一次性准备

**Pi：**

```bash
hostname -I                     # 记下 IP，例如 192.168.1.10
ip -4 addr | grep inet
sudo apt install -y python3-opencv python3-picamera2 python3-zmq
python -c "import cv2, picamera2, zmq; print(cv2.__version__)"
```

**PC：**

```bash
ping <PI_IP>                    # 先确认二层可达
python -c "import cv2, zmq; print(cv2.__version__)"
```

**端口：** Pi 绑定 `5555`（视频帧）和 `5556`（时间戳旁路）。两端防火墙都要放行 TCP 5555/5556。
PC 是客户端（SUB connect），不需要开放入站。

## 1. 最短链路：先不要标定

目的：确认"Pi 能发、PC 能按 pair_id 收"。

**Pi：**

```bash
cd stereo-depth
python -m pi.service --config configs/pi.yaml \
    --record-root data/recordings/smoke --frames 300
```

**PC（建议先启动它，SUB 连接后 Pi 再发，避免 PUB/SUB 慢加入丢首帧）：**

```bash
cd stereo-depth
python -m pc.receive.record_pairs --endpoint tcp://<PI_IP>:5555 \
    --out data/recordings/smoke_pc --max-pairs 100 --idle-timeout 30
```

看到什么才算通：

- PC 端 `summary.json` 里 `emitted > 0`，`orphans ≈ 0`，`duplicates ≈ 0`；
- `ts_diff_ms.median` 有具体数字；
- `data/recordings/smoke_pc/pairs.jsonl` 每行左右文件名都存在，打开看是同一时刻的左右图。

如果 `emitted=0`：先查 Pi 是否真的 bind 成功（`ss -ltnp | grep 5555`）、IP/端口、防火墙。
如果 `orphans` 很多：左右 `pair_id` 没对齐，或 PC 启动太晚错过了同一对的另一侧。

## 2. 标定

### 2a. 采棋盘格（Pi，录成标准录制目录）

把棋盘格放在 0.4–0.8 m，缓慢平移/倾斜，覆盖画面四角和不同姿态，录 100–200 对：

```bash
python -m pi.tools.record_pairs --out data/recordings/calib_board --frames 150
```

### 2b. 在 PC 上直接标定（无需手工分左右目录）

```bash
python -m pc.calib.cli calibrate --replay-root data/recordings/calib_board \
    --pattern 9x6 --square-mm 30.0 --out data/calib/stereo_calib.npz
python -m pc.calib.cli inspect --calib data/calib/stereo_calib.npz
```

没有棋盘格的帧会被自动跳过（日志会列出下标），所以多录一些没关系。
方格尺寸用卡尺实测，**不要用标称值**。

通过标准：

- 单目 RMS < 0.3 px，双目 RMS < 0.5 px；
- 极线 `|dy|` 中位数接近 0；
- `B` 与卷尺量到的基线接近（差 1% 以内）；
- 不达标先重采，不要改算法。

> 也可以用 `--left data/calib/left --right data/calib/right` 读两个目录（文件名排序一一对应）。

## 3. 在线联动

**Pi**（保持运行，默认推流 + 同步环）：

```bash
python -m pi.service --config configs/pi.yaml
```

**PC**（另开终端）：

```bash
python -m pc.receive.live --endpoint tcp://<PI_IP>:5555 \
    --calib data/calib/stereo_calib.npz \
    --out runs/live --save-every 30
```

每帧输出一行 `[live] pair=... objects=... total=...ms`，并在 `runs/live/` 写：

- `objects.jsonl`：每帧物体列表（`center_3d`、`bbox_2d`、`depth_source`、`valid_ratio`）；
- `overlay_00030.png`：检测框 + 质心 + 深度；
- `source_00030.png`：每像素来源（绿=SGBM，蓝=学习型）；
- `summary.json`：帧数、fps、各阶段耗时。

先看 overlay 里的框和深度是否合理，再看 `objects.jsonl` 的数字。
**第一次联调不要求精度**，先要求"稳定出物体、不漏帧、深度不跳飞"。

## 4. 离线复算（同一份数据反复调参）

```bash
python -m pc.receive.live --replay-root data/recordings/smoke_pc \
    --calib data/calib/stereo_calib.npz --out runs/replay

python -m tools.compression_report --replay-root data/recordings/smoke_pc \
    --calib data/calib/stereo_calib.npz --out runs/compression --max-frames 200

python -m tools.motion_jitter --replay-root data/recordings/static \
    --calib data/calib/stereo_calib.npz --out runs/jitter_static.json
```

## 5. 端口与常见故障

| 现象 | 检查 |
|---|---|
| PC 一帧都收不到 | Pi `ss -ltnp \| grep 5555`；`ping <PI_IP>`；防火墙；endpoint 是否写成 Pi 的 IP 而不是 127.0.0.1 |
| 只有 orphan | PC 启动晚于 Pi；`--stale` 太小；左右 pair_id 生成逻辑 |
| 有 pair 但解码失败 | JPEG 载荷损坏；检查 MTU/网络；先降 `jpeg_quality` |
| 物体框对但深度离谱 | 是否用了 `pc.calib.cli synthetic` 的假内参；`--calib` 是否指向实机标定 |
| 框的位置整体偏移 | 标定分辨率与运行分辨率不一致；看 pipeline 启动告警 |
| 只有右目/左目 | Pi 侧 `rpicam-hello --list-cameras`；CSI 排线 |
| 在线 fps 低 | PC 端 SGBM 参数、检测器；先 `--detect` 换成 color 排除 YOLO |

## 6. 第一次联调的验收清单

- [ ] 两个相机都能单独出图；
- [ ] PC 收到成对帧，`orphans=0`；
- [ ] 标定 RMS 达标，极线 `|dy|` 接近 0；
- [ ] 在线跑 100 对不崩，overlay 正常；
- [ ] 静止目标深度抖动 < 基线 1–2%；
- [ ] 记录 Pi 全开时的 CPU/温度/是否降频；
- [ ] 记录一段已知距离（0.5 m / 1 m）做绝对误差表。
