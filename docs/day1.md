# 第一天 / 阶段一实机清单

目标：先建立"正确的数据长什么样"的基线，再让系统变聪明。第一天**不推流、不加控制**，只看 `SensorTimestamp`。

## 0. 硬件与系统自检

```bash
# 两块 IMX219 必须都出现；只有一个 → 先查排线方向/接口，不要改代码
rpicam-hello --list-cameras

# 散热与降频
vcgencmd measure_temp
vcgencmd get_throttled        # 必须是 0x0

# governor
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo performance | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
```

`/boot/firmware/config.txt` 中双 IMX219 的典型写法（Pi 5 两个 CSI 口）：

```ini
camera_auto_detect=0
dtoverlay=imx219,cam0
dtoverlay=imx219,cam1
```

改完重启，再跑一次 `rpicam-hello --list-cameras`。

## 1. 采集 / 同步基线（Pi）

```bash
cd stereo-depth

# 第一遍：不推流、不加控制，100 帧，记录"出厂读数"
python -m pi.tools.timestamp_baseline --frames 100 --out runs/day1_raw.json

# 第二遍：开 P 环（Kp=0.05），看相位差是否收敛
python -m pi.tools.timestamp_baseline --frames 100 --with-control --out runs/day1_locked.json
```

看什么：

- `median` 与 `P25/P75`（IQR）是否稳定；`max-min` 是否出现周期性跳变；
- 加控制后中位数是否靠近 0、IQR 是否变小；
- 如果时间戳差很小但后面运动抖动很大，说明**曝光并不同步**，以运动抖动为准（见第 3 节）。

## 2. 推流 + PC 接收

```bash
# Pi：采集 + 同步 + ZMQ 推流，同时无损落盘 300 对
python -m pi.service --config configs/pi.yaml \
    --record-root data/recordings/day1_pi --frames 300

# PC：最小接收器，按 pair_id 落盘
python -m pc.receive.record_pairs --endpoint tcp://<PI_IP>:5555 \
    --out data/recordings/day1_pc --max-pairs 100
```

验收：

- PC 的 `summary.json` 中 `orphans=0`、`duplicates=0`（或极少）；
- `emitted` 与 Pi 的 pair 数一致（允许少量丢包）；
- 手工抽查 `pairs.jsonl`：同一 `pair_id` 的左右帧内容确实是一对，且 `ts_diff_ms` 与 Pi 日志一致。

## 3. 运动抖动基线（比时间戳更硬的验收）

用已知慢速运动的纹理板（手动平移或小转台），分别录"静止"和"匀速慢移"两段，各 ≥ 60 对。
然后 PC 上：

```bash
python -m tools.motion_jitter --replay-root data/recordings/static \
    --calib data/calib/stereo_calib.npz --out runs/jitter_static.json

python -m tools.motion_jitter --replay-root data/recordings/moving \
    --calib data/calib/stereo_calib.npz --out runs/jitter_moving.json \
    --baseline-json runs/jitter_static.json
```

输出 `depth_std_mm`、`frame_to_frame_median_abs_mm` 与相对增量。这个增量就是卷帘快门 + 同步残差在你场景里的**实际可用速度边界**。

> 没有标定文件时先跑 `python -m pc.calib.cli synthetic --out data/calib/stereo_calib.npz` 只验证链路；
> 但这组内参不能用于实机测距。

## 4. 压缩伪影量化

Pi 端已经无损落盘（PNG）。PC 上：

```bash
python -m tools.compression_report --replay-root data/recordings/day1_pi \
    --calib data/calib/stereo_calib.npz --out runs/compression --max-frames 200
```

看 `runs/compression/compression.md`：

- `有效像素/参考` 掉了多少；
- `弱纹理视差误差中位数`；
- 若有物体掩码/真值，`最大物体相对深度偏移` 是否 < 2%。

不达标时按顺序处理：提高码率 → 降低压缩强度 → 关键区域传原始帧（`stream.raw_max_pairs`）。

## 5. 标定

1. 640×480 下采 15–25 对棋盘格，覆盖画面四角与不同倾角；
2. 左右目录按文件名排序一一对应；
3. 方格尺寸用卡尺**实测**，单位 mm。

```bash
python -m pc.calib.cli calibrate --left data/calib/left --right data/calib/right \
    --pattern 9x6 --square-mm 30.0 --out data/calib/stereo_calib.npz
python -m pc.calib.cli inspect --calib data/calib/stereo_calib.npz
```

验收：单目 RMS < 0.3 px、双目 RMS < 0.5 px、极线 `|dy|` 中位数接近 0。
不达标先重采（角点模糊/反光/覆盖不足），不要急着改算法。

## 6. 已知距离验证

把目标放在 0.3 / 0.8 / 1.5 m（或实际量程），每个距离录 ≥ 30 对，各跑一次：

```bash
python -m pc.receive.live --replay-root data/recordings/dist_08m \
    --calib data/calib/stereo_calib.npz --out runs/dist_08m
```

`runs/dist_08m/objects.jsonl` 每行给出 `center_3d`、`depth_source`、`valid_ratio`。
用卷尺读到的距离做真值，填进验收表（中位数误差、P90、覆盖率、帧率）。

## 7. 排障

| 现象 | 优先检查 |
|---|---|
| 只识别一个相机 | CSI 排线方向/接口、`dtoverlay` 是否 cam0+cam1、`camera_auto_detect` |
| 帧率不稳、时间戳跳变 | governor、温度/降频、编码线程、SD 卡写满 |
| PC 收不到帧 | Pi 是否 bind 0.0.0.0、防火墙、同网段、端口 5555/5556 |
| 配对全是 orphan | 左右 `pair_id` 没对齐（Pi 侧生成逻辑）、`stale_s` 太小 |
| 时间戳差好看但运动抖动大 | 曝光不同步 / 卷帘快门；收紧准静态边界，别只看时间戳 |
| 标定 RMS 大 | 棋盘不平、角点模糊、视图覆盖不足、格子尺寸量错 |
| 深度整体偏大/偏小 | 标定 f 或 B 偏差、视差是否除了 16、符号是否反 |

## 8. 阶段一需要带回的数字

- 时间戳差：median / IQR / min / max（有控制、无控制各一组）；
- 运动纹理板的深度抖动（静止 vs 运动）；
- 压缩前后：有效像素比、弱像素视差误差、物体深度偏移；
- 采集+同步+编码+推流全开时：每核 CPU、温度、是否降频、实际 fps。
