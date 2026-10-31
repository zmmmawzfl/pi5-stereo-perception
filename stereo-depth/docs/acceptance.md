# 交付物与验收映射

## 1. 交付物清单

| # | 交付物 | 代码/文件 | 当前状态 |
|---|---|---|---|
| 1 | Pi 5 采集 + 同步 + 推流服务 | `pi/service.py`、`pi/capture/`、`pi/sync/`、`pi/stream/` | 完成；实机未测 |
| 2 | 时间戳配对分布 + 压缩伪影量化报告 | `pi/tools/timestamp_baseline.py`、`tools/compression_report.py` | 工具完成；实机数据待采 |
| 3 | 双目标定参数 + 校正映射 + 极线检查 | `pc/calib/`、`pc/rectify/`、`data/calib/stereo_calib.npz` | 完成（合成验证） |
| 4 | PC 完整处理管线 | `pc/pipeline.py`、`pc/receive/live.py` | 完成 |
| 5 | 物体级 3D JSON + 点云 PLY | `pc/objects/`、`pc/depth/`、`runs/demo/objects.json`、`cloud.ply` | 完成 |
| 6 | 已知距离验证报告 | `tools/run_demo.py` 报告；实机用 `pc.receive.live` + 真值 | 合成完成；实机待测 |
| 7 | 运动边界报告 | `tools/motion_jitter.py` | 工具完成；实机待测 |
| 8 | 延伸：TSDF 网格 | `pc/tsdf/fuse_offline.py` | 完成（需 Open3D + 位姿） |

## 2. 验收指标与当前证据

| 指标 | 目标 | 当前（合成/Pi 5） | 结论 |
|---|---|---|---|
| 端到端帧率 | 25–30 fps（Pi 推流） | 推流链路未在实机跑；PC 处理约 370 ms/帧（Pi 5 本机） | 待测 |
| 时间戳差 | 中位数/IQR 记录在案 | 仿真 12.8 ms → 0.001 ms | 控制环验证通过，实机待测 |
| 运动纹理板深度抖动 | 记录在案 | 工具就绪 | 待测 |
| Pi CPU | 无核心持续满载 | 未测（本机为 PC 端 SGBM） | 待测 |
| 压缩伪影 | 深度偏移 < 1–2% | H.264 2.5 Mbps：0.40%；800 kbps：0.31% | 合成通过 |
| 单目 RMS | < 0.3 px | 0.047 px | 通过（合成） |
| 双目 RMS | < 0.5 px | 0.047 px | 通过（合成） |
| 极线 \|dy\| | 接近 0 | 0.011 px | 通过（合成） |
| 已知距离质心误差 | 记录中位数 | 1.2 mm / 0.10%（中位；单物体最差 6.3 mm） | 合成通过 |
| 地面约束 | 记录前后两组 | \|ΔY\| 7.1→0.7 mm（近目标） | 合成有效 |
| 学习型三项 | 有明确结论 | **未执行** | 待目标 PC |

## 3. 风险 → 代码落点

| 风险 | 触发信号 | 代码里的应对 |
|---|---|---|
| Pi CPU 余量不足 | 编码/推流时核心满载 | `pi/service.py` 统计 fps/period；`build_publisher` 可关流；`stream.transport=none` 先测采集 |
| 时间戳差 ≠ 曝光同步 | 时间戳小但运动抖动大 | `tools/motion_jitter.py` 以深度抖动为判据 |
| 压缩污染弱纹理 | SGBM 有效像素减少 | `pc/validate/compression.py` 量化；`stream.raw_max_pairs` 预留无损旁路 |
| 学习型超延迟/显存 | >30 ms 或挤占检测 | `learned.py` 显式不可用；`fuse/router.py` 可一键回退纯 SGBM |
| 标定不达标 | RMS >0.3/0.5 px、极线差大 | `calibrate.py` 抛错不落盘；`epipolar_dy` 进 meta |
| 位姿来源未定 | 进入 TSDF 才发现 | `pc/tsdf/fuse_offline.py` 强制要求显式 poses；没有位姿不融合 |

## 4. 尚未完成（必须在目标机/实机完成）

1. Picamera2 双实例真机采集与同步环实测；
2. 实际镜头的标定质量（畸变、装配刚性、温度漂移）；
3. 目标 PC 上 YOLOv8 TensorRT 的检测延迟与精度；
4. 阶段三学习型匹配器的延迟/显存/精度三项实测；
5. 真实运动场景的准静态速度上界；
6. TSDF 的位姿来源确认与网格质量评估。

**没有这些数据之前，任何"已达验收"的结论都不成立。**
