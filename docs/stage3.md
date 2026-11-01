# 阶段三：学习型匹配器对照与融合路由

**这一阶段只回答一个问题：补全后物体级质心深度误差是否下降、有效覆盖率是否显著扩大、端到端帧率是否还在预算内。**
三项都达标才保留补全路径；否则退回纯 SGBM。这是一次可回退的工程对照，不是架构转向。

## 1. 顺序与理由

1. **Fast-FoundationStereo**（TensorRT FP16，640×480）先实测：参考量级 RTX 3090 上 14–23 ms、约 650 MB；RTX 3060 会更高，以实测为准。
2. 超预算 → **HITNet**（速度优先）。
3. 仍不行 → 降匹配分辨率。
4. 最后才考虑纯 SGBM（默认就保留）。

单目深度先验不进入本阶段：相对尺度不一致会污染绝对测距。

## 2. 代码入口

| 功能 | 文件 |
|---|---|
| 学习型适配器 | `pc/stereo/learned.py` |
| SGBM 优先路由 | `pc/fuse/router.py` |
| 每像素来源图 | `FusedDisparity.source`（0 无效 / 1 SGBM / 2 学习型） |
| 物体级来源统计 | `ObjectEstimate.depth_source` |
| 对照报告 | `tools/run_demo.py` 的 `learned_state` 段落 / 自定义脚本 |

路由规则（不可协商）：

```text
SGBM valid && LR(|dL-dR|<=1) && texture>=阈值   -> 用 SGBM
其余像素 && 学习型 valid                        -> 用学习型
否则                                             -> 无效
不做加权平均。
```

`stats["policy"] == "sgbm_priority_no_average"` 会写进每帧结果，报告中可审计。

## 3. 模型接入

### ONNX（先走这条，便于在 PyTorch/ONNX 之间对齐精度）

> 2026-11-01 已预下载 `models/c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx`（NVIDIA `nvidia/c-fast-foundationstereo`，动态 H/W，约 98.4 MiB）。本机没有 onnxruntime/GPU，**尚未运行**；先把下面的适配器约定与模型实际输入/输出对齐，再谈调参。

```bash
# 假设模型输入为左右两图，输出像素视差（不是归一化视差）
python -m tools.run_demo --config configs/pc.yaml --frames 8 \
    --calib data/calib/stereo_calib.npz
# 在 configs/pc.yaml 中设置：
# sgbm:
#   learned_backend: onnx
#   learned_model: models/ffs_640x480.onnx
#   learned_input_mode: batch      # 或 concat
#   learned_input_size: 640x480
```

适配器约定：

- 输入先 BGR→RGB，按 ImageNet mean/std 归一化（如模型不同，改 `OnnxDisparityMatcher` 的 mean/std）；
- `input_mode=batch`：`2×3×H×W`；`input_mode=concat`：`1×6×H×W`；
- 输出必须是**输入分辨率下的像素视差**（无效 ≤0）；若模型输出半分辨率，把 `input_size` 设为实际输入并在导出时保证视差单位，适配器会按宽度比缩放；
- `valid = 0 < disp < num_disparities`。

缺模型/缺 onnxruntime 会抛 `LearnedMatcherUnavailable`。**不会悄悄退回 SGBM 冒充补全。**

### TensorRT FP16

`TensorRTDisparityMatcher` 只保留了显式报错入口。原因：没有目标 GPU 与引擎文件时写一套"看起来能跑"的 TensorRT 代码，最容易产出无法复现的假延迟。
在 PC 上导出（示例）：

```bash
yolo export model=models/yolov8n.pt format=engine half=True  # 检测（权重已预下载）
trtexec --onnx=models/ffs.onnx --fp16 --saveEngine=models/ffs_fp16.engine \
        --minShapes=left:1x3x480x640,right:1x3x480x640 \
        --optShapes=left:1x3x480x640,right:1x3x480x640 \
        --maxShapes=left:1x3x480x640,right:1x3x480x640
```

然后在 `pc/stereo/learned.py` 里按 `docs` 的接口补齐 `compute()`，或先把 TensorRT 输出接成 ONNX Runtime 的同一接口做 A/B。

## 4. 测量协议

同一段数据、同一标定、同一 SGBM 参数，只改 `learned_backend`：

```bash
# A：纯 SGBM
python -m tools.run_demo --out runs/stage3_sgbm --frames 20 --calib data/calib/stereo_calib.npz

# B：SGBM + 学习型补全（配置里填模型后）
python -m tools.run_demo --out runs/stage3_fused --frames 20 --calib data/calib/stereo_calib.npz
```

每帧必须记录：

- `timings_ms`：sgbm / learned / fuse / detect / aggregate / total；
- 显存峰值（`nvidia-smi --query-gpu=memory.used --format=csv -l 1`）；
- `fused.stats`：`learned_used_px`、`coverage`、`learned_overlap_px`；
- 物体级：`rows_constrained` / `rows_unconstrained` 的 `|ΔZ|`、`err_3d_m`、`valid_ratio`、`depth_source`。

## 5. 验收与决策

| 指标 | 通过条件 | 数据来源 |
|---|---|---|
| 物体级质心深度误差中位数 | 相对纯 SGBM **显著下降**（≥10% 且绝对值 > 噪声） | `rows_constrained` 聚合 |
| 有效深度点覆盖率 | 显著扩大（例如 +5 个百分点以上） | `fused.stats.coverage` |
| 端到端帧率 | 10–15 fps 量级（目标 PC 实测，不用 CREStereo 的 56 ms 做设计依据） | `timings_ms.total` |

三项全过 → 保留补全路径，并在运动场景复测抖动；
误差下降不显著或延迟超预算 → 退回纯 SGBM，把结论写进报告。

**当前仓库状态：未执行。** 模型 ONNX 已预下载（见 `docs/PROJECT_STATE.md` 3.4），但没有 GPU/onnxruntime、没有接入、没有实测数字。`runs/demo` 里的学习型段落会显式写"未启用/未实测"。

## 6. 只补 SGBM 的失效区

学习型输出只允许落在 SGBM 无效/低置信的像素上。验收时抽查 `disparity_source.png`：

- 绿色（SGBM）应覆盖物体主体；
- 蓝色（学习型）只出现在弱纹理、遮挡边界；
- 若蓝色大面积覆盖物体主体，说明 SGBM 有效区被错误拒绝（先查 LR 符号、纹理阈值），不是模型更好。
