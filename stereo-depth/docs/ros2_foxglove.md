# ROS 2 / Foxglove / RTAB-Map / Nav2 接入

目标：把 PC 端管线输出接到机器人感知消费链，而不是做高保真建模。

```
ZMQ/回放 → 校正 → SGBM(96 视差, 可选 scale/ROI) → 融合 → 检测/聚合
        → 深度图 + 彩色点云(体素 0.03 m) → ROS 2 话题 → Foxglove / Nav2 / RTAB-Map
```

## 1. 启动发布节点

```bash
source /opt/ros/humble/setup.bash          # 以实际发行版为准
cd stereo-depth

# 在线（先启动 Pi 推流）
python -m pc.ros2.pointcloud_node \
    --calib data/calib/stereo_calib.npz \
    --endpoint tcp://192.168.1.10:5555

# 无相机回放
python -m pc.ros2.pointcloud_node \
    --calib data/calib/stereo_calib.npz \
    --replay-root data/recordings/day1 --max-pairs 300
```

常用参数：`--voxel-size 0.03`（0=不降采样）、`--stride 2`、`--publish-hz 10`、
`--frame-id camera_left`、`--no-images`。

## 2. 话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/stereo/points` | `sensor_msgs/PointCloud2` | 彩色点云，x/y/z float32 + r/g/b uint8，point_step=16，已体素降采样 |
| `/stereo/left/image_rect` | `sensor_msgs/Image` (bgr8) | 校正左图，供可视化/回环 |
| `/stereo/depth` | `sensor_msgs/Image` (16UC1) | 深度，单位 mm，0=无效 |
| `/stereo/left/camera_info` | `sensor_msgs/CameraInfo` | 校正后内参（D=0），RTAB-Map 需要 |
| `/stereo/objects` | `std_msgs/String` | JSON：2D 框、3D 中心、`depth_source`、覆盖率、耗时 |

QoS：sensor data（best effort）。点云坐标系 `camera_left`：X 右、Y 下、Z 前，米。

静态 TF（没有机器人底盘时）：

```bash
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 world camera_left
```

有底盘时把 `world` 换成 `base_link`，并按实际安装位置给平移/旋转。

## 3. Foxglove

```bash
sudo apt install ros-humble-foxglove-bridge
ros2 run foxglove_bridge foxglove_bridge
# Foxglove Studio 打开 ws://<PC_IP>:8765
```

3D 面板：

- 添加 PointCloud，topic `/stereo/points`，Color mode = RGB，点大小 2–3 px。
- 节点**不发布 normals**，面板里不要开法线显示。
- Decay time 设 0（或 0.5 s 以内）；点云已按 0.03 m 体素降采样，不靠面板 LOD 救性能。
  若你的 Foxglove 版本提供 point cloud LOD/降采样选项，可以开启，但主线不依赖它。
- Fixed frame 选 `camera_left`（或补了 TF 后的 `world`）。

图像面板：

- 用 `/stereo/left/image_rect`，把面板的渲染频率限制在 5–10 Hz（不要 30 Hz 全速刷图）。
- 需要压缩时用 `image_transport` 转成 compressed 话题再显示：

```bash
ros2 run image_transport republish raw compressed \
    --ros-args -r in:=/stereo/left/image_rect -r out/compressed:=/stereo/left/image_rect/compressed
```

## 4. RTAB-Map（可选，低频建图/定位）

参数在 [`ros2/config/rtabmap.yaml`](../ros2/config/rtabmap.yaml)：`Rtabmap/MemoryThr=300`
（200–500 区间）、`Rtabmap/DetectionRate=1.0`、栅格 0.05 m，不追高保真网格。

```bash
ros2 launch rtabmap_ros rtabmap.launch.py \
    rgb_topic:=/stereo/left/image_rect \
    depth_topic:=/stereo/depth \
    camera_info_topic:=/stereo/left/camera_info \
    frame_id:=camera_left \
    approx_sync:=true \
    rtabmap_args:="--delete_db_on_start" \
    params_file:=$PWD/ros2/config/rtabmap.yaml
```

若你的 `rtabmap_ros` 版本没有 `params_file` 参数，改成
`rtabmap_args:="--params $PWD/ros2/config/rtabmap.yaml"`，或把 YAML 的
`rtabmap.ros__parameters` 直接并进 launch 的 Node parameters。

注意：深度图与左图必须同一时刻、同一校正坐标系；节点已按同一 `stamp` 发布。
如果 RTAB-Map 报同步失败，先检查 `camera_info` 的 K/P 与深度图尺寸一致。

## 5. Nav2 代价地图

把 [`ros2/config/nav2_obstacle.yaml`](../ros2/config/nav2_obstacle.yaml) 的 obstacle layer
合并进 local costmap。它直接消费 `/stereo/points`，`max_obstacle_height=2.0`、
`raytrace_max_range=5.0`；点云已经体素化，不要再叠一层高密度滤波。

导航用占据栅格/OctoMap，不用 TSDF 网格；TSDF 只在离线扩展里出现。

## 6. 当前状态

- 节点代码已写好并通过 `py_compile`/import 测试；**本机没有安装 ROS 2，未做运行期验证**。
- Foxglove 的字段布局按常见 point cloud 约定（x/y/z/r/g/b）；若你的版本按 `rgb` 打包，
  在 3D 面板里改用 RGB 字段或加一个 `rgb` 字段即可，不影响点几何。
- 实机接入后需要记录：发布帧率、端到端延迟、CPU/内存（`pc.receive.live` 的
  `resources` 字段已提供同一套进程级指标）。
