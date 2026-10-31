"""ROS 2 发布节点：ZMQ/回放 → 校正+SGBM → 点云/深度/对象。

主线的最后一公里：把 PC 端管线结果变成机器人/可视化能消费的 ROS 2 话题。

发布：
  /stereo/points          sensor_msgs/PointCloud2  彩色点云（已体素降采样）
  /stereo/left/image_rect sensor_msgs/Image        bgr8 校正左图
  /stereo/depth           sensor_msgs/Image        16UC1，单位 mm，0=无效
  /stereo/objects         std_msgs/String          JSON：2D 框、3D 中心、来源

默认 QoS 为 sensor data（best effort）。点云字段 x/y/z float32 + r/g/b uint8，
point_step=16，Foxglove 3D 面板可直接按 RGB 上色。

用法（需先安装 ROS 2 + rclpy）：
    source /opt/ros/humble/setup.bash
    python -m pc.ros2.pointcloud_node --calib data/calib/stereo_calib.npz \
        --endpoint tcp://192.168.1.10:5555
    # 无相机回放：
    python -m pc.ros2.pointcloud_node --calib ... --replay-root data/recordings/day1
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from pc.calib.calibrate import CalibrationResult
from pc.depth.pointcloud import cloud_from_result
from pc.pipeline import build_pipeline
from pc.receive.subscriber import ZmqPairSource
from shared.config import load_config
from shared.recording import PairReplay


def _iter_pairs(
    endpoint: str, replay_root: Optional[str], max_pairs: Optional[int], idle: float
) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
    """统一的帧对来源：在线 ZMQ 或录制目录回放。"""
    if replay_root:
        for i, (rec, left, right) in enumerate(PairReplay(replay_root)):
            yield rec.pair_id, rec.ts_diff_ns, left, right
            if max_pairs and i + 1 >= max_pairs:
                break
        return
    src = ZmqPairSource(endpoint)
    for rp in src.pairs(max_pairs=max_pairs, idle_timeout_s=idle):
        yield rp.pair_id, rp.ts_diff_ns, rp.left, rp.right


def _pack_pointcloud2(PointCloud2, PointField, points: np.ndarray, colors: np.ndarray,
                      frame_id: str, stamp) -> object:
    """把 (N,3) float32 点 + (N,3) BGR uint8 打包成 PointCloud2。"""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    n = len(pts)
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="r", offset=12, datatype=PointField.UINT8, count=1),
        PointField(name="g", offset=13, datatype=PointField.UINT8, count=1),
        PointField(name="b", offset=14, datatype=PointField.UINT8, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * n
    msg.is_dense = True
    if n:
        data = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("r", "u1"), ("g", "u1"), ("b", "u1"), ("pad", "u1")])
        data["x"], data["y"], data["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        # 输入是 BGR，ROS/Foxglove 按 RGB 解释
        data["r"], data["g"], data["b"] = cols[:, 2], cols[:, 1], cols[:, 0]
        msg.data = data.tobytes()
    else:
        msg.data = b""
    return msg


def _pack_image(Image, img: np.ndarray, encoding: str, frame_id: str, stamp) -> object:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    if encoding == "bgr8":
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.step = int(img.shape[1] * 3)
        msg.data = np.ascontiguousarray(img, dtype=np.uint8).tobytes()
    elif encoding == "16UC1":
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "16UC1"
        msg.step = int(img.shape[1] * 2)
        mm = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        mm = np.clip(mm * 1000.0, 0, 65535).astype(np.uint16)
        msg.data = np.ascontiguousarray(mm).tobytes()
    else:  # pragma: no cover
        raise ValueError(f"不支持的图像编码 {encoding}")
    msg.is_bigendian = 0
    return msg


def _pack_camera_info(CameraInfo, calib: CalibrationResult, frame_id: str, stamp) -> object:
    """校正后左目的 CameraInfo：D 置零，K/P 用校正内参，供 RTAB-Map/ROS 2 使用。"""
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width, msg.height = int(calib.image_size[0]), int(calib.image_size[1])
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    # 校正后图像的内参应与 P1 的 3x3 一致（D=0）；用 K1 会与 P1 不一致
    msg.k = [float(v) for v in np.asarray(calib.P1, dtype=float)[:3, :3].ravel()]
    msg.r = [float(v) for v in np.eye(3, dtype=float).ravel()]
    msg.p = [float(v) for v in np.asarray(calib.P1, dtype=float).ravel()]
    return msg


def run(args: argparse.Namespace) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
        from std_msgs.msg import String
    except ImportError as exc:  # pragma: no cover - 取决于部署环境
        print(
            f"需要 ROS 2 Python 环境（rclpy/sensor_msgs）：{exc}\n"
            "请先 source /opt/ros/<distro>/setup.bash，或在有 ROS 2 的机器上运行。",
            file=sys.stderr,
        )
        return 2

    cfg = load_config(args.config)
    calib = CalibrationResult.load(args.calib)
    pipeline = build_pipeline(cfg, calib)
    frame_id = args.frame_id or cfg.pointcloud.frame_id
    voxel = args.voxel_size if args.voxel_size is not None else cfg.pointcloud.voxel_size_m
    stride = args.stride if args.stride is not None else cfg.pointcloud.stride
    rate_hz = args.publish_hz if args.publish_hz is not None else cfg.pointcloud.publish_rate_hz

    rclpy.init()
    node = Node("stereo_perception")
    qos = qos_profile_sensor_data
    cloud_pub = node.create_publisher(PointCloud2, cfg.pointcloud.cloud_topic, qos)
    image_pub = node.create_publisher(Image, cfg.pointcloud.left_image_topic, qos)
    depth_pub = node.create_publisher(Image, cfg.pointcloud.depth_topic, qos)
    info_pub = node.create_publisher(CameraInfo, cfg.pointcloud.camera_info_topic, qos)
    obj_pub = node.create_publisher(String, cfg.pointcloud.objects_topic, qos)
    node.get_logger().info(
        f"publishing cloud={cfg.pointcloud.cloud_topic} depth={cfg.pointcloud.depth_topic} "
        f"objects={cfg.pointcloud.objects_topic} voxel={voxel}m rate={rate_hz}Hz"
    )

    stop = threading.Event()
    stats = {"pairs": 0, "points": 0, "latency_ms": []}

    def loop() -> None:
        try:
            for pair_id, skew_ns, left, right in _iter_pairs(
                args.endpoint, args.replay_root, args.max_pairs, args.idle_timeout
            ):
                if stop.is_set():
                    break
                t0 = time.perf_counter()
                result = pipeline.process(left, right)
                stamp = node.get_clock().now().to_msg()
                points, colors = cloud_from_result(
                    result, calib, stride=stride,
                    max_depth_m=cfg.pointcloud.max_depth_m, voxel_size_m=voxel,
                )
                cloud_pub.publish(_pack_pointcloud2(PointCloud2, PointField, points, colors, frame_id, stamp))
                if not args.no_images:
                    image_pub.publish(_pack_image(Image, result.rectified_left, "bgr8", frame_id, stamp))
                    depth_pub.publish(_pack_image(Image, result.depth, "16UC1", frame_id, stamp))
                    info_pub.publish(_pack_camera_info(CameraInfo, calib, frame_id, stamp))
                payload = {
                    "pair_id": pair_id,
                    "skew_ms": skew_ns / 1e6,
                    "objects": [o.to_dict() for o in result.objects],
                    "coverage": float(result.fused.coverage),
                    "timings_ms": {k: round(v, 3) for k, v in result.timings_ms.items()},
                }
                obj_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
                stats["pairs"] += 1
                stats["points"] = len(points)
                stats["latency_ms"].append((time.perf_counter() - t0) * 1000.0)
                if stats["pairs"] % 30 == 0:
                    node.get_logger().info(
                        f"pairs={stats['pairs']} points={len(points)} "
                        f"objects={len(result.objects)}"
                    )
                period = 1.0 / rate_hz
                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)
        except Exception as exc:  # pragma: no cover - 线程内异常要能看见
            node.get_logger().error(f"receive loop 停止: {exc}")
        finally:
            stop.set()

    worker = threading.Thread(target=loop, name="stereo-receive", daemon=True)
    worker.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ROS 2 双目感知发布节点（PointCloud2/深度/对象）")
    ap.add_argument("--calib", required=True, help="标定 npz")
    ap.add_argument("--config", default=None)
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    ap.add_argument("--replay-root", default=None)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--idle-timeout", type=float, default=15.0)
    ap.add_argument("--frame-id", default=None)
    ap.add_argument("--voxel-size", type=float, default=None, help="体素边长（米）；0=不降采样")
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--publish-hz", type=float, default=None)
    ap.add_argument("--no-images", action="store_true", help="只发点云/对象，不发图像")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
