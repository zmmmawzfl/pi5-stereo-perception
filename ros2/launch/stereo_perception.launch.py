"""启动 PC 端双目感知节点 + 静态 TF（在 stereo-depth 仓库根目录运行）。

    source /opt/ros/humble/setup.bash
    ros2 launch ros2/launch/stereo_perception.launch.py \
        calib:=data/calib/stereo_calib.npz endpoint:=tcp://192.168.1.10:5555

注意：这是"仓库内脚本"式 launch，不依赖 ament 安装；python 必须能找到
stereo-depth 包（在仓库根目录执行，或设置 PYTHONPATH）。
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    calib = LaunchConfiguration("calib")
    endpoint = LaunchConfiguration("endpoint")
    parent = LaunchConfiguration("parent_frame")
    child = LaunchConfiguration("child_frame")
    return LaunchDescription(
        [
            DeclareLaunchArgument("calib", default_value="data/calib/stereo_calib.npz"),
            DeclareLaunchArgument("endpoint", default_value="tcp://127.0.0.1:5555"),
            DeclareLaunchArgument("parent_frame", default_value="world"),
            DeclareLaunchArgument("child_frame", default_value="camera_left"),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=["0", "0", "0", "0", "0", "0", parent, child],
                output="screen",
            ),
            ExecuteProcess(
                cmd=[
                    "python3", "-m", "pc.ros2.pointcloud_node",
                    "--calib", calib, "--endpoint", endpoint,
                ],
                output="screen",
            ),
        ]
    )
