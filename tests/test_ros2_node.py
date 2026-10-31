import unittest

import numpy as np


class _FakePointField:
    FLOAT32 = 7
    UINT8 = 2

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeHeader:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""


class _FakePointCloud2:
    def __init__(self):
        self.header = _FakeHeader()


class _FakeImage:
    def __init__(self):
        self.header = _FakeHeader()


class Ros2NodeTest(unittest.TestCase):
    def test_import_without_ros(self):
        # 没有 ROS 2 的机器上也必须能 import；真正启动时才检查 rclpy
        import pc.ros2.pointcloud_node as node
        self.assertTrue(callable(node.main))
        self.assertTrue(callable(node._pack_pointcloud2))

    def test_help_exits_before_ros_import(self):
        import pc.ros2.pointcloud_node as node
        with self.assertRaises(SystemExit) as ctx:
            node.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_pack_pointcloud2_layout(self):
        import pc.ros2.pointcloud_node as node
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
        cols = np.array([[10, 20, 30], [40, 50, 60]], np.uint8)  # BGR
        msg = node._pack_pointcloud2(_FakePointCloud2, _FakePointField, pts, cols, "camera_left", None)
        self.assertEqual(msg.width, 2)
        self.assertEqual(msg.height, 1)
        self.assertEqual(msg.point_step, 16)
        self.assertEqual(msg.row_step, 32)
        self.assertEqual(len(msg.data), 32)
        self.assertEqual([f.name for f in msg.fields], ["x", "y", "z", "r", "g", "b"])
        self.assertEqual([f.offset for f in msg.fields], [0, 4, 8, 12, 13, 14])
        # 第一点：x,y,z 后是 BGR->RGB
        rec = np.frombuffer(msg.data[:16], dtype=np.uint8)
        self.assertEqual((rec[12], rec[13], rec[14]), (30, 20, 10))

    def test_pack_depth_image(self):
        import pc.ros2.pointcloud_node as node
        depth = np.array([[1.5, np.nan], [0.25, 3.0]], np.float32)
        msg = node._pack_image(_FakeImage, depth, "16UC1", "camera_left", None)
        mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(2, 2)
        self.assertEqual(msg.encoding, "16UC1")
        self.assertEqual(msg.step, 4)
        np.testing.assert_array_equal(mm, np.array([[1500, 0], [250, 3000]], np.uint16))


if __name__ == "__main__":
    unittest.main()
