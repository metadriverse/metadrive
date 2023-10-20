import ctypes
import struct
import sys

import numpy as np
import rclpy
import zmq
from builtin_interfaces.msg import Time
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

_DATATYPES = {}
_DATATYPES[PointField.INT8] = ('b', 1)
_DATATYPES[PointField.UINT8] = ('B', 1)
_DATATYPES[PointField.INT16] = ('h', 2)
_DATATYPES[PointField.UINT16] = ('H', 2)
_DATATYPES[PointField.INT32] = ('i', 4)
_DATATYPES[PointField.UINT32] = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)


class LidarPublisher(Node):

    def __init__(self):
        super().__init__('lidar_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, 'metadrive/lidar', qos_profile=10)
        timer_period = 0.05  # seconds
        context = zmq.Context().instance()
        self.socket = context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.set_hwm(5)
        self.socket.connect("ipc:///tmp/lidar")  # configured in gamerunner.py
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def _get_struct_fmt(self, is_bigendian, fields, field_names=None):
        fmt = '>' if is_bigendian else '<'

        offset = 0
        for field in (f for f in sorted(fields, key=lambda f: f.offset)
                      if field_names is None or f.name in field_names):
            if offset < field.offset:
                fmt += 'x' * (field.offset - offset)
                offset = field.offset
            if field.datatype not in _DATATYPES:
                print('Skipping unknown PointField datatype [{}]' % field.datatype, file=sys.stderr)
            else:
                datatype_fmt, datatype_length = _DATATYPES[field.datatype]
                fmt += field.count * datatype_fmt
                offset += field.count * datatype_length

        return fmt

    def timer_callback(self):
        lidar_buffer_msg = self.socket.recv()
        # read the first 32bit int as W
        # second as H to handle different resolutions
        length = struct.unpack('i', lidar_buffer_msg[:4])[0]
        # read the rest of the message as the image buffer
        lidar_buffer = lidar_buffer_msg[4:]
        lidar = np.frombuffer(lidar_buffer, dtype=np.float32)
        lidar = lidar.reshape(-1, 3)
        assert lidar.shape[0] == length
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        header = self.get_msg_header()
        cloud_struct = struct.Struct(self._get_struct_fmt(False, fields))
        buff = ctypes.create_string_buffer(cloud_struct.size * len(lidar))
        point_step, pack_into = cloud_struct.size, cloud_struct.pack_into
        offset = 0
        for p in lidar:
            pack_into(buff, offset, *p)
            offset += point_step

        point_cloud_msg = PointCloud2(
            header=header,
            height=1,
            width=len(lidar),
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=cloud_struct.size,
            row_step=cloud_struct.size * len(lidar),
            data=buff.raw
        )
        self.publisher_.publish(point_cloud_msg)
        self.i += 1

    def get_msg_header(self):
        """
        Get a filled ROS message header
        :return: ROS message header
        :rtype: std_msgs.msg.Header
        """
        header = Header()
        header.frame_id = "map"
        t = self.get_clock().now()
        t = t.seconds_nanoseconds()
        time = Time()
        time.sec = t[0]
        time.nanosec = t[1]
        header.stamp = time
        return header


def main(args=None):
    rclpy.init(args=args)

    camera_publisher = LidarPublisher()

    rclpy.spin(camera_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    camera_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
