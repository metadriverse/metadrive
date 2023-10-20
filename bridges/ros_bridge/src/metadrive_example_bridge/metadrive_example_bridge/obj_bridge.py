import struct

import numpy as np
import rclpy
import zmq
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from rclpy.node import Node
from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox3D, BoundingBox3DArray


class ObjectPublisher(Node):

    def __init__(self):
        super().__init__('obj_publisher')
        self.publisher_ = self.create_publisher(BoundingBox3DArray, 'metadrive/object', qos_profile=10)
        timer_period = 0.05  # seconds
        context = zmq.Context().instance()
        self.socket = context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.set_hwm(5)
        self.socket.connect("ipc:///tmp/obj")  # configured in gamerunner.py
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        obj_buffer_msg = self.socket.recv()
        # read the first 32bit int as W
        # second as H to handle different resolutions
        num_obj = struct.unpack('i', obj_buffer_msg[:4])[0]
        # read the rest of the message as the image buffer
        obj_buffer = obj_buffer_msg[4:]
        obj_infos = np.frombuffer(obj_buffer, dtype=np.float32).reshape((num_obj, 6))
        bboxes = []
        for obj_info in obj_infos:
            obj_info = obj_info.tolist()
            size = Vector3(x=obj_info[3], y=obj_info[4], z=obj_info[5])
            pos = Point(x=obj_info[0], y=obj_info[1], z=0.0)
            rot = Quaternion(x=0.0, y=0.0, z=np.sin(obj_info[2] / 2.0), w=np.cos(obj_info[2] / 2.0))
            pose = Pose(position=pos, orientation=rot)
            bbox = BoundingBox3D(center=pose, size=size)
            bboxes.append(bbox)
        obj_msg = BoundingBox3DArray(header=self.get_msg_header(), boxes=bboxes)
        obj_msg.header = self.get_msg_header()
        self.publisher_.publish(obj_msg)
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

    obj_publisher = ObjectPublisher()

    rclpy.spin(obj_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    obj_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
