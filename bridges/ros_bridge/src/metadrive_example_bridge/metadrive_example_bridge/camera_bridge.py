import struct

import cv2
import numpy as np
import rclpy
import zmq
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header


class CameraPublisher(Node):
    cv_bridge = CvBridge()

    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'metadrive/image', qos_profile=10)
        timer_period = 0.05  # seconds
        context = zmq.Context().instance()
        self.socket = context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.set_hwm(5)
        self.socket.connect("ipc:///tmp/rgb_camera")  # configured in gamerunner.py
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        image_buffer_msg = self.socket.recv()
        # read the first 32bit int as W
        # second as H to handle different resolutions
        W, H = struct.unpack('ii', image_buffer_msg[:8])
        # read the rest of the message as the image buffer
        image_buffer = image_buffer_msg[8:]
        image = np.frombuffer(image_buffer, dtype=np.uint8).reshape((H, W, 3))
        image = image[:, :, :3]  # remove the alpha channel
        image = cv2.resize(image, (W, H))
        img_msg = CameraPublisher.cv_bridge.cv2_to_imgmsg(image)
        img_msg.header = self.get_msg_header()
        self.publisher_.publish(img_msg)
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

    camera_publisher = CameraPublisher()

    rclpy.spin(camera_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    camera_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
