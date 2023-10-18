# Connecting MetaDrive Simulator with ROS2

We provide an example bridge for connecting MetaDrive and ROS2 using zmq sockets, which publishes messages for camera
images, LiDAR point clouds, and object 3D bounding boxes.

## Installation

To install the bridge, first [install ROS2](https://docs.ros.org/en/humble/Installation.html) and follow the scripts
below:

```bash
# update dependencies
pip install -e .[ros] 
cd bridges/ros_bridge
rosdep update
rosdep install --from-paths src --ignore-src -y
# install
colcon build
source install/setup.bash
```

## Usage

```bash
# Terminal 1, launch socket server
python bridges/utils/ros_socket_server.py 
# Terminal 2, launch ROS publishers
ros2 launch metadrive_example_bridge metadrive_example_bridge.launch.py
```

[Demo Video](https://www.youtube.com/watch?v=WWwdnURnOBM&list=TLGGdRGbC4RGzhAxNzEwMjAyMw)