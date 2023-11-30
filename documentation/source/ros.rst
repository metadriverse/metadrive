#####
ROS2
#####

.. raw:: html

    <div style="text-align: center;">
    <iframe width="640" height="400" src="https://www.youtube.com/embed/WWwdnURnOBM?si=aRKqMHQzmblrzhr4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>

ROS is widely used by the robotics community to design and verify planning algorithms.
We provide an example bridge for connecting MetaDrive and ROS2 using zmq sockets, which publishes messages for camera
images, LiDAR point clouds, and object 3D bounding boxes.

Installation
================

To install the bridge, first [install ROS2 **humble**](https://docs.ros.org/en/humble/Installation.html) and follow the scripts
below::

    cd bridges/ros_bridge

    # activate env, ${ROS_DISTRO} is something like foxy, iron, humble
    source /opt/ros/${ROS_DISTRO}/setup.bash
    # You may need to run init, if you are installing ros2 for the first time
    sudo rosdep init
    # zmq should be installed with system python interpreter
    pip install pyzmq
    rosdep update
    rosdep install --from-paths src --ignore-src -y

    # build
    colcon build
    source install/setup.bash

Usage
======

To launch the bridge, run the following code::

    # Terminal 1, launch ROS publishers
    ros2 launch metadrive_example_bridge metadrive_example_bridge.launch.py
    # Terminal 2, launch socket server
    python ros_socket_server.py


Known Issues
==================

* If you are using the `conda`, it is very likely that the interpreter will not match the system interpreter and will be incompatible with ROS 2 binaries.
* The ROS bridge is tested under **ROS2 humble**. Some packages may fail to be installed if your ROS is in other versions.

