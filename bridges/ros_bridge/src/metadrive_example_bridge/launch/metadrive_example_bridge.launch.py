import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    camera_node = Node(package="metadrive_example_bridge", executable="camera_bridge", name='camera_bridge')
    ld.add_action(camera_node)
    lidar_node = Node(package="metadrive_example_bridge", executable="lidar_bridge", name='lidar_bridge')
    ld.add_action(lidar_node)
    obj_node = Node(package="metadrive_example_bridge", executable="obj_bridge", name='obj_bridge')
    ld.add_action(obj_node)

    pkg_dir = get_package_share_directory('metadrive_example_bridge')
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2', arguments=['-d', [os.path.join(pkg_dir, 'default.rviz')]]
    )
    ld.add_action(rviz_node)
    return ld


if __name__ == '__main__':
    generate_launch_description()
