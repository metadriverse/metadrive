from setuptools import setup
from glob import glob

package_name = 'metadrive_example_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, glob('launch/*.launch.py')),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zhizheng Liu',
    maintainer_email='zhizheng@cs.ucla.edu',
    description='example ros2 bridge for metadrive',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_bridge = metadrive_example_bridge.camera_bridge:main',
            'lidar_bridge = metadrive_example_bridge.lidar_bridge:main',
            'obj_bridge = metadrive_example_bridge.obj_bridge:main'
        ],
    },
)
