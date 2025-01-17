import time

from tqdm import tqdm
import pickle
import numpy as np
import cv2
import gymnasium as gym
import mediapy as media
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw, ImageFont
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.obs.image_obs import ImageObservation
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.obs.observation_base import BaseObservation
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import os

def calculate_fov(intrinsic_matrix, image_size):
    f_x = intrinsic_matrix[0, 0]
    f_y = intrinsic_matrix[1, 1]
    w, h = image_size
    fov_x = 2 * np.arctan(w / (2 * f_x)) * 180 / np.pi
    fov_y = 2 * np.arctan(h / (2 * f_y)) * 180 / np.pi

    return fov_x, fov_y

def calculate_camera_intrinsics(fov_x, width, height):
    """
    Calculate the camera intrinsic matrix given the horizontal field of view (fov_x) and image dimensions.

    Parameters:
    - fov_x: float, horizontal field of view in radians
    - width: int, image width in pixels
    - height: int, image height in pixels

    Returns:
    - K: np.ndarray, the 3x3 intrinsic matrix
    """
    # Calculate focal length in pixels
    f_x = width / (2 * np.tan(fov_x / 2))
    f_y = f_x * (height / width)  # Maintain the aspect ratio

    # Principal point (assume image center)
    c_x = width / 2
    c_y = height / 2

    # Intrinsic matrix
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])
    return K


def angular_sampling(points, h_res=0.2, v_bins=40, v_fov=(-30, 30)):
    def cartesian_to_spherical(points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x)  # 水平角度
        phi = np.arcsin(z / r)  # 垂直角度
        return r, theta, phi

    # 转换点云为球坐标
    _, theta, phi = cartesian_to_spherical(points)

    # 水平角度离散化
    theta_bins = np.round(theta / np.deg2rad(h_res))

    # 垂直角度离散化
    v_min, v_max = np.deg2rad(v_fov[0]), np.deg2rad(v_fov[1])
    v_res = (v_max - v_min) / v_bins
    phi_bins = ((phi - v_min) / v_res).astype(np.int32)

    # 过滤掉超出范围的点
    valid_mask = (phi_bins >= 0) & (phi_bins < v_bins)
    theta_bins = theta_bins[valid_mask]
    phi_bins = phi_bins[valid_mask]
    points = points[valid_mask]

    # 构建唯一性键值对并筛选
    bins = theta_bins * v_bins + phi_bins  # 将 (theta, phi) 映射为唯一的线性索引
    unique_bins, unique_indices = np.unique(bins, return_index=True)

    return points[unique_indices]


def simulate_lidar_from_depth(depth_img, camera_intrinsics, camera_translation, camera_rotation):
    """
    Simulate LiDAR points in the world coordinate system from a depth image.

    Parameters:
        depth_img (np.ndarray): Depth image of shape (H, W, 3), where the last dimension represents RGB or grayscale depth values.
        camera_intrinsics (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        camera_translation (np.ndarray): Translation vector of the camera in world coordinates of shape (3,).
        camera_rotation (np.ndarray): Rotation matrix of the camera in world coordinates of shape (3, 3).

    Returns:
        np.ndarray: LiDAR points in the world coordinate system of shape (N, 3), where N is the number of valid points.
    """
    # Extract the depth channel (assuming it's grayscale or depth is in the R channel)
    # Get image dimensions
    height, width = depth_img.shape

    # Create a grid of pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    uv_coords = np.stack([u, v, np.ones_like(u)], axis=-1)  # Shape: (H, W, 3)

    # Reshape to (H*W, 3) for easier matrix multiplication
    uv_coords = uv_coords.reshape(-1, 3)

    # Invert the camera intrinsic matrix to project pixels to camera coordinates
    K_inv = np.linalg.inv(camera_intrinsics)

    # Compute 3D points in the camera coordinate system
    cam_coords = (K_inv @ uv_coords.T).T  # Shape: (H*W, 3)
    cam_coords *= depth_img.flatten()[:, None]  # Scale by depth

    # Remove invalid points (e.g., depth = 0)
    valid_mask = depth_img.flatten() > 0
    cam_coords = cam_coords[valid_mask]

    # Transform points to the world coordinate system
    world_coords = (camera_rotation @ cam_coords.T).T + camera_translation

    return world_coords


def visualize_point_cloud_and_projection(depth_image, fov, image_size, point_cloud):
    """
    Visualizes the point cloud and its projection onto the image plane.

    Parameters:
        depth_image (np.ndarray): Original depth image (H, W).
        fov (float): Horizontal field of view of the camera (in degrees).
        image_size (tuple): Image size as (width, height).
        point_cloud (o3d.geometry.PointCloud): The generated point cloud.
    """

    h, w = depth_image.shape
    img_width, img_height = image_size

    # Ensure input dimensions match
    assert w == img_width and h == img_height, "Image size does not match depth image dimensions"

    # Extract 3D points
    points = np.asarray(point_cloud.points)
    mask = (
            np.ones((len(points)), dtype=bool)
            & (points[:, 2] > 0)
            & (points[:,2] < 200)
    )
    points = points[mask]
    # Camera intrinsics
    fov_rad = np.deg2rad(fov)
    fx = fy = 0.5 * w / np.tan(fov_rad / 2)
    cx = w / 2
    cy = h / 2

    # Project points to image plane
    x = points[:, 0] / points[:, 2] * fx + cx
    y = points[:, 1] / points[:, 2] * fy + cy
    z = points[:, 2]

    # Normalize depth for color mapping
    z_min, z_max = np.min(z), np.max(z)
    z_normalized = (z - z_min) / (z_max - z_min)
    colors = (plt.cm.viridis(z_normalized)[:, :3] * 200).astype(np.uint8)  # Map depth to colors

    # Filter points within image boundaries
    valid_mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x = x[valid_mask].astype(int)
    y = y[valid_mask].astype(int)
    colors = colors[valid_mask]

    # Create a blank RGB image
    img_vis = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw points using cv2.circle
    for i in range(len(x)):
        cv2.circle(img_vis, (x[i], y[i]), radius=2, color=tuple(colors[i].tolist()), thickness=-1)

    # Visualize the original depth image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Depth Image")
    plt.imshow(depth_image, cmap="gray")
    plt.axis("off")

    # Visualize the projected point cloud on the image
    plt.subplot(1, 2, 2)
    plt.title("Point Cloud Projection with cv2.circle")
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()

def rotation_matrix_to_euler_angles(rotation_matrix):
    # 创建旋转对象
    r = R.from_matrix(rotation_matrix)
    # 提取欧拉角，使用 ZYX 顺序
    roll, pitch, heading = r.as_euler('xyz', degrees=True)
    return heading, pitch, roll
class CameraAndLidarObservation(BaseObservation):
    def __init__(self, config):
        super(CameraAndLidarObservation, self).__init__(config)
        assert config["norm_pixel"] is False
        assert config["stack_size"] == 1
        self.rgb_obs = ImageObservation(config, "rgb_camera", config["norm_pixel"])
        self.depth_obs = ImageObservation(config, "depth_camera", config["norm_pixel"])

    @property
    def observation_space(self):
        os = dict(
            rgb=self.rgb_obs.observation_space,
            depth=self.depth_obs.observation_space,
        )
        return gym.spaces.Dict(os)

    def observe(self, vehicle):
        ret = {}
        # get rgb camera
        rgb_cam = self.engine.get_sensor("rgb_camera").cam
        agent = rgb_cam.getParent()
        camera_to_world = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        rgb_data = {}
        lidar_data = []

        for k,v in camera_params.items():
            #camera_intrinsics = v['intrinsics']
            camera_translation = v['sensor2lidar_translation']
            camera_translation[0], camera_translation[1], camera_translation[2] = -camera_translation[1], camera_translation[0], camera_translation[2]
            camera_rotation = v['sensor2lidar_rotation']@camera_to_world
            h,p,r = rotation_matrix_to_euler_angles(camera_rotation)
            rgb_img = self.rgb_obs.observe(agent, position=camera_translation, hpr=[h,p,r])[..., 0]

            depth_img = self.depth_obs.observe(agent, position=camera_translation, hpr=[h,p,r])[...,0, 0]
            camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
            camera_intrinsics.set_intrinsics(
                width=rgb_img.shape[1],
                height=rgb_img.shape[0],
                fx = intrinsics[0,0],
                fy = intrinsics[1,1],
                cx = intrinsics[0,2],
                cy = intrinsics[1,2]

            )
            depth_image = o3d.geometry.Image(depth_img.astype(np.uint16))
            point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                depth=depth_image,
                intrinsic=camera_intrinsics,
            )
            # o3d.visualization.draw_geometries([point_cloud])
            lidar = simulate_lidar_from_depth(depth_img, intrinsics, camera_translation, v['sensor2lidar_rotation'])
            rgb_data[k] = rgb_img
            lidar_data.append(lidar)
        #lidar_data = np.concatenate(lidar_data, axis=0)
        # import trimesh
        # # 转换为 Trimesh 点云
        # cloud = trimesh.points.PointCloud(lidar_data)
        # # 创建坐标轴
        # axis = trimesh.creation.axis(origin_size=20)  # 坐标轴的原点大小
        # from trimesh.scene import Scene
        # scene = Scene()
        # scene.add_geometry(cloud)  # 添加点云
        # scene.add_geometry(axis)  # 添加坐标轴
        # scene.show()
        ret['camera'] = rgb_data
        ret['lidar'] = lidar_data
        return ret

if __name__ =='__main__':

    sensor_size = (1920, 1120)
    sample_per_n_frames = 5
    with open('camera_params.pkl', 'rb') as f:
        camera_params = pickle.load(f)
    intrinsics = camera_params['cam_f0']['intrinsics']
    fov_x, fov_y = calculate_fov(intrinsics, sensor_size)

    env = ScenarioEnv(
        {
            'agent_observation': CameraAndLidarObservation,
            'image_on_cuda': False,
            # To enable onscreen rendering, set this config to True.
            "use_render": False,

            # !!!!! To enable offscreen rendering, set this config to True !!!!!
            "image_observation": True,
            "norm_pixel": False,
            "stack_size": 1,

            # ===== The scenario and MetaDrive config =====
            "agent_policy": ReplayEgoCarPolicy,
            "no_traffic": False,
            "sequential_seed": True,
            "reactive_traffic": False,
            "num_scenarios": 9,
            "horizon": 1000,
            "no_static_vehicles": False,
            "agent_configs": {
                "default_agent": dict(use_special_color=True, vehicle_model="varying_dynamics_bounding_box")
            },
            "vehicle_config": dict(
                show_navi_mark=False,
                show_line_to_dest=False,
                lidar=dict(num_lasers=120, distance=50),
                lane_line_detector=dict(num_lasers=0, distance=50),
                side_detector=dict(num_lasers=12, distance=50),
            ),
            # "use_bounding_box": True,
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
            "height_scale": 1,

            "set_static": True,

            # ===== Set some sensor and visualization configs =====
            "daytime": "08:10",
            "window_size": (sensor_size[0], sensor_size[1]),
            "camera_dist": 0,  # 0.8, 1.71
            "camera_height": 1.5,  # 1.5
            "camera_pitch": None,
            "camera_fov_x": fov_x,  # 60, 66
            "camera_fov_y": fov_y,
            "sensors": dict(
                depth_camera=(DepthCamera, sensor_size[0], sensor_size[1]),
                rgb_camera=(RGBCamera, sensor_size[0], sensor_size[1]),

            ),
            # ===== Remove useless items in the images =====
            "show_logo": False,
            "show_fps": False,
            "show_interface": True,
            "disable_collision": True,
            "force_destroy": True,
        }
    )

    for seed in tqdm(range(9)):
        env.reset(seed)
        for t in tqdm(range(1000)):
            o, r, d, _, _ = env.step([1, 0.88])
            if t % sample_per_n_frames == 0:
                rgb = o['camera']
                lidar = o['lidar']
                plt.imshow(rgb['cam_f0'])
                plt.show()
                pcd = o3d.geometry.PointCloud()
                # 2. 将点云数据赋值到 Open3D 点云对象
                pcd.points = o3d.utility.Vector3dVector(lidar[0])

                o3d.visualization.draw_geometries([pcd])

