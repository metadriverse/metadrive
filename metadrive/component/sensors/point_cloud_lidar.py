import numpy as np
from panda3d.core import Point3

from metadrive.component.sensors.depth_camera import DepthCamera


def euler_to_rotation_matrix(hpr):
    """
    Convert ZYX Euler angles to a rotation matrix.

    Parameters:
        hpr (array-like): [yaw (Z), pitch (Y), roll (X)] in degrees.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    hpr = np.radians(hpr)

    cz, sz = np.cos(hpr[0]), np.sin(hpr[0])  # Yaw (Z)
    cy, sy = np.cos(hpr[1]), np.sin(hpr[1])  # Pitch (Y)
    cx, sx = np.cos(hpr[2]), np.sin(hpr[2])  # Roll (X)

    rotation_matrix = np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], [-sy, cy * sx, cy * cx]
        ]
    )

    return rotation_matrix


class PointCloudLidar(DepthCamera):
    """
    This can be viewed as a special camera sensor, whose RGB channel is (x, y ,z) world coordinate of the point cloud.
    Thus, it is compatible with all image related stuff. For example, you can use it with ImageObservation.
    """
    num_channels = 3  # x, y, z coordinates

    def __init__(self, width, height, ego_centric, engine, *, cuda=False):
        """
        If ego_centric is True, the point cloud will be in the camera's ego coordinate system.
        """
        if cuda:
            raise ValueError("LiDAR does not support CUDA acceleration for now. Ask for support if you need it.")
        super(PointCloudLidar, self).__init__(width, height, engine, cuda=False)
        self.ego_centric = ego_centric

    def get_rgb_array_cpu(self):
        """
        The result of this function is now a 3D array of point cloud coord in shape (H, W, 3)
        The lens parameters can be changed on the fly!
        """

        lens = self.lens
        fov = lens.getFov()
        f_x = self.BUFFER_W / 2 / (np.tan(fov[0] / 2 / 180 * np.pi))
        f_y = self.BUFFER_H / 2 / (np.tan(fov[1] / 2 / 180 * np.pi))
        intrinsics = np.asarray([[f_x, 0, (self.BUFFER_H - 1) / 2], [0, f_y, (self.BUFFER_W - 1) / 2], [0, 0, 1]])
        f = lens.getFar()
        n = lens.getNear()

        depth = super(PointCloudLidar, self).get_rgb_array_cpu()
        hpr = self.cam.getHpr(self.engine.render)
        hpr[0] += 90  # pand3d's y is the camera facing direction, so we need to rotate it 90 degree
        hpr[1] *= -1  # left right handed convert

        rotation_matrix = euler_to_rotation_matrix(hpr)
        translation = Point3(0, 0, 0)
        if not self.ego_centric:
            translation = np.asarray(self.engine.render.get_relative_point(self.cam, Point3(0, 0, 0)))
        z_eye = 2 * n * f / ((f + n) - (2 * depth - 1) * (f - n))
        points = self.simulate_lidar_from_depth(z_eye.squeeze(-1), intrinsics, translation, rotation_matrix)
        return points

    @staticmethod
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
        depth_img = depth_img.T
        depth_img = depth_img[::-1, ::-1]
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
        cam_coords *= depth_img.reshape(-1)[..., None]  # Scale by depth

        # Remove invalid points (e.g., depth = 0)
        # valid_mask = depth_img.flatten() > 0
        # cam_coords = cam_coords[valid_mask]
        cam_coords = cam_coords[..., [2, 1, 0]]

        # Transform points to the world coordinate system
        world_coords = (camera_rotation @ cam_coords.T).T + camera_translation

        # to original shape
        world_coords = world_coords.reshape(height, width, 3)
        return world_coords.swapaxes(0, 1)
