import gymnasium as gym
import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.obs.observation_base import BaseObservation
import torch
import math
from scipy.spatial.transform import Rotation as R
from typing import Dict, Any

lidar2ego = np.array([
    [0, 1, 0,  0.5], 
    [-1, 0, 0,  0.00],
    [0, 0, 1,  1.50], 
    [0, 0, 0,  1.00],
], dtype=np.float32)

def build_camera_params(camera_configs):
    camera_params = {}
    R_ego2cam_base = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0] 
    ], dtype=np.float32)
    
    for cam_name, cam_cfg in camera_configs.items():
        fovx_rad = math.radians(cam_cfg['fovx'])
        fovy_rad = math.radians(cam_cfg['fovy'])

        fx = cam_cfg['W'] / (2 * math.tan(fovx_rad / 2))
        fy = cam_cfg['H'] / (2 * math.tan(fovy_rad / 2))

        K = torch.tensor([
            [fx, 0, cam_cfg['cx']],
            [0, fy, cam_cfg['cy']],
            [0, 0, 1]
        ], dtype=torch.float32)

        hpr = cam_cfg['hpr']
        if isinstance(hpr, list):
            hpr = np.array(hpr, dtype=np.float32)
        hpr_rad = np.deg2rad(hpr)
        R_additional = R.from_euler('ZYX', hpr_rad, degrees=False).as_matrix()

        R_final = R_ego2cam_base @ R_additional

        offset = cam_cfg['offset']
        if isinstance(offset, list):
            offset = np.array(offset, dtype=np.float32)
        else:
            offset = np.array(offset, dtype=np.float32)

        translation = -np.asarray(R_final @ offset, dtype=np.float32)

        ego2camera = np.eye(4, dtype=np.float32)
        ego2camera[:3, :3] = R_final
        ego2camera[:3, 3] = translation
        
        ego2camera = torch.from_numpy(ego2camera)
        
        camera_params[cam_name] = {
            'K': K,
            'H': cam_cfg['H'],
            'W': cam_cfg['W'],
            'ego2camera': ego2camera
        }
    
    return camera_params

class GaussianObservation(BaseObservation):
    """
    Use only image info as input
    """
    STACK_SIZE = 3  # use continuous 3 image as the input

    def __init__(self, config):
        super().__init__(config)
        self.STACK_SIZE = config["stack_size"]
        self.clip_rgb = config['clip_rgb']
        self.camera_configs = config['cameras']

    def reset(self, controller, render_fn, camera_params, **kwargs):
        """
        Clear stack
        :param env: MetaDrive
        :param vehicle: BaseVehicle
        :return: None
        """
        
        self.controller = controller
        self.render_fn = render_fn

        dataset_params = camera_params or {}
        merged_params = dict(dataset_params)

        if self.camera_configs:
            missing_cfg = {
                name: cfg for name, cfg in self.camera_configs.items() if name not in merged_params
            }
            if missing_cfg:
                built_missing = build_camera_params(missing_cfg)
                merged_params.update(built_missing)

        self.params = merged_params

        if self.clip_rgb:
            self.state = {cam_name: np.zeros(self.an_observation_shape(cam['H'], cam['W']), dtype=np.float32) for cam_name, cam in self.params.items()}
        else:
            self.state = {cam_name: np.zeros(self.an_observation_shape(cam['H'], cam['W']), dtype=np.uint8) for cam_name, cam in self.params.items()}


    @property
    def observation_space(self):
        # sensor_cls = self.config["sensors"][self.image_source][0]
        # assert sensor_cls == "MainCamera" or issubclass(sensor_cls, BaseCamera), "Sensor should be BaseCamera"
        
        space = {}
        for name, sensor in self.params.items():
            shape = self.an_observation_shape(sensor['H'], sensor['W'])
            if self.clip_rgb:
                space[name] = gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
            else:
                space[name] = gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        return space

    def an_observation_shape(self, h, w):
        return (self.STACK_SIZE, h, w, 3)
 
    def observe(self):
        """
        Get the image Observation. By setting new_parent_node and the reset parameters, it can capture a new image from
        a different position and pose
        """
        camera_info = {}
        ego_pose = torch.tensor(self.controller.transform).inverse()
        for cam_name, params in self.params.items():
            ego2cam = params.get('ego2camera')
            if ego2cam is None:
                offset = params['offset']
                if isinstance(offset, list):
                    offset = np.array(offset, dtype=np.float32)
                hpr = params['hpr']
                if isinstance(hpr, list):
                    hpr = np.array(hpr, dtype=np.float32)
                hpr_rad = np.deg2rad(hpr)
                R_additional = R.from_euler('ZYX', hpr_rad, degrees=False).as_matrix()
                R_ego2cam_base = np.array([
                    [0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0]
                ], dtype=np.float32)
                R_final = R_ego2cam_base @ R_additional
                translation = -np.asarray(R_final @ offset, dtype=np.float32)
                ego2cam = torch.from_numpy(
                    np.vstack([
                        np.hstack([R_final, translation.reshape(3, 1)]),
                        np.array([0, 0, 0, 1], dtype=np.float32)
                    ])
                )
                params['ego2camera'] = ego2cam

            extrinsics = ego2cam @ ego_pose
            ret = self.render_fn(
                K=params['K'],
                H=params['H'],
                W=params['W'],
                extrinsics=extrinsics,
            )
            if cam_name == 'BACK':
                ret = np.zeros_like(ret)  # UniAD expects a back view; fill with black when data is missing
            self.state[cam_name] = np.roll(self.state[cam_name], -1, axis=0)
            self.state[cam_name][-1] = ret

            K = params['K']
            if hasattr(K, 'cpu'):
                K = K.cpu().numpy()
            else:
                K = np.array(K)

            H = int(params['H'])
            W = int(params['W'])

            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])

            fovx = 2 * math.atan(W / (2 * fx))
            fovy = 2 * math.atan(H / (2 * fy))

            lidar2cam = ego2cam @ lidar2ego
            
            camera_info[cam_name] = {
            'l2c': lidar2cam.numpy().astype(np.float32),
            'intrinsic': {
                'fovx': float(fovx),
                'fovy': float(fovy),
                'H': H,
                'W': W,
                'cx': float(cx),
                'cy': float(cy)
            },
            'ego2camera': ego2cam.numpy().astype(np.float32),
            'K': K.astype(np.float32)
        }

        return {
            'camera_info': camera_info,
            'image': self.state
        }


    def destroy(self):
        """
        Clear memory
        """
        super(GaussianObservation, self).destroy()
        self.state = None