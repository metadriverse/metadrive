from stable_baselines3.common.vec_env import SubprocVecEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.engine.asset_loader import AssetLoader
import numpy as np
import gymnasium as gym
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.instance_camera import InstanceCamera
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.image_obs import ImageObservation
from metadrive.obs.state_obs import StateObservation
from metadrive.policy.replay_policy import ReplayEgoCarPolicy


class MyStateObservation(BaseObservation):
    def __init__(self, config):
        super(MyStateObservation, self).__init__(config)

    @property
    def observation_space(self):
        # Define the observation space for the state observation
        dict = {
            "world_position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, ), dtype=np.float32),
            "world_heading": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, ), dtype=np.float32),
            "world_speed": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)
        }
        return gym.spaces.Dict(dict)

    def observe(self, vehicle):
        world_x, world_y = vehicle.position[0], vehicle.position[1]
        world_heading_x, world_heading_y = vehicle.heading[0], vehicle.heading[1]
        world_position = np.array([world_x, world_y])
        world_heading = np.array([world_heading_x, world_heading_y])
        world_speed = np.float32(vehicle.speed)
        return dict(
            world_position=world_position.astype(np.float32),
            world_heading=world_heading.astype(np.float32),
            world_speed=world_speed.astype(np.float32)
        )


class SOMObseravation(BaseObservation):
    def __init__(self, config):
        super(SOMObseravation, self).__init__(config)
        self.rgb = ImageObservation(config, "rgb", config["norm_pixel"])
        self.depth = ImageObservation(config, "depth", config["norm_pixel"])
        self.semantic = ImageObservation(config, "semantic", config["norm_pixel"])
        self.instance = ImageObservation(config, "instance", config["norm_pixel"])
        self.state = MyStateObservation(config)  #StateObservation(config)

    @property
    def observation_space(self):
        os = {o: getattr(self, o).observation_space for o in ["rgb", "state", "depth", "semantic", "instance"]}
        return gym.spaces.Dict(os)

    def observe(self, vehicle):
        os = {
            o: getattr(self, o).observe(new_parent_node=vehicle.origin)
            for o in ["rgb", "depth", "semantic", "instance"]
        }
        os["state"] = self.state.observe(vehicle)
        return os


def visualize(obs, env_id):
    # visualize image observation
    o_1 = obs["depth"][env_id][..., -1]
    o_1 = np.concatenate([o_1, o_1, o_1], axis=-1)  # align channel
    o_2 = obs["rgb"][env_id][..., -1]
    o_3 = obs["semantic"][env_id][..., -1]
    o_4 = obs["instance"][env_id][..., -1]
    #print(o_1.shape, o_2.shape, o_3.shape, o_4.shape)
    #exit()
    o = cv2.hconcat([o_1, o_2, o_3, o_4])
    o = (o * 255).astype(np.uint8)
    cv2.imshow(f"obs_{env_id}", o)
    cv2.waitKey(1)


import cv2

if __name__ == "__main__":

    asset_path = AssetLoader.asset_path
    cfg = {
        "agent_policy": ReplayEgoCarPolicy,
        "sequential_seed": True,
        "reactive_traffic": False,
        "use_render": True,
        "data_directory": AssetLoader.file_path(asset_path, "nuscenes", unix_style=False),
        "num_scenarios": 10,
        "agent_observation": SOMObseravation,
        "sensors": {
            "rgb": (RGBCamera, 256, 256),
            "depth": (DepthCamera, 256, 256),
            "semantic": (SemanticCamera, 256, 256),
            "instance": (InstanceCamera, 256, 256)
        },
    }

    # Create a vectorized environment with 4 parallel environments
    env = SubprocVecEnv([lambda: ScenarioEnv(cfg) for _ in range(2)])
    for _ in range(100):
        obs = env.reset()
        done = False
        while not done:
            actions = np.random.uniform(-1, 1, size=(env.num_envs, 2))  # Random actions for each environment
            obs, rewards, dones, infos = env.step(actions)
            #for i in range(env.num_envs):
            #    visualize(obs, i)

            print(obs["state"])
            env.render(mode="top_down")
