from metadrive.envs.metadrive_env import MetaDriveEnv
import argparse
import cv2
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE
from metadrive.policy.idm_policy import IDMPolicy
from typing import Union
from metadrive.utils import Config
# from metadrive.envs.intersection_env import IntersectionEnv
# from metadrive.policy.adv_policy import AdvPolicy
from metadrive.envs.adversary_env import AdversaryEnv
import ray

from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy






# running the adversary environment directly in here; rightnow the policy is IDMPolicy


if __name__ == '__main__':
    lane_change_ego_path = "/Users/claire_liu/ray_results/LaneChangeIDM_PPO_ADV5_SuccessR10/PPO_GymEnvWrapper_918f1_00000_0_start_seed=5000_2024-03-18_10-18-42/checkpoint_000100/policies/default_policy"
    policy_CoPO_PPO = PPOTorchPolicy.from_checkpoint(lane_change_ego_path)
    # policy_IPPO_PPO = PPOTorchPolicy.from_checkpoint("/Users/claire_liu/metadrive/metadrive/examples/IDM_Adv_PPO/PPO_GymEnvWrapper_42ff1_00000_0_start_seed=5000_2024-03-13_14-47-18/checkpoint_000040/policies/default_policy")
    env_config = dict(use_render=False,
                      manual_control=False,
                      num_adversary_vehicles=10,
                      crash_vehicle_done=True,
                      crash_object_done=True,
                      out_of_route_done=True,
                      num_scenarios=10,
                      traffic_mode="adversary",
                      #
                      )

    # default_config.update(env_config)

    env = AdversaryEnv(env_config)

    # env = MetaDriveEnv(default_config)
    try:
        o, _ = env.reset()
        # print(HELP_MESSAGE)

        for i in range(1, 1000000000):
            flattened_obs = o.reshape(1, 259)
            action = policy_CoPO_PPO.compute_actions(flattened_obs)[0][0]

            o, r, tm, tc, info = env.step(action)
            env.render(mode="top_down")

            if (tm or tc) or info["arrive_dest"]:
                o, _ = env.reset()
    finally:
        env.close()