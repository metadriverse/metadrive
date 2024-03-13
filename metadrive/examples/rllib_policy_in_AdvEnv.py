import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
from copo_scenario.torch_copo.utils.env_wrappers import get_rllib_compatible_env
from metadrive.examples.load_rl_policy_weights import RLPolicy
from copo_scenario.torch_copo.algo_ippo import IPPOTrainer, IPPOPolicy
from copo_scenario.torch_copo.algo_copo import CoPOTrainer, CoPOPolicy
import ray
from metadrive.envs.adversary_env import AdversaryEnv
from metadrive.policy.idm_policy import IDMPolicy


if __name__ == '__main__':
    path_copo = "/Users/claire_liu/ray_results/copo_5_agents_crachR_1_successR_10_LCF_-90_-180/checkpoint_000310/policies/default/"
    path_ippo = "/Users/claire_liu/ray_results/ippo_5_agents_crachR_1_successR_10/IPPOTrainer_MultiAgentIntersectionEnv_d0949_00000_0_env=MultiAgentIntersectionEnv,seed=0_2024-03-09_12-37-15/checkpoint_000410/policies/default/"
    # policy = CoPOPolicy.from_checkpoint(path_copo)
    policy = IPPOPolicy.from_checkpoint(path_ippo)

    def try_loaded_policy():
        default_config = AdversaryEnv.default_config()
        env_config=dict(crash_vehicle_penalty=-1.0,
                        success_reward=10.0,

                        # traffic_vehicle_config=dict(
                        #
                        # ),
                        num_adversary_vehicles=5,
                        agent_policy=IDMPolicy,
                        )

        default_config.update(env_config)
        env = AdversaryEnv(config=default_config)
        obs, info = env.reset()
        flattened_obs = obs.reshape(1, 91)



        for i in range(1000000):
            action = policy.compute_actions(flattened_obs)[0]
            action = action[0]
            obs, r, tm, tc, info = env.step(action)
            flattened_obs = obs.reshape(1, 91)

            env.render(mode="top_down")

            if tm or tc:
                print("Episode finished")
                obs, info = env.reset()
                flattened_obs = obs.reshape(1, 91)
                continue

        env.close()
    try_loaded_policy()










