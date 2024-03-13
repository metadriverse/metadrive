import numpy as np

from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
from copo_scenario.torch_copo.utils.env_wrappers import get_rllib_compatible_env
from metadrive.examples.load_rl_policy_weights import RLPolicy
from copo_scenario.torch_copo.algo_ippo import IPPOTrainer, IPPOPolicy
from copo_scenario.torch_copo.algo_copo import CoPOTrainer, CoPOPolicy
import ray


if __name__ == '__main__':
    # ray.init()
    # config = dict(
    #     # ===== Environmental Setting =====
    #     # We can grid-search the environmental parameters!
    #     env=get_rllib_compatible_env(MultiAgentIntersectionEnv),
    #
    #     env_config=dict(num_agents=5,
    #                     # crash_vehicle_penalty=-5
    #                     ),
    #     stop=int(100),
    #     num_seeds=1,
    #     num_worker=1,
    #
    #     # evaluation_interval= 1,  # Evaluate every training iteration
    #     # evaluation_num_episodes= 10,  # Number of episodes to run for each evaluation
    #     # Specify evaluation config (if different from training config)
    #     # evaluation_config= {
    #     # # Configurations here will override those in the main config during evaluation
    #     # # For example, to evaluate the policy deterministically, you might set:
    #     # "explore": False,
    #     # },
    # # )


    # trainer = IPPOTrainer(config=config)
    path_copo = "/Users/claire_liu/ray_results/copo_5_agents_crachR_1_successR_10_LCF_-90_-180/checkpoint_000310/policies/default/"
    path_ippo = "/Users/claire_liu/ray_results/ippo_5_agents_crachR_1_successR_10/IPPOTrainer_MultiAgentIntersectionEnv_d0949_00000_0_env=MultiAgentIntersectionEnv,seed=0_2024-03-09_12-37-15/checkpoint_000410/policies/default/"
    # policy = CoPOPolicy.from_checkpoint(path_copo)
    policy = IPPOPolicy.from_checkpoint(path_ippo)



    def try_loaded_policy():
        marl_config=dict(num_agents=5,
                        crash_vehicle_penalty=-1.0,
                        success_reward=10.0,
                        )
        env = MultiAgentIntersectionEnv(config=marl_config)
        # env.observation_space.spaces
        obs, info = env.reset()
        import torch
        flattened_obs = np.stack(obs.values(), axis=0)
        # obs_new = {agent_id: torch.tensor(obs[agent_id]) for agent_id in obs}


        success = 0
        cnt = 0
        for i in range(1000000):
            action = policy.compute_actions(flattened_obs)[0]
            action_dict = {}
            for i, agent_id in enumerate(obs):
                action_dict[agent_id] = np.array(action[i])

            obs, r, tm, tc, info = env.step(action_dict)
            flattened_obs = np.stack(obs.values(), axis=0)

            env.render(mode="top_down")

            if tm["__all__"] or tc["__all__"]:
                print("Episode finished")
                obs, info = env.reset()
                flattened_obs = np.stack(obs.values(), axis=0)
                continue

        env.close()
    try_loaded_policy()



    def get_model_summary():
        rl_policy = RLPolicy("IPPO", "/Users/claire_liu/ray_results/test_ippo/IPPOTrainer_MultiAgentIntersectionEnv_41ca5_00000_0_env=MultiAgentIntersectionEnv,seed=0_2024-03-05_23-29-24/checkpoint_000070"
        , trainer)
        POLICY = rl_policy.load_rllib_policy()
        model = POLICY.model.base_model.summary()
        result = trainer.train()
        # Perform evaluation

    def eval():
        evaluation_results = trainer.evaluate()
        # Print the evaluation results
        print("Evaluation results:", evaluation_results)


    def train_one_iter():
        result = trainer.train()
        # Accessing metrics
        print("Training iteration:", result["training_iteration"])
        print("Average reward:", result["episode_reward_mean"])
        print("Average episode length:", result["episode_len_mean"])

        # Access custom metrics if any
        if "custom_metrics" in result:
            print("Custom metrics:", result["custom_metrics"])






