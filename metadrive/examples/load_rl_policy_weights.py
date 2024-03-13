# import your trainer here
from ray.rllib.agents.ppo import PPOTrainer
from copo_scenario.torch_copo.algo_ippo import IPPOTrainer
from copo_scenario.torch_copo.algo_copo import CoPOTrainer





"""
load a trained rl policy with a fixed network architecture;
the policy can be trained by RLlib or PyTorch;

"""

class RLPolicy:
    def __init__(self, policy_name, policy_ckpt_path, trainer=None):
        self.policy_name = policy_name
        self.policy_ckpt_path = policy_ckpt_path
        self.model = None
        self.trainer = trainer


    def load_rllib_policy(self):
        """
        Load the given trained policy for agents trained by RLlib
        """

        assert self.trainer is not None  # You need to restore the trainer before loading the policy
        try:
            self.trainer.restore(self.policy_ckpt_path)
        except Exception as e:
            print("Cannot load the policy: ", e)
            raise e
        self.policy = self.trainer.get_policy()

        # self.model = self.policy.model
        # self.state_dict = self.model.state_dict()
        # my_model = MyTorchModel()
        # my_model.load_state_dict(pytorch_state_dict)

        return self.policy








if __name__ == '__main__':
    # Assuming you've set up your configuration correctly and used PyTorch
    # Create a new instance of the trainer and restore from checkpoint
    trainer = IPPOTrainer(config={"framework": "torch"})
    trainer.restore("~/ray_results/test_ippo/IPPOTrainer_MultiAgentIntersectionEnv_41ca5_00000_0_env=MultiAgentIntersectionEnv,seed=0_2024-03-05_23-29-24/checkpoint_000070/policies/default")  # example path: "../checkpoint_000001/checkpoint-1"

    # Access the policy model
    policy = trainer.get_policy()
    # model = policy.model

    # pytorch_state_dict = model.state_dict()
    # my_model = MyTorchModel()
    # my_model.load_state_dict(pytorch_state_dict)


