from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.policy.base_policy import BasePolicy
from metadrive.examples.load_rl_policy_weights import RLPolicy
from copo_scenario.torch_copo.algo_ippo import IPPOTrainer
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.engine.engine_utils import get_global_config
import numpy as np

class AdvPolicy(BasePolicy):
    """
    This policy can protect Manual control and EnvInputControl
    """
    def __init__(self, **kwargs):
        super(AdvPolicy, self).__init__(control_object=self.control_object, random_seed=None, config=None)

    def load_adv_policy(self, policy_name, policy_ckpt_path):
        trainer = IPPOTrainer(config={"framework": "torch"})
        rl_policy = RLPolicy("torch", "ippo", "path/to/your/ppo/checkpoint", trainer)
        self.adv_policy = rl_policy.load_rllib_policy()

    def predict(self, vehicle, deterministic=False, need_obs=False):
        expert_obs_cfg = dict(
            lidar=dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0),
            random_agent_model=False
        )
        origin_obs_cfg = dict(
            lidar=dict(num_lasers=240, distance=50, num_others=0, gaussian_noise=0.0, dropout_prob=0.0),
            random_agent_model=False
        )
        config = get_global_config().copy()
        config["vehicle_config"].update(expert_obs_cfg)
        _expert_observation = LidarStateObservation(config)

        assert _expert_observation.observation_space.shape[0] == 275, "Observation not match"

        vehicle.config.update(expert_obs_cfg)
        obs = _expert_observation.observe(vehicle)
        vehicle.config.update(origin_obs_cfg)
        # obs = obs_correction(obs) # TODO: we don't need this??
        action = self.adv_policy.compute_actions(obs)

        assert action.shape == [1,2], "control one adversary at a time"

        # ret = np.clip(ret, -1.0, 1.0) all clip should be implemented in env!
        return (action, obs) if need_obs else action



    def act(self, agent_id=None):
        action = self.predict(self.control_object)
        self.action_info["action"] = action
        return action






