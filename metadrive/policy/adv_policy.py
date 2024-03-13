from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.policy.base_policy import BasePolicy
from metadrive.examples.load_rl_policy_weights import RLPolicy
from copo_scenario.torch_copo.algo_ippo import IPPOPolicy
from copo_scenario.torch_copo.algo_copo import CoPOPolicy

from metadrive.obs.state_obs import LidarStateObservation
from metadrive.engine.engine_utils import get_global_config
import numpy as np
from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
from copo_scenario.torch_copo.utils.env_wrappers import get_rllib_compatible_env

path_copo = "/Users/claire_liu/ray_results/copo_5_agents_crachR_1_successR_10_LCF_-90_-180/checkpoint_000310/policies/default/"
path_ippo = "/Users/claire_liu/ray_results/ippo_5_agents_crachR_1_successR_10/IPPOTrainer_MultiAgentIntersectionEnv_d0949_00000_0_env=MultiAgentIntersectionEnv,seed=0_2024-03-09_12-37-15/checkpoint_000410/policies/default/"
copo_policy = CoPOPolicy.from_checkpoint(path_copo)
ippo_policy = IPPOPolicy.from_checkpoint(path_ippo)
policy = ippo_policy


# _GLOBAL_ENTRY = {}


def obs_correction(obs): # TODO: Do we need this????
    # due to coordinate correction, this observation should be reversed
    obs[15] = 1 - obs[15]
    obs[10] = 1 - obs[10]
    return obs


class AdvPolicy(BasePolicy):
    """
    This policy can protect Manual control and EnvInputControl
    """


    def predict(self, vehicle, deterministic=False, need_obs=False):
        expert_obs_cfg = dict(
            lidar=dict(num_lasers=72, distance=40, num_others=0),
            random_agent_model=False
        )
        origin_obs_cfg = dict(
            lidar=dict(num_lasers=72, distance=40, num_others=0),
            random_agent_model=False
        )
        config = get_global_config().copy()
        config["vehicle_config"].update(expert_obs_cfg)
        _expert_observation = LidarStateObservation(config)

        assert _expert_observation.observation_space.shape[0] == 91, "Observation not match"

        vehicle.config.update(expert_obs_cfg)
        obs = _expert_observation.observe(vehicle)
        vehicle.config.update(origin_obs_cfg)
        obs = obs_correction(obs) # TODO: we don't need this??


        # flattened_obs = np.stack(obs.values(), axis=0)
        flattened_obs = obs.reshape(1, 91)
        action = policy.compute_actions(flattened_obs)[0]
        action = action[0]
        assert action.shape == (2,) , "control one adversary at a time {}".format(action.shape)

        # ret = np.clip(ret, -1.0, 1.0) all clip should be implemented in env!
        return (action, obs) if need_obs else action


    def policy_act(self, obs, need_obs=False):
        # flattened_obs = np.stack(obs.values(), axis=0)
        flattened_obs = obs.reshape(1, 91)
        action = policy.compute_actions(flattened_obs)[0]
        action = action[0]
        assert action.shape == (2,), "control one adversary at a time {}".format(action.shape)

        # ret = np.clip(ret, -1.0, 1.0) all clip should be implemented in env!
        return (action, obs) if need_obs else action



    def act(self, obs=None, **kwargs):
        action = self.policy_act(obs, False)
        self.action_info["action"] = action
        return action






