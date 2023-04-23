import gym
from metadrive.engine.engine_utils import get_global_config
import numpy as np

from metadrive.policy.base_policy import BasePolicy
from metadrive.utils.math import clip


class EnvInputPolicy(BasePolicy):
    DEBUG_MARK_COLOR = (252, 119, 3, 255)

    def __init__(self, obj, seed):
        # Since control object may change
        super(EnvInputPolicy, self).__init__(control_object=obj, random_seed=seed)
        self.discrete_action = self.engine.global_config["discrete_action"]
        self.use_multi_discrete = self.engine.global_config["use_multi_discrete"]
        self.steering_unit = 2.0 / (
            self.engine.global_config["discrete_steering_dim"] - 1
        )  # for discrete actions space
        self.throttle_unit = 2.0 / (
            self.engine.global_config["discrete_throttle_dim"] - 1
        )  # for discrete actions space
        self.discrete_steering_dim = self.engine.global_config["discrete_steering_dim"]
        self.discrete_throttle_dim = self.engine.global_config["discrete_throttle_dim"]

    def act(self, agent_id):
        action = self.engine.external_actions[agent_id]
        if self.engine.global_config["action_check"]:
            # Do action check for external input in EnvInputPolicy
            assert self.get_input_space().contains(action), "Input {} is not compatible with action space {}!".format(
                action, self.get_input_space()
            )
        if not self.discrete_action:
            to_process = action
        else:
            to_process = self.convert_to_continuous_action(action)

        # clip to -1, 1
        action = [clip(to_process[i], -1.0, 1.0) for i in range(len(to_process))]
        self.action_info["action"] = action
        return action

    def convert_to_continuous_action(self, action):
        # if isinstance(action, Iterable):
        #     assert len(action) == 2
        #     steering = action[0] * self.steering_unit - 1.0
        #     throttle = action[1] * self.throttle_unit - 1.0
        # else:
        # A more clear implementation:
        if self.use_multi_discrete:
            steering = action[0] * self.steering_unit - 1.0
            throttle = action[1] * self.throttle_unit - 1.0
        else:
            steering = float(action % self.discrete_steering_dim) * self.steering_unit - 1.0
            throttle = float(action // self.discrete_steering_dim) * self.throttle_unit - 1.0

        return steering, throttle

    @classmethod
    def get_input_space(cls):
        """
        The Input space is a class attribute
        """
        engine_global_config = get_global_config()
        extra_action_dim = engine_global_config["vehicle_config"]["extra_action_dim"]
        discrete_action = engine_global_config["discrete_action"]
        discrete_steering_dim = engine_global_config["discrete_steering_dim"]
        discrete_throttle_dim = engine_global_config["discrete_throttle_dim"]
        use_multi_discrete = engine_global_config["use_multi_discrete"]

        if not discrete_action:
            _input_space = gym.spaces.Box(-1.0, 1.0, shape=(2 + extra_action_dim, ), dtype=np.float32)
        else:
            if use_multi_discrete:
                _input_space = gym.spaces.MultiDiscrete([discrete_steering_dim, discrete_throttle_dim])
            else:
                _input_space = gym.spaces.Discrete(discrete_steering_dim * discrete_throttle_dim)
        return _input_space
