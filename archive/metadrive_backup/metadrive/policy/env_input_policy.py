import gymnasium as gym
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
        to_process = self.convert_to_continuous_action(action) if self.discrete_action else action

        # clip to -1, 1
        action = [clip(to_process[i], -1.0, 1.0) for i in range(len(to_process))]
        self.action_info["action"] = action
        return action

    def convert_to_continuous_action(self, action):
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
        discrete_action = engine_global_config["discrete_action"]
        discrete_steering_dim = engine_global_config["discrete_steering_dim"]
        discrete_throttle_dim = engine_global_config["discrete_throttle_dim"]
        use_multi_discrete = engine_global_config["use_multi_discrete"]

        if not discrete_action:
            _input_space = gym.spaces.Box(-1.0, 1.0, shape=(2, ), dtype=np.float32)
        else:
            if use_multi_discrete:
                _input_space = gym.spaces.MultiDiscrete([discrete_steering_dim, discrete_throttle_dim])
            else:
                _input_space = gym.spaces.Discrete(discrete_steering_dim * discrete_throttle_dim)
        return _input_space


class ExtraEnvInputPolicy(EnvInputPolicy):
    """
    This policy allows the env.step() function accept extra input besides [steering, throttle/brake]
    """
    extra_input_space = None

    def __init__(self, obj, seed):
        """
        Accept one more argument for creating the input space
        Args:
            obj: BaseObject
            seed: random seed. It is usually filled automatically.
        """
        super(ExtraEnvInputPolicy, self).__init__(obj, seed)
        self.extra_input = None

    def act(self, agent_id):
        """
        It retrieves the action from self.engine.external_actions["action"]
        Args:
            agent_id: the id of this agent

        Returns: continuous 2-dim action [steering, throttle]

        """
        action = self.engine.external_actions[agent_id]["action"]
        self.extra_input = self.engine.external_actions[agent_id]["extra"]

        # the following content is the same as EnvInputPolicy
        if self.engine.global_config["action_check"]:
            # Do action check for external input in EnvInputPolicy
            assert self.get_input_space().contains(self.engine.external_actions[agent_id]), \
                "Input {} is not compatible with action space {}!".format(
                self.engine.external_actions[agent_id], self.get_input_space()
            )
        to_process = self.convert_to_continuous_action(action) if self.discrete_action else action

        # clip to -1, 1
        action = [clip(to_process[i], -1.0, 1.0) for i in range(len(to_process))]
        self.action_info["action"] = action
        return action

    @classmethod
    def set_extra_input_space(cls, extra_input_space: gym.spaces.space.Space):
        """
        Set the space for this extra input. Error will be thrown, if this class property is set already.
        Args:
            extra_input_space: gym.spaces.space.Space

        Returns: None

        """
        assert isinstance(extra_input_space, gym.spaces.space.Space)
        ExtraEnvInputPolicy.extra_input_space = extra_input_space

    @classmethod
    def get_input_space(cls):
        """
        Define the input space as a Dict Space
        Returns: Dict action space

        """
        action_space = super(ExtraEnvInputPolicy, cls).get_input_space()
        return gym.spaces.Dict({"action": action_space, "extra": cls.extra_input_space})
