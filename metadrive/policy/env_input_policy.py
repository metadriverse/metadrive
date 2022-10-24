from metadrive.policy.base_policy import BasePolicy
from metadrive.utils.math_utils import clip
from collections.abc import Iterable


class EnvInputPolicy(BasePolicy):
    def __init__(self, obj, seed):
        # Since control object may change
        super(EnvInputPolicy, self).__init__(control_object=None)
        self.discrete_action = self.engine.global_config["discrete_action"]
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

        if not self.discrete_action:
            to_process = action
        else:
            to_process = self.convert_to_continuous_action(action)

        # clip to -1, 1
        action = [clip(to_process[i], -1.0, 1.0) for i in range(len(to_process))]

        return action

    def convert_to_continuous_action(self, action):
        if isinstance(action, Iterable):
            assert len(action) == 2
            steering = action[0] * self.steering_unit - 1.0
            throttle = action[1] * self.throttle_unit - 1.0
        else:
            steering = float(action % self.discrete_steering_dim) * self.steering_unit - 1.0
            throttle = float(action // self.discrete_steering_dim) * self.throttle_unit - 1.0

            # print("Steering: ", steering, " Throttle: ", throttle)
        return steering, throttle
