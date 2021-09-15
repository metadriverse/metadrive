from metadrive.policy.base_policy import BasePolicy
from metadrive.utils.math_utils import clip


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

    def act(self, agent_id):
        # clip to -1, 1
        action = [
            clip(self.engine.external_actions[agent_id][i], -1.0, 1.0)
            for i in range(len(self.engine.external_actions[agent_id]))
        ]
        if not self.discrete_action:
            return action
        else:
            return self.convert_to_continuous_action(action)

    def convert_to_continuous_action(self, action):
        steering = action[0] * self.steering_unit - 1.0
        throttle = action[1] * self.throttle_unit - 1.0
        return steering, throttle
