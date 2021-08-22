from pgdrive.policy.base_policy import BasePolicy


class EnvInputPolicy(BasePolicy):
    def __init__(self):
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
        if not self.discrete_action:
            return self.engine.external_actions[agent_id]
        else:
            return self.convert_to_continuous_action(self.engine.external_actions[agent_id])

    def convert_to_continuous_action(self, action):
        steering = action[0] * self.steering_unit - 1.0
        throttle = action[1] * self.throttle_unit - 1.0
        return steering, throttle
