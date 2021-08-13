from pgdrive.policy.base_policy import BasePolicy


class EnvInputPolicy(BasePolicy):
    def __init__(self):
        # Since control object may change
        super(EnvInputPolicy, self).__init__(control_object=None)

    def act(self, agent_id):
        return self.engine.external_actions[agent_id]
