from metadrive.examples import expert
from metadrive.policy.base_policy import BasePolicy


class ExpertPolicy(BasePolicy):
    def act(self, agent_id=None):
        action = expert(self.control_object)
        self.action_info["action"] = action
        return action
