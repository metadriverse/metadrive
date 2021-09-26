from metadrive.examples import expert
from metadrive.policy.base_policy import BasePolicy


class ExpertPolicy(BasePolicy):
    def act(self, agent_id=None):
        return expert(self.control_object)
