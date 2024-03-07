from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.policy.idm_policy import IDMPolicy

class AdvPolicy(IDMPolicy):
    """
    This policy can protect Manual control and EnvInputControl
    """
    def act(self, agent_id):
        pass

    # TODO: load the given trained policy for adversary agents
