from metadrive.policy.idm_policy import IDMPolicy
class RandomLaneChangePolicy(IDMPolicy):
    def __init__(self, control_object, random_seed):
        super(RandomLaneChangePolicy, self).__init__(control_object, random_seed)
        self.random_lane_change = True
