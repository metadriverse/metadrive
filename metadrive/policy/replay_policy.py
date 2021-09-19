import numpy as np

from metadrive.policy.base_policy import BasePolicy

has_rendered = False


class ReplayPolicy(BasePolicy):
    def __init__(self, control_object, locate_info):
        super(ReplayPolicy, self).__init__(control_object=control_object)
        self.traj_info = locate_info["traj"]
        self.start_index = min(self.traj_info.keys())
        self.init_pos = locate_info["init_pos"]
        self.heading = locate_info["heading"]
        self.timestep = 0
        self.damp = 0
        # how many times the replay data is slowed down
        self.damp_interval = 1

    def act(self, *args, **kwargs):
        self.damp += self.damp_interval
        if self.damp == self.damp_interval:
            self.timestep += 1
            self.damp = 0
        else:
            return [0, 0]

        if str(self.timestep) == self.start_index:
            self.control_object.set_position(self.init_pos)
        elif str(self.timestep) in self.traj_info.keys():
            self.control_object.set_position(self.traj_info[str(self.timestep)])

        if self.heading is None or str(self.timestep - 1) not in self.heading.keys():
            pass
        else:
            this_heading = self.heading[str(self.timestep - 1)]
            self.control_object.set_heading_theta(np.arctan2(this_heading[0], this_heading[1]) - np.pi / 2)

        return [0, 0]
