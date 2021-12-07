import copy

import numpy as np
from metadrive.utils.coordinates_shift import waymo_2_metadrive_heading, waymo_2_metadrive_position
from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.manager.base_manager import BaseManager


class WaymoTrafficManager(BaseManager):
    def __init__(self):
        super(WaymoTrafficManager, self).__init__()
        self.current_traffic_data = None

    def before_reset(self):
        # clean previous episode data
        super(WaymoTrafficManager, self).before_reset()
        self.current_traffic_data = self.engine.data_manager.cases[self.engine.global_random_seed]["tracks"]

    def reset(self):
        # generate vehicle
        for v_id, type_traj in self.current_traffic_data.items():
            info = self.parse_vehicle_state(type_traj["state"], 0)
            self.spawn_object(SVehicle, name=v_id, position=info["position"], heading=info["heading"],
                              vehicle_config=copy.deepcopy(self.engine.global_config["vehicle_config"]))

    @staticmethod
    def parse_vehicle_state(states, time_idx):
        ret = {}
        state = states[time_idx]
        ret["position"] = waymo_2_metadrive_position([state[0], state[1]])
        ret["length"] = state[3]
        ret["width"] = state[4]
        ret["heading"] = waymo_2_metadrive_heading(np.rad2deg(state[6]))
        ret["velocity"] = waymo_2_metadrive_position([state[7], state[8]])
        ret["valid"] = state[9]
        return ret
