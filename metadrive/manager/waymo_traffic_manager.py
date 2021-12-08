import copy

import numpy as np

from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.coordinates_shift import waymo_2_metadrive_heading, waymo_2_metadrive_position
from metadrive.utils.waymo_utils.waymo_utils import AgentType


class WaymoTrafficManager(BaseManager):
    def __init__(self):
        super(WaymoTrafficManager, self).__init__()
        self.current_traffic_data = None
        self.count = 0
        self.sdc_index = None

    def reset(self):
        # generate vehicle
        self.count = 0
        for v_id, type_traj in self.current_traffic_data.items():
            if type_traj["type"] == AgentType.VEHICLE and v_id != self.sdc_index:
                info = self.parse_vehicle_state(type_traj["state"], 0)
                if not info["valid"]:
                    continue
                v_config = copy.deepcopy(self.engine.global_config["vehicle_config"])
                v_config["need_navigation"] = False
                v = self.spawn_object(
                    SVehicle, name=v_id, position=info["position"], heading=info["heading"], vehicle_config=v_config
                )
                v.set_static(True)

    @staticmethod
    def parse_vehicle_state(states, time_idx):
        ret = {}
        if time_idx >= len(states):
            time_idx = -1
        state = states[time_idx]
        ret["position"] = waymo_2_metadrive_position([state[0], state[1]])
        ret["length"] = state[3]
        ret["width"] = state[4]
        ret["heading"] = waymo_2_metadrive_heading(np.rad2deg(state[6]))
        ret["velocity"] = waymo_2_metadrive_position([state[7], state[8]])
        ret["valid"] = state[9]
        return ret

    def after_step(self, *args, **kwargs):
        # generate vehicle
        for v_id, type_traj in self.current_traffic_data.items():
            if v_id in self.spawned_objects.keys():
                info = self.parse_vehicle_state(type_traj["state"], self.count)
                if not info["valid"]:
                    continue
                self.spawned_objects[v_id].set_position(info["position"])
                self.spawned_objects[v_id].set_heading_theta(info["heading"], rad_to_degree=True)
        self.count += 1

    def before_reset(self):
        # clean previous episode data
        super(WaymoTrafficManager, self).before_reset()
        self.current_traffic_data = self.engine.data_manager.get_case(self.engine.global_random_seed)["tracks"]
        self.sdc_index = str(self.engine.data_manager.get_case(self.engine.global_random_seed)["sdc_index"])
