import copy

import numpy as np
from metadrive.utils.coordinates_shift import nuplan_2_metadrive_heading, nuplan_2_metadrive_position
from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.waymo_utils.waymo_utils import AgentType


class NuPlanTrafficManager(BaseManager):
    def __init__(self):
        super(NuPlanTrafficManager, self).__init__()

    # def after_reset(self):
    #     # try:
    #     # generate vehicle
    #     vehicles = self.engine.data_manager.current_scenario
    #     self.count = 0
    #     self.vid_to_obj = {}
    #     for v_id, type_traj in self.current_traffic_data.items():
    #         if type_traj["type"] == AgentType.VEHICLE and v_id != self.sdc_index:
    #             info = self.parse_vehicle_state(type_traj["state"], self.engine.global_config["traj_start_index"])
    #             if not info["valid"]:
    #                 continue
    #             v_config = copy.deepcopy(self.engine.global_config["vehicle_config"])
    #             v_config["need_navigation"] = False
    #             v_config.update(
    #                 dict(
    #                     show_navi_mark=False,
    #                     show_dest_mark=False,
    #                     enable_reverse=False,
    #                     show_lidar=False,
    #                     show_lane_line_detector=False,
    #                     show_side_detector=False,
    #                 )
    #             )
    #             v = self.spawn_object(
    #                 SVehicle, position=info["position"], heading=info["heading"], vehicle_config=v_config
    #             )
    #             self.vid_to_obj[v_id] = v.name
    #             v.set_static(True)
    #         elif type_traj["type"] == AgentType.VEHICLE and v_id == self.sdc_index:
    #             # set Ego V velocity
    #             info = self.parse_vehicle_state(type_traj["state"], self.engine.global_config["traj_start_index"])
    #             ego_v = list(self.engine.agent_manager.active_agents.values())[0]
    #             ego_v.set_velocity(info["velocity"])
    #             ego_v.set_position(info["position"])
    # except:
    #     raise ValueError("Can not LOAD traffic for seed: {}".format(self.engine.global_random_seed))

    @staticmethod
    def parse_vehicle_state(state):
        ret = {}
        ret["position"] = nuplan_2_metadrive_position([state.waypoint.x, state.waypoint.y])
        ret["heading"] = nuplan_2_metadrive_heading(np.rad2deg(state.waypoint.heading))
        ret["velocity"] = nuplan_2_metadrive_position([state.waypoint.velocity.x, state.waypoint.velocity.y])
        ret["valid"] = True
        return ret

    # def after_step(self, *args, **kwargs):
    #     try:
    #         # generate vehicle
    #         for v_id, type_traj in self.current_traffic_data.items():
    #             if v_id in self.vid_to_obj and self.vid_to_obj[v_id] in self.spawned_objects.keys():
    #                 info = self.parse_vehicle_state(type_traj["state"], self.count)
    #                 time_end = self.count > self.engine.global_config["traj_end_index"] and self.engine.global_config[
    #                     "traj_end_index"] != -1
    #                 if (not info["valid"] or time_end) and v_id in self.vid_to_obj:
    #                     self.clear_objects([self.vid_to_obj[v_id]])
    #                     self.vid_to_obj.pop(v_id)
    #                     continue
    #                 self.spawned_objects[self.vid_to_obj[v_id]].set_position(info["position"])
    #                 self.spawned_objects[self.vid_to_obj[v_id]].set_heading_theta(info["heading"], rad_to_degree=False)
    #         self.count += 1
    #     except:
    #         raise ValueError("Can not UPDATE traffic for seed: {}".format(self.engine.global_random_seed))
    #
    # def before_reset(self):
    #     try:
    #         # clean previous episode data
    #         super(NuPlanTrafficManager, self).before_reset()
    #         # self.current_traffic_data = self.engine.data_manager.get_case(self.engine.global_random_seed)["tracks"]
    #         # self.sdc_index = str(self.engine.data_manager.get_case(self.engine.global_random_seed)["sdc_index"])
    #     except:
    #         raise ValueError("Can not CLEAN traffic for seed: {}".format(self.engine.global_random_seed))

    @staticmethod
    def parse_full_trajectory(states):
        traj = []
        for state in states:
            traj.append([state.waypoint.x, state.waypoint.y])

        trajectory = np.array(traj)
        trajectory *= [1, -1]
        return trajectory

    @property
    def current_scenario(self):
        return self.engine.data_manager.current_scenario
