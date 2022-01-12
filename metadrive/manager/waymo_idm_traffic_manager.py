import copy
import signal

import numpy as np

from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.manager.waymo_traffic_manager import WaymoTrafficManager
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils.scene_utils import ray_localization
from metadrive.utils.waymo_utils.waymo_utils import AgentType


def handler(signum, frame):
    raise Exception("end of time")


class WaymoIDMTrafficManager(WaymoTrafficManager):
    def __init__(self):
        super(WaymoIDMTrafficManager, self).__init__()
        self.vehicle_destination_map = {}

    def before_reset(self):
        super(WaymoIDMTrafficManager, self).before_reset()
        self.vehicle_destination_map = {}

    def reset(self):
        try:
            # generate vehicle
            self.count = 0
            for v_id, type_traj in self.current_traffic_data.items():
                if type_traj["type"] == AgentType.VEHICLE and v_id != self.sdc_index:
                    init_info = self.parse_vehicle_state(type_traj["state"], 0)
                    if not init_info["valid"]:
                        continue
                    dest_info = self.parse_vehicle_state(type_traj["state"], -1)
                    start, destinations = self.get_route(init_info, dest_info)
                    if start is None:
                        continue

                    v_config = copy.deepcopy(self.engine.global_config["vehicle_config"])
                    v_config.update(
                        dict(
                            show_navi_mark=False,
                            show_dest_mark=False,
                            enable_reverse=False,
                            show_lidar=False,
                            show_lane_line_detector=False,
                            show_side_detector=False,
                            spawn_lane_index=start,
                            destination=destinations[0]
                        )
                    )
                    v = self.spawn_object(
                        SVehicle, position=init_info["position"], heading=init_info["heading"], vehicle_config=v_config
                    )
                    self.vehicle_destination_map[v.id] = destinations
                    self.add_policy(v.id, IDMPolicy(v, self.generate_seed()))
                elif type_traj["type"] == AgentType.VEHICLE and v_id == self.sdc_index:
                    # set Ego V velocity
                    init_info = self.parse_vehicle_state(type_traj["state"], 0)
                    ego_v = list(self.engine.agent_manager.active_agents.values())[0]
                    ego_v.set_velocity(init_info["velocity"])
                    ego_v.set_position(init_info["position"])
        except:
            raise ValueError("Can not LOAD traffic for seed: {}".format(self.engine.global_random_seed))

    def before_step(self, *args, **kwargs):
        for v in self.spawned_objects.values():
            p = self.engine.get_policy(v.name)
            v.before_step(p.act())

    def after_step(self, *args, **kwargs):
        vehicles_to_clear = []
        for v in self.spawned_objects.values():
            if v.lane in self.vehicle_destination_map[v.id]:
                vehicles_to_clear.append(v)
        self.clear_objects(vehicles_to_clear)

    def get_route(self, init_state, last_state):
        init_position = init_state["position"]
        init_yaw = init_state["heading"]
        last_position = last_state["position"]
        last_yaw = last_state["heading"]
        start_lanes = ray_localization(
            (np.cos(init_yaw), np.sin(init_yaw)),
            init_position,
            self.engine,
            return_all_result=True,
            use_heading_filter=False
        )
        end_lanes = ray_localization(
            (np.cos(last_yaw), np.sin(last_yaw)),
            last_position,
            self.engine,
            return_all_result=True,
            use_heading_filter=False
        )

        start, end = self.filter_path(start_lanes, end_lanes)
        if start is None:
            return None, None
        lane = self.engine.current_map.road_network.get_lane(end)
        destinations = [end]
        if len(lane.left_lanes) > 0:
            destinations += [lane["id"] for lane in lane.left_lanes]
        if len(lane.right_lanes) > 0:
            destinations += [lane["id"] for lane in lane.right_lanes]
        return start, destinations

    def filter_path(self, start_lanes, end_lanes):
        # add some functions to store the filter information to avoid repeat filter when encountering the same cases
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(5)
            for start in start_lanes:
                for end in end_lanes:
                    path = self.engine.current_map.road_network.shortest_path(start[0].index, end[0].index)
                    if len(path) > 0:
                        signal.alarm(0)
                        return (start[0].index, end[0].index)
            signal.alarm(0)
            return None, None
        except:
            return None, None
