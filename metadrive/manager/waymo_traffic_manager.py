import copy

from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.waymo_utils.parse_object_state import parse_vehicle_state
from metadrive.utils.waymo_utils.waymo_type import WaymoAgentType


class WaymoTrafficManager(BaseManager):
    def __init__(self):
        super(WaymoTrafficManager, self).__init__()
        # self.current_traffic_data = None
        self.count = 0
        # self.sdc_track_index = None
        self.vid_to_obj = None

    def after_reset(self):
        # try:
        # generate vehicle
        self.count = 0
        self.vid_to_obj = {}
        for v_id, type_traj in self.current_traffic_data.items():
            if WaymoAgentType.is_vehicle(type_traj["type"]) and v_id != self.sdc_track_index:
                info = parse_vehicle_state(type_traj, self.engine.global_config["traj_start_index"])
                if not info["valid"]:
                    continue
                v_config = copy.deepcopy(self.engine.global_config["vehicle_config"])
                v_config["need_navigation"] = False
                v_config.update(
                    dict(
                        show_navi_mark=False,
                        show_dest_mark=False,
                        enable_reverse=False,
                        show_lidar=False,
                        show_lane_line_detector=False,
                        show_side_detector=False,
                    )
                )
                v = self.spawn_object(
                    SVehicle, position=info["position"], heading=info["heading"], vehicle_config=v_config
                )
                self.vid_to_obj[v_id] = v.name
                v.set_velocity(info["velocity"])
                # v.set_static(True)
        # except:
        #     raise ValueError("Can not LOAD traffic for seed: {}".format(self.engine.global_random_seed))

    def after_step(self, *args, **kwargs):
        try:
            # generate vehicle
            for v_id, type_traj in self.current_traffic_data.items():
                if v_id in self.vid_to_obj and self.vid_to_obj[v_id] in self.spawned_objects.keys():
                    info = parse_vehicle_state(type_traj, self.count)
                    time_end = self.count > self.engine.global_config["traj_end_index"] and self.engine.global_config[
                        "traj_end_index"] != -1
                    if (not info["valid"] or time_end) and v_id in self.vid_to_obj:
                        self.clear_objects([self.vid_to_obj[v_id]])
                        self.vid_to_obj.pop(v_id)
                        continue
                    self.spawned_objects[self.vid_to_obj[v_id]].set_position(info["position"])
                    self.spawned_objects[self.vid_to_obj[v_id]].set_heading_theta(info["heading"])
                    self.spawned_objects[self.vid_to_obj[v_id]].set_velocity(info["velocity"])
            self.count += 1
        except:
            raise ValueError("Can not UPDATE traffic for seed: {}".format(self.engine.global_random_seed))

    def before_reset(self):
        try:
            # clean previous episode data
            super(WaymoTrafficManager, self).before_reset()
            # self.current_traffic_data = self.engine.data_manager.get_case(self.engine.global_random_seed)["tracks"]
            # self.sdc_track_index = str(self.engine.data_manager.get_case(self.engine.global_random_seed)["sdc_track_index"])
        except:
            raise ValueError("Can not CLEAN traffic for seed: {}".format(self.engine.global_random_seed))

    @property
    def current_traffic_data(self):
        return self.engine.data_manager.get_case(self.engine.global_random_seed)["tracks"]

    @property
    def sdc_track_index(self):
        return str(self.engine.data_manager.get_case(self.engine.global_random_seed)["sdc_track_index"])
