import copy

from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.manager.base_manager import BaseManager
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.utils.waymo_utils.parse_object_state import parse_vehicle_state
from metadrive.utils.waymo_utils.waymo_type import WaymoAgentType


class WaymoTrafficManager(BaseManager):
    def __init__(self):
        super(WaymoTrafficManager, self).__init__()
        self.vid_to_obj = None

    def after_reset(self):
        # try:
        # generate vehicle
        self.vid_to_obj = {}
        for v_id, type_traj in self.current_traffic_data.items():
            if WaymoAgentType.is_vehicle(type_traj["type"]) and v_id != self.sdc_track_index:
                info = parse_vehicle_state(
                    type_traj,
                    self.engine.global_config["traj_start_index"],
                    coordinate_transform=self.engine.global_config["coordinate_transform"]
                )
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
                if info["vehicle_class"]:
                    vehicle_class = info["vehicle_class"]
                else:
                    vehicle_class = SVehicle
                obj_name = v_id if self.engine.global_config["force_reuse_object_name"] else None
                v = self.spawn_object(
                    vehicle_class,
                    position=info["position"],
                    heading=info["heading"],
                    vehicle_config=v_config,
                    name=obj_name
                )
                self.vid_to_obj[v_id] = v.name
                v.set_velocity(info["velocity"])
                if "angular_velocity" in info:
                    v.set_angular_velocity(info["angular_velocity"])

                # v.set_static(True)
        # except:
        #     raise ValueError("Can not LOAD traffic for seed: {}".format(self.engine.global_random_seed))

    def after_step(self, *args, **kwargs):
        episode_step = self.engine.episode_step
        try:
            # generate vehicle
            for v_id, type_traj in self.current_traffic_data.items():
                if v_id in self.vid_to_obj and self.vid_to_obj[v_id] in self.spawned_objects.keys():

                    vehicle = self.spawned_objects[self.vid_to_obj[v_id]]

                    info = parse_vehicle_state(
                        type_traj, episode_step, coordinate_transform=self.engine.global_config["coordinate_transform"]
                    )
                    time_end = episode_step > self.engine.global_config["traj_end_index"] and \
                               self.engine.global_config["traj_end_index"] != -1
                    if (not info["valid"] or time_end) and v_id in self.vid_to_obj:
                        self.clear_objects([self.vid_to_obj[v_id]])
                        self.vid_to_obj.pop(v_id)
                        continue

                    vehicle.set_position(info["position"])
                    vehicle.set_heading_theta(float(info["heading"]))
                    vehicle.set_velocity(info["velocity"])
                    if "throttle_brake" in info:
                        vehicle.set_throttle_brake(float(info["throttle_brake"]))
                    if "steering" in info:
                        vehicle.set_steering(float(info["steering"]))
                    if "angular_velocity" in info:
                        vehicle.set_angular_velocity(float(info["angular_velocity"]))

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
        return str(self.engine.data_manager.get_case(self.engine.global_random_seed)[SD.METADATA][SD.SDC_ID])
