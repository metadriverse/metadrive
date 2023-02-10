import copy

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.coordinates_shift import nuplan_2_metadrive_position


class NuPlanTrafficManager(BaseManager):
    def __init__(self):
        super(NuPlanTrafficManager, self).__init__()
        self.vid_to_obj = {}
        self._current_traffic_data = None

    def after_reset(self):
        # try:
        # generate vehicle
        self._current_traffic_data = self._get_current_traffic_data()
        assert self.engine.episode_step == 0
        self.vid_to_obj = {}
        for v_id, obj_state in self._current_traffic_data[0].items():
            if obj_state.tracked_object_type != TrackedObjectType.VEHICLE:
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
                SVehicle,
                position=nuplan_2_metadrive_position(
                    [obj_state.center.x, obj_state.center.y], self.engine.current_map.nuplan_center
                ),
                heading=obj_state.center.heading * 180 / np.pi,
                vehicle_config=v_config
            )
            self.vid_to_obj[v_id] = v.name
            # v.set_static(True)
            v.set_velocity([obj_state.velocity.x, obj_state.velocity.y])

    # except:
    #     raise ValueError("Can not LOAD traffic for seed: {}".format(self.engine.global_random_seed))

    def after_step(self, *args, **kwargs):
        # try:
        # generate vehicle
        if self.episode_step >= self.current_scenario_length:
            return

        vehicles_to_eliminate = self.vid_to_obj.keys() - self._current_traffic_data[self.engine.episode_step].keys()
        for v_id in list(vehicles_to_eliminate):
            self.clear_objects([self.vid_to_obj[v_id]])
            self.vid_to_obj.pop(v_id)

        for v_id, obj_state in self._current_traffic_data[self.engine.episode_step].items():
            if obj_state.tracked_object_type != TrackedObjectType.VEHICLE:
                continue
            if v_id in self.vid_to_obj and self.vid_to_obj[v_id] in self.spawned_objects.keys():
                self.spawned_objects[self.vid_to_obj[v_id]].set_position(
                    nuplan_2_metadrive_position(
                        [obj_state.center.x, obj_state.center.y], self.engine.current_map.nuplan_center
                    )
                )
                self.spawned_objects[self.vid_to_obj[v_id]].set_heading_theta(
                    obj_state.center.heading, rad_to_degree=True
                )
                self.spawned_objects[self.vid_to_obj[v_id]].set_velocity([obj_state.velocity.x, obj_state.velocity.y])
            else:
                # spawn
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
                    SVehicle,
                    position=nuplan_2_metadrive_position(
                        [obj_state.center.x, obj_state.center.y], self.engine.current_map.nuplan_center
                    ),
                    heading=obj_state.center.heading,
                    vehicle_config=v_config
                )
                self.vid_to_obj[v_id] = v.name
                # v.set_static(True)
                v.set_velocity([obj_state.velocity.x, obj_state.velocity.y])
        # except:
        #     raise ValueError("Can not UPDATE traffic for seed: {}".format(self.engine.global_random_seed))

    @property
    def current_scenario(self):
        return self.engine.data_manager.current_scenario

    def _get_current_traffic_data(self):
        length = self.engine.data_manager.current_scenario.get_number_of_iterations()
        detection_ret = {
            i: self.engine.data_manager.current_scenario.get_tracked_objects_at_iteration(i).tracked_objects
            for i in range(length)
        }
        for step, frame_data in detection_ret.items():
            new_frame_data = {}
            for obj in frame_data:
                new_frame_data[obj.track_token] = obj
            detection_ret[step] = new_frame_data
        return detection_ret

    @property
    def current_scenario_length(self):
        return self.engine.data_manager.current_scenario_length
