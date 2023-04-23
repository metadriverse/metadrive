import copy

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.vehicle.vehicle_type import get_vehicle_type
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.policy.replay_policy import NuPlanReplayTrafficParticipantPolicy
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.utils.nuplan.parse_object_state import parse_object_state


class NuPlanTrafficManager(BaseManager):
    EGO_TOKEN = "ego"

    def __init__(self):
        super(NuPlanTrafficManager, self).__init__()
        self.nuplan_id_to_obj_id = {}
        self.obj_id_to_nuplan_id = {}
        self.need_traffic = not self.engine.global_config["no_traffic"]
        self.need_pedestrian = not self.engine.global_config["no_pedestrian"]
        self._episode_traffic_data = None
        self.available_type = self.get_available_type()

    def after_reset(self):
        self._episode_traffic_data = self._get_episode_traffic_data()
        assert self.engine.episode_step == 0
        # according to scenario.initial_ego_state, the ego token is ego
        self.nuplan_id_to_obj_id = {self.EGO_TOKEN: self.engine.agents[DEFAULT_AGENT].id}
        self.obj_id_to_nuplan_id = {self.engine.agents[DEFAULT_AGENT].id: self.EGO_TOKEN}
        for nuplan_id, obj_state in self._episode_traffic_data[0].items():
            if obj_state.tracked_object_type == TrackedObjectType.VEHICLE and self.need_traffic:
                state = parse_object_state(obj_state, self.engine.current_map.nuplan_center)
                self.spawn_vehicle(state, nuplan_id)
            elif obj_state.tracked_object_type == TrackedObjectType.BICYCLE and self.need_pedestrian:
                state = parse_object_state(obj_state, self.engine.current_map.nuplan_center)
                self.spawn_cyclist(state, nuplan_id)
            elif obj_state.tracked_object_type == TrackedObjectType.PEDESTRIAN and self.need_pedestrian:
                state = parse_object_state(obj_state, self.engine.current_map.nuplan_center)
                self.spawn_pedestrian(state, nuplan_id)

    def after_step(self, *args, **kwargs):
        if self.episode_step >= self.current_scenario_length:
            return dict(default_agent=dict(replay_done=True))

        vehicles_to_eliminate = self.nuplan_id_to_obj_id.keys() - self._episode_traffic_data[self.engine.episode_step
                                                                                             ].keys()

        for nuplan_id, obj_state in self._episode_traffic_data[self.engine.episode_step].items():
            if obj_state.tracked_object_type not in self.available_type:
                continue
            state = parse_object_state(obj_state, self.engine.current_map.nuplan_center)
            if nuplan_id in self.nuplan_id_to_obj_id and \
                    self.nuplan_id_to_obj_id[nuplan_id] in self.spawned_objects.keys():
                if self.is_outlier(nuplan_id):
                    vehicles_to_eliminate.add(nuplan_id)
                    continue
                policy = self.get_policy(self.nuplan_id_to_obj_id[nuplan_id])
                policy.act(state)
                # TODO LQY: when using IDM policy, consider add after_step_call
                # policy.control_object.after_step()
            else:
                if obj_state.tracked_object_type == TrackedObjectType.VEHICLE:
                    self.spawn_vehicle(state, nuplan_id)
                elif obj_state.tracked_object_type == TrackedObjectType.BICYCLE:
                    self.spawn_cyclist(state, nuplan_id)
                elif obj_state.tracked_object_type == TrackedObjectType.PEDESTRIAN:
                    self.spawn_pedestrian(state, nuplan_id)

        for nuplan_id in list(vehicles_to_eliminate):
            if nuplan_id != self.EGO_TOKEN:
                self.clear_objects([self.nuplan_id_to_obj_id[nuplan_id]])
                obj_id = self.nuplan_id_to_obj_id.pop(nuplan_id)
                assert nuplan_id == self.obj_id_to_nuplan_id.pop(obj_id)

        assert len(self.nuplan_id_to_obj_id) == len(self.obj_id_to_nuplan_id)
        return dict(default_agent=dict(replay_done=False))

    @property
    def current_scenario(self):
        return self.engine.data_manager.current_scenario

    def _get_episode_traffic_data(self):
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

    def spawn_vehicle(self, state, nuplan_id):
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
                spawn_velocity=state["velocity"],
                spawn_velocity_car_frame=False,
            )
        )
        v = self.spawn_object(
            get_vehicle_type(state["length"], self.np_random),
            position=state["position"],
            heading=state["heading"],
            vehicle_config=v_config,
        )
        self.nuplan_id_to_obj_id[nuplan_id] = v.name
        self.obj_id_to_nuplan_id[v.name] = nuplan_id
        v.set_velocity(state["velocity"])
        v.set_position(state["position"], 0.5)
        self.add_policy(v.name, NuPlanReplayTrafficParticipantPolicy, v)

    def spawn_pedestrian(self, state, nuplan_id):
        obj = self.spawn_object(
            Pedestrian,
            position=state["position"],
            heading_theta=state["heading"],
        )
        self.nuplan_id_to_obj_id[nuplan_id] = obj.name
        self.obj_id_to_nuplan_id[obj.name] = nuplan_id
        obj.set_velocity(state["velocity"])
        self.add_policy(obj.name, NuPlanReplayTrafficParticipantPolicy, obj)

    def spawn_cyclist(self, state, nuplan_id):
        obj = self.spawn_object(
            Cyclist,
            position=state["position"],
            heading_theta=state["heading"],
        )
        self.nuplan_id_to_obj_id[nuplan_id] = obj.name
        self.obj_id_to_nuplan_id[obj.name] = nuplan_id
        obj.set_velocity(state["velocity"])
        self.add_policy(obj.name, NuPlanReplayTrafficParticipantPolicy, obj)

    def get_available_type(self):
        ret = []
        if self.need_traffic:
            ret.append(TrackedObjectType.VEHICLE)
        if self.need_pedestrian:
            ret += [TrackedObjectType.BICYCLE, TrackedObjectType.PEDESTRIAN]
        return ret

    def is_outlier(self, nuplan_id):
        obj = self.spawned_objects[self.nuplan_id_to_obj_id[nuplan_id]]
        if obj.get_z() > 2. or abs(obj.roll) > np.pi / 12 or abs(obj.pitch) > np.pi / 12:
            return True
        else:
            return False

    def get_state(self):
        # Record mapping from original_id to new_id
        ret = {}
        ret[SD.ORIGINAL_ID_TO_OBJ_ID] = copy.deepcopy(self.nuplan_id_to_obj_id)
        ret[SD.OBJ_ID_TO_ORIGINAL_ID] = copy.deepcopy(self.obj_id_to_nuplan_id)
        return ret
