import copy
from metadrive.component.vehicle.vehicle_type import XLVehicle, SVehicle, MVehicle, LVehicle
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.policy.replay_policy import NuPlanReplayTrafficParticipantPolicy

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.coordinates_shift import nuplan_to_metadrive_vector
from metadrive.utils.nuplan_utils.parse_object_state import parse_object_state


class NuPlanTrafficManager(BaseManager):
    def __init__(self):
        super(NuPlanTrafficManager, self).__init__()
        self.nuplan_id_to_obj_id = {}
        self.need_traffic = not self.engine.global_config["no_traffic"]
        self.need_pedestrian = not self.engine.global_config["no_pedestrian"]
        self._current_traffic_data = None
        self.available_type = self.get_available_type()

    def after_reset(self):
        self._current_traffic_data = self._get_current_traffic_data()
        assert self.engine.episode_step == 0
        self.nuplan_id_to_obj_id = {}
        for nuplan_id, obj_state in self._current_traffic_data[0].items():
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
            return

        vehicles_to_eliminate = self.nuplan_id_to_obj_id.keys() - self._current_traffic_data[self.engine.episode_step
        ].keys()
        for nuplan_id in list(vehicles_to_eliminate):
            self.clear_objects([self.nuplan_id_to_obj_id[nuplan_id]])
            self.nuplan_id_to_obj_id.pop(nuplan_id)

        for nuplan_id, obj_state in self._current_traffic_data[self.engine.episode_step].items():
            if obj_state.tracked_object_type not in self.available_type:
                continue
            state = parse_object_state(obj_state, self.engine.current_map.nuplan_center)
            if nuplan_id in self.nuplan_id_to_obj_id and \
                    self.nuplan_id_to_obj_id[nuplan_id] in self.spawned_objects.keys():
                policy = self.get_policy(self.nuplan_id_to_obj_id[nuplan_id])
                policy.act(state)
            else:
                if obj_state.tracked_object_type == TrackedObjectType.VEHICLE:
                    self.spawn_vehicle(state, nuplan_id)
                elif obj_state.tracked_object_type == TrackedObjectType.BICYCLE:
                    self.spawn_cyclist(state, nuplan_id)
                elif obj_state.tracked_object_type == TrackedObjectType.PEDESTRIAN:
                    self.spawn_pedestrian(state, nuplan_id)

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
            self.get_vehicle_type(state["length"]),
            position=nuplan_to_metadrive_vector(state["position"]),
            heading=state["heading"],
            vehicle_config=v_config,
        )
        self.nuplan_id_to_obj_id[nuplan_id] = v.name
        v.set_velocity(state["velocity"])
        self.add_policy(v.name, NuPlanReplayTrafficParticipantPolicy, v)

    def spawn_pedestrian(self, state, nuplan_id):
        obj = self.spawn_object(
            Pedestrian,
            position=state["position"],
            heading_theta=state["heading"],
        )
        self.nuplan_id_to_obj_id[nuplan_id] = obj.name
        obj.set_velocity(state["velocity"])
        self.add_policy(obj.name, NuPlanReplayTrafficParticipantPolicy, obj)

    def spawn_cyclist(self, state, nuplan_id):
        obj = self.spawn_object(
            Cyclist,
            position=state["position"],
            heading_theta=state["heading"],
        )
        self.nuplan_id_to_obj_id[nuplan_id] = obj.name
        obj.set_velocity(state["velocity"])
        self.add_policy(obj.name, NuPlanReplayTrafficParticipantPolicy, obj)

    def get_available_type(self):
        ret = []
        if self.need_traffic:
            ret.append(TrackedObjectType.VEHICLE)
        if self.need_pedestrian:
            ret += [TrackedObjectType.BICYCLE, TrackedObjectType.PEDESTRIAN]
        return ret

    def get_vehicle_type(self, length):
        return [LVehicle, MVehicle, SVehicle, XLVehicle][self.np_random.randint(4)]
