from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.nuplan.parse_object_state import parse_object_state


class NuplanParticipantManager(BaseManager):
    """
    This manager will control the walker and cyclist in the scenario
    """
    def __init__(self):
        raise DeprecationWarning("No all traffic participants are actuated by TrafficManager")
        super(NuplanParticipantManager, self).__init__()
        self.nuplan_id_to_obj = {}
        self._current_traffic_participants = None

    def after_reset(self):
        self._current_traffic_participants = self._get_current_traffic_participants()
        assert self.engine.episode_step == 0
        self.nuplan_id_to_obj = {}
        for nuplan_id, obj_state in self._current_traffic_participants[0].items():
            state = parse_object_state(obj_state, self.engine.current_map.nuplan_center)
            if obj_state.tracked_object_type == TrackedObjectType.BICYCLE or \
                    obj_state.tracked_object_type == TrackedObjectType.PEDESTRIAN:
                obj = self.spawn_object(
                    Cyclist if obj_state.tracked_object_type == TrackedObjectType.BICYCLE else Pedestrian,
                    position=state["position"],
                    heading_theta=state["heading"],
                )
                self.nuplan_id_to_obj[nuplan_id] = obj.name
                obj.set_velocity(state["velocity"])

    def after_step(self, *args, **kwargs):
        if self.episode_step >= self.current_scenario_length:
            return

        objs_to_eliminate = self.nuplan_id_to_obj.keys() - self._current_traffic_participants[self.engine.episode_step
                                                                                              ].keys()
        for nuplan_id in list(objs_to_eliminate):
            self.clear_objects([self.nuplan_id_to_obj[nuplan_id]])
            self.nuplan_id_to_obj.pop(nuplan_id)

        for nuplan_id, obj_state in self._current_traffic_participants[self.engine.episode_step].items():
            if obj_state.tracked_object_type != TrackedObjectType.PEDESTRIAN or \
                    obj_state.tracked_object_type != TrackedObjectType.BICYCLE:
                continue
            state = parse_object_state(obj_state, self.engine.current_map.nuplan_center)
            if nuplan_id in self.nuplan_id_to_obj and self.nuplan_id_to_obj[nuplan_id] in self.spawned_objects.keys():
                self.spawned_objects[self.nuplan_id_to_obj[nuplan_id]].set_position(state["position"])
                self.spawned_objects[self.nuplan_id_to_obj[nuplan_id]].set_heading_theta(state["heading"])
                self.spawned_objects[self.nuplan_id_to_obj[nuplan_id]].set_velocity(state["velocity"])
            else:
                obj = self.spawn_object(
                    Cyclist if obj_state.tracked_object_type == TrackedObjectType.BICYCLE else Pedestrian,
                    position=state["position"],
                    heading_theta=state["heading"],
                )
                self.nuplan_id_to_obj[nuplan_id] = obj.name
                obj.set_velocity(state["velocity"])

    @property
    def current_scenario(self):
        return self.engine.data_manager.current_scenario

    def _get_current_traffic_participants(self):
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
