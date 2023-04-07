import copy

from metadrive.component.traffic_light.nuplan_traffic_light import NuplanTrafficLight
from metadrive.manager.base_manager import BaseManager
from metadrive.scenario.scenario_description import ScenarioDescription as SD


class NuPlanLightManager(BaseManager):
    CLEAR_LIGHTS = False

    def __init__(self):
        super(NuPlanLightManager, self).__init__()
        self._lane_to_lights = {}
        self.nuplan_id_to_obj_id = {}
        self.obj_id_to_nuplan_id = {}

        self._episode_light_data = None

    def before_reset(self):
        super(NuPlanLightManager, self).before_reset()
        self._lane_to_lights = {}
        self.nuplan_id_to_obj_id = {}
        self.obj_id_to_nuplan_id = {}

        self._episode_light_data = self._get_episode_light_data()

    def after_reset(self):
        for light in self._episode_light_data[0]:
            lane_info = self.engine.current_map.road_network.graph[str(light.lane_connector_id)]
            traffic_light = self.spawn_object(NuplanTrafficLight, lane=lane_info.lane)
            self._lane_to_lights[lane_info.lane.index] = traffic_light
            self.nuplan_id_to_obj_id[str(light.lane_connector_id)] = traffic_light.name
            self.obj_id_to_nuplan_id[traffic_light.name] = str(light.lane_connector_id)
            traffic_light.set_status(light.status)

    def after_step(self, *args, **kwargs):
        if self.episode_step >= self.current_scenario_length:
            return

        step_data = self._episode_light_data[self.engine.episode_step]

        if self.CLEAR_LIGHTS:
            light_to_eliminate = self._lane_to_lights.keys() - set([str(i.lane_connector_id) for i in step_data])
            for lane_id in light_to_eliminate:
                light = self._lane_to_lights.pop(lane_id)
                nuplan_id = self.obj_id_to_nuplan_id.pop(light.id)
                obj_id = self.nuplan_id_to_obj_id.pop(nuplan_id)
                assert obj_id == light.name
                self.clear_objects([obj_id])

        for light in self._episode_light_data[self.episode_step]:
            if str(light.lane_connector_id) in self._lane_to_lights:
                traffic_light = self._lane_to_lights[str(light.lane_connector_id)]
            else:
                try:
                    lane_info = self.engine.current_map.road_network.graph[str(light.lane_connector_id)]
                except KeyError:
                    continue
                traffic_light = self.spawn_object(NuplanTrafficLight, lane=lane_info.lane)
                assert str(light.lane_connector_id) == lane_info.lane.index
                self._lane_to_lights[lane_info.lane.index] = traffic_light
                self.nuplan_id_to_obj_id[str(light.lane_connector_id)] = traffic_light.name
                self.obj_id_to_nuplan_id[traffic_light.name] = str(light.lane_connector_id)
            traffic_light.set_status(light.status)

        assert len(self._lane_to_lights) == len(self.nuplan_id_to_obj_id) == len(self.obj_id_to_nuplan_id)

    def has_traffic_light(self, lane_index):
        return True if lane_index in self._lane_to_lights else False

    @property
    def current_scenario(self):
        return self.engine.data_manager.current_scenario

    def traffic_light_status_at(self, timestep):
        return self.current_scenario.get_traffic_light_status_at_iteration(timestep)

    @property
    def current_scenario_length(self):
        return self.engine.data_manager.current_scenario_length

    def _get_episode_light_data(self):
        length = self.engine.data_manager.current_scenario.get_number_of_iterations()
        ret = {i: [t for t in self.traffic_light_status_at(i)] for i in range(length)}
        return ret

    def get_state(self):
        return {
            SD.OBJ_ID_TO_ORIGINAL_ID: copy.deepcopy(self.obj_id_to_nuplan_id),
            SD.ORIGINAL_ID_TO_OBJ_ID: copy.deepcopy(self.nuplan_id_to_obj_id)
        }
