import copy

from metadrive.component.traffic_light.scenario_traffic_light import ScenarioTrafficLight
from metadrive.utils.coordinates_shift import right_hand_to_left_vector
from metadrive.type import MetaDriveType
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.manager.base_manager import BaseManager


class ScenarioLightManager(BaseManager):
    CLEAR_LIGHTS = False

    def __init__(self):
        super(ScenarioLightManager, self).__init__()
        self._scenario_id_to_obj = {}
        self._obj_id_to_scenario_id = {}
        self._lane_index_to_obj = {}
        self._episode_light_data = None

    def before_reset(self):
        super(ScenarioLightManager, self).before_reset()
        self._scenario_id_to_obj = {}
        self._lane_index_to_obj = {}
        self._obj_id_to_scenario_id = {}
        self._episode_light_data = self._get_episode_light_data()

    def after_reset(self):
        for scenario_lane_id, light_info in self._episode_light_data.items():
            lane_info = self.engine.current_map.road_network.graph[str(scenario_lane_id)]
            position = light_info[ScenarioDescription.TRAFFIC_LIGHT_POSITION][0]
            name = scenario_lane_id if self.engine.global_config["force_reuse_object_name"] else None
            traffic_light = self.spawn_object(ScenarioTrafficLight, lane=lane_info.lane, position=position, name=name)
            self._scenario_id_to_obj[scenario_lane_id] = traffic_light
            self._obj_id_to_scenario_id[traffic_light.id] = scenario_lane_id
            if self.engine.global_config["force_reuse_object_name"]:
                assert scenario_lane_id == traffic_light.id, "Original id should be assigned to traffic lights"
            self._lane_index_to_obj[lane_info.lane.index] = traffic_light
            status = light_info[ScenarioDescription.TRAFFIC_LIGHT_STATUS][self.episode_step]
            traffic_light.set_status(status)

    def after_step(self, *args, **kwargs):
        if self.episode_step >= self.current_scenario_length:
            return

        for scenario_light_id, light_obj in self._scenario_id_to_obj.items():
            status = self._episode_light_data[scenario_light_id][
                ScenarioDescription.TRAFFIC_LIGHT_STATUS][self.episode_step]
            light_obj.set_status(status)

    def has_traffic_light(self, lane_index):
        return True if lane_index in self._lane_index_to_obj else False

    @property
    def current_scenario(self):
        return self.engine.data_manager.current_scenario

    @property
    def current_scenario_length(self):
        return self.engine.data_manager.current_scenario_length

    def _get_episode_light_data(self):
        ret = dict()
        for lane_id, light_info in self.current_scenario[ScenarioDescription.DYNAMIC_MAP_STATES].items():
            ret[lane_id] = copy.deepcopy(light_info[ScenarioDescription.STATE])
            if self.engine.data_manager.coordinate_transform:
                # ignore height and convert coordinate, if necessary
                ret[lane_id][ScenarioDescription.TRAFFIC_LIGHT_POSITION] = right_hand_to_left_vector(
                    ret[lane_id][ScenarioDescription.TRAFFIC_LIGHT_POSITION])[..., :-1]
            type = light_info[ScenarioDescription.TYPE]
            assert type == MetaDriveType.TRAFFIC_LIGHT, "Can not handle {}".format(type)
        return ret
