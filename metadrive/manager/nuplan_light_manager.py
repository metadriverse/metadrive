from metadrive.component.traffic_light.nuplan_traffic_light import NuplanTrafficLight
from metadrive.manager.base_manager import BaseManager


class NuPlanLightManager(BaseManager):
    def __init__(self):
        super(NuPlanLightManager, self).__init__()
        self._lane_to_lights = {}

    def before_reset(self):
        super(NuPlanLightManager, self).before_reset()
        self._lane_to_lights = {}

    def after_reset(self):
        for light in self.traffic_light_status_at(0):
            lane_info = self.engine.current_map.road_network.graph[str(light.lane_connector_id)]
            traffic_light = self.spawn_object(NuplanTrafficLight, lane=lane_info.lane)
            self._lane_to_lights[lane_info.lane.index] = traffic_light
            traffic_light.set_status(light.status)

    def after_step(self, *args, **kwargs):
        for light in self.traffic_light_status_at(0):
            if str(light.lane_connector_id) in self._lane_to_lights:
                traffic_light = self._lane_to_lights[str(light.lane_connector_id)]
            else:
                lane_info = self.engine.current_map.road_network.graph[str(light.lane_connector_id)]
                traffic_light = self.spawn_object(NuplanTrafficLight, lane=lane_info.lane)
                self._lane_to_lights[lane_info.lane.index] = traffic_light
            traffic_light.set_status(light.status)

    @property
    def current_scenario(self):
        return self.engine.data_manager.current_scenario

    def traffic_light_status_at(self, timestep):
        return self.current_scenario.get_traffic_light_status_at_iteration(timestep)

    @property
    def current_scenario_length(self):
        return self.engine.data_manager.current_scenario_length
