import copy
import numpy as np

from metadrive.component.traffic_light.scenario_traffic_light import ScenarioTrafficLight
from metadrive.type import MetaDriveType
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.manager.base_manager import BaseManager
import logging

logger = logging.getLogger(__file__)


class ScenarioLightManager(BaseManager):
    CLEAR_LIGHTS = False

    OBJECT_PREFIX = "traffic_light_"

    def __init__(self):
        super(ScenarioLightManager, self).__init__()
        self._scenario_id_to_obj_id = {}
        self._obj_id_to_scenario_id = {}
        self._lane_index_to_obj = {}
        self.skip_missing_light = self.engine.global_config["skip_missing_light"]
        self._episode_light_data = None

    def before_reset(self):
        super(ScenarioLightManager, self).before_reset()
        self._scenario_id_to_obj_id = {}
        self._lane_index_to_obj = {}
        self._obj_id_to_scenario_id = {}
        self._episode_light_data = self._get_episode_light_data()

    def after_reset(self):
        for scenario_lane_id, light_info in self._episode_light_data.items():
            if str(scenario_lane_id) not in self.engine.current_map.road_network.graph:
                logger.warning("Can not find lane for this traffic light. Skip!")
                if self.skip_missing_light:
                    continue
                else:
                    raise ValueError(
                        "Can not find lane for this traffic light. "
                        "Set skip_missing_light=True for skipping missing light!"
                    )
            lane_info = self.engine.current_map.road_network.graph[str(scenario_lane_id)]
            position = self._get_light_position(light_info)
            name = self.OBJECT_PREFIX + scenario_lane_id if self.engine.global_config["force_reuse_object_name"
                                                                                      ] else None
            traffic_light = self.spawn_object(ScenarioTrafficLight, lane=lane_info.lane, position=position, name=name)
            self._scenario_id_to_obj_id[scenario_lane_id] = traffic_light.id
            self._obj_id_to_scenario_id[traffic_light.id] = scenario_lane_id
            if self.engine.global_config["force_reuse_object_name"]:
                assert self.OBJECT_PREFIX + scenario_lane_id == traffic_light.id, (
                    "Original id should be assigned to traffic lights"
                )
            self._lane_index_to_obj[lane_info.lane.index] = traffic_light
            status = light_info[SD.TRAFFIC_LIGHT_STATUS][self.episode_step]
            traffic_light.set_status(status)

    def _get_light_position(self, light_info):
        if SD.TRAFFIC_LIGHT_POSITION in light_info:
            # New format where the position is a 3-dim vector.
            return light_info[SD.TRAFFIC_LIGHT_POSITION]

        else:
            index = np.where(light_info[SD.TRAFFIC_LIGHT_LANE] > 0)[0][0]
            return light_info[SD.TRAFFIC_LIGHT_POSITION][index]

    def after_step(self, *args, **kwargs):
        if self.episode_step >= self.current_scenario_length:
            return

        for scenario_light_id, light_id, in self._scenario_id_to_obj_id.items():
            light_obj = self.spawned_objects[light_id]
            status = self._episode_light_data[scenario_light_id][SD.TRAFFIC_LIGHT_STATUS][self.episode_step]
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
        for lane_id, light_info in self.current_scenario[SD.DYNAMIC_MAP_STATES].items():
            ret[lane_id] = copy.deepcopy(light_info[SD.STATE])
            ret[lane_id]["metadata"] = copy.deepcopy(light_info[SD.METADATA])

            if SD.TRAFFIC_LIGHT_POSITION in ret[lane_id]:
                # Old data format where position is a 2D array with shape [T, 2]
                traffic_light_position = ret[lane_id][SD.TRAFFIC_LIGHT_POSITION]

                if not np.any(ret[lane_id][SD.TRAFFIC_LIGHT_LANE].astype(bool)):
                    # This traffic light has no effect.
                    first_pos = -1
                else:
                    first_pos = np.argwhere(ret[lane_id][SD.TRAFFIC_LIGHT_LANE] != 0)[0, 0]
                traffic_light_position = traffic_light_position[first_pos]
            else:
                # New data format where position is a [3, ] array.
                traffic_light_position = light_info[SD.TRAFFIC_LIGHT_POSITION][:2]

            ret[lane_id][SD.TRAFFIC_LIGHT_POSITION] = traffic_light_position

            assert light_info[SD.TYPE] == MetaDriveType.TRAFFIC_LIGHT, "Can not handle {}".format(light_info[SD.TYPE])
        return ret

    def get_state(self):
        return {
            SD.OBJ_ID_TO_ORIGINAL_ID: copy.deepcopy(self._obj_id_to_scenario_id),
            SD.ORIGINAL_ID_TO_OBJ_ID: copy.deepcopy(self._scenario_id_to_obj_id)
        }
