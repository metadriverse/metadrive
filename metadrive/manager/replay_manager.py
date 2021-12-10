import copy
import logging

from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import PGMap, MapGenerateMethod
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.constants import ObjectState
from metadrive.constants import REPLAY_DONE
from metadrive.manager.base_manager import BaseManager
from metadrive.obs.state_obs import LidarStateObservation


class ReplayManager(BaseManager):
    def __init__(self):
        super(ReplayManager, self).__init__()
        self.restore_episode_info = None
        self.current_map = None
        self.current_frame = None
        self.replay_done = False
        self.record_name_to_current_name = dict()
        self.current_name_to_record_name = dict()
        self.observation = self.get_observation()

    def before_reset(self, *args, **kwargs):
        """
        Clean generated objects
        """
        self.clear_objects([name for name in self.spawned_objects])
        self.replay_done = False
        return super(ReplayManager, self).before_reset()

    def reset(self):
        if not self.engine.replay_episode:
            return
        self.record_name_to_current_name = dict()
        self.current_name_to_record_name = dict()
        self.restore_episode_info = self.engine.global_config["replay_episode"]
        self.restore_episode_info["frame"].reverse()
        # Since in episode data map data only contains one map, values()[0] is the map_parameters
        map_data = self.restore_episode_info["map_data"]
        assert len(map_data) > 0, "Can not find map info in episode data"

        map_config = copy.deepcopy(map_data["map_config"])
        map_config[BaseMap.GENERATE_TYPE] = MapGenerateMethod.PG_MAP_FILE
        map_config[BaseMap.GENERATE_CONFIG] = map_data["block_sequence"]
        self.current_map = self.spawn_object(
            PGMap, map_config=map_config, auto_fill_random_seed=False, force_spawn=True
        )
        self.replay_frame()

    def step(self, *args, **kwargs):
        if self.engine.replay_episode:
            self.replay_frame()

    def after_step(self, *args, **kwargs):
        if self.engine.replay_episode:
            return self.engine.agent_manager.for_each_active_agents(lambda v: {REPLAY_DONE: self.replay_done})
        else:
            return dict()

    def destroy(self):
        self.record_name_to_current_name = dict()
        self.current_name_to_record_name = dict()
        self.restore_episode_info = None
        self.current_map = None

    def replay_frame(self):
        if len(self.restore_episode_info["frame"]) == 0:
            self.replay_done = True
            return
        self.current_frame = self.restore_episode_info["frame"].pop()
        # create
        for name, config in self.current_frame.spawn_info.items():
            if config[ObjectState.CLASS] == DefaultVehicle:
                config[ObjectState.INIT_KWARGS]["vehicle_config"]["use_special_color"] = True
            obj = self.spawn_object(object_class=config[ObjectState.CLASS], **config[ObjectState.INIT_KWARGS])
            self.current_name_to_record_name[obj.name] = name
            self.record_name_to_current_name[name] = obj.name
            if issubclass(config[ObjectState.CLASS], BaseVehicle):
                obj.navigation.set_route(
                    self.restore_episode_info["frame"][-1].step_info[name]["spawn_road"],
                    self.restore_episode_info["frame"][-1].step_info[name]["destination"][-1]
                )
        for name, state in self.current_frame.step_info.items():
            self.spawned_objects[self.record_name_to_current_name[name]].before_step()
            self.spawned_objects[self.record_name_to_current_name[name]].set_state(state)
            self.spawned_objects[self.record_name_to_current_name[name]].after_step()
        self.clear_objects([self.record_name_to_current_name[name] for name in self.current_frame.clear_info])
        self.replay_done = False

    @property
    def replay_agents(self):
        return {k: self.get_object_from_agent(k) for k in self.current_frame.agents}

    def __del__(self):
        logging.debug("Replay system is destroyed")

    def get_object_from_agent(self, agent_id):
        return self.spawned_objects[self.record_name_to_current_name[self.current_frame.agent_to_object(agent_id)]]

    def get_observation(self):
        """
        Override me in the future for collecting other modality
        """
        return LidarStateObservation(self.engine.global_config["vehicle_config"])

    def get_replay_agent_observations(self):
        return {k: self.observation for k in self.current_frame.agents}
