import copy
import logging

from metadrive.component.blocks.base_block import BaseBlock
from metadrive.component.map.base_map import BaseMap, MapGenerateMethod
from metadrive.component.map.pg_map import PGMap
from metadrive.constants import ObjectState, REPLAY_DONE
from metadrive.manager.base_manager import BaseManager


class FrameInfo:
    def __init__(self, episode_step):
        self.episode_step = episode_step
        # used to track the objects spawn info
        self.spawn_info = {}
        # used to track update objects state in the scene
        self.step_info = {}
        # used to track the objects cleared
        self.clear_info = []
        self.agents = []
        self._object_to_agent = None
        self._agent_to_object = None

    def object_to_agent(self, object):
        return self._object_to_agent[object]

    def agent_to_object(self, agent):
        return self._agent_to_object[agent]


class RecordManager(BaseManager):
    """
    Record the episode information for replay
    """
    PRIORITY = 100  # lowest priority

    def __init__(self):
        super(RecordManager, self).__init__()
        self.episode_info = None
        self.current_frame = None

    def before_reset(self):
        self.episode_info = {}
        self.current_frame = FrameInfo(self.engine.episode_step)

    def after_reset(self):
        """
        create a new log to record, note: after_step will be called after calling after_reset()
        """
        if self.engine.record_episode:
            self.episode_info = dict(
                map_data=self.engine.current_map.save_map(),
                frame=[],
            )

    def before_step(self, *args, **kwargs):
        if self.engine.record_episode:
            self.current_frame = FrameInfo(self.engine.episode_step)
        return {}

    def after_step(self, *args, **kwargs):
        if self.engine.record_episode:
            self._update_objects_states()
            self.episode_info["frame"].append(self.current_frame)
            return {}

    def _update_objects_states(self):
        for name, obj in self.engine.get_objects().items():
            if not isinstance(obj, BaseBlock) and not isinstance(obj, BaseMap):
                self.current_frame.step_info[name] = obj.get_state()
        self.current_frame.agents = list(self.engine.agents.keys())
        self.current_frame._agent_to_object = self.engine.agent_manager._agent_to_object
        self.current_frame._object_to_agent = self.engine.agent_manager._object_to_agent

    def dump_episode(self):
        return copy.deepcopy(self.episode_info)

    def destroy(self):
        self.episode_info = None

    def add_spawn_info(self, object_class, kwargs, name):
        """
        Call when spawn new objects, ignore map related things
        """
        if not issubclass(object_class, BaseBlock) and not issubclass(object_class, BaseMap):
            assert name not in self.current_frame.spawn_info, "Duplicated record!"
            self.current_frame.spawn_info[name] = {ObjectState.CLASS: object_class, ObjectState.INIT_KWARGS: kwargs,
                                                   ObjectState.NAME: name}

    def add_clear_info(self, obj):
        """
        Call when clear objects, ignore map related things
        """
        if not isinstance(obj, BaseBlock) and not isinstance(obj, BaseMap):
            self.current_frame.clear_info.append(obj.name)

    def __del__(self):
        logging.debug("Record system is destroyed")


class ReplayManager(BaseManager):

    def __init__(self):
        super(ReplayManager, self).__init__()
        self.restore_episode_info = None
        self.current_map = None
        self.current_frame = None

    def before_reset(self, *args, **kwargs):
        """
        Clean generated objects
        """
        self.clear_objects([name for name in self.spawned_objects])

    def reset(self):
        if not self.engine.replay_episode:
            return
        self.restore_episode_info = self.engine.global_config["replay_episode"]
        self.restore_episode_info["frame"].reverse()
        # Since in episode data map data only contains one map, values()[0] is the map_parameters
        map_data = self.restore_episode_info["map_data"]
        assert len(map_data) > 0, "Can not find map info in episode data"

        map_config = copy.deepcopy(map_data["map_config"])
        map_config[BaseMap.GENERATE_TYPE] = MapGenerateMethod.PG_MAP_FILE
        map_config[BaseMap.GENERATE_CONFIG] = map_data["block_sequence"]
        self.current_map = self.spawn_object(PGMap, map_config=map_config, auto_fill_random_seed=False)
        self.replay_frame()

    def after_step(self, *args, **kwargs):
        if self.engine.replay_episode:
            return self.replay_frame()
        return dict(REPLAY_DONE=False)

    def destroy(self):
        self.restore_episode_info = None
        self.current_map = None

    def replay_frame(self):
        if len(self.restore_episode_info["frame"]) == 0:
            return {REPLAY_DONE: True}
        self.current_frame = self.restore_episode_info["frame"].pop()
        # create
        for name, config in self.current_frame.spawn_info.items():
            obj = self.spawn_object(config[ObjectState.CLASS], config[ObjectState.INIT_KWARGS])
            # self.change_object_name(obj, name)
        for name, state in self.current_frame.step_info.items():
            self.spawned_objects[name].before_step()
            self.spawned_objects[name].set_state(state)
        self.clear_objects(self.current_frame.clear_info)
        return {REPLAY_DONE: False}

    def __del__(self):
        logging.debug("Replay system is destroyed")

    def get_object_from_agent(self, agent_id):
        return self.spawned_objects[self.current_frame.agent_to_object(agent_id)]
