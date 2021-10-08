import copy
import logging

from metadrive.component.blocks.base_block import BaseBlock
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import PGMap
from metadrive.component.road.road import Road
from metadrive.constants import ObjectState
from metadrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT
from metadrive.engine.engine_utils import get_engine
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.traffic_manager import TrafficManager


class FrameInfo:
    def __init__(self, episode_step):
        self.episode_step = episode_step
        # used to track the objects spawn info
        self.spawn_info = {}
        # used to track update objects state in the scene
        self.step_info = {}
        # used to track the objects cleared
        self.clear_info = {}


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
        self.episode_info = dict(
            map_data=self.engine.current_map.save_map(),
            frame=[],
        )

    def before_step(self, *args, **kwargs):
        self.current_frame = FrameInfo(self.engine.episode_step)
        return {}

    def after_step(self, *args, **kwargs):
        self._update_objects_states()
        self.episode_info["frame"].append(self.current_frame)
        return {}

    def _update_objects_states(self):
        for name, obj in self.engine.get_objects().items():
            if not isinstance(obj, BaseBlock) and not isinstance(obj, BaseMap):
                self.current_frame.step_info[name] = obj.get_state()

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
            self.current_frame.clear_info = {ObjectState.NAME: obj.name}

    def __del__(self):
        logging.debug("Record system is destroyed")


class ReplayManager(BaseManager):

    def __init__(self):
        super(ReplayManager, self).__init__()
        self.restore_episode_info = None

    

    def __del__(self):
        logging.debug("Replay system is destroyed")
