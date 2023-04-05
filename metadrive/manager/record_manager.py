import copy
from metadrive.utils.utils import get_time_str
import logging

from metadrive.base_class.base_object import BaseObject
from metadrive.constants import ObjectState, PolicyState
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.utils import is_map_related_instance, is_map_related_class


class FrameInfo:
    def __init__(self, episode_step):
        self.episode_step = episode_step
        # used to track the objects spawn info
        self.spawn_info = {}
        self.policy_info = {}
        self.policy_spawn_info = {}
        # used to track update objects state in the scene
        self.step_info = {}
        # used to track the objects cleared
        self.clear_info = []
        # manager state
        self.manager_info = {}
        self.agents = []
        self._object_to_agent = None
        self._agent_to_object = None

    def object_to_agent(self, object):
        return self._object_to_agent[object]

    def agent_to_object(self, agent):
        return self._agent_to_object[agent]


class RecordManager(BaseManager):
    """
    Record the episode information for replay or reloading episode
    """
    PRIORITY = 100  # lowest priority

    def __init__(self):
        super(RecordManager, self).__init__()
        self.episode_info = None
        self.current_frames = None
        self.current_frame_count = 0
        self.reset_frame = None
        # for debug, we don't allow assign the same id to different vehicles
        # previous recycling mechanism will bring such issue, which is fixed now
        self._episode_obj_names = set()

    def before_reset(self):
        if self.engine.record_episode:
            self.episode_info = {}
            self._episode_obj_names = set()
            self.reset_frame = FrameInfo(self.engine.episode_step)

    def after_reset(self):
        """
        create a new log to record, note: after_step will be called after calling after_reset()
        """
        if self.engine.record_episode:
            self.episode_info = dict(
                map_data=self.engine.current_map.get_meta_data(),
                frame=[[self.reset_frame]],
                scenario_index=self.engine.global_seed,
                global_config=self.engine.global_config,
                global_seed=self.engine.global_seed,
                manager_metadata={},
                coordinate="MetaDrive",
                time=get_time_str()
            )

            self.collect_objects_states()
            self.collect_manager_states()
            self.collect_manager_metadata()
            self.current_frames = None
            self.reset_frame = None
            self.current_frame_count = 0

    def collect_manager_metadata(self):
        assert self.episode_step == 0, "This func can only be called after env.reset() without any env.step() called"
        ret = {}
        for manager in self.engine.managers.values():
            mgr_meta = manager.get_metadata()
            assert mgr_meta is not None, "No return value for manager.get_state()"
            ret[manager.class_name] = mgr_meta
        self.episode_info["manager_metadata"] = ret

    def collect_manager_states(self):
        ret = {}
        for manager in self.engine.managers.values():
            mgr_state = manager.get_state()
            assert mgr_state is not None, "No return value for manager.get_state()"
            ret[manager.class_name] = mgr_state
        self.current_frame.manager_info = ret

    def before_step(self, *args, **kwargs) -> dict:
        if self.engine.record_episode:
            self.current_frames = [
                FrameInfo(self.engine.episode_step) for _ in range(self.engine.global_config["decision_repeat"])
            ]
            self.current_frame_count = 0
        return {}

    def step(self, *args, **kwargs):
        # Note: Update object state must be written in step, because the simulator will step 5 times for each RL step.
        # We need to record the intermediate states.
        if self.engine.record_episode:
            self.collect_objects_states()
            self.collect_manager_states()
            self.current_frame_count += 1 if self.current_frame_count < len(self.current_frames) - 1 else 0

    def after_step(self, *args, **kwargs) -> dict:
        # frame count ==0 is the reset frame, so don't append
        if self.engine.record_episode and self.current_frame_count:
            self.step()
            assert len(self.current_frames) == self.engine.global_config["decision_repeat"], "Number of Frame Mismatch!"
            self.episode_info["frame"].append(self.current_frames)
        return {}

    def collect_objects_states(self):
        policy_mapping = self.engine.get_policies()
        for name, obj in self.engine.get_objects().items():
            if not is_map_related_instance(obj):
                self.current_frame.step_info[name] = obj.get_state()
                if name in policy_mapping:
                    self.current_frame.policy_info[name] = policy_mapping[name].get_state()

        self.current_frame.agents = list(self.engine.agents.keys())
        self.current_frame._agent_to_object = copy.deepcopy(self.engine.agent_manager._agent_to_object)
        self.current_frame._object_to_agent = copy.deepcopy(self.engine.agent_manager._object_to_agent)

    def get_episode_metadata(self):
        assert self.engine.record_episode, "Turn on recording episode and then dump it"
        return copy.deepcopy(self.episode_info)

    def destroy(self):
        self.episode_info = None

    def add_spawn_info(self, obj, object_class, kwargs):
        """
        Call when spawn new objects, ignore map related things
        """
        if not is_map_related_class(object_class) and self.engine.record_episode:
            name = obj.name
            assert name not in self.current_frame.spawn_info, "Duplicated record!"
            assert name not in self._episode_obj_names, "Duplicated name using!"
            self._episode_obj_names.add(name)
            self.current_frame.spawn_info[name] = {
                ObjectState.CLASS: object_class,
                ObjectState.INIT_KWARGS: kwargs,
                ObjectState.NAME: name
            }

            # update step info in after step so exempt the requirement for adding info here
            # self.current_frame.step_info[name] = obj.get_state()
            # policy_mapping = self.engine.get_policies()
            # if name in policy_mapping:
            #     self.current_frame.policy_info[name] = policy_mapping[name].get_state()

    def add_policy_info(self, name, policy_class, *args, **kwargs):
        """
        Call when spawn new objects, ignore map related stuff
        """
        filtered_args = []
        for arg in args:
            filtered_args.append(arg) if not isinstance(arg, BaseObject) else filtered_args.append(BaseObject)
        filtered_kwargs = {}
        for k, v in kwargs.items():
            filtered_kwargs[k] = v if not isinstance(v, BaseObject) else BaseObject
        if self.engine.record_episode:
            assert name not in self.current_frame.policy_spawn_info, "Duplicated record!"
            self.current_frame.policy_spawn_info[name] = {
                PolicyState.POLICY_CLASS: policy_class,
                PolicyState.ARGS: filtered_args,
                PolicyState.KWARGS: filtered_kwargs,
                PolicyState.OBJ_NAME: name
            }

    def add_clear_info(self, obj):
        """
        Call when clear objects, ignore map related things
        """
        if not is_map_related_instance(obj) and self.engine.record_episode and self.episode_step != 0:
            self.current_frame.clear_info.append(obj.name)

    def __del__(self):
        logging.debug("Record system is destroyed")

    @property
    def current_frame(self):
        return self.current_frames[self.current_frame_count] if self.reset_frame is None else self.reset_frame

    def set_state(self, state: dict, old_name_to_current=None):
        return {}

    def get_state(self):
        return {}
