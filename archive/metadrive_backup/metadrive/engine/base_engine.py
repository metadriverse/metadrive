import os
import pickle
import time
from collections import OrderedDict
from typing import Callable, Optional, Union, List, Dict, AnyStr

import numpy as np

from metadrive.base_class.randomizable import Randomizable
from metadrive.constants import RENDER_MODE_NONE
from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.interface import Interface
from metadrive.engine.logger import get_logger, reset_logger

from metadrive.pull_asset import pull_asset
from metadrive.utils import concat_step_infos
from metadrive.utils.utils import is_map_related_class
from metadrive.version import VERSION, asset_version

logger = get_logger()


def generate_distinct_rgb_values():
    # Try to avoid (0,0,0) and (255,255,255) to avoid confusion with the background and other objects.
    r = np.linspace(16, 256 - 16, 16).astype(int)
    g = np.linspace(16, 256 - 16, 16).astype(int)
    b = np.linspace(16, 256 - 16, 16).astype(int)

    # Create a meshgrid and reshape to get all combinations of r, g, b
    rgbs = np.array(np.meshgrid(r, g, b)).T.reshape(-1, 3)

    # Normalize the values to be between 0 and 1
    rgbs = rgbs / 255.0

    return tuple(tuple(round(vv, 5) for vv in v) for v in rgbs)


COLOR_SPACE = generate_distinct_rgb_values()


class BaseEngine(EngineCore, Randomizable):
    """
    Due to the feature of Panda3D, BaseEngine should only be created once(Singleton Pattern)
    It is a pure game engine, which is not task-specific, while BaseEngine connects the
    driving task and the game engine modified from Panda3D Engine.
    """
    singleton = None
    global_random_seed = None

    MAX_COLOR = len(COLOR_SPACE)
    COLORS_OCCUPIED = set()
    COLORS_FREE = set(COLOR_SPACE)

    def __init__(self, global_config):
        self.c_id = dict()
        self.id_c = dict()
        self.try_pull_asset()
        EngineCore.__init__(self, global_config)
        Randomizable.__init__(self, self.global_random_seed)
        self.episode_step = 0
        BaseEngine.singleton = self
        self.interface = Interface(self)

        # managers
        self.task_manager = self.taskMgr  # use the inner TaskMgr of Panda3D as MetaDrive task manager
        self._managers = OrderedDict()

        # for recovering, they can not exist together
        self.record_episode = False
        self.replay_episode = False
        self.only_reset_when_replay = False
        # self.accept("s", self._stop_replay)

        # add camera or not
        self.main_camera = self.setup_main_camera()

        self._spawned_objects = dict()
        self._object_policies = dict()
        self._object_tasks = dict()

        # the clear function is a fake clear, objects cleared is stored for future use
        self._dying_objects = dict()

        # store external actions
        self.external_actions = None

        # topdown renderer
        self.top_down_renderer = None

        # warm up
        self.warmup()

        # curriculum reset
        self._max_level = self.global_config.get("curriculum_level", 1)
        self._current_level = 0
        self._num_scenarios_per_level = int(self.global_config.get("num_scenarios", 1) / self._max_level)

    def add_policy(self, object_id, policy_class, *args, **kwargs):
        policy = policy_class(*args, **kwargs)
        self._object_policies[object_id] = policy
        if self.record_episode:
            assert self.record_manager is not None, "No record manager"
            self.record_manager.add_policy_info(object_id, policy_class, *args, **kwargs)
        return policy

    def get_policy(self, object_id):
        """
        Return policy of specific object with id
        :param object_id: a filter function, only return objects satisfying this condition
        :return: policy
        """
        if object_id in self._object_policies:
            return self._object_policies[object_id]
        else:
            # print("Can not find the policy for object(id: {})".format(object_id))
            return None

    def has_policy(self, object_id, policy_cls=None):
        if policy_cls is None:
            return True if object_id in self._object_policies else False
        else:
            return True if object_id in self._object_policies and isinstance(
                self._object_policies[object_id], policy_cls
            ) else False

    def spawn_object(self, object_class, force_spawn=False, auto_fill_random_seed=True, record=True, **kwargs):
        """
        Call this func to spawn one object
        :param object_class: object class
        :param force_spawn: spawn a new object instead of fetching from _dying_objects list
        :param auto_fill_random_seed: whether to set random seed using purely random integer
        :param record: record the spawn information
        :param kwargs: class init parameters
        :return: object spawned
        """
        if ("random_seed" not in kwargs) and auto_fill_random_seed:
            kwargs["random_seed"] = self.generate_seed()
        if force_spawn or object_class.__name__ not in self._dying_objects or len(
                self._dying_objects[object_class.__name__]) == 0:
            obj = object_class(**kwargs)
        else:
            obj = self._dying_objects[object_class.__name__].pop()
            obj.reset(**kwargs)
            if not is_map_related_class(object_class) and ("name" not in kwargs or kwargs["name"] is None):
                obj.random_rename()

        if "name" in kwargs and kwargs["name"] is not None:
            assert kwargs["name"] == obj.name == obj.id
        if "id" in kwargs and kwargs["name"] is not None:
            assert kwargs["id"] == obj.id == obj.name

        if self.global_config["record_episode"] and not self.replay_episode and record:
            self.record_manager.add_spawn_info(obj, object_class, kwargs)
        self._spawned_objects[obj.id] = obj
        color = self._pick_color(obj.id)
        if color == (-1, -1, -1):
            raise ValueError(
                "No color available for object: {} instance segment mask. We already used all {} colors...".format(
                    obj.id, BaseEngine.MAX_COLOR
                )
            )

        obj.attach_to_world(self.worldNP, self.physics_world)
        return obj

    def _pick_color(self, id):
        """
        Return a color multiplier representing a unique color for an object if some colors are available.
        Return -1,-1,-1 if no color available

        SideEffect: COLOR_PTR will no longer point to the available color
        SideEffect: COLORS_OCCUPIED[COLOR_PTR] will not be avilable
        """
        if len(BaseEngine.COLORS_OCCUPIED) == BaseEngine.MAX_COLOR:
            return (-1, -1, -1)
        assert (len(BaseEngine.COLORS_FREE) > 0)
        my_color = BaseEngine.COLORS_FREE.pop()
        BaseEngine.COLORS_OCCUPIED.add(my_color)
        # print("After picking:", len(BaseEngine.COLORS_OCCUPIED), len(BaseEngine.COLORS_FREE))
        self.id_c[id] = my_color
        self.c_id[my_color] = id
        return my_color

    def _clean_color(self, id):
        """
        Relinquish a color once the object is focibly destroyed
        SideEffect:
        BaseEngins.COLORS_OCCUPIED += 1
        BaseEngine.COLOR_PTR now points to the idx just released
        BaseEngine.COLORS_RECORED
        Mapping Destroyed

        """
        if id in self.id_c.keys():
            my_color = self.id_c.pop(id)
            if my_color in BaseEngine.COLORS_OCCUPIED:
                BaseEngine.COLORS_OCCUPIED.remove(my_color)
            BaseEngine.COLORS_FREE.add(my_color)
            # print("After cleaning:,", len(BaseEngine.COLORS_OCCUPIED), len(BaseEngine.COLORS_FREE))
            # if id in self.id_c.keys():
            #     self.id_c.pop(id)
            assert my_color in self.c_id.keys()
            self.c_id.pop(my_color)

    def id_to_color(self, id):
        if id in self.id_c.keys():
            return self.id_c[id]
        else:
            print("Invalid ID: ", id)
            return -1, -1, -1

    def color_to_id(self, color):
        if color in self.c_id.keys():
            return self.c_id[color]
        else:
            print("Invalid color:", color)
            return "NA"

    def get_objects(self, filter: Optional[Union[Callable, List]] = None):
        """
        Return objects spawned, default all objects. Filter_func will be applied on all objects.
        It can be a id list or a function
        Since we don't expect a iterator, and the number of objects is not so large, we don't use built-in filter()
        :param filter: a filter function, only return objects satisfying this condition
        :return: return all objects or objects satisfying the filter_func
        """
        if filter is None:
            return self._spawned_objects
        elif isinstance(filter, (list, tuple)):
            return {id: self._spawned_objects[id] for id in filter}
        elif callable(filter):
            res = dict()
            for id, obj in self._spawned_objects.items():
                if filter(obj):
                    res[id] = obj
            return res
        else:
            raise ValueError("filter should be a list or a function")

    def get_policies(self):
        """
        Return a mapping from object ID to policy instance.
        """
        return self._object_policies

    def get_object(self, object_id):
        return self.get_objects([object_id])

    def clear_objects(self, filter: Optional[Union[Callable, List]], force_destroy=False, record=True):
        """
        Destroy all self-generated objects or objects satisfying the filter condition
        Since we don't expect a iterator, and the number of objects is not so large, we don't use built-in filter()
        If force_destroy=True, we will destroy this element instead of storing them for next time using

        filter: A list of object ids or a function returning a list of object id
        """
        """
        In addition, we need to remove a color mapping whenever an object is destructed.
        
        """
        force_destroy_this_obj = True if force_destroy or self.global_config["force_destroy"] else False

        if isinstance(filter, (list, tuple)):
            exclude_objects = {obj_id: self._spawned_objects[obj_id] for obj_id in filter}
        elif callable(filter):
            exclude_objects = dict()
            for id, obj in self._spawned_objects.items():
                if filter(obj):
                    exclude_objects[id] = obj
        else:
            raise ValueError("filter should be a list or a function")
        for id, obj in exclude_objects.items():
            self._clean_color(id)
            self._spawned_objects.pop(id)
            if id in self._object_tasks:
                self._object_tasks.pop(id)
            if id in self._object_policies:
                policy = self._object_policies.pop(id)
                policy.destroy()
            if force_destroy_this_obj:
                #self._clean_color(obj.id)
                obj.destroy()
            else:
                obj.detach_from_world(self.physics_world)

                # We might want to remove some episode-relevant information when recycling some objects
                if hasattr(obj, "before_reset"):
                    obj.before_reset()

                if obj.class_name not in self._dying_objects:
                    self._dying_objects[obj.class_name] = []
                # We have a limit for buffering objects
                if len(self._dying_objects[obj.class_name]) < self.global_config["num_buffering_objects"]:
                    self._dying_objects[obj.class_name].append(obj)
                else:
                    #self._clean_color(obj.id)
                    obj.destroy()
            if self.global_config["record_episode"] and not self.replay_episode and record:
                self.record_manager.add_clear_info(obj)
        return exclude_objects.keys()

    def clear_object_if_possible(self, obj, force_destroy):
        if isinstance(obj, dict):
            return
        if obj in self._spawned_objects:
            self.clear_objects([obj], force_destroy=force_destroy)
        if force_destroy and \
                obj.class_name in self._dying_objects and \
                obj in self._dying_objects[obj.class_name]:
            self._dying_objects[obj.class_name].remove(obj)
            if hasattr(obj, "destroy"):
                self._clean_color(obj.id)
                obj.destroy()
        del obj

    def reset(self):
        """
        Clear and generate the whole scene
        """
        # reset logger
        reset_logger()
        step_infos = {}

        # initialize
        self._episode_start_time = time.time()
        self.episode_step = 0
        if self.global_config["debug_physics_world"]:
            self.addTask(self.report_body_nums, "report_num")

        # Update record replay
        self.replay_episode = True if self.global_config["replay_episode"] is not None else False
        self.record_episode = self.global_config["record_episode"]
        self.only_reset_when_replay = self.global_config["only_reset_when_replay"]

        _debug_memory_usage = False

        if _debug_memory_usage:

            def process_memory():
                import psutil
                import os
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                return mem_info.rss

            cm = process_memory()

        # reset manager
        for manager_name, manager in self._managers.items():
            # clean all manager
            new_step_infos = manager.before_reset()
            step_infos = concat_step_infos([step_infos, new_step_infos])
            if _debug_memory_usage:
                lm = process_memory()
                if lm - cm != 0:
                    print("{}: Before Reset! Mem Change {:.3f}MB".format(manager_name, (lm - cm) / 1e6))
                cm = lm
        self.terrain.before_reset()
        self._object_clean_check()

        for manager_name, manager in self.managers.items():
            if self.replay_episode and self.only_reset_when_replay and manager is not self.replay_manager:
                # The scene will be generated from replay manager in only reset replay mode
                continue
            new_step_infos = manager.reset()
            step_infos = concat_step_infos([step_infos, new_step_infos])

            if _debug_memory_usage:
                lm = process_memory()
                if lm - cm != 0:
                    print("{}: Reset! Mem Change {:.3f}MB".format(manager_name, (lm - cm) / 1e6))
                cm = lm

        for manager_name, manager in self.managers.items():
            new_step_infos = manager.after_reset()
            step_infos = concat_step_infos([step_infos, new_step_infos])

            if _debug_memory_usage:
                lm = process_memory()
                if lm - cm != 0:
                    print("{}: After Reset! Mem Change {:.3f}MB".format(manager_name, (lm - cm) / 1e6))
                cm = lm

        # reset terrain
        # center_p = self.current_map.get_center_point() if isinstance(self.current_map, PGMap) else [0, 0]
        center_p = [0, 0]
        self.terrain.reset(center_p)

        # move skybox
        if self.sky_box is not None:
            self.sky_box.set_position(center_p)

        # refresh graphics to support multi-thread rendering, avoiding bugs like shadow disappearance at first frame
        for _ in range(5):
            self.graphicsEngine.renderFrame()

        # reset colors
        BaseEngine.COLORS_FREE = set(COLOR_SPACE)
        BaseEngine.COLORS_OCCUPIED = set()
        new_i2c = {}
        new_c2i = {}
        # print("rest objects", len(self.get_objects()))
        for object in self.get_objects().values():
            if object.id in self.id_c.keys():
                id = object.id
                color = self.id_c[object.id]
                BaseEngine.COLORS_OCCUPIED.add(color)
                BaseEngine.COLORS_FREE.remove(color)
                new_i2c[id] = color
                new_c2i[color] = id
        # print(len(BaseEngine.COLORS_FREE), len(BaseEngine.COLORS_OCCUPIED))
        self.c_id = new_c2i
        self.id_c = new_i2c
        return step_infos

    def before_step(self, external_actions: Dict[AnyStr, np.array]):
        """
        Entities make decision here, and prepare for step
        All entities can access this global manager to query or interact with others
        :param external_actions: Dict[agent_id:action]
        :return:
        """
        self.episode_step += 1
        step_infos = {}
        self.external_actions = external_actions
        for manager in self.managers.values():
            new_step_infos = manager.before_step()
            step_infos = concat_step_infos([step_infos, new_step_infos])
        return step_infos

    def step(self, step_num: int = 1) -> None:
        """
        Step the dynamics of each entity on the road.
        :param step_num: Decision of all entities will repeat *step_num* times
        """
        for i in range(step_num):
            # simulate or replay
            for name, manager in self.managers.items():
                if name != "record_manager":
                    manager.step()
            self.step_physics_world()
            # the recording should happen after step physics world
            if "record_manager" in self.managers and i < step_num - 1:
                # last recording should be finished in after_step(), as some objects may be created in after_step.
                # We repeat run simulator ```step_num``` frames, and record after each frame.
                # The recording of last frame is actually finished when all managers finish the ```after_step()```
                # function. So the recording for the last time should be done after that.
                # An example is that in ```PGTrafficManager``` we may create new vehicles in
                # ```after_step()``` of the traffic manager. Therefore, we can't record the frame before that.
                # These new cars' states can be recorded only if we run ```record_managers.step()```
                # after the creation of new cars and then can be recorded in ```record_managers.after_step()```
                self.record_manager.step()

            if self.force_fps.real_time_simulation and i < step_num - 1:
                self.task_manager.step()

        #  Do rendering
        self.task_manager.step()
        if self.on_screen_message is not None:
            self.on_screen_message.render()

    def after_step(self, *args, **kwargs) -> Dict:
        """
        Update states after finishing movement
        :return: if this episode is done
        """

        step_infos = {}
        if self.record_episode:
            assert list(self.managers.keys())[-1] == "record_manager", "Record Manager should have lowest priority"
        for manager in self.managers.values():
            new_step_info = manager.after_step(*args, **kwargs)
            step_infos = concat_step_infos([step_infos, new_step_info])
        self.interface.after_step()

        # === Option 1: Set episode_step to "num of calls to env.step"
        # We want to make sure that the episode_step is always aligned to the "number of calls to env.step"
        # So if this function is called in env.reset, we will not increment episode_step.
        # if call_from_reset:
        #     pass
        # else:
        #     self.episode_step += 1

        # === Option 2: Following old code.
        # Note that this function will be called in _get_reset_return.
        # Therefore, after reset the episode_step is immediately goes to 1
        # even if no env.step is called.

        # Episode_step should be increased before env.step(). I moved it to engine.before_step() now.

        # cull distant blocks
        # poses = [v.position for v in self.agent_manager.active_agents.values()]
        return step_infos

    def dump_episode(self, pkl_file_name=None) -> None:
        """Dump the data of an episode."""
        assert self.record_manager is not None
        episode_state = self.record_manager.get_episode_metadata()
        if pkl_file_name is not None:
            with open(pkl_file_name, "wb+") as file:
                pickle.dump(episode_state, file)
        return episode_state

    def close(self):
        """
        Note:
        Instead of calling this func directly, close Engine by using engine_utils.close_engine
        """
        if len(self._managers) > 0:
            for name, manager in self._managers.items():
                setattr(self, name, None)
                if manager is not None:
                    manager.destroy()
        # clear all objects in spawned_object
        # self.clear_objects([id for id in self._spawned_objects.keys()])
        for id, obj in self._spawned_objects.items():
            if id in self._object_policies:
                self._object_policies.pop(id).destroy()
            if id in self._object_tasks:
                self._object_tasks.pop(id).destroy()
            self._clean_color(obj.id)
            obj.destroy()
        for cls, pending_obj in self._dying_objects.items():
            for obj in pending_obj:
                self._clean_color(obj.id)
                obj.destroy()
        self._dying_objects = {}
        if self.main_camera is not None:
            self.main_camera.destroy()
        self.interface.destroy()
        self.close_engine()

        if self.top_down_renderer is not None:
            self.top_down_renderer.close()
            del self.top_down_renderer
            self.top_down_renderer = None

        Randomizable.destroy(self)

    def __del__(self):
        logger.debug("{} is destroyed".format(self.__class__.__name__))

    def _stop_replay(self):
        raise DeprecationWarning
        if not self.IN_REPLAY:
            return
        self.STOP_REPLAY = not self.STOP_REPLAY

    def register_manager(self, manager_name: str, manager):
        """
        Add a manager to BaseEngine, then all objects can communicate with this class
        :param manager_name: name shouldn't exist in self._managers and not be same as any class attribute
        :param manager: subclass of BaseManager
        """
        assert manager_name not in self._managers, "Manager {} already exists in BaseEngine, Use update_manager() to " \
                                                   "overwrite".format(manager_name)
        assert not hasattr(self, manager_name), "Manager name can not be same as the attribute in BaseEngine"
        self._managers[manager_name] = manager
        setattr(self, manager_name, manager)
        self._managers = OrderedDict(sorted(self._managers.items(), key=lambda k_v: k_v[-1].PRIORITY))

    def seed(self, random_seed):
        start_seed = self.gets_start_index(self.global_config)
        random_seed = ((random_seed - start_seed) % self._num_scenarios_per_level) + start_seed
        random_seed += self._current_level * self._num_scenarios_per_level
        self.global_random_seed = random_seed
        super(BaseEngine, self).seed(random_seed)
        for mgr in self._managers.values():
            mgr.seed(random_seed)

    @staticmethod
    def gets_start_index(config):
        start_seed = config.get("start_seed", None)
        start_scenario_index = config.get("start_scenario_index", None)
        assert start_seed is None or start_scenario_index is None, \
            "It is not allowed to define `start_seed` and `start_scenario_index`"
        if start_seed is not None:
            return start_seed
        elif start_scenario_index is not None:
            return start_scenario_index
        else:
            logger.warning("Can not find `start_seed` or `start_scenario_index`. Use 0 as `start_seed`")
            return 0

    @property
    def max_level(self):
        return self._max_level

    @property
    def current_level(self):
        return self._current_level

    def level_up(self):
        old_level = self._current_level
        self._current_level = min(self._current_level + 1, self._max_level - 1)
        if old_level != self._current_level:
            self.seed(self.current_seed + self._num_scenarios_per_level)

    @property
    def num_scenarios_per_level(self):
        return self._num_scenarios_per_level

    @property
    def current_map(self):
        if self.replay_episode:
            return self.replay_manager.current_map
        else:
            if hasattr(self, "map_manager"):
                return self.map_manager.current_map
            else:
                return None

    @property
    def current_track_agent(self):
        if self.main_camera is not None:
            return self.main_camera.current_track_agent
        elif "default_agent" in self.agents:
            return self.agents["default_agent"]
        else:
            return None

    @property
    def agents(self):
        if not self.replay_episode:
            return self.agent_manager.active_agents
        else:
            return self.replay_manager.replay_agents

    def setup_main_camera(self):
        from metadrive.engine.core.main_camera import MainCamera
        # Not we should always enable main camera RGBCamera will return incorrect result, as we are using PSSM!
        if self.mode != RENDER_MODE_NONE:
            return MainCamera(self, self.global_config["camera_height"], self.global_config["camera_dist"])
        else:
            return None

    @property
    def current_seed(self):
        return self.global_random_seed

    @property
    def global_seed(self):
        return self.global_random_seed

    def _object_clean_check(self):
        # objects check
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        from metadrive.component.static_object.traffic_object import TrafficObject
        for manager in self._managers.values():
            assert len(manager.spawned_objects) == 0

        objs_need_to_release = self.get_objects(
            filter=lambda obj: isinstance(obj, BaseVehicle) or isinstance(obj, TrafficObject)
        )
        assert len(
            objs_need_to_release) == 0, "You should clear all generated objects by using engine.clear_objects " \
                                        "in each manager.before_step()"

        # rigid body check
        bodies = []
        for world in [self.physics_world.dynamic_world, self.physics_world.static_world]:
            bodies += world.getRigidBodies()
            bodies += world.getSoftBodies()
            bodies += world.getGhosts()
            bodies += world.getVehicles()
            bodies += world.getCharacters()
            # bodies += world.getManifolds()

        filtered = []
        for body in bodies:
            if body.getName() in ["detector_mask", "debug"]:
                continue
            filtered.append(body)
        assert len(filtered) == 0, "Physics Bodies should be cleaned before manager.reset() is called. " \
                                   "Uncleared bodies: {}".format(filtered)

        children = self.worldNP.getChildren()
        assert len(children) == 0, "NodePath are not cleaned thoroughly. Remaining NodePath: {}".format(children)

    def update_manager(self, manager_name: str, manager, destroy_previous_manager=True):
        """
        Update an existing manager with a new one
        :param manager_name: existing manager name
        :param manager: new manager
        """
        assert manager_name in self._managers, "You may want to call register manager, since {} is not in engine".format(
            manager_name
        )
        existing_manager = self._managers.pop(manager_name)
        if destroy_previous_manager:
            existing_manager.destroy()
        self._managers[manager_name] = manager
        setattr(self, manager_name, manager)
        self._managers = OrderedDict(sorted(self._managers.items(), key=lambda k_v: k_v[-1].PRIORITY))

    @property
    def managers(self):
        # whether to froze other managers
        return {"replay_manager": self.replay_manager} if self.replay_episode and not \
            self.only_reset_when_replay else self._managers

    def object_to_agent(self, obj_name):
        if self.replay_episode:
            return self.replay_manager.current_frame.object_to_agent(obj_name)
        else:
            return self.agent_manager.object_to_agent(obj_name)

    def agent_to_object(self, agent_name):
        if self.replay_episode:
            return self.replay_manager.current_frame.agent_to_object(agent_name)
        else:
            return self.agent_manager.agent_to_object(agent_name)

    def render_topdown(self, text, *args, **kwargs):
        if self.top_down_renderer is None:
            from metadrive.engine.top_down_renderer import TopDownRenderer
            self.top_down_renderer = TopDownRenderer(*args, **kwargs)
        return self.top_down_renderer.render(text, *args, **kwargs)

    def _get_window_image(self, return_bytes=False):
        window_count = self.graphicsEngine.getNumWindows() - 1
        texture = self.graphicsEngine.getWindow(window_count).getDisplayRegion(0).getScreenshot()

        assert texture.getXSize() == self.global_config["window_size"][0], (
            texture.getXSize(), texture.getYSize(), self.global_config["window_size"]
        )
        assert texture.getYSize() == self.global_config["window_size"][1], (
            texture.getXSize(), texture.getYSize(), self.global_config["window_size"]
        )

        image_bytes = texture.getRamImage().getData()

        if return_bytes:
            return image_bytes, (texture.getXSize(), texture.getYSize())

        img = np.frombuffer(image_bytes, dtype=np.uint8)
        img = img.reshape((texture.getYSize(), texture.getXSize(), 4))
        img = img[::-1]  # Flip vertically
        img = img[..., :-1]  # Discard useless alpha channel
        img = img[..., ::-1]  # Correct the colors

        return img

    def warmup(self):
        """
        This function automatically initialize models/objects. It can prevent the lagging when creating some objects
        for the first time.
        """
        if self.global_config["preload_models"] and self.mode != RENDER_MODE_NONE:
            from metadrive.component.traffic_participants.pedestrian import Pedestrian
            from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
            from metadrive.component.static_object.traffic_object import TrafficBarrier
            from metadrive.component.static_object.traffic_object import TrafficCone
            Pedestrian.init_pedestrian_model()
            warm_up_pedestrian = self.spawn_object(Pedestrian, position=[0, 0], heading_theta=0, record=False)
            warm_up_light = self.spawn_object(BaseTrafficLight, lane=None, position=[0, 0], record=False)
            barrier = self.spawn_object(TrafficBarrier, position=[0, 0], heading_theta=0, record=False)
            cone = self.spawn_object(TrafficCone, position=[0, 0], heading_theta=0, record=False)
            for vel in Pedestrian.SPEED_LIST:
                warm_up_pedestrian.set_velocity([1, 0], vel - 0.1)
                self.taskMgr.step()
            self.clear_objects([warm_up_pedestrian.id, warm_up_light.id, barrier.id, cone.id], record=False)
            warm_up_pedestrian = None
            warm_up_light = None
            barrier = None
            cone = None

    @staticmethod
    def try_pull_asset():
        from metadrive.engine.asset_loader import AssetLoader
        msg = "Assets folder doesn't exist. Begin to download assets..."
        if not os.path.exists(AssetLoader.asset_path):
            AssetLoader.logger.warning(msg)
            pull_asset(update=False)
        else:
            if AssetLoader.should_update_asset():
                AssetLoader.logger.warning(
                    "Assets outdated! Current: {}, Expected: {}. "
                    "Updating the assets ...".format(asset_version(), VERSION)
                )
                pull_asset(update=True)
            else:
                AssetLoader.logger.info("Assets version: {}".format(VERSION))

    def change_object_name(self, obj, new_name):
        raise DeprecationWarning("This function is too dangerous to be used")
        """
        Change the name of one object, Note: it may bring some bugs if abusing
        """
        obj = self._spawned_objects.pop(obj.name)
        self._spawned_objects[new_name] = obj

    def add_task(self, object_id, task):
        raise DeprecationWarning
        self._object_tasks[object_id] = task

    def has_task(self, object_id):
        raise DeprecationWarning
        return True if object_id in self._object_tasks else False

    def get_task(self, object_id):
        """
        Return task of specific object with id
        :param object_id: a filter function, only return objects satisfying this condition
        :return: task
        """
        raise DeprecationWarning
        assert object_id in self._object_tasks, "Can not find the task for object(id: {})".format(object_id)
        return self._object_tasks[object_id]


if __name__ == "__main__":
    from metadrive.envs.base_env import BASE_DEFAULT_CONFIG

    BASE_DEFAULT_CONFIG["use_render"] = True
    BASE_DEFAULT_CONFIG["show_interface"] = False
    BASE_DEFAULT_CONFIG["render_pipeline"] = True
    world = BaseEngine(BASE_DEFAULT_CONFIG)

    from metadrive.engine.asset_loader import AssetLoader

    car_model = world.loader.loadModel(AssetLoader.file_path("models", "vehicle", "lada", "vehicle.gltf"))
    car_model.reparentTo(world.render)
    car_model.set_pos(0, 0, 190)
    # world.render_pipeline.prepare_scene(env.engine.render)

    world.run()
