import logging
from typing import Dict, AnyStr
import numpy as np
from collections import OrderedDict

from pgdrive.engine.core.engine_core import EngineCore
from pgdrive.engine.scene_cull import SceneCull
from pgdrive.manager.base_manager import BaseManager

logger = logging.getLogger(__name__)


class BaseEngine(EngineCore):
    """
    Due to the feature of Panda3D, BaseEngine should only be created once(Singleton Pattern)
    PGWorld is a pure game engine, which is not task-specific, while BaseEngine connects the
    driving task and the game engine modified from Panda3D Engine.
    """
    singleton = None
    global_random_seed = None

    IN_REPLAY = False
    STOP_REPLAY = False

    def __init__(self, global_config):
        self.global_config = global_config
        super(BaseEngine, self).__init__(self.global_config["engine_config"])
        self.task_manager = self.taskMgr  # use the inner TaskMgr of Panda3D as PGDrive task manager
        self._managers = OrderedDict()

        # for recovering, they can not exist together
        # TODO new record/replay
        self.record_episode = False
        self.replay_system = None
        self.record_system = None
        self.accept("s", self._stop_replay)

        # cull scene
        self.cull_scene = self.global_config["cull_scene"]
        self.detector_mask = None

        # add default engines

    def reset(self, episode_data=None):
        """
        For garbage collecting using, ensure to release the memory of all traffic vehicles
        """

        for manager in self._managers.values():
            manager.before_reset()
        if self.detector_mask is not None:
            self.detector_mask.clear()

        if self.replay_system is not None:
            self.replay_system.destroy()
            self.replay_system = None
        if self.record_system is not None:
            self.record_system.destroy()
            self.record_system = None

        if episode_data is None:
            for manager in self._managers.values():
                manager.reset()
            for manager in self._managers.values():
                manager.after_reset()
            self.IN_REPLAY = False
        else:
            self.replay_system = None
            logging.warning("You are replaying episodes! Delete detector mask!")
            self.detector_mask = None
            self.IN_REPLAY = True

        # TODO recorder
        # if engine.highway_render is not None:
        #     engine.highway_render.set_scene_manager(self)
        # if self.record_episode:
        #     if episode_data is None:
        #         init_states = self.traffic_manager.get_global_init_states()
        #         self.record_system = None
        #     else:
        #         logging.warning("Temporally disable episode recorder, since we are replaying other episode!")

    def before_step(self, target_actions: Dict[AnyStr, np.array]):
        """
        Entities make decision here, and prepare for step
        All entities can access this global manager to query or interact with others
        :param target_actions: Dict[agent_id:action]
        :return:
        """
        step_infos = {}
        if self.replay_system is None:
            # not in replay mode
            for k in self.agent_manager.active_agents.keys():
                a = target_actions[k]
                step_infos[k] = self.agent_manager.get_agent(k).before_step(a)
            for manager in self._managers.values():
                manager.before_step()
        return step_infos

    def step(self, step_num: int = 1) -> None:
        """
        Step the dynamics of each entity on the road.
        :param step_num: Decision of all entities will repeat *step_num* times
        """
        engine = self
        for i in range(step_num):
            # simulate or replay
            if self.replay_system is None:
                # not in replay mode
                for manager in self._managers.values():
                    if isinstance(manager, BaseManager):
                        manager.step()
                engine.step_physics_world()
            else:
                if not self.STOP_REPLAY:
                    self.replay_system.replay_frame(self.target_vehicles, i == step_num - 1)
            # # record every step
            # if self.record_system is not None:
            #     # didn't record while replay
            #     frame_state = self.traffic_manager.get_global_states()
            #     self.record_system.record_frame(frame_state)

            if engine.force_fps.real_time_simulation and i < step_num - 1:
                # insert frame to render in min step_size
                engine.task_manager.step()
        #  panda3d render and garbage collecting loop
        engine.task_manager.step()

    def after_step(self) -> Dict:
        """
        Update states after finishing movement
        :return: if this episode is done
        """

        if self.replay_system is None:
            for manager in self._managers.values():
                manager.after_step()

        # TODO make detector mask a manager and do after_step!
        step_infos = self.update_state_for_all_target_vehicles()

        # cull distant blocks
        poses = [v.position for v in self.agent_manager.active_agents.values()]
        if self.cull_scene:
            # TODO use a for loop
            SceneCull.cull_distant_blocks(self, self.current_map.blocks, poses, self.world_config["max_distance"])

            SceneCull.cull_distant_traffic_vehicles(
                self, self.traffic_manager.traffic_vehicles, poses, self.world_config["max_distance"]
            )
            SceneCull.cull_distant_objects(self, self.object_manager.objects, poses, self.world_config["max_distance"])

        return step_infos

    def update_state_for_all_target_vehicles(self):

        # TODO(pzh): What is this function? Should we need to call it all steps?

        if self.detector_mask is not None:
            is_target_vehicle_dict = {
                v_obj.name: self.agent_manager.is_active_object(v_obj.name)
                for v_obj in self.get_interactive_objects() + self.traffic_manager.traffic_vehicles
            }
            self.detector_mask.update_mask(
                position_dict={
                    v_obj.name: v_obj.position
                    for v_obj in self.get_interactive_objects() + self.traffic_manager.traffic_vehicles
                },
                heading_dict={
                    v_obj.name: v_obj.heading_theta
                    for v_obj in self.get_interactive_objects() + self.traffic_manager.traffic_vehicles
                },
                is_target_vehicle_dict=is_target_vehicle_dict
            )
        step_infos = self.agent_manager.for_each_active_agents(
            lambda v: v.after_step(detector_mask=self.detector_mask.get_mask(v.name) if self.detector_mask else None)
        )
        return step_infos

    def get_interactive_objects(self):
        objs = self.agent_manager.get_vehicle_list() + self.object_manager.objects
        return objs

    def dump_episode(self) -> None:
        """Dump the data of an episode."""
        assert self.record_system is not None
        return self.record_system.dump_episode()

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

        self.clear_world()
        self.close_world()

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def is_target_vehicle(self, v):
        return v in self.agent_manager.active_agents.values()

    @property
    def target_vehicles(self):
        return {k: v for k, v in self.agent_manager.active_agents.items()}

    def _stop_replay(self):
        if not self.IN_REPLAY:
            return
        self.STOP_REPLAY = not self.STOP_REPLAY

    def register_manager(self, manger_name: str, manager: BaseManager):
        """
        Add a manager to BaseEngine, then all objects can communicate with this class
        :param manger_name: name shouldn't exist in self._managers and not be same as any class attribute
        :param manager: subclass of BaseManager
        """
        assert manger_name not in self._managers, "Manager already exists in BaseEngine"
        assert not hasattr(self, manger_name), "Manager name can not be same as the attribute in BaseEngine"
        self._managers[manger_name] = manager
        self._managers.move_to_end(manger_name)
        setattr(self, manger_name, manager)

    def seed(self, random_seed):
        self.global_random_seed = random_seed
        for mgr in self._managers.values():
            mgr.seed(random_seed)

    @property
    def current_map(self):
        return self.map_manager.current_map
