import logging
from typing import Optional, Dict, AnyStr, Union

import numpy as np
from pgdrive.scene_creator.map import Map
from pgdrive.scene_manager.PGLOD import PGLOD
from pgdrive.scene_manager.agent_manager import AgentManager
from pgdrive.scene_manager.object_manager import ObjectManager
from pgdrive.scene_manager.replay_record_system import PGReplayer, PGRecorder
from pgdrive.scene_manager.traffic_manager import TrafficManager
from pgdrive.utils import PGConfig
from pgdrive.world.pg_world import PGWorld

logger = logging.getLogger(__name__)


class SceneManager:
    """Manage all traffic vehicles, and all runtime elements (in the future)"""
    IN_REPLAY = False
    STOP_REPLAY = False

    def __init__(
        self,
        pg_world: PGWorld,
        traffic_config: Union[Dict, "PGConfig"],
        record_episode: bool,
        cull_scene: bool,
        agent_manager: "AgentManager",
    ):
        """
        :param traffic_mode: respawn/trigger mode
        :param random_traffic: if True, map seed is different with traffic manager seed
        """
        # scene manager control all movements in pg_world
        self.pg_world = pg_world

        self.traffic_manager = self._get_traffic_manager(traffic_config)
        self.object_manager = self._get_object_manager()
        self.agent_manager = agent_manager  # Only a reference

        # common variable
        self.map = None

        # for recovering, they can not exist together
        self.record_episode = record_episode
        self.replay_system: Optional[PGReplayer] = None
        self.record_system: Optional[PGRecorder] = None
        pg_world.accept("s", self._stop_replay)

        # cull scene
        self.cull_scene = cull_scene
        self.detector_mask = None

    def _get_traffic_manager(self, traffic_config):
        return TrafficManager(self, traffic_config["traffic_mode"], traffic_config["random_traffic"])

    def _get_object_manager(self, object_config=None):
        return ObjectManager()

    def reset(self, map: Map, traffic_density: float, accident_prob: float, episode_data=None):
        """
        For garbage collecting using, ensure to release the memory of all traffic vehicles
        """
        pg_world = self.pg_world
        self.map = map

        self.traffic_manager.reset(pg_world, map, traffic_density)
        self.object_manager.reset(pg_world, map, accident_prob)
        if self.detector_mask is not None:
            self.detector_mask.clear()

        if self.replay_system is not None:
            self.replay_system.destroy(pg_world)
            self.replay_system = None
        if self.record_system is not None:
            self.record_system.destroy(pg_world)
            self.record_system = None

        if episode_data is None:
            self.object_manager.generate(self, pg_world)
            self.traffic_manager.generate(pg_world=pg_world, map=self.map, traffic_density=traffic_density)
            self.IN_REPLAY = False
        else:
            self.replay_system = PGReplayer(self.traffic_manager, map, episode_data, pg_world)
            logging.warning("You are replaying episodes! Delete detector mask!")
            self.detector_mask = None
            self.IN_REPLAY = True

        # if pg_world.highway_render is not None:
        #     pg_world.highway_render.set_scene_manager(self)
        if self.record_episode:
            if episode_data is None:
                init_states = self.traffic_manager.get_global_init_states()
                self.record_system = PGRecorder(map, init_states)
            else:
                logging.warning("Temporally disable episode recorder, since we are replaying other episode!")

    def prepare_step(self, target_actions: Dict[AnyStr, np.array]):
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
                step_infos[k] = self.agent_manager.get_agent(k).prepare_step(a)
            self.traffic_manager.prepare_step()
        return step_infos

    def step(self, step_num: int = 1) -> None:
        """
        Step the dynamics of each entity on the road.
        :param pg_world: World
        :param step_num: Decision of all entities will repeat *step_num* times
        """
        pg_world = self.pg_world
        dt = pg_world.world_config["physics_world_step_size"]
        for i in range(step_num):
            # simulate or replay
            if self.replay_system is None:
                # not in replay mode
                self.traffic_manager.step(dt)
                pg_world.step()
            else:
                if not self.STOP_REPLAY:
                    self.replay_system.replay_frame(self.target_vehicles, self.pg_world, i == step_num - 1)
            # record every step
            if self.record_system is not None:
                # didn't record while replay
                frame_state = self.traffic_manager.get_global_states()
                self.record_system.record_frame(frame_state)

            if pg_world.force_fps.real_time_simulation and i < step_num - 1:
                # insert frame to render in min step_size
                pg_world.taskMgr.step()
        #  panda3d render and garbage collecting loop
        pg_world.taskMgr.step()

    def update_state(self) -> Dict:
        """
        Update states after finishing movement
        :return: if this episode is done
        """

        if self.replay_system is None:
            self.traffic_manager.update_state()

        step_infos = self.update_state_for_all_target_vehicles()

        # cull distant blocks
        poses = [v.position for v in self.agent_manager.active_agents.values()]
        if self.cull_scene:
            PGLOD.cull_distant_blocks(self.map.blocks, poses, self.pg_world, self.pg_world.world_config["max_distance"])
            # PGLOD.cull_distant_blocks(self.map.blocks, self.ego_vehicle.position, self.pg_world)

            if self.replay_system is None:
                # TODO add objects to replay system and add new cull method

                PGLOD.cull_distant_traffic_vehicles(
                    self.traffic_manager.traffic_vehicles, poses, self.pg_world,
                    self.pg_world.world_config["max_distance"]
                )
                PGLOD.cull_distant_objects(
                    self.object_manager._spawned_objects, poses, self.pg_world,
                    self.pg_world.world_config["max_distance"]
                )

        return step_infos

    def update_state_for_all_target_vehicles(self):
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
            lambda v: v.update_state(detector_mask=self.detector_mask.get_mask(v.name) if self.detector_mask else None)
        )
        return step_infos

    def get_interactive_objects(self):
        objs = self.agent_manager.get_vehicle_list() + self.object_manager.objects
        return objs

    def dump_episode(self) -> None:
        """Dump the data of an episode."""
        assert self.record_system is not None
        return self.record_system.dump_episode()

    def destroy(self, pg_world: PGWorld = None):
        pg_world = self.pg_world if pg_world is None else pg_world
        self.traffic_manager.destroy(pg_world)
        self.traffic_manager = None

        self.object_manager.destroy(pg_world)
        self.object_manager = None

        self.map = None
        if self.record_system is not None:
            self.record_system.destroy(pg_world)
            self.record_system = None
        if self.replay_system is not None:
            self.replay_system.destroy(pg_world)
            self.replay_system = None

    def __repr__(self):
        info = "traffic:" + self.traffic_manager.__repr__()
        return info

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
