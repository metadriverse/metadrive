import logging
from typing import List, Tuple, Optional, Dict, AnyStr

import numpy as np

from pgdrive.scene_creator.map import Map
from pgdrive.scene_manager.PGLOD import PGLOD
from pgdrive.scene_manager.object_manager import ObjectsManager
from pgdrive.scene_manager.replay_record_system import PGReplayer, PGRecorder
from pgdrive.scene_manager.traffic_manager import TrafficManager, TrafficMode
from pgdrive.world.pg_world import PGWorld

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]


class SceneManager:
    """Manage all traffic vehicles, and all runtime elements (in the future)"""
    def __init__(
        self,
        pg_world: PGWorld,
        traffic_mode=TrafficMode.Trigger,
        random_traffic: bool = False,
        record_episode: bool = False,
        cull_scene: bool = True,
    ):
        """
        :param traffic_mode: reborn/trigger mode
        :param random_traffic: if True, map seed is different with traffic manager seed
        """
        # scene manager control all movements in pg_world
        self.pg_world = pg_world

        # TODO more manager will be added in the future to manager traffic light/pedestrian
        self.traffic_mgr = TrafficManager(traffic_mode, random_traffic)
        self.objects_mgr = ObjectsManager()

        # common variable

        # FIXME! We need multi-agent variant of vehicles here!
        self.target_vehicles = None
        self.map = None

        # for recovering, they can not exist together
        self.record_episode = record_episode
        self.replay_system: Optional[PGReplayer] = None
        self.record_system: Optional[PGRecorder] = None

        # cull scene
        self.cull_scene = cull_scene

    def reset(self, map: Map, target_vehicles, traffic_density: float, accident_prob: float, episode_data=None):
        """
        For garbage collecting using, ensure to release the memory of all traffic vehicles
        """
        pg_world = self.pg_world
        assert isinstance(target_vehicles, dict)
        self.target_vehicles = target_vehicles
        self.map = map

        # FIXME
        self.traffic_mgr.reset(pg_world, map, target_vehicles, traffic_density)
        self.objects_mgr.reset(pg_world, map, accident_prob)

        if self.replay_system is not None:
            self.replay_system.destroy(pg_world)
            self.replay_system = None
        if self.record_system is not None:
            self.record_system.destroy(pg_world)
            self.record_system = None

        if episode_data is None:
            # FIXME
            self.objects_mgr.generate(self, pg_world)
            self.traffic_mgr.generate(
                pg_world=pg_world, map=self.map, target_vehicles=self.target_vehicles, traffic_density=traffic_density
            )
        else:
            self.replay_system = PGReplayer(self.traffic_mgr, map, episode_data, pg_world)

        # if pg_world.highway_render is not None:
        #     pg_world.highway_render.set_scene_mgr(self)
        if self.record_episode:
            if episode_data is None:
                self.record_system = PGRecorder(map, self.traffic_mgr.get_global_init_states())
            else:
                logging.warning("Temporally disable episode recorder, since we are replaying other episode!")

    def prepare_step(self, target_actions: Dict[AnyStr, np.array]):
        """
        Entities make decision here, and prepare for step
        All entities can access this global manager to query or interact with others
        :param pg_world: World
        :param ego_vehicle_action: Ego_vehicle action
        :return: None
        """
        step_infos = {}
        if self.replay_system is None:
            # not in replay mode
            for k, a in target_actions.items():
                step_infos[k] = self.target_vehicles[k].prepare_step(a)
            self.traffic_mgr.prepare_step(self)
        return step_infos

    def step(self, step_num: int = 1) -> None:
        """
        Step the dynamics of each entity on the road.
        :param pg_world: World
        :param step_num: Decision of all entities will repeat *step_num* times
        """
        pg_world = self.pg_world
        dt = pg_world.pg_config["physics_world_step_size"]
        for i in range(step_num):
            if self.replay_system is None:
                # not in replay mode
                self.traffic_mgr.step(dt)
                pg_world.step()
            if pg_world.force_fps.real_time_simulation and i < step_num - 1:
                # insert frame to render in min step_size
                pg_world.taskMgr.step()

        #  panda3d render and garbage collecting loop
        pg_world.taskMgr.step()

    def update_state(self) -> bool:
        """
        Update states after finishing movement
        :return: if this episode is done
        """
        dones = {k: False for k in self.target_vehicles.keys()}
        # done = False

        if self.replay_system is not None:
            self.for_each_target_vehicle(lambda v: self.replay_system.replay_frame(v, self.pg_world))
            # self.replay_system.replay_frame(self.ego_vehicle, self.pg_world)
        else:
            global_done = self.traffic_mgr.update_state(self, self.pg_world)
            if global_done:
                dones = {k: True for k in self.target_vehicles.keys()}

        if self.record_system is not None:
            # didn't record while replay
            self.record_system.record_frame(self.traffic_mgr.get_global_states())

        step_infos = self.for_each_target_vehicle(lambda v: v.update_state())
        # self.ego_vehicle.update_state()

        # cull distant objects
        self.for_each_target_vehicle(lambda v: PGLOD.cull_distant_blocks(self.map.blocks, v.position, self.pg_world))
        # PGLOD.cull_distant_blocks(self.map.blocks, self.ego_vehicle.position, self.pg_world)

        if self.replay_system is None:
            # TODO add objects to replay system and add new cull method

            def _advance(v):
                PGLOD.cull_distant_traffic_vehicles(self.traffic_mgr.traffic_vehicles, v.position, self.pg_world)
                PGLOD.cull_distant_objects(self.objects_mgr._spawned_objects, v.position, self.pg_world)

            self.for_each_target_vehicle(_advance)
            # PGLOD.cull_distant_traffic_vehicles(
            #     self.traffic_mgr.traffic_vehicles, self.ego_vehicle.position, self.pg_world
            # )
            # PGLOD.cull_distant_objects(self.objects_mgr.spawned_objects, self.ego_vehicle.position, self.pg_world)

        return dones, step_infos

    def for_each_target_vehicle(self, func):
        """Apply the func (a function take only the vehicle as argument) to each target vehicles and return a dict!"""
        ret = dict()
        for k, v in self.target_vehicles.items():
            ret[k] = func(v)
        return ret
        if self.cull_scene:
            PGLOD.cull_distant_blocks(self.map.blocks, self.ego_vehicle.position, self.pg_world)
            if self.replay_system is None:
                PGLOD.cull_distant_traffic_vehicles(
                    self.traffic_mgr.traffic_vehicles, self.ego_vehicle.position, self.pg_world
                )
                PGLOD.cull_distant_objects(self.objects_mgr._spawned_objects, self.ego_vehicle.position, self.pg_world)
        return done

    def dump_episode(self) -> None:
        """Dump the data of an episode."""
        assert self.record_system is not None
        return self.record_system.dump_episode()

    def destroy(self, pg_world: PGWorld = None):
        pg_world = self.pg_world if pg_world is None else pg_world
        self.traffic_mgr.destroy(pg_world)
        self.traffic_mgr = None

        self.objects_mgr.destroy(pg_world)
        self.objects_mgr = None

        self.map = None
        if self.record_system is not None:
            self.record_system.destroy(pg_world)
            self.record_system = None
        if self.replay_system is not None:
            self.replay_system.destroy(pg_world)
            self.replay_system = None

    def __repr__(self):
        info = "traffic:" + self.traffic_mgr.__repr__()
        return info

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def is_target_vehicle(self, v):
        return v in self.target_vehicles.values()
