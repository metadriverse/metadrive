import logging
from typing import List, Tuple, Optional, Dict, AnyStr, Union

import numpy as np
from pgdrive.scene_creator.map import Map
from pgdrive.scene_manager.PGLOD import PGLOD
from pgdrive.scene_manager.object_manager import ObjectsManager
from pgdrive.scene_manager.replay_record_system import PGReplayer, PGRecorder
from pgdrive.scene_manager.traffic_manager import TrafficManager
from pgdrive.utils import PGConfig
from pgdrive.world.pg_world import PGWorld

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]


class SceneManager:
    """Manage all traffic vehicles, and all runtime elements (in the future)"""
    def __init__(
        self,
        config,
        pg_world: PGWorld,
        traffic_config: Union[Dict, "PGConfig"],
        # traffic_mode=TrafficMode.Trigger,
        # random_traffic: bool = False,
        record_episode: bool = False,
        cull_scene: bool = True,
    ):
        """
        :param traffic_mode: respawn/trigger mode
        :param random_traffic: if True, map seed is different with traffic manager seed
        """
        # scene manager control all movements in pg_world
        self.pg_world = pg_world

        self.traffic_mgr = self._get_traffic_manager(traffic_config)
        self.objects_mgr = self._get_object_manager()

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
        self.detector_mask = None

    def _get_traffic_manager(self, traffic_config):
        return TrafficManager(traffic_config["traffic_mode"], traffic_config["random_traffic"])

    def _get_object_manager(self, object_config=None):
        return ObjectsManager()

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
        if self.detector_mask is not None:
            self.detector_mask.clear()

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
            logging.warning("You are replaying episodes! Delete detector mask!")
            self.detector_mask = None

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
            for k in self.target_vehicles.keys():
                a = target_actions[k]
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
        dt = pg_world.world_config["physics_world_step_size"]
        for i in range(step_num):
            if self.replay_system is None:
                # not in replay mode
                self.traffic_mgr.step(dt)
                pg_world.step()
            if pg_world.force_fps.real_time_simulation and i < step_num - 1:
                # insert frame to render in min step_size
                pg_world.taskMgr.step()

            # print("Step {}/{}. Steering {}. Acceleration {}. Brake {}.".format(
            #     i + 1, step_num,
            #     [self.target_vehicles['default_agent'].system.get_steering_value(ii) for ii in range(4)],
            #     [self.target_vehicles['default_agent'].system.get_wheel(ii).engine_force for ii in range(4)],
            #     [self.target_vehicles['default_agent'].system.get_wheel(ii).brake for ii in range(4)],
            # ))

        #  panda3d render and garbage collecting loop
        pg_world.taskMgr.step()

    def update_state(self) -> Dict:
        """
        Update states after finishing movement
        :return: if this episode is done
        """

        if self.replay_system is not None:
            self.for_each_target_vehicle(lambda v: self.replay_system.replay_frame(v, self.pg_world))
            # self.replay_system.replay_frame(self.ego_vehicle, self.pg_world)
        else:
            self.traffic_mgr.update_state(self, self.pg_world)

        if self.record_system is not None:
            # didn't record while replay
            self.record_system.record_frame(self.traffic_mgr.get_global_states())

        step_infos = self.update_state_for_all_target_vehicles()

        # cull distant blocks
        poses = [v.position for v in self.target_vehicles.values()]
        if self.cull_scene:
            PGLOD.cull_distant_blocks(self.map.blocks, poses, self.pg_world, self.pg_world.world_config["max_distance"])
            # PGLOD.cull_distant_blocks(self.map.blocks, self.ego_vehicle.position, self.pg_world)

            if self.replay_system is None:
                # TODO add objects to replay system and add new cull method

                PGLOD.cull_distant_traffic_vehicles(
                    self.traffic_mgr.traffic_vehicles, poses, self.pg_world, self.pg_world.world_config["max_distance"]
                )
                PGLOD.cull_distant_objects(
                    self.objects_mgr._spawned_objects, poses, self.pg_world, self.pg_world.world_config["max_distance"]
                )

        return step_infos

    def update_state_for_all_target_vehicles(self):
        if self.detector_mask is not None:
            # a = set([v.name for v in self.traffic_mgr.vehicles])
            # b = set([v.name for v in self.target_vehicles.values()])
            # assert b.issubset(a)  # This may only happen during episode replays!
            is_target_vehicle_dict = {v.name: self.traffic_mgr.is_target_vehicle(v) for v in self.traffic_mgr.vehicles}
            self.detector_mask.update_mask(
                position_dict={v.name: v.position
                               for v in self.traffic_mgr.vehicles},
                heading_dict={v.name: v.heading_theta
                              for v in self.traffic_mgr.vehicles},
                is_target_vehicle_dict=is_target_vehicle_dict
            )
        step_infos = self.for_each_target_vehicle(
            lambda v: v.update_state(detector_mask=self.detector_mask.get_mask(v.name) if self.detector_mask else None)
        )
        return step_infos

    def for_each_target_vehicle(self, func):
        """Apply the func (a function take only the vehicle as argument) to each target vehicles and return a dict!"""
        assert len(self.target_vehicles) > 0
        ret = dict()
        for k, v in self.target_vehicles.items():
            ret[k] = func(v)
        return ret

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
