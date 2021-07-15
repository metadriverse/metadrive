import logging
from typing import List
from typing import Optional, Dict, AnyStr, Union

import numpy as np

from pgdrive.engine.world.pg_world import PGWorld
from pgdrive.utils.math_utils import norm

logger = logging.getLogger(__name__)


class PGDriveEngine(PGWorld):
    """
    Due to the feature of Panda3D, PGDriveEngine should only be created once(Singleton Pattern)
    PGWorld is a pure game engine, which is not task-specific, while PGDriveEngine connects the
    driving task and the game engine modified from Panda3D Engine.
    """
    singleton = None

    IN_REPLAY = False
    STOP_REPLAY = False

    # Used to cull distant rendering object to improve rendering efficiency

    # Visualization cull
    LOD_MAP_VIS_DIST = 300  # highly related to the render efficiency !
    LOD_VEHICLE_VIS_DIST = 500
    LOD_OBJECT_VIS_DIST = 500

    # Physics world cull, which can save the time used to do collision detection
    LOD_MAP_PHYSICS_DIST = 50
    LOD_VEHICLE_PHYSICS_DIST = 50
    LOD_OBJECT_PHYSICS_DIST = 50

    def __init__(
        self,
        pgdrive_config: Union[Dict, "PGConfig"],
        agent_manager: "AgentManager",
    ):
        self.pgdrive_config = pgdrive_config
        super(PGDriveEngine, self).__init__(pgdrive_config["pg_world_config"])
        self.task_manager = self.taskMgr  # use the inner TaskMgr of Panda3D as PGDrive task manager
        traffic_config = {
            "traffic_mode": pgdrive_config["traffic_mode"],
            "random_traffic": pgdrive_config["random_traffic"]
        }
        self.traffic_manager = self._get_traffic_manager(traffic_config)
        self.object_manager = self._get_object_manager()
        self.agent_manager = agent_manager  # Only a reference

        # common variable
        self.map = None

        # for recovering, they can not exist together
        # TODO new record/replay
        self.record_episode = False
        self.replay_system = None
        self.record_system = None
        self.accept("s", self._stop_replay)

        # cull scene
        self.cull_scene = pgdrive_config["cull_scene"]
        self.detector_mask = None

    @staticmethod
    def _get_traffic_manager(traffic_config):
        from pgdrive.scene_manager.traffic_manager import TrafficManager
        return TrafficManager(traffic_config["traffic_mode"], traffic_config["random_traffic"])

    def _get_object_manager(self, object_config=None):
        from pgdrive.scene_manager.object_manager import ObjectManager
        return ObjectManager()

    def reset(self, map, traffic_density: float, accident_prob: float, episode_data=None):
        """
        For garbage collecting using, ensure to release the memory of all traffic vehicles
        """
        self.map = map

        self.traffic_manager.reset(map, traffic_density)
        self.object_manager.reset(map, accident_prob)
        if self.detector_mask is not None:
            self.detector_mask.clear()

        if self.replay_system is not None:
            self.replay_system.destroy()
            self.replay_system = None
        if self.record_system is not None:
            self.record_system.destroy()
            self.record_system = None

        if episode_data is None:
            self.object_manager.generate()
            self.traffic_manager.generate(map=self.map, traffic_density=traffic_density)
            self.IN_REPLAY = False
        else:
            self.replay_system = None
            logging.warning("You are replaying episodes! Delete detector mask!")
            self.detector_mask = None
            self.IN_REPLAY = True

        # if pg_world.highway_render is not None:
        #     pg_world.highway_render.set_scene_manager(self)
        if self.record_episode:
            if episode_data is None:
                init_states = self.traffic_manager.get_global_init_states()
                self.record_system = None
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
        pg_world = self
        dt = pg_world.world_config["physics_world_step_size"]
        for i in range(step_num):
            # simulate or replay
            if self.replay_system is None:
                # not in replay mode
                self.traffic_manager.step(dt)
                pg_world.step_physics_world()
            else:
                if not self.STOP_REPLAY:
                    self.replay_system.replay_frame(self.target_vehicles, i == step_num - 1)
            # record every step
            if self.record_system is not None:
                # didn't record while replay
                frame_state = self.traffic_manager.get_global_states()
                self.record_system.record_frame(frame_state)

            if pg_world.force_fps.real_time_simulation and i < step_num - 1:
                # insert frame to render in min step_size
                pg_world.task_manager.step()
        #  panda3d render and garbage collecting loop
        pg_world.task_manager.step()

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
            self.cull_distant_blocks(self.map.blocks, poses, self.world_config["max_distance"])

            if self.replay_system is None:
                # TODO add objects to replay system and add new cull method

                self.cull_distant_traffic_vehicles(
                    self.traffic_manager.traffic_vehicles, poses, self.world_config["max_distance"]
                )
                self.cull_distant_objects(
                    self.object_manager._spawned_objects, poses, self.world_config["max_distance"]
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

    def close(self):
        """
        Note:
        Instead of calling this func directly, close Engine by using engine_utils.close_pgdrive_engine
        """
        self.traffic_manager.destroy()
        self.traffic_manager = None

        self.object_manager.destroy()
        self.object_manager = None

        self.map = None
        if self.record_system is not None:
            self.record_system.destroy()
            self.record_system = None
        if self.replay_system is not None:
            self.replay_system.destroy()
            self.replay_system = None
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

    def cull_distant_blocks(self, blocks: list, poses: List[tuple], max_distance=None):
        # A distance based LOD rendering like GTA
        for block in blocks:
            if not self.all_out_of_bounding_box(block.bounding_box, poses, self.LOD_MAP_VIS_DIST):
                if not block.node_path.hasParent():
                    block.node_path.reparentTo(self.worldNP)
            else:
                if block.node_path.hasParent():
                    block.node_path.detachNode()
            if not self.all_out_of_bounding_box(block.bounding_box, poses, max_distance or self.LOD_MAP_PHYSICS_DIST):
                block.dynamic_nodes.attach_to_physics_world(self.physics_world.dynamic_world)
            else:
                block.dynamic_nodes.detach_from_physics_world(self.physics_world.dynamic_world)

    def cull_distant_traffic_vehicles(self, vehicles: list, poses: List[tuple], max_distance=None):
        self._cull_elements(vehicles, poses, self.LOD_VEHICLE_VIS_DIST, max_distance or self.LOD_VEHICLE_PHYSICS_DIST)

    def cull_distant_objects(self, objects: list, poses: List[tuple], max_distance=None):
        self._cull_elements(objects, poses, self.LOD_OBJECT_VIS_DIST, max_distance or self.LOD_OBJECT_PHYSICS_DIST)

    def _cull_elements(self, elements: list, poses: List[tuple], vis_distance: float, physics_distance: float):
        for obj in elements:
            v_p = obj.position
            if not self.all_distance_greater_than(vis_distance, poses, v_p):
                if not obj.node_path.hasParent():
                    obj.node_path.reparentTo(self.pbr_worldNP)
            else:
                if obj.node_path.hasParent():
                    obj.node_path.detachNode()

            if not self.all_distance_greater_than(physics_distance, poses, v_p):
                obj.dynamic_nodes.attach_to_physics_world(self.physics_world.dynamic_world)
            else:
                obj.dynamic_nodes.detach_from_physics_world(self.physics_world.dynamic_world)

    @staticmethod
    def all_distance_greater_than(distance, poses, target_pos):
        v_p = target_pos
        for pos in poses:
            if norm(v_p[0] - pos[0], v_p[1] - v_p[1]) < distance:
                return False
        return True

    @staticmethod
    def all_out_of_bounding_box(bounding_box, poses, margin_distance):
        for pos in poses:
            if bounding_box[0] - margin_distance < pos[0] < bounding_box[1] + margin_distance and \
                    bounding_box[2] - margin_distance < pos[1] < bounding_box[3] + margin_distance:
                return False
        return True
