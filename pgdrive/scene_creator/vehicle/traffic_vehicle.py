from typing import Union
import math

import numpy as np
from panda3d.bullet import BulletRigidBodyNode, BulletBoxShape
from panda3d.core import BitMask32, TransformState, Point3, NodePath, Vec3
from pgdrive.constants import BodyName, CollisionGroup
from pgdrive.scene_creator.highway_vehicle.behavior import IDMVehicle
from pgdrive.scene_creator.lane.circular_lane import CircularLane
from pgdrive.scene_creator.lane.straight_lane import StraightLane
from pgdrive.scene_manager.traffic_manager import TrafficManager
from pgdrive.utils import get_np_random
from pgdrive.utils.asset_loader import AssetLoader
from pgdrive.utils.coordinates_shift import panda_position, panda_heading
from pgdrive.utils.element import DynamicElement
from pgdrive.utils.scene_utils import ray_localization
from pgdrive.world.pg_world import PGWorld


class TrafficVehicleNode(BulletRigidBodyNode):

    # for lidar detection and other purposes
    def __init__(self, node_name, kinematics_model: IDMVehicle):
        BulletRigidBodyNode.__init__(self, node_name)
        TrafficVehicleNode.setPythonTag(self, BodyName.Traffic_vehicle, self)
        self.kinematic_model = kinematics_model

    def reset(self, kinematics_model):
        self.kinematic_model = IDMVehicle.create_from(kinematics_model)


class PGTrafficVehicle(DynamicElement):
    COLLISION_MASK = CollisionGroup.TrafficVehicle
    HEIGHT = 1.8
    LENGTH = 4
    WIDTH = 2
    path = None
    break_down = False
    model_collection = {}  # save memory, load model once

    def __init__(self, index: int, kinematic_model: IDMVehicle, enable_respawn: bool = False, np_random=None):
        """
        A traffic vehicle class.
        :param index: Each Traffic vehicle has an unique index, and the name of this vehicle will contain this index
        :param kinematic_model: IDM Model or other models
        :param enable_respawn: It will be generated at the spawn place again when arriving at the destination
        :param np_random: Random Engine
        """
        kinematic_model.LENGTH = self.LENGTH
        kinematic_model.WIDTH = self.WIDTH
        super(PGTrafficVehicle, self).__init__()
        self.vehicle_node = TrafficVehicleNode(BodyName.Traffic_vehicle, IDMVehicle.create_from(kinematic_model))
        chassis_shape = BulletBoxShape(Vec3(self.LENGTH / 2, self.WIDTH / 2, self.HEIGHT / 2))
        self.index = index
        self.vehicle_node.addShape(chassis_shape, TransformState.makePos(Point3(0, 0, self.HEIGHT / 2)))
        self.vehicle_node.setMass(800.0)
        self.vehicle_node.setIntoCollideMask(BitMask32.bit(self.COLLISION_MASK))
        self.vehicle_node.setKinematic(False)
        self.vehicle_node.setStatic(True)
        self.enable_respawn = enable_respawn
        self._initial_state = kinematic_model if enable_respawn else None
        self.dynamic_nodes.append(self.vehicle_node)
        self.node_path = NodePath(self.vehicle_node)
        self.out_of_road = False

        np_random = np_random or get_np_random()
        [path, scale, x_y_z_offset, H] = self.path[np_random.randint(0, len(self.path))]
        if self.render:
            if path not in PGTrafficVehicle.model_collection:
                carNP = self.loader.loadModel(AssetLoader.file_path("models", path))
                PGTrafficVehicle.model_collection[path] = carNP
            else:
                carNP = PGTrafficVehicle.model_collection[path]
            carNP.setScale(scale)
            carNP.setH(H)
            carNP.setPos(x_y_z_offset)

            carNP.instanceTo(self.node_path)
        self.step(1e-1)
        # self.carNP.setQuat(LQuaternionf(math.cos(-1 * np.pi / 4), 0, 0, math.sin(-1 * np.pi / 4)))

    def prepare_step(self, scene_manager) -> None:
        """
        Determine the action according to the elements in scene
        :param scene_manager: scene
        :return: None
        """
        self.vehicle_node.kinematic_model.act(scene_manager=scene_manager)

    def step(self, dt):
        if self.break_down:
            return
        self.vehicle_node.kinematic_model.step(dt)
        position = panda_position(self.vehicle_node.kinematic_model.position, 0)
        self.node_path.setPos(position)
        heading = np.rad2deg(panda_heading(self.vehicle_node.kinematic_model.heading))
        self.node_path.setH(heading)

    def update_state(self, pg_world: PGWorld):
        dir = np.array([math.cos(self.heading), math.sin(self.heading)])
        lane, lane_index = ray_localization(dir, self.position, pg_world)
        if lane is not None:
            self.vehicle_node.kinematic_model.update_lane_index(lane_index, lane)
        self.out_of_road = not self.vehicle_node.kinematic_model.lane.on_lane(
            self.vehicle_node.kinematic_model.position, margin=2
        )

    def need_remove(self):
        if self._initial_state is not None:
            return False
        else:
            self.vehicle_node.clearTag(BodyName.Traffic_vehicle)
            self.node_path.removeNode()
            return True

    def reset(self):
        self.vehicle_node.reset(self._initial_state)
        self.out_of_road = False

    def destroy(self, pg_world):
        self.vehicle_node.kinematic_model.destroy(pg_world)
        self.vehicle_node.clearTag(BodyName.Traffic_vehicle)
        super(PGTrafficVehicle, self).destroy(pg_world)

    def get_name(self):
        return self.vehicle_node.getName() + "_" + str(self.index)

    def set_position(self, position):
        """
        Should only be called when restore traffic from episode data
        :param position: 2d array or list
        :return: None
        """
        self.node_path.setPos(panda_position(position, 0))

    def set_heading(self, heading_theta) -> None:
        """
        Should only be called when restore traffic from episode data
        :param heading_theta: float in rad
        :return: None
        """
        self.node_path.setH(panda_heading(heading_theta * 180 / np.pi))

    def get_state(self):
        return {"heading": self.heading, "position": self.position, "done": self.out_of_road}

    def set_state(self, state: dict):
        self.set_heading(state["heading"])
        self.set_position(state["position"])

    @property
    def heading(self):
        return self.vehicle_node.kinematic_model.heading

    @property
    def position(self):
        return self.vehicle_node.kinematic_model.position.tolist()

    @classmethod
    def create_random_traffic_vehicle(
        cls,
        index: int,
        traffic_mgr: TrafficManager,
        lane: Union[StraightLane, CircularLane],
        longitude: float,
        seed=None,
        enable_lane_change: bool = True,
        enable_respawn=False
    ):
        v = IDMVehicle.create_random(traffic_mgr, lane, longitude, random_seed=seed)
        v.enable_lane_change = enable_lane_change
        return cls(index, v, enable_respawn, np_random=v.np_random)

    @classmethod
    def create_traffic_vehicle_from_config(cls, traffic_mgr: TrafficManager, config: dict):
        v = IDMVehicle(traffic_mgr, config["position"], config["heading"], np_random=None)
        return cls(config["index"], v, config["enable_respawn"])

    def set_break_down(self, break_down=True):
        self.break_down = break_down

    def __del__(self):
        self.vehicle_node.clearTag(BodyName.Traffic_vehicle)
        super(PGTrafficVehicle, self).__del__()
