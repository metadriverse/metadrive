from typing import Union

import numpy as np
from panda3d.bullet import BulletBoxShape
from panda3d.core import BitMask32, TransformState, Point3, Vec3

from pgdrive.component.base_class.base_object import BaseObject
from pgdrive.component.highway_vehicle.behavior import IDMVehicle
from pgdrive.component.lane.circular_lane import CircularLane
from pgdrive.component.lane.straight_lane import StraightLane
from pgdrive.constants import BodyName, CollisionGroup
from pgdrive.engine.asset_loader import AssetLoader
from pgdrive.engine.physics_node import TrafficVehicleNode
from pgdrive.manager.traffic_manager import TrafficManager
from pgdrive.utils.coordinates_shift import panda_position, panda_heading
from pgdrive.engine.engine_utils import get_engine


class TrafficVehicle(BaseObject):
    COLLISION_MASK = CollisionGroup.TrafficVehicle
    HEIGHT = 1.8
    LENGTH = 4
    WIDTH = 2
    path = None
    break_down = False
    model_collection = {}  # save memory, load model once

    def __init__(self, index: int, kinematic_model: IDMVehicle, enable_respawn: bool = False, random_seed=None):
        """
        A traffic vehicle class.
        :param index: Each Traffic vehicle has an unique index, and the name of this vehicle will contain this index
        :param kinematic_model: IDM Model or other models
        :param enable_respawn: It will be generated at the spawn place again when arriving at the destination
        :param random_seed: Random Engine seed
        """
        kinematic_model.LENGTH = self.LENGTH
        kinematic_model.WIDTH = self.WIDTH
        super(TrafficVehicle, self).__init__(random_seed=random_seed)
        self.add_physics_body(TrafficVehicleNode(BodyName.Traffic_vehicle, IDMVehicle.create_from(kinematic_model)))
        self.vehicle_node = self.body
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
        # self.out_of_road = False

        [path, scale, x_y_z_offset, H] = self.path[self.np_random.randint(0, len(self.path))]
        if self.render:
            if path not in TrafficVehicle.model_collection:
                carNP = self.loader.loadModel(AssetLoader.file_path("models", path))
                TrafficVehicle.model_collection[path] = carNP
            else:
                carNP = TrafficVehicle.model_collection[path]
            carNP.setScale(scale)
            carNP.setH(H)
            carNP.setPos(x_y_z_offset)

            carNP.instanceTo(self.origin)
        self.step(1e-1, None)
        # self.carNP.setQuat(LQuaternionf(math.cos(-1 * np.pi / 4), 0, 0, math.sin(-1 * np.pi / 4)))

    # def before_step(self) -> None:
    #     """
    #     Determine the action according to the elements in scene
    #     :return: None
    #     """
    #     self.vehicle_node.kinematic_model.act()

    def step(self, dt, action=None):
        if self.break_down:
            return

        # TODO: We ignore this part here! Because the code is in IDM policy right now!
        #  Is that OK now?
        if action is None:
            action = {"steering": 0, "acceleration": 0}
        self.vehicle_node.kinematic_model.step(dt, action)

        position = panda_position(self.vehicle_node.kinematic_model.position, 0)
        self.origin.setPos(position)
        heading = np.rad2deg(panda_heading(self.vehicle_node.kinematic_model.heading))
        self.origin.setH(heading)

    def after_step(self):
        # engine = get_engine()
        # dir = np.array([math.cos(self.heading), math.sin(self.heading)])
        # lane, lane_index = ray_localization(dir, self.position, engine)
        # if lane is not None:
        # e = get_engine()
        # p = e.policy_manager.get_policy(self.name)
        # p.update_lane_index(lane_index, lane)
        # self.vehicle_node.kinematic_model.update_lane_index(lane_index, lane)

        # self.out_of_road = not self.vehicle_node.kinematic_model.lane.on_lane(
        #     self.vehicle_node.kinematic_model.position, margin=2
        # )
        # if self.out_of_road:
        #     print('stop here')
        pass

    @property
    def out_of_road(self):
        p = get_engine().policy_manager.get_policy(self.name)
        ret = not p.lane.on_lane(self.vehicle_node.kinematic_model.position, margin=2)
        return ret

    def need_remove(self):
        if self._initial_state is not None:
            return False
        else:
            self.vehicle_node.clearTag(BodyName.Traffic_vehicle)
            self.origin.removeNode()
            print("The vehicle is removed!")
            return True

    def reset(self):
        self.vehicle_node.reset(self._initial_state)
        # self.out_of_road = False

    def destroy(self):
        self.vehicle_node.kinematic_model.destroy()
        self.vehicle_node.clearTag(BodyName.Traffic_vehicle)
        super(TrafficVehicle, self).destroy()

    def get_name(self):
        return self.vehicle_node.getName() + "_" + str(self.index)

    def set_position(self, position):
        """
        Should only be called when restore traffic from episode data
        :param position: 2d array or list
        :return: None
        """
        self.origin.setPos(panda_position(position, 0))

    def set_heading(self, heading_theta) -> None:
        """
        Should only be called when restore traffic from episode data
        :param heading_theta: float in rad
        :return: None
        """
        self.origin.setH(panda_heading(heading_theta * 180 / np.pi))

    def get_state(self):
        return {"heading": self.heading, "position": self.position, "done": self.out_of_road}

    def set_state(self, state: dict):
        self.set_heading(state["heading"])
        self.set_position(state["position"])

    @property
    def heading(self):
        return self.vehicle_node.kinematic_model.heading

    @property
    def heading_theta(self):
        return self.vehicle_node.kinematic_model.heading_theta

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
        random_seed=None,
        enable_lane_change: bool = True,
        enable_respawn=False
    ):
        v = IDMVehicle.create_random(traffic_mgr, lane, longitude, random_seed=random_seed)
        v.enable_lane_change = enable_lane_change
        return cls(index, v, enable_respawn, random_seed=random_seed)

    @classmethod
    def create_traffic_vehicle_from_config(cls, traffic_mgr: TrafficManager, config: dict):
        v = IDMVehicle(traffic_mgr, config["position"], config["heading"], np_random=None)
        return cls(config["index"], v, config["enable_respawn"])

    def set_break_down(self, break_down=True):
        self.break_down = break_down

    def __del__(self):
        self.vehicle_node.clearTag(BodyName.Traffic_vehicle)
        super(TrafficVehicle, self).__del__()

    # TODO(pzh): This is only workaround! We should not have so many speed all around!
    @property
    def speed(self):
        return self.vehicle_node.kinematic_model.speed
