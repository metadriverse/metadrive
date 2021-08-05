from typing import Union

import numpy as np

from pgdrive.component.highway_vehicle.behavior import IDMVehicle
from pgdrive.component.lane.circular_lane import CircularLane
from pgdrive.component.lane.straight_lane import StraightLane
from pgdrive.component.vehicle.base_vehicle import BaseVehicle
from pgdrive.constants import CollisionGroup
from pgdrive.engine.asset_loader import AssetLoader
from pgdrive.engine.engine_utils import get_engine
from pgdrive.manager.traffic_manager import TrafficManager
from pgdrive.utils.coordinates_shift import panda_position, panda_heading


class TrafficVehicle(BaseVehicle):
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
        :param kinematic_model: IDM Model or other models
        :param enable_respawn: It will be generated at the spawn place again when arriving at the destination
        :param random_seed: Random Engine seed
        """
        kinematic_model.LENGTH = self.LENGTH
        kinematic_model.WIDTH = self.WIDTH
        engine = get_engine()
        self._initial_state = kinematic_model if enable_respawn else None
        self.kinematic_model = IDMVehicle.create_from(kinematic_model)
        # TODO random seed work_around
        super(TrafficVehicle, self).__init__(engine.global_config["vehicle_config"], random_seed=random_seed)
        self.step(0.01, {"steering": 0, "acceleration": 0})

    def _add_visualization(self):
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

    def before_step(self):
        pass

    def step(self, dt, action=None):
        if self.break_down:
            return

        # TODO: We ignore this part here! Because the code is in IDM policy right now!
        #  Is that OK now?
        if action is None:
            action = {"steering": 0, "acceleration": 0}
        self.kinematic_model.step(dt, action)

        position = panda_position(self.kinematic_model.position, 0)
        self.origin.setPos(position)
        heading = np.rad2deg(panda_heading(self.kinematic_model.heading))
        self.origin.setH(heading - 90)

    def after_step(self):
        pass

    @property
    def out_of_road(self):
        p = get_engine().policy_manager.get_policy(self.name)
        ret = not p.lane.on_lane(self.kinematic_model.position, margin=2)
        return ret

    def need_remove(self):
        if self._initial_state is not None:
            return False
        else:
            print("The vehicle is removed!")
            return True

    def reset(self):
        self.kinematic_model = IDMVehicle.create_from(self._initial_state)
        self.step(0.01, {"steering": 0, "acceleration": 0})

    def destroy(self):
        self.kinematic_model.destroy()
        super(TrafficVehicle, self).destroy()

    def set_position(self, position, height=0.4):
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
        return self.kinematic_model.heading

    @property
    def heading_theta(self):
        return self.kinematic_model.heading_theta

    @property
    def position(self):
        return self.kinematic_model.position.tolist()

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
        super(TrafficVehicle, self).__del__()

    @property
    def speed(self):
        return self.kinematic_model.speed
